#!/usr/bin/env python3
"""
DualSense PS5 teleop for Reachy2 mobile base.

Reads the DualSense controller via evdev and publishes velocity
commands directly to /cmd_vel (Twist).

Controls:
  Left stick X/Y     → linear velocity (forward-back, left-right)
  Right stick X       → angular velocity (yaw)
  L1 / R1             → decrease / increase speed
  Triangle            → print current speed scale
  L2 analog           → fine-control mode (proportional slow-down)

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  /usr/bin/python3 dualsense_teleop.py
"""

import argparse
import threading
import numpy as np

import evdev
from evdev import ecodes

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

ROBOT_LIST = ["reachy", "spot"]
ROBOT_TOPICS = {
    "reachy": "/cmd_vel",
    "spot": "/spot/cmd_vel",
}

# ── DualSense constants ──────────────────────────────────────────────────

DS_AXIS_CENTER = 128  # midpoint for 0-255 range
DS_AXIS_MAX = 127.0
DS_DEADZONE = 15      # raw units (out of 255)

# Axis codes
AX_LX  = ecodes.ABS_X      # 0: Left stick horizontal
AX_LY  = ecodes.ABS_Y      # 1: Left stick vertical
AX_L2  = ecodes.ABS_Z      # 2: L2 trigger (0=released, 255=full)
AX_RX  = ecodes.ABS_RX     # 3: Right stick horizontal
AX_RY  = ecodes.ABS_RY     # 4: Right stick vertical
AX_R2  = ecodes.ABS_RZ     # 5: R2 trigger
AX_DX  = ecodes.ABS_HAT0X  # 16: D-pad X (-1/0/1)
AX_DY  = ecodes.ABS_HAT0Y  # 17: D-pad Y (-1/0/1)

# Button codes
BTN_CROSS    = ecodes.BTN_SOUTH   # 304
BTN_CIRCLE   = ecodes.BTN_EAST    # 305
BTN_TRIANGLE = ecodes.BTN_NORTH   # 307
BTN_SQUARE   = ecodes.BTN_WEST    # 308
BTN_L1       = ecodes.BTN_TL      # 310
BTN_R1       = ecodes.BTN_TR      # 311
BTN_R2       = ecodes.BTN_TR2     # 313


def _normalize_stick(raw, center=DS_AXIS_CENTER, deadzone=DS_DEADZONE):
    """Normalize 0-255 raw value to -1..+1 with deadzone."""
    val = raw - center
    if abs(val) < deadzone:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return np.clip(sign * (abs(val) - deadzone) / (DS_AXIS_MAX - deadzone), -1.0, 1.0)


# ── ROS2 Node ─────────────────────────────────────────────────────────────

class DualSenseTeleop(Node):
    def __init__(self, robot="reachy"):
        super().__init__("dualsense_teleop")

        # ── Robot switching ──────────────────────────────────────────────
        self._robot_idx = ROBOT_LIST.index(robot)

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("linear_speed", 1.0)      # m/s at full stick
        self.declare_parameter("angular_speed", 1.0)     # rad/s at full stick
        self.declare_parameter("publish_rate", 50.0)     # Hz

        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value
        self.pub_rate = self.get_parameter("publish_rate").value

        # ── Speed multiplier (L1/R1 adjustable) ──────────────────────────
        self.speed_scale = 0.75

        # ── Joystick state ────────────────────────────────────────────────
        self.axes = {
            AX_LX: DS_AXIS_CENTER, AX_LY: DS_AXIS_CENTER,
            AX_RX: DS_AXIS_CENTER, AX_RY: DS_AXIS_CENTER,
            AX_L2: 0, AX_R2: 0,
            AX_DX: 0, AX_DY: 0,
        }
        self.buttons = {}
        self.r1_held = False       # dead man's switch
        self.sent_zero = False     # track if we already sent zero on release

        # ── Publishers (one per robot, switch at runtime) ────────────────
        self._cmd_pubs = {}
        for name, topic in ROBOT_TOPICS.items():
            self._cmd_pubs[name] = self.create_publisher(Twist, topic, 1)
        self.cmd_pub = self._cmd_pubs[self.active_robot]

        # ── Timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.pub_rate, self._control_tick)

        # ── Start evdev reader thread ─────────────────────────────────────
        self._find_and_start_reader()

        self._log_active_robot()
        self.get_logger().info(
            "  R1 (hold)   → Dead man's switch (must hold to drive)\n"
            "  Left stick  → linear velocity (X/Y)\n"
            "  Right stick X → angular velocity (yaw)\n"
            "  L1          → Speed -\n"
            "  L2 trigger  → Fine-control (hold)\n"
            "  R2 (press)  → Switch robot (reachy / spot)\n"
            "  Triangle    → Print speed\n"
            f"\n  linear_speed={self.linear_speed} m/s"
            f"\n  angular_speed={self.angular_speed} rad/s"
        )

    # ── Robot switching ────────────────────────────────────────────────

    @property
    def active_robot(self):
        return ROBOT_LIST[self._robot_idx]

    def _switch_robot(self):
        # Stop the current robot first
        self.cmd_pub.publish(Twist())
        # Cycle to next robot
        self._robot_idx = (self._robot_idx + 1) % len(ROBOT_LIST)
        self.cmd_pub = self._cmd_pubs[self.active_robot]
        self.sent_zero = False
        self._log_active_robot()

    def _log_active_robot(self):
        robot = self.active_robot
        topic = ROBOT_TOPICS[robot]
        bar = "=" * 44
        self.get_logger().info(
            f"\n{bar}\n"
            f"  ACTIVE ROBOT : {robot.upper()}\n"
            f"  TOPIC        : {topic}\n"
            f"{bar}"
        )

    # ── evdev reader ──────────────────────────────────────────────────────

    def _find_and_start_reader(self):
        dev = None
        for path in evdev.list_devices():
            d = evdev.InputDevice(path)
            if "DualSense" in d.name:
                dev = d
                break
        if dev is None:
            self.get_logger().error("DualSense not found! Connect and restart.")
            return
        self.get_logger().info(f"Found DualSense at {dev.path}")
        t = threading.Thread(target=self._evdev_loop, args=(dev,), daemon=True)
        t.start()

    def _evdev_loop(self, dev):
        for event in dev.read_loop():
            if event.type == ecodes.EV_ABS:
                self.axes[event.code] = event.value
            elif event.type == ecodes.EV_KEY:
                if event.code == BTN_R1:
                    self.r1_held = (event.value == 1)
                elif event.value == 1:  # other button press
                    self._on_button(event.code)

    def _on_button(self, code):
        if code == BTN_R2:
            self._switch_robot()
        elif code == BTN_TRIANGLE:
            self.get_logger().info(f"Speed: {self.speed_scale:.2f}x")
        elif code == BTN_L1:
            self.speed_scale = max(0.1, self.speed_scale - 0.25)
            self.get_logger().info(f"Speed: {self.speed_scale:.2f}x")
        elif code == BTN_R1:
            self.speed_scale = min(3.0, self.speed_scale + 0.25)
            self.get_logger().info(f"Speed: {self.speed_scale:.2f}x")

    # ── Control loop ──────────────────────────────────────────────────────

    def _control_tick(self):
        if not self.r1_held:
            # Send one zero twist on release, then stop publishing
            if not self.sent_zero:
                self.cmd_pub.publish(Twist())
                self.sent_zero = True
            return

        self.sent_zero = False

        # Read sticks
        lx = _normalize_stick(self.axes[AX_LX])
        ly = _normalize_stick(self.axes[AX_LY])
        rx = _normalize_stick(self.axes[AX_RX])

        # L2 fine control: 0=normal, 255=maximum slow-down (0.2x)
        l2_raw = self.axes[AX_L2]
        fine_factor = 1.0 - 0.8 * (l2_raw / 255.0)

        scale = self.speed_scale * fine_factor

        # LY up (negative raw) → forward (+X), LX right → left (-Y for ROS)
        cmd = Twist()
        cmd.linear.x = -ly * self.linear_speed * scale
        cmd.linear.y = -lx * self.linear_speed * scale
        cmd.angular.z = -rx * self.angular_speed * scale

        self.cmd_pub.publish(cmd)


def main():
    parser = argparse.ArgumentParser(description="DualSense PS5 teleop")
    parser.add_argument(
        "robot",
        nargs="?",
        default="reachy",
        choices=ROBOT_TOPICS.keys(),
        help="Robot to control: reachy (default) or spot",
    )
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = DualSenseTeleop(robot=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
