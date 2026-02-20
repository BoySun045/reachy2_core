#!/usr/bin/env python3
"""
DualSense PS5 base teleop for Reachy2.

Reads the DualSense controller via evdev and publishes velocity commands
to /cmd_vel (Twist) to drive the mobile base.

Controls:
  R1 (hold)          → dead man's switch (must hold to drive)
  Left stick X/Y     → base linear X/Y (forward-back, strafe)
  Right stick X      → base angular Z (rotate)
  L1                 → decrease speed
  L2 analog          → fine-control mode (proportional slow-down)
  Triangle           → print current speed settings

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  /usr/bin/python3 dualsense_base_teleop.py
"""

import threading
import numpy as np

import evdev
from evdev import ecodes

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── DualSense constants ──────────────────────────────────────────────────

DS_AXIS_CENTER = 128
DS_AXIS_MAX = 127.0
DS_DEADZONE = 15

# Axis codes
AX_LX = ecodes.ABS_X
AX_LY = ecodes.ABS_Y
AX_L2 = ecodes.ABS_Z
AX_RX = ecodes.ABS_RX
AX_RY = ecodes.ABS_RY
AX_R2 = ecodes.ABS_RZ
AX_DX = ecodes.ABS_HAT0X
AX_DY = ecodes.ABS_HAT0Y

# Button codes
BTN_CROSS    = ecodes.BTN_SOUTH
BTN_CIRCLE   = ecodes.BTN_EAST
BTN_TRIANGLE = ecodes.BTN_NORTH
BTN_SQUARE   = ecodes.BTN_WEST
BTN_L1       = ecodes.BTN_TL
BTN_R1       = ecodes.BTN_TR


def _normalize_stick(raw, center=DS_AXIS_CENTER, deadzone=DS_DEADZONE):
    """Normalize 0-255 raw value to -1..+1 with deadzone."""
    val = raw - center
    if abs(val) < deadzone:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return np.clip(sign * (abs(val) - deadzone) / (DS_AXIS_MAX - deadzone), -1.0, 1.0)


# ── ROS2 Node ─────────────────────────────────────────────────────────────

class DualSenseBaseTeleop(Node):
    def __init__(self):
        super().__init__("dualsense_base_teleop")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("max_linear_vel", 0.5)    # m/s at full stick
        self.declare_parameter("max_angular_vel", 1.0)   # rad/s at full stick
        self.declare_parameter("publish_rate", 20.0)     # Hz

        self.max_lin = self.get_parameter("max_linear_vel").value
        self.max_ang = self.get_parameter("max_angular_vel").value
        self.pub_rate = self.get_parameter("publish_rate").value

        # ── Speed multiplier (L1 adjustable) ─────────────────────────────
        self.speed_scale = 1.0

        # ── Joystick state ────────────────────────────────────────────────
        self.axes = {
            AX_LX: DS_AXIS_CENTER, AX_LY: DS_AXIS_CENTER,
            AX_RX: DS_AXIS_CENTER, AX_RY: DS_AXIS_CENTER,
            AX_L2: 0, AX_R2: 0,
            AX_DX: 0, AX_DY: 0,
        }
        self.r1_held = False       # dead man's switch
        self.sent_zero = False     # track if we already sent zero on release

        # ── Publisher ─────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.pub_rate, self._control_tick)

        # ── Start evdev reader thread ─────────────────────────────────────
        self._find_and_start_reader()

        self.get_logger().info(
            "DualSense base teleop started.\n"
            "  R1 (hold)   → Dead man's switch (must hold to drive)\n"
            "  Left stick  → linear X/Y (base frame)\n"
            "  Right stick X → angular Z (yaw)\n"
            "  L1          → Speed -\n"
            "  L2 trigger  → Fine-control (hold)\n"
            "  Triangle    → Print speed\n"
            f"\n  max_linear_vel={self.max_lin} m/s"
            f"\n  max_angular_vel={self.max_ang} rad/s"
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
                # Track R1 hold/release
                if event.code == BTN_R1:
                    self.r1_held = (event.value == 1)
                elif event.value == 1:  # other button press
                    self._on_button(event.code)

    def _on_button(self, code):
        if code == BTN_TRIANGLE:
            self.get_logger().info(
                f"Speed scale: {self.speed_scale:.2f}x | "
                f"max_lin={self.max_lin:.2f} m/s | max_ang={self.max_ang:.2f} rad/s"
            )
        elif code == BTN_L1:
            self.speed_scale = max(0.1, self.speed_scale - 0.25)
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

        msg = Twist()
        # LY up (negative raw) → forward (+X in base frame)
        # LX right (positive raw) → strafe right (-Y in base frame)
        msg.linear.x = -ly * self.max_lin * scale
        msg.linear.y = -lx * self.max_lin * scale
        # RX right → rotate clockwise (negative angular.z)
        msg.angular.z = -rx * self.max_ang * scale

        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = DualSenseBaseTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
