#!/usr/bin/env python3
"""
DualSense PS5 arm teleop for Reachy2 (torso IK controller).

Reads the DualSense controller via evdev and publishes end-effector
pose targets to /target_ee_pose_torso (PoseStamped, torso frame).

Controls:
  R1 (hold)          → dead man's switch (must hold to move)
  Left stick Y       → EE X (forward / back)
  Left stick X       → EE Y (left / right)
  Right stick Y      → EE Z (up / down)
  Right stick X      → EE yaw (rotate wrist)
  D-pad up/down      → EE pitch
  D-pad left/right   → EE roll
  L1                 → decrease speed
  L2 analog          → fine-control mode (proportional slow-down)
  Triangle           → print current EE pose & speed

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  /usr/bin/python3 dualsense_arm_teleop.py
"""

import threading
import numpy as np
from scipy.spatial.transform import Rotation

import evdev
from evdev import ecodes

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from wholebody_ik_controller import arm_fk, R_ARM_JOINTS

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

class DualSenseArmTeleop(Node):
    def __init__(self):
        super().__init__("dualsense_arm_teleop")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("max_linear_speed", 0.15)    # m/s at full stick
        self.declare_parameter("max_angular_speed", 0.8)    # rad/s at full stick
        self.declare_parameter("publish_rate", 20.0)        # Hz

        self.max_lin = self.get_parameter("max_linear_speed").value
        self.max_ang = self.get_parameter("max_angular_speed").value
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
        self.r1_held = False

        # ── Current EE pose in torso frame ────────────────────────────────
        self.current_joints = np.zeros(7)
        self.joints_received = False
        self.ee_pos = None       # [x, y, z]
        self.ee_rot = None       # Rotation object
        self.initialized = False

        # ── Publisher / Subscriber ────────────────────────────────────────
        self.target_pub = self.create_publisher(
            PoseStamped, "/target_ee_pose_torso", 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

        # ── Timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.pub_rate, self._control_tick)

        # ── Start evdev reader thread ─────────────────────────────────────
        self._find_and_start_reader()

        self.get_logger().info(
            "DualSense arm teleop started.\n"
            "  R1 (hold)       → Dead man's switch\n"
            "  Left stick Y    → EE X (forward/back)\n"
            "  Left stick X    → EE Y (left/right)\n"
            "  Right stick Y   → EE Z (up/down)\n"
            "  Right stick X   → EE yaw\n"
            "  D-pad up/down   → EE pitch\n"
            "  D-pad left/right→ EE roll\n"
            "  L1              → Speed -\n"
            "  L2 trigger      → Fine-control (hold)\n"
            "  Triangle        → Print pose\n"
            f"\n  max_linear_speed={self.max_lin} m/s"
            f"\n  max_angular_speed={self.max_ang} rad/s"
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
                elif event.value == 1:
                    self._on_button(event.code)

    def _on_button(self, code):
        if code == BTN_TRIANGLE:
            if self.ee_pos is not None:
                rpy = self.ee_rot.as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f"EE pose (torso): ({self.ee_pos[0]:.3f}, {self.ee_pos[1]:.3f}, "
                    f"{self.ee_pos[2]:.3f}) | RPY: ({rpy[0]:.1f}, {rpy[1]:.1f}, "
                    f"{rpy[2]:.1f}) deg | Speed: {self.speed_scale:.2f}x"
                )
            else:
                self.get_logger().info("No joint states yet")
        elif code == BTN_L1:
            self.speed_scale = max(0.1, self.speed_scale - 0.25)
            self.get_logger().info(f"Speed: {self.speed_scale:.2f}x")

    # ── Joint state callback ──────────────────────────────────────────────

    def _joint_state_cb(self, msg):
        name_list = list(msg.name)
        for i, jname in enumerate(R_ARM_JOINTS):
            if jname in name_list:
                self.current_joints[i] = msg.position[name_list.index(jname)]
        self.joints_received = True

        if not self.initialized:
            T = arm_fk(self.current_joints)
            self.ee_pos = T[:3, 3].copy()
            self.ee_rot = Rotation.from_matrix(T[:3, :3])
            self.initialized = True
            self.get_logger().info(
                f"Initialized EE at ({self.ee_pos[0]:.3f}, {self.ee_pos[1]:.3f}, "
                f"{self.ee_pos[2]:.3f})"
            )

    # ── Control loop ──────────────────────────────────────────────────────

    def _control_tick(self):
        if not self.initialized or not self.r1_held:
            return

        dt = 1.0 / self.pub_rate

        # Read sticks
        lx = _normalize_stick(self.axes[AX_LX])
        ly = _normalize_stick(self.axes[AX_LY])
        rx = _normalize_stick(self.axes[AX_RX])
        ry = _normalize_stick(self.axes[AX_RY])
        dx = self.axes[AX_DX]   # -1, 0, +1
        dy = self.axes[AX_DY]   # -1, 0, +1

        # L2 fine control
        l2_raw = self.axes[AX_L2]
        fine_factor = 1.0 - 0.8 * (l2_raw / 255.0)
        scale = self.speed_scale * fine_factor

        # ── Position delta ────────────────────────────────────────────────
        # LY up (negative raw) → +X (forward in torso frame)
        # LX right (positive raw) → -Y (right in torso frame)
        # RY up (negative raw) → +Z (up in torso frame)
        vx = -ly * self.max_lin * scale
        vy = -lx * self.max_lin * scale
        vz = -ry * self.max_lin * scale

        self.ee_pos[0] += vx * dt
        self.ee_pos[1] += vy * dt
        self.ee_pos[2] += vz * dt

        # ── Orientation delta ─────────────────────────────────────────────
        # RX → yaw, D-pad up/down → pitch, D-pad left/right → roll
        wyaw  = -rx * self.max_ang * scale
        wpitch = -float(dy) * self.max_ang * scale
        wroll  = -float(dx) * self.max_ang * scale

        if abs(wyaw) > 1e-6 or abs(wpitch) > 1e-6 or abs(wroll) > 1e-6:
            delta_rot = Rotation.from_euler(
                'xyz', [wroll * dt, wpitch * dt, wyaw * dt]
            )
            self.ee_rot = self.ee_rot * delta_rot

        # ── Publish target ────────────────────────────────────────────────
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "torso"

        msg.pose.position.x = float(self.ee_pos[0])
        msg.pose.position.y = float(self.ee_pos[1])
        msg.pose.position.z = float(self.ee_pos[2])

        q = self.ee_rot.as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

        self.target_pub.publish(msg)


def main():
    rclpy.init()
    node = DualSenseArmTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
