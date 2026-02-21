#!/usr/bin/env python3
"""
ROS2 services for Reachy arm control:

  /reachy/arm_on   – torque ON (both arms + gripper warmup)
  /reachy/arm_up   – move to the ready (up) pose
  /reachy/arm_down – move to the stow (down) pose
  /reachy/arm_off  – torque OFF (both arms)
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import DynamicJointState, InterfaceValue
from tf2_ros import Buffer, TransformListener

# ---- Predefined poses (4x4 homogeneous, arm-tip in torso frame) ----------

POSE_DOWN = np.array([
    [0.9668163527361653, -0.12825098100552948, -0.2209475638088324, 0.02642997791631217],
    [0.16118069519858105, 0.9772216339814511, 0.13805311142428067, -0.21534684606020874],
    [0.19820929235845847, -0.1690844876082429, 0.9654654382591937, -0.6542274965885696],
    [0.0, 0.0, 0.0, 1.0],
])

POSE_UP = np.array([
    [0.07258878540034808, -0.5459663755430385, -0.834656567104399, 0.13250980557880576],
    [0.09896798853414465, 0.8366769296767651, -0.5386808448350162, -0.19161335317236736],
    [0.9924395223284931, -0.04350209331812946, 0.11476655609250075, -0.292346821578255],
    [0.0, 0.0, 0.0, 1.0],
])

L_ARM_JOINTS = ["l_shoulder", "l_elbow", "l_wrist", "l_hand"]
R_ARM_JOINTS = ["r_shoulder", "r_elbow", "r_wrist", "r_hand"]

TORQUE_SETTLE_SEC = 2.0
POSE_SETTLE_SEC = 7.0


def matrix_to_pose_stamped(T: np.ndarray, stamp) -> PoseStamped:
    """Convert a 4x4 homogeneous matrix to a PoseStamped in torso frame."""
    pos = T[:3, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]

    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "torso"
    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


class ArmOnOffService(Node):
    def __init__(self):
        super().__init__("arm_on_off_service")

        self._torque_pub = self.create_publisher(
            DynamicJointState, "/dynamic_joint_commands", 10
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, "/target_ee_pose_torso", 10
        )
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, "/gripper_forward_position_controller/commands", 10
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_service(Trigger, "/reachy/arm_on", self._on_arm_on)
        self.create_service(Trigger, "/reachy/arm_up", self._on_arm_up)
        self.create_service(Trigger, "/reachy/arm_down", self._on_arm_down)
        self.create_service(Trigger, "/reachy/arm_off", self._on_arm_off)

        self.get_logger().info(
            "Arm services ready: /reachy/arm_on, /reachy/arm_up, "
            "/reachy/arm_down, /reachy/arm_off"
        )

    # ------------------------------------------------------------------
    def _set_torque(self, joints: list[str], on: bool):
        val = 1.0 if on else 0.0
        msg = DynamicJointState()
        msg.joint_names = joints
        msg.interface_values = [
            InterfaceValue(interface_names=["torque"], values=[val])
            for _ in joints
        ]
        self._torque_pub.publish(msg)
        label = "ON" if on else "OFF"
        self.get_logger().info(f"Torque {label}: {joints}")

    def _set_gripper(self, value: float):
        msg = Float64MultiArray()
        msg.data = [value, value]
        self._gripper_pub.publish(msg)
        self.get_logger().info(f"Gripper set to {value}")

    def _send_pose(self, T: np.ndarray):
        msg = matrix_to_pose_stamped(T, self.get_clock().now().to_msg())
        self._pose_pub.publish(msg)
        self.get_logger().info("Sent target pose")

    def _send_pose_msg(self, msg: PoseStamped):
        self._pose_pub.publish(msg)
        self.get_logger().info("Sent current arm pose as target")

    def _get_current_ee_pose(self) -> PoseStamped | None:
        """Look up r_arm_tip -> torso from TF."""
        try:
            t = self._tf_buffer.lookup_transform(
                "torso", "r_arm_tip", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "torso"
        msg.pose.position.x = t.transform.translation.x
        msg.pose.position.y = t.transform.translation.y
        msg.pose.position.z = t.transform.translation.z
        msg.pose.orientation = t.transform.rotation
        return msg

    # ------------------------------------------------------------------
    def _on_arm_on(self, _request, response):
        """Warmup: both arms torque ON + gripper open/close."""
        # Read current pose so the right arm doesn't jump when torque enables
        current = self._get_current_ee_pose()
        if current is not None:
            self._send_pose_msg(current)
        else:
            self._send_pose(POSE_DOWN)
        time.sleep(0.1)

        self._set_torque(L_ARM_JOINTS, True)
        time.sleep(TORQUE_SETTLE_SEC)
        self._set_torque(R_ARM_JOINTS, True)
        time.sleep(TORQUE_SETTLE_SEC)
        self._set_gripper(0.0)
        time.sleep(TORQUE_SETTLE_SEC)
        self._set_gripper(2.6)

        response.success = True
        response.message = "Both arms ON, grippers warmed up"
        return response

    def _on_arm_up(self, _request, response):
        """Move to ready (up) pose."""
        self._send_pose(POSE_UP)

        response.success = True
        response.message = "Moving to up pose"
        return response

    def _on_arm_down(self, _request, response):
        """Move to stow (down) pose."""
        self._send_pose(POSE_DOWN)

        response.success = True
        response.message = "Moving to down pose"
        return response

    def _on_arm_off(self, _request, response):
        """Torque OFF for both arms."""
        self._set_torque(L_ARM_JOINTS, False)
        time.sleep(0.5)
        self._set_torque(R_ARM_JOINTS, False)

        response.success = True
        response.message = "Both arms OFF"
        return response


def main():
    rclpy.init()
    node = ArmOnOffService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
