#!/usr/bin/env python3
"""
Command end-effector poses with automatic fallback to nearest feasible pose.

Subscribes to a target EE pose, looks it up against the sampled workspace,
and commands the joint configuration of the nearest reachable pose.  If the
target is within `feasible_thresh` of a known sample it is considered
feasible; otherwise the nearest sample is used as a fallback.

Published feedback:
  /ee_command_feedback  (geometry_msgs/PoseStamped)  — the actual pose
                                                        being commanded

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  python3 ee_pose_commander.py --ros-args \
      -p csv_path:=/home/marwan/marwan_ws/reachy_sim/ee_poses_right.csv \
      -p arm:=right \
      -p feasible_thresh:=0.02 \
      -p orientation_weight:=0.1
"""

import csv
import time
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

R_ARM_JOINTS = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
]
L_ARM_JOINTS = [
    "l_shoulder_pitch", "l_shoulder_roll", "l_elbow_yaw", "l_elbow_pitch",
    "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw",
]

CMD_RATE = 50.0  # Hz for holding the joint command


class EEPoseCommander(Node):
    def __init__(self):
        super().__init__("ee_pose_commander")

        self.declare_parameter(
            "csv_path",
            str(Path(__file__).parent / "ee_poses_right.csv"),
        )
        self.declare_parameter("arm", "right")
        # Max distance (metres) to consider a target directly feasible
        self.declare_parameter("feasible_thresh", 0.02)
        # Weight for orientation error relative to position error in the
        # combined nearest-neighbour metric.  0 = position only.
        self.declare_parameter("orientation_weight", 0.1)

        csv_path = self.get_parameter("csv_path").value
        self.arm = self.get_parameter("arm").value
        self.feasible_thresh = self.get_parameter("feasible_thresh").value
        self.ori_weight = self.get_parameter("orientation_weight").value

        self.joints_key = R_ARM_JOINTS if self.arm == "right" else L_ARM_JOINTS

        # ── Load workspace samples ──────────────────────────────────────
        self.joint_configs, self.ee_positions, self.ee_quats = self._load_csv(
            csv_path
        )
        self.n_samples = len(self.ee_positions)
        self.get_logger().info(
            f"Loaded {self.n_samples} workspace samples from {csv_path}"
        )

        # KD-tree on positions for fast lookup
        self.pos_tree = KDTree(self.ee_positions)

        # ── ROS interfaces ──────────────────────────────────────────────
        ctrl_topic = (
            "/r_arm_forward_position_controller/commands"
            if self.arm == "right"
            else "/l_arm_forward_position_controller/commands"
        )
        self.joint_pub = self.create_publisher(Float64MultiArray, ctrl_topic, 10)

        latching = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.fb_pub = self.create_publisher(
            PoseStamped, "/ee_command_feedback", latching
        )

        self.create_subscription(
            PoseStamped, "/target_ee_pose", self._on_target, 10
        )

        # Hold the last commanded joints at CMD_RATE
        self._active_joints = None
        self.create_timer(1.0 / CMD_RATE, self._hold_cmd)

        self.get_logger().info(
            f"Ready. Publish a PoseStamped to /target_ee_pose "
            f"(frame should be 'torso')"
        )

    # ── CSV loading ─────────────────────────────────────────────────────
    def _load_csv(self, path: str):
        joint_configs = []
        positions = []
        quats = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                jv = [float(row[j]) for j in self.joints_key]
                pos = [float(row["ee_x"]), float(row["ee_y"]), float(row["ee_z"])]
                quat = [
                    float(row["ee_qx"]),
                    float(row["ee_qy"]),
                    float(row["ee_qz"]),
                    float(row["ee_qw"]),
                ]
                joint_configs.append(jv)
                positions.append(pos)
                quats.append(quat)
        return (
            np.array(joint_configs),
            np.array(positions),
            np.array(quats),
        )

    # ── Nearest-neighbour search ────────────────────────────────────────
    def _find_nearest(self, target_pos: np.ndarray, target_quat: np.ndarray):
        """Return (index, pos_distance, is_feasible)."""
        if self.ori_weight == 0.0:
            dist, idx = self.pos_tree.query(target_pos)
            return idx, dist, dist <= self.feasible_thresh

        # Position + orientation weighted search:
        # Query K nearest by position, then re-rank with orientation cost
        k = min(50, self.n_samples)
        dists, idxs = self.pos_tree.query(target_pos, k=k)

        target_rot = Rotation.from_quat(target_quat)  # [x,y,z,w]
        best_cost = np.inf
        best_idx = idxs[0]
        best_pos_dist = dists[0]

        for d, i in zip(dists, idxs):
            sample_rot = Rotation.from_quat(self.ee_quats[i])
            # Geodesic angular distance (radians)
            ori_dist = (target_rot.inv() * sample_rot).magnitude()
            cost = d + self.ori_weight * ori_dist
            if cost < best_cost:
                best_cost = cost
                best_idx = i
                best_pos_dist = d

        return best_idx, best_pos_dist, best_pos_dist <= self.feasible_thresh

    # ── Callbacks ───────────────────────────────────────────────────────
    def _on_target(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        target_pos = np.array([p.x, p.y, p.z])
        target_quat = np.array([o.x, o.y, o.z, o.w])

        idx, pos_dist, feasible = self._find_nearest(target_pos, target_quat)

        actual_pos = self.ee_positions[idx]
        actual_quat = self.ee_quats[idx]
        joint_cmd = self.joint_configs[idx].tolist()

        if feasible:
            self.get_logger().info(
                f"FEASIBLE  target=({p.x:.3f},{p.y:.3f},{p.z:.3f})  "
                f"dist={pos_dist:.4f}m"
            )
        else:
            self.get_logger().warn(
                f"NOT FEASIBLE  target=({p.x:.3f},{p.y:.3f},{p.z:.3f})  "
                f"nearest=({actual_pos[0]:.3f},{actual_pos[1]:.3f},{actual_pos[2]:.3f})  "
                f"dist={pos_dist:.4f}m  — falling back"
            )

        # Command joints
        self._active_joints = joint_cmd

        # Publish feedback pose (what we're actually commanding)
        fb = PoseStamped()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.header.frame_id = "torso"
        fb.pose.position.x = float(actual_pos[0])
        fb.pose.position.y = float(actual_pos[1])
        fb.pose.position.z = float(actual_pos[2])
        fb.pose.orientation.x = float(actual_quat[0])
        fb.pose.orientation.y = float(actual_quat[1])
        fb.pose.orientation.z = float(actual_quat[2])
        fb.pose.orientation.w = float(actual_quat[3])
        self.fb_pub.publish(fb)

    def _hold_cmd(self):
        """Keep publishing active joint command so the controller holds position."""
        if self._active_joints is not None:
            msg = Float64MultiArray()
            msg.data = self._active_joints
            self.joint_pub.publish(msg)


def main():
    rclpy.init()
    node = EEPoseCommander()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
