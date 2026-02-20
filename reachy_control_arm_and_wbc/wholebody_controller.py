#!/usr/bin/env python3
"""
Whole-body controller: mobile base (x,y) + arm.

Given a target EE pose in the odom (world) frame:
  1. Transform target into the torso frame
  2. Query the workspace KD-tree for the nearest feasible arm pose
  3. If reachable → command arm joints directly
  4. If not      → compute the base translation needed to bring the
                    target inside the arm's workspace, drive the base
                    there, then command the arm

Subscribe:  /target_ee_pose_world   (PoseStamped, frame: odom)
Publish:    /cmd_vel                (Twist)
            /r_arm_forward_position_controller/commands
            /wholebody_feedback     (PoseStamped — the pose being commanded)

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  python3 wholebody_controller.py --ros-args \
      -p csv_path:=ee_poses_right.csv \
      -p feasible_thresh:=0.05 \
      -p base_linear_speed:=0.25 \
      -p base_pos_tolerance:=0.02
"""

import csv
import time
from pathlib import Path
from enum import Enum, auto

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener

R_ARM_JOINTS = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
]

ARM_CMD_RATE = 50.0


class State(Enum):
    WAITING_FOR_ODOM = auto()
    IDLE = auto()
    DRIVING = auto()
    COMMANDING_ARM = auto()


class WholeBodyController(Node):
    def __init__(self):
        super().__init__("wholebody_controller")

        # Parameters
        self.declare_parameter("csv_path", str(Path(__file__).parent / "ee_poses_right.csv"))
        self.declare_parameter("feasible_thresh", 0.05)
        self.declare_parameter("orientation_weight", 0.1)
        self.declare_parameter("base_linear_speed", 0.25)
        self.declare_parameter("base_pos_tolerance", 0.02)

        self.feasible_thresh = self.get_parameter("feasible_thresh").value
        self.ori_weight = self.get_parameter("orientation_weight").value
        self.base_speed = self.get_parameter("base_linear_speed").value
        self.base_tol = self.get_parameter("base_pos_tolerance").value

        # Load workspace
        csv_path = self.get_parameter("csv_path").value
        self.joint_configs, self.ee_positions, self.ee_quats = self._load_csv(csv_path)
        self.pos_tree = KDTree(self.ee_positions)
        self.get_logger().info(f"Loaded {len(self.ee_positions)} workspace samples")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.arm_pub = self.create_publisher(
            Float64MultiArray, "/r_arm_forward_position_controller/commands", 10
        )
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        latching = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.fb_pub = self.create_publisher(PoseStamped, "/wholebody_feedback", latching)

        # Odom
        self.base_xy = None
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        # Target subscriber
        self.create_subscription(
            PoseStamped, "/target_ee_pose_world", self._on_target, 10
        )

        # State machine
        self.state = State.WAITING_FOR_ODOM
        self._active_joints = None
        self._drive_target_xy = None
        self._pending_target = None  # target to process after driving

        # Single high-rate timer drives everything
        self.create_timer(1.0 / ARM_CMD_RATE, self._tick)

        self.get_logger().info("Waiting for odom...")

    # ── Data loading ────────────────────────────────────────────────────
    def _load_csv(self, path):
        joints, positions, quats = [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                joints.append([float(row[j]) for j in R_ARM_JOINTS])
                positions.append([float(row["ee_x"]), float(row["ee_y"]), float(row["ee_z"])])
                quats.append([float(row[k]) for k in ("ee_qx", "ee_qy", "ee_qz", "ee_qw")])
        return np.array(joints), np.array(positions), np.array(quats)

    # ── Callbacks ───────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.base_xy = np.array([p.x, p.y])

    def _on_target(self, msg: PoseStamped):
        """Queue a target; process immediately if idle, else after drive."""
        self._pending_target = msg
        if self.state == State.IDLE:
            self._process_target()

    # ── Main tick (50 Hz) ───────────────────────────────────────────────
    def _tick(self):
        # Hold arm joints
        if self._active_joints is not None:
            arm_msg = Float64MultiArray()
            arm_msg.data = self._active_joints
            self.arm_pub.publish(arm_msg)

        # State transitions
        if self.state == State.WAITING_FOR_ODOM:
            if self.base_xy is not None:
                self.state = State.IDLE
                self.get_logger().info(
                    f"Odom received. Base at ({self.base_xy[0]:.3f}, {self.base_xy[1]:.3f})\n"
                    "  Publish PoseStamped to /target_ee_pose_world (frame: odom)"
                )
            return

        if self.state == State.DRIVING:
            self._drive_tick()
            return

    # ── Process incoming target ─────────────────────────────────────────
    def _process_target(self):
        msg = self._pending_target
        if msg is None:
            return
        self._pending_target = None

        target_world = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        ])
        target_quat_world = np.array([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        ])

        self.get_logger().info(
            f"Target in odom: ({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f})"
        )

        # Transform target → torso frame
        try:
            t_o2t, R_o2t = self._get_transform("torso", "odom")
        except Exception as e:
            self.get_logger().error(f"TF odom→torso failed: {e}")
            return

        target_torso = R_o2t @ target_world + t_o2t
        target_quat_torso = (
            Rotation.from_matrix(R_o2t) * Rotation.from_quat(target_quat_world)
        ).as_quat()

        # Find nearest workspace point
        idx, pos_dist = self._find_nearest(target_torso, target_quat_torso)

        if pos_dist <= self.feasible_thresh:
            self.get_logger().info(
                f"FEASIBLE from current base. dist={pos_dist:.4f}m"
            )
            self._command_arm(idx)
            return

        # Need to move the base
        ws_point_torso = self.ee_positions[idx]
        offset_torso = target_torso - ws_point_torso

        try:
            t_t2o, R_t2o = self._get_transform("odom", "torso")
        except Exception as e:
            self.get_logger().error(f"TF torso→odom failed: {e}")
            return

        offset_odom = R_t2o @ offset_torso
        new_base_xy = self.base_xy + offset_odom[:2]

        self.get_logger().warn(
            f"NOT FEASIBLE (dist={pos_dist:.3f}m). "
            f"Moving base: ({self.base_xy[0]:.3f},{self.base_xy[1]:.3f}) → "
            f"({new_base_xy[0]:.3f},{new_base_xy[1]:.3f})"
        )

        # Store context for post-drive arm command
        self._drive_target_xy = new_base_xy
        self._drive_world_target = target_world
        self._drive_quat_world = target_quat_world
        self.state = State.DRIVING

    # ── Base driving (called from _tick) ────────────────────────────────
    def _drive_tick(self):
        if self.base_xy is None:
            return

        error = self._drive_target_xy - self.base_xy
        dist = np.linalg.norm(error)

        if dist < self.base_tol:
            # Stop base
            self.base_pub.publish(Twist())
            self.get_logger().info(f"Base arrived. error={dist:.4f}m")
            self.state = State.IDLE
            # Now command the arm from the new position
            self._finalize_after_drive()
            return

        direction = error / dist
        speed = min(self.base_speed, dist * 2.0)
        cmd = Twist()
        cmd.linear.x = float(direction[0] * speed)
        cmd.linear.y = float(direction[1] * speed)
        self.base_pub.publish(cmd)

    def _finalize_after_drive(self):
        """After base arrives, re-check target in new torso frame and command arm."""
        try:
            t_o2t, R_o2t = self._get_transform("torso", "odom")
        except Exception as e:
            self.get_logger().error(f"Post-drive TF failed: {e}")
            return

        target_torso = R_o2t @ self._drive_world_target + t_o2t
        target_quat_torso = (
            Rotation.from_matrix(R_o2t) * Rotation.from_quat(self._drive_quat_world)
        ).as_quat()

        idx, dist = self._find_nearest(target_torso, target_quat_torso)
        self.get_logger().info(
            f"After base move: nearest dist={dist:.4f}m → commanding arm"
        )
        self._command_arm(idx)

        # Process any queued target
        if self._pending_target is not None:
            self._process_target()

    # ── Nearest workspace lookup ────────────────────────────────────────
    def _find_nearest(self, target_pos, target_quat):
        if self.ori_weight == 0.0:
            dist, idx = self.pos_tree.query(target_pos)
            return idx, dist

        k = min(50, len(self.ee_positions))
        dists, idxs = self.pos_tree.query(target_pos, k=k)
        target_rot = Rotation.from_quat(target_quat)
        best_cost, best_idx, best_dist = np.inf, idxs[0], dists[0]
        for d, i in zip(dists, idxs):
            ori_dist = (target_rot.inv() * Rotation.from_quat(self.ee_quats[i])).magnitude()
            cost = d + self.ori_weight * ori_dist
            if cost < best_cost:
                best_cost, best_idx, best_dist = cost, i, d
        return best_idx, best_dist

    # ── TF helper ───────────────────────────────────────────────────────
    def _get_transform(self, target_frame, source_frame):
        t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        p = t.transform.translation
        r = t.transform.rotation
        trans = np.array([p.x, p.y, p.z])
        rot = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        return trans, rot

    # ── Arm command ─────────────────────────────────────────────────────
    def _command_arm(self, ws_idx: int):
        self._active_joints = self.joint_configs[ws_idx].tolist()

        # Publish feedback
        pos = self.ee_positions[ws_idx]
        quat = self.ee_quats[ws_idx]
        try:
            t_t2o, R_t2o = self._get_transform("odom", "torso")
            pos_odom = R_t2o @ pos + t_t2o
            quat_odom = (Rotation.from_matrix(R_t2o) * Rotation.from_quat(quat)).as_quat()
        except Exception:
            pos_odom, quat_odom = pos, quat

        fb = PoseStamped()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.header.frame_id = "odom"
        fb.pose.position.x, fb.pose.position.y, fb.pose.position.z = (
            float(pos_odom[0]), float(pos_odom[1]), float(pos_odom[2])
        )
        fb.pose.orientation.x, fb.pose.orientation.y = float(quat_odom[0]), float(quat_odom[1])
        fb.pose.orientation.z, fb.pose.orientation.w = float(quat_odom[2]), float(quat_odom[3])
        self.fb_pub.publish(fb)


def main():
    rclpy.init()
    node = WholeBodyController()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
