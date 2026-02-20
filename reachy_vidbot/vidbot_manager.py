#!/usr/bin/env python3
"""
VidBot grasp manager node.

Subscribes to vidbot pre/post grasp pose arrays, then sequences:
  1. Publish pre-grasp path → monitor EE until close to last waypoint
  2. Close gripper
  3. Publish post-grasp path → monitor EE until close to last waypoint
  4. Open gripper

Topics:
  Subscribes:
    /vidbot/pre_poses   (PoseArray)  — pre-grasp trajectory from vidbot
    /vidbot/post_poses  (PoseArray)  — post-grasp trajectory from vidbot
    /tf                              — TF tree for r_arm_tip pose

  Publishes:
    /target_ee_path_torso (PoseArray) — path for torso_ik_controller_fast
    /gripper_forward_position_controller/commands (Float64MultiArray)

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  /usr/bin/python3 vidbot_manager.py
"""

import json
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64MultiArray, String
import tf2_ros

from vidbot_utils import (
    T_BASE_CAM,
    T_BASE_EE_DEFAULT,
    get_action_type,
    mat4_to_pose,
    pose_to_mat4,
    shift_pose_along_local_z,
)


class VidBotManager(Node):
    def __init__(self):
        super().__init__("vidbot_manager")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("convergence_timeout", 10.0)   # seconds 30
        self.declare_parameter("pos_threshold", 0.02)         # metres 0.02
        self.declare_parameter("ori_threshold", 0.15)         # radians 0.15
        self.declare_parameter("check_rate", 5.0)             # Hz for EE polling

        self.convergence_timeout = self.get_parameter("convergence_timeout").value
        self.pos_threshold = self.get_parameter("pos_threshold").value
        self.ori_threshold = self.get_parameter("ori_threshold").value
        self.check_rate = self.get_parameter("check_rate").value

        # ── State ─────────────────────────────────────────────────────────
        self.pre_poses = None   # PoseArray
        self.post_poses = None  # PoseArray
        self.busy = False
        self.current_object = None   # object name from /vidbot/trigger
        self.current_instruction = None  # instruction from /vidbot/trigger

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ee_frame = "r_arm_tip"
        self.reference_frame = "torso"

        # ── Publishers ────────────────────────────────────────────────────
        self.path_pub = self.create_publisher(
            PoseArray, "/target_ee_path_torso", 10
        )
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/gripper_forward_position_controller/commands", 10
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            PoseArray, "/vidbot/pre_poses", self._pre_cb, 10
        )
        self.create_subscription(
            PoseArray, "/vidbot/post_poses", self._post_cb, 10
        )
        self.create_subscription(String, "/vidbot/trigger", self._trigger_cb, 10)

        self.get_logger().info("VidBot manager ready. Waiting for pre/post poses...")

    # ── TF helper ──────────────────────────────────────────────────────────

    def _lookup_tf_as_matrix(self, target_frame, source_frame):
        """Look up a TF transform and return it as a 4x4 numpy matrix."""
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            T = np.eye(4)
            tr = t.transform.translation
            ro = t.transform.rotation
            T[:3, :3] = Rotation.from_quat([ro.x, ro.y, ro.z, ro.w]).as_matrix()
            T[0, 3] = tr.x
            T[1, 3] = tr.y
            T[2, 3] = tr.z
            return T
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=5.0)
            return None

    def _get_ee_pose_in_torso(self):
        """Look up r_arm_tip pose in torso frame via TF."""
        return self._lookup_tf_as_matrix(self.reference_frame, self.ee_frame)

    def _get_T_torso_cam(self):
        """Return hardcoded camera → torso extrinsic calibration."""
        return T_BASE_CAM

    # ── Vidbot callbacks ──────────────────────────────────────────────────

    def _trigger_cb(self, msg: String):
        """Capture the object name and instruction from the prompt node."""
        try:
            data = json.loads(msg.data)
            self.current_object = data.get("object", "").strip().lower()
            self.current_instruction = data.get("instruction", "").strip().lower()
            self.get_logger().info(
                f"Trigger received: object='{self.current_object}', "
                f"instruction='{self.current_instruction}'"
            )
        except json.JSONDecodeError:
            self.current_object = None
            self.current_instruction = None

    def _pre_cb(self, msg: PoseArray):
        self.pre_poses = msg
        self.get_logger().info(f"Received pre-grasp path: {len(msg.poses)} poses")
        self._try_execute()

    def _post_cb(self, msg: PoseArray):
        self.post_poses = msg
        self.get_logger().info(f"Received post-grasp path: {len(msg.poses)} poses")
        self._try_execute()

    # ── Execution ─────────────────────────────────────────────────────────

    def _try_execute(self):
        """Start grasp sequence when both paths are available."""
        if self.busy:
            self.get_logger().warn("Already executing, ignoring new trigger")
            return
        if self.pre_poses is None or self.post_poses is None:
            self.get_logger().info("Waiting for both pre and post poses...")
            return

        self.busy = True
        t = threading.Thread(target=self._run_sequence, daemon=True)
        t.start()

    def _truncate_path_early(self, poses, offset_m=0.10):
        """Remove poses from the end so the path stops offset_m before the original endpoint.

        Walks backwards from the last pose and keeps only poses whose
        distance to the original endpoint is >= offset_m.
        Returns the truncated list of Pose objects.
        """
        if len(poses) < 2:
            return list(poses)

        final_pos = np.array([
            poses[-1].position.x,
            poses[-1].position.y,
            poses[-1].position.z,
        ])

        # Walk backwards to find the cut point
        for i in range(len(poses) - 1, -1, -1):
            p = poses[i]
            pos = np.array([p.position.x, p.position.y, p.position.z])
            if np.linalg.norm(pos - final_pos) >= offset_m:
                truncated = list(poses[: i + 1])
                return truncated

        # All poses are within offset_m of the end — keep just the first pose
        return [poses[0]]

    def _skip_path_start(self, poses, ref_pos, offset_m=0.10):
        """Skip initial poses that are within offset_m of ref_pos.

        Returns the trimmed list of Pose objects.
        """
        if len(poses) < 2:
            return list(poses)

        for i, p in enumerate(poses):
            pos = np.array([p.position.x, p.position.y, p.position.z])
            if np.linalg.norm(pos - ref_pos) >= offset_m:
                return list(poses[i:])

        # All poses are within offset_m — keep just the last
        return [poses[-1]]

    def _run_sequence(self):
        # Grab local copies immediately so callbacks can accept new data
        local_pre = self.pre_poses
        local_post = self.post_poses
        local_instruction = self.current_instruction
        self.pre_poses = None
        self.post_poses = None
        self.current_object = None
        self.current_instruction = None

        try:
            action_type = get_action_type(local_instruction)
            self.get_logger().info(f"Action type: {action_type} (instruction: '{local_instruction}')")

            pre_poses_list = list(local_pre.poses)
            post_poses_list = list(local_post.poses)

            # Compensate palm-to-tip offset for press actions:
            # tracking frame is at the palm, tip is 3cm further along local Z
            if action_type == "press":
                pre_poses_list = [shift_pose_along_local_z(p, 0.03) for p in pre_poses_list]
                post_poses_list = [shift_pose_along_local_z(p, 0.03) for p in post_poses_list]

            # Build PoseArrays
            pre_msg = PoseArray()
            pre_msg.header = local_pre.header
            pre_msg.poses = pre_poses_list

            post_msg = PoseArray()
            post_msg.header = local_post.header
            post_msg.poses = post_poses_list

            if action_type == "press":
                # ── Press/push/click: half-close gripper → pre → post → return to default ──
                self.get_logger().info("Press action: half-closing gripper before approach...")
                msg = Float64MultiArray()
                msg.data = [0.0, 0.0]
                self.gripper_pub.publish(msg)
                time.sleep(0.5)

                self.get_logger().info(
                    f"Step 1/3: Sending pre path ({len(pre_poses_list)} poses)..."
                )
                self._send_path_and_wait(pre_msg, pre_poses_list[-1])

                self.get_logger().info(
                    f"Step 2/3: Sending post path ({len(post_poses_list)} poses)..."
                )
                self._send_path_and_wait(post_msg, post_poses_list[-1])

                self.get_logger().info("Step 3/3: Returning arm to default EE pose...")
                default_pose = mat4_to_pose(T_BASE_EE_DEFAULT)
                home_msg = PoseArray()
                home_msg.header = local_pre.header
                home_msg.poses = [default_pose]
                self._send_path_and_wait(home_msg, default_pose)

            elif action_type == "place":
                # ── Put/leave/drop/place: pre → post → open gripper → return to default ──
                self.get_logger().info(
                    f"Step 1/4: Sending pre path ({len(pre_poses_list)} poses)..."
                )
                self._send_path_and_wait(pre_msg, pre_poses_list[-1])

                self.get_logger().info(
                    f"Step 2/4: Sending post path ({len(post_poses_list)} poses)..."
                )
                self._send_path_and_wait(post_msg, post_poses_list[-1])

                self.get_logger().info("Step 3/4: Opening gripper...")
                self._open_gripper()
                time.sleep(0.5)

                self.get_logger().info("Step 4/4: Returning arm to default EE pose...")
                default_pose = mat4_to_pose(T_BASE_EE_DEFAULT)
                home_msg = PoseArray()
                home_msg.header = local_pre.header
                home_msg.poses = [default_pose]
                self._send_path_and_wait(home_msg, default_pose)

            else:
                # ── Other actions (pick, etc.): open gripper → pre → close → post ──
                self.get_logger().info("Opening gripper fully before approach...")
                self._open_gripper()
                time.sleep(0.5)

                self.get_logger().info(
                    f"Step 1/3: Sending pre path ({len(pre_poses_list)} poses)..."
                )
                self._send_path_and_wait(pre_msg, pre_poses_list[-1])

                self.get_logger().info("Step 2/3: Closing gripper...")
                self._close_gripper()
                time.sleep(0.5)

                self.get_logger().info(
                    f"Step 3/3: Sending post path ({len(post_poses_list)} poses)..."
                )
                self._send_path_and_wait(post_msg, post_poses_list[-1])

            self.get_logger().info("Sequence complete!")

        except Exception as e:
            self.get_logger().error(f"Grasp sequence failed: {e}")
        finally:
            self.busy = False
            # If new data arrived during execution, kick off next sequence
            self._try_execute()

    def _wait_for_first_waypoint(self, first_pose: Pose, arrival_threshold=0.08, max_wait=5.0):
        """Block until the EE is close to the first waypoint of the path."""
        T_first = pose_to_mat4(first_pose)
        dt = 1.0 / self.check_rate
        t0 = time.monotonic()

        while True:
            elapsed = time.monotonic() - t0
            if elapsed > max_wait:
                self.get_logger().warn(
                    f"First-waypoint wait timeout ({max_wait}s). Starting convergence timer."
                )
                break

            T_ee = self._get_ee_pose_in_torso()
            if T_ee is None:
                time.sleep(dt)
                continue

            pos_err = np.linalg.norm(T_ee[:3, 3] - T_first[:3, 3])

            if pos_err < arrival_threshold:
                self.get_logger().info(
                    f"Reached first waypoint (pos_err={pos_err:.4f}m, waited {elapsed:.1f}s)"
                )
                break

            time.sleep(dt)

    def _send_path_and_wait(self, pose_array: PoseArray, goal_pose: Pose, timeout=None, first_wp_timeout=15.0):
        """Publish path, then poll EE pose until it converges to goal_pose."""
        self.path_pub.publish(pose_array)

        # Wait for the arm to reach the first waypoint before starting timeout
        if len(pose_array.poses) > 0:
            self._wait_for_first_waypoint(pose_array.poses[0], max_wait=first_wp_timeout)

        # Target transform (in torso frame from vidbot)
        T_goal = pose_to_mat4(goal_pose)

        effective_timeout = timeout if timeout is not None else self.convergence_timeout
        t0 = time.monotonic()
        dt = 1.0 / self.check_rate

        while True:
            elapsed = time.monotonic() - t0
            if elapsed > effective_timeout:
                self.get_logger().warn(
                    f"Convergence timeout ({effective_timeout}s). Continuing."
                )
                break

            T_ee = self._get_ee_pose_in_torso()
            if T_ee is None:
                time.sleep(dt)
                continue

            # Both T_ee and T_goal are in torso frame — compare directly
            pos_err = np.linalg.norm(T_ee[:3, 3] - T_goal[:3, 3])

            R_ee = Rotation.from_matrix(T_ee[:3, :3])
            R_goal = Rotation.from_matrix(T_goal[:3, :3])
            ori_err = (R_ee.inv() * R_goal).magnitude()

            if pos_err < self.pos_threshold and ori_err < self.ori_threshold:
                self.get_logger().info(
                    f"Converged: pos_err={pos_err:.4f}m, ori_err={ori_err:.4f}rad "
                    f"({elapsed:.1f}s)"
                )
                break

            time.sleep(dt)

    # ── Gripper ────────────────────────────────────────────────────────────

    def _close_gripper(self):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0]
        self.gripper_pub.publish(msg)
        self.get_logger().info("  Gripper closed")

    def _open_gripper(self):
        msg = Float64MultiArray()
        msg.data = [2.6, 0.0]
        self.gripper_pub.publish(msg)
        self.get_logger().info("  Gripper opened")


def main():
    rclpy.init()
    node = VidBotManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
