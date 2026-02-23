#!/usr/bin/env python3
"""
VidBot grasp manager node for Boston Dynamics Spot.

Subscribes to vidbot pre/post grasp pose arrays, then sequences arm commands
one waypoint at a time via /spot/arm_pose_commands and gripper via Trigger services.

Action types:
  press/push/click → close gripper → pre → post → stow
  put/leave/drop/place → pre → post → open gripper → stow
  other (pick, etc.) → open gripper → pre → close gripper → post → stow

Topics:
  Subscribes:
    /vidbot/pre_poses   (PoseArray)  — pre-grasp trajectory from vidbot
    /vidbot/post_poses  (PoseArray)  — post-grasp trajectory from vidbot
    /vidbot/trigger     (String)     — JSON trigger with object + instruction
    /tf                              — TF tree for EE pose

  Publishes:
    /spot/arm_pose_commands (PoseStamped) — one waypoint at a time

  Service clients:
    /spot/arm_stow        (Trigger) — stow arm after task
    /spot/open_gripper    (Trigger) — open gripper
    /spot/close_gripper   (Trigger) — close gripper

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  python3 vidbot_spot_manager.py
"""

import json
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros

from vidbot_utils import (
    get_action_type,
    mat4_to_pose,
    pose_to_mat4,
    shift_pose_along_local_z,
)


class VidBotSpotManager(Node):
    def __init__(self):
        super().__init__("vidbot_spot_manager")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("base_frame", "spot/body")
        self.declare_parameter("ee_frame", "spot/hand")
        self.declare_parameter("arm_cmd_frame", "body")
        self.declare_parameter("waypoint_pause", 0.1)          # seconds between waypoints
        self.declare_parameter("interp_steps", 10)               # interpolation points between waypoints
        self.declare_parameter("convergence_timeout", 10.0)    # seconds
        self.declare_parameter("pos_threshold", 0.02)          # metres
        self.declare_parameter("ori_threshold", 0.15)          # radians
        self.declare_parameter("check_rate", 5.0)              # Hz for EE polling

        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.arm_cmd_frame = self.get_parameter("arm_cmd_frame").value
        self.waypoint_pause = self.get_parameter("waypoint_pause").value
        self.interp_steps = int(self.get_parameter("interp_steps").value)
        self.convergence_timeout = self.get_parameter("convergence_timeout").value
        self.pos_threshold = self.get_parameter("pos_threshold").value
        self.ori_threshold = self.get_parameter("ori_threshold").value
        self.check_rate = self.get_parameter("check_rate").value

        # ── State ─────────────────────────────────────────────────────────
        self.pre_poses = None   # PoseArray
        self.post_poses = None  # PoseArray
        self.busy = False
        self.current_object = None
        self.current_instruction = None

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Publisher: arm commands ────────────────────────────────────────
        self.arm_cmd_pub = self.create_publisher(
            PoseStamped, "/spot/arm_pose_commands", 10
        )

        # ── Service clients ───────────────────────────────────────────────
        self.stow_client = self.create_client(Trigger, "/spot/arm_stow")
        self.open_gripper_client = self.create_client(Trigger, "/spot/open_gripper")
        self.close_gripper_client = self.create_client(Trigger, "/spot/close_gripper")

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            PoseArray, "/vidbot/pre_poses", self._pre_cb, 10
        )
        self.create_subscription(
            PoseArray, "/vidbot/post_poses", self._post_cb, 10
        )
        self.create_subscription(String, "/vidbot/trigger", self._trigger_cb, 10)

        self.get_logger().info("VidBot Spot manager ready. Waiting for pre/post poses...")

    # ── TF helper ────────────────────────────────────────────────────────

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

    def _get_ee_pose(self):
        """Look up EE pose in body frame via TF."""
        return self._lookup_tf_as_matrix(self.base_frame, self.ee_frame)

    # ── Service call helpers ─────────────────────────────────────────────

    def _call_trigger_service(self, client, name, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"Service {name} not available")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"  {name}: OK")
            return True
        msg = future.result().message if future.result() else "timeout"
        self.get_logger().error(f"  {name} failed: {msg}")
        return False

    def _arm_stow(self):
        return self._call_trigger_service(self.stow_client, "/spot/arm_stow")

    def _open_gripper(self):
        return self._call_trigger_service(self.open_gripper_client, "/spot/open_gripper")

    def _close_gripper(self):
        return self._call_trigger_service(self.close_gripper_client, "/spot/close_gripper")

    # ── Arm command helper ───────────────────────────────────────────────

    def _send_arm_pose(self, pose: Pose):
        """Publish a single PoseStamped to /spot/arm_pose_commands."""
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.arm_cmd_frame
        ps.pose = pose
        self.arm_cmd_pub.publish(ps)

    # ── Vidbot callbacks ─────────────────────────────────────────────────

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

    # ── Execution ────────────────────────────────────────────────────────

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

    def _interpolate_poses(self, poses):
        """Insert interpolation points between consecutive poses."""
        if len(poses) < 2 or self.interp_steps <= 0:
            return list(poses)

        result = []
        for i in range(len(poses) - 1):
            p0, p1 = poses[i], poses[i + 1]
            pos0 = np.array([p0.position.x, p0.position.y, p0.position.z])
            pos1 = np.array([p1.position.x, p1.position.y, p1.position.z])
            q0 = [p0.orientation.x, p0.orientation.y, p0.orientation.z, p0.orientation.w]
            q1 = [p1.orientation.x, p1.orientation.y, p1.orientation.z, p1.orientation.w]

            rots = Rotation.from_quat([q0, q1])
            slerp = Slerp([0.0, 1.0], rots)

            result.append(p0)
            for j in range(1, self.interp_steps + 1):
                t = j / (self.interp_steps + 1)
                pos = pos0 + t * (pos1 - pos0)
                rot = slerp([t])
                quat = rot.as_quat()[0]
                p = Pose()
                p.position.x = float(pos[0])
                p.position.y = float(pos[1])
                p.position.z = float(pos[2])
                p.orientation.x = float(quat[0])
                p.orientation.y = float(quat[1])
                p.orientation.z = float(quat[2])
                p.orientation.w = float(quat[3])
                result.append(p)
        result.append(poses[-1])
        return result

    def _send_poses_sequentially(self, poses, label="trajectory"):
        """Send each pose one at a time with a pause between them."""
        interp_poses = self._interpolate_poses(poses)
        self.get_logger().info(
            f"  {label}: {len(poses)} waypoints -> {len(interp_poses)} with interpolation"
        )
        for i, pose in enumerate(interp_poses):
            self._send_arm_pose(pose)
            time.sleep(self.waypoint_pause)

        # Wait for convergence to the last pose
        if interp_poses:
            self._wait_for_convergence(interp_poses[-1])

    def _wait_for_convergence(self, goal_pose: Pose, timeout=None):
        """Poll EE pose until it converges to goal_pose."""
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

            T_ee = self._get_ee_pose()
            if T_ee is None:
                time.sleep(dt)
                continue

            pos_err = np.linalg.norm(T_ee[:3, 3] - T_goal[:3, 3])

            R_ee = Rotation.from_matrix(T_ee[:3, :3])
            R_goal = Rotation.from_matrix(T_goal[:3, :3])
            ori_err = (R_ee.inv() * R_goal).magnitude()

            if pos_err < self.pos_threshold and ori_err < self.ori_threshold:
                self.get_logger().info(
                    f"  Converged: pos_err={pos_err:.4f}m, ori_err={ori_err:.4f}rad "
                    f"({elapsed:.1f}s)"
                )
                break

            time.sleep(dt)

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

            # Compensate palm-to-tip offset for press actions
            if action_type == "press":
                pre_poses_list = [shift_pose_along_local_z(p, 0.03) for p in pre_poses_list]
                post_poses_list = [shift_pose_along_local_z(p, 0.03) for p in post_poses_list]

            if action_type == "press":
                # ── Press/push/click: close gripper → pre → post → stow ──
                self.get_logger().info("Press action: closing gripper before approach...")
                self._close_gripper()
                time.sleep(0.5)

                self.get_logger().info(
                    f"Step 1/3: Sending pre path ({len(pre_poses_list)} poses)..."
                )
                self._send_poses_sequentially(pre_poses_list, "pre")

                self.get_logger().info(
                    f"Step 2/3: Sending post path ({len(post_poses_list)} poses)..."
                )
                self._send_poses_sequentially(post_poses_list, "post")

                self.get_logger().info("Step 3/3: Stowing arm...")
                self._arm_stow()

            elif action_type == "place":
                # ── Put/leave/drop/place: pre → post → open gripper → stow ──
                self.get_logger().info(
                    f"Step 1/4: Sending pre path ({len(pre_poses_list)} poses)..."
                )
                self._send_poses_sequentially(pre_poses_list, "pre")

                self.get_logger().info(
                    f"Step 2/4: Sending post path ({len(post_poses_list)} poses)..."
                )
                self._send_poses_sequentially(post_poses_list, "post")

                self.get_logger().info("Step 3/4: Opening gripper...")
                self._open_gripper()
                time.sleep(0.5)

                self.get_logger().info("Step 4/4: Stowing arm...")
                self._arm_stow()

            else:
                # ── Other actions (pick, etc.): open gripper → pre → close → post → stow ──
                self.get_logger().info("Opening gripper fully before approach...")
                self._open_gripper()
                time.sleep(0.5)

                self.get_logger().info(
                    f"Step 1/4: Sending pre path ({len(pre_poses_list)} poses)..."
                )
                self._send_poses_sequentially(pre_poses_list, "pre")

                self.get_logger().info("Step 2/4: Closing gripper...")
                self._close_gripper()
                time.sleep(0.5)

                self.get_logger().info(
                    f"Step 3/4: Sending post path ({len(post_poses_list)} poses)..."
                )
                self._send_poses_sequentially(post_poses_list, "post")

                self.get_logger().info("Step 4/4: Stowing arm...")
                self._arm_stow()

            self.get_logger().info("Sequence complete!")

        except Exception as e:
            self.get_logger().error(f"Grasp sequence failed: {e}")
        finally:
            self.busy = False
            self._try_execute()


def main():
    rclpy.init()
    node = VidBotSpotManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
