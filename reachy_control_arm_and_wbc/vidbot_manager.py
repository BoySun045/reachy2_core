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
    /odom               (Odometry)   — base pose for FK
    /joint_states       (JointState) — arm joints for FK

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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from wholebody_ik_controller import wholebody_fk, R_ARM_JOINTS


def _pose_to_T(pose: Pose) -> np.ndarray:
    """Convert a geometry_msgs Pose to a 4x4 transform."""
    T = np.eye(4)
    o = pose.orientation
    T[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
    T[0, 3] = pose.position.x
    T[1, 3] = pose.position.y
    T[2, 3] = pose.position.z
    return T


class VidBotManager(Node):
    def __init__(self):
        super().__init__("vidbot_manager")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("convergence_timeout", 15.0)   # seconds 30
        self.declare_parameter("pos_threshold", 0.05)         # metres 0.02
        self.declare_parameter("ori_threshold", 0.30)         # radians 0.15
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

        # Robot state (updated by callbacks)
        self.base_x = None
        self.base_y = None
        self.base_yaw = None
        self.current_joints = np.zeros(7)
        self.joints_received = False

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
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(String, "/vidbot/trigger", self._trigger_cb, 10)

        self.get_logger().info("VidBot manager ready. Waiting for pre/post poses...")

    # ── Robot state callbacks ─────────────────────────────────────────────

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.base_x = p.x
        self.base_y = p.y
        self.base_yaw = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')[2]

    def _joint_cb(self, msg):
        name_list = list(msg.name)
        for i, jname in enumerate(R_ARM_JOINTS):
            if jname in name_list:
                self.current_joints[i] = msg.position[name_list.index(jname)]
        self.joints_received = True

    # ── Vidbot callbacks ──────────────────────────────────────────────────

    def _trigger_cb(self, msg: String):
        """Capture the object name and instruction from the prompt node."""
        try:
            data = json.loads(msg.data)
            robot = data.get("robot", "reachy").strip().lower()
            if robot != "reachy":
                self.get_logger().info(
                    f"Ignoring trigger for robot '{robot}' (this manager handles reachy)"
                )
                return
            self.current_object = data.get("object", "").strip().lower()
            self.current_instruction = data.get("instruction", "").strip().lower()
            self.get_logger().info(
                f"Trigger received: robot={robot}, object='{self.current_object}', "
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
        local_object = self.current_object
        local_instruction = self.current_instruction
        self.pre_poses = None
        self.post_poses = None
        self.current_object = None
        self.current_instruction = None

        try:
            is_light_switch = (local_object is not None
                               and "light switch" in local_object)
            is_drop = (local_instruction is not None
                       and "drop" in local_instruction)

            pre_poses_list = list(local_pre.poses)
            post_poses_list = list(local_post.poses)


            # Build PoseArrays from (possibly modified) lists
            pre_msg = PoseArray()
            pre_msg.header = local_pre.header
            pre_msg.poses = pre_poses_list

            post_msg = PoseArray()
            post_msg.header = local_post.header
            post_msg.poses = post_poses_list

            # Light switch: close gripper BEFORE approach so the fingers
            # act as the contact surface for pressing the button.
            if is_light_switch:
                self.get_logger().info("Light switch: half-closing gripper before approach...")
                msg = Float64MultiArray()
                msg.data = [1.3, 0.0]
                self.gripper_pub.publish(msg)
                time.sleep(0.5)

            timeout = 15.0 if is_light_switch else None

            # Light switch: shift first pre-grasp pose +0.1 Y and go there first
            if is_light_switch:
                import copy
                pre_poses_list[0].position.y += 0.1
                ls_home = pre_poses_list[0]
                self.get_logger().info("Light switch: moving to start pose (+0.1 Y)...")
                ls_home_msg = PoseArray()
                ls_home_msg.header = local_pre.header
                ls_home_msg.poses = [ls_home]
                self._send_path_and_wait(ls_home_msg, ls_home, timeout=3.0, first_wp_timeout=15.0)

            # Step 1: Send pre-grasp path
            self.get_logger().info(
                f"Step 1/4: Sending pre-grasp path ({len(pre_poses_list)} poses)..."
            )
            first_wp_to = 15.0 if is_light_switch else 30.0
            self._send_path_and_wait(pre_msg, pre_poses_list[-1], timeout=timeout, first_wp_timeout=first_wp_to)

            # Light switch wiggle: sweep ±3cm on Y (torso frame) to ensure toggle
            if is_light_switch:
                import copy
                last_pose = pre_poses_list[-1]
                wiggle_y = 0.005  # 0.5 cm

                pose_plus = copy.deepcopy(last_pose)
                pose_plus.position.y += wiggle_y
                pose_minus = copy.deepcopy(last_pose)
                pose_minus.position.y -= wiggle_y

                for label, wp in [("  +3cm Y", pose_plus), ("  -3cm Y", pose_minus), ("  center", last_pose)]:
                    self.get_logger().info(f"Light switch wiggle:{label}")
                    wig_msg = PoseArray()
                    wig_msg.header = local_pre.header
                    wig_msg.poses = [wp]
                    self._send_path_and_wait(wig_msg, wp, timeout=2.0)

            # Step 2: Gripper action between pre-grasp and post-grasp
            if is_light_switch:
                pass  # already closed before approach; skip post-grasp entirely
            elif is_drop:
                # Drop: move +15cm on X from last pre-grasp pose, then open gripper
                import copy
                drop_pose = copy.deepcopy(pre_poses_list[-1])
                drop_pose.position.x += 0.30
                self.get_logger().info("Step 2 (drop): Moving +30cm X from pre-grasp endpoint...")
                drop_msg = PoseArray()
                drop_msg.header = local_pre.header
                drop_msg.poses = [drop_pose]
                self._send_path_and_wait(drop_msg, drop_pose, timeout=timeout)

                self.get_logger().info("Step 2 (drop): Opening gripper...")
                self._open_gripper()
                time.sleep(0.5)
            else:
                self.get_logger().info("Step 2/4: Closing gripper...")
                self._close_gripper()

            # Step 3: Send post-grasp path (skip for drop)
            if not is_drop:
                self.get_logger().info(
                    f"Step 3/4: Sending post-grasp path ({len(post_poses_list)} poses)..."
                )
                self._send_path_and_wait(post_msg, post_poses_list[-1], timeout=timeout)

            # Step 5: Return arm to first pre-grasp pose
            if is_light_switch:
                self.get_logger().info("Step 5: Returning arm to start pose (+0.1 Y)...")
                home_msg = PoseArray()
                home_msg.header = local_pre.header
                home_msg.poses = [ls_home]
                self._send_path_and_wait(home_msg, ls_home)
                self.get_logger().info("Light switch: opening gripper...")
                self._open_gripper()
            else:
                self.get_logger().info("Step 5: Returning arm to first pre-grasp pose...")
                home_msg = PoseArray()
                home_msg.header = local_pre.header
                home_msg.poses = [pre_poses_list[0]]
                self._send_path_and_wait(home_msg, pre_poses_list[0])

            self.get_logger().info("Grasp sequence complete!")

        except Exception as e:
            self.get_logger().error(f"Grasp sequence failed: {e}")
        finally:
            self.busy = False
            # If new data arrived during execution, kick off next sequence
            self._try_execute()

    def _wait_for_first_waypoint(self, first_pose: Pose, arrival_threshold=0.08, max_wait=5.0):
        """Block until the EE is close to the first waypoint of the path."""
        T_first = _pose_to_T(first_pose)
        dt = 1.0 / self.check_rate
        t0 = time.monotonic()

        while True:
            elapsed = time.monotonic() - t0
            if elapsed > max_wait:
                self.get_logger().warn(
                    f"First-waypoint wait timeout ({max_wait}s). Starting convergence timer."
                )
                break

            if self.base_x is None or not self.joints_received:
                time.sleep(dt)
                continue

            T_ee_odom = wholebody_fk(
                self.base_x, self.base_y, self.base_yaw, self.current_joints
            )

            c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
            T_base_odom = np.eye(4)
            T_base_odom[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            T_base_odom[0, 3] = self.base_x
            T_base_odom[1, 3] = self.base_y

            T_torso_base = np.eye(4)
            T_torso_base[0, 3] = -0.01
            T_torso_base[2, 3] = 0.996

            T_first_odom = T_base_odom @ T_torso_base @ T_first
            pos_err = np.linalg.norm(T_ee_odom[:3, 3] - T_first_odom[:3, 3])

            if pos_err < arrival_threshold:
                self.get_logger().info(
                    f"Reached first waypoint (pos_err={pos_err:.4f}m, waited {elapsed:.1f}s)"
                )
                break

            time.sleep(dt)

    def _send_path_and_wait(self, pose_array: PoseArray, goal_pose: Pose, timeout=None, first_wp_timeout=30.0):
        """Publish path, then poll EE pose until it converges to goal_pose."""
        self.path_pub.publish(pose_array)

        # Wait for the arm to reach the first waypoint before starting timeout
        if len(pose_array.poses) > 0:
            self._wait_for_first_waypoint(pose_array.poses[0], max_wait=first_wp_timeout)

        # Target transform (in torso frame from vidbot)
        T_goal = _pose_to_T(goal_pose)

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

            if self.base_x is None or not self.joints_received:
                time.sleep(dt)
                continue

            # Compute current EE in odom frame via FK
            T_ee_odom = wholebody_fk(
                self.base_x, self.base_y, self.base_yaw, self.current_joints
            )

            # Convert goal from torso to odom:
            # T_torso_odom = T_base_odom * T_torso_base
            c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
            T_base_odom = np.eye(4)
            T_base_odom[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            T_base_odom[0, 3] = self.base_x
            T_base_odom[1, 3] = self.base_y

            # torso is fixed offset from base_link
            T_torso_base = np.eye(4)
            T_torso_base[0, 3] = -0.01
            T_torso_base[2, 3] = 0.996

            T_goal_odom = T_base_odom @ T_torso_base @ T_goal

            # Position error
            pos_err = np.linalg.norm(T_ee_odom[:3, 3] - T_goal_odom[:3, 3])

            # Orientation error
            R_ee = Rotation.from_matrix(T_ee_odom[:3, :3])
            R_goal = Rotation.from_matrix(T_goal_odom[:3, :3])
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
