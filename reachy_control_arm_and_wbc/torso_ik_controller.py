#!/usr/bin/env python3
"""
Fast torso-frame IK controller for Reachy2.

Same as torso_ik_controller.py but with IK speed optimizations:
  1. Batch pre-solve: all waypoint IK solutions computed upfront in _on_target
  2. Warm-start: each waypoint IK seeded from previous waypoint's solution
  3. Relaxed params for intermediate waypoints (fewer iters, no restarts, looser tol)

Subscribe:  /target_ee_pose_torso   (PoseStamped, frame: torso)
            /odom                   (Odometry)
            /joint_states           (JointState)
Publish:    /cmd_vel                (Twist)
            /r_arm_forward_position_controller/commands
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped, Twist, Point
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

from wholebody_ik_controller import (
    arm_fk, wholebody_fk, pose_error_6d,
    solve_arm_only_ik, solve_wholebody_ik, load_fallback_csv,
    _Rz, T_BASE_TO_TORSO, R_ARM_JOINTS,
    JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, CTRL_RATE,
)
from scipy.spatial import KDTree


# ── Cartesian interpolation ─────────────────────────────────────────────

def interpolate_poses(T_start, T_end, n_steps):
    """Interpolate n_steps poses between T_start and T_end (exclusive of start).

    Position: linear.  Orientation: SLERP.
    Returns list of n_steps 4x4 transforms (the last one equals T_end).
    """
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_end = Rotation.from_matrix(T_end[:3, :3])
    key_rots = Rotation.concatenate([R_start, R_end])
    slerp = Slerp([0.0, 1.0], key_rots)

    p_start = T_start[:3, 3]
    p_end = T_end[:3, 3]

    waypoints = []
    for i in range(1, n_steps + 1):
        t = i / n_steps
        T = np.eye(4)
        T[:3, 3] = (1.0 - t) * p_start + t * p_end
        T[:3, :3] = slerp(t).as_matrix()
        waypoints.append(T)
    return waypoints


# ── ROS2 Node ────────────────────────────────────────────────────────────

class TorsoIKControllerFast(Node):
    def __init__(self):
        super().__init__("torso_ik_controller")

        # Parameters
        self.declare_parameter("csv_path", str(Path(__file__).parent / "ee_poses_right_dense.csv"))

        self.declare_parameter("base_speed", 0.12)
        self.declare_parameter("base_angular_speed", 0.3)

        self.declare_parameter("target_time", 0.01)

        self.declare_parameter("base_pos_tolerance", 0.02)
        self.declare_parameter("base_yaw_tolerance", 0.03)

        self.declare_parameter("tracking_pos_threshold", 0.01)
        self.declare_parameter("tracking_ori_threshold", 0.05)
        self.declare_parameter("tracking_timeout", 500.0)

        self.declare_parameter("ik_pos_tol", 0.002)
        self.declare_parameter("ik_ori_tol", 0.02)
        self.declare_parameter("ik_damping", 0.05)
        self.declare_parameter("ik_base_pos_weight", 50.0)
        self.declare_parameter("ik_base_yaw_weight", 30.0)
        self.declare_parameter("arm_only_first", False)
        self.declare_parameter("max_joint_vel", 200.0)
        self.declare_parameter("joint_smoothing", 0.08)

        # Interpolation parameters
        self.declare_parameter("interp_pos_step", 0.009)
        self.declare_parameter("interp_ori_step", 0.09)
        self.declare_parameter("interp_pos_threshold", 0.01)
        self.declare_parameter("interp_ori_threshold", 0.05)

        self.base_speed = float(self.get_parameter("base_speed").value)
        self.base_ang_speed = float(self.get_parameter("base_angular_speed").value)
        self.target_time = float(self.get_parameter("target_time").value)
        self.base_pos_tol = float(self.get_parameter("base_pos_tolerance").value)
        self.base_yaw_tol = float(self.get_parameter("base_yaw_tolerance").value)

        self.tracking_pos_thr = float(self.get_parameter("tracking_pos_threshold").value)
        self.tracking_ori_thr = float(self.get_parameter("tracking_ori_threshold").value)
        self.tracking_timeout = float(self.get_parameter("tracking_timeout").value)

        self.ik_pos_tol = float(self.get_parameter("ik_pos_tol").value)
        self.ik_ori_tol = float(self.get_parameter("ik_ori_tol").value)
        self.ik_damping = float(self.get_parameter("ik_damping").value)
        self.ik_base_pos_w = float(self.get_parameter("ik_base_pos_weight").value)
        self.ik_base_yaw_w = float(self.get_parameter("ik_base_yaw_weight").value)
        self.arm_only_first = self.get_parameter("arm_only_first").value
        self.max_joint_vel = float(self.get_parameter("max_joint_vel").value)
        self.joint_smoothing = float(self.get_parameter("joint_smoothing").value)

        self.interp_pos_step = float(self.get_parameter("interp_pos_step").value)
        self.interp_ori_step = float(self.get_parameter("interp_ori_step").value)
        self.interp_pos_thr = float(self.get_parameter("interp_pos_threshold").value)
        self.interp_ori_thr = float(self.get_parameter("interp_ori_threshold").value)

        # Load fallback CSV
        csv_path = self.get_parameter("csv_path").value
        self.fb_joints, self.fb_positions, self.fb_quats = load_fallback_csv(csv_path)
        self.fb_tree = KDTree(self.fb_positions)
        self.get_logger().info(f"Loaded {len(self.fb_positions)} fallback samples")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.arm_pub = self.create_publisher(
            Float64MultiArray, "/r_arm_forward_position_controller/commands", 10
        )
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/target_ee_marker", 10)
        self.wp_marker_pub = self.create_publisher(MarkerArray, "/waypoint_markers", 10)

        # State
        self.base_x = None
        self.base_y = None
        self.base_yaw = None
        self.current_joints = np.zeros(7)
        self.commanded_joints = np.zeros(7)
        self.smoothed_joints = None
        self.joints_received = False

        self.target_base_x = None
        self.target_base_y = None
        self.target_base_yaw = None
        self.target_joints = None
        self.T_target_world = None
        self.active = False
        self.base_arrived = False
        self.last_resolve_time = None

        # Trajectory interpolation state
        self.traj_start_time = None
        self.traj_duration = 1.0
        self.traj_start_joints = np.zeros(7)
        self.traj_start_base = (0.0, 0.0, 0.0)

        # Smoothed velocity
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        # Pre-solved waypoint solutions: list of (bx, by, byaw, joints, T_odom)
        self.waypoint_solutions = []
        self.is_final_waypoint = True
        self.last_target_torso = None

        # Subscribers
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, "/target_ee_pose_torso", self._on_target, 10)
        self.create_subscription(PoseArray, "/target_ee_path_torso", self._on_path, 10)

        # Control timer
        self.create_timer(1.0 / CTRL_RATE, self._tick)

        self.get_logger().info(
            "Torso-frame IK controller (FAST) ready.\n"
            "  Waiting for /odom and /joint_states...\n"
            "  PoseStamped → /target_ee_pose_torso\n"
            "  PoseArray   → /target_ee_path_torso"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _torso_to_odom(self, T_torso):
        """Convert a 4x4 pose in torso frame to odom frame using current base state."""
        T_odom_base = _Rz(self.base_yaw)
        T_odom_base[0, 3] = self.base_x
        T_odom_base[1, 3] = self.base_y
        return T_odom_base @ T_BASE_TO_TORSO @ T_torso

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        first = self.base_x is None
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.base_x = p.x
        self.base_y = p.y
        self.base_yaw = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')[2]
        if first:
            self.get_logger().info(
                f"Odom OK. Base at ({self.base_x:.3f}, {self.base_y:.3f}), "
                f"yaw={np.degrees(self.base_yaw):.1f}deg. Ready for targets."
            )

    def _joint_state_cb(self, msg):
        name_list = list(msg.name)
        for i, jname in enumerate(R_ARM_JOINTS):
            if jname in name_list:
                self.current_joints[i] = msg.position[name_list.index(jname)]
        self.joints_received = True

    def _on_target(self, msg):
        if self.base_x is None:
            self.get_logger().error("No odom yet")
            return
        if not self.joints_received:
            self.get_logger().warn("No joint states yet, using zeros as seed")

        # Parse target pose in torso frame
        p = msg.pose.position
        o = msg.pose.orientation
        T_target_torso = np.eye(4)
        T_target_torso[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        T_target_torso[:3, 3] = [p.x, p.y, p.z]

        # Skip if the torso-frame target hasn't changed
        if self.last_target_torso is not None:
            if np.allclose(T_target_torso, self.last_target_torso, atol=1e-4):
                return
        self.last_target_torso = T_target_torso.copy()

        self.get_logger().info(
            f"Target EE (torso frame): ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )

        # Current EE in torso frame (from FK)
        T_current_torso = arm_fk(self.current_joints)

        # Compute distance between current and target
        pos_dist = float(np.linalg.norm(
            T_target_torso[:3, 3] - T_current_torso[:3, 3]
        ))
        R_rel = T_target_torso[:3, :3] @ T_current_torso[:3, :3].T
        ori_dist = float(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()))

        # Decide whether to interpolate
        need_interp = (pos_dist > self.interp_pos_thr or
                       ori_dist > self.interp_ori_thr)

        if need_interp:
            n_pos = max(1, int(np.ceil(pos_dist / self.interp_pos_step)))
            n_ori = max(1, int(np.ceil(ori_dist / self.interp_ori_step)))
            n_steps = max(n_pos, n_ori)
            waypoints_torso = interpolate_poses(T_current_torso, T_target_torso, n_steps)
        else:
            n_steps = 1
            waypoints_torso = [T_target_torso]

        # Convert ALL waypoints to odom frame NOW (freeze in world frame)
        waypoints_odom = [self._torso_to_odom(T) for T in waypoints_torso]

        # Visualize all waypoints in RViz
        self._publish_waypoint_markers(waypoints_odom)

        # ── Batch pre-solve all waypoints ─────────────────────────────────
        t0 = time.monotonic()
        solutions = self._batch_solve_ik(waypoints_odom)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self.get_logger().info(
            f"Batch IK: {len(solutions)}/{n_steps} solved in {elapsed_ms:.1f}ms"
        )

        if not solutions:
            self.get_logger().error("All waypoint IK solves failed!")
            return

        # Store pre-computed solutions and start executing
        self.waypoint_solutions = solutions
        self._execute_next_waypoint()

    def _on_path(self, msg):
        """Handle a full trajectory (PoseArray) in torso frame.

        Accepts a sequence of waypoints, sub-interpolates between consecutive
        pairs for smoothness, batch-solves IK for all, then executes.
        """
        if self.base_x is None:
            self.get_logger().error("No odom yet")
            return
        if not self.joints_received:
            self.get_logger().warn("No joint states yet, using zeros as seed")
        if len(msg.poses) == 0:
            self.get_logger().warn("Empty path received, ignoring")
            return

        # Parse all poses into 4x4 transforms (torso frame)
        path_torso = []
        for pose in msg.poses:
            p = pose.position
            o = pose.orientation
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
            T[:3, 3] = [p.x, p.y, p.z]
            path_torso.append(T)

        self.get_logger().info(
            f"Path received: {len(path_torso)} waypoints in torso frame"
        )

        # Sub-interpolate between consecutive waypoints for smooth Cartesian motion
        T_current_torso = arm_fk(self.current_joints)
        all_waypoints_torso = []

        prev = T_current_torso
        for T_wp in path_torso:
            pos_dist = float(np.linalg.norm(T_wp[:3, 3] - prev[:3, 3]))
            R_rel = T_wp[:3, :3] @ prev[:3, :3].T
            ori_dist = float(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()))

            if pos_dist > self.interp_pos_thr or ori_dist > self.interp_ori_thr:
                n_pos = max(1, int(np.ceil(pos_dist / self.interp_pos_step)))
                n_ori = max(1, int(np.ceil(ori_dist / self.interp_ori_step)))
                n_steps = max(n_pos, n_ori)
                sub_wps = interpolate_poses(prev, T_wp, n_steps)
                all_waypoints_torso.extend(sub_wps)
            else:
                all_waypoints_torso.append(T_wp)

            prev = T_wp

        # Convert to odom frame
        waypoints_odom = [self._torso_to_odom(T) for T in all_waypoints_torso]

        self.get_logger().info(
            f"Path expanded: {len(path_torso)} waypoints → "
            f"{len(waypoints_odom)} sub-waypoints"
        )

        # Visualize
        self._publish_waypoint_markers(waypoints_odom)

        # Batch solve IK
        t0 = time.monotonic()
        solutions = self._batch_solve_ik(waypoints_odom)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self.get_logger().info(
            f"Batch IK: {len(solutions)}/{len(waypoints_odom)} solved in {elapsed_ms:.1f}ms"
        )

        if not solutions:
            self.get_logger().error("All path IK solves failed!")
            return

        self.waypoint_solutions = solutions
        self._execute_next_waypoint()

    def _batch_solve_ik(self, waypoints_odom):
        """Pre-solve IK for all waypoints with warm-start and relaxed intermediate params.

        Returns list of (bx, by, byaw, joints, T_odom) tuples.
        """
        solutions = []
        seed_joints = self.current_joints.copy()
        n_total = len(waypoints_odom)

        for i, T_target_odom in enumerate(waypoints_odom):
            is_final = (i == n_total - 1)

            # Relaxed params for intermediate waypoints, full params for final
            if is_final:
                pos_tol = self.ik_pos_tol
                ori_tol = self.ik_ori_tol
                max_iter = 200
                n_restarts = 4
            else:
                pos_tol = 0.005
                ori_tol = 0.05
                max_iter = 30
                n_restarts = 0

            # Try arm-only IK first (warm-started from previous solution)
            if self.arm_only_first:
                T_odom_base = _Rz(self.base_yaw)
                T_odom_base[0, 3] = self.base_x
                T_odom_base[1, 3] = self.base_y
                T_odom_torso = T_odom_base @ T_BASE_TO_TORSO
                T_torso = np.linalg.inv(T_odom_torso) @ T_target_odom

                arm_ok, arm_joints, _, arm_iters = solve_arm_only_ik(
                    T_torso,
                    joints_init=seed_joints,
                    pos_tol=pos_tol,
                    ori_tol=ori_tol,
                    damping=self.ik_damping,
                    max_iter=max_iter,
                    n_restarts=n_restarts,
                )
                if arm_ok:
                    seed_joints = arm_joints.copy()
                    solutions.append((
                        self.base_x, self.base_y, self.base_yaw,
                        arm_joints, T_target_odom,
                    ))
                    continue

            # Whole-body IK (warm-started)
            ok, bx, by, byaw, joints, iters = solve_wholebody_ik(
                T_target_odom,
                base_xy_init=np.array([self.base_x, self.base_y]),
                base_yaw_init=self.base_yaw,
                joints_init=seed_joints,
                pos_tol=pos_tol,
                ori_tol=ori_tol,
                damping=self.ik_damping,
                base_pos_weight=self.ik_base_pos_w,
                base_yaw_weight=self.ik_base_yaw_w,
                max_iter=max_iter,
                n_restarts=n_restarts,
            )

            if ok:
                seed_joints = joints.copy()
                solutions.append((bx, by, byaw, joints, T_target_odom))
                continue

            # Check residual — accept if close enough
            T_check = wholebody_fk(bx, by, byaw, joints)
            err = pose_error_6d(T_check, T_target_odom)
            p_err = float(np.linalg.norm(err[:3]))

            if p_err < 0.05:
                seed_joints = joints.copy()
                solutions.append((bx, by, byaw, joints, T_target_odom))
                continue

            # Intermediate waypoint failed — skip it (arm will interpolate through)
            if not is_final:
                continue

            # Final waypoint failed — use CSV fallback
            self.get_logger().warn(
                f"Final waypoint IK failed (pos_err={p_err:.3f}m). CSV fallback."
            )
            fb_sol = self._fallback_csv_solve(T_target_odom)
            if fb_sol is not None:
                solutions.append(fb_sol)

        return solutions

    def _fallback_csv_solve(self, T_target_odom):
        """CSV fallback for a single waypoint. Returns (bx, by, byaw, joints, T_odom) or None."""
        T_odom_base = _Rz(self.base_yaw)
        T_odom_base[0, 3] = self.base_x
        T_odom_base[1, 3] = self.base_y
        T_odom_torso = T_odom_base @ T_BASE_TO_TORSO
        T_torso_odom = np.linalg.inv(T_odom_torso)

        T_target_torso = T_torso_odom @ T_target_odom
        target_pos_torso = T_target_torso[:3, 3]

        dist, idx = self.fb_tree.query(target_pos_torso)
        fb_joints = self.fb_joints[idx]
        fb_pos_torso = self.fb_positions[idx]

        offset_torso = target_pos_torso - fb_pos_torso
        offset_odom = T_odom_torso[:3, :3] @ offset_torso

        new_bx = self.base_x + offset_odom[0]
        new_by = self.base_y + offset_odom[1]

        return (new_bx, new_by, self.base_yaw, fb_joints, T_target_odom)

    def _execute_next_waypoint(self):
        """Pop the next pre-solved waypoint and activate it."""
        if not self.waypoint_solutions:
            return

        bx, by, byaw, joints, T_odom = self.waypoint_solutions.pop(0)
        self.is_final_waypoint = len(self.waypoint_solutions) == 0
        self.T_target_world = T_odom.copy()

        self._publish_target_marker(T_odom)
        self._set_targets(bx, by, byaw, joints)

    def _set_targets(self, bx, by, byaw, joints):
        """Activate a new target, snapshotting current state for trajectory start."""
        self.target_base_x = float(bx)
        self.target_base_y = float(by)
        self.target_base_yaw = float(byaw)
        self.target_joints = np.array(joints, dtype=np.float64)

        if self.active and self.smoothed_joints is not None:
            self.traj_start_joints = self.smoothed_joints.copy()
        else:
            self.traj_start_joints = self.current_joints.copy()
        self.traj_start_base = (float(self.base_x), float(self.base_y), float(self.base_yaw))
        self.traj_start_time = self.get_clock().now()

        max_joint_delta = float(np.max(np.abs(self.target_joints - self.traj_start_joints)))
        base_delta = float(np.hypot(self.target_base_x - self.base_x,
                                     self.target_base_y - self.base_y))

        min_dur_from_vel = 1.875 * max_joint_delta / max(self.max_joint_vel, 1e-6)

        motion_scale = max(max_joint_delta / 0.5, base_delta / 0.15)
        self.traj_duration = float(max(
            self.target_time * motion_scale,
            min_dur_from_vel,
            0.15,
        ))

        self.commanded_joints = self.traj_start_joints.copy()

        if not self.active:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0

        self.active = True
        self.base_arrived = False
        self.last_resolve_time = None

    # ── Minimum-jerk trajectory ─────────────────────────────────────────

    @staticmethod
    def _min_jerk(t, T):
        if T <= 0.0:
            return 1.0
        tau = min(t / T, 1.0)
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    # ── Control loop ─────────────────────────────────────────────────────

    def _tick(self):
        if not self.active or self.target_joints is None:
            return
        if self.base_x is None or self.traj_start_time is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.traj_start_time).nanoseconds * 1e-9
        s = self._min_jerk(elapsed, self.traj_duration)

        # ── Interpolate arm joints + low-pass filter ────────────────────
        self.commanded_joints = (
            self.traj_start_joints + s * (self.target_joints - self.traj_start_joints)
        )

        if self.smoothed_joints is None:
            self.smoothed_joints = self.commanded_joints.copy()
        else:
            a = self.joint_smoothing
            self.smoothed_joints += a * (self.commanded_joints - self.smoothed_joints)

        arm_msg = Float64MultiArray()
        arm_msg.data = self.smoothed_joints.tolist()
        self.arm_pub.publish(arm_msg)

        # ── Base error ───────────────────────────────────────────────────
        dx = self.target_base_x - self.base_x
        dy = self.target_base_y - self.base_y
        dyaw = self._angle_diff(self.target_base_yaw, self.base_yaw)
        pos_dist = float(np.hypot(dx, dy))
        yaw_dist = float(abs(dyaw))

        # ── Waypoint completion check ────────────────────────────────────
        if s >= 1.0:
            if not self.is_final_waypoint:
                self._execute_next_waypoint()
                return

            # Final waypoint: full EE tracking
            if self.T_target_world is not None:
                T_actual = wholebody_fk(
                    self.base_x, self.base_y, self.base_yaw, self.current_joints
                )
                ee_err = pose_error_6d(T_actual, self.T_target_world)
                ee_pos_err = float(np.linalg.norm(ee_err[:3]))
                ee_ori_err = float(np.linalg.norm(ee_err[3:]))

                timed_out = False
                if self.tracking_timeout > 0.0:
                    tracking_elapsed = elapsed - self.traj_duration
                    if tracking_elapsed > self.tracking_timeout:
                        timed_out = True

                if timed_out or (ee_pos_err < self.tracking_pos_thr
                                 and ee_ori_err < self.tracking_ori_thr):
                    if not self.base_arrived:
                        self.base_arrived = True
                        self.cmd_vx = 0.0
                        self.cmd_vy = 0.0
                        self.cmd_wz = 0.0
                        self.base_pub.publish(Twist())
                        reason = "timeout" if timed_out else "converged"
                        self.get_logger().info(
                            f"Tracking {reason}. ee_pos_err={ee_pos_err:.4f}m, "
                            f"ee_ori_err={ee_ori_err:.4f}rad"
                        )

                    else:
                        arm_msg = Float64MultiArray()
                        arm_msg.data = self.target_joints.tolist()
                        self.arm_pub.publish(arm_msg)
                        self.base_pub.publish(Twist())
                    return

                # Re-solve IK if base near target but EE still off
                if pos_dist < self.base_pos_tol * 2 and yaw_dist < self.base_yaw_tol * 2:
                    should_resolve = False
                    if self.last_resolve_time is None:
                        should_resolve = True
                    else:
                        dt = (now - self.last_resolve_time).nanoseconds * 1e-9
                        if dt > 0.5:
                            should_resolve = True

                    if should_resolve:
                        self.last_resolve_time = now
                        ok, bx, by, byaw, joints, iters = solve_wholebody_ik(
                            self.T_target_world,
                            base_xy_init=np.array([self.base_x, self.base_y]),
                            base_yaw_init=self.base_yaw,
                            joints_init=self.current_joints,
                            pos_tol=self.ik_pos_tol,
                            ori_tol=self.ik_ori_tol,
                            damping=self.ik_damping,
                            base_pos_weight=self.ik_base_pos_w,
                            base_yaw_weight=self.ik_base_yaw_w,
                        )
                        if ok:
                            self.get_logger().debug(
                                f"Tracking re-solve: ee_pos_err={ee_pos_err:.4f}m → "
                                f"new IK in {iters} iters"
                            )
                            self._set_targets(bx, by, byaw, joints)
                            return

        # ── Base velocity: P-control ─────────────────────────────────────
        c = np.cos(self.base_yaw)
        sn = np.sin(self.base_yaw)
        vx_base = c * dx + sn * dy
        vy_base = -sn * dx + c * dy

        p_gain = 2.0
        vx_cmd = vx_base * p_gain
        vy_cmd = vy_base * p_gain
        wz_cmd = dyaw * p_gain

        brake_radius = 0.15
        yaw_brake_radius = 0.30
        lin_brake = float(np.clip(pos_dist / brake_radius, 0.0, 1.0))
        yaw_brake = float(np.clip(yaw_dist / yaw_brake_radius, 0.0, 1.0))

        lin_cap = self.base_speed * lin_brake
        lin_speed = float(np.hypot(vx_cmd, vy_cmd))
        if lin_speed > lin_cap and lin_speed > 1e-6:
            vx_cmd *= lin_cap / lin_speed
            vy_cmd *= lin_cap / lin_speed
        ang_cap = self.base_ang_speed * yaw_brake
        wz_cmd = float(np.clip(wz_cmd, -ang_cap, ang_cap))

        alpha = 0.3
        self.cmd_vx += alpha * (vx_cmd - self.cmd_vx)
        self.cmd_vy += alpha * (vy_cmd - self.cmd_vy)
        self.cmd_wz += alpha * (wz_cmd - self.cmd_wz)

        cmd = Twist()
        cmd.linear.x = float(self.cmd_vx)
        cmd.linear.y = float(self.cmd_vy)
        cmd.angular.z = float(self.cmd_wz)
        self.base_pub.publish(cmd)

    @staticmethod
    def _angle_diff(target, current):
        d = target - current
        return (d + np.pi) % (2 * np.pi) - np.pi

    def _publish_waypoint_markers(self, waypoints_odom):
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        delete = Marker()
        delete.header.frame_id = "odom"
        delete.header.stamp = stamp
        delete.ns = "waypoints"
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        odom_positions = [T[:3, 3] for T in waypoints_odom]

        for i, pos in enumerate(odom_positions):
            is_final = (i == len(odom_positions) - 1)
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = stamp
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            sz = 0.04 if is_final else 0.025
            m.scale.x = sz
            m.scale.y = sz
            m.scale.z = sz
            if is_final:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            m.color.a = 1.0
            m.lifetime.sec = 30
            markers.markers.append(m)

        if len(odom_positions) > 1:
            line = Marker()
            line.header.frame_id = "odom"
            line.header.stamp = stamp
            line.ns = "waypoints"
            line.id = len(odom_positions)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.008
            line.color.r = 1.0
            line.color.g = 0.6
            line.color.b = 0.0
            line.color.a = 0.8
            line.lifetime.sec = 30
            line.pose.orientation.w = 1.0
            for pos in odom_positions:
                line.points.append(
                    Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                )
            markers.markers.append(line)

        self.wp_marker_pub.publish(markers)

    def _publish_target_marker(self, T_target):
        stamp = self.get_clock().now().to_msg()
        pos = T_target[:3, 3]
        R = T_target[:3, :3]

        axis_len = 0.10
        shaft_d = 0.008
        head_d = 0.015
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        markers = MarkerArray()
        for i in range(3):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = stamp
            m.ns = "target_ee_axes"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            tip = pos + R[:, i] * axis_len
            m.points = [
                Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])),
            ]
            m.scale.x = shaft_d
            m.scale.y = head_d
            m.scale.z = 0.0

            m.color.r = float(colors[i][0])
            m.color.g = float(colors[i][1])
            m.color.b = float(colors[i][2])
            m.color.a = 1.0
            m.lifetime.sec = 1
            markers.markers.append(m)

        self.marker_pub.publish(markers)



def main():
    rclpy.init()
    node = TorsoIKControllerFast()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
