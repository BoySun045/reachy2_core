#!/usr/bin/env python3
"""
ROS2 path planner node for Spot.

- Loads the reachability map and sets up the planner at startup.
- Subscribes to /object_pose (PoseStamped) for target object locations.
- Finds the closest reachable point to the object, orients to face it.
- Plans a path from the current robot pose to the nav goal.
- Publishes the trajectory as nav_msgs/Path.
- Publishes the reachability point cloud for rviz visualization.

Usage:
    python3 path_planner_node.py --config pathplanner_config.yaml
"""

import math
import os
import argparse
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_srvs.srv import Trigger

import open3d as o3d

from rrt_point3d import PathPlanner
from path_planner_o3d import (
    VOXEL_SIZE,
    Z_PLANE,
    Z_BOUND_EPS,
    BOUND_MARGIN,
    TIME_LIMIT,
    METHOD,
    aabb_to_bound,
    force_z,
    yaw_to_quat,
    quat_to_yaw,
    assign_look_at_orientations,
)


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve paths relative to the config file directory."""
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)
    data_dir = os.path.join(config_dir, cfg["data_dir"])
    cfg["_data_dir"] = data_dir
    cfg["_pcd_path"] = os.path.join(data_dir, cfg["pcd_file"])
    cfg["_reachability_path"] = os.path.join(data_dir, cfg["reachability_file_spot"])
    return cfg

FRAME_ID = "map"

# Fallback start pose (used when no live localization is available)
START_POS_FALLBACK = np.array([0.0, 0.0, 0.0], dtype=float)
START_YAW_FALLBACK = -np.pi / 2

# Arrival tolerances
GOAL_POS_TOL = 1.0  # metres
GOAL_YAW_TOL = math.radians(25.0)  # radians (~15 degrees)

# Scan pose position in body frame (must match spot_scan_pose.py defaults)
SCAN_POSE_X = 0.4
SCAN_POSE_Y = 0.0
SCAN_POSE_Z = 0.7


class PathPlannerNode(Node):
    def __init__(self, cfg: dict):
        super().__init__("path_planner")

        self._min_dist2goal = float(cfg.get("spot_min_dist2goal", 0.3))
        self.get_logger().info(f"min_dist2goal = {self._min_dist2goal:.3f} m")

        # ---- Live pose from localization ----
        self._current_pos = None
        self._current_yaw = None

        # ---- Build planner (reachability mode) ----
        self._reach_pts = None
        self._planner = PathPlanner()
        self._setup_reachability(cfg)

        # ---- Publishers ----
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._path_pub = self.create_publisher(
            Path, "/spot/planned_path", latched_qos
        )
        self._pose_array_pub = self.create_publisher(
            PoseArray, "/spot/planned_path_poses", latched_qos
        )
        self._reach_pub = self.create_publisher(
            PointCloud2, "~/reachability", 10
        )
        self._arm_cmd_pub = self.create_publisher(
            PoseStamped, "/spot/arm_pose_commands", 10
        )

        # ---- Service clients ----
        self._scan_pose_client = self.create_client(Trigger, "/scan_pose")
        self._stow_client = self.create_client(Trigger, "/spot/arm_stow")

        # ---- Arrival monitoring state ----
        self._nav_goal_pos = None   # XYZ of the nav goal (map frame)
        self._nav_goal_yaw = None   # yaw at the nav goal
        self._target_object_pos = None  # XYZ of the object we want to look at
        self._arrival_timer = None  # 1 Hz timer, created after trajectory publish
        self._arrival_timeout_timer = None  # 30s fallback timer
        self._arm_active = False    # True while arm look-at thread is running

        # ---- Subscriber: object pose (target location) ----
        self.create_subscription(
            PoseStamped, "/object_pose", self._on_object_pose, 10
        )

        # ---- Subscriber: live robot pose ----
        self.create_subscription(
            Odometry, "/spot/odometry/corrected", self._on_robot_pose, 10
        )

        # ---- Timer: publish reachability point cloud at 0.1 Hz ----
        self.create_timer(10.0, self._publish_reachability)

        self.get_logger().info("PathPlanner node ready, waiting for /object_pose...")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_reachability(self, cfg: dict):
        """Load reachability cloud and configure the planner."""
        reach_path = cfg["_reachability_path"]
        self.get_logger().info(f"Loading reachability cloud: {reach_path}")
        pcd_reach = o3d.io.read_point_cloud(reach_path)
        if len(pcd_reach.points) == 0:
            raise RuntimeError(f"Empty reachability cloud: {reach_path}")

        # Project reachable points to the planning plane
        reach_pts = np.asarray(pcd_reach.points).copy()
        reach_pts[:, 2] = Z_PLANE
        self._reach_pts = reach_pts
        self.get_logger().info(f"Reachable points: {reach_pts.shape[0]}")

        bound = aabb_to_bound(
            pcd_reach.get_axis_aligned_bounding_box(), BOUND_MARGIN
        )
        bound["low_z"] = float(Z_PLANE - Z_BOUND_EPS)
        bound["high_z"] = float(Z_PLANE + Z_BOUND_EPS)

        self._planner.use_state(use_invx=False)
        self._planner.update_collision_radius(VOXEL_SIZE * 1.0, 0)
        self._planner.update_sp(
            bound, reach_pts, None, input_vx_size=VOXEL_SIZE
        )
        self._planner.use_validity_checker("default")

        # Build PointCloud2 message for reachability visualization
        self._reach_pc2_msg = self._build_pointcloud2_msg(pcd_reach)

    # ------------------------------------------------------------------
    # PointCloud2 helpers
    # ------------------------------------------------------------------
    def _build_pointcloud2_msg(self, pcd):
        """Convert an Open3D point cloud to a ROS2 PointCloud2 message."""
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            rgb_uint32 = (
                colors[:, 0].astype(np.uint32) << 16
                | colors[:, 1].astype(np.uint32) << 8
                | colors[:, 2].astype(np.uint32)
            )
            rgb_float32 = rgb_uint32.view(np.float32)
        else:
            rgb_float32 = np.full(pts.shape[0], 0.5, dtype=np.float32)
        xyzrgb = np.column_stack([pts, rgb_float32.reshape(-1, 1)])
        from std_msgs.msg import Header
        header = Header()
        header.frame_id = FRAME_ID
        header.stamp = self.get_clock().now().to_msg()
        return pc2.create_cloud(header, fields, xyzrgb)

    def _publish_reachability(self):
        """Publish the reachability point cloud periodically."""
        self._reach_pc2_msg.header.stamp = self.get_clock().now().to_msg()
        self._reach_pub.publish(self._reach_pc2_msg)

    # ------------------------------------------------------------------
    # Arrival monitoring & arm look-at
    # ------------------------------------------------------------------
    def _start_arrival_monitor(self):
        """Start a 1 Hz timer to check if the robot has reached the nav goal."""
        if self._arrival_timer is not None:
            self._arrival_timer.cancel()
        if self._arrival_timeout_timer is not None:
            self._arrival_timeout_timer.cancel()
        self._arrival_timer = self.create_timer(1.0, self._check_arrival)
        self._arrival_timeout_timer = self.create_timer(30.0, self._on_arrival_timeout)
        self.get_logger().info("Arrival monitor started (30s timeout).")

    def _cancel_arrival_timers(self):
        """Cancel both arrival check and timeout timers."""
        if self._arrival_timer is not None:
            self._arrival_timer.cancel()
            self._arrival_timer = None
        if self._arrival_timeout_timer is not None:
            self._arrival_timeout_timer.cancel()
            self._arrival_timeout_timer = None

    def _on_arrival_timeout(self):
        """30s timeout: stop waiting and trigger arm look-at anyway."""
        self.get_logger().warn("Arrival timeout (30s), triggering arm look-at anyway.")
        self._cancel_arrival_timers()
        self._publish_stop_path()
        if self._target_object_pos is not None:
            thread = threading.Thread(
                target=self._execute_arm_look_at, daemon=True
            )
            thread.start()

    def _publish_stop_path(self):
        """Publish a single-waypoint path at the robot's current pose to stop it."""
        if self._current_pos is None:
            return
        path_msg = Path()
        path_msg.header.frame_id = FRAME_ID
        path_msg.header.stamp = self.get_clock().now().to_msg()

        ps = PoseStamped()
        ps.header = path_msg.header
        ps.pose.position.x = float(self._current_pos[0])
        ps.pose.position.y = float(self._current_pos[1])
        ps.pose.position.z = float(self._current_pos[2])
        q = yaw_to_quat(self._current_yaw)
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])
        path_msg.poses.append(ps)

        self._path_pub.publish(path_msg)

        pose_array_msg = PoseArray()
        pose_array_msg.header = path_msg.header
        pose_array_msg.poses = [ps.pose]
        self._pose_array_pub.publish(pose_array_msg)

        self.get_logger().info("Published stop path at current pose.")

    def _stow_arm(self):
        """Call /spot/arm_stow service (blocking). Returns True on success."""
        if self._stow_client.wait_for_service(timeout_sec=5.0):
            future = self._stow_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() and future.result().success:
                self.get_logger().info("Arm stowed.")
                return True
            else:
                self.get_logger().error("Arm stow failed.")
        else:
            self.get_logger().error("/spot/arm_stow service not available.")
        return False

    def _check_arrival(self):
        """Periodic check: has the robot reached the nav goal (position OR yaw)?"""
        if self._current_pos is None or self._nav_goal_pos is None:
            return

        dx = self._current_pos[0] - self._nav_goal_pos[0]
        dy = self._current_pos[1] - self._nav_goal_pos[1]
        dist = math.hypot(dx, dy)

        yaw_err = abs(self._current_yaw - self._nav_goal_yaw)
        if yaw_err > math.pi:
            yaw_err = 2 * math.pi - yaw_err

        pos_ok = dist < GOAL_POS_TOL
        yaw_ok = yaw_err < GOAL_YAW_TOL

        if pos_ok and yaw_ok:
            self.get_logger().info(
                f"Arrived at goal! dist={dist:.3f}m, "
                f"yaw_err={math.degrees(yaw_err):.1f}deg"
            )
            self._cancel_arrival_timers()

            # Stop the robot by publishing current pose as the new trajectory
            self._publish_stop_path()

            # Trigger arm look-at in a separate thread
            if self._target_object_pos is not None:
                thread = threading.Thread(
                    target=self._execute_arm_look_at, daemon=True
                )
                thread.start()

    def _execute_arm_look_at(self):
        """Move arm to scan pose, then orient EE toward the target object."""
        self._arm_active = True

        # Step 1: Publish scan pose directly (skip service round-trip)
        ps_scan = PoseStamped()
        ps_scan.header.stamp = self.get_clock().now().to_msg()
        ps_scan.header.frame_id = "body"
        ps_scan.pose.position.x = SCAN_POSE_X
        ps_scan.pose.position.y = SCAN_POSE_Y
        ps_scan.pose.position.z = SCAN_POSE_Z
        ps_scan.pose.orientation.w = 1.0
        self._arm_cmd_pub.publish(ps_scan)
        self.get_logger().info("Scan pose published directly.")
        time.sleep(1.5)  # brief settle for arm to reach scan pose

        # Step 2: Compute direction from current EE to target, align EE X axis
        if self._current_pos is None:
            self.get_logger().warn("No robot pose, cannot compute arm target.")
            self._arm_active = False
            return

        obj = self._target_object_pos
        rob = self._current_pos
        yaw = self._current_yaw

        # Target position in map frame → body frame
        dx_map = obj[0] - rob[0]
        dy_map = obj[1] - rob[1]
        dz_map = obj[2] - rob[2]

        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        target_body = np.array([
            dx_map * cos_y - dy_map * sin_y,
            dx_map * sin_y + dy_map * cos_y,
            dz_map,
        ])

        # EE position in body frame (scan pose)
        ee_body = np.array([SCAN_POSE_X, SCAN_POSE_Y, SCAN_POSE_Z])

        # Direction vector from EE to target in body frame
        direction = target_body - ee_body
        dist = np.linalg.norm(direction)
        if dist < 1e-3:
            self.get_logger().warn("Target too close to EE, skipping look-at.")
            self._arm_active = False
            return
        direction = direction / dist  # unit vector

        # Rotation that aligns EE X axis [1,0,0] with the direction vector
        rot, _ = Rot.align_vectors([direction], [[1.0, 0.0, 0.0]])
        q = rot.as_quat()  # [x, y, z, w]

        self.get_logger().info(
            f"Look-at: EE={ee_body}, target_body={target_body}, "
            f"dir={direction}"
        )

        # Step 3: Publish arm pose at scan position with look-at orientation
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "body"
        ps.pose.position.x = float(ee_body[0])
        ps.pose.position.y = float(ee_body[1])
        ps.pose.position.z = float(ee_body[2])
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])
        self._arm_cmd_pub.publish(ps)

        self.get_logger().info(
            f"Arm look-at published: pos={ee_body} quat={q}"
        )

        # Step 4: Wait 5s then stow the arm
        time.sleep(5.0)
        self.get_logger().info("Look-at hold done, stowing arm...")
        self._stow_arm()
        self._arm_active = False

    # ------------------------------------------------------------------
    # Nav goal computation
    # ------------------------------------------------------------------
    def _compute_nav_goal(self, object_pos):
        """Find the closest reachable point to the object, facing it.

        Respects min_dist2goal: the nav goal must be at least this far
        from the object (XY). Returns (goal_pos, goal_yaw) or None.
        """
        obj_xy = object_pos[:2]
        dists = np.linalg.norm(self._reach_pts[:, :2] - obj_xy, axis=1)

        far_enough = dists >= self._min_dist2goal
        if np.any(far_enough):
            best = np.argmin(dists[far_enough])
            goal_pt = self._reach_pts[far_enough][best].copy()
        else:
            # No point meets minimum distance; fall back to farthest
            best = np.argmax(dists)
            goal_pt = self._reach_pts[best].copy()

        # Face the object
        dx = obj_xy[0] - goal_pt[0]
        dy = obj_xy[1] - goal_pt[1]
        yaw = float(np.arctan2(dy, dx))

        return goal_pt, yaw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _snap_to_nearest_valid(self, pos_xyz):
        """Find the closest reachable point (XY) to the given position."""
        if self._reach_pts is None:
            return pos_xyz
        dists = np.linalg.norm(self._reach_pts[:, :2] - pos_xyz[:2], axis=1)
        nearest = self._reach_pts[np.argmin(dists)]
        return nearest.copy()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_robot_pose(self, msg: Odometry):
        pose = msg.pose.pose
        self._current_pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ])
        self._current_yaw = quat_to_yaw(np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]))
        self.get_logger().info(
            f"Pose updated: [{self._current_pos[0]:.3f}, "
            f"{self._current_pos[1]:.3f}] yaw={np.degrees(self._current_yaw):.1f}",
            throttle_duration_sec=5.0,
        )

    def _on_object_pose(self, msg: PoseStamped):
        """Receive an object location, compute nav goal, plan path."""
        # Cancel any ongoing arrival monitoring / timeout
        self._cancel_arrival_timers()

        # If arm is active from previous target, stow it before moving
        if self._arm_active:
            self.get_logger().info("New goal received, stowing arm first...")
            self._arm_active = False
            thread = threading.Thread(target=self._stow_arm, daemon=True)
            thread.start()
            thread.join(timeout=15.0)  # wait for stow to finish before planning

        object_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self._target_object_pos = object_pos
        self.get_logger().info(
            f"Object pose received: [{object_pos[0]:.3f}, "
            f"{object_pos[1]:.3f}, {object_pos[2]:.3f}]"
        )

        # Find closest reachable point facing the object
        result = self._compute_nav_goal(object_pos)
        if result is None:
            self.get_logger().warn("Could not compute nav goal.")
            return
        goal_pos, goal_yaw = result
        goal_quat = yaw_to_quat(goal_yaw)

        # Store nav goal for arrival monitoring
        self._nav_goal_pos = goal_pos
        self._nav_goal_yaw = goal_yaw

        self.get_logger().info(
            f"Nav goal: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}] "
            f"yaw={np.degrees(goal_yaw):.1f} (facing object)"
        )

        # ---- Determine start pose ----
        if self._current_pos is not None:
            start_pos = self._current_pos
            start_yaw = self._current_yaw
            self.get_logger().info("Using live localized pose as start")
        else:
            start_pos = START_POS_FALLBACK
            start_yaw = START_YAW_FALLBACK
            self.get_logger().warn("No live pose, using fallback start")

        # ---- Solve ----
        start_xz = force_z(start_pos, Z_PLANE)
        goal_xz = force_z(goal_pos, Z_PLANE)

        start_ok = self._planner.isStateValid(start_xz)
        goal_ok = self._planner.isStateValid(goal_xz)
        self.get_logger().info(f"Validity: start={start_ok}  goal={goal_ok}")

        if not start_ok:
            snapped = self._snap_to_nearest_valid(start_xz)
            self.get_logger().warn(
                f"Start outside reachable space, snapping to nearest valid: "
                f"[{snapped[0]:.3f}, {snapped[1]:.3f}]"
            )
            start_xz = snapped

        if not goal_ok:
            self.get_logger().warn("Goal outside reachable space, solving anyway...")

        start = {"pos": start_xz, "quat": yaw_to_quat(start_yaw)}
        goal = {"pos": goal_xz, "quat": goal_quat}
        self._planner.update_start_goal(start, goal)

        self._planner.solve(time_limit=TIME_LIMIT, method=METHOD)
        solution = self._planner.get_solution()

        if solution is None or len(solution) == 0:
            self.get_logger().warn("No solution found.")
            return

        # Set Z to odometry height
        publish_z = float(start_pos[2]) if self._current_pos is not None else Z_PLANE
        for sol in solution:
            sol["pos"][2] = publish_z
        solution = assign_look_at_orientations(solution, start_yaw, goal_yaw)

        self.get_logger().info(f"Solution: {len(solution)} waypoints")
        for i, s in enumerate(solution):
            yaw_deg = np.degrees(quat_to_yaw(s["quat"]))
            self.get_logger().info(
                f"  {i}: [{s['pos'][0]:.3f}, {s['pos'][1]:.3f}] "
                f"yaw={yaw_deg:.1f}"
            )

        # ---- Publish trajectory ----
        path_msg = Path()
        path_msg.header.frame_id = FRAME_ID
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for sol in solution:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(sol["pos"][0])
            ps.pose.position.y = float(sol["pos"][1])
            ps.pose.position.z = float(sol["pos"][2])
            q = sol["quat"]
            ps.pose.orientation.x = float(q[0])
            ps.pose.orientation.y = float(q[1])
            ps.pose.orientation.z = float(q[2])
            ps.pose.orientation.w = float(q[3])
            path_msg.poses.append(ps)

        self._path_pub.publish(path_msg)

        pose_array_msg = PoseArray()
        pose_array_msg.header = path_msg.header
        pose_array_msg.poses = [ps.pose for ps in path_msg.poses]
        self._pose_array_pub.publish(pose_array_msg)

        self.get_logger().info(
            f"Published trajectory on {self._path_pub.topic_name} "
            f"and {self._pose_array_pub.topic_name}"
        )

        # Start monitoring for arrival to trigger arm look-at
        self._start_arrival_monitor()


def main():
    parser = argparse.ArgumentParser(description="Path Planner Node (Spot)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    cfg = load_config(args.config)
    node = PathPlannerNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
