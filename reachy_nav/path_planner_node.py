#!/usr/bin/env python3
"""
ROS2 node wrapping the path planner from path_planner_o3d.py.

- Loads the TSDF scene and sets up the collision checker once at startup.
- Subscribes to the nav goal from pathplanner_manager (PoseStamped).
- Subscribes to the live robot pose from the localization node.
- On each goal: solves, assigns look-at orientations, publishes the
  trajectory as nav_msgs/Path.

Supports two planning modes (set in the YAML config or via --mode):
  collision    – obstacle point cloud → invalid voxels, polygon-stack checker
  reachability – reachability point cloud → valid voxels, stay-inside checker

Usage:
    python3 path_planner_node.py --config pathplanner_config.yaml [--mode collision|reachability]
"""

import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

import open3d as o3d

from rrt_point3d import PathPlanner
from path_planner_o3d import (
    VOXEL_SIZE,
    CROP_VERT,
    Z_PLANE,
    Z_BOUND_EPS,
    BOUND_MARGIN,
    POLY_VERT_BAND,
    POLY_MARGIN,
    ENABLE_FLOOR_CUT,
    FLOOR_CUT,
    TIME_LIMIT,
    METHOD,
    pcd_to_invalid_voxels,
    aabb_to_bound,
    force_z,
    build_robot_poly_slices,
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
    cfg["_reachability_path"] = os.path.join(data_dir, cfg["reachability_file"])
    cfg["_objects_path"] = os.path.join(data_dir, cfg["objects_file"])
    return cfg

FRAME_ID = "map"

# Fallback start pose (used when no live localization is available)
START_POS_FALLBACK = np.array([0.0, 0.0, 0.0], dtype=float)
START_YAW_FALLBACK = -np.pi / 2


class PathPlannerNode(Node):
    def __init__(self, cfg: dict, mode: str):
        super().__init__("path_planner")

        self._mode = mode
        self.get_logger().info(f"Planning mode: {mode}")

        # ---- Live pose from localization node ----
        self._current_pos = None  # np.array([x, y, z]) in z-up frame
        self._current_yaw = None  # float, radians

        # ---- Build planner (once) ----
        self._planner = PathPlanner()

        if mode == "collision":
            self._setup_collision_mode(cfg)
        elif mode == "reachability":
            self._setup_reachability_mode(cfg)
        else:
            raise ValueError(f"Unknown planning mode: {mode!r}")

        # ---- Publishers ----
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._path_pub = self.create_publisher(Path, "~/trajectory", latched_qos)

        # ---- Subscriber: goal from manager ----
        self.create_subscription(
            PoseStamped,
            "/pathplanner_manager/nav_goal",
            self._on_goal,
            latched_qos,
        )

        # ---- Subscriber: live robot pose from localization node ----
        self.create_subscription(
            Odometry,
            "/slam/base_odom",
            self._on_robot_pose,
            10,
        )

        self.get_logger().info("PathPlanner node ready, waiting for goals...")

    # ------------------------------------------------------------------
    # Mode setup helpers
    # ------------------------------------------------------------------
    def _setup_collision_mode(self, cfg: dict):
        """Obstacle avoidance: PCD → invalid voxels + polygon-stack checker."""
        pcd_path = cfg["_pcd_path"]
        self.get_logger().info(f"[collision] Loading obstacle cloud: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if len(pcd.points) == 0:
            raise RuntimeError(f"Empty point cloud: {pcd_path}")

        vg, invalid_vx_all, pcd_used = pcd_to_invalid_voxels(
            pcd, VOXEL_SIZE, crop_vert=CROP_VERT
        )
        self.get_logger().info(f"Invalid voxels: {invalid_vx_all.shape[0]}")

        if ENABLE_FLOOR_CUT:
            invalid_vx_all = invalid_vx_all[invalid_vx_all[:, 2] > FLOOR_CUT]
            self.get_logger().info(f"After floor cut: {invalid_vx_all.shape[0]}")

        bound = aabb_to_bound(
            pcd_used.get_axis_aligned_bounding_box(), BOUND_MARGIN
        )
        bound["low_z"] = float(Z_PLANE - Z_BOUND_EPS)
        bound["high_z"] = float(Z_PLANE + Z_BOUND_EPS)

        poly_slices, _ = build_robot_poly_slices()
        self.get_logger().info(f"Robot polygon slices: {len(poly_slices)}")

        self._planner.use_state(use_invx=True)
        self._planner.update_collision_radius(0, VOXEL_SIZE * 2.0)
        self._planner.update_sp(
            bound, None, invalid_vx_all, input_vx_size=VOXEL_SIZE
        )
        self._planner.set_robot_polygon_stack(
            slices=poly_slices,
            z_mode="offset",
            z_band=POLY_VERT_BAND,
            margin=POLY_MARGIN,
        )
        self._planner.use_validity_checker("poly_stack")
        self._bound = bound

    def _setup_reachability_mode(self, cfg: dict):
        """Stay-inside-reachable-space: reachability cloud → valid voxels."""
        reach_path = cfg["_reachability_path"]
        self.get_logger().info(f"[reachability] Loading reachability cloud: {reach_path}")
        pcd_reach = o3d.io.read_point_cloud(reach_path)
        if len(pcd_reach.points) == 0:
            raise RuntimeError(f"Empty reachability cloud: {reach_path}")

        # Project reachable points to the planning plane
        reach_pts = np.asarray(pcd_reach.points).copy()
        reach_pts[:, 2] = Z_PLANE
        self.get_logger().info(f"Reachable points: {reach_pts.shape[0]}")

        bound = aabb_to_bound(
            pcd_reach.get_axis_aligned_bounding_box(), BOUND_MARGIN
        )
        bound["low_z"] = float(Z_PLANE - Z_BOUND_EPS)
        bound["high_z"] = float(Z_PLANE + Z_BOUND_EPS)

        self._planner.use_state(use_invx=False)
        self._planner.update_collision_radius(VOXEL_SIZE * 2.0, 0)
        self._planner.update_sp(
            bound, reach_pts, None, input_vx_size=VOXEL_SIZE
        )
        self._planner.use_validity_checker("default")
        self._bound = bound

    # ------------------------------------------------------------------
    def _check_state_valid(self, state_xyz):
        """Check validity using the mode-appropriate checker."""
        if self._mode == "collision":
            return self._planner.isStateValid_poly_stack(state_xyz)
        else:
            return self._planner.isStateValid(state_xyz)

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
            f"Robot pose updated: [{self._current_pos[0]:.3f}, "
            f"{self._current_pos[1]:.3f}] yaw={np.degrees(self._current_yaw):.1f}°",
            throttle_duration_sec=5.0,
        )

    # ------------------------------------------------------------------
    def _on_goal(self, msg: PoseStamped):
        goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        goal_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        goal_yaw = quat_to_yaw(goal_quat)

        self.get_logger().info(
            f"Goal received: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, "
            f"{goal_pos[2]:.3f}] yaw={np.degrees(goal_yaw):.1f}°"
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

        start_ok = self._check_state_valid(start_xz)
        goal_ok = self._check_state_valid(goal_xz)
        self.get_logger().info(
            f"Validity: start={start_ok}  goal={goal_ok}"
        )

        if not start_ok or not goal_ok:
            label = "in collision" if self._mode == "collision" else "outside reachable space"
            self.get_logger().warn(f"Start or goal is {label}, solving anyway...")

        start = {"pos": start_xz, "quat": yaw_to_quat(start_yaw)}
        goal = {"pos": goal_xz, "quat": goal_quat}
        self._planner.update_start_goal(start, goal)

        self._planner.solve(time_limit=TIME_LIMIT, method=METHOD)
        solution = self._planner.get_solution()

        if solution is None or len(solution) == 0:
            self.get_logger().warn("No solution found.")
            return

        # Clamp Z & assign orientations
        for sol in solution:
            sol["pos"][2] = Z_PLANE
        solution = assign_look_at_orientations(solution, start_yaw, goal_yaw)

        self.get_logger().info(f"Solution: {len(solution)} waypoints")
        for i, s in enumerate(solution):
            yaw_deg = np.degrees(quat_to_yaw(s["quat"]))
            self.get_logger().info(
                f"  {i}: [{s['pos'][0]:.3f}, {s['pos'][1]:.3f}] "
                f"yaw={yaw_deg:.1f}°"
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
        self.get_logger().info("Published trajectory on ~/trajectory")


def main():
    parser = argparse.ArgumentParser(description="Path Planner Node")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--mode", choices=["collision", "reachability"], default=None,
        help="Planning mode (overrides config file)"
    )
    args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    cfg = load_config(args.config)
    mode = args.mode if args.mode else cfg.get("planning_mode", "collision")
    node = PathPlannerNode(cfg, mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
