#!/usr/bin/env python3
"""
ROS2 node wrapping the path planner from path_planner_o3d.py.

- Loads the TSDF scene and sets up the collision checker once at startup.
- Subscribes to the nav goal from pathplanner_manager (PoseStamped).
- On each goal: solves, assigns look-at orientations, publishes the
  trajectory as nav_msgs/Path.
- Start pose is hardcoded until the localisation pipeline is ready.

Usage:
    python3 path_planner_node.py
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import open3d as o3d

from rrt_point3d import PathPlanner
from path_planner_o3d import (
    PCD_PATH,
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

FRAME_ID = "map"

# Hardcoded start until localisation is ready
START_POS = np.array([0.0, 0.0, 0.0], dtype=float)
START_YAW = np.pi / 2


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner")

        # ---- Load scene & build planner (once) ----
        self.get_logger().info(f"Loading point cloud: {PCD_PATH}")
        pcd = o3d.io.read_point_cloud(PCD_PATH)
        if len(pcd.points) == 0:
            raise RuntimeError(f"Empty point cloud: {PCD_PATH}")

        vg, invalid_vx_all, pcd_used = pcd_to_invalid_voxels(
            pcd, VOXEL_SIZE, crop_vert=CROP_VERT
        )
        self.get_logger().info(f"Invalid voxels: {invalid_vx_all.shape[0]}")

        if ENABLE_FLOOR_CUT:
            invalid_vx_all = invalid_vx_all[invalid_vx_all[:, 2] > FLOOR_CUT]
            self.get_logger().info(
                f"After floor cut: {invalid_vx_all.shape[0]}"
            )

        bound = aabb_to_bound(
            pcd_used.get_axis_aligned_bounding_box(), BOUND_MARGIN
        )
        bound["low_z"] = float(Z_PLANE - Z_BOUND_EPS)
        bound["high_z"] = float(Z_PLANE + Z_BOUND_EPS)

        poly_slices, _ = build_robot_poly_slices()
        self.get_logger().info(f"Robot polygon slices: {len(poly_slices)}")

        self._planner = PathPlanner()
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

        self.get_logger().info("PathPlanner node ready, waiting for goals...")

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

        # ---- Solve ----
        start_xz = force_z(START_POS, Z_PLANE)
        goal_xz = force_z(goal_pos, Z_PLANE)

        start_ok = self._planner.isStateValid_poly_stack(start_xz)
        goal_ok = self._planner.isStateValid_poly_stack(goal_xz)
        self.get_logger().info(
            f"Validity: start={start_ok}  goal={goal_ok}"
        )

        if not start_ok or not goal_ok:
            self.get_logger().warn("Start or goal is in collision, solving anyway...")

        start = {"pos": start_xz, "quat": yaw_to_quat(START_YAW)}
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
        solution = assign_look_at_orientations(solution, START_YAW, goal_yaw)

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


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
