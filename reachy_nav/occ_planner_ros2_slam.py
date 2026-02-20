#!/usr/bin/env python3
"""
ROS2 node wrapping OccupancyGrid3DPathPlanner.

Provides a GetPlan service. Plans once when called, publishes the result
on ~/planned_path, and keeps it latched for path followers.

Usage:
    python3 occ_planner_ros2.py

    # Trigger from another terminal (only goal needed, start comes from /odom):
    ros2 service call /occ_planner/plan_path nav_msgs/srv/GetPlan \
      "{goal: {header: {frame_id: 'map'}, pose: {position: {x: 2, y: 2, z: 0}, orientation: {w: 1.0}}}}"
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from nav_msgs.srv import GetPlan

from occ_planner import OccupancyGrid3DPathPlanner


def pose_msg_to_pos_quat(pose: Pose) -> dict:
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    quat = np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])
    return {"pos": pos, "quat": quat}


def pos_quat_to_pose_msg(pos: np.ndarray, quat: np.ndarray) -> Pose:
    p = Pose()
    p.position.x = float(pos[0])
    p.position.y = float(pos[1])
    p.position.z = float(pos[2])
    p.orientation.x = float(quat[0])
    p.orientation.y = float(quat[1])
    p.orientation.z = float(quat[2])
    p.orientation.w = float(quat[3])
    return p


class OccPlannerNode(Node):
    def __init__(self):
        super().__init__("occ_planner")

        # Planner: no free/occ grid for now, open space
        params = {
            "bounds": [-5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
            "use_free_grid": False,
            "use_occ_grid": False,
        }
        self._planner = OccupancyGrid3DPathPlanner(params)

        # Latest odom pose (updated continuously)
        self._latest_odom_pose = None
        self.create_subscription(Odometry, "/slam/base_odom", self._odom_cb, 10)

        # Transient local QoS so late subscribers still get the last path
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._path_pub = self.create_publisher(Path, "~/planned_path", latched_qos)
        self._waypoints_pub = self.create_publisher(PoseArray, "~/waypoints", latched_qos)

        # Service
        self.create_service(GetPlan, "~/plan_path", self._plan_path_cb)

        self.get_logger().info("OccPlanner ready. Listening on /slam/base_odom. Call ~/plan_path to plan.")

    def _odom_cb(self, msg: Odometry):
        self._latest_odom_pose = msg.pose.pose

    def _plan_path_cb(self, request: GetPlan.Request, response: GetPlan.Response):
        if self._latest_odom_pose is None:
            self.get_logger().error("No odom received yet, cannot plan")
            return response

        start_dict = pose_msg_to_pos_quat(self._latest_odom_pose)
        goal_dict = pose_msg_to_pos_quat(request.goal.pose)
        frame_id = request.goal.header.frame_id or "map"

        self.get_logger().info(
            f"Planning: {start_dict['pos']} (from odom) -> {goal_dict['pos']}"
        )

        if not self._planner.update_start_goal(start_dict, goal_dict):
            self.get_logger().error("Failed to set start/goal")
            return response

        if not self._planner.solve(time_limit=5.0, method="rrtstar"):
            self.get_logger().error("OMPL failed to find a solution")
            return response

        dense = self._planner.interpolate_path(num_interp_points=1)
        if dense is None:
            self.get_logger().error("Path interpolation failed")
            return response

        # Build and publish path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        for wp in dense:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose = pos_quat_to_pose_msg(wp["pos"], wp["quat"])
            path_msg.poses.append(ps)

        response.plan = path_msg
        self._path_pub.publish(path_msg)

        # Publish intermediate poses as PoseArray
        pa = PoseArray()
        pa.header = path_msg.header
        for wp in dense:
            pa.poses.append(pos_quat_to_pose_msg(wp["pos"], wp["quat"]))
        self._waypoints_pub.publish(pa)

        self.get_logger().info(f"Published path with {len(dense)} waypoints")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = OccPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
