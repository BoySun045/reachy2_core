#!/usr/bin/env python3
"""
ROS2 node wrapping OccupancyGrid3DPathPlanner.

Provides a GetPlan service. Plans once when called, publishes the result
on ~/planned_path, and keeps it latched for path followers.

Usage:
    python3 occ_planner_ros2.py

    # Trigger from another terminal:
    ros2 service call /occ_planner/plan_path nav_msgs/srv/GetPlan \
      "{start: {header: {frame_id: 'map'}, pose: {position: {x: 0, y: 0, z: 0}, orientation: {w: 1.0}}}, \
        goal: {header: {frame_id: 'map'}, pose: {position: {x: 2, y: 2, z: 0}, orientation: {w: 1.0}}}}"
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path
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

        # Transient local QoS so late subscribers still get the last path
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._path_pub = self.create_publisher(Path, "~/planned_path", latched_qos)

        # Service
        self.create_service(GetPlan, "~/plan_path", self._plan_path_cb)

        self.get_logger().info("OccPlanner ready. Call ~/plan_path to plan.")

    def _plan_path_cb(self, request: GetPlan.Request, response: GetPlan.Response):
        start_dict = pose_msg_to_pos_quat(request.start.pose)
        goal_dict = pose_msg_to_pos_quat(request.goal.pose)
        frame_id = request.goal.header.frame_id or "map"

        self.get_logger().info(
            f"Planning: {start_dict['pos']} -> {goal_dict['pos']}"
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
