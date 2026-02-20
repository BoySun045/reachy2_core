#!/usr/bin/env python3
"""
ROS2 node for relative base movement commands.

Receives locomotion commands (body-frame dx, dy, dyaw) from the LLM
command router, computes a target pose from the current robot pose,
and publishes a 2-pose Path (start, goal) for the trajectory follower.

Subscriptions:
    /locomotion/command       — JSON {"dx", "dy", "dyaw"} (std_msgs/String)
    /localization/robot_pose  — current robot pose (PoseStamped)

Publications:
    /path_planner/trajectory  — 2-pose path [current, target] (nav_msgs/Path)

Usage:
    python3 locomotion_manager.py
"""

import json
import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String

FRAME_ID = "map"


def quat_to_yaw(q):
    """Extract yaw from a quaternion (x, y, z, w)."""
    return Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]


def yaw_to_quat(yaw):
    """Convert yaw to quaternion (x, y, z, w)."""
    return Rot.from_euler("z", yaw).as_quat()


class LocomotionManager(Node):
    def __init__(self):
        super().__init__("locomotion_manager")

        self.declare_parameter("trajectory_topic", "/occ_planner/planned_path")
        traj_topic = self.get_parameter("trajectory_topic").value

        # Current pose (updated by localization)
        self._current_pose = None

        # Subscribers
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            Odometry, "/slam/base_odom",
            self._on_odom, 2,
        )
        self.create_subscription(
            String, "/locomotion/command",
            self._on_command, 10,
        )

        # Publisher
        self._traj_pub = self.create_publisher(Path, traj_topic, latched_qos)

        self.get_logger().info(
            f"Locomotion manager ready. Publishing to {traj_topic}"
        )

    def _on_odom(self, msg: Odometry):
        self._current_pose = msg.pose.pose

    def _on_command(self, msg: String):
        try:
            data = json.loads(msg.data)
            dx = float(data.get("dx", 0.0))
            dy = float(data.get("dy", 0.0))
            dyaw = float(data.get("dyaw", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.get_logger().error(f"Bad locomotion command: {e}")
            return

        if self._current_pose is None:
            self.get_logger().error(
                "No robot pose available — cannot execute locomotion command"
            )
            return

        # Current pose
        p = self._current_pose.position
        cur_yaw = quat_to_yaw(self._current_pose.orientation)
        cur_x, cur_y = p.x, p.y

        # Transform body-frame deltas to map frame
        cos_yaw = math.cos(cur_yaw)
        sin_yaw = math.sin(cur_yaw)
        map_dx = cos_yaw * dx - sin_yaw * dy
        map_dy = sin_yaw * dx + cos_yaw * dy

        # Target pose
        tgt_x = cur_x + map_dx
        tgt_y = cur_y + map_dy
        tgt_yaw = cur_yaw + dyaw

        self.get_logger().info(
            f"Locomotion: body dx={dx:.3f} dy={dy:.3f} dyaw={math.degrees(dyaw):.1f}deg"
        )
        self.get_logger().info(
            f"  Current: [{cur_x:.3f}, {cur_y:.3f}] yaw={math.degrees(cur_yaw):.1f}deg"
        )
        self.get_logger().info(
            f"  Target:  [{tgt_x:.3f}, {tgt_y:.3f}] yaw={math.degrees(tgt_yaw):.1f}deg"
        )

        # Build 2-pose path
        path = Path()
        path.header.frame_id = FRAME_ID
        path.header.stamp = self.get_clock().now().to_msg()

        # Start pose
        start = PoseStamped()
        start.header = path.header
        start.pose.position.x = cur_x
        start.pose.position.y = cur_y
        start.pose.position.z = p.z
        q = yaw_to_quat(cur_yaw)
        start.pose.orientation.x = q[0]
        start.pose.orientation.y = q[1]
        start.pose.orientation.z = q[2]
        start.pose.orientation.w = q[3]

        # Goal pose
        goal = PoseStamped()
        goal.header = path.header
        goal.pose.position.x = tgt_x
        goal.pose.position.y = tgt_y
        goal.pose.position.z = p.z
        q = yaw_to_quat(tgt_yaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        path.poses = [start, goal]
        self._traj_pub.publish(path)
        self.get_logger().info("Published 2-pose trajectory")


def main(args=None):
    rclpy.init(args=args)
    node = LocomotionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
