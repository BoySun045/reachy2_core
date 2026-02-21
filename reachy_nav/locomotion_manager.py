#!/usr/bin/env python3
"""
ROS2 node for relative base movement commands.

Receives locomotion commands (body-frame dx, dy, dyaw) from the LLM
command router, computes a target pose from the current robot pose,
and publishes a 2-pose Path (start, goal) for the trajectory follower.

Subscriptions:
    /locomotion/command  — JSON {"robot", "dx", "dy", "dyaw"} (std_msgs/String)
    /slam/base_odom      — reachy odometry (Odometry)
    /spot/odometry       — spot odometry (Odometry)

Publications:
    /path_planner/trajectory   — reachy 2-pose path (nav_msgs/Path)
    /spot/plan_path            — spot 2-pose path (nav_msgs/Path)

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

        # Per-robot current pose (updated by odometry callbacks)
        self._current_poses = {
            "reachy": None,
            "spot": None,
        }

        # Subscribers
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            Odometry, "/slam/base_odom",
            self._on_reachy_odom, 2,
        )
        self.create_subscription(
            Odometry, "/spot/odometry/corrected",
            self._on_spot_odom, 2,
        )
        self.create_subscription(
            String, "/locomotion/command",
            self._on_command, 10,
        )

        # Per-robot trajectory publishers
        self._traj_pubs = {
            "reachy": self.create_publisher(
                Path, "/path_planner/trajectory", latched_qos
            ),
            "spot": self.create_publisher(
                Path, "/spot/planned_path", latched_qos
            ),
        }

        self.get_logger().info("Locomotion manager ready.")

    def _on_reachy_odom(self, msg: Odometry):
        self._current_poses["reachy"] = msg.pose.pose

    def _on_spot_odom(self, msg: Odometry):
        self._current_poses["spot"] = msg.pose.pose

    def _on_command(self, msg: String):
        try:
            data = json.loads(msg.data)
            robot = data.get("robot", "reachy")
            dx = float(data.get("dx", 0.0))
            dy = float(data.get("dy", 0.0))
            dyaw = float(data.get("dyaw", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.get_logger().error(f"Bad locomotion command: {e}")
            return

        traj_pub = self._traj_pubs.get(robot)
        if traj_pub is None:
            self.get_logger().warn(
                f"No trajectory publisher for robot '{robot}'"
            )
            return

        current_pose = self._current_poses.get(robot)
        if current_pose is None:
            self.get_logger().error(
                f"No pose available for '{robot}' — cannot execute locomotion command"
            )
            return

        # Current pose
        p = current_pose.position
        cur_yaw = quat_to_yaw(current_pose.orientation)
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
            f"[{robot}] Locomotion: body dx={dx:.3f} dy={dy:.3f} "
            f"dyaw={math.degrees(dyaw):.1f}deg"
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
        traj_pub.publish(path)
        self.get_logger().info(f"[{robot}] Published 2-pose trajectory")


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
