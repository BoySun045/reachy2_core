#!/usr/bin/env python3
"""
ROS2 Pure Pursuit path-following controller for a planar (differential-drive) robot.

Subscribes to:
    /odom                           - nav_msgs/Odometry (robot pose)
    /occ_planner/planned_path       - nav_msgs/Path (from occ_planner_ros2)

Publishes:
    /cmd_vel                        - geometry_msgs/Twist (velocity commands)
    ~/target_waypoint               - geometry_msgs/PointStamped (for RViz debug)

Usage:
    python3 pure_pursuit_ros2.py

    # With parameter overrides:
    python3 pure_pursuit_ros2.py --ros-args \
        -p Vcmd:=0.5 -p Lfw:=0.8 -p goal_radius:=0.3
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry, Path
from transforms3d.euler import quat2euler


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

        # --- Parameters ---
        self.declare_parameter("Vcmd", 0.5)           # reference forward speed (m/s), when on path (can be 0 for pure rotation in place)
        self.declare_parameter("Lfw", 0.3)             # look-ahead distance (m)
        self.declare_parameter("goal_radius", 0.1)     # stop when this close to goal (m)
        self.declare_parameter("goal_yaw_tol", 0.1)    # stop when yaw error below this (rad, ~5.7 deg)
        self.declare_parameter("controller_freq", 30)   # control loop Hz
        self.declare_parameter("linear_gain", 0.8)
        self.declare_parameter("steering_gain", 0.3)
        self.declare_parameter("max_v", 0.5)           # max linear velocity (m/s)
        self.declare_parameter("max_w", 0.3)           # max angular velocity (rad/s)
        self.declare_parameter("base_angle", 0.0)      # neutral steering offset (rad)
        self.declare_parameter("smooth_accel", True)
        self.declare_parameter("speed_incremental", 0.05)  # m/s per tick
        self.declare_parameter("path_topic", "/occ_planner/planned_path")
        self.declare_parameter("waypoint_advance_hz", 1.0)  # how fast to advance through waypoints

        self._Vcmd = self.get_parameter("Vcmd").value
        self._Lfw = self.get_parameter("Lfw").value
        self._goal_radius = self.get_parameter("goal_radius").value
        self._goal_yaw_tol = self.get_parameter("goal_yaw_tol").value
        self._freq = self.get_parameter("controller_freq").value
        self._linear_gain = self.get_parameter("linear_gain").value
        self._steering_gain = self.get_parameter("steering_gain").value
        self._max_v = self.get_parameter("max_v").value
        self._max_w = self.get_parameter("max_w").value
        self._base_angle = self.get_parameter("base_angle").value
        self._smooth_accel = self.get_parameter("smooth_accel").value
        self._speed_inc = self.get_parameter("speed_incremental").value
        path_topic = self.get_parameter("path_topic").value
        self._wp_advance_hz = self.get_parameter("waypoint_advance_hz").value

        # --- State ---
        self._odom = None          # latest Odometry msg
        self._path = None          # latest Path msg
        self._goal_reached = False
        self._velocity = 0.0
        self._wp_idx = 0           # current target waypoint index

        # --- Subscribers ---
        self.create_subscription(Odometry, "/odom", self._odom_cb, 2)

        # Transient local QoS to receive latched path from planner
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, path_topic, self._path_cb, latched_qos)

        # --- Publishers ---
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._target_pub = self.create_publisher(PointStamped, "~/target_waypoint", 10)

        # --- Control timer (high freq for smooth commands) ---
        self.create_timer(1.0 / self._freq, self._control_loop)

        # --- Waypoint advance timer (separate, controllable rate) ---
        self.create_timer(1.0 / self._wp_advance_hz, self._advance_waypoint)

        self.get_logger().info(
            f"PurePursuit ready. Vcmd={self._Vcmd}, Lfw={self._Lfw}, "
            f"goal_radius={self._goal_radius}, freq={self._freq}Hz, "
            f"path_topic={path_topic}"
        )

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    def _path_cb(self, msg: Path):
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty path, ignoring")
            return
        self._path = msg
        self._goal_reached = False
        self._velocity = 0.0
        self._wp_idx = 0
        self.get_logger().info(f"Received new path with {len(msg.poses)} poses")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_yaw(self) -> float:
        """Extract yaw from current odom quaternion."""
        q = self._odom.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        return yaw

    def _get_pos(self):
        """Return (x, y) from current odom."""
        p = self._odom.pose.pose.position
        return p.x, p.y

    def _is_forward(self, wx: float, wy: float, cx: float, cy: float, yaw: float) -> bool:
        """Check if waypoint (wx, wy) is in front of the robot."""
        dx = wx - cx
        dy = wy - cy
        # Transform to robot frame
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        return local_x > 0.0

    def _dist(self, x1, y1, x2, y2) -> float:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # ------------------------------------------------------------------ #
    #  Waypoint advance (separate timer)
    # ------------------------------------------------------------------ #

    def _is_last_waypoint(self) -> bool:
        return self._path is not None and self._wp_idx >= len(self._path.poses) - 1

    def _advance_waypoint(self):
        """Advance to the next waypoint at waypoint_advance_hz, unless on the last one."""
        if self._path is None or self._goal_reached:
            return
        if self._is_last_waypoint():
            return  # stay on last waypoint until goal reached
        self._wp_idx = min(self._wp_idx + 1, len(self._path.poses) - 1)
        self.get_logger().debug(
            f"Advanced to waypoint {self._wp_idx}/{len(self._path.poses) - 1}"
        )

    # ------------------------------------------------------------------ #
    #  Core pure pursuit control loop
    # ------------------------------------------------------------------ #

    def _control_loop(self):
        cmd = Twist()

        if self._odom is None or self._path is None or self._goal_reached:
            self._cmd_pub.publish(cmd)  # zero velocity
            return

        cx, cy = self._get_pos()
        yaw = self._get_yaw()

        # Current target waypoint
        target_pose = self._path.poses[self._wp_idx].pose
        tx = target_pose.position.x
        ty = target_pose.position.y

        # --- Final waypoint: check goal reached (position + orientation) ---
        if self._is_last_waypoint():
            dist_to_goal = self._dist(cx, cy, tx, ty)
            goal_q = target_pose.orientation
            _, _, goal_yaw = quat2euler([goal_q.x, goal_q.y, goal_q.z, goal_q.w])
            yaw_error = abs(math.atan2(math.sin(yaw - goal_yaw), math.cos(yaw - goal_yaw)))

            pos_reached = dist_to_goal < self._goal_radius
            yaw_reached = yaw_error < self._goal_yaw_tol

            if pos_reached and yaw_reached:
                self._goal_reached = True
                self._velocity = 0.0
                self._cmd_pub.publish(cmd)
                self.get_logger().info(
                    f"Goal reached! dist={dist_to_goal:.3f}m, "
                    f"yaw_err={math.degrees(yaw_error):.1f}deg"
                )
                return

            # Close in position but not yaw: rotate in place
            if pos_reached and not yaw_reached:
                yaw_diff = math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw))
                w = self._steering_gain * yaw_diff
                w = max(-self._max_w, min(self._max_w, w))
                cmd.angular.z = w
                self._cmd_pub.publish(cmd)
                return

        # --- Publish target for RViz ---
        tp = PointStamped()
        tp.header.stamp = self.get_clock().now().to_msg()
        tp.header.frame_id = self._path.header.frame_id or "odom"
        tp.point.x = tx
        tp.point.y = ty
        self._target_pub.publish(tp)

        # --- Heading error to current target ---
        alpha = math.atan2(ty - cy, tx - cx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Angular velocity: pure pursuit formula
        w = self._base_angle + self._steering_gain * (2.0 * math.sin(alpha) / self._Lfw)
        w = max(-self._max_w, min(self._max_w, w))

        # Linear velocity
        if self._smooth_accel:
            self._velocity = min(self._velocity + self._speed_inc, self._Vcmd)
        else:
            self._velocity = self._Vcmd

        v = self._linear_gain * self._velocity
        v = max(-self._max_v, min(self._max_v, v))

        cmd.linear.x = v
        cmd.angular.z = w
        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot on shutdown
        stop = Twist()
        node._cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
