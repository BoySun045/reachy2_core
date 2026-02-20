#!/usr/bin/env python3
"""
ROS2 node that follows a planned trajectory by publishing velocity commands.

Subscribes to the trajectory from the path planner and the live robot pose
from the localization node. Computes body-frame velocity commands (linear x/y
+ angular z) using proportional control and publishes them on /cmd_vel.

The robot's omni drive allows simultaneous translation and rotation.

Subscriptions:
    /path_planner/trajectory — planned waypoints (nav_msgs/Path)
    /localization/robot_pose — current robot pose (PoseStamped)

Publications:
    /cmd_vel — velocity commands (geometry_msgs/Twist)

Usage:
    python3 trajectory_follower_node.py
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
CONTROL_HZ = 10.0        # control loop rate (Hz)

KP_LINEAR = 0.8          # proportional gain for position (m/s per m error)
KP_ANGULAR = 1.5         # proportional gain for yaw (rad/s per rad error)

MAX_LINEAR_VEL = 0.3     # m/s clamp on linear velocity
MAX_ANGULAR_VEL = 0.8    # rad/s clamp on angular velocity

WAYPOINT_TOL = 0.15      # m — advance to next waypoint when within this
GOAL_TOL = 0.10          # m — position tolerance at final waypoint
YAW_TOL = 0.1            # rad — yaw tolerance at final waypoint

POSE_TIMEOUT = 1.0       # seconds — stop if no fresh pose for this long


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quat_to_yaw(quat) -> float:
    """Quaternion [x, y, z, w] → yaw angle (radians, around Z-up)."""
    return Rot.from_quat(quat).as_euler("xyz")[2]


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TrajectoryFollowerNode(Node):
    def __init__(self):
        super().__init__("trajectory_follower")

        # State
        self._path = None           # nav_msgs/Path
        self._waypoint_idx = 0
        self._active = False

        self._robot_x = None
        self._robot_y = None
        self._robot_yaw = None
        self._last_pose_time = 0.0

        # QoS for latched topics
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscribers
        self.create_subscription(
            Path,
            "/path_planner/trajectory",
            self._on_trajectory,
            latched_qos,
        )
        self.create_subscription(
            PoseStamped,
            "/localization/robot_pose",
            self._on_robot_pose,
            latched_qos,
        )

        # Publisher
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Control loop timer
        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

        self.get_logger().info(
            f"Trajectory follower ready (control @ {CONTROL_HZ} Hz, "
            f"Kp_lin={KP_LINEAR}, Kp_ang={KP_ANGULAR})"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_trajectory(self, msg: Path):
        n = len(msg.poses)
        if n == 0:
            self.get_logger().warn("Received empty trajectory, ignoring.")
            return

        self._path = msg
        self._waypoint_idx = 0
        self._active = True
        self.get_logger().info(f"New trajectory received: {n} waypoints")

    def _on_robot_pose(self, msg: PoseStamped):
        self._robot_x = msg.pose.position.x
        self._robot_y = msg.pose.position.y
        self._robot_yaw = quat_to_yaw(np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]))
        self._last_pose_time = self.get_clock().now().nanoseconds / 1e9

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def _control_loop(self):
        # Guard: nothing to do
        if not self._active or self._path is None:
            return

        if self._robot_x is None:
            self.get_logger().warn(
                "No robot pose yet, waiting...", throttle_duration_sec=5.0
            )
            return

        # Safety: stop if pose is stale
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_pose_time > POSE_TIMEOUT:
            self.get_logger().warn(
                "Robot pose timeout, stopping.", throttle_duration_sec=2.0
            )
            self._publish_stop()
            return

        # Current target waypoint
        target_pose = self._path.poses[self._waypoint_idx].pose
        tx = target_pose.position.x
        ty = target_pose.position.y
        target_yaw = quat_to_yaw(np.array([
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w,
        ]))

        # Position error in global frame
        dx = tx - self._robot_x
        dy = ty - self._robot_y
        dist = math.hypot(dx, dy)

        # Yaw error
        yaw_err = angle_wrap(target_yaw - self._robot_yaw)

        is_last = self._waypoint_idx >= len(self._path.poses) - 1
        pos_tol = GOAL_TOL if is_last else WAYPOINT_TOL

        # Check waypoint reached
        if dist < pos_tol:
            if is_last:
                # Final waypoint — also check yaw
                if abs(yaw_err) < YAW_TOL:
                    self.get_logger().info("Trajectory complete!")
                    self._publish_stop()
                    self._active = False
                    return
                # Still need to rotate to final heading
                twist = Twist()
                twist.angular.z = clamp(
                    KP_ANGULAR * yaw_err, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL
                )
                self._cmd_pub.publish(twist)
                return
            else:
                # Advance to next waypoint
                self._waypoint_idx += 1
                self.get_logger().info(
                    f"Waypoint {self._waypoint_idx}/{len(self._path.poses)} reached"
                )
                return

        # Rotate global error into body frame
        yaw = self._robot_yaw
        ex_body = math.cos(yaw) * dx + math.sin(yaw) * dy
        ey_body = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # P-control with clamping
        vx = clamp(KP_LINEAR * ex_body, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        vy = clamp(KP_LINEAR * ey_body, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        wz = clamp(KP_ANGULAR * yaw_err, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

        # Publish
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = wz
        self._cmd_pub.publish(twist)

    # ------------------------------------------------------------------
    def _publish_stop(self):
        self._cmd_pub.publish(Twist())

    def destroy_node(self):
        # Stop the robot on shutdown
        self._publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
