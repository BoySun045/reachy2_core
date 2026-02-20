#!/usr/bin/env python3
"""
Test node for torso_ik_controller.

Publishes a target EE pose (torso frame) to /target_ee_pose_torso.
Every 30 seconds toggles between two positions.

Publishes ONCE per change (not continuously).  Subscribes to /odom so
it can show a world-frame marker that stays fixed even as the base moves.

RViz topics:
  /test_target_torso_marker  — axes in torso frame (moves with base)
  /test_target_world_marker  — axes in odom frame  (stays fixed)
"""

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


def make_target(x_offset=0.0):
    theta = np.deg2rad(0.0)
    theta2 = np.deg2rad(90.0)
    Rz0 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 0.0],
            [np.sin(theta),  np.cos(theta), 0.0, 0.0],
            [0.0,            0.0,           1.0, 0.0],
            [0.0,            0.0,           0.0, 1.0],
        ],
        dtype=float,
    )

    Rx0 = np.array(
        [
            [1, 0, 0.0, 0.0],
            [0,np.cos(theta2), -np.sin(theta2), 0.0],
            [0,np.sin(theta2),  np.cos(theta2), 0.0],
            [0.0,            0.0,           0.0, 1.0],

        ],
        dtype=float,
    )

    Ry0 = np.array(
        [
            [ np.cos(theta2), 0, np.sin(theta2), 0.0],
            [ 0,             1, 0,              0.0],
            [-np.sin(theta2), 0, np.cos(theta2),  0.0],
            [ 0.0,           0, 0.0,            1.0],
        ],
        dtype=float,
    )
    
    A = np.array(
        [
            [0, 0, -1, 0.1],
            [0, 1,  0, -0.4],
            [1, 0,  0, -0.2],
            [0, 0,  0,  1.0],
        ],
        dtype=float,
    )

    T =  A @ Rz0
    T =   Rz0 @ A
    T[0, 3] += x_offset
    return T


# Fixed transform: base_link → torso (same as in wholebody_ik_controller)
T_BASE_TO_TORSO = np.eye(4)
T_BASE_TO_TORSO[0, 3] = -0.01
T_BASE_TO_TORSO[2, 3] = 0.996


def _axes_markers(T, frame_id, ns, stamp):
    """Create RGB arrow markers for XYZ axes at a 4x4 pose."""
    pos = T[:3, 3]
    R = T[:3, :3]
    axis_len, shaft_d, head_d = 0.10, 0.008, 0.015
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    markers = []
    for i in range(3):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
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
        m.color.r, m.color.g, m.color.b = [float(c) for c in colors[i]]
        m.color.a = 1.0
        m.lifetime.sec = 0  # persistent until overwritten
        markers.append(m)
    return markers


class TestTorsoTargetPub(Node):
    def __init__(self):
        super().__init__("test_torso_target_pub")
        self.pub = self.create_publisher(PoseStamped, "/target_ee_pose_torso", 10)
        self.torso_marker_pub = self.create_publisher(
            MarkerArray, "/test_target_torso_marker", 10)
        self.world_marker_pub = self.create_publisher(
            MarkerArray, "/test_target_world_marker", 10)

        self.start_time = self.get_clock().now()
        self.offset = False
        self.published_offset = None  # track what was last published

        # Base state from /odom (for torso→odom conversion)
        self.base_x = None
        self.base_y = None
        self.base_yaw = None
        self.T_target_world = None  # frozen odom-frame target

        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_timer(0.1, self._tick)           # check for changes at 10 Hz
        self.create_timer(0.1, self._marker_tick)     # re-publish markers at 10 Hz

        self.get_logger().info(
            "Test publisher ready. Toggles every 30s.\n"
            "  /test_target_torso_marker — torso frame (moves with base)\n"
            "  /test_target_world_marker — odom frame  (stays fixed)"
        )

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.base_x = p.x
        self.base_y = p.y
        self.base_yaw = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')[2]

    def _tick(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        cycle = int(elapsed / 20.0)#30
        new_offset = (cycle % 2) == 1

        # Only publish when the target actually changes
        if new_offset == self.published_offset:
            return

        self.offset = new_offset
        self.published_offset = new_offset
        label = "+0.2m" if self.offset else "-0.2m"
        self.get_logger().info(f"Target changed → {label} (t={elapsed:.1f}s)")

        x_off = 1.4 if self.offset else 0.0
        T = make_target(x_off)

        # Publish PoseStamped (once)
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "torso"
        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pub.publish(msg)

        # Freeze the world-frame target NOW
        if self.base_x is not None:
            c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
            T_odom_base = np.array([
                [c, -s, 0, self.base_x],
                [s,  c, 0, self.base_y],
                [0,  0, 1, 0],
                [0,  0, 0, 1],
            ])
            self.T_target_world = T_odom_base @ T_BASE_TO_TORSO @ T
        else:
            self.T_target_world = None
            self.get_logger().warn("No odom yet — world marker not available")

        self.T_target_torso = T

    def _marker_tick(self):
        """Re-publish markers so they stay visible in RViz."""
        if self.published_offset is None:
            return

        stamp = self.get_clock().now().to_msg()

        # Torso-frame markers (these WILL move with the base — expected)
        ma_torso = MarkerArray()
        ma_torso.markers = _axes_markers(
            self.T_target_torso, "torso", "target_torso", stamp)
        self.torso_marker_pub.publish(ma_torso)

        # World-frame markers (these stay FIXED — the actual target)
        if self.T_target_world is not None:
            ma_world = MarkerArray()
            ma_world.markers = _axes_markers(
                self.T_target_world, "odom", "target_world", stamp)
            self.world_marker_pub.publish(ma_world)


def main():
    rclpy.init()
    node = TestTorsoTargetPub()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
