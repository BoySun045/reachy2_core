#!/usr/bin/env python3
"""Publish a PLY point cloud as a ROS2 PointCloud2 message at 0.1 Hz.

Usage:
    python3 publish_scene.py /path/to/tsdf_fused.ply
"""

import sys
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

FRAME_ID = "map"


class ScenePublisher(Node):
    def __init__(self, ply_path: str):
        super().__init__("scene_publisher")

        self.get_logger().info(f"Loading {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        self.get_logger().info(f"Loaded {len(pcd.points)} points")

        self._fields = [
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

        self._xyzrgb = np.column_stack([pts, rgb_float32.reshape(-1, 1)])

        self._pub = self.create_publisher(PointCloud2, "/scene_pointcloud", 10)
        self.create_timer(10.0, self._publish)
        self.get_logger().info("Publishing on /scene_pointcloud at 0.1 Hz")

    def _publish(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = FRAME_ID
        msg = pc2.create_cloud(header, self._fields, self._xyzrgb)
        self._pub.publish(msg)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <path/to/pointcloud.ply>")
        sys.exit(1)

    rclpy.init()
    node = ScenePublisher(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
