#!/usr/bin/env python3
"""Publish a PLY point cloud as sensor_msgs/PointCloud2 for RViz."""

import struct
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def read_ply(path: Path) -> np.ndarray:
    """Read binary little-endian PLY with float x,y,z."""
    with open(path, "rb") as f:
        # Parse header
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_points = int(line.split()[-1])
            if line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_points, 3)
    return data


def make_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Build a PointCloud2 message from an Nx3 float32 array."""
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12  # 3 x float32
    msg.row_step = 12 * len(points)
    msg.data = points.astype(np.float32).tobytes()
    msg.is_dense = True
    return msg


class PointCloudPublisher(Node):
    def __init__(self, ply_path: str, frame_id: str):
        super().__init__("workspace_pointcloud_publisher")

        # Latching QoS so RViz picks it up even if it subscribes later
        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(
            PointCloud2, "/workspace_pointcloud", latching_qos
        )

        self.points = read_ply(Path(ply_path))
        self.frame_id = frame_id
        self.get_logger().info(
            f"Loaded {len(self.points)} points from {ply_path}, "
            f"publishing on /workspace_pointcloud in frame '{frame_id}'"
        )

        # Publish once immediately, then periodically at 1 Hz
        self._publish()
        self.timer = self.create_timer(1.0, self._publish)

    def _publish(self):
        msg = make_pointcloud2(
            self.points, self.frame_id, self.get_clock().now().to_msg()
        )
        self.pub.publish(msg)


def main():
    rclpy.init()
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "ee_pointcloud_right.ply"
    frame_id = sys.argv[2] if len(sys.argv) > 2 else "torso"
    node = PointCloudPublisher(ply_path, frame_id)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
