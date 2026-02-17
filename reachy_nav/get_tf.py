#!/usr/bin/env python3
"""
Get the 4x4 transformation matrix between two TF frames.

Usage:
    python3 get_tf.py <source_frame> <target_frame>

Example:
    python3 get_tf.py base_footprint map
    python3 get_tf.py camera_link odom

Outputs the 4x4 homogeneous transform T such that:
    point_in_target = T @ point_in_source
"""

import sys
import numpy as np

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class TFGetter(Node):
    def __init__(self, source_frame: str, target_frame: str):
        super().__init__("get_tf")
        self._source = source_frame
        self._target = target_frame
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(0.5, self._try_lookup)

    def _try_lookup(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._target, self._source, rclpy.time.Time()
            )
        except Exception:
            self.get_logger().info(
                f"Waiting for transform {self._source} -> {self._target} ..."
            )
            return

        # Extract translation
        tr = t.transform.translation
        tx, ty, tz = tr.x, tr.y, tr.z

        # Extract rotation (quaternion xyzw -> rotation matrix)
        q = t.transform.rotation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),      1 - 2*(qx*qx + qy*qy)],
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        print(f"\nTransform: {self._source} -> {self._target}")
        print(f"Translation: [{tx:.6f}, {ty:.6f}, {tz:.6f}]")
        print(f"Quaternion (xyzw): [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]")
        print(f"\n4x4 Matrix:")
        np.set_printoptions(precision=6, suppress=True)
        print(T)

        # Also print as copy-pasteable Python
        print(f"\nnp.array({T.tolist()})")

        raise SystemExit(0)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 get_tf.py <source_frame> <target_frame>")
        print("Example: python3 get_tf.py base_footprint map")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]

    rclpy.init()
    node = TFGetter(source, target)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
