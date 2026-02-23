#!/usr/bin/env python3
"""
Long-running node that exposes a /scan_pose service.
Call it to move Spot arm to scan pose and open gripper.

Usage:
  python3 spot_scan_pose.py

  # From another terminal:
  ros2 service call /scan_pose std_srvs/srv/Trigger
"""
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


class SpotScanPoseNode(Node):
    def __init__(self):
        super().__init__("spot_scan_pose")

        self.declare_parameter("arm_cmd_frame", "body")
        self.declare_parameter("scan_pose_x", 0.25)
        self.declare_parameter("scan_pose_y", 0.0)
        self.declare_parameter("scan_pose_z", 0.25)
        self.declare_parameter("settle_time", 10.0)
        self.declare_parameter("gripper_wait", 3.0)

        self.arm_cmd_frame = self.get_parameter("arm_cmd_frame").value
        self.scan_pose = np.array([
            self.get_parameter("scan_pose_x").value,
            self.get_parameter("scan_pose_y").value,
            self.get_parameter("scan_pose_z").value,
        ])
        self.settle_time = self.get_parameter("settle_time").value
        self.gripper_wait = self.get_parameter("gripper_wait").value

        self.arm_cmd_pub = self.create_publisher(PoseStamped, "/spot/arm_pose_commands", 10)
        self.open_gripper_client = self.create_client(Trigger, "/spot/open_gripper")

        self.create_service(Trigger, "/scan_pose", self._scan_pose_cb)
        self.get_logger().info("Ready. Call: ros2 service call /scan_pose std_srvs/srv/Trigger")

    def _scan_pose_cb(self, request, response):
        # Run in thread so the service doesn't block the executor
        thread = threading.Thread(target=self._execute, daemon=True)
        thread.start()
        thread.join()
        response.success = True
        response.message = "Scan pose reached, gripper open."
        return response

    def _execute(self):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.arm_cmd_frame
        ps.pose.position.x = float(self.scan_pose[0])
        ps.pose.position.y = float(self.scan_pose[1])
        ps.pose.position.z = float(self.scan_pose[2])
        ps.pose.orientation.w = 1.0
        self.arm_cmd_pub.publish(ps)
        self.get_logger().info(
            f"Sent scan pose: ({self.scan_pose[0]}, {self.scan_pose[1]}, {self.scan_pose[2]}). "
            f"Waiting {self.settle_time}s..."
        )
        time.sleep(self.settle_time)

        if self.open_gripper_client.wait_for_service(timeout_sec=5.0):
            future = self.open_gripper_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.result() and future.result().success:
                self.get_logger().info("Gripper opened.")
            else:
                self.get_logger().error("Failed to open gripper.")
        else:
            self.get_logger().error("Open gripper service not available.")

        time.sleep(self.gripper_wait)
        self.get_logger().info("Scan pose ready.")


def main():
    rclpy.init()
    node = SpotScanPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
