#!/usr/bin/env python3
"""
Sample end-effector poses by publishing random joint values within limits.

Publishes random joint configurations to both arms via the
forward_position_controllers, continuously holds the command until the
robot settles, then reads the resulting EE pose from TF.  Saves a CSV
with joint values + EE poses AND a PLY point cloud for visualisation.

Usage:
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=0
  python3 ee_pose_sampler.py --ros-args \
      -p num_samples:=5000 \
      -p settle_time:=0.3 \
      -p arm:=right \
      -p output_dir:=/home/marwan/marwan_ws/reachy_sim
"""

import csv
import struct
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException

# Joint names in controller order (must match forward_position_controller params)
R_ARM_JOINTS = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_elbow_yaw",
    "r_elbow_pitch",
    "r_wrist_roll",
    "r_wrist_pitch",
    "r_wrist_yaw",
]

L_ARM_JOINTS = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_elbow_yaw",
    "l_elbow_pitch",
    "l_wrist_roll",
    "l_wrist_pitch",
    "l_wrist_yaw",
]

# Joint limits (lower, upper) in radians — from reachy2.urdf
JOINT_LIMITS = {
    "r_shoulder_pitch": (-1.5708, 1.5708),
    "r_shoulder_roll":  (-1.5708, 0.0),
    "r_elbow_yaw":      (-1.5708, 1.5708),
    "r_elbow_pitch":    (-2.25,   0.1),
    "r_wrist_roll":     (-0.7854, 0.7854),
    "r_wrist_pitch":    (-0.7854, 0.7854),
    "r_wrist_yaw":      (-1.57,   1.57),

    "l_shoulder_pitch": (-1.5708, 1.5708),
    "l_shoulder_roll":  (0.0,     1.5708),
    "l_elbow_yaw":      (-1.5708, 1.5708),
    "l_elbow_pitch":    (-2.25,   0.1),
    "l_wrist_roll":     (-0.7854, 0.7854),
    "l_wrist_pitch":    (-0.7854, 0.7854),
    "l_wrist_yaw":      (-1.57,   1.57),
}

# TF frames for end-effectors
EE_FRAMES = {
    "right": "r_arm_tip",
    "left":  "l_arm_tip",
}
BASE_FRAME = "torso"

# Publish rate during settle period (Hz)
CMD_RATE = 50.0


def write_ply(path: Path, points: np.ndarray):
    """Write an Nx3 float array as a binary PLY point cloud."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype(np.float32).tobytes())


class EEPoseSampler(Node):
    def __init__(self):
        super().__init__("ee_pose_sampler")

        # Parameters
        self.declare_parameter("num_samples", 5000)
        self.declare_parameter("settle_time", 0.3)   # seconds to hold command
        self.declare_parameter("arm", "right")        # "right", "left", or "both"
        self.declare_parameter("output_dir", str(Path.cwd()))
        self.declare_parameter("margin", 0.05)        # stay this far (rad) from limits

        self.num_samples = self.get_parameter("num_samples").value
        self.settle_time = self.get_parameter("settle_time").value
        self.arm = self.get_parameter("arm").value
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.margin = self.get_parameter("margin").value

        # Publishers
        self.r_pub = self.create_publisher(
            Float64MultiArray,
            "/r_arm_forward_position_controller/commands",
            10,
        )
        self.l_pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10,
        )

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Latest joint state
        self.latest_joint_state = None
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )

        # Wait for TF tree and subscriptions to populate
        self.get_logger().info("Waiting for TF tree to populate...")
        self.warmup_timer = self.create_timer(1.0, self._check_tf_ready)

    def _check_tf_ready(self):
        """Wait until the torso -> arm_tip TF chain is available."""
        try:
            self.tf_buffer.lookup_transform(
                BASE_FRAME, EE_FRAMES["right"], rclpy.time.Time()
            )
            self.warmup_timer.cancel()
            self.get_logger().info("TF tree ready.")
            self._start_sampling()
        except Exception:
            self.get_logger().info("Still waiting for TF...")

    def _joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def _publish_and_hold(self, pub, joint_values, duration):
        """Publish the command repeatedly for `duration` seconds at CMD_RATE Hz,
        spinning between publishes so TF and joint_states stay up to date."""
        msg = Float64MultiArray()
        msg.data = joint_values
        interval = 1.0 / CMD_RATE
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=interval)

    def _start_sampling(self):
        self.warmup_timer.cancel()

        est_time = self.num_samples * self.settle_time
        self.get_logger().info(
            f"Starting EE pose sampling: {self.num_samples} samples, "
            f"arm={self.arm}, settle_time={self.settle_time}s  "
            f"(~{est_time:.0f}s / {est_time/60:.1f}min)"
        )

        arms_to_sample = []
        if self.arm in ("right", "both"):
            arms_to_sample.append("right")
        if self.arm in ("left", "both"):
            arms_to_sample.append("left")

        for side in arms_to_sample:
            self._sample_arm(side)

        self.get_logger().info("Sampling complete. Shutting down.")
        rclpy.shutdown()

    def _sample_arm(self, side: str):
        joints = R_ARM_JOINTS if side == "right" else L_ARM_JOINTS
        pub = self.r_pub if side == "right" else self.l_pub
        ee_frame = EE_FRAMES[side]

        csv_path = self.output_dir / f"ee_poses_{side}.csv"
        ply_path = self.output_dir / f"ee_pointcloud_{side}.ply"
        self.get_logger().info(f"[{side}] Saving CSV to {csv_path}")
        self.get_logger().info(f"[{side}] Saving PLY to {ply_path}")

        ee_points = []
        start_time = time.monotonic()

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = (
                [j for j in joints]
                + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
            )
            writer.writerow(header)

            for i in range(self.num_samples):
                # Generate random joint values within limits (with margin)
                joint_values = []
                for j in joints:
                    lo, hi = JOINT_LIMITS[j]
                    lo_m = lo + self.margin
                    hi_m = hi - self.margin
                    if lo_m >= hi_m:
                        lo_m, hi_m = lo, hi
                    joint_values.append(np.random.uniform(lo_m, hi_m))

                # Continuously publish while waiting for the robot to settle
                self._publish_and_hold(pub, joint_values, self.settle_time)

                # Look up EE pose from TF
                try:
                    t = self.tf_buffer.lookup_transform(
                        BASE_FRAME, ee_frame, rclpy.time.Time()
                    )
                    pos = t.transform.translation
                    rot = t.transform.rotation
                    row = joint_values + [
                        pos.x, pos.y, pos.z,
                        rot.x, rot.y, rot.z, rot.w,
                    ]
                    writer.writerow(row)
                    ee_points.append([pos.x, pos.y, pos.z])
                except (LookupException, ExtrapolationException) as e:
                    self.get_logger().warn(
                        f"[{side}] Sample {i+1}: TF lookup failed: {e}"
                    )
                    continue

                # Progress every 100 samples
                if (i + 1) % 100 == 0 or (i + 1) == self.num_samples:
                    elapsed = time.monotonic() - start_time
                    rate = (i + 1) / elapsed
                    eta = (self.num_samples - i - 1) / rate if rate > 0 else 0
                    self.get_logger().info(
                        f"[{side}] {i+1}/{self.num_samples}  "
                        f"({rate:.1f} samples/s, ETA {eta:.0f}s)"
                    )

        # Write PLY point cloud
        if ee_points:
            pts = np.array(ee_points, dtype=np.float32)
            write_ply(ply_path, pts)
            self.get_logger().info(
                f"[{side}] Done. {len(ee_points)} points saved.\n"
                f"  CSV: {csv_path}\n"
                f"  PLY: {ply_path}"
            )
        else:
            self.get_logger().error(f"[{side}] No valid samples collected!")


def main():
    rclpy.init()
    node = EEPoseSampler()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
