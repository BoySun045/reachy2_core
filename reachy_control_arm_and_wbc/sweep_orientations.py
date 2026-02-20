#!/usr/bin/env python3
"""
Sweep EE orientations at each sampled arm position.

Takes the existing ee_poses CSV, subsamples arm configurations
(shoulder + elbow joints), then grids the 3 wrist joints to get
dense orientation coverage at each position.  Appends results to a
new CSV and PLY.

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  python3 sweep_orientations.py --ros-args \
      -p n_positions:=500 \
      -p wrist_steps:=4 \
      -p settle_time:=0.12
"""

import csv
import time
from pathlib import Path
from itertools import product

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException

R_ARM_JOINTS = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
]
WRIST_LIMITS = {
    "r_wrist_roll":  (-0.7854, 0.7854),
    "r_wrist_pitch": (-0.7854, 0.7854),
    "r_wrist_yaw":   (-1.57,   1.57),
}
BASE_FRAME = "torso"
EE_FRAME = "r_arm_tip"
CMD_RATE = 80.0


def write_ply(path: Path, points: np.ndarray):
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


def make_wrist_grid(steps: int, margin: float = 0.05):
    """Generate grid of (wrist_roll, wrist_pitch, wrist_yaw) values."""
    grids = []
    for joint in ("r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"):
        lo, hi = WRIST_LIMITS[joint]
        grids.append(np.linspace(lo + margin, hi - margin, steps))
    # Give wrist_yaw more steps since it has a wider range
    yaw_lo, yaw_hi = WRIST_LIMITS["r_wrist_yaw"]
    yaw_steps = max(steps, int(steps * 1.5))
    grids[2] = np.linspace(yaw_lo + margin, yaw_hi - margin, yaw_steps)
    return list(product(*grids))


class OrientationSweeper(Node):
    def __init__(self):
        super().__init__("orientation_sweeper")

        self.declare_parameter("n_positions", 500)
        self.declare_parameter("wrist_steps", 4)
        self.declare_parameter("settle_time", 0.12)
        self.declare_parameter(
            "input_csv",
            str(Path(__file__).parent / "ee_poses_right.csv"),
        )
        self.declare_parameter("output_dir", str(Path(__file__).parent))

        self.n_positions = self.get_parameter("n_positions").value
        self.wrist_steps = self.get_parameter("wrist_steps").value
        self.settle_time = self.get_parameter("settle_time").value
        self.input_csv = self.get_parameter("input_csv").value
        self.output_dir = Path(self.get_parameter("output_dir").value)

        self.pub = self.create_publisher(
            Float64MultiArray,
            "/r_arm_forward_position_controller/commands",
            10,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info("Waiting for TF...")
        self.warmup_timer = self.create_timer(1.0, self._check_tf_ready)

    def _check_tf_ready(self):
        try:
            self.tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
            self.warmup_timer.cancel()
            self.get_logger().info("TF ready.")
            self._run()
        except Exception:
            self.get_logger().info("Still waiting for TF...")

    def _publish_and_hold(self, joint_values, duration):
        msg = Float64MultiArray()
        msg.data = joint_values
        interval = 1.0 / CMD_RATE
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            pub_msg = Float64MultiArray()
            pub_msg.data = joint_values
            self.pub.publish(pub_msg)
            rclpy.spin_once(self, timeout_sec=interval)

    def _run(self):
        # Load existing arm configurations (first 4 joints)
        arm_configs = []
        with open(self.input_csv) as f:
            for row in csv.DictReader(f):
                arm4 = [float(row[j]) for j in R_ARM_JOINTS[:4]]
                arm_configs.append(arm4)

        # Subsample positions uniformly
        n_total = len(arm_configs)
        if self.n_positions < n_total:
            indices = np.linspace(0, n_total - 1, self.n_positions, dtype=int)
            arm_configs = [arm_configs[i] for i in indices]
        n_pos = len(arm_configs)

        # Build wrist grid
        wrist_grid = make_wrist_grid(self.wrist_steps)
        n_ori = len(wrist_grid)
        n_samples = n_pos * n_ori
        est_time = n_samples * self.settle_time

        self.get_logger().info(
            f"Sweep: {n_pos} positions x {n_ori} orientations = "
            f"{n_samples} samples  (~{est_time:.0f}s / {est_time/60:.1f}min)"
        )

        csv_path = self.output_dir / "ee_poses_right_dense.csv"
        ply_path = self.output_dir / "ee_pointcloud_right_dense.ply"

        ee_points = []
        start_time = time.monotonic()
        sample_count = 0

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [j for j in R_ARM_JOINTS]
                + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
            )

            for pi, arm4 in enumerate(arm_configs):
                # First move the arm to this position (longer hold)
                first_wrist = list(wrist_grid[0])
                self._publish_and_hold(arm4 + first_wrist, self.settle_time * 2)

                for wi, wrist3 in enumerate(wrist_grid):
                    joint_values = arm4 + list(wrist3)
                    self._publish_and_hold(joint_values, self.settle_time)

                    try:
                        t = self.tf_buffer.lookup_transform(
                            BASE_FRAME, EE_FRAME, rclpy.time.Time()
                        )
                        pos = t.transform.translation
                        rot = t.transform.rotation
                        writer.writerow(
                            joint_values
                            + [pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w]
                        )
                        ee_points.append([pos.x, pos.y, pos.z])
                    except (LookupException, ExtrapolationException):
                        pass

                    sample_count += 1

                # Progress per position
                elapsed = time.monotonic() - start_time
                rate = sample_count / elapsed if elapsed > 0 else 0
                eta = (n_samples - sample_count) / rate if rate > 0 else 0
                self.get_logger().info(
                    f"Position {pi+1}/{n_pos}  "
                    f"({sample_count}/{n_samples} total, "
                    f"{rate:.1f}/s, ETA {eta:.0f}s)"
                )

        # Write PLY
        if ee_points:
            write_ply(ply_path, np.array(ee_points, dtype=np.float32))
            self.get_logger().info(
                f"Done! {len(ee_points)} samples saved.\n"
                f"  CSV: {csv_path}\n"
                f"  PLY: {ply_path}"
            )
        else:
            self.get_logger().error("No samples collected!")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = OrientationSweeper()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
