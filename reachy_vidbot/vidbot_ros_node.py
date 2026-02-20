#!/usr/bin/env python3
"""
VidBot ROS 2 node.

Subscribes to camera color/depth topics, listens for a trigger on
/vidbot/trigger (JSON: {"object": "...", "instruction": "..."}),
runs VidBot inference, and publishes the resulting trajectory as
nav_msgs/Path on /vidbot/pre_trajectory and /vidbot/post_trajectory.
"""
from __future__ import annotations

import os
import glob
import json
import subprocess
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation


# Encoding -> (numpy dtype, number of channels)
_ENCODING_MAP = {
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "8UC3": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a numpy array (no cv_bridge needed)."""
    enc = msg.encoding
    if enc not in _ENCODING_MAP:
        raise ValueError(f"Unsupported image encoding: {enc}")
    dtype, channels = _ENCODING_MAP[enc]
    raw = np.frombuffer(msg.data, dtype=dtype)
    if channels > 1:
        img = raw.reshape(msg.height, msg.width, channels)
    else:
        img = raw.reshape(msg.height, msg.width)
    # Convert RGB → BGR so cv2.imwrite saves colors correctly
    if enc == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# -----------------------------
# CONFIG
# -----------------------------
VDBOT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET_NAME = "reachy2"
DATASET_DIR = os.path.join(VDBOT_ROOT, "datasets", DATASET_NAME)

COLOR_DIR = os.path.join(DATASET_DIR, "color")
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
PRED_DIR = os.path.join(DATASET_DIR, "prediction")

FRAME_ID = "000000"
COLOR_PATH = os.path.join(COLOR_DIR, f"{FRAME_ID}.png")
DEPTH_PATH = os.path.join(DEPTH_DIR, f"{FRAME_ID}.png")

T_BASE_CAM = np.array(
    [
        [-0.0, -0.7372773368,  0.6755902076,  0.0580000000],
        [-1.0, -0.0,          -0.0,           0.0250000000],
        [-0.0, -0.6755902076, -0.7372773368, -0.0300000000],
        [ 0.0,  0.0,           0.0,           1.0],
    ],
    dtype=float,
)


def _make_T_base_ee_default():
    theta = np.deg2rad(0.0)
    Rz0 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 0.0],
            [np.sin(theta),  np.cos(theta), 0.0, 0.0],
            [0.0,            0.0,           1.0, 0.0],
            [0.0,            0.0,           0.0, 1.0],
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
    return Rz0 @ A


T_BASE_EE = _make_T_base_ee_default()
R_PATH = T_BASE_EE[:3, :3]

BRIDGE_STEPS = 10


# -----------------------------
# Helpers (same logic as vidbot_server.py)
# -----------------------------
def ensure_dirs():
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]


def lerp_positions(p0, p1, num_steps):
    p0 = np.asarray(p0, float).reshape(3)
    p1 = np.asarray(p1, float).reshape(3)
    alphas = np.linspace(0.0, 1.0, num_steps + 2)[1:-1]
    return (1 - alphas)[:, None] * p0[None, :] + alphas[:, None] * p1[None, :]


def make_T(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, float).reshape(3)
    return T


def find_latest_prediction_npz():
    files = glob.glob(os.path.join(PRED_DIR, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No prediction .npz found in {PRED_DIR}")
    files.sort(key=os.path.getmtime)
    return files[-1]


def squeeze_pred(pred):
    pred = np.asarray(pred)
    pred = np.squeeze(pred)
    if pred.ndim == 3 and pred.shape[-1] == 3:
        return pred
    raise ValueError(f"Unexpected pred_trajectories shape after squeeze: {pred.shape}")


def squeeze_loss(loss):
    loss = np.asarray(loss)
    loss = np.squeeze(loss)
    if loss.ndim == 1:
        return loss
    raise ValueError(f"Unexpected loss shape after squeeze: {loss.shape}")


def load_pred_and_loss(pred_npz_path):
    z = np.load(pred_npz_path, allow_pickle=True)

    pred = None
    loss = None

    if "pred_trajectories" in z.files:
        pred = z["pred_trajectories"]
    if "guide_losses-total_loss" in z.files:
        loss = z["guide_losses-total_loss"]

    base_dir = os.path.dirname(pred_npz_path)
    if pred is None:
        p = os.path.join(base_dir, "pred_trajectories.npy")
        if os.path.exists(p):
            pred = np.load(p)
    if loss is None:
        p = os.path.join(base_dir, "guide_losses-total_loss.npy")
        if os.path.exists(p):
            loss = np.load(p)

    if pred is None or loss is None:
        raise FileNotFoundError(
            f"Could not find pred/loss in {pred_npz_path} or sibling .npy files."
        )

    pred = squeeze_pred(pred)
    loss = squeeze_loss(loss)

    if pred.shape[0] != loss.shape[0]:
        raise ValueError(f"Mismatch: pred N={pred.shape[0]} vs loss N={loss.shape[0]}")

    return pred, loss


def build_pre_post(pred_cam, loss):
    best_idx = int(np.argmin(loss))
    best_traj_cam = pred_cam[best_idx]

    best_traj_base = transform_points(T_BASE_CAM, best_traj_cam)

    ee_start = T_BASE_EE[:3, 3]
    traj_start = best_traj_base[0]

    bridge = lerp_positions(ee_start, traj_start, BRIDGE_STEPS)

    pre_pos = np.vstack([
        ee_start[None, :],
        bridge,
        traj_start[None, :],
    ])

    post_pos = best_traj_base.copy()

    pre_T = np.stack([make_T(R_PATH, p) for p in pre_pos], axis=0)
    post_T = np.stack([make_T(R_PATH, p) for p in post_pos], axis=0)

    return best_idx, float(loss[best_idx]), pre_T, post_T


def run_infer_affordance(obj, action, logger=None):
    cmd = [
        "python3",
        os.path.join(VDBOT_ROOT, "demos", "infer_affordance.py"),
        "-d", DATASET_NAME,
        "-f", FRAME_ID,
        "-o", obj,
        "-i", action,
    ]

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = VDBOT_ROOT + (":" + old_pp if old_pp else "")

    if logger:
        logger.info(f"Running: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=VDBOT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"infer_affordance failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    return proc.stdout


# -----------------------------
# ROS helpers
# -----------------------------
def mat4_to_pose_stamped(T, stamp, frame_id):
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(T[0, 3])
    ps.pose.position.y = float(T[1, 3])
    ps.pose.position.z = float(T[2, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    ps.pose.orientation.x = float(q[0])
    ps.pose.orientation.y = float(q[1])
    ps.pose.orientation.z = float(q[2])
    ps.pose.orientation.w = float(q[3])
    return ps


def transforms_to_path(transforms, stamp, frame_id):
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id
    for T in transforms:
        path.poses.append(mat4_to_pose_stamped(T, stamp, frame_id))
    return path


# -----------------------------
# Node
# -----------------------------
class VidBotNode(Node):
    def __init__(self):
        super().__init__("vidbot_node")

        self.declare_parameter("base_frame", "torso")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")

        self.base_frame = self.get_parameter("base_frame").value
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value

        self.latest_color = None
        self.latest_depth = None
        self.busy = False

        # Subscribers
        self.create_subscription(Image, color_topic, self._color_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(String, "/vidbot/trigger", self._trigger_cb, 10)

        # Publishers
        self.pre_traj_pub = self.create_publisher(Path, "/vidbot/pre_trajectory", 10)
        self.post_traj_pub = self.create_publisher(Path, "/vidbot/post_trajectory", 10)

        self.get_logger().info(
            f"VidBot node ready. Subscribed to {color_topic}, {depth_topic}. "
            "Waiting for /vidbot/trigger ..."
        )

    # ---- callbacks ----
    def _color_cb(self, msg: Image):
        self.latest_color = imgmsg_to_cv2(msg)

    def _depth_cb(self, msg: Image):
        self.latest_depth = imgmsg_to_cv2(msg)

    def _trigger_cb(self, msg: String):
        if self.busy:
            self.get_logger().warn("Inference already running, ignoring trigger")
            return

        try:
            data = json.loads(msg.data)
            obj = data["object"]
            instruction = data["instruction"]
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"Bad trigger payload: {e}")
            return

        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().error("No images received yet, ignoring trigger")
            return

        self.busy = True
        thread = threading.Thread(
            target=self._run_inference, args=(obj, instruction), daemon=True
        )
        thread.start()

    # ---- inference (runs in background thread) ----
    def _run_inference(self, obj, instruction):
        try:
            self.get_logger().info(
                f'Inference started: object="{obj}", instruction="{instruction}"'
            )

            # Snapshot current images
            ensure_dirs()
            cv2.imwrite(COLOR_PATH, self.latest_color)
            cv2.imwrite(DEPTH_PATH, self.latest_depth)

            # Run vidbot
            run_infer_affordance(obj, instruction, logger=self.get_logger())

            # Load results
            pred_npz = find_latest_prediction_npz()
            pred, loss = load_pred_and_loss(pred_npz)
            best_idx, best_loss, pre_T, post_T = build_pre_post(pred, loss)

            # Publish trajectories
            stamp = self.get_clock().now().to_msg()
            self.pre_traj_pub.publish(
                transforms_to_path(pre_T, stamp, self.base_frame)
            )
            self.post_traj_pub.publish(
                transforms_to_path(post_T, stamp, self.base_frame)
            )

            self.get_logger().info(
                f"Published: pre={pre_T.shape[0]} poses, post={post_T.shape[0]} poses "
                f"(best_idx={best_idx}, loss={best_loss:.4f})"
            )

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
        finally:
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = VidBotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
