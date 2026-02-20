#!/usr/bin/env python3
"""
VidBot + GraspNet ROS 2 node.

Same as vidbot_ros_node but runs inference with --use_graspnet and uses the
AnyGrasp orientation for every waypoint.  Publishes PoseArray on
/vidbot/pre_poses and /vidbot/post_poses so you can visualize individual
pose arrows in RViz (add a PoseArray display, set the topic).
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
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped, Point, Vector3
from nav_msgs.msg import Path
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from vidbot_utils import (
    T_BASE_CAM,
    T_BASE_EE_DEFAULT,
    is_press_action,
    get_scale_for_instruction,
    make_T,
    transform_points,
    lerp_positions,
    mat4_to_pose,
)


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
    enc = msg.encoding
    if enc not in _ENCODING_MAP:
        raise ValueError(f"Unsupported image encoding: {enc}")
    dtype, channels = _ENCODING_MAP[enc]
    raw = np.frombuffer(msg.data, dtype=dtype)
    if channels > 1:
        img = raw.reshape(msg.height, msg.width, channels)
    else:
        img = raw.reshape(msg.height, msg.width)
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

INTRINSIC_PATH = os.path.join(DATASET_DIR, "camera_intrinsic.json")

FRAME_ID = "000000"
COLOR_PATH = os.path.join(COLOR_DIR, f"{FRAME_ID}.png")
DEPTH_PATH = os.path.join(DEPTH_DIR, f"{FRAME_ID}.png")

R_PATH = T_BASE_EE_DEFAULT[:3, :3]

BRIDGE_STEPS = 10

GRASP_POSE_PATH = os.path.join(VDBOT_ROOT, "grasp_pose.npy")
APPLY_Z_FLIP_FIX = True


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)



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


# ---- GraspNet orientation helpers ----
def load_T_any(path):
    T = np.load(path)
    T = np.asarray(T, dtype=float)
    if T.shape == (1, 4, 4):
        T = T[0]
    if T.shape != (4, 4):
        raise ValueError(f"{path} must be (4,4) or (1,4,4); got {T.shape}")
    return T


def rot_y_pi():
    return np.array(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
    )


def project_to_so3(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def get_constant_grasp_R_base():
    if not os.path.exists(GRASP_POSE_PATH):
        raise FileNotFoundError(f"Missing {GRASP_POSE_PATH}")

    T_grasp_cam = load_T_any(GRASP_POSE_PATH)
    T_grasp_base = T_BASE_CAM @ T_grasp_cam

    R = T_grasp_base[:3, :3].copy()
    R = project_to_so3(R)

    if APPLY_Z_FLIP_FIX:
        R = R @ rot_y_pi()
        R = project_to_so3(R)

    T_grasp_base[:3, :3] = R
    return T_grasp_base, R


def build_pre_post(pred_cam, loss, R_const):
    best_idx = int(np.argmin(loss))
    best_traj_cam = pred_cam[best_idx]

    best_traj_base = transform_points(T_BASE_CAM, best_traj_cam)

    ee_start = T_BASE_EE_DEFAULT[:3, 3]
    traj_start = best_traj_base[0]

    bridge = lerp_positions(ee_start, traj_start, BRIDGE_STEPS)

    pre_pos = np.vstack([
        ee_start[None, :],
        bridge,
        traj_start[None, :],
    ])
    post_pos = best_traj_base.copy()

    pre_T = np.stack([make_T(R_const, p) for p in pre_pos], axis=0)
    post_T = np.stack([make_T(R_const, p) for p in post_pos], axis=0)

    return best_idx, float(loss[best_idx]), pre_T, post_T


def run_infer_affordance(obj, action, use_graspnet=False, logger=None):
    scale = get_scale_for_instruction(action)
    if logger:
        logger.info(f"Scale for instruction '{action}': {scale}")
    cmd = [
        "python3",
        os.path.join(VDBOT_ROOT, "demos", "infer_affordance.py"),
        "-d", DATASET_NAME,
        "-f", FRAME_ID,
        "-o", obj,
        "-i", action,
        "-v",
        "-s", str(scale),
    ]
    if use_graspnet:
        cmd.append("--use_graspnet")

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = VDBOT_ROOT + (":" + old_pp if old_pp else "")

    if logger:
        logger.info(f"Running: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd, cwd=VDBOT_ROOT, env=env, capture_output=True, text=True,
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
    ps.pose = mat4_to_pose(T)
    return ps


def transforms_to_pose_array(transforms, stamp, frame_id):
    pa = PoseArray()
    pa.header.stamp = stamp
    pa.header.frame_id = frame_id
    for T in transforms:
        pa.poses.append(mat4_to_pose(T))
    return pa


def transforms_to_path(transforms, stamp, frame_id):
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id
    for T in transforms:
        path.poses.append(mat4_to_pose_stamped(T, stamp, frame_id))
    return path


# XYZ axis colors: X=red, Y=green, Z=blue
_AXIS_COLORS = [
    ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
]

AXIS_LENGTH = 0.05
AXIS_SHAFT_DIAMETER = 0.005
AXIS_HEAD_DIAMETER = 0.01
AXIS_HEAD_LENGTH = 0.01


def transforms_to_axes_markers(transforms, stamp, frame_id, ns="traj_axes"):
    """Build a MarkerArray with 3 arrow markers (X, Y, Z) per waypoint."""
    ma = MarkerArray()
    marker_id = 0
    for T in transforms:
        origin = T[:3, 3]
        R = T[:3, :3]
        for axis_idx in range(3):
            direction = R[:, axis_idx]
            tip = origin + direction * AXIS_LENGTH

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame_id
            m.ns = ns
            m.id = marker_id
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [
                Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])),
            ]
            m.scale = Vector3(
                x=AXIS_SHAFT_DIAMETER, y=AXIS_HEAD_DIAMETER, z=AXIS_HEAD_LENGTH
            )
            m.color = _AXIS_COLORS[axis_idx]
            ma.markers.append(m)
            marker_id += 1
    return ma


# -----------------------------
# Node
# -----------------------------
class VidBotGraspNetNode(Node):
    def __init__(self):
        super().__init__("vidbot_graspnet_node")

        self.declare_parameter("base_frame", "torso")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("use_graspnet", False)       # use AnyGrasp orientation vs default EE rotation
        self.declare_parameter("execute", True)            # send path to torso_ik_controller
        self.declare_parameter("post_skip", 8)             # keep every (k+1)-th waypoint in post path (0 = keep all)
        self.declare_parameter("press_skip_k", 3)           # multiplier for post_skip in press/push/click mode

        self.base_frame = self.get_parameter("base_frame").value
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self.use_graspnet = self.get_parameter("use_graspnet").value
        self.do_execute = self.get_parameter("execute").value
        self.post_skip = int(self.get_parameter("post_skip").value)
        self.press_skip_k = int(self.get_parameter("press_skip_k").value)

        self.latest_color = None
        self.latest_depth = None
        self.busy = False
        self._intrinsics_saved = False

        # Subscribers
        self.create_subscription(Image, color_topic, self._color_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(String, "/vidbot/trigger", self._trigger_cb, 10)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self._camera_info_cb, 10)

        # Path publishers (trajectory line in RViz)
        self.pre_traj_pub = self.create_publisher(Path, "/vidbot/pre_trajectory", 10)
        self.post_traj_pub = self.create_publisher(Path, "/vidbot/post_trajectory", 10)

        # PoseArray publishers (pose arrows in RViz)
        self.pre_poses_pub = self.create_publisher(PoseArray, "/vidbot/pre_poses", 10)
        self.post_poses_pub = self.create_publisher(PoseArray, "/vidbot/post_poses", 10)

        # MarkerArray publishers (XYZ axes per waypoint in RViz)
        self.pre_axes_pub = self.create_publisher(MarkerArray, "/vidbot/pre_axes", 10)
        self.post_axes_pub = self.create_publisher(MarkerArray, "/vidbot/post_axes", 10)

        self.get_logger().info(
            f"VidBot+GraspNet node ready. Subscribed to {color_topic}, {depth_topic}. "
            "Waiting for /vidbot/trigger ..."
        )

    def _color_cb(self, msg: Image):
        self.latest_color = imgmsg_to_cv2(msg)

    def _depth_cb(self, msg: Image):
        self.latest_depth = imgmsg_to_cv2(msg)

    def _camera_info_cb(self, msg: CameraInfo):
        if self._intrinsics_saved:
            return
        # ROS K is row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        # VidBot expects column-major: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        k = msg.k
        intrinsic = {
            "width": int(msg.width),
            "height": int(msg.height),
            "intrinsic_matrix": [
                k[0], 0.0, 0.0,   # column 0: fx, 0, 0
                0.0, k[4], 0.0,   # column 1: 0, fy, 0
                k[2], k[5], 1.0,  # column 2: cx, cy, 1
            ],
        }
        ensure_dirs()
        with open(INTRINSIC_PATH, "w") as f:
            json.dump(intrinsic, f, indent=4)
        self._intrinsics_saved = True
        self.get_logger().info(
            f"Saved camera intrinsics: {msg.width}x{msg.height}, "
            f"fx={k[0]:.2f}, fy={k[4]:.2f}, cx={k[2]:.2f}, cy={k[5]:.2f}"
        )

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

    def _run_inference(self, obj, instruction):
        try:
            self.get_logger().info(
                f'Inference started: object="{obj}", instruction="{instruction}"'
            )

            ensure_dirs()
            cv2.imwrite(COLOR_PATH, self.latest_color)
            cv2.imwrite(DEPTH_PATH, self.latest_depth)

            run_infer_affordance(
                obj, instruction,
                use_graspnet=self.use_graspnet,
                logger=self.get_logger(),
            )

            # Load predictions
            pred_npz = find_latest_prediction_npz()
            pred, loss = load_pred_and_loss(pred_npz)

            # Choose orientation: AnyGrasp if available, else default EE rotation
            if self.use_graspnet:
                _, R_const = get_constant_grasp_R_base()
                self.get_logger().info(
                    f"Using AnyGrasp orientation (z_flip={APPLY_Z_FLIP_FIX})"
                )
            else:
                R_const = R_PATH
                self.get_logger().info("Using default EE orientation (no graspnet)")

            best_idx, best_loss, pre_T, post_T = build_pre_post(
                pred, loss, R_const
            )

            # Thin post-grasp path: keep every (k+1)-th waypoint, always keep the last
            # For press actions, multiply skip by press_skip_k for denser waypoints
            is_press = is_press_action(instruction)
            effective_skip = self.post_skip * self.press_skip_k if is_press else self.post_skip

            if effective_skip > 0:
                step = effective_skip + 1
                indices = list(range(0, len(post_T), step))
                if indices[-1] != len(post_T) - 1:
                    indices.append(len(post_T) - 1)
                self.get_logger().info(
                    f"Post-grasp thinned: {len(post_T)} -> {len(indices)} waypoints "
                    f"(skip={effective_skip}, press={is_press})"
                )
                post_T = post_T[indices]

            # Publish
            stamp = self.get_clock().now().to_msg()
            frame = self.base_frame

            self.pre_traj_pub.publish(transforms_to_path(pre_T, stamp, frame))
            self.post_traj_pub.publish(transforms_to_path(post_T, stamp, frame))
            self.pre_poses_pub.publish(transforms_to_pose_array(pre_T, stamp, frame))
            self.post_poses_pub.publish(transforms_to_pose_array(post_T, stamp, frame))
            self.pre_axes_pub.publish(transforms_to_axes_markers(pre_T, stamp, frame, ns="pre_axes"))
            self.post_axes_pub.publish(transforms_to_axes_markers(post_T, stamp, frame, ns="post_axes"))

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
    node = VidBotGraspNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
