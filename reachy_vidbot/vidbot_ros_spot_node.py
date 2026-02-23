#!/usr/bin/env python3
"""
VidBot ROS 2 node for Boston Dynamics Spot.

Adapted from vidbot_ros_graspnet_node.py (Reachy-2) for Spot's hand camera
and arm_pose_commands interface.

Flow:
  1. Trigger arrives on /vidbot/trigger
  2. Unstow arm to scan pose + open gripper
  3. Capture RGB-D from hand camera
  4. Run VidBot inference (subprocess)
  5. Build & publish pre/post trajectories
  6. Execute trajectory via /spot/arm_pose_commands (one PoseStamped at a time)
  7. Stow arm

Topics:
  Subscribes:
    /spot/camera/hand/image          (Image)      — hand RGB
    /spot/depth_registered/hand/image (Image)     — registered depth
    /spot/depth_registered/hand/camera_info (CameraInfo)
    /vidbot/trigger                   (String)    — JSON trigger

  Publishes:
    /spot/arm_pose_commands           (PoseStamped) — arm waypoints
    /vidbot/pre_trajectory, /vidbot/post_trajectory (Path)
    /vidbot/pre_poses, /vidbot/post_poses (PoseArray)
    /vidbot/pre_axes, /vidbot/post_axes (MarkerArray)

  Service clients:
    /spot/arm_stow, /spot/open_gripper, /spot/close_gripper (Trigger)
"""
from __future__ import annotations

import os
import glob
import json
import subprocess
import threading
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped, Point, Vector3, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import String, ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
import tf2_ros

from vidbot_utils import (
    is_press_action,
    get_scale_for_instruction,
    make_T,
    transform_points,
    lerp_positions,
    mat4_to_pose,
)


# ---------------------------------------------------------------------------
# Image conversion (no cv_bridge dependency)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VDBOT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET_NAME = "spot"
DATASET_DIR = os.path.join(VDBOT_ROOT, "datasets", DATASET_NAME)

COLOR_DIR = os.path.join(DATASET_DIR, "color")
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
PRED_DIR = os.path.join(DATASET_DIR, "prediction")
INTRINSIC_PATH = os.path.join(DATASET_DIR, "camera_intrinsic.json")

FRAME_ID = "000000"
COLOR_PATH = os.path.join(COLOR_DIR, f"{FRAME_ID}.png")
DEPTH_PATH = os.path.join(DEPTH_DIR, f"{FRAME_ID}.png")

BRIDGE_STEPS = 10

GRASP_POSE_PATH = os.path.join(VDBOT_ROOT, "grasp_pose.npy")
APPLY_Z_FLIP_FIX = True

# VidBot expects ~16:9 input (resizes to 456x256 uniformly).
# Spot hand camera is 4:3 (640x480). Letterbox to match expected aspect ratio.
VIDBOT_TARGET_ASPECT = 456.0 / 256.0  # 1.78125


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)


def letterbox_to_aspect(color, depth, fx, fy, cx, cy, target_aspect=VIDBOT_TARGET_ASPECT):
    """Pad images horizontally to match VidBot's expected 16:9 aspect ratio.

    Returns padded color, padded depth, and adjusted (fx, fy, cx, cy).
    Depth is padded with 0 (no data).  Color is padded with 0 (black).
    """
    h, w = color.shape[:2]
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 0.01:
        return color, depth, fx, fy, cx, cy  # already close enough

    target_w = int(round(h * target_aspect))
    pad_total = target_w - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    color_pad = cv2.copyMakeBorder(
        color, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    depth_pad = cv2.copyMakeBorder(
        depth, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    cx_new = cx + pad_left
    return color_pad, depth_pad, fx, fy, cx_new, cy


def densify_depth(depth):
    """Fill zero-valued depth pixels with nearest valid neighbor's value."""
    mask = depth == 0
    if not mask.any():
        return depth.copy()
    _, indices = distance_transform_edt(mask, return_distances=True, return_indices=True)
    return depth[indices[0], indices[1]]


def nearest_valid_depth(depth_img, u, v, max_radius=50):
    """Find the nearest non-zero depth value around (u, v).

    Returns (depth_value, nu, nv) of the closest valid pixel,
    or (0, u, v) if nothing found within max_radius.
    """
    h, w = depth_img.shape[:2]
    d = depth_img[v, u] if 0 <= v < h and 0 <= u < w else 0
    if d > 0:
        return int(d), u, v
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue  # only check perimeter
                ny, nx = v + dy, u + dx
                if 0 <= ny < h and 0 <= nx < w and depth_img[ny, nx] > 0:
                    return int(depth_img[ny, nx]), nx, ny
    return 0, u, v


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


def get_grasp_R_base(T_base_cam):
    """Compute grasp orientation using runtime T_base_cam."""
    if not os.path.exists(GRASP_POSE_PATH):
        raise FileNotFoundError(f"Missing {GRASP_POSE_PATH}")

    T_grasp_cam = load_T_any(GRASP_POSE_PATH)
    T_grasp_base = T_base_cam @ T_grasp_cam

    R = T_grasp_base[:3, :3].copy()
    R = project_to_so3(R)

    if APPLY_Z_FLIP_FIX:
        R = R @ rot_y_pi()
        R = project_to_so3(R)

    T_grasp_base[:3, :3] = R
    return T_grasp_base, R


def build_pre_post_spot(pred_cam, loss, R_const, T_base_cam, ee_start):
    """Build pre/post trajectories using runtime transforms."""
    best_idx = int(np.argmin(loss))
    best_traj_cam = pred_cam[best_idx]

    best_traj_base = transform_points(T_base_cam, best_traj_cam)

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
    scale = get_scale_for_instruction(action) * 3
    scale = min(1 ,scale)
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


# ---------------------------------------------------------------------------
# ROS helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class VidBotSpotNode(Node):
    def __init__(self):
        super().__init__("vidbot_spot_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("base_frame", "spot/body")
        self.declare_parameter("camera_frame", "spot/hand_color_image_sensor")
        self.declare_parameter("ee_frame", "spot/arm_link_wr1")
        self.declare_parameter("arm_cmd_frame", "body")
        self.declare_parameter("color_topic", "/spot/camera/hand/image")
        self.declare_parameter("depth_topic", "/spot/depth_registered/hand/image")
        self.declare_parameter("camera_info_topic", "/spot/depth_registered/hand/camera_info")
        self.declare_parameter("use_graspnet", False)
        self.declare_parameter("post_skip", 8)
        self.declare_parameter("press_skip_k", 3)
        self.declare_parameter("scan_pose_x", 0.25)
        self.declare_parameter("scan_pose_y", 0.0)
        self.declare_parameter("scan_pose_z", 0.25)
        self.declare_parameter("scan_settle_time", 10.0)
        self.declare_parameter("prediction_z_offset", 0.00)  # metres, shift predictions up

        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.arm_cmd_frame = self.get_parameter("arm_cmd_frame").value
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        self.use_graspnet = self.get_parameter("use_graspnet").value
        self.post_skip = int(self.get_parameter("post_skip").value)
        self.press_skip_k = int(self.get_parameter("press_skip_k").value)
        self.scan_pose = np.array([
            self.get_parameter("scan_pose_x").value,
            self.get_parameter("scan_pose_y").value,
            self.get_parameter("scan_pose_z").value,
        ], dtype=float)
        self.scan_settle_time = self.get_parameter("scan_settle_time").value
        self.prediction_z_offset = self.get_parameter("prediction_z_offset").value

        # ── State ─────────────────────────────────────────────────────────
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.busy = False

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Image, color_topic, self._color_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(String, "/vidbot/trigger", self._trigger_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.arm_cmd_pub = self.create_publisher(PoseStamped, "/spot/arm_pose_commands", 10) #/spot/arm_pose_commands
        self.pre_traj_pub = self.create_publisher(Path, "/vidbot/pre_trajectory", 10)
        self.post_traj_pub = self.create_publisher(Path, "/vidbot/post_trajectory", 10)
        self.pre_poses_pub = self.create_publisher(PoseArray, "/vidbot/pre_poses", 10)
        self.post_poses_pub = self.create_publisher(PoseArray, "/vidbot/post_poses", 10)
        self.pre_axes_pub = self.create_publisher(MarkerArray, "/vidbot/pre_axes", 10)
        self.post_axes_pub = self.create_publisher(MarkerArray, "/vidbot/post_axes", 10)

        # ── Service clients ───────────────────────────────────────────────
        self.stow_client = self.create_client(Trigger, "/spot/arm_stow")
        self.open_gripper_client = self.create_client(Trigger, "/spot/open_gripper")
        self.close_gripper_client = self.create_client(Trigger, "/spot/close_gripper")

        self.get_logger().info(
            f"VidBot Spot node ready. Subscribed to {color_topic}, {depth_topic}. "
            "Waiting for /vidbot/trigger ..."
        )

    # ── TF helpers ────────────────────────────────────────────────────────

    def _lookup_tf(self, target_frame, source_frame):
        """Look up TF and return as 4x4 numpy matrix, or None on failure."""
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            T = np.eye(4)
            tr = t.transform.translation
            ro = t.transform.rotation
            T[:3, :3] = Rotation.from_quat([ro.x, ro.y, ro.z, ro.w]).as_matrix()
            T[0, 3] = tr.x
            T[1, 3] = tr.y
            T[2, 3] = tr.z
            return T
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed ({target_frame}->{source_frame}): {e}")
            return None

    def _get_T_base_cam(self):
        return self._lookup_tf(self.base_frame, self.camera_frame)

    def _get_ee_pose(self):
        return self._lookup_tf(self.base_frame, self.ee_frame)

    # ── Service call helpers ──────────────────────────────────────────────

    def _call_trigger_service(self, client, name, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"Service {name} not available")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"Service {name}: OK")
            return True
        msg = future.result().message if future.result() else "timeout"
        self.get_logger().error(f"Service {name} failed: {msg}")
        return False

    def _arm_stow(self):
        return self._call_trigger_service(self.stow_client, "/spot/arm_stow")

    def _open_gripper(self):
        return self._call_trigger_service(self.open_gripper_client, "/spot/open_gripper")

    def _close_gripper(self):
        return self._call_trigger_service(self.close_gripper_client, "/spot/close_gripper")

    # ── Arm command helper ────────────────────────────────────────────────

    def _send_arm_pose(self, T):
        """Publish a single PoseStamped to /spot/arm_pose_commands."""
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.arm_cmd_frame
        ps.pose = mat4_to_pose(T)
        self.arm_cmd_pub.publish(ps)

    def _send_scan_pose(self):
        """Send arm to the scan position with identity orientation."""
        T = np.eye(4)
        T[:3, 3] = self.scan_pose
        self._send_arm_pose(T)

    # ── Image callbacks ───────────────────────────────────────────────────

    def _color_cb(self, msg: Image):
        self.latest_color = imgmsg_to_cv2(msg)

    def _depth_cb(self, msg: Image):
        self.latest_depth = imgmsg_to_cv2(msg)

    def _camera_info_cb(self, msg: CameraInfo):
        if self.latest_camera_info is None:
            k = msg.k
            self.get_logger().info(
                f"Camera intrinsics: {msg.width}x{msg.height}, "
                f"fx={k[0]:.2f}, fy={k[4]:.2f}, cx={k[2]:.2f}, cy={k[5]:.2f}"
            )
        self.latest_camera_info = msg

    def _save_intrinsics(self, width, height, fx, fy, cx, cy):
        """Save letterbox-adjusted camera intrinsics to disk."""
        intrinsic = {
            "width": int(width),
            "height": int(height),
            "intrinsic_matrix": [
                fx, 0.0, 0.0,
                0.0, fy, 0.0,
                cx, cy, 1.0,
            ],
        }
        ensure_dirs()
        with open(INTRINSIC_PATH, "w") as f:
            json.dump(intrinsic, f, indent=4)
        self.get_logger().info(
            f"Saved intrinsics: {width}x{height}, "
            f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
        )
        return True

    # ── Trigger ───────────────────────────────────────────────────────────

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

        self.busy = True
        thread = threading.Thread(
            target=self._run_inference, args=(obj, instruction), daemon=True
        )
        thread.start()

    # ── Main inference pipeline ───────────────────────────────────────────

    def _run_inference(self, obj, instruction):
        try:
            self.get_logger().info(
                f'Inference started: object="{obj}", instruction="{instruction}"'
            )

            # ── Step 2: Capture ───────────────────────────────────────
            if self.latest_color is None or self.latest_depth is None:
                raise RuntimeError("No images received from hand camera")
            if self.latest_camera_info is None:
                raise RuntimeError("No CameraInfo received yet")

            ensure_dirs()
            k = self.latest_camera_info.k
            raw_color = self.latest_color.copy()
            raw_depth = self.latest_depth.copy()
            raw_fx, raw_fy, raw_cx, raw_cy = k[0], k[4], k[2], k[5]
            h_raw, w_raw = raw_color.shape[:2]

            color_pad, depth_pad, pad_fx, pad_fy, pad_cx, pad_cy = letterbox_to_aspect(
                raw_color, raw_depth, raw_fx, raw_fy, raw_cx, raw_cy,
            )
            h_pad, w_pad = color_pad.shape[:2]

            T_base_cam = self._get_T_base_cam()
            if T_base_cam is None:
                raise RuntimeError(
                    f"Cannot look up TF: {self.base_frame} -> {self.camera_frame}"
                )

            # ── Step 2b: Save letterboxed images + intrinsics ────────────
            cv2.imwrite(COLOR_PATH, color_pad)
            cv2.imwrite(DEPTH_PATH, depth_pad)
            self._save_intrinsics(w_pad, h_pad, pad_fx, pad_fy, pad_cx, pad_cy)
            self.get_logger().info(
                f"Step 2: Captured RGB-D, letterboxed {w_raw}x{h_raw} -> {w_pad}x{h_pad}"
            )

            # ── Step 3: Run inference ─────────────────────────────────────
            run_infer_affordance(
                obj, instruction,
                use_graspnet=self.use_graspnet,
                logger=self.get_logger(),
            )

            # ── Step 4: Load predictions ──────────────────────────────────
            pred_npz = find_latest_prediction_npz()
            pred, loss = load_pred_and_loss(pred_npz)
            z = np.load(pred_npz, allow_pickle=True)

            # ── DEBUG: 3-panel contact visualization ──────────────────────
            try:
                debug_dir = os.path.join(DATASET_DIR, "debug")
                os.makedirs(debug_dir, exist_ok=True)

                # Panel 1: Object patch (256x256) with heatmap
                obj_color = z["object_color_vis"][0]
                obj_color = (obj_color.transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1]
                contact_scores = z["contact_scores"][0]
                cs_norm = (contact_scores - contact_scores.min()) / (contact_scores.max() - contact_scores.min() + 1e-8)
                cs_jet = cv2.applyColorMap((cs_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                patch_vis = cv2.addWeighted(obj_color, 0.5, cs_jet, 0.5, 0)
                cp_patch = z["contact_pix_patch"][0, 0]
                cpu, cpv = int(cp_patch[0]), int(cp_patch[1])
                cv2.circle(patch_vis, (cpu, cpv), 6, (0, 255, 0), 2)
                cv2.putText(patch_vis, f"patch ({cpu},{cpv})", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Mapped contact coords (reuse from depth correction)
                cp_rs = z["contact_pix"][0, 0]
                w_rs2 = z["color"].shape[3]
                crop_off2 = (456 - w_rs2) / 2.0
                iscale2 = h_pad / 256.0
                cu = int((cp_rs[0] + crop_off2) * iscale2)
                cv_pt = int(cp_rs[1] * iscale2)
                start_pos = z["start_pos"][0]

                # Panel 2: Full letterboxed color with bbox + contact
                color_vis = color_pad.copy()
                bbox_raw = z["bbox_raw_all"][0, 0].astype(int) if "bbox_raw_all" in z.files else None
                if bbox_raw is not None:
                    cv2.rectangle(color_vis, (bbox_raw[0], bbox_raw[1]), (bbox_raw[2], bbox_raw[3]),
                                  (255, 255, 0), 1)
                cv2.circle(color_vis, (cu, cv_pt), 10, (0, 255, 0), 2)
                cv2.putText(color_vis, f"contact ({cu},{cv_pt})", (cu + 12, cv_pt - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_vis, f"3D: ({start_pos[0]:.3f},{start_pos[1]:.3f},{start_pos[2]:.3f})",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Panel 3: Depth jet with contact
                depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
                depth_f = depth_raw.astype(np.float32)
                depth_f[depth_f == 0] = np.nan
                d_min, d_max = np.nanmin(depth_f), np.nanmax(depth_f)
                depth_norm = (depth_f - d_min) / (d_max - d_min + 1e-8)
                depth_norm = np.nan_to_num(depth_norm, nan=0.0)
                depth_jet = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                depth_jet[depth_raw == 0] = 0
                cv2.circle(depth_jet, (cu, cv_pt), 10, (0, 255, 0), 2)
                d_at, d_nu, d_nv = nearest_valid_depth(depth_raw, cu, cv_pt)
                if d_nu != cu or d_nv != cv_pt:
                    cv2.circle(depth_jet, (d_nu, d_nv), 6, (0, 200, 255), 2)
                    cv2.line(depth_jet, (cu, cv_pt), (d_nu, d_nv), (0, 200, 255), 1)
                    cv2.putText(depth_jet, f"nearest: {d_at} ({d_at/1000:.3f}m) @({d_nu},{d_nv})",
                                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                else:
                    cv2.putText(depth_jet, f"depth@contact: {d_at} ({d_at/1000:.3f}m)",
                                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Assemble
                ph = color_vis.shape[0]
                patch_resized = cv2.resize(patch_vis, (ph, ph))
                combined = np.hstack([patch_resized, color_vis, depth_jet])
                debug_path = os.path.join(debug_dir, "contact_pipeline.png")
                cv2.imwrite(debug_path, combined)

                self.get_logger().info(
                    f"DEBUG: patch=({cpu},{cpv}), full=({cu},{cv_pt}), "
                    f"depth={d_at}, saved {debug_path}"
                )
            except Exception as e:
                self.get_logger().warn(f"DEBUG vis failed: {e}")

            # ── Step 6: Get EE pose + build trajectories ──────────────────
            T_ee = self._get_ee_pose()
            if T_ee is None:
                raise RuntimeError(
                    f"Cannot look up TF: {self.base_frame} -> {self.ee_frame}"
                )
            ee_start = T_ee[:3, 3]

            # Orientation: use GraspNet if available, else current EE rotation
            if self.use_graspnet:
                _, R_const = get_grasp_R_base(T_base_cam)
                self.get_logger().info("Using GraspNet orientation")
            else:
                R_const = T_ee[:3, :3]
                self.get_logger().info("Using current EE orientation from TF")

            best_idx, best_loss, pre_T, post_T = build_pre_post_spot(
                pred, loss, R_const, T_base_cam, ee_start
            )

            # Apply Z offset to compensate depth bias
            if self.prediction_z_offset != 0.0:
                pre_T[:, 2, 3] += self.prediction_z_offset
                post_T[:, 2, 3] += self.prediction_z_offset
                self.get_logger().info(
                    f"Applied prediction_z_offset={self.prediction_z_offset:.3f}m"
                )

            # ── Step 7: Thin post trajectory ──────────────────────────────
            is_press = is_press_action(instruction)
            effective_skip = self.post_skip * self.press_skip_k if is_press else self.post_skip

            if effective_skip > 0:
                step = effective_skip + 1
                indices = list(range(0, len(post_T), step))
                if indices[-1] != len(post_T) - 1:
                    indices.append(len(post_T) - 1)
                self.get_logger().info(
                    f"Post thinned: {len(post_T)} -> {len(indices)} waypoints "
                    f"(skip={effective_skip}, press={is_press})"
                )
                post_T = post_T[indices]

            # ── Step 8: Publish visualization ─────────────────────────────
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

            self.get_logger().info("Inference complete! Manager will handle execution.")

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
        finally:
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = VidBotSpotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
