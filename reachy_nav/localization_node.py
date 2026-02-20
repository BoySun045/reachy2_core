#!/usr/bin/env python3
"""
ROS2 node for real-time visual localization using hloc.

Uses the pre-built SfM map (from build_hloc_map.py) to localize the robot
from live camera images via SuperPoint + NetVLAD + LightGlue + PnP.

Subscriptions:
    /camera/color/image_raw   — RGB image (sensor_msgs/Image)
    /camera/color/camera_info — live camera intrinsics (sensor_msgs/CameraInfo)

Publications:
    ~/robot_pose — localized robot base pose in z-up frame (PoseStamped)

Usage:
    python3 localization_node.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import h5py
import torch
import pycolmap

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = THIS_DIR / "data_dso" / "2026_02_16-12_02_34-default_experiment"
HLOC_DIR = DATA_DIR / "hloc"
SFM_DIR = HLOC_DIR / "sfm"
EXTRINSIC_TXT = DATA_DIR / "camera_extrinsics.txt"

# Add hloc to path
sys.path.insert(0, str(THIS_DIR / "third_party" / "hloc"))

from hloc.utils.base_model import dynamic_load
from hloc import extractors, matchers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRAME_ID = "map"
TOP_K_RETRIEVAL = 10
RANSAC_THRESH = 12
LOCALIZE_RATE_HZ = 2.0  # max localization frequency


def to_z_up(pts: np.ndarray) -> np.ndarray:
    """(x, y, z) native -> (x, z, -y) z-up."""
    out = np.array(pts, dtype=np.float64)
    if out.ndim == 1:
        y = out[1].copy()
        out[1] = out[2]
        out[2] = -y
    else:
        y = out[:, 1].copy()
        out[:, 1] = out[:, 2]
        out[:, 2] = -y
    return out


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Yaw (radians, around Z-up) -> quaternion [x, y, z, w]."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler("z", yaw).as_quat()


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization")

        self.get_logger().info("Loading SfM model...")
        self._reconstruction = pycolmap.Reconstruction(str(SFM_DIR))
        self.get_logger().info(
            f"SfM model: {self._reconstruction.num_reg_images()} images, "
            f"{self._reconstruction.num_points3D()} 3D points"
        )

        # Build name -> image_id map
        self._db_name_to_id = {
            img.name: i for i, img in self._reconstruction.images.items()
        }

        # Load camera extrinsics T_base_cam
        self._T_base_cam = np.loadtxt(str(EXTRINSIC_TXT))
        assert self._T_base_cam.shape == (4, 4)
        self.get_logger().info("Loaded T_base_cam extrinsics")

        # Load feature + global descriptor files
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self._device}")

        self._load_models()
        self._load_reference_data()

        # Camera intrinsics from CameraInfo
        self._camera_K = None  # will be set from CameraInfo
        self._img_width = None
        self._img_height = None

        # Rate limiting
        self._min_interval = 1.0 / LOCALIZE_RATE_HZ
        self._last_localize_time = 0.0

        # Publisher
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, "~/robot_pose", latched_qos
        )

        # Subscribers
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self._on_camera_info,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._on_image,
            sensor_qos,
        )

        self.get_logger().info("Localization node ready.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self):
        """Load SuperPoint, NetVLAD, and LightGlue models."""
        # SuperPoint
        sp_conf = {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        }
        SuperPoint = dynamic_load(extractors, "superpoint")
        self._superpoint = SuperPoint(sp_conf).eval().to(self._device)

        # NetVLAD
        netvlad_conf = {"name": "netvlad"}
        NetVLAD = dynamic_load(extractors, "netvlad")
        self._netvlad = NetVLAD(netvlad_conf).eval().to(self._device)

        # LightGlue
        lg_conf = {"name": "lightglue", "features": "superpoint"}
        LightGlueModel = dynamic_load(matchers, "lightglue")
        self._lightglue = LightGlueModel(lg_conf).eval().to(self._device)

        self.get_logger().info("Loaded SuperPoint, NetVLAD, LightGlue models")

    def _load_reference_data(self):
        """Load pre-computed reference features and global descriptors."""
        # Find feature and global descriptor files
        feat_files = list(HLOC_DIR.glob("feats-superpoint*.h5"))
        global_files = list(HLOC_DIR.glob("global-feats-netvlad*.h5"))

        if not feat_files:
            raise FileNotFoundError(f"No SuperPoint feature file in {HLOC_DIR}")
        if not global_files:
            raise FileNotFoundError(f"No NetVLAD descriptor file in {HLOC_DIR}")

        self._features_path = feat_files[0]
        self._global_desc_path = global_files[0]

        # Load all reference global descriptors into memory for retrieval
        self._ref_names = []
        self._ref_global_descs = []
        with h5py.File(str(self._global_desc_path), "r") as f:
            for name in f:
                if name in self._db_name_to_id:
                    self._ref_names.append(name)
                    self._ref_global_descs.append(
                        f[name]["global_descriptor"].__array__()
                    )

        self._ref_global_descs = np.stack(self._ref_global_descs)  # (N, D)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self._ref_global_descs, axis=1, keepdims=True)
        self._ref_global_descs = self._ref_global_descs / np.clip(norms, 1e-6, None)

        # Load all reference local features into memory
        self._ref_keypoints = {}
        self._ref_descriptors = {}
        with h5py.File(str(self._features_path), "r") as f:
            for name in self._ref_names:
                self._ref_keypoints[name] = f[name]["keypoints"].__array__()
                self._ref_descriptors[name] = f[name]["descriptors"].__array__()

        self.get_logger().info(
            f"Loaded {len(self._ref_names)} reference images "
            f"(features + global descriptors)"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo):
        """Store live camera intrinsics."""
        K = np.array(msg.k).reshape(3, 3)
        self._camera_K = K
        self._img_width = msg.width
        self._img_height = msg.height

    def _on_image(self, msg: Image):
        """Localize from a camera image."""
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_localize_time < self._min_interval:
            return

        if self._camera_K is None:
            self.get_logger().warn("No CameraInfo received yet, skipping.", throttle_duration_sec=5.0)
            return

        self._last_localize_time = now

        # Decode image
        if msg.encoding in ("rgb8", "bgr8"):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return

        # Convert to grayscale for SuperPoint
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        try:
            pose = self._localize(gray, img)
        except Exception as e:
            self.get_logger().warn(f"Localization failed: {e}", throttle_duration_sec=2.0)
            return

        if pose is not None:
            self._publish_pose(pose, msg.header.stamp)

    # ------------------------------------------------------------------
    # Core localization pipeline
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _localize(self, gray: np.ndarray, color: np.ndarray):
        """Run the full localization pipeline on a query image.

        Returns T_world_base in z-up frame as (position_xyz, yaw) or None.
        """
        # 1. Preprocess: resize to match training resolution (max 1024)
        h, w = gray.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            gray_resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            gray_resized = gray

        # 2. Extract SuperPoint features
        img_tensor = torch.from_numpy(gray_resized)[None, None].to(self._device)
        sp_pred = self._superpoint({"image": img_tensor})
        sp_pred = {k: v[0].cpu().numpy() for k, v in sp_pred.items()}

        keypoints_q = sp_pred["keypoints"]  # (N, 2)
        descriptors_q = sp_pred["descriptors"]  # (D, N) or (N, D)

        if keypoints_q.shape[0] == 0:
            return None

        # Scale keypoints back to original resolution
        if scale != 1.0:
            keypoints_q = keypoints_q / scale

        # 3. Extract NetVLAD global descriptor
        color_resized = cv2.resize(color, (gray_resized.shape[1], gray_resized.shape[0]),
                                   interpolation=cv2.INTER_AREA)
        color_tensor = torch.from_numpy(
            color_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        )[None].to(self._device)
        netvlad_pred = self._netvlad({"image": color_tensor})
        global_desc_q = netvlad_pred["global_descriptor"][0].cpu().numpy()

        # 4. Retrieval: find top-k reference images
        global_desc_q_norm = global_desc_q / max(np.linalg.norm(global_desc_q), 1e-6)
        similarities = self._ref_global_descs @ global_desc_q_norm
        top_k_idx = np.argsort(-similarities)[:TOP_K_RETRIEVAL]
        retrieved_names = [self._ref_names[i] for i in top_k_idx]

        # 5. Match against retrieved references using LightGlue
        kp_idx_to_3D = defaultdict(list)
        num_matches_total = 0

        # Ensure descriptors_q is (D, N) for LightGlue
        if descriptors_q.ndim == 2 and descriptors_q.shape[0] == keypoints_q.shape[0]:
            descriptors_q_DN = descriptors_q.T  # (N, D) -> (D, N)
        else:
            descriptors_q_DN = descriptors_q

        for ref_name in retrieved_names:
            db_id = self._db_name_to_id.get(ref_name)
            if db_id is None:
                continue

            db_image = self._reconstruction.images[db_id]
            if db_image.num_points3D == 0:
                continue

            kps_ref = self._ref_keypoints[ref_name]
            desc_ref = self._ref_descriptors[ref_name]

            # Ensure desc_ref is (D, N) for LightGlue
            if desc_ref.ndim == 2 and desc_ref.shape[0] == kps_ref.shape[0]:
                desc_ref_DN = desc_ref.T
            else:
                desc_ref_DN = desc_ref

            # LightGlue expects descriptors as (1, N, D)
            match_data = {
                "image0": torch.zeros(1),  # placeholder
                "image1": torch.zeros(1),
                "keypoints0": torch.from_numpy(keypoints_q).float()[None].to(self._device),
                "keypoints1": torch.from_numpy(kps_ref).float()[None].to(self._device),
                "descriptors0": torch.from_numpy(descriptors_q_DN).float()[None].to(self._device),
                "descriptors1": torch.from_numpy(desc_ref_DN).float()[None].to(self._device),
            }
            match_pred = self._lightglue(match_data)

            # Parse matches
            matches0 = match_pred["matches0"][0].cpu().numpy()  # (N_q,) index into ref or -1
            valid = matches0 != -1
            q_idxs = np.where(valid)[0]
            r_idxs = matches0[valid]

            if len(q_idxs) == 0:
                continue

            # Get 3D point IDs from the database image
            points3D_ids = np.array(
                [p.point3D_id if p.has_point3D() else -1 for p in db_image.points2D]
            )

            for q_idx, r_idx in zip(q_idxs, r_idxs):
                if r_idx < len(points3D_ids) and points3D_ids[r_idx] != -1:
                    p3d_id = points3D_ids[r_idx]
                    if p3d_id not in kp_idx_to_3D[q_idx]:
                        kp_idx_to_3D[q_idx].append(p3d_id)
                        num_matches_total += 1

        if num_matches_total < 10:
            self.get_logger().info(f"Too few 2D-3D matches ({num_matches_total})")
            return None

        # 6. PnP + RANSAC
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]

        points2D = keypoints_q[mkp_idxs] + 0.5  # COLMAP convention
        points3D = np.array([
            self._reconstruction.points3D[pid].xyz for pid in mp3d_ids
        ])

        # Create query camera with LIVE intrinsics
        K = self._camera_K
        query_cam = pycolmap.Camera(
            model=pycolmap.CameraModelId.PINHOLE,
            width=self._img_width,
            height=self._img_height,
            params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        )

        ret = pycolmap.estimate_and_refine_absolute_pose(
            points2D,
            points3D,
            query_cam,
            estimation_options={"ransac": {"max_error": RANSAC_THRESH}},
            refinement_options={},
        )

        if ret is None:
            self.get_logger().info("PnP failed")
            return None

        self.get_logger().info(
            f"Localized: {ret['num_inliers']} inliers / {len(points2D)} matches"
        )

        # ret["cam_from_world"] is a Rigid3d
        T_cam_world = ret["cam_from_world"].matrix()  # 3x4 or 4x4
        T_cam_world_full = np.eye(4)
        T_cam_world_full[:T_cam_world.shape[0], :T_cam_world.shape[1]] = T_cam_world
        T_wc = np.linalg.inv(T_cam_world_full)
        self.get_logger().info(
            f"T_world_cam (native): pos=[{T_wc[0,3]:.3f}, {T_wc[1,3]:.3f}, {T_wc[2,3]:.3f}]"
        )
        if T_cam_world.shape == (3, 4):
            T_cam_world_4x4 = np.eye(4)
            T_cam_world_4x4[:3, :] = T_cam_world
        else:
            T_cam_world_4x4 = T_cam_world

        T_world_cam = np.linalg.inv(T_cam_world_4x4)

        # 7. Apply extrinsics: T_world_base = T_world_cam @ inv(T_base_cam)
        T_cam_base = self._T_base_cam  # T_base_cam = base <- cam
        T_base_cam_inv = np.linalg.inv(T_cam_base)
        T_world_base = T_world_cam @ T_base_cam_inv
        self.get_logger().info(
            f"T_world_base (native): pos=[{T_world_base[0,3]:.3f}, {T_world_base[1,3]:.3f}, {T_world_base[2,3]:.3f}]"
        )

        # 8. Convert to z-up frame
        pos_native = T_world_base[:3, 3]
        pos_zup = to_z_up(pos_native)

        # Extract yaw from the rotation in z-up frame
        R_native = T_world_base[:3, :3]
        # Forward direction (X-axis of the base) in native frame
        fwd_native = R_native[:, 0]
        fwd_zup = to_z_up(fwd_native)
        yaw = np.arctan2(fwd_zup[1], fwd_zup[0])

        self.get_logger().info(
            f"Final (z-up): pos=[{pos_zup[0]:.3f}, {pos_zup[1]:.3f}, {pos_zup[2]:.3f}] yaw={np.degrees(yaw):.1f} deg"
        )

        return pos_zup, yaw

    # ------------------------------------------------------------------
    def _publish_pose(self, pose_data, stamp):
        pos_zup, yaw = pose_data
        quat = yaw_to_quat(yaw)

        msg = PoseStamped()
        msg.header.frame_id = FRAME_ID
        msg.header.stamp = stamp
        msg.pose.position.x = float(pos_zup[0])
        msg.pose.position.y = float(pos_zup[1])
        msg.pose.position.z = float(pos_zup[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self._pose_pub.publish(msg)
        self.get_logger().info(
            f"Published pose: [{pos_zup[0]:.3f}, {pos_zup[1]:.3f}, {pos_zup[2]:.3f}] "
            f"yaw={np.degrees(yaw):.1f} deg"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
