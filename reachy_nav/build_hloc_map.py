#!/usr/bin/env python3
"""
Offline map-building script for hloc visual localization.

Creates an hloc reference database from the existing RGB images + known camera
poses captured during TSDF mapping.  The output is a triangulated SfM model
that can be used by the online localization node.

Steps:
  1. Build a COLMAP reconstruction from known per-frame poses & intrinsics.
  2. Extract SuperPoint local features.
  3. Extract NetVLAD global descriptors for retrieval.
  4. Generate image pairs via NetVLAD retrieval.
  5. Match features with LightGlue.
  6. Triangulate 3D points using known poses + matches.

Usage:
    python3 build_hloc_map.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pycolmap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = THIS_DIR / "data_dso" / "2026_02_16-12_02_34-default_experiment"
RGB_DIR = DATA_DIR / "rgb"
POSE_DIR = DATA_DIR / "camera_poses"
INTRINSIC_DIR = DATA_DIR / "camera_intrinsics"

HLOC_DIR = DATA_DIR / "hloc"
REF_MODEL_DIR = HLOC_DIR / "ref_model"
SFM_DIR = HLOC_DIR / "sfm"

# Add hloc to path
sys.path.insert(0, str(THIS_DIR / "third_party" / "hloc"))

from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
from tsdf_generator import collect_step_files

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_RETRIEVAL_PAIRS = 20


def build_colmap_model() -> pycolmap.Reconstruction:
    """Create a COLMAP reconstruction from known per-frame poses & intrinsics.

    Each frame may have different intrinsics, so we create one camera per image.
    """
    rgb_files = collect_step_files(str(RGB_DIR), "rgb", ".png")
    pose_files = collect_step_files(str(POSE_DIR), "camera_pose", ".npy")
    intrinsic_files = collect_step_files(str(INTRINSIC_DIR), "camera_intrinsics", ".npy")

    common_steps = sorted(set(rgb_files) & set(pose_files) & set(intrinsic_files))
    print(f"Common steps (RGB + pose + intrinsics): {len(common_steps)}")

    if not common_steps:
        raise RuntimeError("No common steps found.")

    rec = pycolmap.Reconstruction()

    # First pass: add all cameras and build the rig
    rig = pycolmap.Rig(rig_id=1)
    for idx, step in enumerate(common_steps):
        camera_id = idx + 1

        K = np.load(intrinsic_files[step])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        cam = pycolmap.Camera(
            model=pycolmap.CameraModelId.PINHOLE,
            width=1280,
            height=720,
            params=[fx, fy, cx, cy],
            camera_id=camera_id,
        )
        rec.add_camera(cam)

        sensor_id = cam.sensor_id
        if idx == 0:
            rig.add_ref_sensor(sensor_id)
        else:
            rig.add_sensor(sensor_id, pycolmap.Rigid3d())

    # Rig must be added before any frames that reference it
    rec.add_rig(rig)

    # Second pass: add frames and images with poses
    for idx, step in enumerate(common_steps):
        image_id = idx + 1
        camera_id = idx + 1
        frame_id = idx + 1

        # Load pose: T_world_cam (world <- camera)
        T_world_cam = np.load(pose_files[step])
        T_cam_world = np.linalg.inv(T_world_cam)

        rot = pycolmap.Rotation3d(T_cam_world[:3, :3])
        trans = T_cam_world[:3, 3]
        rigid = pycolmap.Rigid3d(rot, trans)

        img_name = os.path.basename(rgb_files[step])

        img = pycolmap.Image(
            name=img_name,
            camera_id=camera_id,
            image_id=image_id,
            frame_id=frame_id,
        )

        frame = pycolmap.Frame(frame_id=frame_id, rig_id=1)
        frame.rig_from_world = rigid
        frame.add_data_id(img.data_id)

        rec.add_frame(frame)
        rec.register_frame(frame_id)
        rec.add_image(img)

    print(f"COLMAP model: {rec.num_cameras()} cameras, "
          f"{rec.num_reg_images()} registered images")

    # Write the reference model to disk
    REF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rec.write(REF_MODEL_DIR)
    print(f"Saved reference model to {REF_MODEL_DIR}")

    return rec


def main():
    HLOC_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build COLMAP model from known poses
    # ------------------------------------------------------------------
    print("\n=== Step 1: Building COLMAP model from known poses ===")
    build_colmap_model()

    # ------------------------------------------------------------------
    # Step 2: Extract SuperPoint features
    # ------------------------------------------------------------------
    print("\n=== Step 2: Extracting SuperPoint features ===")
    feature_path = extract_features.main(
        extract_features.confs["superpoint_aachen"],
        image_dir=RGB_DIR,
        export_dir=HLOC_DIR,
    )
    print(f"Features: {feature_path}")

    # ------------------------------------------------------------------
    # Step 3: Extract NetVLAD global descriptors
    # ------------------------------------------------------------------
    print("\n=== Step 3: Extracting NetVLAD global descriptors ===")
    global_desc_path = extract_features.main(
        extract_features.confs["netvlad"],
        image_dir=RGB_DIR,
        export_dir=HLOC_DIR,
    )
    print(f"Global descriptors: {global_desc_path}")

    # ------------------------------------------------------------------
    # Step 4: Find image pairs via retrieval
    # ------------------------------------------------------------------
    print("\n=== Step 4: Finding image pairs via retrieval ===")
    pairs_path = HLOC_DIR / "pairs-netvlad.txt"
    pairs_from_retrieval.main(
        descriptors=global_desc_path,
        output=pairs_path,
        num_matched=NUM_RETRIEVAL_PAIRS,
    )
    print(f"Pairs: {pairs_path}")

    # ------------------------------------------------------------------
    # Step 5: Match features with LightGlue
    # ------------------------------------------------------------------
    print("\n=== Step 5: Matching features with LightGlue ===")
    feat_conf = extract_features.confs["superpoint_aachen"]
    match_path = match_features.main(
        match_features.confs["superpoint+lightglue"],
        pairs=pairs_path,
        features=feat_conf["output"],
        export_dir=HLOC_DIR,
    )
    print(f"Matches: {match_path}")

    # ------------------------------------------------------------------
    # Step 6: Triangulate 3D points using known poses + matches
    # ------------------------------------------------------------------
    print("\n=== Step 6: Triangulating 3D points ===")
    reconstruction = triangulation.main(
        sfm_dir=SFM_DIR,
        reference_model=REF_MODEL_DIR,
        image_dir=RGB_DIR,
        pairs=pairs_path,
        features=feature_path,
        matches=match_path,
        verbose=True,
    )
    print(f"\nTriangulation complete!")
    print(reconstruction.summary())

    print(f"\n=== Done! hloc map saved to {HLOC_DIR} ===")


if __name__ == "__main__":
    main()
