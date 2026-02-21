#!/usr/bin/env python3
"""Export posed RGBD from sfm_outputs.

Reads a COLMAP-style reconstruction (rec/), images/, and depths/ from an
sfm_outputs directory and writes per-frame files ready for downstream use.

Only depends on numpy + stdlib (no pycolmap, no hloc import).
COLMAP binary parsing copied verbatim from hloc/utils/read_write_model.py.

Output structure (flat, one file per registered image):
    <out_dir>/
        frame_000001.jpg          # RGB (copied or symlinked)
        frame_000001.npy          # depth  (float32, meters)
        frame_000001.txt          # 4x4 cam_from_world matrix
        intrinsics.txt            # 3x3 K matrix (shared across all frames)
        poses.txt                 # single file with all poses

Usage:
    python scripts/export_posed_rgbd.py <sfm_outputs_dir> [--out <output_dir>] [--symlink]
"""

import argparse
import collections
import shutil
import struct
from pathlib import Path

import numpy as np

# ── COLMAP binary parsing (copied from hloc/utils/read_write_model.py) ──────

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
_BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {cm.model_id: cm for cm in CAMERA_MODELS}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name,
                width=width, height=height, params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid, num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            ) if num_points2D > 0 else np.zeros((0, 2))
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = _BaseImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids,
            )
        assert len(images) == num_reg_images
    return images


def qvec2rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) → 3x3 rotation matrix.

    Copied verbatim from hloc/utils/read_write_model.py.
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def cam_from_world_4x4(image):
    """Image namedtuple → 4x4 cam_from_world matrix."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def intrinsics_matrix(camera):
    """Camera namedtuple → 3x3 K matrix."""
    p = camera.params
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sfm_dir", type=Path, help="Path to sfm_outputs directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: <sfm_dir>/posed_rgbd)")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink images instead of copying")
    args = parser.parse_args()

    sfm_dir = args.sfm_dir.resolve()
    rec_dir = sfm_dir / "rec"
    images_dir = sfm_dir / "images"
    depths_dir = sfm_dir / "depths"
    out_dir = (args.out or sfm_dir / "posed_rgbd").resolve()

    assert rec_dir.exists(), f"Reconstruction not found: {rec_dir}"
    assert images_dir.exists(), f"Images not found: {images_dir}"
    assert depths_dir.exists(), f"Depths not found: {depths_dir}"

    # Load reconstruction
    cameras = read_cameras_binary(str(rec_dir / "cameras.bin"))
    images = read_images_binary(str(rec_dir / "images.bin"))
    print(f"Loaded reconstruction: {len(images)} registered images, "
          f"{len(cameras)} camera(s)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write shared intrinsics (first camera)
    cam = next(iter(cameras.values()))
    K = intrinsics_matrix(cam)
    np.savetxt(out_dir / "intrinsics.txt", K, fmt="%.8f")
    print(f"Camera: {cam.model} {cam.width}x{cam.height}, fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    # Collect all poses for the summary file
    poses_lines = []
    exported = 0
    skipped_depth = 0
    skipped_image = 0

    for img_id in sorted(images):
        img = images[img_id]
        name = img.name
        stem = Path(name).stem

        src_img = images_dir / name
        if not src_img.exists():
            skipped_image += 1
            continue

        src_depth = depths_dir / f"{stem}.npy"
        if not src_depth.exists():
            skipped_depth += 1
            continue

        # 4x4 cam_from_world
        T = cam_from_world_4x4(img)

        # Write outputs
        dst_img = out_dir / name
        if args.symlink:
            if not dst_img.exists():
                dst_img.symlink_to(src_img)
        else:
            shutil.copy2(src_img, dst_img)

        shutil.copy2(src_depth, out_dir / f"{stem}.npy")
        np.savetxt(out_dir / f"{stem}.txt", T, fmt="%.8f")

        poses_lines.append(f"{name} " + " ".join(f"{v:.8f}" for v in T.flatten()))
        exported += 1

    # Write combined poses file
    with open(out_dir / "poses.txt", "w") as f:
        f.write(f"# Posed RGBD export from {sfm_dir.name}\n")
        f.write(f"# Format: image_name <16 floats: 4x4 cam_from_world row-major>\n")
        f.write(f"# Camera: {cam.model} {cam.width}x{cam.height} "
                f"fx={K[0,0]:.4f} fy={K[1,1]:.4f} cx={K[0,2]:.4f} cy={K[1,2]:.4f}\n")
        for line in poses_lines:
            f.write(line + "\n")

    print(f"\nExported {exported} posed RGBD frames to {out_dir}")
    if skipped_depth:
        print(f"  Skipped {skipped_depth} frames (missing depth)")
    if skipped_image:
        print(f"  Skipped {skipped_image} frames (missing image)")


if __name__ == "__main__":
    main()
