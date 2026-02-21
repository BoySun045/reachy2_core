import os
import sys
import re
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm

# ----------------------------
# User config
# ----------------------------
DEPTH_SCALE = 1.0        # depth .npy is already in meters
DEPTH_TRUNC = 3

VOXEL_LENGTH = 0.02
SDF_TRUNC = 0.05

STAT_NB_NEIGHBORS = 20
STAT_STD_RATIO = 2.0

MAX_FRAMES = 2000


# ----------------------------
# Helpers
# ----------------------------
def extract_frame_number(filename: str) -> int | None:
    """Extract frame number from filenames like 'frame_000001.jpg'."""
    m = re.search(r"frame_(\d+)", filename)
    return int(m.group(1)) if m else None


def collect_frame_files(folder: str, ext: str) -> dict[int, str]:
    """Return {frame_number: filepath} for files matching frame_NNNNNN.ext."""
    result = {}
    for path in glob.glob(os.path.join(folder, f"frame_*{ext}")):
        fnum = extract_frame_number(os.path.basename(path))
        if fnum is not None:
            result[fnum] = path
    return result


def check_image_sizes(color_img, depth_arr, width, height, rgb_path):
    c = np.asarray(color_img)
    if c.shape[1] != width or c.shape[0] != height:
        raise ValueError(
            f"RGB image size {c.shape[1]}x{c.shape[0]} does not match intrinsics "
            f"{width}x{height}. File: {rgb_path}"
        )
    if depth_arr.shape[1] != width or depth_arr.shape[0] != height:
        raise ValueError(
            f"Depth size {depth_arr.shape[1]}x{depth_arr.shape[0]} does not match intrinsics "
            f"{width}x{height}."
        )


def to_z_up(pts: np.ndarray) -> np.ndarray:
    """Transform from native (XZ ground, Y-down) to (XY ground, Z-up).

    Mapping: (x, y, z) -> (x, z, -y)
    """
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


def make_trajectory_line(points_xyz: np.ndarray) -> o3d.geometry.LineSet:
    if len(points_xyz) < 2:
        return o3d.geometry.LineSet()

    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_xyz),
        lines=o3d.utility.Vector2iVector(lines),
    )


# ----------------------------
# Main
# ----------------------------
def run_tsdf_fusion(data_dir: str):
    # Load shared intrinsics (3x3 text file)
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")
    K = np.loadtxt(intrinsics_path)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics.txt must be 3x3, got {K.shape}")

    FX, FY = float(K[0, 0]), float(K[1, 1])
    CX, CY = float(K[0, 2]), float(K[1, 2])

    # Collect per-frame files (all in the same flat folder)
    rgb_files = collect_frame_files(data_dir, ".jpg")
    depth_files = collect_frame_files(data_dir, ".npy")
    pose_files = collect_frame_files(data_dir, ".txt")
    # Remove intrinsics.txt and poses.txt from pose_files if they snuck in
    pose_files = {k: v for k, v in pose_files.items()
                  if os.path.basename(v).startswith("frame_")}

    common_frames = sorted(set(rgb_files) & set(depth_files) & set(pose_files))

    print(f"RGB frames:   {len(rgb_files)}")
    print(f"Depth frames: {len(depth_files)}")
    print(f"Pose frames:  {len(pose_files)}")
    print(f"Common frames: {len(common_frames)}")

    if not common_frames:
        raise RuntimeError("No common frames found across rgb, depth, and poses.")

    if len(common_frames) > MAX_FRAMES:
        common_frames = common_frames[:MAX_FRAMES]

    # Get image dimensions from first frame
    first_rgb = o3d.io.read_image(rgb_files[common_frames[0]])
    HEIGHT, WIDTH = np.asarray(first_rgb).shape[:2]

    print(f"Intrinsics: {WIDTH}x{HEIGHT}, fx={FX:.2f}, fy={FY:.2f}, cx={CX:.2f}, cy={CY:.2f}")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Camera frustum visualization buffers
    cam_positions = []
    cam_frustums = []
    FRUSTUM_EVERY = 1
    FRUSTUM_SCALE = 0.25

    integrated = 0
    size_checked = False

    for fnum in tqdm(common_frames, desc="Integrating frames"):
        # Load per-frame pose: 4x4 cam-from-world (already what Open3D wants)
        T_cam_world = np.loadtxt(pose_files[fnum])
        if T_cam_world.shape != (4, 4):
            print(f"Skipping frame {fnum}: pose is {T_cam_world.shape}, expected 4x4")
            continue

        # World-from-camera for camera position extraction
        T_world_cam = np.linalg.inv(T_cam_world)

        color = o3d.io.read_image(rgb_files[fnum])

        # Depth is float32 .npy in meters — convert to uint16 mm for Open3D
        depth_m = np.load(depth_files[fnum])

        if not size_checked:
            check_image_sizes(color, depth_m, WIDTH, HEIGHT, rgb_files[fnum])
            size_checked = True

        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        depth = o3d.geometry.Image(depth_mm)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )

        volume.integrate(rgbd, intrinsic, T_cam_world)

        # Collect camera pose + frustum
        cam_positions.append(T_world_cam[:3, 3].copy())

        if integrated % FRUSTUM_EVERY == 0:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=WIDTH,
                view_height_px=HEIGHT,
                intrinsic=K,
                extrinsic=T_cam_world,
                scale=FRUSTUM_SCALE
            )
            cam_frustums.append(frustum)

        integrated += 1

    print(f"Done. Integrated {integrated} frames.")

    pcd = volume.extract_point_cloud()
    pcd = pcd.voxel_down_sample(VOXEL_LENGTH)
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO)
    print(f"Statistical outlier removal: {n_before} -> {len(pcd.points)} points ({n_before - len(pcd.points)} removed)")

    # World frame is already Z-up for this data
    pcd_path = os.path.join(data_dir, "tsdf_fused.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud: {pcd_path}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = os.path.join(data_dir, "tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved mesh: {mesh_path}")

    # Visualize: cloud + trajectory + frustums
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    cam_positions_np = np.array(cam_positions) if cam_positions else np.zeros((0, 3))
    traj = make_trajectory_line(cam_positions_np)

    geoms = [pcd, world_frame, traj] + cam_frustums
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <path/to/posed_rgbd/>")
        sys.exit(1)
    run_tsdf_fusion(sys.argv[1])
