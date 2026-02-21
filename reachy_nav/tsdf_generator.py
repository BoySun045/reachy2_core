import os
import re
import glob
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 3.5

VOXEL_LENGTH = 0.02
SDF_TRUNC = 0.03

STAT_NB_NEIGHBORS = 20
STAT_STD_RATIO = 2.0

MAX_FRAMES = 2000


# ----------------------------
# Helpers
# ----------------------------
def load_extrinsic_T_base_cam(path: str) -> np.ndarray:
    """
    Expect a 4x4 matrix in text file (space-separated).
    T_base_cam = (base <- cam), i.e. camera-to-base transform.
    """
    T = np.loadtxt(path)
    if T.shape != (4, 4):
        raise ValueError(f"Extrinsic must be 4x4, got {T.shape}")
    return T


def extract_step_number(filename: str) -> int | None:
    """Extract step number from filenames like 'rgb_step_42.png' or 'camera_pose_step_42.npy'."""
    m = re.search(r"step_(\d+)", filename)
    return int(m.group(1)) if m else None


def collect_step_files(folder: str, prefix: str, ext: str) -> dict[int, str]:
    """Return {step_number: filepath} for files matching prefix_step_N.ext."""
    result = {}
    for path in glob.glob(os.path.join(folder, f"{prefix}_step_*{ext}")):
        step = extract_step_number(os.path.basename(path))
        if step is not None:
            result[step] = path
    return result


def check_image_sizes(color_img, depth_img, width, height, rgb_path, depth_path):
    c = np.asarray(color_img)
    d = np.asarray(depth_img)
    if c.shape[1] != width or c.shape[0] != height:
        raise ValueError(
            f"RGB image size {c.shape[1]}x{c.shape[0]} does not match intrinsics "
            f"{width}x{height}. Example file: {rgb_path}"
        )
    if d.shape[1] != width or d.shape[0] != height:
        raise ValueError(
            f"Depth image size {d.shape[1]}x{d.shape[0]} does not match intrinsics "
            f"{width}x{height}. Example file: {depth_path}"
        )


def to_z_up(pts: np.ndarray) -> np.ndarray:
    """Transform from native (XZ ground, Y-down) to (XY ground, Z-up).

    Mapping: (x, y, z) → (x, z, -y)
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
def run_tsdf_fusion(data_dir: str, z_up: bool = False):
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    pose_dir = os.path.join(data_dir, "camera_poses")
    intrinsic_dir = os.path.join(data_dir, "camera_intrinsics")

    # Collect per-step files
    rgb_files = collect_step_files(rgb_dir, "rgb", ".png")
    depth_files = collect_step_files(depth_dir, "depth", ".png")
    pose_files = collect_step_files(pose_dir, "camera_pose", ".npy")
    intrinsic_files = collect_step_files(intrinsic_dir, "camera_intrinsics", ".npy")

    # Find common steps across all modalities
    common_steps = sorted(
        set(rgb_files) & set(depth_files) & set(pose_files) & set(intrinsic_files)
    )

    print(f"RGB frames:   {len(rgb_files)}")
    print(f"Depth frames: {len(depth_files)}")
    print(f"Pose frames:  {len(pose_files)}")
    print(f"Common steps: {len(common_steps)}")

    if not common_steps:
        raise RuntimeError("No common steps found across rgb, depth, poses, and intrinsics.")

    if len(common_steps) > MAX_FRAMES:
        common_steps = common_steps[:MAX_FRAMES]

    # Load intrinsics from first step to set up image dimensions
    K = np.load(intrinsic_files[common_steps[0]])
    first_rgb = o3d.io.read_image(rgb_files[common_steps[0]])
    HEIGHT, WIDTH = np.asarray(first_rgb).shape[:2]
    FX, FY = float(K[0, 0]), float(K[1, 1])
    CX, CY = float(K[0, 2]), float(K[1, 2])

    print(f"Intrinsics: {WIDTH}x{HEIGHT}, fx={FX:.2f}, fy={FY:.2f}, cx={CX:.2f}, cy={CY:.2f}")

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

    for step in tqdm(common_steps, desc="Integrating frames"):
        # Load per-step camera pose (T_world_cam: camera pose in world frame)
        T_world_cam = np.load(pose_files[step])

        # Open3D wants world->camera (cam <- world)
        T_cam_world = np.linalg.inv(T_world_cam)

        color = o3d.io.read_image(rgb_files[step])
        depth = o3d.io.read_image(depth_files[step])

        if not size_checked:
            check_image_sizes(color, depth, WIDTH, HEIGHT,
                              rgb_files[step], depth_files[step])
            size_checked = True

        # Load per-step intrinsics (in case they vary)
        K_step = np.load(intrinsic_files[step])
        intrinsic_step = o3d.camera.PinholeCameraIntrinsic(
            WIDTH, HEIGHT,
            float(K_step[0, 0]), float(K_step[1, 1]),
            float(K_step[0, 2]), float(K_step[1, 2])
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=DEPTH_SCALE,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )

        volume.integrate(rgbd, intrinsic_step, T_cam_world)

        # Collect camera pose + frustum
        cam_positions.append(T_world_cam[:3, 3].copy())

        if integrated % FRUSTUM_EVERY == 0:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=WIDTH,
                view_height_px=HEIGHT,
                intrinsic=K_step,
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
    print(f"Statistical outlier removal: {n_before} → {len(pcd.points)} points ({n_before - len(pcd.points)} removed)")

    # Convert to XY-ground, Z-up frame before saving
    if z_up:
        pcd.points = o3d.utility.Vector3dVector(to_z_up(np.asarray(pcd.points)))
    pcd_path = os.path.join(data_dir, "tsdf_fused.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud: {pcd_path}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if z_up:
        mesh.vertices = o3d.utility.Vector3dVector(to_z_up(np.asarray(mesh.vertices)))
        mesh.vertex_normals = o3d.utility.Vector3dVector(to_z_up(np.asarray(mesh.vertex_normals)))
    mesh_path = os.path.join(data_dir, "tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved mesh: {mesh_path}")

    # Visualize: cloud + trajectory + frustums
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    cam_positions_np = np.array(cam_positions) if cam_positions else np.zeros((0, 3))
    if z_up:
        cam_positions_np = to_z_up(cam_positions_np)
    traj = make_trajectory_line(cam_positions_np)

    # Transform frustum line-sets to Z-up
    if z_up:
        for f in cam_frustums:
            f.points = o3d.utility.Vector3dVector(to_z_up(np.asarray(f.points)))

    geoms = [pcd, world_frame, traj] + cam_frustums
    o3d.visualization.draw_geometries(geoms)

    # Also visualize the saved tsdf_fused.ply to confirm
    pcd_saved = o3d.io.read_point_cloud(pcd_path)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="tsdf_fused.ply")
    vis.add_geometry(pcd_saved)
    vis.add_geometry(origin)
    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF fusion from RGB-D + poses")
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--z-up", action="store_true",
                        help="Convert to Z-up coordinate frame (default is Y-up)")
    args = parser.parse_args()
    run_tsdf_fusion(args.data_dir, z_up=args.z_up)
