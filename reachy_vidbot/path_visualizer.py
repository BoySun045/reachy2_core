import numpy as np
import open3d as o3d

# -----------------------------
# Helpers
# -----------------------------
def origin_sphere(position, color, radius=0.012):
    position = np.asarray(position, dtype=float).reshape(3)
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sph.translate(position)
    sph.paint_uniform_color(color)
    return sph

def frame_at_pose(position, R=None, size=0.035):
    """Create an Open3D coordinate frame centered at 'position' with rotation R (3x3)."""
    position = np.asarray(position, dtype=float).reshape(3)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if R is not None:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        frame.rotate(R, center=[0, 0, 0])
    frame.translate(position)
    return frame

def load_T_any(path: str) -> np.ndarray:
    """Accept (4,4) or (1,4,4) from npy/npz-like saves."""
    T = np.load(path)
    T = np.asarray(T, dtype=float)
    if T.shape == (1, 4, 4):
        T = T[0]
    if T.shape != (4, 4):
        raise ValueError(f"{path} must be (4,4) or (1,4,4); got {T.shape}")
    return T

def make_line_set(points: np.ndarray) -> o3d.geometry.LineSet:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("Need at least 2 points to make a line set.")
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls

def Ry_pi() -> np.ndarray:
    """
    180° about local Y axis:
      - keeps Y axis direction
      - flips X and Z
    Useful if AnyGrasp's Z is flipped w.r.t your EE convention.
    """
    R = np.eye(3, dtype=float)
    R[0, 0] = -1.0
    R[2, 2] = -1.0
    return R

# -----------------------------
# Load inputs
# -----------------------------
full_traj_positions = np.load("exec_traj_positions.npy")
full_traj_positions = np.asarray(full_traj_positions, dtype=float)
if full_traj_positions.ndim != 2 or full_traj_positions.shape[1] != 3:
    raise ValueError(f"exec_traj_positions.npy must be (T,3), got {full_traj_positions.shape}")

# Optional: load start EE pose from saved transforms (if available)
T_base_ee = None
try:
    full_traj_T = np.load("exec_traj_T_base_ee.npy")
    full_traj_T = np.asarray(full_traj_T, dtype=float)
    if full_traj_T.ndim == 3 and full_traj_T.shape[1:] == (4, 4):
        T_base_ee = full_traj_T[0].copy()
except Exception:
    pass

# Base <- Camera extrinsics (your known setup)
T_base_cam = np.array(
    [
        [-0.0000000000, -0.7372773368,  0.6755902076,  0.0580000000],
        [-1.0000000000, -0.0000000000, -0.0000000000,  0.0150000000],
        [-0.0000000000, -0.6755902076, -0.7372773368, -0.0300000000],
        [ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000],
    ],
    dtype=float,
)

# If T_base_ee wasn't loaded, build a minimal one from the first waypoint position
if T_base_ee is None:
    T_base_ee = np.eye(4, dtype=float)
    T_base_ee[:3, 3] = full_traj_positions[0]

# -----------------------------
# Load AnyGrasp pose (CAM frame), fix Z flip, convert rotation to BASE frame
# -----------------------------
T_cam_grasp = load_T_any("grasp_pose.npy")     # AnyGrasp output pose in CAMERA frame
R_cam_grasp = T_cam_grasp[:3, :3]

# Fix: Z is flipped (apply 180° about local Y; flips X and Z)
R_cam_grasp_fixed = R_cam_grasp @ Ry_pi()

# Convert rotation to BASE frame
R_base_cam = T_base_cam[:3, :3]
R_grasp_base = R_base_cam @ R_cam_grasp_fixed

# Debug prints
print("exec_traj_positions.npy:", full_traj_positions.shape)
try:
    print("exec_traj_T_base_ee.npy:", np.load("exec_traj_T_base_ee.npy").shape)
except Exception:
    print("exec_traj_T_base_ee.npy: not found (OK)")
print("grasp_pose.npy:", np.load("grasp_pose.npy").shape)
print("Using grasp orientation (base frame) with Z-flip fix (Ry(pi)).")

# -----------------------------
# Visualize: base + camera + ee start + waypoint frames with grasp orientation
# -----------------------------
geoms = []

# Base/world frame at origin
geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12, origin=[0, 0, 0]))

# Camera frame
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
cam_frame.transform(T_base_cam)
geoms.append(cam_frame)

# EE start frame (whatever you saved as start)
ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
ee_frame.transform(T_base_ee)
geoms.append(ee_frame)

# Show grasp orientation frame AT the FIRST waypoint (not using AnyGrasp translation)
T0 = np.eye(4, dtype=float)
T0[:3, :3] = R_grasp_base
T0[:3, 3]  = full_traj_positions[0]
grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
grasp_frame.transform(T0)
geoms.append(grasp_frame)

# Origin spheres
base_origin = np.array([0.0, 0.0, 0.0])
cam_origin  = T_base_cam[:3, 3]
ee_origin   = T_base_ee[:3, 3]
geoms.append(origin_sphere(base_origin, color=(1.0, 1.0, 1.0), radius=0.012))  # base
geoms.append(origin_sphere(cam_origin,  color=(1.0, 0.0, 0.0), radius=0.012))  # camera
geoms.append(origin_sphere(ee_origin,   color=(0.0, 1.0, 0.0), radius=0.012))  # ee start
geoms.append(origin_sphere(full_traj_positions[0], color=(1.0, 0.2, 1.0), radius=0.012))  # grasp ref at start

# Path line
geoms.append(make_line_set(full_traj_positions))

# Waypoint frames: position from path, orientation = corrected AnyGrasp orientation (constant)
STEP = 3
FRAME_SIZE = 0.035

for i in range(0, len(full_traj_positions), STEP):
    geoms.append(frame_at_pose(full_traj_positions[i], R=R_grasp_base, size=FRAME_SIZE))

# Highlight first & last waypoint with bigger frames
geoms.append(frame_at_pose(full_traj_positions[0],  R=R_grasp_base, size=0.06))
geoms.append(frame_at_pose(full_traj_positions[-1], R=R_grasp_base, size=0.06))

o3d.visualization.draw_geometries(
    geoms,
    window_name="Waypoints with constant AnyGrasp orientation (Z flipped fixed) + base/cam/EE frames",
    width=1280,
    height=800,
)
