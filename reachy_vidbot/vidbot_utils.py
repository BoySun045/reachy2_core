"""
Shared constants and helpers for VidBot ROS nodes.

Used by vidbot_ros_graspnet_node.py and vidbot_manager.py.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

T_BASE_CAM = np.array(
    [
        [-0.0, -0.7372773368,  0.6755902076,  0.0580000000],
        [-1.0, -0.0,          -0.0,           0.0150000000],
        [-0.0, -0.6755902076, -0.7372773368, -0.0300000000],
        [ 0.0,  0.0,           0.0,           1.0],
    ],
    dtype=float,
)

T_BASE_EE_DEFAULT = np.array(
    [
        [ 0.167, -0.092, -0.982,  0.248],
        [-0.035,  0.994, -0.099, -0.128],
        [ 0.985,  0.051,  0.163, -0.245],
        [ 0.000,  0.000,  0.000,  1.000],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Action verb classification
# ---------------------------------------------------------------------------

PRESS_VERBS = {"press", "push", "click"}
PLACE_VERBS = {"put", "leave", "drop", "place"}


def get_action_type(instruction: str | None) -> str:
    """Return 'press', 'place', or 'other' based on the instruction verb."""
    if not instruction:
        return "other"
    verb = instruction.strip().split()[0].lower()
    if verb in PRESS_VERBS:
        return "press"
    if verb in PLACE_VERBS:
        return "place"
    return "other"


def is_press_action(instruction: str | None) -> bool:
    return get_action_type(instruction) == "press"


def get_scale_for_instruction(instruction: str | None) -> float:
    """Return trajectory scale: small for press actions, 1.0 otherwise."""
    return 0.075 if is_press_action(instruction) else 1.0


# ---------------------------------------------------------------------------
# Pure-numpy transform helpers
# ---------------------------------------------------------------------------

def make_T(R, p) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from a 3x3 rotation and a 3-vector."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, float).reshape(3)
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an Nx3 array of points."""
    pts = np.asarray(pts, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]


def lerp_positions(p0, p1, num_steps: int) -> np.ndarray:
    """Linearly interpolate between two 3-vectors (excluding endpoints)."""
    p0 = np.asarray(p0, float).reshape(3)
    p1 = np.asarray(p1, float).reshape(3)
    alphas = np.linspace(0.0, 1.0, num_steps + 2)[1:-1]
    return (1 - alphas)[:, None] * p0[None, :] + alphas[:, None] * p1[None, :]


# ---------------------------------------------------------------------------
# ROS Pose ↔ 4x4 conversions
# ---------------------------------------------------------------------------

def mat4_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 transform to a geometry_msgs/Pose."""
    p = Pose()
    p.position.x = float(T[0, 3])
    p.position.y = float(T[1, 3])
    p.position.z = float(T[2, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p


def pose_to_mat4(pose: Pose) -> np.ndarray:
    """Convert a geometry_msgs/Pose to a 4x4 transform."""
    T = np.eye(4)
    o = pose.orientation
    T[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
    T[0, 3] = pose.position.x
    T[1, 3] = pose.position.y
    T[2, 3] = pose.position.z
    return T


def shift_pose_along_local_z(pose: Pose, offset: float) -> Pose:
    """Return a new Pose shifted by `offset` metres along its own local Z-axis."""
    R = Rotation.from_quat([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w,
    ]).as_matrix()
    local_z = R[:, 2]
    shifted = Pose()
    shifted.position.x = pose.position.x + local_z[0] * offset
    shifted.position.y = pose.position.y + local_z[1] * offset
    shifted.position.z = pose.position.z + local_z[2] * offset
    shifted.orientation = pose.orientation
    return shifted
