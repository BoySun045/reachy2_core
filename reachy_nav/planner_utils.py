import numpy as np
from scipy.spatial.transform import Rotation as R


def pose2posquat(pose):
    """
    Convert a 4x4 pose matrix to position and quaternion.

    Args:
        pose (np.ndarray): A 4x4 numpy array representing the pose.

    Returns:
        tuple: A dictionary containing position (np.ndarray) and quaternion (np.ndarray).
    """
    if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
        raise ValueError("Input pose must be a 4x4 numpy array.")

    position = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    return {"pos": position, "quat": quaternion}


def posquat2pose(posquat):
    """
    Convert position and quaternion to a 4x4 pose matrix.

    Args:
        pos (np.ndarray): A 3-element numpy array representing the position.
        quat (np.ndarray): A 4-element numpy array representing the quaternion.

    Returns:
        np.ndarray: A 4x4 numpy array representing the pose.
    """
    pos = np.array(posquat["pos"])
    quat = np.array(posquat["quat"])
    if not isinstance(pos, np.ndarray) or pos.shape != (3,):
        raise ValueError("Position must be a 3-element numpy array.")
    if not isinstance(quat, np.ndarray) or quat.shape != (4,):
        raise ValueError("Quaternion must be a 4-element numpy array.")

    rotation_matrix = R.from_quat(quat).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = pos

    return pose


def slerp(quat0, quat1, t):
    # SLERP (Spherical Linear Interpolation) for quaternions
    dot = np.dot(quat0, quat1)

    if dot < 0.0:
        quat1 = -quat1
        dot = -dot

    if dot > 0.9995:
        result = quat0 + t * (quat1 - quat0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    quat2 = quat1 - quat0 * dot
    quat2 = quat2 / np.linalg.norm(quat2)

    return quat0 * np.cos(theta) + quat2 * np.sin(theta)


def interpolate_waypoints(start, end, num_interp_points=None,
                          dist_step=0.30, angle_step_deg=15.0):
    """
    Interpolate between start and end poses using linear position interpolation
    and SLERP for quaternion rotation.

    Parameters
    ----------
    start : dict
        {"pos": [x, y, z], "quat": [x, y, z, w]}
    end : dict
        {"pos": [x, y, z], "quat": [x, y, z, w]}
    num_interp_points : int or None
        Number of intermediate points (excluding start and end).
        If None, automatically computed from dist_step and angle_step_deg.
    dist_step : float
        Insert one point every this many metres of translation (default 0.20).
    angle_step_deg : float
        Insert one point every this many degrees of rotation (default 15.0).

    Returns
    -------
    list of dict
        [start, ...intermediates..., end]
    """
    start_pos = np.asarray(start["pos"], dtype=float)
    end_pos = np.asarray(end["pos"], dtype=float)

    start_quat = np.asarray(start["quat"], dtype=float)
    end_quat = np.asarray(end["quat"], dtype=float)

    if num_interp_points is None:
        dist = np.linalg.norm(end_pos - start_pos)
        dot = np.clip(np.abs(np.dot(start_quat, end_quat)), 0.0, 1.0)
        angle_deg = np.degrees(2.0 * np.arccos(dot))
        n_from_dist = int(dist / dist_step) if dist_step > 0 else 0
        n_from_angle = int(angle_deg / angle_step_deg) if angle_step_deg > 0 else 0
        num_interp_points = max(n_from_dist, n_from_angle)

    if num_interp_points <= 0:
        return [{"pos": start_pos, "quat": start_quat},
                {"pos": end_pos, "quat": end_quat}]

    # Time steps (excluding endpoints for mids)
    t_values = np.linspace(0.0, 1.0, num_interp_points + 2)[1:-1]

    # Build path
    path = [{"pos": start_pos, "quat": start_quat}]
    for t in t_values:
        interp_pos = (1 - t) * start_pos + t * end_pos
        interp_quat = slerp(start_quat, end_quat, t)
        path.append({"pos": interp_pos, "quat": interp_quat})
    path.append({"pos": end_pos, "quat": end_quat})

    return path