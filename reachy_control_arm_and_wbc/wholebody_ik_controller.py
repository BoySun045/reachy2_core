#!/usr/bin/env python3
"""
Whole-body IK controller for Reachy2: mobile base (x, y, yaw) + 7-DOF arm.

Uses damped least-squares numerical IK to solve for all 10 DOFs
(base_x, base_y, base_yaw, 7 arm joints) to reach a desired EE pose
in the odom (world) frame.

If IK fails, falls back to a CSV lookup table of pre-sampled feasible poses.

Trajectory execution uses minimum-jerk (quintic) time scaling so that
both arm joints and base follow a coordinated profile completing in
exactly `target_time` seconds, with zero velocity/acceleration at
start and end.

Subscribe:  /target_ee_pose_world   (PoseStamped, frame: odom)
            /odom                   (Odometry)
            /joint_states           (JointState)
Publish:    /cmd_vel                (Twist)
            /r_arm_forward_position_controller/commands
            /wholebody_feedback     (PoseStamped)
"""

import csv
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, Point
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

# ── Constants ────────────────────────────────────────────────────────────

R_ARM_JOINTS = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
]

JOINT_LIMITS_LOWER = np.array([-1.5708, -1.5708, -1.5708, -2.25, -0.7854, -0.7854, -1.57])
JOINT_LIMITS_UPPER = np.array([ 1.5708,  0.0,     1.5708,  0.1,   0.7854,  0.7854,  1.57])

CTRL_RATE = 100.0  # Hz

# ── Homogeneous transform helpers ────────────────────────────────────────

def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float64)

def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float64)

def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)

def _Trans(x, y, z):
    T = np.eye(4)
    T[0,3], T[1,3], T[2,3] = x, y, z
    return T

def _rpy_to_T(xyz, rpy):
    """4x4 transform from xyz translation + rpy rotation."""
    T = _Rz(rpy[2]) @ _Ry(rpy[1]) @ _Rx(rpy[0])
    T[0,3], T[1,3], T[2,3] = xyz[0], xyz[1], xyz[2]
    return T


# ── Kinematic chain from URDF ───────────────────────────────────────────
# Each entry: (origin_xyz, origin_rpy, joint_type, joint_axis_index)
# joint_type: None=fixed, 'x'/'y'/'z'=revolute about that axis

_CHAIN = [
    # torso → r_arm_tip  (from URDF)
    ([0.0, -0.2, 0.0],   [1.3089969389957472, 0.0, 0.17453292519943295], None, None),  # r_shoulder_base_joint
    ([0.0, 0.0, 0.0],    [-1.5707963267948966, 0.0, 0.0],               None, None),  # r_shoulder_dummy_2
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'y',  0),     # r_shoulder_pitch
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'x',  1),     # r_shoulder_roll
    ([0.0, 0.0, 0.0],    [1.5707963267948966, 0.0, 0.0],                None, None),  # r_shoulder_dummy_out
    ([0.0, 0.0, 0.0],    [0.0, 1.5707963267948966, -1.5707963267948966], None, None),  # r_elbow_arm_joint
    ([0.0, 0.0, 0.28],   [0.0, 0.0, 0.0],                              None, None),  # r_elbow_base_joint
    ([0.0, 0.0, 0.0],    [3.141592, 0.0, 1.5707963267948966],           None, None),  # r_elbow_dummy_2
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'z',  2),     # r_elbow_yaw
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'y',  3),     # r_elbow_pitch
    ([0.0, 0.0, 0.0],    [3.141592, 0.0, 1.5707963267948966],           None, None),  # r_elbow_dummy_out
    ([0.0, 0.0, 0.28],   [3.141592653589793, 0.0, 1.5707963267948966],  None, None),  # r_wrist_base_joint
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'x',  4),     # r_wrist_roll
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'y',  5),     # r_wrist_pitch
    ([0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],                               'z',  6),     # r_wrist_yaw
    ([0.0, 0.0, 0.0],    [3.141592, 0.0, 0.0],                          None, None),  # r_wrist_out_joint
    ([0.0, 0.0, 0.1],    [3.141592653589793, 0.0, 0.0],                 None, None),  # r_tip_joint
]

# Fixed transform: base_link → torso  (from actual TF at runtime)
T_BASE_TO_TORSO = _Trans(-0.01, 0.0, 0.996)

# Pre-compute per-segment origin transforms (constant)
_SEG_ORIGINS = [_rpy_to_T(xyz, rpy) for xyz, rpy, _, _ in _CHAIN]
_SEG_TYPES   = [(jtype, jidx) for _, _, jtype, jidx in _CHAIN]

_ROT_FN = {'x': _Rx, 'y': _Ry, 'z': _Rz}
_AXIS_COL = {'x': 0, 'y': 1, 'z': 2}


# ── Forward kinematics ──────────────────────────────────────────────────

def arm_fk(joint_values):
    """FK from torso frame to EE. Returns 4x4 transform."""
    T = np.eye(4)
    for T_origin, (jtype, jidx) in zip(_SEG_ORIGINS, _SEG_TYPES):
        T = T @ T_origin
        if jtype is not None:
            T = T @ _ROT_FN[jtype](joint_values[jidx])
    return T


def wholebody_fk(base_x, base_y, base_yaw, joint_values):
    """FK from odom frame to EE. Returns 4x4 transform."""
    T_odom_base = _Rz(base_yaw)
    T_odom_base[0, 3] = base_x
    T_odom_base[1, 3] = base_y
    return T_odom_base @ T_BASE_TO_TORSO @ arm_fk(joint_values)


# ── Geometric Jacobian ──────────────────────────────────────────────────

def arm_jacobian(joint_values):
    """6×7 geometric Jacobian of the arm in torso frame.
    Rows 0-2: linear velocity,  rows 3-5: angular velocity."""
    J = np.zeros((6, 7))
    T_ee = arm_fk(joint_values)
    p_ee = T_ee[:3, 3]

    T = np.eye(4)
    for T_origin, (jtype, jidx) in zip(_SEG_ORIGINS, _SEG_TYPES):
        T = T @ T_origin
        if jtype is not None:
            col = _AXIS_COL[jtype]
            z_i = T[:3, col]          # joint axis in torso frame
            p_i = T[:3, 3]            # joint origin in torso frame
            J[:3, jidx] = np.cross(z_i, p_ee - p_i)
            J[3:, jidx] = z_i
            T = T @ _ROT_FN[jtype](joint_values[jidx])
    return J


def wholebody_jacobian(base_x, base_y, base_yaw, joint_values):
    """6×10 whole-body Jacobian in odom frame.
    Columns: [base_x, base_y, base_yaw, j0..j6]."""
    J_wb = np.zeros((6, 10))

    c_yaw = np.cos(base_yaw)
    s_yaw = np.sin(base_yaw)
    R_yaw = np.array([[c_yaw, -s_yaw, 0],
                       [s_yaw,  c_yaw, 0],
                       [0,      0,     1]])

    # Arm Jacobian in torso frame → rotate to odom frame
    J_arm = arm_jacobian(joint_values)
    J_arm[:3, :] = R_yaw @ J_arm[:3, :]
    J_arm[3:, :] = R_yaw @ J_arm[3:, :]
    J_wb[:, 3:] = J_arm

    # Base x, y: pure translation in odom frame
    J_wb[0, 0] = 1.0
    J_wb[1, 1] = 1.0

    # Base yaw: rotation about odom Z
    T_torso_ee = arm_fk(joint_values)
    p_torso_ee = T_torso_ee[:3, 3]
    p_base_ee = T_BASE_TO_TORSO[:3, :3] @ p_torso_ee + T_BASE_TO_TORSO[:3, 3]
    p_odom_ee_rel = R_yaw @ p_base_ee

    z_odom = np.array([0.0, 0.0, 1.0])
    J_wb[:3, 2] = np.cross(z_odom, p_odom_ee_rel)
    J_wb[3:, 2] = z_odom

    return J_wb


# ── Pose error ───────────────────────────────────────────────────────────

def pose_error_6d(T_current, T_target):
    """6D error: [position_error(3), orientation_error(3)]."""
    pos_err = T_target[:3, 3] - T_current[:3, 3]
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    ori_err = Rotation.from_matrix(R_err).as_rotvec()
    return np.concatenate([pos_err, ori_err])


# ── IK solver ────────────────────────────────────────────────────────────

def _arm_ik_once(T_target_torso, joints_init, max_iter, pos_tol, ori_tol,
                 damping, ori_weight):
    """Arm-only IK in torso frame. Returns (converged, joints, pos_err, ori_err, iters)."""
    q = joints_init.copy()

    for it in range(max_iter):
        T_cur = arm_fk(q)
        err = pose_error_6d(T_cur, T_target_torso)
        err[3:] *= ori_weight

        p_err = np.linalg.norm(err[:3])
        o_err = np.linalg.norm(err[3:])

        if p_err < pos_tol and o_err < ori_tol:
            return True, q, p_err, o_err, it

        J = arm_jacobian(q)
        J[3:, :] *= ori_weight

        A = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)

        max_step = 0.15
        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq *= max_step / norm

        q += dq
        q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    return False, q, p_err, o_err, max_iter


def solve_arm_only_ik(
    T_target_torso,
    joints_init,
    max_iter=200,
    pos_tol=0.002,
    ori_tol=0.02,
    damping=0.05,
    ori_weight=1.0,
    n_restarts=4,
):
    """Solve 7-DOF arm-only IK in torso frame.

    Returns (converged, joints[7], pos_err, iters).
    """
    ok, q_best, best_perr, best_oerr, iters = _arm_ik_once(
        T_target_torso, joints_init, max_iter, pos_tol, ori_tol, damping, ori_weight)
    if ok:
        return True, q_best, best_perr, iters

    best_cost = best_perr + 0.3 * best_oerr

    for _ in range(n_restarts):
        q_rand = JOINT_LIMITS_LOWER + np.random.rand(7) * (JOINT_LIMITS_UPPER - JOINT_LIMITS_LOWER)
        ok, q, perr, oerr, it = _arm_ik_once(
            T_target_torso, q_rand, max_iter, pos_tol, ori_tol, damping, ori_weight)
        if ok:
            return True, q, perr, it
        cost = perr + 0.3 * oerr
        if cost < best_cost:
            best_cost = cost
            q_best = q
            best_perr = perr

    return False, q_best, best_perr, max_iter


def _ik_once(T_target, q_init, max_iter, pos_tol, ori_tol, damping,
             base_pos_weight, base_yaw_weight, ori_weight):
    """Single IK solve attempt. Returns (converged, q, pos_err, ori_err, iters)."""
    q = q_init.copy()

    w = np.ones(10)
    w[0] = base_pos_weight
    w[1] = base_pos_weight
    w[2] = base_yaw_weight
    W_inv = np.diag(1.0 / w)

    for it in range(max_iter):
        T_cur = wholebody_fk(q[0], q[1], q[2], q[3:])
        err = pose_error_6d(T_cur, T_target)
        err[3:] *= ori_weight

        p_err = np.linalg.norm(err[:3])
        o_err = np.linalg.norm(err[3:])

        if p_err < pos_tol and o_err < ori_tol:
            return True, q, p_err, o_err, it

        J = wholebody_jacobian(q[0], q[1], q[2], q[3:])
        J[3:, :] *= ori_weight

        JW = J @ W_inv
        A = JW @ J.T + damping**2 * np.eye(6)
        dq = W_inv @ J.T @ np.linalg.solve(A, err)

        max_step = 0.15
        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq *= max_step / norm

        q += dq
        q[3:] = np.clip(q[3:], JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    return False, q, p_err, o_err, max_iter


def solve_wholebody_ik(
    T_target,
    base_xy_init,
    base_yaw_init,
    joints_init,
    max_iter=200,
    pos_tol=0.002,
    ori_tol=0.02,
    damping=0.05,
    base_pos_weight=0.5,
    base_yaw_weight=0.3,
    ori_weight=1.0,
    n_restarts=4,
):
    """Solve 10-DOF whole-body IK with random restarts.

    Returns (converged, base_x, base_y, base_yaw, joints[7], iters).
    """
    q0 = np.zeros(10)
    q0[0] = base_xy_init[0]
    q0[1] = base_xy_init[1]
    q0[2] = base_yaw_init
    q0[3:] = joints_init.copy()

    ok, q_best, best_perr, best_oerr, iters = _ik_once(
        T_target, q0, max_iter, pos_tol, ori_tol, damping,
        base_pos_weight, base_yaw_weight, ori_weight)
    if ok:
        return True, q_best[0], q_best[1], q_best[2], q_best[3:].copy(), iters

    best_cost = best_perr + 0.3 * best_oerr

    for _ in range(n_restarts):
        q_rand = np.zeros(10)
        q_rand[0] = base_xy_init[0]
        q_rand[1] = base_xy_init[1]
        q_rand[2] = base_yaw_init
        q_rand[3:] = JOINT_LIMITS_LOWER + np.random.rand(7) * (JOINT_LIMITS_UPPER - JOINT_LIMITS_LOWER)

        ok, q, perr, oerr, it = _ik_once(
            T_target, q_rand, max_iter, pos_tol, ori_tol, damping,
            base_pos_weight, base_yaw_weight, ori_weight)
        if ok:
            return True, q[0], q[1], q[2], q[3:].copy(), it

        cost = perr + 0.3 * oerr
        if cost < best_cost:
            best_cost = cost
            q_best = q
            best_perr = perr
            best_oerr = oerr

    return False, q_best[0], q_best[1], q_best[2], q_best[3:].copy(), max_iter


# ── CSV fallback ─────────────────────────────────────────────────────────

def load_fallback_csv(path):
    """Load pre-sampled workspace for fallback nearest-neighbour lookup."""
    joints_list, pos_list, quat_list = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            joints_list.append([float(row[j]) for j in R_ARM_JOINTS])
            pos_list.append([float(row["ee_x"]), float(row["ee_y"]), float(row["ee_z"])])
            quat_list.append([float(row[k]) for k in ("ee_qx", "ee_qy", "ee_qz", "ee_qw")])
    return np.array(joints_list), np.array(pos_list), np.array(quat_list)


# ── ROS2 Node ────────────────────────────────────────────────────────────

class WholeBodyIKController(Node):
    def __init__(self):
        super().__init__("wholebody_ik_controller")

        # Parameters
        self.declare_parameter("csv_path", str(Path(__file__).parent / "ee_poses_right_dense.csv"))

        self.declare_parameter("base_speed", 0.12)
        self.declare_parameter("base_angular_speed", 0.3)

        self.declare_parameter("target_time", 0.3)               # trajectory duration in seconds

        self.declare_parameter("base_pos_tolerance", 0.02)
        self.declare_parameter("base_yaw_tolerance", 0.03)

        self.declare_parameter("tracking_pos_threshold", 0.01)   # EE position threshold (m)
        self.declare_parameter("tracking_ori_threshold", 0.05)   # EE orientation threshold (rad)
        self.declare_parameter("tracking_timeout", 500.0)          # max tracking time (s), 0=no timeout

        self.declare_parameter("ik_pos_tol", 0.002)
        self.declare_parameter("ik_ori_tol", 0.02)
        self.declare_parameter("ik_damping", 0.05)
        self.declare_parameter("ik_base_pos_weight", 5.0)
        self.declare_parameter("ik_base_yaw_weight", 3.0)
        self.declare_parameter("arm_only_first", False)

        self.base_speed = float(self.get_parameter("base_speed").value)
        self.base_ang_speed = float(self.get_parameter("base_angular_speed").value)
        self.target_time = float(self.get_parameter("target_time").value)
        self.base_pos_tol = float(self.get_parameter("base_pos_tolerance").value)
        self.base_yaw_tol = float(self.get_parameter("base_yaw_tolerance").value)

        self.tracking_pos_thr = float(self.get_parameter("tracking_pos_threshold").value)
        self.tracking_ori_thr = float(self.get_parameter("tracking_ori_threshold").value)
        self.tracking_timeout = float(self.get_parameter("tracking_timeout").value)

        self.ik_pos_tol = float(self.get_parameter("ik_pos_tol").value)
        self.ik_ori_tol = float(self.get_parameter("ik_ori_tol").value)
        self.ik_damping = float(self.get_parameter("ik_damping").value)
        self.ik_base_pos_w = float(self.get_parameter("ik_base_pos_weight").value)
        self.ik_base_yaw_w = float(self.get_parameter("ik_base_yaw_weight").value)
        self.arm_only_first = self.get_parameter("arm_only_first").value

        # Load fallback CSV
        csv_path = self.get_parameter("csv_path").value
        self.fb_joints, self.fb_positions, self.fb_quats = load_fallback_csv(csv_path)
        self.fb_tree = KDTree(self.fb_positions)
        self.get_logger().info(f"Loaded {len(self.fb_positions)} fallback samples")

        # TF (for reading current state / validation)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.arm_pub = self.create_publisher(
            Float64MultiArray, "/r_arm_forward_position_controller/commands", 10
        )
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        latching = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.fb_pub = self.create_publisher(PoseStamped, "/wholebody_feedback", latching)
        self.marker_pub = self.create_publisher(MarkerArray, "/target_ee_marker", 10)

        # State
        self.base_x = None
        self.base_y = None
        self.base_yaw = None
        self.current_joints = np.zeros(7)
        self.commanded_joints = np.zeros(7)
        self.joints_received = False

        self.target_base_x = None
        self.target_base_y = None
        self.target_base_yaw = None
        self.target_joints = None
        self.T_target_world = None       # 4x4 target EE pose in odom frame
        self.active = False
        self.base_arrived = False
        self.last_resolve_time = None    # rate-limit IK re-solves during tracking

        # Trajectory interpolation state
        self.traj_start_time = None
        self.traj_duration = 1.0
        self.traj_start_joints = np.zeros(7)
        self.traj_start_base = (0.0, 0.0, 0.0)  # (x, y, yaw)

        # Smoothed velocity (for base tracking)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        # Subscribers
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, "/target_ee_pose_world", self._on_target, 10)

        # Control timer
        self.create_timer(1.0 / CTRL_RATE, self._tick)

        self.get_logger().info(
            "Whole-body IK controller ready.\n"
            "  Waiting for /odom and /joint_states...\n"
            "  Then publish PoseStamped to /target_ee_pose_world (frame: odom)"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        first = self.base_x is None
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.base_x = p.x
        self.base_y = p.y
        self.base_yaw = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')[2]
        if first:
            self.get_logger().info(
                f"Odom OK. Base at ({self.base_x:.3f}, {self.base_y:.3f}), "
                f"yaw={np.degrees(self.base_yaw):.1f}deg. Ready for targets."
            )

    def _joint_state_cb(self, msg):
        name_list = list(msg.name)
        for i, jname in enumerate(R_ARM_JOINTS):
            if jname in name_list:
                self.current_joints[i] = msg.position[name_list.index(jname)]
        self.joints_received = True

    def _on_target(self, msg):
        if self.base_x is None:
            self.get_logger().error("No odom yet")
            return
        if not self.joints_received:
            self.get_logger().warn("No joint states yet, using zeros as seed")

        p = msg.pose.position
        o = msg.pose.orientation
        T_target = np.eye(4)
        T_target[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        T_target[:3, 3] = [p.x, p.y, p.z]

        self.T_target_world = T_target.copy()
        self.get_logger().info(f"Target EE: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        self._publish_target_marker(T_target)

        # ── 1. Optionally try arm-only IK first ─────────────────────────
        if self.arm_only_first:
            T_odom_base = _Rz(self.base_yaw)
            T_odom_base[0, 3] = self.base_x
            T_odom_base[1, 3] = self.base_y
            T_odom_torso = T_odom_base @ T_BASE_TO_TORSO
            T_target_torso = np.linalg.inv(T_odom_torso) @ T_target

            arm_ok, arm_joints, arm_perr, arm_iters = solve_arm_only_ik(
                T_target_torso,
                joints_init=self.current_joints,
                pos_tol=self.ik_pos_tol,
                ori_tol=self.ik_ori_tol,
                damping=self.ik_damping,
            )

            if arm_ok:
                self.get_logger().info(
                    f"Arm-only IK converged in {arm_iters} iters (no base motion needed)"
                )
                self._set_targets(self.base_x, self.base_y, self.base_yaw, arm_joints)
                return

            self.get_logger().info(
                f"Arm-only IK failed (pos_err={arm_perr:.3f}m), trying whole-body IK..."
            )

        # ── 2. Whole-body IK ─────────────────────────────────────────────
        converged, bx, by, byaw, joints, iters = solve_wholebody_ik(
            T_target,
            base_xy_init=np.array([self.base_x, self.base_y]),
            base_yaw_init=self.base_yaw,
            joints_init=self.current_joints,
            pos_tol=self.ik_pos_tol,
            ori_tol=self.ik_ori_tol,
            damping=self.ik_damping,
            base_pos_weight=self.ik_base_pos_w,
            base_yaw_weight=self.ik_base_yaw_w,
        )

        if converged:
            self.get_logger().info(
                f"Whole-body IK converged in {iters} iters. "
                f"Base: ({self.base_x:.3f},{self.base_y:.3f}) → ({bx:.3f},{by:.3f}), "
                f"yaw: {self.base_yaw:.3f} → {byaw:.3f}"
            )
            self._set_targets(bx, by, byaw, joints)
            return

        # IK failed → check residual
        T_check = wholebody_fk(bx, by, byaw, joints)
        err = pose_error_6d(T_check, T_target)
        p_err = np.linalg.norm(err[:3])

        if p_err < 0.05:
            self.get_logger().warn(
                f"Whole-body IK did not fully converge (pos_err={p_err:.4f}m) but close enough."
            )
            self._set_targets(bx, by, byaw, joints)
            return

        # Fallback to CSV lookup
        self.get_logger().warn(
            f"IK failed (pos_err={p_err:.3f}m). Falling back to CSV lookup."
        )
        self._fallback_csv(T_target)

    def _fallback_csv(self, T_target):
        """Use pre-sampled workspace CSV as fallback."""
        T_odom_base = _Rz(self.base_yaw)
        T_odom_base[0, 3] = self.base_x
        T_odom_base[1, 3] = self.base_y
        T_odom_torso = T_odom_base @ T_BASE_TO_TORSO
        T_torso_odom = np.linalg.inv(T_odom_torso)

        T_target_torso = T_torso_odom @ T_target
        target_pos_torso = T_target_torso[:3, 3]

        dist, idx = self.fb_tree.query(target_pos_torso)
        fb_joints = self.fb_joints[idx]
        fb_pos_torso = self.fb_positions[idx]

        offset_torso = target_pos_torso - fb_pos_torso
        offset_odom = T_odom_torso[:3, :3] @ offset_torso

        new_bx = self.base_x + offset_odom[0]
        new_by = self.base_y + offset_odom[1]

        self.get_logger().info(
            f"Fallback: nearest sample dist={dist:.4f}m, "
            f"moving base ({self.base_x:.3f},{self.base_y:.3f}) → ({new_bx:.3f},{new_by:.3f})"
        )

        self._set_targets(new_bx, new_by, self.base_yaw, fb_joints)

    def _set_targets(self, bx, by, byaw, joints):
        """Activate a new target, snapshotting current state for trajectory start."""
        self.target_base_x = float(bx)
        self.target_base_y = float(by)
        self.target_base_yaw = float(byaw)
        self.target_joints = np.array(joints, dtype=np.float64)

        # Snapshot current state as trajectory start.
        # If mid-trajectory use commanded_joints (smooth restart);
        # otherwise use current_joints (avoids zeros-at-init).
        if self.active:
            self.traj_start_joints = self.commanded_joints.copy()
        else:
            self.traj_start_joints = self.current_joints.copy()
        self.traj_start_base = (float(self.base_x), float(self.base_y), float(self.base_yaw))
        self.traj_start_time = self.get_clock().now()

        # Scale duration proportional to displacement so peak velocity
        # stays consistent regardless of motion size (prevents speed
        # burst when teleop commands stop arriving).
        max_joint_delta = float(np.max(np.abs(self.target_joints - self.traj_start_joints)))
        base_delta = float(np.hypot(self.target_base_x - self.base_x,
                                     self.target_base_y - self.base_y))
        # Normalize: target_time is the duration for a "full" motion
        # (1 rad joint move or 0.15m base move, whichever is larger)
        motion_scale = max(max_joint_delta / 1.0, base_delta / 0.15)
        self.traj_duration = float(np.clip(
            self.target_time * motion_scale,
            0.05,               # minimum 50ms to avoid instant snaps
            self.target_time,   # never longer than target_time
        ))

        self.commanded_joints = self.traj_start_joints.copy()

        # Only reset velocity filter on fresh starts, not mid-teleop restarts.
        # Resetting every frame kills base velocity during continuous commands.
        if not self.active:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0

        self.active = True
        self.base_arrived = False
        self.last_resolve_time = None

    # ── Minimum-jerk trajectory ─────────────────────────────────────────

    @staticmethod
    def _min_jerk(t, T):
        """Minimum-jerk (quintic) time scaling: s in [0, 1].

        s(t) = 10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5
        Zero velocity and acceleration at start and end.
        """
        if T <= 0.0:
            return 1.0
        tau = min(t / T, 1.0)
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    # ── Control loop (50 Hz) ──────────────────────────────────────────────

    def _tick(self):
        if not self.active or self.target_joints is None:
            return
        if self.base_x is None or self.traj_start_time is None:
            return

        # ── 1) Compute trajectory progress s ∈ [0, 1] ───────────────────
        now = self.get_clock().now()
        elapsed = (now - self.traj_start_time).nanoseconds * 1e-9
        s = self._min_jerk(elapsed, self.traj_duration)

        # ── 2) Interpolate arm joints ────────────────────────────────────
        self.commanded_joints = self.traj_start_joints + s * (self.target_joints - self.traj_start_joints)

        arm_msg = Float64MultiArray()
        arm_msg.data = self.commanded_joints.tolist()
        self.arm_pub.publish(arm_msg)

        # ── 3) Base: track FINAL target directly with P-control ─────────
        #    (no trajectory interpolation — the base uses cmd_vel so
        #     proportional control + braking handles smooth deceleration)
        dx = self.target_base_x - self.base_x
        dy = self.target_base_y - self.base_y
        dyaw = self._angle_diff(self.target_base_yaw, self.base_yaw)

        pos_dist = float(np.hypot(dx, dy))
        yaw_dist = float(abs(dyaw))

        # ── 4) Tracking check: use actual EE pose error as stopping criterion ──
        if s >= 1.0 and self.T_target_world is not None:
            T_actual = wholebody_fk(self.base_x, self.base_y, self.base_yaw,
                                     self.current_joints)
            ee_err = pose_error_6d(T_actual, self.T_target_world)
            ee_pos_err = float(np.linalg.norm(ee_err[:3]))
            ee_ori_err = float(np.linalg.norm(ee_err[3:]))

            # Timeout: stop tracking if we've been at it too long
            timed_out = False
            if self.tracking_timeout > 0.0:
                tracking_elapsed = elapsed - self.traj_duration
                if tracking_elapsed > self.tracking_timeout:
                    timed_out = True

            if timed_out or (ee_pos_err < self.tracking_pos_thr and ee_ori_err < self.tracking_ori_thr):
                # EE is within threshold → done
                if not self.base_arrived:
                    self.base_arrived = True
                    self.cmd_vx = 0.0
                    self.cmd_vy = 0.0
                    self.cmd_wz = 0.0
                    self.base_pub.publish(Twist())
                    reason = "timeout" if timed_out else "converged"
                    self.get_logger().info(
                        f"Tracking {reason}. ee_pos_err={ee_pos_err:.4f}m, "
                        f"ee_ori_err={ee_ori_err:.4f}rad"
                    )
                    self._publish_feedback()
                else:
                    # Keep publishing target joints to hold position
                    arm_msg = Float64MultiArray()
                    arm_msg.data = self.target_joints.tolist()
                    self.arm_pub.publish(arm_msg)
                    self.base_pub.publish(Twist())
                return

            # EE error still too large — check if base reached its IK target
            # but EE is still off → need to re-solve IK from current state
            if pos_dist < self.base_pos_tol * 2 and yaw_dist < self.base_yaw_tol * 2:
                should_resolve = False
                if self.last_resolve_time is None:
                    should_resolve = True
                else:
                    dt = (now - self.last_resolve_time).nanoseconds * 1e-9
                    if dt > 0.5:  # re-solve at most 2Hz
                        should_resolve = True

                if should_resolve:
                    self.last_resolve_time = now
                    ok, bx, by, byaw, joints, iters = solve_wholebody_ik(
                        self.T_target_world,
                        base_xy_init=np.array([self.base_x, self.base_y]),
                        base_yaw_init=self.base_yaw,
                        joints_init=self.current_joints,
                        pos_tol=self.ik_pos_tol,
                        ori_tol=self.ik_ori_tol,
                        damping=self.ik_damping,
                        base_pos_weight=self.ik_base_pos_w,
                        base_yaw_weight=self.ik_base_yaw_w,
                    )
                    if ok:
                        self.get_logger().debug(
                            f"Tracking re-solve: ee_pos_err={ee_pos_err:.4f}m, "
                            f"ee_ori_err={ee_ori_err:.4f}rad → new IK in {iters} iters"
                        )
                        self._set_targets(bx, by, byaw, joints)
                        return

        # ── 5) Base velocity: P-control toward final target ──────────────
        c = np.cos(self.base_yaw)
        sn = np.sin(self.base_yaw)
        vx_base =  c * dx + sn * dy
        vy_base = -sn * dx + c * dy

        p_gain = 2.0
        vx_cmd = vx_base * p_gain
        vy_cmd = vy_base * p_gain
        wz_cmd = dyaw * p_gain

        # Braking near target: decelerate smoothly on approach
        brake_radius = 0.15  # m
        yaw_brake_radius = 0.30  # rad
        lin_brake = float(np.clip(pos_dist / brake_radius, 0.0, 1.0))
        yaw_brake = float(np.clip(yaw_dist / yaw_brake_radius, 0.0, 1.0))

        lin_cap = self.base_speed * lin_brake
        lin_speed = float(np.hypot(vx_cmd, vy_cmd))
        if lin_speed > lin_cap and lin_speed > 1e-6:
            vx_cmd *= lin_cap / lin_speed
            vy_cmd *= lin_cap / lin_speed
        ang_cap = self.base_ang_speed * yaw_brake
        wz_cmd = float(np.clip(wz_cmd, -ang_cap, ang_cap))

        # ── 6) Smoothing ─────────────────────────────────────────────────
        alpha = 0.3
        self.cmd_vx += alpha * (vx_cmd - self.cmd_vx)
        self.cmd_vy += alpha * (vy_cmd - self.cmd_vy)
        self.cmd_wz += alpha * (wz_cmd - self.cmd_wz)

        # ── 7) Publish ───────────────────────────────────────────────────
        cmd = Twist()
        cmd.linear.x = float(self.cmd_vx)
        cmd.linear.y = float(self.cmd_vy)
        cmd.angular.z = float(self.cmd_wz)
        self.base_pub.publish(cmd)

    @staticmethod
    def _angle_diff(target, current):
        """Shortest angular difference, result in [-pi, pi]."""
        d = target - current
        return (d + np.pi) % (2 * np.pi) - np.pi

    def _publish_target_marker(self, T_target):
        """Publish XYZ axes marker at target pose for RViz visualization."""
        stamp = self.get_clock().now().to_msg()
        pos = T_target[:3, 3]
        R = T_target[:3, :3]

        axis_len = 0.10
        shaft_d = 0.008
        head_d = 0.015
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        markers = MarkerArray()
        for i in range(3):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = stamp
            m.ns = "target_ee_axes"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            tip = pos + R[:, i] * axis_len
            m.points = [
                Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])),
            ]
            m.scale.x = shaft_d
            m.scale.y = head_d
            m.scale.z = 0.0

            m.color.r = float(colors[i][0])
            m.color.g = float(colors[i][1])
            m.color.b = float(colors[i][2])
            m.color.a = 1.0
            m.lifetime.sec = 1  # auto-expire if no updates
            markers.markers.append(m)

        self.marker_pub.publish(markers)

    def _publish_feedback(self):
        T_ee = wholebody_fk(self.base_x, self.base_y, self.base_yaw, self.target_joints)
        fb = PoseStamped()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.header.frame_id = "odom"
        fb.pose.position.x = float(T_ee[0, 3])
        fb.pose.position.y = float(T_ee[1, 3])
        fb.pose.position.z = float(T_ee[2, 3])
        quat = Rotation.from_matrix(T_ee[:3, :3]).as_quat()
        fb.pose.orientation.x = float(quat[0])
        fb.pose.orientation.y = float(quat[1])
        fb.pose.orientation.z = float(quat[2])
        fb.pose.orientation.w = float(quat[3])
        self.fb_pub.publish(fb)


def main():
    rclpy.init()
    node = WholeBodyIKController()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
