#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import glob
import subprocess
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

app = FastAPI(title="Vidbot Inference Server")

# -----------------------------
# CONFIG (edit these paths once)
# -----------------------------

VDBOT_ROOT = os.path.abspath(os.path.dirname(__file__))  # assumes script lives in vdbot_t/
DATASET_NAME = "reachy2"
DATASET_DIR = os.path.join(VDBOT_ROOT, "datasets", DATASET_NAME)

COLOR_DIR = os.path.join(DATASET_DIR, "color")
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
PRED_DIR = os.path.join(DATASET_DIR, "prediction")

FRAME_ID = "000000"
COLOR_PATH = os.path.join(COLOR_DIR, f"{FRAME_ID}.png")
DEPTH_PATH = os.path.join(DEPTH_DIR, f"{FRAME_ID}.png")

# Your extrinsics (camera -> base)
T_BASE_CAM = np.array(
    [
        [-0.0, -0.7372773368,  0.6755902076,  0.0580000000],
        [-1.0, -0.0,          -0.0,          0.0250000000],
        [-0.0, -0.6755902076, -0.7372773368, -0.0300000000],
        [ 0.0,  0.0,           0.0,           1.0],
    ],
    dtype=float,
)

# Default start EE pose in base frame (your example)
def make_T_base_ee_default() -> np.ndarray:
    theta = np.deg2rad(0.0)
    Rz0 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 0.0],
            [np.sin(theta),  np.cos(theta), 0.0, 0.0],
            [0.0,            0.0,           1.0, 0.0],
            [0.0,            0.0,           0.0, 1.0],
        ],
        dtype=float,
    )
    A = np.array(
        [
            [0, 0, -1, 0.1],
            [0, 1,  0, -0.4],
            [1, 0,  0, -0.2],
            [0, 0,  0,  1.0],
        ],
        dtype=float,
    )
    return Rz0 @ A

T_BASE_EE = make_T_base_ee_default()

BRIDGE_STEPS = 10

# ---- Anygrasp pose file (produced by your pipeline) ----
# Put it wherever your inference writes it (repo root works fine if your code saves there)
GRASP_POSE_PATH = os.path.join(VDBOT_ROOT, "grasp_pose.npy")

# ---- Convention fix ----
# You said: "anygrasp pose is good but z is flipped"
# Flipping only Z is not a proper rotation (det=-1), so we do a 180° rotation about Y:
# Ry(pi) flips X and Z, keeps Y intact, and has det=+1.
APPLY_Z_FLIP_FIX = True


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]

def lerp_positions(p0: np.ndarray, p1: np.ndarray, num_steps: int) -> np.ndarray:
    p0 = np.asarray(p0, float).reshape(3)
    p1 = np.asarray(p1, float).reshape(3)
    alphas = np.linspace(0.0, 1.0, num_steps + 2)[1:-1]
    return (1 - alphas)[:, None] * p0[None, :] + alphas[:, None] * p1[None, :]

def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, float).reshape(3)
    return T

def find_latest_prediction_npz() -> str:
    files = glob.glob(os.path.join(PRED_DIR, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No prediction .npz found in {PRED_DIR}")
    files.sort(key=os.path.getmtime)
    return files[-1]

def squeeze_pred(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    pred = np.squeeze(pred)
    if pred.ndim == 3 and pred.shape[-1] == 3:
        return pred
    raise ValueError(f"Unexpected pred_trajectories shape after squeeze: {pred.shape}")

def squeeze_loss(loss: np.ndarray) -> np.ndarray:
    loss = np.asarray(loss)
    loss = np.squeeze(loss)
    if loss.ndim == 1:
        return loss
    raise ValueError(f"Unexpected loss shape after squeeze: {loss.shape}")

def load_pred_and_loss_from_prediction_file(pred_npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
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

def load_T_any(path: str) -> np.ndarray:
    """Accept (4,4) or (1,4,4)."""
    T = np.load(path)
    T = np.asarray(T, dtype=float)
    if T.shape == (1, 4, 4):
        T = T[0]
    if T.shape != (4, 4):
        raise ValueError(f"{path} must be (4,4) or (1,4,4); got {T.shape}")
    return T

def rot_y_pi() -> np.ndarray:
    """180° around Y: flips X and Z, keeps Y intact, det=+1."""
    return np.array(
        [
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ],
        dtype=float,
    )

def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix to the closest valid SO(3) rotation."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    # ensure det +1
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn

def get_constant_grasp_R_base() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads grasp_pose.npy (camera frame), converts to base,
    makes rotation a proper SO(3), applies convention fix, returns:
      - T_grasp_base (4,4)
      - R_grasp_base (3,3)
    """
    if not os.path.exists(GRASP_POSE_PATH):
        raise FileNotFoundError(f"Missing {GRASP_POSE_PATH}")

    T_grasp_cam = load_T_any(GRASP_POSE_PATH)
    T_grasp_base = T_BASE_CAM @ T_grasp_cam

    R = T_grasp_base[:3, :3].copy()
    R = project_to_so3(R)  # <-- key robustness step

    if APPLY_Z_FLIP_FIX:
        R = R @ rot_y_pi()     # proper 180° about Y: flips Z (approach) without flipping Y
        R = project_to_so3(R)  # keep it clean

    T_grasp_base[:3, :3] = R
    return T_grasp_base, R


def build_pre_post(pred_cam: np.ndarray, loss: np.ndarray, R_const: np.ndarray):
    best_idx = int(np.argmin(loss))
    best_traj_cam = pred_cam[best_idx]  # (H,3)

    # cam -> base for positions
    best_traj_base = transform_points(T_BASE_CAM, best_traj_cam)

    ee_start = T_BASE_EE[:3, 3]
    traj_start = best_traj_base[0]

    bridge = lerp_positions(ee_start, traj_start, BRIDGE_STEPS)

    # PRE: go to traj start
    pre_pos = np.vstack([
        ee_start[None, :],
        bridge,
        traj_start[None, :],
    ])

    # POST: execute the predicted path
    post_pos = best_traj_base.copy()

    # IMPORTANT: use R_const (AnyGrasp) for every waypoint
    pre_T = np.stack([make_T(R_const, p) for p in pre_pos], axis=0)
    post_T = np.stack([make_T(R_const, p) for p in post_pos], axis=0)

    return best_idx, float(loss[best_idx]), pre_T, post_T

def run_infer_affordance(obj: str, action: str, visualize: bool = False):
    cmd = [
        "python3",
        os.path.join(VDBOT_ROOT, "demos", "infer_affordance.py"),
        "-d", DATASET_NAME,
        "-f", FRAME_ID,
        "-o", obj,
        "-i", action,
        "--use_graspnet",
    ]
    if visualize:
        cmd.append("-v")

    env = os.environ.copy()

    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = VDBOT_ROOT + (":" + old_pp if old_pp else "")

    proc = subprocess.run(
        cmd,
        cwd=VDBOT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"infer_affordance failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    return proc.stdout


# -----------------------------
# API
# -----------------------------
@app.post("/infer_and_return_trajectories")
async def infer_and_return_trajectories(
    object_name: str = Form(...),
    action: str = Form(...),
    rgb_png: UploadFile = File(...),
    depth_png: UploadFile = File(...),
    visualize: Optional[bool] = Form(False), 
):
    """
    Uploads:
      - rgb_png: 000000.png (8-bit)
      - depth_png: 000000.png (likely 16-bit mm)

    Saves to:
      datasets/reachy2/color/000000.png
      datasets/reachy2/depth/000000.png

    Runs inference and returns transforms with orientation from AnyGrasp:
      - pre_T  (T1,4,4)
      - post_T (T2,4,4)
      - T_grasp_base (4,4)
      - T_base_cam, T_base_ee
    """
    ensure_dirs()

    rgb_bytes = await rgb_png.read()
    depth_bytes = await depth_png.read()

    with open(COLOR_PATH, "wb") as f:
        f.write(rgb_bytes)
    with open(DEPTH_PATH, "wb") as f:
        f.write(depth_bytes)

    # Run vidbot inference (this should also write grasp_pose.npy)
    _stdout = run_infer_affordance(object_name, action, visualize=bool(visualize))

    pred_npz_path = find_latest_prediction_npz()
    pred, loss = load_pred_and_loss_from_prediction_file(pred_npz_path)

    # Load AnyGrasp pose (camera->base) and use its rotation along the path
    T_grasp_base, R_grasp_base = get_constant_grasp_R_base()

    best_idx, best_loss, pre_T, post_T = build_pre_post(pred, loss, R_grasp_base)

    # Pack a binary npz (recommended)
    buf = io.BytesIO()
    np.savez(
        buf,
        pre_T=pre_T,
        post_T=post_T,
        best_idx=np.array(best_idx),
        best_loss=np.array(best_loss),
        T_grasp_base=T_grasp_base,
        T_base_cam=T_BASE_CAM,
        T_base_ee=T_BASE_EE,
        z_flip_fix_applied=np.array(bool(APPLY_Z_FLIP_FIX)),
        prediction_file=np.array(pred_npz_path),
    )
    buf.seek(0)

    # Keep your "hex bytes" style response (debug-friendly)
    return {
        "prediction_file": pred_npz_path,
        "best_idx": best_idx,
        "best_loss": best_loss,
        "pre_len": int(pre_T.shape[0]),
        "post_len": int(post_T.shape[0]),
        "z_flip_fix_applied": bool(APPLY_Z_FLIP_FIX),
        "payload_npz_bytes_hex": buf.getvalue().hex(),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
