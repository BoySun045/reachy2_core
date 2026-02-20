#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import glob
import subprocess
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

app = FastAPI(title="Vidbot Inference Server")

# -----------------------------
# CONFIG (edit these paths once)
# -----------------------------

# anygrasp graspgen edgegraspnet
VDBOT_ROOT = os.path.abspath(os.path.dirname(__file__))  # assumes script lives in vdbot_t/
DATASET_NAME = "reachy2"
DATASET_DIR = os.path.join(VDBOT_ROOT, "datasets", DATASET_NAME)

COLOR_DIR = os.path.join(DATASET_DIR, "color")
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
PRED_DIR = os.path.join(DATASET_DIR, "prediction")

# The exact frame name convention you asked for:
FRAME_ID = "000000"
COLOR_PATH = os.path.join(COLOR_DIR, f"{FRAME_ID}.png")
DEPTH_PATH = os.path.join(DEPTH_DIR, f"{FRAME_ID}.png")

# Your transforms (must match your earlier working scripts)
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
def make_T_base_ee_default():
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
R_PATH = T_BASE_EE[:3, :3]

BRIDGE_STEPS = 10


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

def transform_points(T, pts):
    pts = np.asarray(pts, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]

def lerp_positions(p0, p1, num_steps):
    p0 = np.asarray(p0, float).reshape(3)
    p1 = np.asarray(p1, float).reshape(3)
    alphas = np.linspace(0.0, 1.0, num_steps + 2)[1:-1]
    return (1 - alphas)[:, None] * p0[None, :] + alphas[:, None] * p1[None, :]

def make_T(R, p):
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

def squeeze_pred(pred):
    pred = np.asarray(pred)
    pred = np.squeeze(pred)
    # expected (N,H,3)
    if pred.ndim == 3 and pred.shape[-1] == 3:
        return pred
    raise ValueError(f"Unexpected pred_trajectories shape after squeeze: {pred.shape}")

def squeeze_loss(loss):
    loss = np.asarray(loss)
    loss = np.squeeze(loss)
    if loss.ndim == 1:
        return loss
    raise ValueError(f"Unexpected loss shape after squeeze: {loss.shape}")

def load_pred_and_loss_from_prediction_file(pred_npz_path: str):
    z = np.load(pred_npz_path, allow_pickle=True)

    pred = None
    loss = None

    # If the file stores them directly
    if "pred_trajectories" in z.files:
        pred = z["pred_trajectories"]
    if "guide_losses-total_loss" in z.files:
        loss = z["guide_losses-total_loss"]

    # Otherwise assume sibling npy files exist in the same folder
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


def build_pre_post(pred_cam: np.ndarray, loss: np.ndarray):
    best_idx = int(np.argmin(loss))
    best_traj_cam = pred_cam[best_idx]  # (H,3)

    # cam -> base
    best_traj_base = transform_points(T_BASE_CAM, best_traj_cam)

    # We define "grasp pose" = first waypoint of predicted trajectory
    ee_start = T_BASE_EE[:3, 3]
    traj_start = best_traj_base[0]

    # Smooth bridge from current EE pose to traj start
    bridge = lerp_positions(ee_start, traj_start, BRIDGE_STEPS)

    # PRE: go to traj start (include start and end)
    pre_pos = np.vstack([
        ee_start[None, :],        # current/default EE pose
        bridge,                   # optional intermediate points
        traj_start[None, :],      # ensure we end exactly at traj start
    ])

    # POST: execute full predicted trajectory from the start
    post_pos = best_traj_base.copy()

    pre_T = np.stack([make_T(R_PATH, p) for p in pre_pos], axis=0)
    post_T = np.stack([make_T(R_PATH, p) for p in post_pos], axis=0)

    return best_idx, float(loss[best_idx]), pre_T, post_T

def run_infer_affordance(obj: str, action: str, visualize: bool = False):
    cmd = [
        "python3",
        os.path.join(VDBOT_ROOT, "demos", "infer_affordance.py"),
        "-d", DATASET_NAME,
        "-f", FRAME_ID,
        "-o", obj,
        "-i", action,
    ]
    if visualize:
        cmd.append("-v")

    env = os.environ.copy()

    # Ensure the repo root is on PYTHONPATH so imports like diffuser_utils work
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = VDBOT_ROOT + (":" + old_pp if old_pp else "")

    proc = subprocess.run(
        cmd,
        cwd=VDBOT_ROOT,          # run from repo root
        env=env,                 # <-- important
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
    Reachy uploads:
      - rgb_png: 000000.png (8-bit color)
      - depth_png: 000000.png (typically 16-bit depth in mm, whatever your pipeline expects)

    Server saves them to:
      datasets/reachy2/color/000000.png
      datasets/reachy2/depth/000000.png

    Runs inference and returns an npz payload containing:
      - pre_T  (T1,4,4)
      - post_T (T2,4,4)
      - best_idx, best_loss
    """
    ensure_dirs()

    rgb_bytes = await rgb_png.read()
    depth_bytes = await depth_png.read()

    # Save bytes exactly as PNG to the required locations
    with open(COLOR_PATH, "wb") as f:
        f.write(rgb_bytes)
    with open(DEPTH_PATH, "wb") as f:
        f.write(depth_bytes)

    # Run vidbot inference
    _stdout = run_infer_affordance(object_name, action, visualize=bool(visualize))

    # Find the latest prediction file and load trajectories
    pred_npz_path = find_latest_prediction_npz()
    pred, loss = load_pred_and_loss_from_prediction_file(pred_npz_path)
    best_idx, best_loss, pre_T, post_T = build_pre_post(pred, loss)

    # Return as bytes: an npz file in response body
    buf = io.BytesIO()
    np.savez(buf, pre_T=pre_T, post_T=post_T, best_idx=np.array(best_idx), best_loss=np.array(best_loss))
    buf.seek(0)

    return {
        "prediction_file": pred_npz_path,
        "best_idx": best_idx,
        "best_loss": best_loss,
        "pre_len": int(pre_T.shape[0]),
        "post_len": int(post_T.shape[0]),
        "payload_npz_b64": None,  # unused (kept simple)
        "payload_npz_bytes": buf.getvalue().hex(),  # raw bytes as hex for simplicity
    }

# If you prefer actual binary file response instead of hex, tell me and I’ll switch it to StreamingResponse.
# Hex is easy to debug but ~2x larger.

if __name__ == "__main__":
    # accessible from other machines
    uvicorn.run(app, host="0.0.0.0", port=9000)
