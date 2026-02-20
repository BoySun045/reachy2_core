# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VidBot (CVPR 2025) learns generalizable 3D robotic manipulation actions from 2D human videos for zero-shot execution. It predicts end-effector trajectories and contact points from RGB-D images conditioned on natural language instructions.

Paper: https://arxiv.org/abs/2503.07135

## Setup Commands

```bash
# Environment
conda create -n vidbot python=3.10.9
conda activate vidbot
pip install pytorch-lightning==1.8.6
pip install -r requirements.txt

# PyTorch Scatter (required, separate install)
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.1%2Bpt113cu117-cp310-cp310-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt113cu117-cp310-cp310-linux_x86_64.whl

# Pretrained weights and demo data
sh scripts/download_ckpt_testdata.sh

# (Optional) Third-party modules (GroundingDINO, EfficientSAM, GraspNet)
sh scripts/prepare_third_party_modules.sh
```

**Pinned versions**: PyTorch 1.13.1+cu117, `transformers==4.26.1` (GroundingDINO install may change this — reinstall if needed), `numpy==1.26.3`.

## Running Inference

```bash
# Run all demos (shuffled order)
bash scripts/test_demo.sh

# Single inference
python demos/infer_affordance.py \
  --config ./config/test_config.yaml \
  --dataset YOUR_DATASET_NAME \
  --frame FRAME_ID \
  --instruction "pickup sponge" \
  --object "sponge" \
  --visualize

# With GraspNet gripper pose estimation
python demos/infer_affordance.py ... --use_graspnet

# Load cached results (skip expensive detection/prediction)
python demos/infer_affordance.py ... --load_results --no_save

# Skip stages for faster iteration
python demos/infer_affordance.py ... --skip_coarse_stage --load_results  # requires cached results
python demos/infer_affordance.py ... --skip_fine_stage --load_results
```

## Architecture

### Three-Stage Pipeline

1. **Detection**: Open-vocabulary object detection via GroundingDINO + EfficientSAM segmentation. Results cached in `datasets/{name}/scene_meta/`.
2. **Coarse Stage**: `ContactPredictor` (ResNet50 + Perceiver) predicts contact points; `GoalPredictor` (VAE encoder-decoder) predicts goal end-effector positions. Both conditioned on CLIP text embeddings.
3. **Fine Stage**: `DiffuserModel` (temporal U-Net) generates trajectories via diffusion with guidance losses (goal-reaching, collision avoidance, surface normal alignment). Trajectories smoothed via Savitzky-Golay filtering.

### Module Layout

- **`algos/`** — Inference orchestration. `AffordanceInferenceEngine` in `afford_algos.py` is the main entry point that loads all models and runs the pipeline. Lightning modules for each predictor in `contact_algos.py`, `goal_algos.py`, `traj_algos.py`. `traj_optimizer.py` handles multi-frame trajectory optimization with scale/pose refinement.
- **`models/`** — Neural network architectures. `diffuser.py` (trajectory diffusion), `contact.py` (contact prediction), `goal.py` (goal prediction), `temporal.py` (U-Net backbone), `perceiver.py` (cross-attention), `clip/` (CLIP encoder, vendored).
- **`diffuser_utils/`** — Guidance losses (`guidance_loss.py`), action-specific guidance parameters (`guidance_params.py`), dataset utilities (`dataset_utils.py` for TSDF construction, 3D backprojection, CLIP encoding, RANSAC voting).
- **`demos/`** — Entry-point scripts. `infer_affordance.py` is the main CLI. `optimize_affordance.py` handles multi-frame trajectory optimization using COLMAP results.
- **`config/`** — YAML configs (OmegaConf). `test_config.yaml` points to model checkpoints and third-party module paths.
- **`pretrained/`** — Model checkpoints (`contact/`, `goal/`, `traj/` each with `config.yaml` + `final.ckpt`).

### Deployment Files (root)

- `vidbot_server.py` — FastAPI server wrapping VidBot inference for HTTP-based deployment. Saves RGB-D frames to disk, runs `infer_affordance.py` as subprocess, returns predicted trajectories.
- `vidbot_server_graspnet.py` — Variant with GraspNet integration.
- `vidbot_ros_node.py` / `vidbot_ros_graspnet_node.py` — ROS nodes for robot integration.
- `vidbot_manager.py`, `vidbot_prompt_node.py` — Supporting ROS orchestration.

### Key Patterns

- **`data_batch` dict as central state**: All pipeline stages read from and write to a shared `data_batch` dictionary. Detection populates `bbox_all`, `object_mask_all`, etc. Contact prediction adds `start_pos`, `contact_pix`. Goal prediction adds `end_pos`, `normal_sign`. Trajectory prediction adds `pred_trajectories`. The `update_outputs_to_databatch()` method merges stage outputs back.
- **Results serialization**: `AffordanceInferenceEngine.export_results()` / `load_results()` save/load `data_batch` as compressed NPZ files. String values get special encoding (`_singlestr` / `_strlist` suffixes). Load with `--load_results` to skip expensive stages.
- All models are PyTorch Lightning modules (`pl.LightningModule`), loaded via `load_from_checkpoint()`.
- Language conditioning uses CLIP ViT-B/16 throughout; the global `VLM` model is loaded once in `infer_affordance.py` and passed to `encode_action()`.
- 3D scene representation uses TSDF volumes built from RGB-D via backprojection.
- GraspNet integration is optional — the system falls back to heuristic normal-based grasp poses when unavailable.
- Third-party modules are added via `sys.path.append("./third_party/...")` in `afford_algos.py`, not installed as packages.

### Guidance Parameters

Action-specific guidance parameters in `diffuser_utils/guidance_params.py` control trajectory generation quality. Three parameter sets map to action categories:
- **PARAMS1** (open, close, pull, push, press): Higher normal weight, lower voxel resolution, excludes object points.
- **PARAMS2** (pick, take, put, place, drop): Higher voxel resolution (128), includes object points. Also used as default for unknown actions.
- **PARAMS3** (wipe, move): Highest noncollide/contact weights, no normal guidance.

To add a new action verb, add an entry to `GUIDANCE_PARAMS_DICT` mapping to PARAMS1/2/3 or a custom dict. Instructions must be in "verb object" format (e.g., "pickup sponge").

### Data Format

Datasets live in `datasets/{name}/` with:
- `camera_intrinsic.json` — column-major 3x3 intrinsic matrix, width, height
- `color/` — RGB frames as `000000.png`, `000001.png`, ...
- `depth/` — Depth maps as PNGs (same naming), values in millimeters (divided by 1000 at load time)

Recommended resolution: 1280x720 (GraspNet requires exactly this).

Results cached as NPZ files in `datasets/{name}/scene_meta/` (detection) and `datasets/{name}/prediction/` (trajectories).
