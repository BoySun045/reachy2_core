# Reachy2 Core

This repository provides the core ROS 2 packages required to simulate, configure, and launch the **Reachy 2** humanoid robot. It includes URDF descriptions, launch files, Gazebo simulation assets, and control interfaces used across real and simulated platforms.

## Overview

The repo is structured as a multi-package ROS 2 workspace and serves as the foundation for developing and testing Reachy 2 behaviors, whether in hardware or in simulation.

### Included Packages

- **`reachy_bringup`**  
  Launch files and runtime orchestration for both real and simulated deployments.

- **`reachy_config`**  
  YAML-based configuration files for Reachy’s kinematics, dynamics, and URDF parameterization.

- **`reachy_controllers`**  
  ROS 2 control nodes for interfacing with Reachy hardware and simulated joints.

- **`reachy_description`**  
  Robot description files (URDF, meshes, xacro) for Reachy 2, used across simulation and visualization.

- **`reachy_fake`**  
  Fake interfaces for mimicking joint states and simulating Reachy's response without hardware.

- **`reachy_gazebo`**  
  Gazebo simulation environment for Reachy 2, including plugins and world integration.

- **`reachy_gazebo_gripper_glue`**  
  Bridges and constraints for integrating the gripper with Gazebo’s physics and control pipeline.

- **`reachy_utils`**
  Common utility functions and shared tools used by multiple packages.

- **`reachy_nav`**
  Navigation stack: TSDF generation, path planning (RRT with collision/reachability modes), pathplanner manager with object scene graph queries, and locomotion management.

- **`reachy_llm`**
  Voice command pipeline: VAD-based speech capture, Whisper transcription, rule-based + Ollama LLM command parsing, and multi-robot command routing.

- **`reachy_control_arm_and_wbc`**
  Arm control: end-effector IK (torso-frame and whole-body), DualSense teleop, arm on/off/up/down services, and VidBot manipulation manager.

- **`reachy_vidbot`**
  Visual manipulation: GroundingDINO + SAM object detection, grasp planning, and manipulation execution.

---

## Multi-Robot Support (Reachy + Spot)

The system supports two robots: **Reachy** (humanoid) and **Spot** (quadruped). Voice commands are parsed to identify which robot is being addressed, with "reachy" as the default.

### Architecture

```
Voice → [Whisper STT] → [Command Parser] → [STT Node Router]
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         │                        │                        │
                    navigation              manipulation              service
                         │                        │                        │
              /pathplanner_manager/query    /vidbot/trigger         direct ROS2
              (JSON: robot + object)     (JSON: robot + ...)     service calls
                         │                        │                        │
                   ┌─────┴─────┐            ┌─────┴─────┐          ┌───────┴───────┐
                   │           │            │           │          │               │
              ~/nav_goal  ~/spot/nav_goal  (reachy     (spot    /reachy/*      /spot/*
              (reachy)    (spot)            only)      filtered)
```

All managers receive commands on **shared topics** with a `robot` field in the JSON payload. Each manager routes to robot-specific output topics internally.

### Robot Name Detection

Whisper often mishears "reachy" as "richie", "ritchie", "reachie", etc. The command parser normalizes all these variants before processing. Robot name is extracted from the beginning of the command (e.g., "reachy go to the table", "spot stand up", "tell reachy to grab the cup").

---

## Voice Command Reference

### Navigation
Move the robot's base to a named object or location.

| Example phrase | Robot | Action |
|---|---|---|
| "go to the table" | reachy | Navigate to table |
| "spot navigate to the chair" | spot | Navigate to chair |
| "find the bottle" | reachy | Navigate to bottle |

### Manipulation
Use the robot's arm to interact with an object.

| Example phrase | Robot | Action |
|---|---|---|
| "pick up the red cup" | reachy | Grasp red cup |
| "grab the ball" | reachy | Grasp ball |
| "put the bottle on the shelf" | reachy | Place bottle |

### Locomotion
Move the robot's base by a relative amount.

| Example phrase | Robot | Action |
|---|---|---|
| "move forward 50 centimeters" | reachy | +0.5m dx |
| "turn left 90 degrees" | reachy | +90 deg yaw |
| "step right" | reachy | -0.5m dy |
| "turn around" | reachy | 180 deg turn |
| "stop" | reachy | Halt |

### Reachy Arm Services

| Example phrase | Service called |
|---|---|
| "turn on" / "arm on" / "wake up" | `/reachy/arm_on` |
| "arm up" / "ready position" | `/reachy/arm_up` |
| "arm down" / "rest position" | `/reachy/arm_down` |
| "turn off" / "arm off" / "shut down" | `/reachy/arm_off` |

### Spot Body Services

| Example phrase | Service called |
|---|---|
| "spot stand" / "spot get up" | `/spot/stand` |
| "spot sit" / "spot sit down" | `/spot/sit` |
| "spot stow arm" | `/spot/arm_stow` |
| "spot unstow arm" | `/spot/arm_unstow` |
| "spot open gripper" | `/spot/open_gripper` |
| "spot close gripper" | `/spot/close_gripper` |
| "spot power on" | `/spot/power_on` |
| "spot power off" | `/spot/power_off` |
| "spot claim" | `/spot/claim` |
| "spot kill" | `/spot/sit` + `/spot/rollover` |

---

## Navigation & Path Planning

### TSDF Generation

Generate a fused point cloud from posed RGB-D frames:

```bash
python3 reachy_nav/tsdf_generator_fromslam.py <path/to/posed_rgbd/>
```

Input folder must contain:
- `intrinsics.txt` (3x3 camera matrix)
- `frame_NNNNNN.jpg` (RGB images)
- `frame_NNNNNN.npy` (depth maps, float32 metres)
- `frame_NNNNNN.txt` (4x4 camera poses)

Outputs: `tsdf_fused.ply` and `tsdf_mesh.ply`.

### Path Planner Config

YAML config file (e.g., `pathplanner_config.yaml`):

```yaml
data_dir: "data_dso/2026_02_20-17_23_36-sfm_map1"
pcd_file: "tsdf_fused.ply"
reachability_file: "reachability.ply"
objects_file: "object_scene_graph/frame_final_objects.pkl.gz"
z_up: false
planning_mode: "reachability"   # or "collision"
```

**Planning modes:**
- `collision` — obstacle point cloud defines invalid voxels; uses polygon-stack validity checker; enforces `MAX_ARM_REACH` (0.8m) when selecting nav goals
- `reachability` — reachability point cloud defines valid voxels; picks closest reachable point to object without distance limit

### Running the Navigation Stack

```bash
# Pathplanner manager (object queries, nav goal publishing)
python3 reachy_nav/pathplanner_manager.py --config reachy_nav/pathplanner_config.yaml

# Path planner node (RRT solving, trajectory publishing)
python3 reachy_nav/path_planner_node.py --config reachy_nav/pathplanner_config.yaml [--mode collision|reachability]
```

### Per-Robot Topics

| Component | Reachy topic | Spot topic |
|---|---|---|
| Nav goal (manager out) | `/pathplanner_manager/nav_goal` | `/pathplanner_manager/spot/nav_goal` |
| Odometry (planner in) | `/slam/base_odom` | `/spot/odometry/corrected` |
| Planned path (planner out) | `/path_planner/trajectory` | `/spot/planned_path` |
| Locomotion trajectory | `/path_planner/trajectory` | `/spot/planned_path` |

---

## Arm Control Services

The `arm_on_off_service.py` node provides four services for Reachy's arm:

```bash
python3 reachy_control_arm_and_wbc/arm_on_off_service.py
```

| Service | Description |
|---|---|
| `/reachy/arm_on` | Reads current arm pose from TF, enables torque on both arms, runs gripper warmup |
| `/reachy/arm_up` | Moves arm to ready (up) pose |
| `/reachy/arm_down` | Moves arm to stow (down) pose |
| `/reachy/arm_off` | Disables torque on both arms |

Call manually:
```bash
ros2 service call /reachy/arm_on std_srvs/srv/Trigger
ros2 service call /reachy/arm_up std_srvs/srv/Trigger
ros2 service call /reachy/arm_down std_srvs/srv/Trigger
ros2 service call /reachy/arm_off std_srvs/srv/Trigger
```

---

## TF Utilities

Get the transform between any two TF frames:

```bash
python3 reachy_nav/get_tf.py <source_frame> <target_frame>
# Example: python3 reachy_nav/get_tf.py r_arm_tip torso
```

---

## Dependencies

Install ROS 2 and the dependencies listed in `requirements.txt` and `pyproject.toml`. You can also use the included `dependencies.sh` script to install system-wide dependencies required for simulation and control.

Additional Python dependencies for the new packages:
- `whisper` (OpenAI Whisper for STT)
- `ollama` (local LLM for command parsing)
- `open3d` (point cloud processing, TSDF fusion)
- `scipy` (transforms, spatial operations)
- `pyaudio` (microphone capture)

## License

This project is licensed under the **Apache License 2.0** – see the `LICENSE` file for details.