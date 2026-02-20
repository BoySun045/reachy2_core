# Running reachy_control_arm_and_wbc

## Prerequisites

```bash
pip3 install numpy scipy
source /opt/ros/humble/setup.bash
```

---

## Offline Tools (run once to prepare data)

### Sample arm workspace (generates CSV + PLY)

```bash
python3 ee_pose_sampler.py --ros-args -p num_samples:=5000 -p arm:=right
```

### Dense orientation sweep

```bash
python3 sweep_orientations.py --ros-args -p n_positions:=500 -p wrist_steps:=4
```

### Build reachability graph from CSV

```bash
python3 build_reachability_graph.py --csv ee_poses_right.csv --radius 0.05
```

### Visualize workspace point cloud in RViz

```bash
python3 publish_pointcloud.py ee_pointcloud_right.ply odom
```

---

## Controllers (pick one)

### Whole-body controller (workspace lookup)

```bash
python3 wholebody_controller.py --ros-args -p csv_path:=ee_poses_right.csv
```

- Subscribes to `/target_ee_pose_world` (PoseStamped, odom frame) and `/odom`
- Publishes `/cmd_vel` and `/r_arm_forward_position_controller/commands`

### Whole-body IK controller (damped least-squares)

```bash
python3 wholebody_ik_controller.py
```

- Subscribes to `/target_ee_pose_world`, `/odom`, `/joint_states`
- Publishes `/cmd_vel` and `/r_arm_forward_position_controller/commands`

### Torso IK controller (batch pre-solving, fast)

```bash
python3 torso_ik_controller.py
```

- Subscribes to `/target_ee_pose_torso` (torso frame), `/odom`, `/joint_states`
- Also accepts waypoint paths via `/target_ee_path_torso` (PoseArray)
- Publishes `/cmd_vel` and `/r_arm_forward_position_controller/commands`

### EE pose commander (CSV fallback to nearest feasible)

```bash
python3 ee_pose_commander.py --ros-args -p csv_path:=ee_poses_right.csv -p arm:=right
```

- Subscribes to `/target_ee_pose`
- Publishes `/r_arm_forward_position_controller/commands`

---

## Teleoperation

### DualSense full teleop (base only)

```bash
python3 dualsense_teleop.py reachy
# or for Spot:
python3 dualsense_teleop.py spot
```

- Left stick = linear X/Y, right stick X = angular Z
- R1 = dead man's switch, L1/R1 = speed adjust

### DualSense base-only teleop

```bash
python3 dualsense_base_teleop.py
```

### DualSense arm teleop (torso frame)

```bash
python3 dualsense_arm_teleop.py
```

- Left/right sticks for position, D-pad for orientation, L1 for speed
- Publishes to `/target_ee_pose_torso`

---

## Grasping (VidBot pipeline)

### VidBot inference node (affordance + GraspNet)

```bash
python3 vidbot_ros_graspnet_node.py
```

- Subscribes to `/camera/color/image_raw`, `/camera/depth/image_raw`, `/vidbot/trigger`
- Publishes grasp trajectories on `/vidbot/pre_poses` and `/vidbot/post_poses`

### VidBot grasp manager (executes grasp sequence)

```bash
python3 vidbot_manager.py
```

- Subscribes to `/vidbot/pre_poses`, `/vidbot/post_poses`, `/odom`, `/joint_states`, `/vidbot/trigger`
- Publishes waypoints on `/target_ee_path_torso` and gripper commands

---

## Testing

### Publish alternating test targets for torso IK controller

```bash
python3 test_torso_target_pub.py
```

---

## Typical launch order (arm manipulation)

```bash
# Terminal 1 — controller
python3 torso_ik_controller.py

# Terminal 2 — VidBot inference
python3 vidbot_ros_graspnet_node.py

# Terminal 3 — grasp manager
python3 vidbot_manager.py
```

## Typical launch order (teleop)

```bash
# Terminal 1 — controller
python3 torso_ik_controller.py

# Terminal 2 — arm teleop
python3 dualsense_arm_teleop.py

# Terminal 3 — base teleop (optional, if also driving)
python3 dualsense_base_teleop.py
```

## Monitor topics

```bash
ros2 topic echo /cmd_vel
ros2 topic echo /r_arm_forward_position_controller/commands
ros2 topic echo /wholebody_feedback
ros2 topic echo /target_ee_pose_torso
```
