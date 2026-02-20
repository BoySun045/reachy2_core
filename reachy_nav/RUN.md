# Running reachy_nav

## Prerequisites

```bash
pip3 install open3d numpy scipy
```

For localization (hloc):
```bash
pip3 install torch torchvision
# hloc must be in third_party/hloc (clone separately)
```

For occupancy planner:
```bash
pip3 install ompl  # or build from source
```

---

## Offline Tools (run once to prepare data)

### 1. Extract images from a ROS2 bag

```bash
python3 extract_bag_images.py /path/to/rosbag2_folder [output_dir]
```

### 2. Generate TSDF scene from RGB-D + poses

```bash
python3 tsdf_generator.py
```

Outputs `tsdf_fused.ply` and `tsdf_mesh.ply` in the data directory.

### 3. Build hloc localization map

```bash
python3 build_hloc_map.py
```

Runs the full pipeline: COLMAP model → SuperPoint features → NetVLAD descriptors → LightGlue matching → triangulation. Outputs to `data_dso/.../hloc/`.

---

## ROS2 Nodes

All nodes are run directly with `python3`. Source ROS2 first:

```bash
source /opt/ros/humble/setup.bash
```

### Scene manager (publishes scene + handles object queries)

```bash
python3 pathplanner_manager.py
```

- Publishes TSDF scene and reachability point clouds
- Accepts object queries via terminal or `/pathplanner_manager/query` topic
- Publishes nav goals to `/pathplanner_manager/nav_goal`

### Path planner (RRT-based)

```bash
python3 path_planner_node.py
```

- Subscribes to `/pathplanner_manager/nav_goal` and `/slam/base_odom`
- Publishes planned trajectory on `~/trajectory`

### Localization

```bash
python3 localization_node.py
```

- Subscribes to `/camera/color/image_raw` and `/camera/color/camera_info`
- Publishes localized pose on `~/robot_pose`

### Locomotion manager (LLM command → nav goal)

```bash
python3 locomotion_manager.py
```

- Subscribes to `/slam/base_odom` and `/locomotion/command` (JSON: `{"dx", "dy", "dyaw"}`)
- Publishes 2-pose trajectory on `/occ_planner/planned_path`

### Trajectory follower (waypoint controller)

```bash
python3 trajectory_follower_node.py
```

- Subscribes to `/path_planner/trajectory` and `/localization/robot_pose`
- Publishes velocity commands on `/cmd_vel`

### Pure pursuit controller (alternative to trajectory follower)

```bash
python3 pure_pursuit_ros2.py
# With parameter overrides:
python3 pure_pursuit_ros2.py --ros-args -p Vcmd:=0.5 -p Lfw:=0.8
```

- Subscribes to `/odom` and `/occ_planner/planned_path`
- Publishes velocity commands on `/cmd_vel`

### Occupancy planner (OMPL service)

```bash
python3 occ_planner_ros2.py
```

- Service: `~/plan_path` (nav_msgs/srv/GetPlan)
- Publishes planned path on `~/planned_path`
- Example call:
  ```bash
  ros2 service call /occ_planner/plan_path nav_msgs/srv/GetPlan \
    "{goal: {header: {frame_id: 'map'}, pose: {position: {x: 2, y: 2, z: 0}, orientation: {w: 1.0}}}}"
  ```

---

## Typical launch order

```bash
# Terminal 1 — scene + object queries
python3 pathplanner_manager.py

# Terminal 2 — path planner
python3 path_planner_node.py

# Terminal 3 — localization
python3 localization_node.py

# Terminal 4 — trajectory follower
python3 trajectory_follower_node.py

# Terminal 5 — locomotion manager (for LLM voice commands)
python3 locomotion_manager.py
```

## Monitor topics

```bash
ros2 topic echo /pathplanner_manager/nav_goal
ros2 topic echo /path_planner/trajectory
ros2 topic echo /cmd_vel
ros2 topic echo /localization/robot_pose
```
