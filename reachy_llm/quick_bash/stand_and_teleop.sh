#!/bin/bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

echo "[spot] Standing..."
ros2 service call /spot/stand std_srvs/srv/Trigger

echo "[spot] Starting teleop (WASD/arrow keys publish to /spot/cmd_vel)..."
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/spot/cmd_vel
