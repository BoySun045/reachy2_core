#!/bin/bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

echo "[spot] Sitting..."
ros2 service call /spot/sit std_srvs/srv/Trigger
echo "[spot] Rolling over..."
ros2 service call /spot/rollover std_srvs/srv/Trigger