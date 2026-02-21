#!/bin/bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

echo "[spot] Claiming..."
ros2 service call /spot/claim std_srvs/srv/Trigger
echo "[spot] Powering on..."
ros2 service call /spot/power_on std_srvs/srv/Trigger
