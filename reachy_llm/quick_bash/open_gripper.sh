#!/bin/bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

ros2 service call /spot/open_gripper std_srvs/srv/Trigger
