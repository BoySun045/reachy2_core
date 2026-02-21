#!/bin/bash
BAG_NAME="${1:-spot}_$(date +%Y%m%d_%H%M%S)"

ros2 bag record -o "$BAG_NAME" \
  /spot/camera/back/camera_info \
  /spot/camera/back/image \
  /spot/camera/frontleft/camera_info \
  /spot/camera/frontleft/image \
  /spot/camera/frontright/camera_info \
  /spot/camera/frontright/image \
  /spot/camera/hand/camera_info \
  /spot/camera/hand/image \
  /spot/camera/left/camera_info \
  /spot/camera/left/image \
  /spot/camera/right/camera_info \
  /spot/camera/right/image \
  /spot/cmd_vel \
  /spot/depth/back/camera_info \
  /spot/depth/back/image \
  /spot/depth/frontleft/camera_info \
  /spot/depth/frontleft/image \
  /spot/depth/frontright/camera_info \
  /spot/depth/frontright/image \
  /spot/depth/hand/camera_info \
  /spot/depth/hand/image \
  /spot/depth/left/camera_info \
  /spot/depth/left/image \
  /spot/depth/right/camera_info \
  /spot/depth/right/image \
  /spot/depth_registered/back/camera_info \
  /spot/depth_registered/back/image \
  /spot/depth_registered/frontleft/camera_info \
  /spot/depth_registered/frontleft/image \
  /spot/depth_registered/frontright/camera_info \
  /spot/depth_registered/frontright/image \
  /spot/depth_registered/hand/camera_info \
  /spot/depth_registered/hand/image \
  /spot/depth_registered/left/camera_info \
  /spot/depth_registered/left/image \
  /spot/depth_registered/right/camera_info \
  /spot/depth_registered/right/image \
  /spot/odometry \
  /tf \
  /tf_static
