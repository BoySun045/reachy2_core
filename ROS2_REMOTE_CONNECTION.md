# Connecting to Reachy2 ROS2 from a Remote Machine

This guide explains how to enable ROS2 communication between a Reachy2 robot and a remote laptop/PC.

## Problem

By default, Reachy2's ROS2 nodes run inside a Docker container with CycloneDDS configured to only allow **localhost** peers. This prevents external machines from discovering and communicating with the robot's ROS2 topics.

## Prerequisites

- Reachy2 robot (tested with reachy2_core v1.7.5.9)
- Remote PC with Ubuntu 22.04 and ROS2 Humble
- Both machines on the same network

## Network Information (Example)

| Machine | IP Address |
|---------|------------|
| Reachy2 Robot | 192.168.1.71 |
| Remote PC | 192.168.1.130 |

---

## Part 1: Configure the Robot

### Step 1: SSH into the robot

```bash
ssh bedrock@192.168.1.71
```

### Step 2: Create CycloneDDS config file on the host

The robot's Docker container mounts `~/.reachy_config` to `/home/reachy/.reachy_config_override`. Create a CycloneDDS config file there:

```bash
cat > /home/bedrock/.reachy_config/cyclonedds.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
      <General>
          <AllowMulticast>false</AllowMulticast>
      </General>
      <Discovery>
          <ParticipantIndex>auto</ParticipantIndex>
          <Peers>
              <Peer Address="localhost"/>
              <Peer Address="192.168.1.130"/>  <!-- Your remote PC's IP -->
              <Peer Address="192.168.1.223"/> 
          </Peers>
          <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
      </Discovery>
  </Domain>
</CycloneDDS>
EOF
```

**Note:** Replace `192.168.1.130` with your remote PC's actual IP address.

### Step 3: Modify the Docker Compose file

Edit the compose file to use the new config:

```bash
sudo nano /pollen/docker/reachy2-core/compose.yaml
```

Find the environment section and uncomment/modify the CYCLONEDDS_URI line:

```yaml
environment:
  - ROS_DOMAIN_ID=$ROS_DOMAIN_ID
  - DISPLAY=$DISPLAY
  - "RCUTILS_CONSOLE_OUTPUT_FORMAT=[{severity}]: {message}"
  - REACHY2_CORE_SERVICE_FAKE=${REACHY2_CORE_SERVICE_FAKE:-false}
  # Add this line:
  - CYCLONEDDS_URI=/home/reachy/.reachy_config_override/cyclonedds.xml
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

### Step 4: Restart the reachy2-core service

```bash
sudo systemctl restart reachy2-core.service
```

### Step 5: Verify the configuration

```bash
# Check container is running
docker ps | grep core

# Verify environment variable is set
docker exec -it core env | grep CYCLONEDDS
# Should show: CYCLONEDDS_URI=/home/reachy/.reachy_config_override/cyclonedds.xml

# Verify config file is accessible
docker exec -it core cat /home/reachy/.reachy_config_override/cyclonedds.xml
```

---

## Part 2: Configure the Remote PC

### Step 1: Install ROS2 Humble (if not installed)

```bash
# Follow official ROS2 Humble installation guide
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
```

### Step 2: Install CycloneDDS

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

### Step 3: Create CycloneDDS config file

```bash
cat > ~/cyclonedds.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <Peers>
        <Peer Address="localhost"/>
        <Peer address="192.168.1.71"/>  <!-- Robot's IP -->
      </Peers>
      <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
```

**Note:** Replace `192.168.1.71` with your robot's actual IP address.

### Step 4: Set environment variables

Add these to your `~/.bashrc` for persistence:

```bash
echo '# ROS2 Reachy Connection
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml' >> ~/.bashrc

source ~/.bashrc
```

Or set them manually each session:

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```

### Step 5: Clear ROS2 daemon cache and test

```bash
ros2 daemon stop
ros2 daemon start
ros2 topic list
```

You should now see all the robot's topics!

---

## Verification

### On remote PC, list topics:

```bash
ros2 topic list
```

Expected output (partial):
```
/joint_states
/cmd_vel
/tf
/tf_static
/camera/color/image_raw
/scan
...
```

### Echo a topic:

```bash
ros2 topic echo /joint_states --once
```

### Check topic info:

```bash
ros2 topic info /cmd_vel
```

---

## Usage Examples

### Keyboard Teleoperation

```bash
sudo apt install ros-humble-teleop-twist-keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Visualize with RViz

```bash
ros2 run rviz2 rviz2
```

### Visualize with Foxglove Studio

1. Install Foxglove Studio: https://foxglove.dev/download
2. Connect to: `ws://192.168.1.71:8765` (if foxglove_bridge is running on robot)

---

## Troubleshooting

### Can't see topics

1. **Check network connectivity:**
   ```bash
   ping 192.168.1.71
   ```

2. **Verify ROS_DOMAIN_ID matches on both machines:**
   ```bash
   echo $ROS_DOMAIN_ID  # Should be 0 on both
   ```

3. **Verify RMW implementation:**
   ```bash
   echo $RMW_IMPLEMENTATION  # Should be rmw_cyclonedds_cpp on both
   ```

4. **Clear daemon cache:**
   ```bash
   ros2 daemon stop
   ros2 daemon start
   ```

5. **Check firewall (disable temporarily for testing):**
   ```bash
   sudo ufw disable
   ```

### Robot config was reset after restart

The Docker container recreates from the image on restart. Make sure:
1. The compose.yaml has the `CYCLONEDDS_URI` environment variable set
2. The cyclonedds.xml file exists in `/home/bedrock/.reachy_config/`

### Test from another container on the robot

To verify the robot's ROS2 is accessible:

```bash
# On the robot
docker run -it --rm --network host pollenrobotics/reachy2_core:1.7.5.9_release bash

# Inside the container
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic list
```

---

## Architecture Overview

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│  Remote PC (192.168.1.130)          │     │  Reachy2 Robot (192.168.1.71)       │
│                                     │     │                                     │
│  ┌─────────────────────────────┐    │     │  ┌─────────────────────────────┐    │
│  │  ROS2 Node                  │    │     │  │  Docker: core               │    │
│  │                             │    │     │  │                             │    │
│  │  CYCLONEDDS_URI:            │    │     │  │  CYCLONEDDS_URI:            │    │
│  │  - Peer: 192.168.1.71       │◄───┼─────┼──│  - Peer: localhost          │    │
│  │                             │    │     │  │  - Peer: 192.168.1.130      │    │
│  └─────────────────────────────┘    │     │  └─────────────────────────────┘    │
│                                     │     │                                     │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
                          UDP Ports 7400-7500
```

---

## Key Files

| Location | Purpose |
|----------|---------|
| `/home/bedrock/.reachy_config/cyclonedds.xml` | Robot's CycloneDDS peer config (host) |
| `/pollen/docker/reachy2-core/compose.yaml` | Docker compose with CYCLONEDDS_URI |
| `~/cyclonedds.xml` | Remote PC's CycloneDDS peer config |

---

## References

- [CycloneDDS Configuration](https://cyclonedds.io/docs/cyclonedds/latest/config/index.html)
- [ROS2 DDS Tuning](https://docs.ros.org/en/humble/How-To-Guides/DDS-tuning.html)
- [Reachy2 Documentation](https://docs.pollen-robotics.com/)
