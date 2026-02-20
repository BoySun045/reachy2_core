# Running dualsense_speech_to_text

## Prerequisites (one-time setup)

```bash
# System dependencies
sudo apt install libportaudio2

# Python dependencies (system-wide, no venv — ROS 2 Humble requires system Python)
pip3 install sounddevice numpy openai-whisper ollama

# Install and pull the Ollama model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

## Build

```bash
source /opt/ros/humble/setup.bash
cd ~/marwan_ws
colcon build --packages-select dualsense_speech_to_text
source install/setup.bash
```

## Run

```bash
# Make sure Ollama is running
ollama serve &

# Launch the node
ros2 launch dualsense_speech_to_text dualsense_stt.launch.py
```

## Monitor output

```bash
# Raw transcription
ros2 topic echo /transcription

# Parsed command JSON (category, object, instruction, dx, dy, dyaw)
ros2 topic echo /command

# Routed topics
ros2 topic echo /pathplanner_manager/query    # navigation commands
ros2 topic echo /vidbot/trigger               # manipulation commands
ros2 topic echo /locomotion/command           # locomotion commands
```

## Tuning

Edit `config/params.yaml` or pass parameters at launch:

```bash
ros2 launch dualsense_speech_to_text dualsense_stt.launch.py \
  --ros-args -p energy_threshold:=0.02 -p silence_duration:=2.0
```
