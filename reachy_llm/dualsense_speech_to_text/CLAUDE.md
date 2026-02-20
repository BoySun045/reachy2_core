# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ROS 2 Humble Python package for VAD-based speech-to-text. A push-to-talk microphone with a hardware mute button controls when audio flows. Energy-based voice activity detection automatically detects speech onset and silence. Whisper transcribes speech, then a local LLM (Ollama) parses it into structured `{object, instruction}` commands.

## Environment Setup

```bash
# Python dependencies (system-wide — ament_python hardcodes #!/usr/bin/python3)
pip3 install sounddevice numpy openai-whisper ollama
sudo apt install libportaudio2

# Ollama (local LLM for command parsing)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

**Note:** Do not use a venv — ROS 2 Humble's `ament_python` build type hardcodes the system Python shebang in entry points.

## Build & Run

```bash
source /opt/ros/humble/setup.bash

# Build (from workspace root ~/marwan_ws)
colcon build --packages-select dualsense_speech_to_text
source install/setup.bash

# Run
ros2 launch dualsense_speech_to_text dualsense_stt.launch.py

# Monitor topics
ros2 topic echo /transcription   # raw text
ros2 topic echo /command          # parsed JSON: {"object": "...", "instruction": "..."}
```

## Architecture

Single ROS 2 node (`SpeechToTextNode` in `stt_node.py`) with three components:

- **`audio_capture.py`** — Continuously monitors microphone via sounddevice. Tracks RMS energy of every audio chunk. `start_recording()` / `stop_recording()` control chunk accumulation. Resamples to 16kHz for Whisper on stop.
- **`transcriber.py`** — Wrapper around `whisper.load_model()` / `model.transcribe()`. Ignores audio < 0.3s.
- **`command_parser.py`** — Sends transcribed text to a local Ollama LLM to extract `{"object", "instruction"}` JSON. Few-shot prompted. Falls back to `None` on error.

**VAD state machine:**
```
IDLE → (energy > threshold) → RECORDING → (silence > duration) → PROCESSING → IDLE
```

**Data flow:**
```
Mic unmuted → VAD detects speech → record → silence detected → Whisper STT → Ollama parse → publish
```

**Published topics:**
- `/transcription` (`std_msgs/String`) — raw transcribed text
- `/command` (`std_msgs/String`) — JSON `{"object": "...", "instruction": "..."}`

## ROS 2 Parameters (config/params.yaml)

| Parameter | Default | Description |
|---|---|---|
| `whisper_model` | `base` | Whisper model size: tiny, base, small, medium, large |
| `audio_device` | `4` | sounddevice index. 4 = laptop mic. -1 = system default |
| `ollama_model` | `llama3.2` | Ollama model for command parsing |
| `debug_save_wav` | `true` | Save last recording to /tmp/stt_debug.wav |
| `energy_threshold` | `0.01` | RMS energy threshold for speech detection |
| `silence_duration` | `1.5` | Seconds of silence before end-of-speech |
| `min_speech_duration` | `0.5` | Ignore speech shorter than this (noise filter) |

## Known Limitations

- Ollama must be running (`ollama serve` or systemd service) for command parsing.
- Energy threshold may need tuning for different microphones and environments.
