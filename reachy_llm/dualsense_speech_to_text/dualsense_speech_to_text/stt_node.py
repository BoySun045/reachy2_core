"""ROS 2 node: VAD-based speech-to-text with command routing.

Uses energy-based voice activity detection to automatically detect speech
from a microphone with a hardware mute button.  When audio energy rises
above a threshold the node starts recording; when silence is detected for
a configurable duration it stops, transcribes with Whisper, parses with
Ollama, and routes the command to the appropriate subsystem:
  - navigation  → /pathplanner_manager/query
  - manipulation → /vidbot/trigger
  - locomotion   → /locomotion/command
"""

import json
import time
import wave
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from dualsense_speech_to_text.audio_capture import AudioCapture
from dualsense_speech_to_text.transcriber import Transcriber
from dualsense_speech_to_text.command_parser import CommandParser


# VAD state machine states
_IDLE = 0
_RECORDING = 1
_PROCESSING = 2


class SpeechToTextNode(Node):
    def __init__(self):
        super().__init__('speech_to_text')

        # Parameters
        self.declare_parameter('whisper_model', 'base')
        self.declare_parameter('audio_device', -1)
        self.declare_parameter('debug_save_wav', True)
        self.declare_parameter('ollama_model', 'llama3.2')
        self.declare_parameter('energy_threshold', 0.01)
        self.declare_parameter('silence_duration', 1.5)
        self.declare_parameter('min_speech_duration', 0.5)

        whisper_model = self.get_parameter('whisper_model').value
        audio_device = self.get_parameter('audio_device').value
        self._debug_wav = self.get_parameter('debug_save_wav').value
        self._energy_threshold = self.get_parameter('energy_threshold').value
        self._silence_duration = self.get_parameter('silence_duration').value
        self._min_speech_duration = self.get_parameter('min_speech_duration').value

        # Publishers
        self._pub = self.create_publisher(String, 'transcription', 10)
        self._cmd_pub = self.create_publisher(String, 'command', 10)

        # Routing publishers
        self._nav_pub = self.create_publisher(
            String, '/pathplanner_manager/query', 10)
        self._manip_pub = self.create_publisher(
            String, '/vidbot/trigger', 10)
        self._loco_pub = self.create_publisher(
            String, '/locomotion/command', 10)

        # Audio
        self.get_logger().info('Initializing microphone...')
        device_idx = audio_device if audio_device >= 0 else None
        self._audio = AudioCapture(device=device_idx)
        self._audio.open_stream()
        self.get_logger().info(f'Microphone: {self._audio.device_name}')

        # Whisper
        self.get_logger().info(f'Loading Whisper model "{whisper_model}"...')
        self._transcriber = Transcriber(model_name=whisper_model)
        self.get_logger().info('Whisper model loaded.')

        # Ollama
        ollama_model = self.get_parameter('ollama_model').value
        self.get_logger().info(f'Command parser using Ollama model "{ollama_model}"...')
        self._parser = CommandParser(model=ollama_model)
        self.get_logger().info('Command parser ready.')

        # VAD state machine
        self._state = _IDLE
        self._silence_start = 0.0
        self._speech_start = 0.0

        # Poll at 50 Hz
        self._timer = self.create_timer(0.02, self._poll_vad)
        self.get_logger().info(
            f'Ready. Listening for speech (threshold={self._energy_threshold}, '
            f'silence={self._silence_duration}s).'
        )

    def _poll_vad(self):
        energy = self._audio.current_energy
        now = time.monotonic()

        if self._state == _IDLE:
            if energy > self._energy_threshold:
                # Speech onset detected
                self.get_logger().info(f'Speech detected (energy={energy:.4f}), recording...')
                self._audio.start_recording()
                self._state = _RECORDING
                self._speech_start = now
                self._silence_start = 0.0

        elif self._state == _RECORDING:
            if energy < self._energy_threshold:
                # Silence — start/continue silence timer
                if self._silence_start == 0.0:
                    self._silence_start = now
                elif now - self._silence_start >= self._silence_duration:
                    # Enough silence — stop recording
                    speech_duration = now - self._speech_start
                    self.get_logger().info(
                        f'Silence detected ({self._silence_duration}s), '
                        f'speech was {speech_duration:.1f}s'
                    )
                    self._state = _PROCESSING
                    self._process_recording()
            else:
                # Still speaking — reset silence timer
                self._silence_start = 0.0

        # _PROCESSING is handled synchronously in _process_recording

    def _process_recording(self):
        audio = self._audio.stop_recording()
        duration = len(audio) / 16000.0

        self.get_logger().info(f'Captured {duration:.2f}s of audio ({len(audio)} samples)')

        if duration < self._min_speech_duration:
            self.get_logger().info('Too short, ignoring.')
            self._state = _IDLE
            return

        if self._debug_wav:
            wav_path = '/tmp/stt_debug.wav'
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            self.get_logger().info(f'Debug WAV saved to {wav_path}')

        self.get_logger().info('Transcribing...')
        text = self._transcriber.transcribe(audio)
        if text:
            msg = String()
            msg.data = text
            self._pub.publish(msg)
            self.get_logger().info(f'Transcription: {text}')

            self.get_logger().info('Parsing command with Ollama...')
            parsed = self._parser.parse(text)
            if parsed:
                cmd_msg = String()
                cmd_msg.data = json.dumps({
                    'category': parsed.category,
                    'object': parsed.object,
                    'instruction': parsed.instruction,
                    'dx': parsed.dx,
                    'dy': parsed.dy,
                    'dyaw': parsed.dyaw,
                })
                self._cmd_pub.publish(cmd_msg)
                self.get_logger().info(
                    f'[{parsed.category}] object="{parsed.object}", '
                    f'instruction="{parsed.instruction}"'
                )
                self._route_command(parsed)
            else:
                self.get_logger().info('Could not parse command from text.')
        else:
            self.get_logger().info('No speech detected (too short or silent).')

        self._state = _IDLE

    def _route_command(self, parsed):
        msg = String()

        if parsed.category == 'navigation':
            msg.data = parsed.object
            self._nav_pub.publish(msg)
            self.get_logger().info(
                f'Routed to navigation: query="{parsed.object}"')

        elif parsed.category == 'manipulation':
            msg.data = json.dumps({
                'object': parsed.object,
                'instruction': parsed.instruction,
            })
            self._manip_pub.publish(msg)
            self.get_logger().info(
                f'Routed to manipulation: object="{parsed.object}", '
                f'instruction="{parsed.instruction}"')

        elif parsed.category == 'locomotion':
            msg.data = json.dumps({
                'dx': parsed.dx,
                'dy': parsed.dy,
                'dyaw': parsed.dyaw_rad,
            })
            self._loco_pub.publish(msg)
            self.get_logger().info(
                f'Routed to locomotion: dx={parsed.dx}, dy={parsed.dy}, '
                f'dyaw={parsed.dyaw}deg')

    def destroy_node(self):
        if self._audio.is_recording:
            self._audio.stop_recording()
        self._audio.close_stream()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
