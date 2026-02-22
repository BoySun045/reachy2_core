"""Speech-to-text wrapper using faster-whisper (CTranslate2 backend)."""

import numpy as np
from faster_whisper import WhisperModel

# Domain vocabulary hint — biases decoder towards robot command words
_INITIAL_PROMPT = (
    "reachy, spot, go to the table, pick up the cup, navigate to the chair, "
    "gripper, arm on, arm off, arm up, arm down, stand, sit, "
    "move forward, turn left, turn right, step back, grab the bottle"
)


class Transcriber:
    """Wraps faster-whisper for single-shot transcription."""

    def __init__(self, model_name: str = 'medium'):
        """
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v3).
        """
        self._model = WhisperModel(
            model_name, device="auto", compute_type="int8"
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (must be 16000 for Whisper).

        Returns:
            Transcribed text string, or empty string if audio is too short.
        """
        if len(audio) < sample_rate * 0.3:  # ignore < 0.3s
            return ''

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            initial_prompt=_INITIAL_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
