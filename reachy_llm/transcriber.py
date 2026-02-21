"""Whisper speech-to-text wrapper."""

import numpy as np
import whisper


class Transcriber:
    """Wraps OpenAI Whisper for single-shot transcription."""

    def __init__(self, model_name: str = 'base'):
        """
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large).
        """
        self._model = whisper.load_model(model_name)

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

        # Whisper expects float32 audio at 16kHz
        result = self._model.transcribe(
            audio,
            fp16=False,
        )
        return result['text'].strip()
