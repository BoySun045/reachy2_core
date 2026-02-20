"""Audio capture from microphone via sounddevice with energy monitoring."""

import threading
from typing import Optional

import numpy as np
import sounddevice as sd


WHISPER_SAMPLE_RATE = 16000


def find_default_mic() -> int:
    """Return the default input device index."""
    return sd.default.device[0] or 0


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if orig_rate == target_rate:
        return audio
    ratio = target_rate / orig_rate
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class AudioCapture:
    """Continuously monitors microphone audio with energy tracking.

    The stream runs continuously once ``open_stream()`` is called.
    ``start_recording()`` / ``stop_recording()`` control whether chunks
    are accumulated into the buffer.  ``current_energy`` always reflects
    the RMS of the most recent audio chunk regardless of recording state.
    """

    def __init__(self, device: Optional[int] = None):
        if device is not None and device >= 0:
            self._device = device
        else:
            self._device = find_default_mic()

        info = sd.query_devices(self._device)
        if info['max_input_channels'] < 1:
            raise RuntimeError(f'Device {self._device} ({info["name"]}) has no input channels.')

        self._native_rate = int(info['default_samplerate'])
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._current_energy = 0.0

    @property
    def device_name(self) -> str:
        info = sd.query_devices(self._device)
        return info['name']

    @property
    def native_rate(self) -> int:
        return self._native_rate

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_energy(self) -> float:
        """RMS energy of the most recent audio chunk (always updated)."""
        return self._current_energy

    def open_stream(self) -> None:
        """Open the microphone stream (always-on monitoring)."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._native_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
        )
        self._stream.start()

    def close_stream(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def start_recording(self) -> None:
        """Start accumulating audio chunks into the buffer."""
        with self._lock:
            self._chunks = []
            self._recording = True

    def stop_recording(self) -> np.ndarray:
        """Stop accumulating and return audio resampled to 16kHz.

        Returns:
            1-D float32 numpy array at 16kHz.
        """
        with self._lock:
            self._recording = False
            if not self._chunks:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []

        return _resample(audio, self._native_rate, WHISPER_SAMPLE_RATE)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # Always update energy (even when not recording)
        rms = float(np.sqrt(np.mean(indata ** 2)))
        self._current_energy = rms

        if self._recording:
            with self._lock:
                self._chunks.append(indata.copy())
