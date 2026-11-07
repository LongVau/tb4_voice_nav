"""WSLg PulseAudio microphone capture via libpulse-simple.so.0 and WAV utilities."""

from __future__ import annotations

import ctypes
from pathlib import Path
import wave

import numpy as np

PA_STREAM_RECORD = 2
PA_SAMPLE_S16LE = 3


class _PaSampleSpec(ctypes.Structure):
    _fields_ = [
        ('format', ctypes.c_int),
        ('rate', ctypes.c_uint32),
        ('channels', ctypes.c_uint8),
    ]


def _load_pulse_simple() -> ctypes.CDLL:
    pulse = ctypes.CDLL('libpulse-simple.so.0')
    pulse.pa_simple_new.restype = ctypes.c_void_p
    pulse.pa_simple_new.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_PaSampleSpec),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    pulse.pa_simple_read.restype = ctypes.c_int
    pulse.pa_simple_read.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
    ]
    pulse.pa_simple_free.restype = None
    pulse.pa_simple_free.argtypes = [ctypes.c_void_p]
    return pulse


def record_pulse_audio(
    duration_sec: float = 4.0,
    sample_rate: int = 16000,
    warmup_sec: float = 0.2,
) -> np.ndarray:
    """Record 16-bit mono PCM from PulseAudio and return normalized float32 [-1.0, 1.0]."""
    pulse = _load_pulse_simple()
    spec = _PaSampleSpec(PA_SAMPLE_S16LE, int(sample_rate), 1)
    error = ctypes.c_int(0)

    handle = pulse.pa_simple_new(
        None,
        b'tb4_voice_nav',
        PA_STREAM_RECORD,
        None,
        b'voice_command_mic',
        ctypes.byref(spec),
        None,
        None,
        ctypes.byref(error),
    )
    if not handle:
        raise RuntimeError(
            f'Failed to open PulseAudio recording stream (error code={error.value}). '
            'Verify PULSE_SERVER is active (e.g. unix:/mnt/wslg/PulseServer).'
        )

    try:
        warmup_samples = max(0, int(round(float(warmup_sec) * sample_rate)))
        if warmup_samples > 0:
            warm_buf = (ctypes.c_int16 * warmup_samples)()
            pulse.pa_simple_read(
                handle, warm_buf, ctypes.sizeof(warm_buf), ctypes.byref(error)
            )

        total_samples = max(1, int(round(float(duration_sec) * sample_rate)))
        pcm_buf = (ctypes.c_int16 * total_samples)()
        res = pulse.pa_simple_read(
            handle, pcm_buf, ctypes.sizeof(pcm_buf), ctypes.byref(error)
        )
        if res < 0:
            raise RuntimeError(f'PulseAudio read error (code={error.value})')

        raw_i16 = np.frombuffer(pcm_buf, dtype=np.int16).copy()
        audio_f32 = (raw_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        return audio_f32
    finally:
        pulse.pa_simple_free(handle)


def save_wav(
    path: str | Path,
    audio_f32: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Save float32 mono audio [-1.0, 1.0] to a 16-bit PCM WAV file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(out_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_i16.tobytes())


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a mono/stereo 16-bit PCM WAV file into float32 mono [-1.0, 1.0]."""
    in_path = Path(path)
    with wave.open(str(in_path), 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32), int(sample_rate)

