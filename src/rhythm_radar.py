"""Small, cached display envelopes for the live rhythm radar (20 ms bins)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .audio_engine import AudioTrack


@dataclass(frozen=True)
class RhythmRadar:
    rms: np.ndarray
    peaks: np.ndarray
    step: float


def build_rhythm_radar(track: AudioTrack, step: float = 0.02) -> RhythmRadar:
    if not np.isfinite(step) or not 0.01 <= step <= 0.05:
        raise ValueError("Radar step must be 10–50 ms")
    width = max(1, round(track.sample_rate * step))
    count = (len(track.pcm) + width - 1) // width
    mono = track.pcm.mean(axis=1, dtype=np.float32)
    rms = np.zeros(count, np.float32)
    peaks = np.zeros(count, np.float32)
    flux = np.abs(np.diff(mono, prepend=mono[:1]))
    for i in range(count):
        block = mono[i * width:(i + 1) * width]
        if len(block):
            rms[i] = np.sqrt(np.mean(block * block, dtype=np.float64))
            peaks[i] = np.max(flux[i * width:(i + 1) * width])
    scale = max(float(rms.max()), float(peaks.max()), 1e-8)
    rms /= scale
    peaks /= max(float(peaks.max()), 1e-8)
    rms.flags.writeable = peaks.flags.writeable = False
    return RhythmRadar(rms, peaks, width / track.sample_rate)
