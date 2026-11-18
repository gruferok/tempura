"""Centred, exactly 100 Hz audio features, with a versioned non-pickle cache.

Columns: 64 log-Mel + 64 first differences + onset strength + spectral flux.
Frame k is centred at k / 100 seconds, NOT at the end of its FFT window. The
16 kHz analysis resampling is independent of the playback device/sample rate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from ..audio_engine import AudioTrack
from . import FEATURE_CACHE, FEATURE_VERSION, FRAME_RATE, check_cancel, report


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 512
    n_mels: int = 64
    fmin: float = 30.0
    log_scale: float = 10000.0

    @property
    def feature_dim(self) -> int:
        return self.n_mels * 2 + 2

    def validate(self) -> None:
        if self.sample_rate != 16000 or self.hop_length != 160 or self.n_mels != 64 or self.n_fft != 512:
            raise ValueError("Feature schema v1: 16 kHz / hop 160 / FFT 512 / 64 Mel (ровно 100 Hz).")
        if self.fmin != 30.0 or self.log_scale != 10000.0:
            raise ValueError("Неизвестные параметры feature schema v1.")


FEATURE_CONFIG = FeatureConfig()
FEATURE_DIM = FEATURE_CONFIG.feature_dim


def frame_count(duration: float) -> int:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Длительность должна быть положительной и конечной.")
    return max(1, math.ceil(duration * FRAME_RATE - 1e-8))


@dataclass
class FeatureSequence:
    values: np.ndarray
    duration: float
    track_id: str
    cache_hit: bool = False

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.values), dtype=np.float64) / FRAME_RATE

    @property
    def fps(self) -> int:
        return FRAME_RATE


def _cache_key(track: AudioTrack, config: FeatureConfig) -> str:
    identity = {"version": FEATURE_VERSION, "config": asdict(config),
                "track_id": track.track_id, "duration": track.duration,
                "source_sample_rate": track.sample_rate, "frames": len(track.pcm)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def extract_features(track: AudioTrack, *, cache_dir: str | Path | None = FEATURE_CACHE,
                     config: FeatureConfig = FEATURE_CONFIG, progress=None, cancel=None) -> FeatureSequence:
    config.validate()
    check_cancel(cancel)
    expected_frames = frame_count(track.duration)
    key = _cache_key(track, config)
    path = Path(cache_dir) / f"{key}.npz" if cache_dir is not None else None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as stored:
                values = stored["values"]
                valid = (int(stored["version"]) == FEATURE_VERSION and str(stored["key"]) == key
                         and values.shape == (expected_frames, config.feature_dim)
                         and values.dtype == np.float32 and np.isfinite(values).all())
                if valid:
                    report(progress, phase="features", message=f"Кэш: {track.name}", cached=True)
                    return FeatureSequence(np.ascontiguousarray(values), track.duration, track.track_id, True)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # An interrupted/old cache is recomputed, never unpickled.
    report(progress, phase="features", message=f"Признаки 100 Hz: {track.name}", cached=False)
    import librosa

    mono = track.pcm.mean(axis=1, dtype=np.float32)
    if track.sample_rate != config.sample_rate:
        mono = librosa.resample(mono, orig_sr=track.sample_rate, target_sr=config.sample_rate)
    check_cancel(cancel)
    spectrum = np.abs(librosa.stft(mono, n_fft=config.n_fft, hop_length=config.hop_length,
                                  center=True, pad_mode="constant")) ** 2
    filters = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mels,
                                 fmin=config.fmin, fmax=config.sample_rate / 2)
    mel = np.maximum(filters @ spectrum, 0)
    # STFT has an extra frame at an exact EOF; only times strictly before EOF
    # belong to the 100 Hz chart grid. No hidden onset_strength centring shift.
    mel = mel[:, :expected_frames]
    if mel.shape[1] < expected_frames:
        mel = np.pad(mel, ((0, 0), (0, expected_frames - mel.shape[1])))
    log_mel = np.log1p(config.log_scale * mel)
    delta = np.diff(log_mel, axis=1, prepend=log_mel[:, :1])
    onset = np.maximum(delta, 0).mean(axis=0)
    amplitude = np.sqrt(mel)
    flux = np.sqrt(np.mean(np.maximum(np.diff(amplitude, axis=1, prepend=amplitude[:, :1]), 0) ** 2, axis=0))
    values = np.concatenate((log_mel.T, delta.T, onset[:, None], flux[:, None]), axis=1)
    # Per-track, per-column z-score is identical in training and generation;
    # never normalize independently per window. Silence has finite zero features.
    mean = values.mean(axis=0, keepdims=True)
    std = np.maximum(values.std(axis=0, keepdims=True), 1e-4)
    values = np.ascontiguousarray(np.clip((values - mean) / std, -8, 8), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Неконечные аудиопризнаки: {track.name}")
    check_cancel(cancel)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False) as file:
                temporary = file.name
                np.savez_compressed(file, values=values, version=np.int64(FEATURE_VERSION), key=np.array(key))
                file.flush()
                os.fsync(file.fileno())
            check_cancel(cancel)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
    return FeatureSequence(values, track.duration, track.track_id)
