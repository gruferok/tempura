"""Offline overlap-blended inference and deterministic, playable note decoding."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import time

import numpy as np
from scipy.signal import find_peaks

from ..audio_engine import AudioTrack
from ..chart import Chart, Note
from ..storage import CHART_DIR
from . import FEATURE_CACHE, FRAME_RATE, MAX_LANES, MODEL_PATH, check_cancel, report, require_torch
from .features import FEATURE_DIM, extract_features
from .generator_keys import keys_for_lanes
from .trainer import load_checkpoint

torch = require_torch()


def threshold_for_sensitivity(sensitivity: float) -> float:
    if not math.isfinite(sensitivity) or not 0 <= sensitivity <= 1:
        raise ValueError("sensitivity должна быть от 0 до 1.")
    return round(0.8 - 0.6 * sensitivity, 6)


def peak_pick(probability: np.ndarray, *, threshold: float = 0.55, min_interval_frames: int = 7) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Ожидается конечный вектор вероятностей в [0, 1].")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold должен быть между 0 и 1.")
    if not isinstance(min_interval_frames, int) or min_interval_frames < 1:
        raise ValueError("min_interval_frames должен быть положительным целым числом.")
    if len(probability) == 0:
        return np.empty(0, dtype=np.int64)
    # Do not turn float32 overlap-blending roundoff into many fake peaks on an
    # otherwise flat plateau. This changes probabilities by at most 0.0000005.
    probability = np.round(probability, decimals=6)
    # Padding includes genuine peaks at frame 0 and at the final frame. SciPy
    # centres plateaus and performs height-prioritised minimum-distance picking.
    peaks, _ = find_peaks(np.pad(probability, 1, constant_values=-np.inf), height=threshold,
                          distance=min_interval_frames)
    return (peaks - 1).astype(np.int64)


def _hold_runs(probability: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mask = probability >= threshold
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    # Bridge only short internal 10/20 ms dips, not leading/trailing silence.
    for left, right in zip(ends[:-1], starts[1:]):
        if right - left <= 2:
            mask[left:right] = True
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)


def decode_probabilities(track: AudioTrack, onset_prob: np.ndarray, hold_prob: np.ndarray, num_lanes: int,
                         *, threshold: float = 0.55, hold_threshold: float = 0.5,
                         min_interval_ms: float = 70.0, keys: list[str] | None = None,
                         metadata: dict | None = None) -> Chart:
    keys = keys_for_lanes(num_lanes, keys)
    onset_prob, hold_prob = np.asarray(onset_prob), np.asarray(hold_prob)
    if (onset_prob.ndim != 2 or onset_prob.shape != hold_prob.shape
            or onset_prob.shape[1] < num_lanes or onset_prob.shape[1] > MAX_LANES):
        raise ValueError("Вероятности должны иметь одинаковую форму [frames, lanes<=8].")
    if not np.isfinite(onset_prob).all() or not np.isfinite(hold_prob).all():
        raise ValueError("Неконечные вероятности.")
    if np.any((hold_prob < 0) | (hold_prob > 1)) or np.any((onset_prob < 0) | (onset_prob > 1)):
        raise ValueError("Вероятности должны быть в [0, 1].")
    if not math.isfinite(hold_threshold) or not 0 < hold_threshold < 1:
        raise ValueError("hold_threshold должен быть между 0 и 1.")
    if not math.isfinite(min_interval_ms) or not 10 <= min_interval_ms <= 1000:
        raise ValueError("min_interval_ms: от 10 до 1000.")
    distance = max(1, math.ceil(min_interval_ms * FRAME_RATE / 1000 - 1e-8))
    notes = []
    for lane in range(num_lanes):
        peaks = peak_pick(onset_prob[:, lane], threshold=threshold, min_interval_frames=distance)
        starts, ends = _hold_runs(hold_prob[:, lane], hold_threshold)
        for i, peak in enumerate(peaks):
            time_at = float(peak) / FRAME_RATE
            if time_at >= track.duration:
                continue
            next_peak = int(peaks[i + 1]) if i + 1 < len(peaks) else len(onset_prob)
            run = int(np.searchsorted(starts, peak + 3, side="right")) - 1
            end_frame = int(ends[run]) if run >= 0 and ends[run] > peak - 2 else int(peak)
            end_time = min(track.duration, min(end_frame, next_peak) / FRAME_RATE)
            duration = max(0.0, end_time - time_at)
            is_hold = duration > 0.150 + 1e-9
            notes.append(Note(time_at, lane, duration if is_hold else 0.0, "Hold" if is_hold else "Tap"))
    generated = {"ai_generated": True, "human_reviewed": False, "timestamps_are_corrected": True,
                 "created_utc": datetime.now(timezone.utc).isoformat(), "generator": "RhythmCRNN",
                 "frame_rate": FRAME_RATE, "threshold": threshold, "hold_threshold": hold_threshold,
                 "min_interval_ms": min_interval_ms,
                 "note": "Offline AI draft on the raw track grid; no audio/visual offset baked in."}
    generated.update(metadata or {})
    # Required provenance flags cannot be accidentally disabled by a caller.
    generated["ai_generated"] = True
    chart = Chart(track.reference(), keys, sorted(notes, key=lambda n: (n.time, n.lane)), generated)
    chart.validate()
    return chart


def infer_probabilities(model, features: np.ndarray, num_lanes: int, *, window_frames: int = 1200,
                        overlap_frames: int = 200, progress=None, cancel=None) -> tuple[np.ndarray, np.ndarray]:
    keys_for_lanes(num_lanes)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(features) < 1 or not np.isfinite(features).all():
        raise ValueError("Ожидаются конечные признаки [frames, 130].")
    if not isinstance(window_frames, int) or not 10 <= window_frames <= 6000:
        raise ValueError("window_frames: от 10 до 6000.")
    if not isinstance(overlap_frames, int) or not 0 <= overlap_frames < window_frames:
        raise ValueError("overlap_frames должен быть меньше window_frames.")
    device = next(model.parameters()).device
    last = max(0, len(features) - window_frames)
    starts = list(range(0, last + 1, window_frames - overlap_frames))
    if starts[-1] != last:
        starts.append(last)
    onset = np.zeros((len(features), MAX_LANES), np.float32)
    hold = np.zeros_like(onset)
    weights_sum = np.zeros((len(features), 1), np.float32)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for i, start in enumerate(starts):
                check_cancel(cancel)
                end = min(len(features), start + window_frames)
                batch = torch.from_numpy(np.ascontiguousarray(features[start:end])).unsqueeze(0).to(device)
                prediction = model(batch, num_lanes)
                onset_part = torch.sigmoid(prediction["onset_logits"])[0].cpu().numpy()
                hold_part = torch.sigmoid(prediction["hold_logits"])[0].cpu().numpy()
                weights = np.maximum(0.05, np.hanning(end - start + 2)[1:-1]).astype(np.float32)[:, None]
                onset[start:end] += onset_part * weights
                hold[start:end] += hold_part * weights
                weights_sum[start:end] += weights
                report(progress, phase="inference", step=i + 1, steps=len(starts),
                       message=f"Инференс {i + 1}/{len(starts)} · {num_lanes}K")
    finally:
        model.train(was_training)
    onset /= weights_sum
    hold /= weights_sum
    onset[:, num_lanes:] = 0
    hold[:, num_lanes:] = 0
    if not np.isfinite(onset).all() or not np.isfinite(hold).all():
        raise ValueError("Инференс вернул неконечные вероятности.")
    return np.clip(onset, 0, 1), np.clip(hold, 0, 1)


def ai_chart_path(track: AudioTrack, num_lanes: int, directory: str | Path = CHART_DIR) -> Path:
    keys_for_lanes(num_lanes)
    name = re.sub(r"[^\w.-]+", "_", track.name, flags=re.UNICODE).strip("._")[:100] or "track"
    return Path(directory) / f"{name}_ai_{num_lanes}K.json"


@dataclass
class GenerationResult:
    chart: Chart
    path: Path | None
    seconds: float

    def to_dict(self):
        return {"chart": str(self.path) if self.path is not None else None, "lanes": self.chart.lanes,
                "notes": len(self.chart.notes), "holds": sum(n.type == "Hold" for n in self.chart.notes),
                "seconds": round(self.seconds, 3), "metadata": self.chart.metadata}


def generate_chart(track: AudioTrack, num_lanes: int = 2, *, checkpoint_path: str | Path = MODEL_PATH,
                   threshold: float = 0.55, sensitivity: float | None = None, hold_threshold: float = 0.5,
                   min_interval_ms: float = 70, keys: list[str] | None = None, device="auto",
                   output_dir: str | Path = CHART_DIR, output_path: str | Path | None = None,
                   cache_dir=FEATURE_CACHE, save: bool = True, progress=None, cancel=None) -> GenerationResult:
    started = time.perf_counter()
    keys_for_lanes(num_lanes, keys)
    if sensitivity is not None:
        threshold = threshold_for_sensitivity(sensitivity)
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold должен быть между 0 и 1.")
    check_cancel(cancel)
    model, metadata = load_checkpoint(checkpoint_path, device=str(device))
    sequence = extract_features(track, cache_dir=cache_dir, progress=progress, cancel=cancel)
    onset, hold = infer_probabilities(model, sequence.values, num_lanes, progress=progress, cancel=cancel)
    check_cancel(cancel)
    chart = decode_probabilities(track, onset, hold, num_lanes, threshold=threshold,
                                 hold_threshold=hold_threshold, min_interval_ms=min_interval_ms, keys=keys,
                                 metadata={"checkpoint": str(checkpoint_path), "checkpoint_id": metadata.get("checkpoint_id"),
                                           "training_origin": metadata.get("origin", "unknown"),
                                           "model_user_lane_counts": metadata.get("user_lane_counts", []),
                                           "device": str(next(model.parameters()).device)})
    path = (Path(output_path) if output_path is not None else ai_chart_path(track, num_lanes, output_dir)) if save else None
    if path is not None:
        check_cancel(cancel)
        chart.save(path)
    seconds = time.perf_counter() - started
    report(progress, phase="done", message=f"Готово: {len(chart.notes)} нот", notes=len(chart.notes),
           chart=str(path) if path is not None else None, seconds=seconds)
    return GenerationResult(chart, path, seconds)
