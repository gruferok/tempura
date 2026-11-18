"""Validated chart/audio pairs, Gaussian labels, and masked overlapping windows.

Live chart times are already audio-offset corrected. No offset is applied here.
AI-generated and automatically snapped charts are excluded unless explicitly
marked human_reviewed=true; do not silently learn automatic labels/live duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

import numpy as np

from ..audio_engine import AudioTrack, generate_test_track, synth_click
from ..chart import Chart, Note
from ..storage import CHART_DIR, ROOT
from . import FEATURE_CACHE, FRAME_RATE, MAX_LANES, check_cancel, report, require_torch
from .features import FEATURE_DIM, extract_features, frame_count

torch = require_torch()


@dataclass
class TrainingExample:
    features: np.ndarray
    onset_target: np.ndarray
    hold_target: np.ndarray
    num_lanes: int
    group_id: str
    source: str
    kind: str = "user"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = len(self.features)
        if not isinstance(self.num_lanes, int) or not 1 <= self.num_lanes <= MAX_LANES:
            raise ValueError("num_lanes: от 1 до 8.")
        if frames < 1 or self.features.shape != (frames, FEATURE_DIM):
            raise ValueError("Некорректная матрица признаков.")
        if any(a.shape != (frames, MAX_LANES) for a in (self.onset_target, self.hold_target)):
            raise ValueError("Некорректные размеры таргетов.")
        if not all(np.isfinite(a).all() for a in (self.features, self.onset_target, self.hold_target)):
            raise ValueError("Признаки и таргеты должны быть конечными.")
        if any(np.any((a < 0) | (a > 1)) for a in (self.onset_target, self.hold_target)):
            raise ValueError("Таргеты должны быть в [0, 1].")


def encode_targets(chart: Chart, frames: int | None = None, *, sigma_frames: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    chart.validate()
    if not math.isfinite(sigma_frames) or sigma_frames <= 0:
        raise ValueError("sigma_frames должен быть положительным.")
    frames = frame_count(chart.track["duration"]) if frames is None else frames
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("Нужен хотя бы один frame.")
    onset = np.zeros((frames, MAX_LANES), dtype=np.float32)
    hold = np.zeros_like(onset)
    radius = math.ceil(4 * sigma_frames)
    for note in chart.notes:
        center = note.time * FRAME_RATE
        start, end = max(0, math.floor(center) - radius), min(frames, math.ceil(center) + radius + 1)
        if start < end:
            gaussian = np.exp(-0.5 * ((np.arange(start, end) - center) / sigma_frames) ** 2)
            onset[start:end, note.lane] = np.maximum(onset[start:end, note.lane], gaussian)
        if note.type == "Hold":
            first = max(0, math.ceil(note.time * FRAME_RATE - 1e-8))
            last = min(frames, math.ceil(note.end * FRAME_RATE - 1e-8))
            hold[first:last, note.lane] = 1.0  # [start, end), on the same centred time grid
    return onset, hold


def training_exclusion(chart: Chart) -> str | None:
    if not isinstance(chart.metadata, dict):
        return "metadata должна быть объектом"
    if chart.metadata.get("include_in_training") is False:
        return "исключено пользователем: include_in_training=false"
    if (chart.metadata.get("auto_quantized") and chart.metadata.get("human_reviewed") is not True
            and chart.metadata.get("include_in_training") is not True):
        return "Beat-Snap-копия без явного include_in_training=true / human_reviewed=true"
    if chart.metadata.get("example_only"):
        return "демонстрационный JSON, не пользовательская разметка"
    if (chart.metadata.get("ai_generated") and chart.metadata.get("human_reviewed") is not True
            and chart.metadata.get("include_in_training") is not True):
        return "AI-карта без явного include_in_training=true / human_reviewed=true"
    if (chart.metadata.get("auto_quantized") and chart.metadata.get("human_reviewed") is not True
            and chart.metadata.get("include_in_training") is not True):
        return "Beat-Snap-копия без явного включения в обучение"
    if chart.metadata.get("output_underruns", 0):
        return "запись содержит audio underruns; перезапишите для точного обучения"
    if not chart.notes:
        return "пустая карта"
    return None


def catalog_charts(charts_dir: str | Path = CHART_DIR) -> tuple[list[tuple[Path, Chart]], list[dict]]:
    accepted, skipped = [], []
    directory = Path(charts_dir)
    if not directory.exists():
        return accepted, skipped
    if not directory.is_dir():
        raise ValueError(f"Не папка карт: {directory}")
    for path in sorted(directory.glob("*.json")):
        try:
            chart = Chart.load(path)
            reason = training_exclusion(chart)
            if reason:
                skipped.append({"path": str(path), "reason": reason})
            else:
                accepted.append((path, chart))
        except (ValueError, OSError, TypeError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    return accepted, skipped


def load_chart_track(chart: Chart, chart_path: str | Path | None = None) -> AudioTrack:
    """No dialogs. Missing/mismatched source audio is reported, never substituted."""
    reference = chart.track
    if reference["kind"] == "synthetic":
        if reference.get("generator_version") != 1:
            raise ValueError("Неизвестная версия генератора тестового трека.")
        try:
            track = generate_test_track(duration=reference["duration"], sample_rate=reference["sample_rate"],
                                        bpm=reference["bpm"], lead_in=reference["lead_in"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"Нет параметров синтетического трека: {error}") from error
    else:
        source = reference.get("audio_path")
        if not isinstance(source, str) or not source:
            raise ValueError("В карте нет audio_path.")
        source = Path(source).expanduser()
        candidates = [source] if source.is_absolute() else [ROOT / source]
        if chart_path is not None and not source.is_absolute():
            candidates.append(Path(chart_path).resolve().parent / source)
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise ValueError(f"Аудиофайл не найден: {source}")
        track = AudioTrack.from_file(found, sample_rate=reference.get("sample_rate", 48000))
    if track.track_id != reference["id"] or abs(track.duration - reference["duration"]) > 0.02:
        raise ValueError("Аудио не совпадает с картой по SHA-256/id или длительности.")
    return track


def load_examples(charts_dir: str | Path = CHART_DIR, *, cache_dir=FEATURE_CACHE,
                  progress=None, cancel=None) -> tuple[list[TrainingExample], list[dict]]:
    catalog, skipped = catalog_charts(charts_dir)
    for item in skipped:
        check_cancel(cancel)
        report(progress, phase="skipped", message=f"{Path(item['path']).name}: {item['reason']}")
    examples = []
    for i, (path, chart) in enumerate(catalog):
        check_cancel(cancel)
        report(progress, phase="dataset", step=i + 1, steps=len(catalog), message=f"Карта: {path.name}")
        try:
            track = load_chart_track(chart, path)
            sequence = extract_features(track, cache_dir=cache_dir, progress=progress, cancel=cancel)
            onset, hold = encode_targets(chart, len(sequence.values))
            provenance = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "track_id": track.track_id, "lanes": chart.lanes, "notes": len(chart.notes),
                          "duration": track.duration}
            examples.append(TrainingExample(sequence.values, onset, hold, chart.lanes, track.track_id,
                                            str(path), "user", provenance))
        except (OSError, ValueError, RuntimeError) as error:
            check_cancel(cancel)
            skipped.append({"path": str(path), "reason": str(error)})
            report(progress, phase="skipped", message=f"{path.name}: {error}")
    return examples, skipped


class SequenceDataset(torch.utils.data.Dataset):
    """Full-track labels sliced into windows; no label shift at chunk boundaries."""
    def __init__(self, examples: list[TrainingExample], window_seconds: float = 10.0,
                 stride_seconds: float | None = None):
        if not math.isfinite(window_seconds) or not 0.1 <= window_seconds <= 30:
            raise ValueError("window_seconds: от 0.1 до 30.")
        self.examples = examples
        self.window_frames = round(window_seconds * FRAME_RATE)
        stride_seconds = window_seconds / 2 if stride_seconds is None else stride_seconds
        if not math.isfinite(stride_seconds) or not 0 < stride_seconds <= window_seconds:
            raise ValueError("stride_seconds должен быть > 0 и <= window_seconds.")
        stride = max(1, round(stride_seconds * FRAME_RATE))
        self.windows: list[tuple[int, int]] = []
        for index, example in enumerate(examples):
            last = max(0, len(example.features) - self.window_frames)
            starts = list(range(0, last + 1, stride))
            if starts[-1] != last:
                starts.append(last)
            self.windows.extend((index, start) for start in starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        example_index, start = self.windows[index]
        example = self.examples[example_index]
        length = min(self.window_frames, len(example.features) - start)
        features = np.zeros((self.window_frames, FEATURE_DIM), dtype=np.float32)
        onset = np.zeros((self.window_frames, MAX_LANES), dtype=np.float32)
        hold = np.zeros_like(onset)
        mask = np.zeros_like(onset)
        features[:length] = example.features[start:start + length]
        onset[:length] = example.onset_target[start:start + length]
        hold[:length] = example.hold_target[start:start + length]
        mask[:length, :example.num_lanes] = 1
        return {"features": torch.from_numpy(features), "onset_target": torch.from_numpy(onset),
                "hold_target": torch.from_numpy(hold), "mask": torch.from_numpy(mask),
                "num_lanes": torch.tensor(example.num_lanes, dtype=torch.long),
                "length": torch.tensor(length, dtype=torch.long)}


def procedural_track(seed: int, num_lanes: int, duration: float = 6.0) -> tuple[AudioTrack, Chart]:
    """Small, explicitly synthetic teaching task; NOT a pretrained music model.

    Variable tempo, rests, alternating lanes, occasional chords and sustained
    tones provide a cold start. Its patterns are not evidence of human style.
    """
    if not isinstance(num_lanes, int) or not 1 <= num_lanes <= MAX_LANES or not 1 <= duration <= 30:
        raise ValueError("Некорректные параметры bootstrap.")
    rng = np.random.default_rng(seed)
    rate = 16000
    bpm = float(rng.uniform(90, 165))
    step = 60 / bpm / 2
    pcm = np.zeros((round(duration * rate), 2), dtype=np.float32)
    notes = []
    previous_end = [-1.0] * num_lanes
    tick = synth_click(rate)[:, 0]
    for beat, time in enumerate(np.arange(0.5, duration - 0.2, step)):
        if (beat % 2 and rng.random() < 0.55) or rng.random() < 0.08:
            continue
        lane = beat % num_lanes if rng.random() < 0.75 else int(rng.integers(num_lanes))
        selected = [lane]
        if num_lanes > 1 and rng.random() < 0.13:
            selected.append((lane + 1) % num_lanes)
        for lane in selected:
            if previous_end[lane] > time - 0.07:
                continue
            duration_note = min(float(rng.uniform(0.22, 0.60)), duration - time - 0.05) if rng.random() < 0.22 else 0
            notes.append(Note(float(time), lane, duration_note, "Hold" if duration_note else "Tap"))
            previous_end[lane] = time + duration_note
            start = round(time * rate)
            length = min(round((duration_note or 0.09) * rate), len(pcm) - start)
            t = np.arange(length, dtype=np.float64) / rate
            frequency = 180 * 2 ** (lane / 3)
            envelope = (np.minimum(1, t / 0.008) * np.minimum(1, (length / rate - t) / 0.012)
                        if duration_note else np.exp(-t * 36) * np.minimum(1, t / 0.002))
            signal = (0.18 * np.sin(2 * np.pi * frequency * t)
                      + 0.035 * np.sin(2 * np.pi * frequency * 2 * t)) * envelope
            pcm[start:start + length] += signal.astype(np.float32)[:, None]
            count = min(len(tick), len(pcm) - start)
            pcm[start:start + count] += tick[:count, None] * 0.5
    np.clip(pcm, -0.95, 0.95, out=pcm)
    track = AudioTrack(pcm, rate, f"bootstrap_{seed}_{num_lanes}K", bpm=bpm, lead_in=0.5,
                       track_id=f"bootstrap-v1-{seed}-{num_lanes}-{len(pcm)}")
    # Only used in memory. It is deliberately never saved as a user Live chart.
    from .generator_keys import keys_for_lanes
    chart = Chart(track.reference(), keys_for_lanes(num_lanes), notes, {"bootstrap": True})
    chart.validate()
    return track, chart


def make_bootstrap_examples(count: int = 12, *, duration: float = 6.0, seed: int = 1234,
                            cache_dir=FEATURE_CACHE, progress=None, cancel=None) -> list[TrainingExample]:
    if not isinstance(count, int) or not 1 <= count <= 256:
        raise ValueError("Число bootstrap примеров: от 1 до 256.")
    result = []
    for i in range(count):
        check_cancel(cancel)
        lanes = i % MAX_LANES + 1
        report(progress, phase="bootstrap_data", step=i + 1, steps=count,
               message=f"Процедурное аудио {i + 1}/{count} · {lanes}K")
        track, chart = procedural_track(seed + i, lanes, duration)
        features = extract_features(track, cache_dir=cache_dir, progress=progress, cancel=cancel)
        onset, hold = encode_targets(chart, len(features.values))
        result.append(TrainingExample(features.values, onset, hold, lanes, track.track_id,
                                      track.name, "bootstrap"))
    return result
