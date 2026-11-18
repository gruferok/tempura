"""Conservative, opt-in onset snapping on the already-corrected chart timeline.

No playback clock, calibration offset, Torch, display or file is touched during
analysis. Librosa is imported only by the offline worker. A Hold moves as a whole;
its type/duration, the lane order, note count and exact chords are preserved.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .audio_engine import AudioTrack
from .chart import Chart, Note
from .ml import check_cancel, report

ALGORITHM = "librosa-onsets-energy-v1"
ANALYSIS_RATE = 22050
HOP_LENGTH = 128


@dataclass(frozen=True)
class SnapGrid:
    track_id: str
    duration: float
    onsets: tuple[float, ...]
    beats: tuple[float, ...] = ()
    tempo_bpm: float | None = None

    def __post_init__(self):
        if not isinstance(self.track_id, str) or not self.track_id or not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Нужны track_id и положительная длительность Beat-Snap.")
        for name in ("onsets", "beats"):
            values = tuple(float(t) for t in getattr(self, name))
            if any(not math.isfinite(t) or not 0 <= t < self.duration for t in values):
                raise ValueError("Beat-Snap targets должны находиться в пределах трека.")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        if self.tempo_bpm is not None and (not math.isfinite(self.tempo_bpm) or self.tempo_bpm <= 0):
            raise ValueError("Некорректный темп Beat-Snap.")


@dataclass(frozen=True)
class SnapChange:
    index: int
    lane: int
    before: float
    after: float

    @property
    def delta_ms(self) -> float:
        return (self.after - self.before) * 1000


@dataclass(frozen=True)
class SnapResult:
    chart: Chart
    grid: SnapGrid
    changes: tuple[SnapChange, ...]
    max_shift_ms: float
    blocked_notes: int = 0

    @property
    def largest_shift_ms(self) -> float:
        return max((abs(c.delta_ms) for c in self.changes), default=0.0)


def _refine_attacks(mono: np.ndarray, coarse_times: np.ndarray) -> tuple[float, ...]:
    """Refine delayed spectral-flux peaks against ~1 ms energy attack edges.

    Only accept a local rise above its preceding energy floor. Steady/noisy
    regions without a clear amplitude attack are deliberately not snap targets.
    This is an onset estimate, not a claim of sample-accurate musical ground truth.
    """
    step = round(ANALYSIS_RATE / 1000)
    size = math.ceil(len(mono) / step) * step
    squared = np.pad(np.square(mono), (0, size - len(mono)))
    rms = np.sqrt(squared.reshape(-1, step).mean(axis=1))
    times = np.arange(len(rms)) * step / ANALYSIS_RATE
    rise = rms - np.concatenate((np.zeros(3, dtype=rms.dtype), rms[:-3])) if len(rms) >= 3 else rms
    found = []
    for coarse in coarse_times:
        first = int(np.searchsorted(times, max(0, coarse - 0.060)))
        last = min(len(rms), int(np.searchsorted(times, coarse + 0.012)) + 1)
        if first >= last:
            continue
        attack = first + int(np.argmax(rise[first:last]))
        baseline = float(np.median(rms[max(0, attack - 20):max(1, attack - 4)])) if attack > 4 else 0.0
        peak = float(np.max(rms[max(0, attack - 1):min(len(rms), attack + 10)]))
        if peak <= baseline * 1.25 + 1e-5:
            continue
        threshold = baseline + (peak - baseline) * 0.10
        start = max(0, attack - 30)
        below = np.flatnonzero(rms[start:attack + 1] <= threshold)
        edge = start + int(below[-1]) + 1 if len(below) else start
        if not len(below) and start != 0:
            continue
        refined = float(times[min(edge, len(times) - 1)])
        if not found or refined - found[-1] > 0.020:
            found.append(refined)
    return tuple(found)


def analyze_beats(track: AudioTrack, *, progress=None, cancel=None) -> SnapGrid:
    """Find supported transients, not a forced constant-tempo quantization grid.

    Beat tracking is only a tempo/preview hint. Unsupported inferred beats and
    silent subdivisions are NEVER used as targets. Time zero is the raw PCM
    origin, exactly the same timeline as the input-corrected saved notes.
    """
    check_cancel(cancel)
    report(progress, phase="audio", message="Подготовка аудио (без playback/offset)")
    channels = np.mean(np.square(track.pcm), axis=0)
    mono = track.pcm.mean(axis=1, dtype=np.float32)
    # Do not turn an anti-phase stereo transient into silence by downmixing.
    if float(np.mean(np.square(mono))) < float(np.max(channels)) * 0.1:
        mono = track.pcm[:, int(np.argmax(channels))]
    if track.duration < 0.08 or float(np.max(np.abs(mono))) <= 1e-6:
        return SnapGrid(track.track_id, track.duration, ())
    import librosa

    if track.sample_rate != ANALYSIS_RATE:
        mono = librosa.resample(mono, orig_sr=track.sample_rate, target_sr=ANALYSIS_RATE)
    check_cancel(cancel)
    report(progress, phase="onsets", message="Librosa: спектральные атаки и уточнение по энергии")
    strength = librosa.onset.onset_strength(y=mono, sr=ANALYSIS_RATE, hop_length=HOP_LENGTH, n_fft=1024)
    peaks = librosa.onset.onset_detect(onset_envelope=strength, sr=ANALYSIS_RATE, hop_length=HOP_LENGTH,
                                      backtrack=False, units="time", wait=4)
    onsets = _refine_attacks(mono, np.asarray(peaks))
    onsets = tuple(t for t in onsets if 0 <= t < track.duration)
    check_cancel(cancel)
    tempo, beats = None, ()
    if len(onsets) >= 4 and track.duration >= 2:
        report(progress, phase="beats", message="Librosa: оценка темпа (не обязательная жёсткая сетка)")
        bpm, beat_frames = librosa.beat.beat_track(onset_envelope=strength, sr=ANALYSIS_RATE,
                                                  hop_length=HOP_LENGTH, trim=False)
        if np.size(bpm) and math.isfinite(float(np.ravel(bpm)[0])) and float(np.ravel(bpm)[0]) > 0:
            tempo = float(np.ravel(bpm)[0])
        supported = set()
        for time in np.asarray(beat_frames) * HOP_LENGTH / ANALYSIS_RATE:
            # The onset edge can precede its spectral peak by up to 60 ms.
            nearest = _nearest(float(time), onsets)
            if nearest is not None and abs(nearest - float(time)) <= 0.060:
                supported.add(nearest)
        beats = tuple(sorted(supported))
    check_cancel(cancel)
    report(progress, phase="done", message=f"Найдено атак: {len(onsets)}")
    return SnapGrid(track.track_id, track.duration, onsets, beats, tempo)


def _nearest(time: float, targets: tuple[float, ...]) -> float | None:
    index = bisect_left(targets, time)
    candidates = targets[max(0, index - 1):index + 1]
    return min(candidates, key=lambda t: (abs(t - time), t)) if candidates else None


def snap_chart(chart: Chart, grid: SnapGrid, *, max_shift_ms: float = 25.0) -> SnapResult:
    """Return a separate preview; never mutate the original or reapply offsets.

    Conflicts revert whole exact chords rather than shorten Holds, delete notes,
    merge repeated taps, change lanes or cross the track boundary. Reverted
    shifts can cause neighbour conflicts, so recheck their edges to a fixed point.
    Each proposed note can be reverted only once; the queue is bounded by O(N).
    """
    chart.validate()
    if not isinstance(chart.metadata, dict):
        raise ValueError("metadata карты должна быть объектом.")
    if not math.isfinite(max_shift_ms) or not 0 < max_shift_ms <= 50:
        raise ValueError("Окно Beat-Snap должно быть >0 и <=50 ms.")
    # The same file decoded at another device rate can differ by one output
    # sample at EOF. Allow <=1 ms rounding, but always keep the chart's bounds.
    if chart.track["id"] != grid.track_id or abs(chart.track["duration"] - grid.duration) > 0.001:
        raise ValueError("Аудио Beat-Snap не совпадает с картой по id/длительности.")
    original = chart.notes
    moved = list(original)
    chords = defaultdict(list)
    lanes = defaultdict(list)
    for i, note in enumerate(original):
        chords[note.time].append(i)
        lanes[note.lane].append(i)
    blocked = set()
    for time, indices in chords.items():
        target = _nearest(time, grid.onsets)
        if target is None or abs(target - time) > max_shift_ms / 1000 + 1e-10 or abs(target - time) <= 1e-10:
            continue
        if target >= chart.track["duration"] or any(target + original[i].duration > chart.track["duration"] + 1e-10 for i in indices):
            blocked.update(indices)
            continue
        for i in indices:
            note = original[i]
            moved[i] = Note(target, note.lane, note.duration, note.type)
    edges = deque()
    neighbours = defaultdict(list)
    for indices in lanes.values():
        indices.sort(key=lambda i: (original[i].time, i))
        for left, right in zip(indices, indices[1:]):
            edges.append((left, right))
            neighbours[left].append((left, right))
            neighbours[right].append((left, right))
    while edges:
        left, right = edges.popleft()
        crossing = moved[left].end > moved[right].time + 1e-10
        collapsed = original[left].time < original[right].time and moved[left].time >= moved[right].time - 1e-10
        if not (crossing or collapsed):
            continue
        for index in (left, right):
            for member in chords[original[index].time]:
                if moved[member] != original[member]:
                    moved[member] = original[member]
                    blocked.add(member)
                    edges.extend(neighbours[member])
    changes = tuple(SnapChange(i, note.lane, original[i].time, note.time)
                    for i, note in enumerate(moved) if note != original[i])
    metadata = deepcopy(chart.metadata)
    metadata["auto_quantized"] = True
    metadata["human_reviewed"] = False
    metadata["beat_snap"] = {"version": 1, "algorithm": ALGORITHM, "max_shift_ms": max_shift_ms,
                             "moved_notes": len(changes), "blocked_notes": len(blocked),
                             "detected_onsets": len(grid.onsets), "tempo_bpm": grid.tempo_bpm,
                             "hold_policy": "translate_preserving_duration"}
    result = Chart(deepcopy(chart.track), list(chart.keys), moved, metadata, chart.schema_version, chart.time_unit)
    result.validate()
    return SnapResult(result, grid, changes, max_shift_ms, len(blocked))


def save_snap_copy(result: SnapResult, source_path: str | Path) -> Path:
    """Publish only on explicit acceptance, with a new name; keep the source.

    Each accepted result gets *_snap.json / *_snap_2.json, never replacing the
    source or a previous copy. Metadata is copied, not mutated on the preview.
    """
    source = Path(source_path)
    if not result.changes:
        raise ValueError("Beat-Snap не изменил ни одной ноты; копия не нужна.")
    if not source.is_file():
        raise ValueError("Сначала сохраните исходную карту.")
    stem = source.stem
    path = source.with_name(f"{stem}_snap.json")
    index = 2
    while path.exists():
        path = source.with_name(f"{stem}_snap_{index}.json")
        index += 1
    chart = deepcopy(result.chart)
    chart.metadata["beat_snap"]["source_chart"] = source.name
    chart.metadata["beat_snap"]["source_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return chart.save(path)
