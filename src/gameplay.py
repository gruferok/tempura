"""Lane-aware tap/hold judgment, independent of rendering and audio hardware."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import math
from pathlib import Path
import re

import pygame

from .audio_engine import AudioTrack
from .chart import Chart, Note, validate_keys
from .storage import CHART_DIR
from .live_mapper import LiveRecorder
from .ui import App, CYAN, GOLD, MUTED, RED, TEXT, VIOLET, draw_lanes, event_snapshot, transport_hud

WINDOWS = ((25.0, "Perfect"), (50.0, "Great"), (90.0, "Good"))
POINTS = {"Perfect": 1000, "Great": 700, "Good": 400, "Miss": 0}
RANK = {name: i for i, name in enumerate(POINTS)}


def grade(delta_ms: float) -> str:
    for window, label in WINDOWS:
        if abs(delta_ms) <= window + 1e-7:
            return label
    return "Miss"


@dataclass(frozen=True)
class Judgment:
    label: str
    delta_ms: float | None
    note_index: int | None
    phase: str = "tap"

    @property
    def timing_text(self) -> str:
        if self.delta_ms is None:
            return "—"
        if abs(self.delta_ms) < 0.05:
            return "0.0 ms · точно"
        return f"{self.delta_ms:+.1f} ms · {'early' if self.delta_ms < 0 else 'late'}"


class Judge:
    def __init__(self, chart: Chart):
        chart.validate()
        self.chart = chart
        self.notes = sorted(chart.notes, key=lambda n: (n.time, n.lane))
        self.states = ["pending"] * len(self.notes)
        self.holding: dict[int, tuple[int, str]] = {}
        self.by_lane = [[i for i, note in enumerate(self.notes) if note.lane == lane] for lane in range(chart.lanes)]
        self.times = [[self.notes[i].time for i in indexes] for indexes in self.by_lane]
        self.counts = {name: 0 for name in POINTS}
        self.score = self.combo = self.max_combo = self.ghosts = 0
        self.last = Judgment("Готов", None, None)
        self.hit_errors_ms: list[float] = []
        self._miss_cursor = 0

    def _complete(self, index: int, label: str, delta_ms: float | None, phase="tap") -> Judgment:
        self.states[index] = "miss" if label == "Miss" else "hit"
        self.counts[label] += 1
        self.score += POINTS[label]
        self.combo = 0 if label == "Miss" else self.combo + 1
        self.max_combo = max(self.max_combo, self.combo)
        self.last = Judgment(label, delta_ms, index, phase)
        return self.last

    def key_down(self, lane: int, time: float) -> Judgment:
        if lane not in range(self.chart.lanes) or lane in self.holding:
            return Judgment("Пусто", None, None)
        indexes, times = self.by_lane[lane], self.times[lane]
        left = bisect_left(times, time - 0.090 - 1e-9)
        right = bisect_right(times, time + 0.090 + 1e-9)
        possible = [i for i in indexes[left:right] if self.states[i] == "pending"]
        if not possible:
            self.ghosts += 1
            self.last = Judgment("Пусто", None, None)
            return self.last
        index = min(possible, key=lambda i: abs(time - self.notes[i].time))
        note = self.notes[index]
        delta = (time - note.time) * 1000
        label = grade(delta)
        if label != "Miss":
            self.hit_errors_ms.append(delta)
        if note.type == "Hold":
            self.states[index] = "holding"
            self.holding[lane] = (index, label)
            self.last = Judgment(label, delta, index, "hold start")
            return self.last
        return self._complete(index, label, delta)

    def key_up(self, lane: int, time: float) -> Judgment | None:
        active = self.holding.pop(lane, None)
        if active is None:
            return None
        index, start_grade = active
        delta = (time - self.notes[index].end) * 1000
        label = max((start_grade, grade(delta)), key=lambda value: RANK[value])
        return self._complete(index, label, delta, "hold end")

    def release_all(self, time: float) -> None:
        for lane in list(self.holding):
            self.key_up(lane, time)

    def update(self, time: float) -> None:
        while self._miss_cursor < len(self.notes) and self.notes[self._miss_cursor].time < time - 0.090 - 1e-9:
            index = self._miss_cursor
            self._miss_cursor += 1
            if self.states[index] == "pending":
                self._complete(index, "Miss", (time - self.notes[index].time) * 1000)
        for lane, (index, _) in list(self.holding.items()):
            if time > self.notes[index].end + 0.090 + 1e-9:
                del self.holding[lane]
                self._complete(index, "Miss", (time - self.notes[index].end) * 1000, "hold end")

    def finish(self) -> None:
        for index, state in enumerate(self.states):
            if state in ("pending", "holding"):
                self._complete(index, "Miss", None)
        self.holding.clear()

    @property
    def accuracy(self) -> float:
        judged = sum(self.counts.values())
        return 100 * self.score / (judged * 1000) if judged else 100.0


@dataclass
class PunchInEditor:
    """Replace chart onsets inside completed punch intervals, preserving outside notes."""
    chart: Chart
    intervals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.intervals)

    def replace(self, start: float, end: float, notes: list[Note]) -> None:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= self.chart.track["duration"]):
            raise ValueError("Некорректная граница Punch-In.")
        candidate = []
        for note in self.chart.notes:
            if start - 1e-9 <= note.time <= end + 1e-9:
                continue
            if note.type == "Hold" and note.time < start < note.end:
                prefix = start - note.time
                note = Note(note.time, note.lane, prefix, "Hold") if prefix > 0.150 else Note(note.time, note.lane)
            candidate.append(note)
        candidate.extend(n for n in notes if start - 1e-9 <= n.time <= end + 1e-9)
        candidate.sort(key=lambda n: (n.time, n.lane))
        repaired: list[Note] = []
        for note in candidate:
            following = next((n for n in candidate if n.lane == note.lane and n.time > note.time + 1e-9), None)
            if note.type == "Hold" and following is not None and note.end > following.time:
                length = following.time - note.time
                if length <= 0.150:
                    continue
                note = Note(note.time, note.lane, length, "Hold")
            repaired.append(note)
        updated = Chart(self.chart.track.copy(), list(self.chart.keys), repaired,
                        dict(self.chart.metadata), self.chart.schema_version, self.chart.time_unit)
        updated.validate()
        self.chart = updated
        self.intervals.append((start, end))


def error_statistics(judge: Judge) -> dict:
    values = list(getattr(judge, "hit_errors_ms", []))
    mean = sum(values) / len(values) if values else None
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if values else None
    bins = [0] * 18
    for value in values:
        bins[max(0, min(17, int((value + 90) / 10)))] += 1
    return {"count": len(values), "mean_error_ms": mean, "std_error_ms": std,
            "unstable_rate": std * 10 if std is not None else None, "histogram": bins}


def save_overdub_copy(chart: Chart, charts_dir=CHART_DIR) -> Path:
    directory = Path(charts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w.-]+", "_",  str(chart.track.get("name", "track"))).strip("._")[:100] or "track"
    path = directory / f"{stem}_{chart.lanes}K_overdub.json"
    index = 2
    while path.exists():
        path = directory / f"{stem}_{chart.lanes}K_overdub_{index}.json"
        index += 1
    return chart.save(path)


def draw_gameplay(app: App, chart: Chart, judge: Judge, raw_time: float, held: set[int], punch_active=False) -> None:
    app.begin(f"Плейтест · {chart.lanes}K", str(chart.track.get("name", "track")))
    draw_lanes(app, chart.keys, app.config.display_time(raw_time), judge.notes,
               held=held, states=judge.states)
    app.text(f"{judge.score:,}", 42, 178, 28, CYAN)
    app.text(f"COMBO {judge.combo}", 42, 225, 18, TEXT)
    app.text(f"ACC {judge.accuracy:.1f}%", 42, 258, 16, MUTED)
    for i, (label, count) in enumerate(judge.counts.items()):
        app.text(f"{label}  {count}", 42, 325 + i * 30, 16, MUTED)
    color = RED if judge.last.label == "Miss" else CYAN if judge.last.label == "Perfect" else GOLD
    app.text(judge.last.label, app.width - 195, 180, 22, color)
    app.text(judge.last.timing_text, app.width - 215, 220, 14, color)
    app.text(judge.last.phase, app.width - 195, 253, 14, MUTED)
    if punch_active:
        app.text("● PUNCH-IN REC · R завершить сегмент", app.width // 2, 158, 18, RED, center=True)
    app.footer("R — Punch-In запись/завершить   V — ориентация   F3/F4 — скорость   P — пауза   Esc — выход")


def save_offset_adjustment(app: App, mean_error_ms: float) -> float:
    """Add measured signed lateness to the input offset, clamped to config limits."""
    app.config.audio_offset_ms = max(-500.0, min(500.0, app.config.audio_offset_ms + mean_error_ms))
    app.config.save()
    return app.config.audio_offset_ms


def _post_gameplay(app: App, judge: Judge) -> None:
    stats = error_statistics(judge)
    while app.running:
        app.begin("Результаты плейтеста", "Hit timing · ошибки только по засчитанным нажатиям")
        app.text(f"Точность: {stats['count']} hits", 48, 170, 22, CYAN)
        mean, std = stats["mean_error_ms"], stats["std_error_ms"]
        app.text(f"Mean: {mean:+.1f} ms" if mean is not None else "Mean: —", 48, 218, 18, TEXT)
        app.text(f"SD: {std:.1f} ms    Unstable Rate: {stats['unstable_rate']:.1f}" if std is not None else "Unstable Rate: —",
                 48, 250, 18, MUTED)
        x0, y0, width, height = 170, 330, 760, 130
        pygame.draw.line(app.screen, MUTED, (x0, y0 + height), (x0 + width, y0 + height), 1)
        center = x0 + width // 2
        pygame.draw.line(app.screen, CYAN, (center, y0), (center, y0 + height), 2)
        peak = max(stats["histogram"], default=1) or 1
        for i, amount in enumerate(stats["histogram"]):
            bar_h = round((amount / peak) * (height - 12))
            x = x0 + i * width // 18
            pygame.draw.rect(app.screen, VIOLET if i < 9 else GOLD,
                             (x + 2, y0 + height - bar_h, max(2, width // 18 - 4), bar_h))
        app.text("−90 ms · early", x0, y0 + height + 12, 14, MUTED)
        app.text("late · +90 ms", x0 + width - 130, y0 + height + 12, 14, MUTED)
        suggestion = stats["count"] >= 16 and mean is not None and abs(mean) >= 4.0
        if suggestion:
            app.text(f"O — offset adjustment {mean:+.1f} ms", 48, 535, 18, GOLD)
        app.footer("O — применить audio offset   Enter/Esc — меню")
        for event in app.events():
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    return
                if event.key == pygame.K_o and suggestion:
                    new_value = save_offset_adjustment(app, mean)
                    app.toast = f"Audio offset: {new_value:+.1f} ms"
                    return
        app.frame(fast=False)


def _offer_punch_save(app: App, chart: Chart) -> None:
    include = False
    while app.running:
        app.begin("Сохранить Punch-In?", "Исходная карта остаётся нетронутой; будет создан отдельный overdub JSON")
        app.text("T — включать эту запись в обучение", 48, 190, 20, CYAN if include else MUTED)
        app.text("Enter — сохранить копию    Esc — отказаться", 48, 235, 18, TEXT)
        app.footer("Active Learning: явно проверьте сегмент перед включением в обучение")
        for event in app.events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    include = not include
                elif event.key == pygame.K_RETURN:
                    chart.metadata["include_in_training"] = include
                    path = save_overdub_copy(chart)
                    app.toast = f"Overdub сохранён: {path.name}"
                    return
                elif event.key == pygame.K_ESCAPE:
                    return
        app.frame(fast=False)


def run_gameplay(app: App, track: AudioTrack, chart: Chart) -> None:
    if track.track_id != chart.track["id"]:
        raise ValueError("Аудиофайл не совпадает с картой (SHA-256/id). Выберите исходный трек.")
    if abs(track.duration - chart.track["duration"]) > 0.02:
        raise ValueError("Длительность аудиофайла не совпадает с картой.")
    judge = Judge(chart)
    editor = PunchInEditor(chart)
    punch_recorder = None
    punch_start = None
    held: set[int] = set()
    key_to_lane = {pygame.key.key_code(key): lane for lane, key in enumerate(chart.keys)}

    def finish_punch(end_time):
        nonlocal judge, punch_recorder, punch_start
        if punch_recorder is None:
            return
        end_time = max(punch_start, min(chart.track["duration"], end_time))
        punch_recorder.release_all(end_time)
        previous = judge
        editor.replace(punch_start, end_time, list(punch_recorder.notes))
        chart.notes = list(editor.chart.notes)
        updated = Judge(chart)
        old_states = {(n.time, n.lane, n.duration, n.type): previous.states[i]
                      for i, n in enumerate(previous.notes)}
        for i, note in enumerate(updated.notes):
            if punch_start - 1e-9 <= note.time <= end_time + 1e-9:
                updated.states[i] = "hit"  # replaced notes are edits, not scored twice
            else:
                updated.states[i] = old_states.get((note.time, note.lane, note.duration, note.type), "pending")
        updated.counts = previous.counts.copy()
        updated.score, updated.combo, updated.max_combo = previous.score, previous.combo, previous.max_combo
        updated.ghosts, updated.hit_errors_ms = previous.ghosts, list(previous.hit_errors_ms)
        updated.last = previous.last
        judge = updated
        punch_recorder = punch_start = None
    leave_requested = False
    with app.engine(track) as engine:
        engine.play()
        while app.running:
            engine.check_health()
            now, state = engine.snapshot()
            # Process timestamped input BEFORE sweeping overdue notes this frame.
            for event in app.events():
                if event.type == pygame.WINDOWFOCUSLOST:
                    judge.release_all(app.config.input_time(engine.song_time))
                    held.clear()
                    engine.pause()
                if event.type == pygame.KEYDOWN:
                    if getattr(event, "repeat", False):
                        continue
                    if event.key == pygame.K_ESCAPE:
                        if punch_recorder is not None:
                            finish_punch(app.config.input_time(engine.song_time))
                        engine.pause()
                        leave_requested = True
                        break
                    if event.key == pygame.K_v:
                        app.config.orientation = "horizontal" if app.config.orientation == "vertical" else "vertical"
                        app.config.save()
                    elif event.key == pygame.K_F3:
                        app.config.scroll_seconds = min(6.0, app.config.scroll_seconds + 0.25)
                        app.config.save()
                    elif event.key == pygame.K_F4:
                        app.config.scroll_seconds = max(0.5, app.config.scroll_seconds - 0.25)
                        app.config.save()
                    elif event.key == pygame.K_p:
                        judge.release_all(app.config.input_time(engine.song_time))
                        held.clear()
                        engine.toggle_pause()
                    elif event.key == pygame.K_r:
                        if punch_recorder is None:
                            punch_start = max(0.0, min(chart.track["duration"], app.config.input_time(engine.song_time)))
                            punch_recorder = LiveRecorder(list(chart.keys), chart.track["duration"],
                                                          app.config.hold_threshold_ms / 1000)
                        else:
                            finish_punch(app.config.input_time(engine.song_time))
                        held.clear()
                    elif event.key in key_to_lane and engine.requested_state == "playing" and not engine.transport_pending:
                        lane = key_to_lane[event.key]
                        if lane not in held:
                            event_time, event_state = event_snapshot(engine, event)
                            if event_state == "playing":
                                held.add(lane)
                                chart_time = app.config.input_time(event_time)
                                if punch_recorder is not None:
                                    punch_recorder.key_down(lane, chart_time)
                                else:
                                    result = judge.key_down(lane, chart_time)
                                    if result.note_index is not None:
                                        engine.trigger_hit()
                elif event.type == pygame.KEYUP and event.key in key_to_lane:
                    lane = key_to_lane[event.key]
                    held.discard(lane)
                    event_time, _ = event_snapshot(engine, event)
                    chart_time = app.config.input_time(event_time)
                    if punch_recorder is not None:
                        punch_recorder.key_up(lane, chart_time)
                    else:
                        judge.key_up(lane, chart_time)
            if leave_requested:
                break
            now, state = engine.snapshot()
            if state == "finished" and punch_recorder is not None:
                finish_punch(app.config.input_time(now))
            if not engine.transport_pending and punch_recorder is None:
                judge.update(app.config.input_time(now))
            if state == "finished" and not engine.transport_pending:
                judge.finish()
                break
            draw_gameplay(app, chart, judge, now, held, punch_recorder is not None)
            transport_hud(app, engine, now)
            if engine.requested_state == "paused":
                app.text("ПАУЗА · P для продолжения", app.width // 2, app.height // 2, 22, GOLD, center=True)
            app.frame()
    if app.running:
        if editor.changed:
            _offer_punch_save(app, chart)
        if app.running:
            _post_gameplay(app, judge)
