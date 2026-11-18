"""Live key-down/key-up recording with audio-offset correction exactly once."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import pygame

from .audio_engine import AudioTrack
from .chart import Chart, Note, recording_metadata, validate_keys
from .rhythm_radar import RhythmRadar, build_rhythm_radar
from .ui import App, CYAN, GOLD, MUTED, RED, draw_lanes, event_snapshot, transport_hud


@dataclass
class LiveRecorder:
    keys: list[str]
    duration: float
    hold_threshold: float = 0.150
    notes: list[Note] = field(default_factory=list)
    pressed: dict[int, float] = field(default_factory=dict)
    rewind_count: int = 0

    def __post_init__(self) -> None:
        validate_keys(self.keys)
        if not math.isfinite(self.duration) or self.duration <= 0 or not math.isfinite(self.hold_threshold) or self.hold_threshold <= 0:
            raise ValueError("Некорректная длительность или hold threshold.")

    def key_down(self, lane: int, time: float) -> bool:
        if lane not in range(len(self.keys)) or lane in self.pressed or not math.isfinite(time) or not 0 <= time < self.duration:
            return False
        # Avoid overlapping same-lane notes if a backend discontinuity jumps back.
        if any(n.lane == lane and n.end > time + 1e-6 for n in self.notes):
            return False
        self.pressed[lane] = time
        return True

    def _finished_note(self, lane: int, start: float, time: float) -> Note:
        duration = min(self.duration, max(start, time)) - start
        # One threshold rule for KEYUP and for holds cut by a rewind boundary.
        hold = duration > self.hold_threshold + 1e-9
        return Note(start, lane, duration if hold else 0.0, "Hold" if hold else "Tap")

    def key_up(self, lane: int, time: float) -> Note | None:
        if lane not in self.pressed or not math.isfinite(time):
            return None
        note = self._finished_note(lane, self.pressed.pop(lane), time)
        self.notes.append(note)
        return note

    def rewind_to(self, corrected_time: float) -> float:
        """Keep only the prefix before a corrected chart-time boundary.

        Remove onsets at/after the boundary; trim completed AND currently held
        notes crossing it. Short prefixes become Taps. No physical key remains
        active: its later KEYUP/repeat must not create a phantom continuation.
        Completion order is retained so Backspace still undoes the last note.
        """
        if not math.isfinite(corrected_time):
            raise ValueError("Граница отката должна быть конечной.")
        cutoff = max(0.0, min(self.duration, corrected_time))
        kept = []
        for note in self.notes:
            if note.time < cutoff - 1e-9:
                kept.append(self._finished_note(note.lane, note.time, cutoff)
                            if note.end > cutoff else note)
        for lane, start in self.pressed.items():
            if start < cutoff - 1e-9:
                kept.append(self._finished_note(lane, start, cutoff))
        self.notes = kept
        self.pressed.clear()
        self.rewind_count += 1
        return cutoff

    def release_all(self, time: float) -> None:
        for lane in list(self.pressed):
            self.key_up(lane, time)

    def chart(self, track: AudioTrack, metadata: dict | None = None) -> Chart:
        chart = Chart(track.reference(), list(self.keys), list(self.notes), metadata or {})
        chart.validate()
        return chart

    def visible_notes(self, time: float) -> list[Note]:
        result = list(self.notes)
        for lane, start in self.pressed.items():
            length = max(0, time - start)
            hold = length > self.hold_threshold
            result.append(Note(start, lane, length if hold else 0.0, "Hold" if hold else "Tap"))
        return result


def draw_mapper(app: App, recorder: LiveRecorder, track: AudioTrack, raw_time: float,
                radar: RhythmRadar | None = None, show_radar: bool = True) -> None:
    app.begin(f"Live-разметка · {len(recorder.keys)}K", track.name)
    corrected = app.config.input_time(raw_time)
    display_time = app.config.display_time(raw_time)
    draw_lanes(app, recorder.keys, display_time, recorder.visible_notes(corrected), held=set(recorder.pressed))
    if radar is not None and show_radar:
        x0, y0, width, height = app.width - 300, 170, 250, 72
        pygame.draw.rect(app.screen, (22, 29, 47), (x0, y0, width, height), border_radius=8)
        pygame.draw.line(app.screen, CYAN, (x0 + 8, y0 + height // 2), (x0 + width - 8, y0 + height // 2), 1)
        pygame.draw.line(app.screen, RED, (x0 + 8, y0 + height // 2), (x0 + 8, y0 + height - 5), 2)
        first = max(0, int(display_time / radar.step))
        count = min(len(radar.rms) - first, int(1.8 / radar.step))
        for j in range(max(0, count)):
            xx = x0 + 10 + j * (width - 20) / max(1, int(1.8 / radar.step))
            amp = float(radar.rms[first + j]) * (height * .38)
            onset = float(radar.peaks[first + j])
            pygame.draw.line(app.screen, CYAN, (round(xx), round(y0 + height/2 - amp)), (round(xx), round(y0 + height/2 + amp)), 1)
            if onset > .22:
                pygame.draw.line(app.screen, GOLD, (round(xx), y0 + 4), (round(xx), y0 + height - 4), 2)
        app.text("WAVE · GOLD = атака · 1.8 s", x0 + 8, y0 - 22, 14, MUTED)
    app.text("REC", 45, 180, 28, RED)
    app.text(f"{len(recorder.notes)} нот", 45, 224, 18, CYAN)
    app.text(f"Input {app.config.audio_offset_ms:+.0f} ms", 45, 265, 14, MUTED)
    app.text(f"Hold >{app.config.hold_threshold_ms:g} ms", 45, 292, 14, MUTED)
    app.text(f"Rewind {recorder.rewind_count}", 45, 322, 14, MUTED)
    rewind_keys = "F2" if "left" in recorder.keys else "Left/F2"
    app.footer(f"{rewind_keys} — назад 5с   W — волна   V — ориентация   F3/F4 — скорость   P — пауза   S — сохранить   Esc — выход")


def run_live_mapper(app: App, track: AudioTrack, keys: list[str]) -> tuple[Chart, Path] | None:
    """Save the raw take first; return it AFTER closing the recording stream.

    The menu may then offer offline Beat-Snap, never inside this audio context.
    """
    recorder = LiveRecorder(keys, track.duration, app.config.hold_threshold_ms / 1000)
    radar = build_rhythm_radar(track)
    show_radar = True
    key_to_lane = {pygame.key.key_code(key): lane for lane, key in enumerate(keys)}
    with app.engine(track) as engine:
        engine.play()

        def save() -> tuple[Chart, Path]:
            recorder.release_all(app.config.input_time(engine.song_time))
            metadata = recording_metadata(app.config.audio_offset_ms, engine.underruns,
                                          "SDL event timestamp if exposed; otherwise queue-poll time")
            metadata["include_in_training"] = True
            metadata["sample_rate"] = track.sample_rate
            metadata["hold_threshold_ms"] = app.config.hold_threshold_ms
            metadata["rewind_count"] = recorder.rewind_count
            metadata["rewind_seconds"] = 5.0
            chart = recorder.chart(track, metadata)
            path = chart.save()
            app.toast = f"Сохранено {len(recorder.notes)} нот: {path.name}"
            return chart, path

        while app.running:
            engine.check_health()
            raw_time, _ = engine.snapshot()
            draw_mapper(app, recorder, track, raw_time, radar, show_radar)
            transport_hud(app, engine, raw_time)
            for event in app.events():
                if event.type == pygame.WINDOWFOCUSLOST:
                    recorder.release_all(app.config.input_time(engine.song_time))
                    engine.pause()
                if event.type == pygame.KEYDOWN:
                    if getattr(event, "repeat", False):
                        continue
                    if event.key == pygame.K_w:
                        show_radar = not show_radar
                    elif event.key == pygame.K_v:
                        app.config.orientation = "horizontal" if app.config.orientation == "vertical" else "vertical"
                        app.config.save()
                    elif event.key == pygame.K_F3:
                        app.config.scroll_seconds = min(6.0, app.config.scroll_seconds + 0.25)
                        app.config.save()
                    elif event.key == pygame.K_F4:
                        app.config.scroll_seconds = max(0.5, app.config.scroll_seconds - 0.25)
                        app.config.save()
                    elif event.key == pygame.K_ESCAPE:
                        recorder.release_all(app.config.input_time(engine.song_time))
                        engine.pause()
                        if not recorder.notes:
                            return
                        if app.confirm("Сохранить карту и выйти?",
                                       "Enter сохранит карту; Esc вернёт к записи. Существующий файл будет заменён.",
                                       "Сохранить"):
                            return save()
                    elif event.key == pygame.K_s:
                        engine.pause()
                        return save()
                    elif event.key == pygame.K_p:
                        recorder.release_all(app.config.input_time(engine.song_time))
                        engine.toggle_pause()
                    elif event.key == pygame.K_r:
                        recorder.release_all(app.config.input_time(engine.song_time))
                        engine.pause()
                        if app.confirm("Начать запись заново?", "Текущие несохранённые ноты будут удалены.", "Начать"):
                            recorder = LiveRecorder(keys, track.duration, app.config.hold_threshold_ms / 1000)
                            engine.seek(0)
                            engine.play()
                    elif event.key == pygame.K_F2 or (event.key == pygame.K_LEFT and event.key not in key_to_lane):
                        target = engine.rewind(5.0)
                        recorder.rewind_to(app.config.input_time(target))
                    elif event.key == pygame.K_BACKSPACE and recorder.notes:
                        recorder.notes.pop()
                    elif event.key in key_to_lane and engine.requested_state == "playing" and not engine.transport_pending:
                        event_time, event_state = event_snapshot(engine, event)
                        if event_state == "playing" and recorder.key_down(key_to_lane[event.key], app.config.input_time(event_time)):
                            engine.trigger_hit()
                elif event.type == pygame.KEYUP and event.key in key_to_lane:
                    event_time, event_state = event_snapshot(engine, event)
                    if engine.requested_state == "playing" and not engine.transport_pending and event_state == "playing":
                        recorder.key_up(key_to_lane[event.key], app.config.input_time(event_time))
            # A final KEYUP can already be queued when the DAC reaches EOF.
            # Consume it before closing held notes and persisting the chart.
            _, state = engine.snapshot()
            if state == "finished" and not engine.transport_pending:
                return save()
            if engine.requested_state == "paused":
                app.text("ПАУЗА · P для продолжения", app.width // 2, app.height // 2, 22, CYAN, center=True)
            elif engine.transport_pending:
                app.text("ОЖИДАНИЕ DAC", app.width // 2, app.height // 2, 22, CYAN, center=True)
            app.frame()
