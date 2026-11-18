"""Optional post-record Beat-Snap review; analysis starts only on user request.

The caller must close playback/recording before entering. The shared bounded
worker imports no Torch, touches no Pygame, and never saves the preview itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pygame

from .ai_screen import BackgroundJob
from .audio_engine import AudioTrack
from .beat_snap import SnapGrid, SnapResult, analyze_beats, save_snap_copy, snap_chart
from .chart import Chart
from .ui import App, CYAN, GOLD, MUTED, RED, TEXT, key_label


@dataclass
class SnapState:
    source_name: str = "Live-карта"
    total_notes: int = 0
    window_ms: float = 25.0
    busy: bool = False
    elapsed: float = 0.0
    message: str = "Исходная карта сохранена. Анализ — только по кнопке B."
    error: str = ""
    result: SnapResult | None = None
    include_in_training: bool = False


def draw_snap_screen(app: App, state: SnapState) -> dict:
    app.begin("Smart Beat-Snap", "Опциональная копия · исходная Live-разметка не изменяется")
    app.text("01 / ORIGINAL", 44, 165, 14, CYAN)
    app.fit_text(state.source_name, 44, 194, app.width - 88, 22, TEXT)
    app.text(f"Нот: {state.total_notes}   ·   окно притяжения:", 44, 239, 18, MUTED)
    buttons = {}
    for i, window in enumerate((15, 20, 25)):
        buttons[f"window_{window}"] = app.button((44 + i * 120, 275, 105, 44), f"{window} ms",
                                                selected=state.window_ms == window)
    buttons["analyze"] = app.button((44, 340, 345, 52), "B   Найти атаки / Beat-Snap", accent=not state.busy)
    app.text("Только близкие атаки, без принудительной сетки.", 44, 412, 16, MUTED)
    app.text("Клавиши, типы и длительности Hold сохраняются.", 44, 440, 16, MUTED)
    app.text("Конфликты / выход за трек: сдвиг пропускается.", 44, 468, 16, MUTED)
    buttons["play"] = app.button((600, 275, 450, 52), "P   Плейтест результата")
    buttons["training"] = app.button((600, 345, 450, 52),
                                     "T   В обучении: " + ("ДА" if state.include_in_training else "НЕТ"),
                                     selected=state.include_in_training)
    buttons["save"] = app.button((600, 410, 450, 52), "Enter   Сохранить отдельную копию", accent=state.result is not None)
    buttons["back"] = app.button((600, 415, 450, 48), "Esc   Оставить только исходную карту")
    if state.busy:
        app.text(f"Анализ в фоне · {state.elapsed:.1f} s", 44, 519, 22, CYAN)
    elif state.result is not None:
        result = state.result
        app.text(f"Сдвинуто: {len(result.changes)} / {state.total_notes}   ·   пропущено конфликтов: {result.blocked_notes}",
                 44, 509, 18, CYAN)
        tempo = f" · темп ~{result.grid.tempo_bpm:.1f} BPM" if result.grid.tempo_bpm else ""
        app.text(f"Атак: {len(result.grid.onsets)} · макс. сдвиг {result.largest_shift_ms:.2f} ms{tempo}",
                 44, 537, 16, MUTED)
        for i, change in enumerate(result.changes[:4]):
            key = key_label(result.chart.keys[change.lane])
            app.text(f"{key:>5}   {change.before:.4f} -> {change.after:.4f} s    {change.delta_ms:+.2f} ms",
                     44, 566 + i * 24, 16, TEXT)
    app.fit_text(state.error or state.message, 44, app.height - 82, app.width - 88, 14, RED if state.error else MUTED)
    app.footer("B — анализ   P — плейтест   Enter — сохранить копию   Esc — без изменений / отмена анализа")
    return buttons


def run_snap_screen(app: App, track: AudioTrack, chart: Chart, source_path: str | Path) -> Path | None:
    state = SnapState(source_name=Path(source_path).name, total_notes=len(chart.notes))
    grid: SnapGrid | None = None
    job: BackgroundJob | None = None
    leave_after_job = False
    try:
        while app.running:
            if job is not None:
                state.elapsed = time.monotonic() - job.started
                for event in job.poll():
                    state.message = event.get("message", state.message)
                if job.done.is_set():
                    state.busy = False
                    if job.error:
                        state.error = job.error
                    elif job.cancelled:
                        state.message = "Анализ отменён. Исходная карта не изменена."
                    else:
                        grid = job.result
                        try:
                            state.result = snap_chart(chart, grid, max_shift_ms=state.window_ms)
                            state.message = ("Сравните сдвиги и попробуйте P. Enter сохраняет копию; Esc отбрасывает её."
                                             if state.result.changes else "Близких безопасных сдвигов нет. Исходная карта остаётся без изменений.")
                        except (ValueError, TypeError) as error:
                            state.error = str(error)
                    job = None
                    if leave_after_job:
                        return None
            buttons = draw_snap_screen(app, state)
            action = None
            for event in app.events():
                if event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                    action = {pygame.K_b: "analyze", pygame.K_p: "play", pygame.K_RETURN: "save",
                              pygame.K_ESCAPE: "back", pygame.K_t: "training", pygame.K_1: "window_15",
                              pygame.K_2: "window_20", pygame.K_3: "window_25"}.get(event.key, action)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    action = next((name for name, rect in buttons.items() if rect.collidepoint(event.pos)), action)
            if not app.running:
                break
            if action == "back":
                if job is None:
                    return None
                job.cancel.set()
                leave_after_job = True
                state.message = "Отмена на ближайшей безопасной границе анализа…"
            if not state.busy:
                try:
                    if action == "training":
                        state.include_in_training = not state.include_in_training
                    elif action == "analyze" and chart.notes:
                        state.error = ""
                        state.result = None
                        state.busy = True
                        job = BackgroundJob(lambda progress, cancel: analyze_beats(track, progress=progress, cancel=cancel),
                                            name="Tempura Beat-Snap").start()
                    elif action and action.startswith("window_"):
                        state.window_ms = float(action.split("_")[1])
                        if grid is not None:
                            state.result = snap_chart(chart, grid, max_shift_ms=state.window_ms)
                    elif action == "play" and state.result is not None and state.result.changes:
                        from .gameplay import run_gameplay
                        run_gameplay(app, track, state.result.chart)
                    elif action == "save":
                        if state.result is not None and state.result.changes:
                            state.result.chart.metadata["include_in_training"] = state.include_in_training
                            path = save_snap_copy(state.result, source_path)
                            app.toast = f"Beat-Snap: {len(state.result.changes)} сдвигов; сохранено {path.name}. Оригинал не изменён."
                            return path
                        state.message = "Сначала B — анализ. Если сдвигов нет, копия не создаётся. Esc — назад."
                except (OSError, ValueError, RuntimeError) as error:
                    state.error = str(error)
            app.frame(fast=False)
    finally:
        if job is not None:
            job.cancel.set()
            if job.thread is not None:
                job.thread.join(timeout=0.25)  # no worker writes, even after QUIT
    return None
