"""Two independent tapping passes; one mixed pass cannot identify two offsets.

A = median(audio-only tap - beat), V = median(visual-only tap - crossing).
Input judgment uses t - A; chart rendering uses t + (V - A).
These include the player's response bias, NOT a pure hardware latency measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pygame

from .audio_engine import metronome_track
from .storage import CONFIG_PATH
from .ui import App, CYAN, GOLD, MUTED, PANEL, RED, TEXT, VIOLET, event_snapshot

MIN_TAPS = 16


def robust_summary(errors_ms: list[float], minimum: int = MIN_TAPS) -> dict:
    values = np.asarray(errors_ms, dtype=np.float64)
    if len(values) < minimum or not np.isfinite(values).all():
        raise ValueError(f"Нужно минимум {minimum} конечных измерений.")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # MAD filtering protects the estimate against accidental off-beat presses.
    inliers = values[np.abs(values - median) <= max(15.0, 3 * 1.4826 * mad)]
    return {"median_ms": float(np.median(inliers)), "mad_ms": mad,
            "robust_jitter_ms": 1.4826 * mad, "accepted_taps": len(values),
            "inliers": len(inliers), "errors_ms": [round(float(x), 3) for x in values]}


def estimate_offsets(audio_errors_ms: list[float], visual_errors_ms: list[float]) -> dict:
    audio, visual = robust_summary(audio_errors_ms), robust_summary(visual_errors_ms)
    return {"audio_offset_ms": round(audio["median_ms"], 3),
            "visual_offset_ms": round(visual["median_ms"] - audio["median_ms"], 3),
            "audio": audio, "visual": visual}


@dataclass
class TapCollector:
    bpm: float = 120.0
    first_beat: float = 2.0
    errors_ms: list[float] = field(default_factory=list)
    used_beats: set[int] = field(default_factory=set)
    rejected: int = 0

    def add(self, song_time: float) -> bool:
        if not math.isfinite(song_time):
            self.rejected += 1
            return False
        interval = 60 / self.bpm
        beat = round((song_time - self.first_beat) / interval)
        delta = song_time - (self.first_beat + beat * interval)
        if beat < 0 or beat in self.used_beats or abs(delta) > interval * 0.42:
            self.rejected += 1
            return False
        self.used_beats.add(beat)
        self.errors_ms.append(delta * 1000)
        return True

    @property
    def complete(self) -> bool:
        return len(self.errors_ms) >= MIN_TAPS


def draw_calibration(app: App, stage: str, collector: TapCollector, song_time: float) -> None:
    audio_only = stage == "audio"
    app.begin("Калибровка 1 / 2 · звук" if audio_only else "Калибровка 2 / 2 · изображение",
              "Нажимайте SPACE в такт звуку. Без визуальной подсказки." if audio_only else
              "Звук выключен. Нажимайте SPACE, когда шарик пересекает центральную линию.")
    center = app.width // 2
    app.text(f"{len(collector.errors_ms):02d} / {MIN_TAPS}", center, 230, 48, CYAN, center=True)
    app.text("зачтённых нажатий", center, 280, 18, MUTED, center=True)
    if audio_only:
        app.text("Слушайте метроном · 120 BPM", center, 400, 28, TEXT, center=True)
        app.text("Не пытайтесь компенсировать задержку заранее.", center, 450, 16, MUTED, center=True)
    else:
        width = min(620, app.width - 120)
        y = 410
        pygame.draw.rect(app.screen, PANEL, (center - width // 2, y - 45, width, 90), border_radius=15)
        pygame.draw.line(app.screen, VIOLET, (center, y - 65), (center, y + 65), 3)
        if song_time >= collector.first_beat - 0.25:
            interval = 60 / collector.bpm
            relative = ((song_time - collector.first_beat + interval / 2) % interval) - interval / 2
            x = center + relative / interval * width
            pygame.draw.circle(app.screen, CYAN, (round(x), y), 15)
        app.text("Пересечение линии = доля", center, 525, 18, MUTED, center=True)
    app.fit_text(f"Повторы/слишком далёкие нажатия: {collector.rejected}", 45, app.height - 110, app.width - 90, 14)
    app.footer("SPACE — нажать   R — повторить проход   Esc — отменить без сохранения")


def run_calibration(app: App, config_path=CONFIG_PATH) -> None:
    if not app.confirm("Калибровка · два прохода",
                       "По 16 нажатий SPACE: сначала на слух, затем по изображению. Esc отменяет.", "Начать"):
        return
    measured: dict[str, list[float]] = {}
    latency = None
    for stage in ("audio", "visual"):
        if stage == "visual" and not app.confirm("Теперь только изображение",
                                                "Звук будет выключен. Нажимайте в момент пересечения линии.", "Начать"):
            return
        collector = TapCollector()
        track = metronome_track(sample_rate=app.config.sample_rate)
        with app.engine(track) as engine:
            engine.set_metronome(stage == "audio")
            engine.play()
            latency = engine.output_latency
            while app.running and not collector.complete:
                engine.check_health()
                now, state = engine.snapshot()
                if engine.underruns:
                    app.notice("Калибровка прервана", ["Обнаружен audio underrun. Результат не сохранён.",
                                                         "Увеличьте latency/blocksize и повторите калибровку."])
                    return
                if state == "finished":
                    app.notice("Недостаточно нажатий", ["Время прохода истекло. Настройки не изменены."])
                    return
                draw_calibration(app, stage, collector, now)
                for event in app.events():
                    if event.type == pygame.WINDOWFOCUSLOST:
                        return  # Focus/input delays would corrupt this pass.
                    if event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                        if event.key == pygame.K_ESCAPE:
                            return
                        if event.key == pygame.K_r:
                            collector = TapCollector()
                            engine.seek(0)
                            engine.play()
                        elif event.key == pygame.K_SPACE:
                            event_time, event_state = event_snapshot(engine, event)
                            if event_state == "playing" and not engine.transport_pending:
                                collector.add(event_time)
                app.frame()
        if not app.running:
            return
        measured[stage] = collector.errors_ms
    result = estimate_offsets(measured["audio"], measured["visual"])
    audio, visual = result["audio"], result["visual"]
    lines = [f"Audio offset: {result['audio_offset_ms']:+.1f} ms (вычитается из времени ввода)",
             f"Visual offset: {result['visual_offset_ms']:+.1f} ms (прибавляется к времени рендера)",
             f"Разброс MAD: звук {audio['mad_ms']:.1f} ms / изображение {visual['mad_ms']:.1f} ms",
             "Это эффективная поправка игрока, не аппаратное измерение latency."]
    app.notice("Результат калибровки", lines)
    if not app.running:
        return
    if app.confirm("Сохранить калибровку?", "В config.json будут записаны обе поправки и измерения.", "Сохранить"):
        if abs(result["audio_offset_ms"]) > 500 or abs(result["visual_offset_ms"]) > 500:
            app.notice("Слишком большое смещение", ["Результат не сохранён. Повторите измерение."])
            return
        app.config.audio_offset_ms = result["audio_offset_ms"]
        app.config.visual_offset_ms = result["visual_offset_ms"]
        app.config.calibrated = True
        app.config.calibration = {**result, "output_device": app.config.output_device,
                                  "sample_rate": app.config.sample_rate,
                                  "reported_output_latency_seconds": latency,
                                  "method": "two-pass tapping; audio=A, visual=V-A",
                                  "timing_source": "SDL event timestamp if exposed; otherwise 300+ target FPS polling",
                                  "warning": "Includes human response bias. Recalibrate after changing device/monitor."}
        app.config.save(config_path)
        app.toast = "Калибровка сохранена в config.json."
