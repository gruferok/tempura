#!/usr/bin/env python3
"""Native rhythm lab + offline AI CLI. ML does not need display/audio hardware."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.audio_engine import AudioTrack, generate_test_track
from src.diagnostics import environment_report
from src.storage import CHART_DIR, Config, ROOT, seed_from_bundle, write_json


def draw_menu(app, track: AudioTrack, audio_status: str = "") -> dict:
    import pygame
    from src.ui import BORDER, CYAN, GOLD, MUTED, TEXT, VIOLET

    app.begin("Музыка. Движение. Ваш ритм.", "DAC clock · Live charting · Personal AI Studio")
    app.text("01 / AUDIO", 44, 168, 14, CYAN)
    app.text(track.name, 44, 204, 28)
    app.text(f"{track.duration:.1f} s   /   {track.sample_rate:,} Hz   /   stereo float32", 46, 249, 16, MUTED)
    if track.kind == "synthetic":
        app.text("Тестовый трек · 120 BPM · первые 2 секунды — вступление", 46, 285, 16, MUTED)
    file_rect = app.button((44, 330, 245, 48), "Свой аудиофайл · O")
    test_rect = app.button((305, 330, 245, 48), "Тестовый трек · T")
    right_x = max(600, app.width - 460)
    controls = [("calibration", "1   Калибровка", "Два прохода по 16 нажатий"),
                ("record", "2   Live-разметка", "1K / 2K / 4K / своя раскладка"),
                ("play", "3   Плейтест карты", "Perfect ±25 · Great ±50 · Good ±90 ms"),
                ("ai", "4   AI Studio", "Train / Fine-tune · Generate · Playtest")]
    buttons = {"file": file_rect, "test": test_rect,
               "snap": app.button((44, 405, 506, 48), "Beat-Snap сохранённой карты · B")}
    for i, (action, title, caption) in enumerate(controls):
        rect = app.button((right_x, 165 + i * 82, 410, 54), title, accent=action == "ai")
        app.fit_text(caption, right_x + 10, rect.bottom + 6, 390, 14, MUTED)
        buttons[action] = rect
    pygame.draw.line(app.screen, BORDER, (44, 500), (app.width - 44, 500))
    app.text("02 / SYNC", 44, 530, 14, VIOLET)
    app.text(f"Audio {app.config.audio_offset_ms:+.1f} ms     Visual {app.config.visual_offset_ms:+.1f} ms",
             44, 565, 22, TEXT)
    label = "Калибровка сохранена" if app.config.calibrated else "Не откалибровано — начните с пункта 1"
    app.text(label, 45, 605, 16, CYAN if app.config.calibrated else GOLD)
    app.fit_text(audio_status, 45, 640, app.width - 90, 14)
    if app.toast:
        app.fit_text(app.toast, 45, app.height - 77, app.width - 90, 14, CYAN)
    app.footer("1 — калибровка   2 — запись   3 — плейтест   4/A — AI Studio   B — Beat-Snap   O/T — аудио   Esc — выход")
    return buttons


def track_for_chart(app, chart) -> AudioTrack | None:
    from src.ui import file_picker

    reference = chart.track
    if reference["kind"] == "synthetic":
        if reference.get("generator_version") != 1:
            raise ValueError("Неизвестная версия генератора синтетического трека.")
        return generate_test_track(duration=reference["duration"], sample_rate=reference["sample_rate"],
                                   bpm=reference["bpm"], lead_in=reference["lead_in"])
    source = Path(reference.get("audio_path") or "__missing_audio__").expanduser()
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        app.notice("Нужен исходный аудиофайл", ["Путь в карте недоступен. Выберите тот же аудиофайл.",
                                               "Идентичность проверяется по SHA-256, не только по названию."])
        selected = file_picker(app)
        if selected is None:
            return None
        source = selected
    return AudioTrack.from_file(source, app.config.sample_rate)


def headless_smoke(config: Config, screenshot: str | None = None) -> dict:
    """Draw screens without Torch, physical playback, or latency/FPS claims."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    from src.ai_screen import StudioState, draw_ai_screen
    from src.calibration import TapCollector, draw_calibration
    from src.chart import Chart, Note
    from src.gameplay import Judge, draw_gameplay
    from src.live_mapper import LiveRecorder, draw_mapper
    from src.snap_screen import SnapState, draw_snap_screen
    from src.ui import App

    app = App(config)
    try:
        track = generate_test_track(duration=8, sample_rate=config.sample_rate)
        notes = [Note(2, 0), Note(3, 1, 0.75, "Hold"), Note(4, 2), Note(4.5, 3)]
        chart = Chart(track.reference(), ["d", "f", "j", "k"], notes)
        judge = Judge(chart)
        recorder = LiveRecorder(chart.keys, track.duration, notes=list(notes))
        draw_calibration(app, "audio", TapCollector(), 3)
        draw_calibration(app, "visual", TapCollector(), 3.1)
        draw_mapper(app, recorder, track, 2.4)
        draw_gameplay(app, chart, judge, 2.4, set())
        draw_ai_screen(app, StudioState(device="CPU · headless render only", track_name=track.name))
        draw_snap_screen(app, SnapState(source_name="test_track_4K.json", total_notes=len(notes)))
        draw_menu(app, track, "HEADLESS: audio/device latency not measured")
        pygame.display.flip()
        if screenshot:
            path = Path(screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(app.screen, path)
        return {"rendered_screens": ["menu", "audio_calibration", "visual_calibration", "live_mapper", "gameplay", "ai_studio", "beat_snap"],
                "audio_stream_opened": False, "physical_latency_measured": False}
    finally:
        pygame.quit()


def run_menu(config: Config, audio_path: str | None = None, chart_path: str | None = None) -> None:
    import pygame
    from src.calibration import run_calibration
    from src.chart import Chart
    from src.gameplay import run_gameplay
    from src.live_mapper import run_live_mapper
    from src.snap_screen import run_snap_screen
    from src.ui import App, choose_keys, file_picker

    app = App(config)
    pygame.key.set_repeat()
    try:
        track = AudioTrack.from_file(audio_path, config.sample_rate) if audio_path else generate_test_track(sample_rate=config.sample_rate)
        env = environment_report()
        outputs = env["audio"]["output_devices"]
        audio_status = (f"Аудиовыходов: {len(outputs)}  ·  target {config.fps} FPS  ·  Pygame mixer выключен"
                        if outputs else "Аудиовыход не найден. Меню работает; звук требует локального устройства. См. --diagnose.")
        launch_chart = Path(chart_path) if chart_path else None
        while app.running:
            buttons = draw_menu(app, track, audio_status)
            action = "play" if launch_chart else None
            for event in app.events():
                if event.type == pygame.KEYDOWN:
                    action = {pygame.K_1: "calibration", pygame.K_2: "record", pygame.K_3: "play",
                              pygame.K_4: "ai", pygame.K_a: "ai", pygame.K_b: "snap", pygame.K_o: "file",
                              pygame.K_t: "test", pygame.K_ESCAPE: "quit"}.get(event.key)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    hit = next((name for name, rect in buttons.items() if rect.collidepoint(event.pos)), None)
                    if hit is not None:
                        action = hit
            if not app.running:
                break
            try:
                if action == "quit":
                    break
                if action == "test":
                    track = generate_test_track(sample_rate=config.sample_rate)
                elif action == "file":
                    selected = file_picker(app)
                    if selected:
                        track = AudioTrack.from_file(selected, config.sample_rate)
                elif action == "calibration":
                    run_calibration(app)
                elif action == "record":
                    keys = choose_keys(app)
                    if keys:
                        recorded = run_live_mapper(app, track, keys)
                        if recorded is not None and app.running:
                            recorded_chart, recorded_path = recorded
                            run_snap_screen(app, track, recorded_chart, recorded_path)
                elif action == "snap":
                    selected = file_picker(app, charts=True)
                    if selected:
                        chart = Chart.load(selected)
                        chart_track = track_for_chart(app, chart)
                        if chart_track:
                            run_snap_screen(app, chart_track, chart, selected)
                elif action == "ai":
                    from src.ai_screen import run_ai_screen
                    run_ai_screen(app, track)
                elif action == "play":
                    selected, launch_chart = launch_chart, None
                    if selected is None:
                        selected = file_picker(app, charts=True)
                    if selected:
                        chart = Chart.load(selected)
                        chart_track = track_for_chart(app, chart)
                        if chart_track:
                            run_gameplay(app, chart_track, chart)
            except Exception as error:
                print(f"{type(error).__name__}: {error}", file=sys.stderr)
                app.notice("Не удалось выполнить действие", [str(error)[i:i + 100] for i in range(0, len(str(error)), 100)])
            app.frame(fast=False)
    finally:
        pygame.quit()


def run_ml_cli(args) -> dict:
    """No pygame import/window and no PortAudio stream is required."""
    from src.ml import FEATURE_CACHE

    def progress(event):
        print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

    result = {}
    cache = None if args.no_cache else FEATURE_CACHE
    if args.train:
        from src.ml.trainer import train_model
        trained = train_model(charts_dir=args.charts_dir, checkpoint_path=args.model,
                              epochs=args.epochs, bootstrap_epochs=args.bootstrap_epochs,
                              bootstrap_samples=args.bootstrap_samples, batch_size=args.batch_size,
                              learning_rate=args.learning_rate, window_seconds=args.window_seconds,
                              validation_fraction=args.validation_fraction, device=args.ml_device,
                              reset=args.reset_model, seed=args.seed, cache_dir=cache, progress=progress)
        result["training"] = trained.to_dict()
    if args.generate:
        from src.ml.generator import generate_chart
        track = (generate_test_track() if args.generate == "synthetic"
                 else AudioTrack.from_file(args.generate))
        generated = generate_chart(track, args.lanes, checkpoint_path=args.model, threshold=args.threshold,
                                   sensitivity=args.sensitivity, keys=args.keys, device=args.ml_device,
                                   output_dir=args.charts_dir, output_path=args.output, cache_dir=cache, progress=progress)
        result["generation"] = generated.to_dict()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tempura: DAC-clocked rhythm lab + offline AI Studio")
    parser.add_argument("--audio", help="Предварительно загрузить аудиофайл в меню")
    parser.add_argument("--chart", help="Сразу открыть JSON-карту в плейтесте")
    parser.add_argument("--device", help="Индекс/имя аудиовыхода PortAudio (не ML-устройство)")
    parser.add_argument("--diagnose", action="store_true", help="Локальная диагностика без окна")
    parser.add_argument("--report", help="Сохранить JSON-диагностику (--diagnose)")
    parser.add_argument("--headless-smoke", action="store_true", help="Проверить рендер без дисплея/аудио")
    parser.add_argument("--screenshot", help="PNG headless-меню (--headless-smoke)")
    ml = parser.add_argument_group("Offline AI (дисплей и аудиоустройство не нужны)")
    ml.add_argument("--train", action="store_true", help="Cold start / fine-tune на пользовательских картах")
    ml.add_argument("--generate", metavar="AUDIO", help="Аудиофайл или synthetic → AI-карта")
    ml.add_argument("--lanes", type=int, choices=range(1, 9), default=2)
    ml.add_argument("--keys", nargs="+", help="Свои клавиши слева направо; длина равна --lanes")
    density = ml.add_mutually_exclusive_group()
    density.add_argument("--threshold", type=float, default=0.55, help="Порог пиков (0..1): выше → меньше нот")
    density.add_argument("--sensitivity", type=float, help="Плотность (0..1): выше → больше нот")
    ml.add_argument("--model", default=str(ROOT / "data" / "models" / "rhythm_model.pt"))
    ml.add_argument("--charts-dir", default=str(CHART_DIR))
    ml.add_argument("--output", help="Имя выходного JSON (--generate)")
    ml.add_argument("--ml-device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ml.add_argument("--epochs", type=int, default=8, help="Эпох fine-tune")
    ml.add_argument("--bootstrap-epochs", type=int, default=3, help="Процедурный претрейн новой модели; 0 выключает")
    ml.add_argument("--bootstrap-samples", type=int, default=12)
    ml.add_argument("--batch-size", type=int, default=4)
    ml.add_argument("--learning-rate", type=float, default=0.001)
    ml.add_argument("--window-seconds", type=float, default=10.0)
    ml.add_argument("--validation-fraction", type=float, default=0.0, help="0 = обучать на всех; holdout .2 требует >=3 разных треков")
    ml.add_argument("--seed", type=int, default=1234)
    ml.add_argument("--reset-model", action="store_true", help="Обучить новую модель; старый файл заменяется только при успехе")
    ml.add_argument("--no-cache", action="store_true", help="Не читать/не писать кэш признаков")
    args = parser.parse_args(argv)
    offline_ai = args.train or args.generate is not None
    if (args.diagnose and (args.headless_smoke or offline_ai)) or (args.headless_smoke and offline_ai):
        parser.error("Выберите диагностику, headless-рендер или offline AI.")
    if args.output and args.generate is None:
        parser.error("--output используется с --generate.")
    if offline_ai and (args.chart or args.audio):
        parser.error("Offline AI использует --generate AUDIO; --audio/--chart предназначены для меню.")
    if args.diagnose:
        report = environment_report()
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    try:
        if offline_ai:
            print(json.dumps(run_ml_cli(args), ensure_ascii=False, indent=2))
            return 0
        seed_from_bundle()
        config = Config.load()
        if args.device is not None:
            device = int(args.device) if args.device.isdigit() else args.device
            if device != config.output_device:
                config.calibrated = False
            config.output_device = device
        if args.headless_smoke:
            print(json.dumps(headless_smoke(config, args.screenshot), ensure_ascii=False, indent=2))
        else:
            run_menu(config, args.audio, args.chart)
        return 0
    except KeyboardInterrupt:
        print("Tempura: операция прервана; сохранённые файлы остаются атомарными.", file=sys.stderr)
        return 130
    except Exception as error:
        hint = ("Проверьте requirements-ml.txt, пути аудио/карт и --ml-device. См. README."
                if offline_ai else "Запустите python main.py --diagnose. Для игры нужны локальные дисплей и аудиовыход.")
        print(f"Tempura: {error}\n{hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
