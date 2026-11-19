"""Native AI Studio. Pygame stays on the UI thread; ML runs in one worker.

This module can render without Torch. It is imported lazily by the menu. No
training, inference, or checkpoint loading occurs during gameplay/audio output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import queue
import threading
import time
import traceback

import pygame

from .audio_engine import AudioTrack, generate_test_track
from .chart import Chart, DEFAULT_LAYOUTS
from .ml import Cancelled, MODEL_PATH, metadata_path, read_checkpoint
from .storage import CHART_DIR, read_json
from .ui import App, BORDER, CYAN, GOLD, MUTED, RED, TEXT, VIOLET, choose_keys, file_picker, key_label


@dataclass
class StudioState:
    model_exists: bool = False
    personalized: bool = False
    dataset_count: int = 0
    skipped_count: int = 0
    device: str = "определяется…"
    track_name: str = "test_track"
    keys: list[str] = field(default_factory=lambda: ["f", "j"])
    sensitivity: float = 0.5
    busy: bool = False
    phase: str = "Готово"
    epoch: int = 0
    epochs: int = 0
    loss: float | None = None
    progress: float = 0.0
    elapsed: float = 0.0
    message: str = "Train создаст cold start даже без пользовательских карт."
    error: str = ""
    generated_path: str = ""
    generated_notes: int = 0
    profile: str = "personal_default"

    def update(self, event: dict) -> None:
        phase = event.get("phase", self.phase)
        if phase != self.phase:
            self.epoch = self.epochs = 0
            self.progress = 0
        self.phase = phase
        self.epoch = event.get("epoch", self.epoch)
        self.epochs = event.get("epochs", self.epochs)
        if "loss" in event:
            self.loss = event["loss"]
        self.message = event.get("message", self.message)
        if "device" in event:
            self.device = event["device"]
        steps = max(1, event.get("steps", 1))
        part = event.get("step", 0) / steps
        self.progress = max(0, min(1, ((self.epoch - 1 + part) / self.epochs)
                                   if self.epochs and self.epoch > 0 else part))


class BackgroundJob:
    """Bounded progress queue; terminal result cannot be lost to queue overflow."""
    def __init__(self, work, *, name="Tempura AI"):
        self.work = work
        self.name = name
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.events: queue.Queue = queue.Queue(maxsize=64)
        self.result = None
        self.error = ""
        self.cancelled = False
        self.started = 0.0
        self.thread: threading.Thread | None = None

    def _progress(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def _run(self):
        try:
            self.result = self.work(self._progress, self.cancel)
        except Cancelled:
            self.cancelled = True
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        finally:
            self.done.set()

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Задание уже запускалось.")
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self.thread.start()
        return self

    def poll(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events


def inspect_studio(charts_dir=CHART_DIR, checkpoint_path=MODEL_PATH) -> dict:
    # This runs in the worker too: importing Torch must not freeze the event loop.
    from .ml.dataset import catalog_charts
    from .ml.model import select_device

    accepted, skipped = catalog_charts(charts_dir)
    path = read_checkpoint(checkpoint_path)  # ROOT first, then BUNDLE_ROOT
    metadata = {}
    if metadata_path(path).is_file():
        try:
            metadata = read_json(metadata_path(path))
        except (OSError, ValueError):
            pass  # A sidecar is only a status hint; inference validates .pt itself.
    return {"dataset_count": len(accepted), "skipped_count": len(skipped),
            "device": str(select_device()), "model_exists": path.is_file(),
            "personalized": bool(metadata.get("personalized", False))}


def draw_ai_screen(app: App, state: StudioState) -> dict:
    app.begin("AI Studio", "Offline CRNN · 100 Hz · Live-карты / личная модель / JSON для плейтеста")
    app.text(f"01 / DATA & MODEL · profile: {state.profile}", 44, 165, 14, CYAN)
    app.text(f"Устройство: {state.device}", 44, 196, 22)
    app.text(f"Карт-кандидатов: {state.dataset_count}   /   исключено: {state.skipped_count}", 44, 232, 16, MUTED)
    model_label = ("Модель: дообучена на ваших картах" if state.personalized else "Модель: процедурный cold start") if state.model_exists else "Модель отсутствует — начните с Train"
    app.fit_text(model_label, 44, 261, 500, 16, CYAN if state.model_exists else GOLD)
    app.text("02 / TRACK", 44, 305, 14, VIOLET)
    app.fit_text(state.track_name, 44, 334, 500, 22, TEXT)
    buttons = {"file": app.button((44, 374, 235, 43), "Аудиофайл · O"),
               "test": app.button((295, 374, 235, 43), "Тестовый трек · Y")}
    for i, (count, label) in enumerate(((1, "1K"), (2, "2K"), (4, "4K"))):
        buttons[f"lanes_{count}"] = app.button((44 + i * 122, 440, 106, 43), label,
                                               selected=state.keys == DEFAULT_LAYOUTS[count])
    custom = state.keys not in DEFAULT_LAYOUTS.values()
    buttons["custom"] = app.button((410, 440, 120, 43), f"Своя {len(state.keys)}K" if custom else "Своя · C", selected=custom)
    app.fit_text("Клавиши: " + "  ".join(key_label(k) for k in state.keys), 44, 488, 486, 14, CYAN)
    app.text(f"Плотность {state.sensitivity:.2f}   /   threshold {0.8 - 0.6 * state.sensitivity:.2f}", 44, 510, 16, MUTED)
    slider = pygame.Rect(44, 548, 486, 12)
    pygame.draw.rect(app.screen, BORDER, slider, border_radius=6)
    pygame.draw.rect(app.screen, CYAN, (slider.x, slider.y, round(slider.width * state.sensitivity), slider.height), border_radius=6)
    pygame.draw.circle(app.screen, TEXT, (round(slider.x + slider.width * state.sensitivity), slider.centery), 8)
    buttons["density"] = slider.inflate(0, 18)
    app.text("Меньше нот: влево / больше: вправо", 44, 574, 14, MUTED)
    buttons["train"] = app.button((600, 180, 450, 64), "T   Train / Fine-tune on My Charts", accent=True)
    app.text("Новая: bootstrap, затем ваши карты. Далее: fine-tune.", 608, 253, 14, MUTED)
    buttons["generate"] = app.button((600, 290, 450, 64), "G   Generate AI Chart", accent=state.model_exists)
    app.text("1–8 дорожек · peak-picking · Tap / Hold", 608, 365, 14, MUTED)
    buttons["play"] = app.button((600, 403, 450, 52), "Enter   Playtest последней AI-карты")
    if state.generated_path:
        app.fit_text(f"{Path(state.generated_path).name} · {state.generated_notes} нот", 608, 469, 430, 14, CYAN)
    else:
        app.text("После генерации карта сохраняется в data/charts/.", 608, 469, 14, MUTED)
    if state.busy:
        buttons["cancel"] = app.button((600, 505, 450, 46), "Отменить задачу · Esc")
    else:
        app.text("Cold start — базовый ритм, не выученный личный стиль.", 608, 522, 14, GOLD)
        app.text("1–3 карты: нужен плейтест на новых треках.", 608, 551, 14, MUTED)
    label = state.phase + (f" · эпоха {state.epoch}/{state.epochs}" if state.epochs else "")
    if state.loss is not None:
        label += f" · loss {state.loss:.4f}"
    if state.busy:
        label += f" · {state.elapsed:.1f}s"
    app.fit_text(label, 44, 614, app.width - 88, 16, CYAN)
    pygame.draw.rect(app.screen, BORDER, (44, 647, app.width - 88, 6), border_radius=3)
    pygame.draw.rect(app.screen, CYAN, (44, 647, round((app.width - 88) * state.progress), 6), border_radius=3)
    app.fit_text(state.error or state.message, 44, 674, app.width - 88, 14, RED if state.error else MUTED)
    app.footer("T — train   G — generate   D — dataset   [ / ] — профиль   Enter — playtest   O/Y — аудио   1/2/4/C — клавиши   Esc — назад")
    return buttons


def run_ai_screen(app: App, track: AudioTrack, *, checkpoint_path=MODEL_PATH, charts_dir=CHART_DIR) -> None:
    state = StudioState(track_name=track.name)
    profile_names = ("personal_default", "style_a", "style_b")
    base_checkpoint = Path(checkpoint_path)
    managed_profiles = base_checkpoint == MODEL_PATH

    def profile_checkpoint(name: str) -> Path:
        return MODEL_PATH.with_name(f"rhythm_model_{name}.pt") if managed_profiles else base_checkpoint

    checkpoint_path = profile_checkpoint(state.profile)
    job: BackgroundJob | None = None
    kind = ""
    leave_after_job = False
    generated = None
    generated_track = None
    submitted_track = None

    def start_job(new_kind, work):
        nonlocal job, kind
        kind = new_kind
        state.busy = True
        state.error = ""
        state.phase = new_kind
        state.progress = 0
        state.epoch = state.epochs = 0
        state.loss = None
        job = BackgroundJob(work).start()

    start_job("status", lambda progress, cancel: inspect_studio(charts_dir, checkpoint_path))
    try:
        while app.running:
            if job is not None:
                state.elapsed = time.monotonic() - job.started
                for event in job.poll():
                    state.update(event)
                if job.done.is_set():
                    state.busy = False
                    if job.cancelled:
                        state.message = "Отменено. Старый чекпоинт сохранён."
                    elif job.error:
                        state.error = job.error
                        if kind == "status":
                            state.device = "ошибка диагностики (см. ниже)"
                    elif kind == "status":
                        for key, value in job.result.items():
                            setattr(state, key, value)
                        state.message = "Готово. На малом датасете качество проверяйте плейтестом."
                    elif kind == "train":
                        state.model_exists = True
                        state.personalized = job.result.metadata["personalized"]
                        state.dataset_count = job.result.metadata["current_dataset_maps"]
                        state.skipped_count = job.result.metadata["skipped_count"]
                        state.message = "Сохранено: " + ("личная модель" if state.personalized else "процедурный cold start; добавьте Live-карты")
                        state.progress = 1
                    elif kind == "generate":
                        generated, generated_track = job.result, submitted_track
                        state.generated_path = str(generated.path)
                        state.generated_notes = len(generated.chart.notes)
                        state.message = "Карта готова. Enter — плейтест; AI-карты не идут в обучение без human_reviewed=true."
                        if not generated.chart.notes:
                            state.message = "Нет пиков выше порога. Увеличьте плотность / дообучите модель и повторите."
                        state.progress = 1
                    job = None
                    if leave_after_job:
                        return
            buttons = draw_ai_screen(app, state)
            action = None
            for event in app.events():
                if event.type == pygame.KEYDOWN:
                    action = {pygame.K_t: "train", pygame.K_g: "generate", pygame.K_RETURN: "play",
                              pygame.K_o: "file", pygame.K_y: "test", pygame.K_1: "lanes_1",
                              pygame.K_2: "lanes_2", pygame.K_4: "lanes_4", pygame.K_c: "custom",
                              pygame.K_LEFT: "sparser", pygame.K_RIGHT: "denser", pygame.K_LEFTBRACKET: "profile_prev",
                              pygame.K_RIGHTBRACKET: "profile_next", pygame.K_d: "dataset", pygame.K_ESCAPE: "back"}.get(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    action = next((name for name, rect in buttons.items() if rect.collidepoint(event.pos)), None)
                    if action == "density" and not state.busy:
                        state.sensitivity = max(0, min(1, (event.pos[0] - buttons["density"].x) / buttons["density"].width))
            if not app.running:
                break
            if action in ("back", "cancel"):
                if job is not None:
                    job.cancel.set()
                    leave_after_job = leave_after_job or action == "back"
                    state.message = "Отмена на ближайшей безопасной границе…"
                else:
                    return
            if action in ("profile_prev", "profile_next") and not state.busy and managed_profiles:
                current = profile_names.index(state.profile)
                offset = -1 if action == "profile_prev" else 1
                state.profile = profile_names[(current + offset) % len(profile_names)]
                checkpoint_path = profile_checkpoint(state.profile)
                start_job("status", lambda progress, cancel: inspect_studio(charts_dir, checkpoint_path))
            if state.busy:
                app.frame(fast=False)
                continue
            try:
                if action == "dataset":
                    from .dataset_manager import run_dataset_manager
                    run_dataset_manager(app, charts_dir)
                    start_job("status", lambda progress, cancel: inspect_studio(charts_dir, checkpoint_path))
                elif action == "train":
                    def train(progress, cancel):
                        from .ml.trainer import train_model
                        return train_model(charts_dir=charts_dir, checkpoint_path=checkpoint_path,
                                           progress=progress, cancel=cancel)
                    start_job("train", train)
                elif action == "generate":
                    if not state.model_exists:
                        app.notice("Сначала обучите модель", ["Нажмите T / Train. Без Live-карт будет создан процедурный cold start."])
                    else:
                        submitted_track = track
                        keys, sensitivity = list(state.keys), state.sensitivity
                        def generate(progress, cancel):
                            from .ml.generator import generate_chart
                            return generate_chart(submitted_track, len(keys), keys=keys, sensitivity=sensitivity,
                                                  checkpoint_path=checkpoint_path, output_dir=charts_dir,
                                                  progress=progress, cancel=cancel)
                        start_job("generate", generate)
                elif action == "play" and generated is not None:
                    from .gameplay import run_gameplay
                    run_gameplay(app, generated_track, generated.chart)
                elif action == "file":
                    selected = file_picker(app)
                    if selected:
                        track = AudioTrack.from_file(selected, app.config.sample_rate)
                        state.track_name = track.name
                elif action == "test":
                    track = generate_test_track(sample_rate=app.config.sample_rate)
                    state.track_name = track.name
                elif action and action.startswith("lanes_"):
                    state.keys = list(DEFAULT_LAYOUTS[int(action[-1])])
                elif action == "custom":
                    chosen = choose_keys(app)
                    if chosen:
                        state.keys = chosen
                elif action in ("sparser", "denser"):
                    state.sensitivity = round(max(0, min(1, state.sensitivity + (0.05 if action == "denser" else -0.05))), 2)
            except Exception as error:
                state.error = str(error)
            app.frame(fast=False)
    finally:
        if job is not None:
            job.cancel.set()
            if job.thread is not None:
                job.thread.join(timeout=0.25)  # daemon + atomic publication protect exit
