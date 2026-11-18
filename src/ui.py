"""Shared native Pygame UI. No pygame.mixer; audio uses one PortAudio stream."""
from __future__ import annotations

import math
from pathlib import Path

import pygame

from .audio_engine import AudioEngine, AudioTrack
from .chart import DEFAULT_LAYOUTS, RESERVED_KEYS, Note, validate_keys
from .storage import Config, ROOT

BG = (12, 16, 28)
PANEL = (22, 29, 47)
BORDER = (47, 61, 84)
TEXT = (235, 241, 253)
MUTED = (146, 161, 189)
CYAN = (81, 222, 209)
VIOLET = (168, 141, 255)
RED = (255, 113, 139)
GOLD = (255, 205, 111)
LANE_COLORS = [CYAN, VIOLET, (105, 178, 255), (255, 158, 180), GOLD, (154, 228, 137), CYAN, VIOLET]


def key_label(name: str) -> str:
    return "SPACE" if name == "space" else name.upper()


def event_snapshot(engine: AudioEngine, event) -> tuple[float, str]:
    """Use an SDL event timestamp if exposed, otherwise timestamp at queue polling.

    Tick subtraction handles SDL's 32-bit wrap. Look up the historic DAC segment
    rather than subtracting age from song_time across pause/seek boundaries.
    """
    stamp = getattr(event, "timestamp", None)
    age = 0.0
    if isinstance(stamp, (int, float)) and math.isfinite(stamp):
        ticks = (pygame.time.get_ticks() - int(stamp)) & 0xFFFFFFFF
        if ticks <= 1000:
            age = ticks / 1000.0
    return engine.snapshot(seconds_ago=age)


class App:
    def __init__(self, config: Config, *, engine_factory=AudioEngine):
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE, vsync=0)
        pygame.display.set_caption("Tempura · rhythm lab")
        self.clock = pygame.time.Clock()
        self.fonts = {size: pygame.font.SysFont("dejavusans,arial", size) for size in (14, 16, 18, 22, 28, 36, 48)}
        self.config, self.engine_factory = config, engine_factory
        self.running = True
        self.toast = ""

    @property
    def width(self) -> int:
        return self.screen.get_width()

    @property
    def height(self) -> int:
        return self.screen.get_height()

    def engine(self, track: AudioTrack) -> AudioEngine:
        c = self.config
        return self.engine_factory(track, device=c.output_device, latency=c.latency,
                                   blocksize=c.blocksize, volume=c.volume)

    def text(self, value: str, x: float, y: float, size=18, color=TEXT, *, center=False):
        surface = self.fonts[size].render(str(value), True, color)
        rect = surface.get_rect(center=(round(x), round(y))) if center else surface.get_rect(topleft=(round(x), round(y)))
        self.screen.blit(surface, rect)
        return rect

    def fit_text(self, value, x, y, width, size=16, color=MUTED):
        value = str(value)
        font = self.fonts[size]
        if font.size(value)[0] > width:
            while value and font.size(value + "…")[0] > width:
                value = value[:-1]
            value = value + "…" if value else ""
        return self.text(value, x, y, size, color)

    def begin(self, title: str, subtitle: str = "") -> None:
        self.screen.fill(BG)
        self.text("TEMPURA / RHYTHM LAB", 40, 24, 14, CYAN)
        self.text(title, 40, 52, 36)
        self.fit_text(subtitle, 42, 105, self.width - 84)
        pygame.draw.line(self.screen, BORDER, (40, 140), (self.width - 40, 140))

    def button(self, rect, label: str, *, accent=False, selected=False):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos())
        fill = (35, 55, 68) if hover or selected else PANEL
        pygame.draw.rect(self.screen, fill, rect, border_radius=12)
        pygame.draw.rect(self.screen, CYAN if accent or selected else BORDER, rect, width=1, border_radius=12)
        self.text(label, rect.centerx, rect.centery, 18, CYAN if accent else TEXT, center=True)
        return rect

    def footer(self, value: str) -> None:
        self.fit_text(value, 40, self.height - 36, self.width - 80, 14)

    def events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
        return events

    def frame(self, *, fast=True) -> None:
        pygame.display.flip()
        if fast:
            self.clock.tick_busy_loop(self.config.fps)
        else:
            self.clock.tick(90)

    def notice(self, title: str, lines: list[str]) -> None:
        while self.running:
            self.begin(title)
            for i, line in enumerate(lines):
                self.fit_text(line, 48, 190 + i * 35, self.width - 96, 18, TEXT)
            back = self.button((48, self.height - 110, 230, 48), "Вернуться · Enter", accent=True)
            for event in self.events():
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and back.collidepoint(event.pos):
                    return
            self.frame(fast=False)

    def confirm(self, title: str, detail: str, yes="Да") -> bool:
        while self.running:
            self.begin(title, detail)
            accept = self.button((48, 210, 260, 60), yes + " · Enter", accent=True)
            cancel = self.button((330, 210, 220, 60), "Отмена · Esc")
            for event in self.events():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return True
                    if event.key == pygame.K_ESCAPE:
                        return False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if accept.collidepoint(event.pos):
                        return True
                    if cancel.collidepoint(event.pos):
                        return False
            self.frame(fast=False)
        return False


def file_picker(app: App, *, charts=False) -> Path | None:
    """Built-in file browser: does not depend on Tk, a desktop shell or a web app."""
    directory = ROOT / "data" / ("charts" if charts else "audio")
    suffixes = {".json"} if charts else {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".opus"}
    selected, top, error = 0, 0, ""
    entries: list[Path] = []

    def refresh():
        nonlocal entries, selected, top, error
        try:
            entries = sorted((p for p in directory.iterdir() if not p.name.startswith(".") and
                              (p.is_dir() or p.suffix.lower() in suffixes)),
                             key=lambda p: (not p.is_dir(), p.name.casefold()))
            selected = top = 0
            error = ""
        except OSError as exception:
            entries = []
            error = str(exception)

    refresh()
    while app.running:
        app.begin("Выбор карты" if charts else "Выбор аудиофайла", str(directory))
        visible = max(1, (app.height - 310) // 38)
        top = max(0, min(top, selected))
        if selected >= top + visible:
            top = selected - visible + 1
        row_rects = []
        for j, item in enumerate(entries[top:top + visible]):
            index = top + j
            rect = pygame.Rect(42, 160 + j * 38, app.width - 84, 34)
            if index == selected:
                pygame.draw.rect(app.screen, PANEL, rect, border_radius=6)
            app.fit_text(("[DIR]  " if item.is_dir() else "       ") + item.name,
                         52, rect.y + 5, rect.width - 20, 16, CYAN if item.is_dir() else TEXT)
            row_rects.append((rect, index))
        if not entries:
            app.text(error or "Папка пуста. Поднимитесь к родительской папке.", 48, 180, 16, MUTED)
        up = app.button((42, app.height - 125, 235, 48), "Выше · Backspace")
        choose = app.button((295, app.height - 125, 210, 48), "Открыть · Enter", accent=True)
        cancel = app.button((525, app.height - 125, 180, 48), "Назад · Esc")
        app.footer("↑ ↓ — выбор   Enter / двойной клик — открыть   Backspace — выше")
        activate = False
        for event in app.events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_BACKSPACE:
                    directory = directory.parent
                    refresh()
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    selected = max(0, min(len(entries) - 1, selected + (1 if event.key == pygame.K_DOWN else -1)))
                elif event.key == pygame.K_RETURN:
                    activate = True
            elif event.type == pygame.MOUSEWHEEL:
                selected = max(0, min(len(entries) - 1, selected - event.y * 3))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if up.collidepoint(event.pos):
                    directory = directory.parent
                    refresh()
                elif cancel.collidepoint(event.pos):
                    return None
                elif choose.collidepoint(event.pos):
                    activate = True
                else:
                    for rect, index in row_rects:
                        if rect.collidepoint(event.pos):
                            selected = index
                            activate = getattr(event, "clicks", 1) >= 2
        if activate and entries:
            item = entries[selected]
            if item.is_dir():
                directory = item
                refresh()
            else:
                return item
        app.frame(fast=False)
    return None


def choose_keys(app: App) -> list[str] | None:
    custom: list[str] | None = None
    error = ""
    while app.running:
        app.begin("Раскладка дорожек", "Стандартная раскладка или от 1 до 8 собственных клавиш")
        buttons = []
        for i, (count, keys) in enumerate(DEFAULT_LAYOUTS.items()):
            rect = app.button((45 + i * 250, 190, 230, 76), f"{count}K  /  {' '.join(key_label(k) for k in keys)}")
            buttons.append((rect, keys))
        custom_rect = app.button((45, 295, 310, 60), "Своя раскладка · C", accent=custom is not None)
        if custom is not None:
            app.text("Нажимайте клавиши слева направо, затем Enter.", 45, 395, 18)
            app.text("   ".join(key_label(k) for k in custom) or "…", 45, 440, 28, CYAN)
            app.fit_text(error, 45, 500, app.width - 90, 16, RED)
        app.footer("1 / 2 / 4 — стандарт   C — своя   Backspace — удалить клавишу   Esc — назад")
        for event in app.events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if custom is None:
                    if event.key in (pygame.K_1, pygame.K_2, pygame.K_4):
                        return list(DEFAULT_LAYOUTS[int(event.unicode)])
                    if event.key == pygame.K_c:
                        custom = []
                else:
                    if event.key == pygame.K_BACKSPACE:
                        custom = custom[:-1]
                    elif event.key == pygame.K_RETURN:
                        try:
                            validate_keys(custom)
                            return custom
                        except ValueError as exception:
                            error = str(exception)
                    else:
                        key = pygame.key.name(event.key)
                        if key not in RESERVED_KEYS and key not in custom and len(custom) < 8:
                            custom.append(key)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if custom_rect.collidepoint(event.pos):
                    custom = []
                for rect, keys in buttons:
                    if rect.collidepoint(event.pos):
                        return list(keys)
        app.frame(fast=False)
    return None


def draw_lanes(app: App, keys: list[str], song_time: float, notes: list[Note],
               *, held: set[int] | None = None, states: list[str] | None = None) -> None:
    held = held or set()
    left, top = 180, 160
    field_width, field_height = min(660, app.width - 350), app.height - 280
    field_width = max(180, field_width)
    line_x, line_y = left, top + field_height
    horizontal = app.config.orientation == "horizontal"
    lane_extent = field_height if horizontal else field_width
    lane_size = lane_extent / len(keys)
    pixel_span = field_width if horizontal else field_height
    speed = max(80, pixel_span / app.config.scroll_seconds)
    for lane, key in enumerate(keys):
        if horizontal:
            rect = pygame.Rect(left, round(top + lane * lane_size), field_width + 30, round(lane_size - 3))
            pygame.draw.rect(app.screen, (30, 48, 62) if lane in held else PANEL, rect, border_radius=8)
            color = LANE_COLORS[lane]
            pygame.draw.line(app.screen, color, (line_x, rect.top + 4), (line_x, rect.bottom - 4), 3)
            app.text(key_label(key), line_x - 30, rect.centery, 18, color, center=True)
        else:
            rect = pygame.Rect(round(left + lane * lane_size), top, round(lane_size - 3), field_height + 64)
            pygame.draw.rect(app.screen, (30, 48, 62) if lane in held else PANEL, rect, border_radius=8)
            color = LANE_COLORS[lane]
            pygame.draw.line(app.screen, color, (rect.left + 5, line_y), (rect.right - 5, line_y), 3)
            app.text(key_label(key), rect.centerx, line_y + 35, 18, color, center=True)
    old_clip = app.screen.get_clip()
    app.screen.set_clip(pygame.Rect(left, top, field_width + 30, field_height + 20))
    for i, note in enumerate(notes):
        state = states[i] if states else "pending"
        if state in ("hit", "miss") or note.time - song_time > app.config.scroll_seconds + 0.2 or note.end < song_time - 0.3:
            continue
        color = LANE_COLORS[note.lane]
        if horizontal:
            y = round(top + note.lane * lane_size + lane_size / 2)
            x = line_x + (note.time - song_time) * speed
            if note.type == "Hold":
                tail = line_x + (note.end - song_time) * speed
                pygame.draw.line(app.screen, tuple(c // 2 for c in color), (round(x), y), (round(tail), y), max(5, round(lane_size / 3)))
            pygame.draw.circle(app.screen, color, (round(x), y), max(7, round(lane_size / 5)))
        else:
            x = round(left + note.lane * lane_size + 10)
            width = max(8, round(lane_size - 23))
            y = line_y - (note.time - song_time) * speed
            if note.type == "Hold":
                tail_y = line_y - (note.end - song_time) * speed
                head = min(y, line_y) if state == "holding" else y
                body = pygame.Rect(x + width // 4, round(tail_y), max(5, width // 2), max(5, round(head - tail_y)))
                pygame.draw.rect(app.screen, tuple(c // 2 for c in color), body, border_radius=5)
                pygame.draw.rect(app.screen, color, (x, round(tail_y), width, 6), border_radius=3)
                y = head
            pygame.draw.rect(app.screen, color, (x, round(y - 8), width, 16), border_radius=5)
    app.screen.set_clip(old_clip)


def transport_hud(app: App, engine: AudioEngine, song_time: float) -> None:
    width = app.width - 80
    progress = max(0.0, min(1.0, song_time / engine.track.duration))
    pygame.draw.rect(app.screen, BORDER, (40, app.height - 62, width, 4), border_radius=2)
    pygame.draw.rect(app.screen, CYAN, (40, app.height - 62, round(width * progress), 4), border_radius=2)
    app.fit_text(f"{song_time:06.2f} / {engine.track.duration:.2f}s  ·  {engine.state.upper()}",
                 42, app.height - 90, app.width - 400, 14)
    app.text(f"{app.clock.get_fps():.0f} FPS · underrun {engine.underruns}", app.width - 300, 25, 14,
             RED if engine.underruns else MUTED)
