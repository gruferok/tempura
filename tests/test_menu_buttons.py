import pygame
import pytest

import main as game
from src.audio_engine import generate_test_track
from src.storage import Config
from src.ui import App

EXPECTED = {"file", "test", "snap", "calibration", "record", "play", "ai"}


@pytest.fixture
def menu(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = App(Config())
    monkeypatch.setattr(app, "frame", lambda **_: None)
    track = generate_test_track(duration=2, sample_rate=app.config.sample_rate)
    buttons = game.draw_menu(app, track, "test")
    yield app, buttons
    pygame.display.quit()


def _click(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0)


def test_menu_has_every_expected_button_on_screen(menu):
    app, buttons = menu
    assert EXPECTED <= set(buttons)
    for name, rect in buttons.items():
        assert rect.width > 0 and rect.height > 0, name
        assert app.screen.get_rect().contains(rect), f"{name} is off-screen"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_click_on_button_centre_selects_that_action(menu, name):
    _, buttons = menu
    assert game.menu_action([_click(buttons[name].center)], buttons) == name


def test_click_outside_buttons_keeps_previous_action(menu):
    _, buttons = menu
    assert game.menu_action([_click((1, 1))], buttons, default="play") == "play"


def test_unmapped_key_does_not_clear_click_in_same_frame(menu):
    _, buttons = menu
    events = [_click(buttons["ai"].center), _key(pygame.K_x)]
    assert game.menu_action(events, buttons) == "ai"


def test_mapped_keys(menu):
    _, buttons = menu
    assert game.menu_action([_key(pygame.K_1)], buttons) == "calibration"
    assert game.menu_action([_key(pygame.K_ESCAPE)], buttons) == "quit"
