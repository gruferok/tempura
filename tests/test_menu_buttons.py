import pygame
import pytest

import main as game
from src.audio_engine import generate_test_track
from src.storage import Config
from src.ui import App, ButtonHitArea, button_hover, wrap_notice_lines

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


def test_disabled_button_has_no_hover_and_does_not_change_action(menu):
    _, buttons = menu
    rect = buttons["ai"]
    pos = rect.center
    disabled = ButtonHitArea(rect, disabled=True)
    assert not button_hover(rect, pos, disabled=True)

    buttons = dict(buttons, ai=disabled)
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    posted = pygame.event.get(pygame.MOUSEBUTTONDOWN)[0]
    assert game.menu_action([posted], buttons, default="play") == "play"


def test_notice_wrap_preserves_text_and_breaks_only_long_tokens():
    text = "Калибровка завершилась: file:///very/long/path/to/audio.flac; retry?  yes!"
    rows = wrap_notice_lines([text], max_chars=18)
    assert "".join(rows) == text
    assert all(len(row) <= 18 for row in rows)
    assert any(row.endswith(" ") for row in rows[:-1])


def test_notice_wrap_uses_render_font_width():
    class TenPixelFont:
        def size(self, value):
            return (len(value) * 10, 12)

    text = "one two three /very/long/path"
    rows = wrap_notice_lines([text], max_chars=72, max_width=40, font=TenPixelFont())
    assert "".join(rows) == text
    assert all(len(row) <= 4 for row in rows)
