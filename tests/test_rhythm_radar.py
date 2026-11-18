import numpy as np
import pygame

from src.audio_engine import AudioTrack, generate_test_track
from src.chart import Chart, Note
from src.ml.dataset import training_exclusion
from src.rhythm_radar import build_rhythm_radar
from src.storage import Config
from src.ui import App, draw_lanes


def test_radar_rms_and_attack_envelope_are_cached_readonly():
    rate = 8000
    pcm = np.zeros((rate, 2), np.float32)
    pcm[rate // 2:rate // 2 + 400] = 0.7
    radar = build_rhythm_radar(AudioTrack(pcm, rate))
    assert radar.step == 0.02
    assert len(radar.rms) == 50
    assert radar.rms[25] > radar.rms[0]
    assert radar.peaks[25] > 0
    assert not radar.rms.flags.writeable
    assert not radar.peaks.flags.writeable


def test_config_persists_orientation_and_scroll_speed(tmp_path):
    config = Config(orientation="horizontal", scroll_seconds=3.25)
    path = tmp_path / "config.json"
    config.save(path)
    restored = Config.load(path)
    assert restored.orientation == "horizontal"
    assert restored.scroll_seconds == 3.25


def test_both_orientations_headless_render():
    pygame.display.init(); pygame.font.init()
    app = App(Config())
    try:
        notes = [Note(0.4, 0), Note(0.8, 1, 0.2, "Hold")]
        for orientation in ("vertical", "horizontal"):
            app.config.orientation = orientation
            app.begin("orientation test")
            draw_lanes(app, ["f", "j"], 0.0, notes)
            app.frame(fast=False)
    finally:
        pygame.quit()


def test_explicit_training_flag_controls_snap_and_regular_chart():
    track = generate_test_track(duration=4, sample_rate=8000)
    base = Chart(track.reference(), ["f"], [Note(2.0, 0)], {})
    assert training_exclusion(base) is None  # legacy charts default to opt-in
    base.metadata.update(auto_quantized=True, human_reviewed=False)
    assert training_exclusion(base) is not None
    base.metadata["include_in_training"] = True
    assert training_exclusion(base) is None
    base.metadata["include_in_training"] = False
    assert "include_in_training=false" in training_exclusion(base)
