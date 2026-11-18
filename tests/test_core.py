"""No physical audio/display is needed. A fake stream exercises the REAL callback.

These tests verify logic, not end-to-end device latency or human calibration.
"""
from __future__ import annotations

from dataclasses import replace
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio_engine import (AudioEngine, AudioError, AudioTrack, ClockAnchor,
                              generate_test_track, metronome_track, position_at, synth_click)
from src.calibration import TapCollector, estimate_offsets, robust_summary
from src.chart import Chart, Note, validate_keys
from src.gameplay import Judge, grade
from src.live_mapper import LiveRecorder
from src.storage import Config


class FakeStream:
    """PortAudio-shaped stream with a manually advanced device clock."""
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.rate = kwargs["samplerate"]
        self.kwargs = kwargs
        self._time = 100.0
        self.latency = 0.04
        self.active = False
        self.closed = False
        self.inside_callback = False

    @property
    def time(self):
        assert not self.inside_callback, "PortAudio get_time called inside callback"
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    def start(self):
        self.active = True

    def abort(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True

    def render(self, frames=4800, dac=100.1, *, underflow=False):
        output = np.full((frames, 2), np.nan, dtype=np.float32)
        self.inside_callback = True
        try:
            self.callback(output, frames, SimpleNamespace(outputBufferDacTime=dac, currentTime=self._time),
                          SimpleNamespace(output_underflow=underflow))
        finally:
            self.inside_callback = False
        assert np.isfinite(output).all(), "Callback did not fill all frames"
        return output


def silence(duration=2, rate=48000):
    return AudioTrack(np.zeros((round(duration * rate), 2), np.float32), rate)


def start_engine(track=None, **kwargs):
    engine = AudioEngine(track or silence(), stream_factory=FakeStream, **kwargs)
    engine.play()
    return engine, engine._stream


@pytest.mark.parametrize("module", ["main", "src.audio_engine", "src.calibration", "src.live_mapper",
                                    "src.gameplay", "src.chart", "src.storage", "src.ui", "src.diagnostics",
                                    "src.beat_snap", "src.snap_screen",
                                    "pygame", "numpy", "soundfile", "librosa"])
def test_imports_have_no_device_or_window_side_effects(module):
    assert importlib.import_module(module) is not None


def test_sounddevice_installed_and_imports_with_native_runtime():
    # Native PortAudio is an OS prerequisite, not a prerequisite for core math.
    import ctypes.util
    if not ctypes.util.find_library("portaudio"):
        pytest.skip("No native PortAudio; core modules still import without it")
    assert importlib.import_module("sounddevice") is not None


def test_generate_track_is_finite_stereo_deterministic_and_beats_aligned():
    track = generate_test_track(duration=4, sample_rate=8000)
    other = generate_test_track(duration=4, sample_rate=8000)
    assert track.pcm.shape == (32000, 2)
    assert track.pcm.dtype == np.float32 and track.pcm.flags.c_contiguous
    assert not track.pcm.flags.writeable
    assert track.duration == 4 and track.bpm == 120
    assert np.isfinite(track.pcm).all() and np.max(np.abs(track.pcm)) < 1
    assert np.array_equal(track.pcm, other.pcm)
    assert np.count_nonzero(track.pcm[:16000]) == 0  # first beat at 2 seconds
    for second in (2, 2.5, 3, 3.5):
        start = round(second * 8000)
        assert np.max(np.abs(track.pcm[start:start + 320])) > 0.15
    assert track.reference()["generator_version"] == 1


@pytest.mark.parametrize("kwargs", [{"duration": 0}, {"duration": math.nan}, {"bpm": 0}, {"sample_rate": 0}])
def test_invalid_synthetic_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_test_track(**kwargs)


def test_soundfile_load_and_offline_resample(tmp_path):
    import soundfile as sf
    path = tmp_path / "имя.wav"
    samples = np.sin(2 * np.pi * 440 * np.arange(2400) / 24000).astype(np.float32) * 0.2
    sf.write(path, samples, 24000, subtype="FLOAT")
    track = AudioTrack.from_file(path, 48000)
    assert track.sample_rate == 48000 and track.pcm.shape == (4800, 2)
    assert track.duration == pytest.approx(0.1)
    assert np.allclose(track.pcm[:, 0], track.pcm[:, 1])
    same_file = AudioTrack.from_file(path, 24000)
    assert same_file.track_id == track.track_id  # file identity is not sample-rate dependent
    assert track.reference()["kind"] == "file"
    with pytest.raises(AudioError):
        AudioTrack.from_file(tmp_path / "missing.wav")


def test_invalid_pcm_rejected():
    for pcm in [np.array([]), np.zeros((3, 3)), np.array([math.nan])]:
        with pytest.raises(ValueError):
            AudioTrack(pcm, 48000)


def test_dac_clock_waits_for_actual_start_and_interpolates():
    engine, stream = start_engine()
    assert engine.state == "starting" and engine.song_time == 0
    assert engine.transport_pending
    stream.render(dac=100.1)
    stream.time = 100.05
    assert engine.song_time == 0 and engine.state == "starting"
    assert engine.transport_pending
    stream.time = 100.125
    assert engine.song_time == pytest.approx(0.025)
    assert not engine.transport_pending
    # Not elapsed from stream creation, and NOT subtracting latency twice.
    assert engine.song_time != pytest.approx(0.125)
    assert engine.output_latency == 0.04
    stream.render(dac=100.2)
    stream.time = 100.255
    assert engine.song_time == pytest.approx(0.155)
    engine.close()
    assert stream.closed


def test_pause_resume_stop_follow_presentation_boundaries():
    engine, stream = start_engine()
    stream.render(dac=100.1)
    stream.time = 100.12
    engine.pause()
    stream.render(dac=100.2)
    stream.time = 100.18
    assert engine.state == "playing" and engine.song_time == pytest.approx(0.08)
    stream.time = 100.5
    assert engine.state == "paused" and engine.song_time == pytest.approx(0.1)
    stream.time = 100.9
    engine.play()
    stream.render(dac=101.0)
    assert engine.state == "paused"
    stream.time = 101.02
    assert engine.song_time == pytest.approx(0.12)
    engine.stop()
    stream.render(dac=101.1)
    stream.time = 101.05
    assert engine.song_time == pytest.approx(0.15)
    stream.time = 101.2
    assert engine.state == "stopped" and engine.song_time == 0
    engine.close()


def test_seek_does_not_apply_a_future_anchor_early():
    engine, stream = start_engine()
    stream.render(dac=100.1)
    stream.time = 100.15
    engine.seek(1.0)
    assert engine.transport_pending
    stream.render(dac=100.2)
    stream.time = 100.19
    assert engine.song_time == pytest.approx(0.09)
    assert engine.transport_pending
    stream.time = 100.21
    assert engine.song_time == pytest.approx(1.01)
    assert not engine.transport_pending
    engine.close()


def test_timestamped_input_looks_up_old_segment_but_rejects_pre_seek_epoch():
    engine, stream = start_engine()
    stream.render(dac=100.1)
    stream.time = 100.18
    assert engine.snapshot(seconds_ago=0.02)[0] == pytest.approx(0.06)
    engine.seek(1)
    stream.render(dac=100.2)
    stream.time = 100.22
    old_time, state = engine.snapshot(seconds_ago=0.03)
    assert old_time == pytest.approx(0.09) and state == "transition"
    engine.close()


def test_seek_while_paused_and_initial_seek():
    engine = AudioEngine(silence(), stream_factory=FakeStream)
    engine.seek(0.5)
    assert engine.song_time == 0.5
    engine.play()
    stream = engine._stream
    stream.render(dac=100.1)
    stream.time = 100.15
    assert engine.song_time == pytest.approx(0.55)
    engine.pause()
    stream.render(dac=100.2)
    engine.seek(1.5)
    stream.render(dac=100.3)
    stream.time = 100.4
    assert engine.state == "paused" and engine.song_time == 1.5
    engine.close()


def test_stop_before_open_resets_initial_seek():
    engine = AudioEngine(silence(), stream_factory=FakeStream)
    engine.seek(1.0)
    engine.stop()
    engine.play()
    engine._stream.render(dac=100.1)
    engine._stream.time = 100.11
    assert engine.song_time == pytest.approx(0.01)
    engine.close()


def test_end_of_track_zeros_partial_buffer_and_waits_for_dac():
    track = AudioTrack(np.full((240, 2), 0.2, np.float32), 48000)
    engine, stream = start_engine(track, volume=1)
    out = stream.render(frames=480, dac=100.1)
    assert np.all(out[:240] == np.float32(0.2)) and np.all(out[240:] == 0)
    stream.time = 100.102
    assert engine.state == "playing"
    stream.time = 100.106
    assert engine.state == "finished" and engine.song_time == track.duration
    assert np.all(stream.render(frames=480, dac=100.11) == 0)
    engine.close()


def test_underflow_count_and_reanchor():
    engine, stream = start_engine()
    stream.render(dac=100.1)
    stream.render(dac=100.25, underflow=True)
    stream.time = 100.27
    assert engine.underruns == 1
    assert engine.song_time == pytest.approx(0.12)
    engine.close()


@pytest.mark.parametrize("bad_timestamp", [0, math.nan, -1])
def test_bad_dac_time_is_not_silently_replaced_by_wall_clock(bad_timestamp):
    engine, stream = start_engine()
    assert np.all(stream.render(dac=bad_timestamp) == 0)
    with pytest.raises(AudioError, match="outputBufferDacTime"):
        engine.check_health()
    engine.close()


def test_backwards_device_clock_is_detected():
    engine, stream = start_engine()
    stream.render(dac=100.2)
    stream.render(dac=100.1)
    assert engine.clock_error
    engine.close()


def test_callback_error_reaches_ui_and_fills_output(monkeypatch):
    engine, stream = start_engine()
    stream.render(dac=100.1)
    engine.trigger_hit()
    def broken(*_):
        raise RuntimeError("test failure")
    monkeypatch.setattr(engine, "_mix_at", broken)
    assert np.all(stream.render(dac=100.2) == 0)
    with pytest.raises(AudioError, match="test failure"):
        engine.check_health()
    engine.close()


def test_open_failure_is_actionable_and_context_closes():
    def fail(**_):
        raise RuntimeError("no output device")
    engine = AudioEngine(silence(), stream_factory=fail)
    with pytest.raises(AudioError, match="output_device"):
        engine.play()
    engine.close()
    with AudioEngine(silence(), stream_factory=FakeStream) as other:
        other.play()
        stream = other._stream
    assert stream.closed


def test_metronome_is_sample_scheduled_inside_callback_and_hit_spills():
    engine, stream = start_engine(volume=1)
    engine.set_metronome(True, first_beat=0.015)
    out = stream.render(frames=960, dac=100.1)  # beat at sample 720 within this block
    assert np.all(out[:720] == 0)
    assert np.max(np.abs(out[721:])) > 0.1
    engine.set_metronome(False)
    # Tail of the click crosses buffer boundaries without decoding/allocating audio.
    assert np.max(np.abs(stream.render(frames=960, dac=100.12))) > 0
    engine.trigger_hit()
    assert np.max(np.abs(stream.render(frames=960, dac=100.14))) > 0.1
    assert np.all(engine.track.pcm == 0)  # the immutable source PCM wasn't modified
    engine.close()


def test_mixer_clips_and_queues_are_bounded():
    engine, stream = start_engine(volume=1)
    stream.render(frames=960, dac=100.1)
    for _ in range(1000):
        engine.trigger_hit()
    assert len(engine._hits) == 64
    out = stream.render(frames=960, dac=100.12)
    assert np.max(np.abs(out)) <= 1
    assert engine.voice_drops > 0
    engine.close()


def test_pure_clock_chooses_current_epoch_and_clamps_end():
    anchors = (ClockAnchor(1, 0, "playing", 1), ClockAnchor(2, 1, "paused", 2),
               ClockAnchor(3, 5, "playing", 3))
    assert position_at(anchors, 1.5, 10) == (0.5, "playing")
    assert position_at(anchors, 2.5, 10) == (1, "paused")
    assert position_at(anchors, 100, 10) == (10, "playing")


def test_calibration_two_offsets_and_outlier_rejection(tmp_path):
    audio = [60 + i % 3 for i in range(16)] + [-180, 190]
    visual = [85 + i % 3 for i in range(16)] + [-180, 190]
    result = estimate_offsets(audio, visual)
    assert result["audio_offset_ms"] == pytest.approx(61)
    assert result["visual_offset_ms"] == pytest.approx(25)
    assert result["audio"]["inliers"] == 16
    config = Config(audio_offset_ms=result["audio_offset_ms"], visual_offset_ms=result["visual_offset_ms"], calibrated=True)
    path = config.save(tmp_path / "config.json")
    assert Config.load(path).audio_offset_ms == 61
    # Signs: a visual tap after V=86 ms must judge exactly on the note after correction.
    raw_tap = 2.0 - config.visual_offset_ms / 1000 + 0.086
    assert config.input_time(raw_tap) == pytest.approx(2)
    assert config.display_time(2) == pytest.approx(2.025)
    with pytest.raises(ValueError):
        robust_summary([1] * 15)
    with pytest.raises(ValueError):
        robust_summary([math.nan] * 16)


def test_calibration_requires_distinct_beats_and_rejects_noise():
    collector = TapCollector()
    assert not collector.add(0.1)
    assert not collector.add(math.nan)
    for i in range(16):
        assert collector.add(2 + i * 0.5 + 0.04)
        assert not collector.add(2 + i * 0.5 + 0.06)
    assert collector.complete and len(collector.errors_ms) == 16
    assert collector.errors_ms == pytest.approx([40] * 16)


@pytest.mark.parametrize("keys", [["d", "d"], [], ["p"], ["d"] * 9])
def test_invalid_key_layout(keys):
    with pytest.raises(ValueError):
        validate_keys(keys)


def test_live_recording_hold_threshold_chords_and_round_trip(tmp_path):
    track = generate_test_track(duration=5, sample_rate=8000)
    recorder = LiveRecorder(["d", "f", "j", "k"], track.duration)
    assert recorder.key_down(0, 2)
    assert recorder.key_down(1, 2)
    assert not recorder.key_down(0, 2.1)  # repeated KEYDOWN
    assert recorder.key_up(0, 2.15).type == "Tap"
    hold = recorder.key_up(1, 2.151)
    assert hold.type == "Hold" and hold.duration == pytest.approx(0.151)
    assert recorder.key_up(1, 2.2) is None
    chart = recorder.chart(track, {"timestamps_are_corrected": True})
    path = chart.save(tmp_path / "chart.json")
    restored = Chart.load(path)
    assert restored.to_dict() == chart.to_dict()
    assert restored.lanes == 4
    assert path.read_text().endswith("\n")
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text())["time_unit"] == "seconds"


def test_recording_applies_audio_offset_once_and_bounds_held_tail():
    config = Config(audio_offset_ms=80)
    track = generate_test_track(duration=4, sample_rate=8000)
    recorder = LiveRecorder(["space"], track.duration)
    assert not recorder.key_down(0, config.input_time(0.01))
    recorder.key_down(0, config.input_time(2.08))
    recorder.key_up(0, config.input_time(2.12))
    chart = recorder.chart(track)
    assert chart.notes[0].time == pytest.approx(2)
    recorder.key_down(0, 3.5)
    recorder.release_all(100)
    assert recorder.notes[-1].end == 4
    assert not recorder.pressed


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(schema_version=99),
    lambda data: data.update(lanes=2),
    lambda data: data["notes"][0].update(lane=5),
    lambda data: data["notes"][0].update(time=-1),
    lambda data: data["notes"][0].update(time=math.nan),
    lambda data: data["notes"][0].update(type="Unknown"),
    lambda data: data["notes"][0].update(type="Hold", duration=0),
    lambda data: data["notes"][0].update(time=100),
])
def test_chart_validation_rejects_invalid_data(mutation):
    chart = Chart(silence().reference(), ["space"], [Note(1, 0)])
    data = chart.to_dict()
    mutation(data)
    with pytest.raises(ValueError):
        Chart.from_dict(data)


def test_chart_rejects_overlapping_same_lane_but_allows_chords():
    with pytest.raises(ValueError, match="Перекрывающиеся"):
        Chart(silence().reference(), ["f", "j"], [Note(0.5, 0, 0.5, "Hold"), Note(0.6, 0)]).validate()
    Chart(silence().reference(), ["f", "j"], [Note(0.5, 0), Note(0.5, 1)]).validate()


@pytest.mark.parametrize("offset,label", [(0, "Perfect"), (25, "Perfect"), (-25, "Perfect"),
                                         (25.1, "Great"), (50, "Great"), (-50, "Great"),
                                         (50.1, "Good"), (90, "Good"), (-90, "Good"), (90.1, "Miss")])
def test_judgment_windows(offset, label):
    assert grade(offset) == label
    judge = Judge(Chart(silence().reference(), ["space"], [Note(1, 0)]))
    result = judge.key_down(0, 1 + offset / 1000)
    if label == "Miss":
        assert result.note_index is None
        judge.update(1.1)
        assert judge.counts["Miss"] == 1
    else:
        assert result.label == label
        assert result.delta_ms == pytest.approx(offset)
        assert judge.combo == 1
        assert ("early" in result.timing_text) == (offset < 0)


def test_hold_uses_worst_start_release_and_misses_early_or_stuck_holds():
    chart = Chart(silence(duration=4).reference(), ["f", "j"],
                  [Note(0.5, 0, 0.5, "Hold"), Note(2, 1, 0.5, "Hold")])
    judge = Judge(chart)
    assert judge.key_down(0, 0.51).phase == "hold start"
    assert sum(judge.counts.values()) == 0
    assert judge.key_up(0, 1.04).label == "Great"
    judge.key_down(1, 2)
    assert judge.key_up(1, 2.1).label == "Miss"
    judge = Judge(chart)
    judge.key_down(0, 0.5)
    judge.update(1.1)
    assert judge.counts["Miss"] == 1 and not judge.holding
    judge.update(3)
    judge.update(3.5)
    assert judge.counts["Miss"] == 2  # not counted again


def test_judgment_is_lane_aware_and_chords_work():
    judge = Judge(Chart(silence().reference(), ["f", "j"], [Note(1, 0), Note(1, 1)]))
    assert judge.key_down(0, 1).label == "Perfect"
    assert judge.key_down(0, 1).note_index is None
    assert judge.key_down(1, 1).label == "Perfect"
    assert judge.combo == 2 and judge.score == 2000 and judge.accuracy == 100
    judge.finish()
    assert judge.counts["Perfect"] == 2


def test_invalid_config_does_not_silently_reset(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"audio_offset_ms": "bad"}')
    with pytest.raises(ValueError):
        Config.load(path)
    with pytest.raises(ValueError):
        Config(fps=60).save(path)
    with pytest.raises(ValueError):
        Config(audio_offset_ms=math.inf).validate()


def test_diagnostics_and_cli_do_not_need_hardware(tmp_path):
    from src.diagnostics import environment_report
    report = environment_report()
    assert "gpu" in report and "output_devices" in report["audio"]
    result = subprocess.run([sys.executable, "main.py", "--diagnose", "--report", str(tmp_path / "env.json")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["python"]
    assert (tmp_path / "env.json").is_file()


def test_all_screens_render_headlessly_without_audio(tmp_path):
    import pygame
    from main import headless_smoke
    result = headless_smoke(Config(), str(tmp_path / "menu.png"))
    assert set(result["rendered_screens"]) == {"menu", "audio_calibration", "visual_calibration", "live_mapper", "gameplay", "ai_studio", "beat_snap"}
    assert not result["audio_stream_opened"]
    assert (tmp_path / "menu.png").stat().st_size > 1000
    assert not pygame.display.get_init()


@pytest.fixture
def scripted_app(monkeypatch):
    """Real drawing/event handlers, but scripted input and a simulated DAC clock."""
    import pygame
    from src.ui import App

    engines = []
    notices = []

    def factory(track, **kwargs):
        engine = AudioEngine(track, stream_factory=FakeStream, **kwargs)
        engines.append(engine)
        return engine

    app = App(Config(sample_rate=24000), engine_factory=factory)
    monkeypatch.setattr(app, "frame", lambda **_: None)
    monkeypatch.setattr(app, "confirm", lambda *_, **__: True)
    monkeypatch.setattr(app, "notice", lambda title, lines: notices.append((title, lines)))

    def advance(song_time):
        engine = engines[-1]
        stream = engine._stream
        now = 100.1 + song_time
        next_dac = getattr(stream, "next_dac", 100.1)
        frames = round(engine.track.sample_rate * 0.05)
        while next_dac <= now + 0.025:
            stream.render(frames=frames, dac=next_dac)
            next_dac += frames / engine.track.sample_rate
        stream.next_dac = next_dac
        stream.time = now

    try:
        yield app, engines, notices, advance
    finally:
        for engine in engines:
            engine.close()
        pygame.quit()


def test_calibration_loop_runs_both_passes_and_saves_only_corrected_offsets(scripted_app, monkeypatch, tmp_path):
    import pygame
    from src.calibration import run_calibration

    app, engines, notices, advance = scripted_app
    taps = [0, 0]

    def events():
        stage = len(engines) - 1
        assert taps[stage] < 16, "Calibration did not finish its pass"
        delay = 0.060 if stage == 0 else 0.085
        advance(2 + taps[stage] * 0.5 + delay)
        taps[stage] += 1
        return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)]

    monkeypatch.setattr(app, "events", events)
    path = tmp_path / "calibrated.json"
    run_calibration(app, path)
    config = Config.load(path)
    assert taps == [16, 16] and len(engines) == 2
    assert config.calibrated
    assert config.audio_offset_ms == pytest.approx(60)
    assert config.visual_offset_ms == pytest.approx(25)
    assert config.calibration["audio"]["accepted_taps"] == 16
    assert config.calibration["visual"]["accepted_taps"] == 16
    assert not engines[1]._metronome[0]  # visual-only pass really disabled clicks
    assert all(engine._stream is None for engine in engines)
    assert notices[0][0] == "Результат калибровки"


def test_live_mapper_loop_records_corrected_input_and_saves(scripted_app, monkeypatch, tmp_path):
    import pygame
    import src.chart as chart_module
    from src.live_mapper import run_live_mapper

    app, engines, _, advance = scripted_app
    app.config.audio_offset_ms = 80
    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)
    actions = iter([(0.58, pygame.KEYDOWN, pygame.K_f), (0.63, pygame.KEYUP, pygame.K_f),
                    (1.08, pygame.KEYDOWN, pygame.K_j), (1.28, pygame.KEYUP, pygame.K_j),
                    (1.5, pygame.KEYDOWN, pygame.K_s)])

    def events():
        when, kind, key = next(actions)
        advance(when)
        return [pygame.event.Event(kind, key=key)]

    monkeypatch.setattr(app, "events", events)
    track = generate_test_track(duration=2, sample_rate=24000)
    recorded, saved_path = run_live_mapper(app, track, ["f", "j"])
    chart = Chart.load(tmp_path / "test_track_2K.json")
    assert saved_path == tmp_path / "test_track_2K.json" and recorded.to_dict() == chart.to_dict()
    assert [(n.lane, n.type) for n in chart.notes] == [(0, "Tap"), (1, "Hold")]
    assert [n.time for n in chart.notes] == pytest.approx([0.5, 1.0])
    assert chart.notes[1].duration == pytest.approx(0.2)
    assert chart.metadata["audio_offset_ms_used"] == 80
    assert engines[0]._stream is None


def test_mapper_processes_queued_keyup_before_automatic_eof_save(scripted_app, monkeypatch, tmp_path):
    import pygame
    import src.chart as chart_module
    from src.live_mapper import run_live_mapper

    app, _, _, advance = scripted_app
    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 10000)
    polls = 0

    def events():
        nonlocal polls
        polls += 1
        if polls == 1:
            advance(0.5)
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)]
        assert polls == 2
        # Released at t=0.8, but a delayed frame retrieves it at t=1.0 (EOF).
        return [pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE, timestamp=9800)]

    monkeypatch.setattr(app, "events", events)
    monkeypatch.setattr(app, "frame", lambda **_: advance(1.0))
    recorded, saved_path = run_live_mapper(app, generate_test_track(duration=1, sample_rate=24000), ["space"])
    chart = Chart.load(tmp_path / "test_track_1K.json")
    assert saved_path == tmp_path / "test_track_1K.json" and recorded.to_dict() == chart.to_dict()
    assert len(chart.notes) == 1
    assert chart.notes[0].end == pytest.approx(0.8)
    assert polls == 2


def test_gameplay_loop_accepts_corrected_tap_and_hold_then_shows_results(scripted_app, monkeypatch):
    import pygame
    from src.gameplay import run_gameplay

    app, engines, notices, advance = scripted_app
    app.config.audio_offset_ms = 40
    track = generate_test_track(duration=2, sample_rate=24000)
    chart = Chart(track.reference(), ["f", "j"], [Note(0.5, 0), Note(1, 1, 0.3, "Hold")])
    actions = iter([(0.54, pygame.KEYDOWN, pygame.K_f), (0.60, pygame.KEYUP, pygame.K_f),
                    (1.04, pygame.KEYDOWN, pygame.K_j), (1.34, pygame.KEYUP, pygame.K_j)])

    finished = False
    def events():
        nonlocal finished
        action = next(actions, None)
        if action is None:
            if not finished:
                finished = True
                advance(2.1)
                return []
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)]
        when, kind, key = action
        advance(when)
        return [pygame.event.Event(kind, key=key)]

    monkeypatch.setattr(app, "events", events)
    run_gameplay(app, track, chart)
    assert engines[0]._stream is None
    assert not notices  # detailed histogram/result screen replaces the old text-only notice


# Rewind regression tests use the same real callback and simulated DAC clock.

def positioned_engine(position=20.0, duration=30.0):
    engine = AudioEngine(silence(duration), stream_factory=FakeStream)
    engine.seek(position)
    engine.play()
    stream = engine._stream
    stream.render(dac=100.1)
    stream.time = 100.1
    return engine, stream


def test_rewind_before_open_is_sample_aligned_and_clamps_to_start():
    engine = AudioEngine(silence(30), stream_factory=FakeStream)
    engine.seek(13.123456)
    assert engine.rewind() == round(8.123456 * 48000) / 48000
    assert engine.rewind() == round(3.123456 * 48000) / 48000
    assert engine.rewind() == 0
    assert engine.rewind() == 0 and engine.song_time == 0
    assert engine.requested_state == "stopped" and engine._stream is None


@pytest.mark.parametrize("seconds", [0, -5, math.nan, math.inf])
def test_rewind_rejects_bad_interval_without_changing_transport(seconds):
    engine = AudioEngine(silence(30), stream_factory=FakeStream)
    engine.seek(12)
    with pytest.raises(ValueError):
        engine.rewind(seconds)
    assert engine.song_time == 12 and engine._stream is None


def test_rapid_rewinds_stack_before_callback_and_before_dac_presentation():
    engine, stream = positioned_engine(13)
    assert engine.rewind() == 8
    assert engine.rewind() == 3
    stream.render(dac=100.2)
    stream.time = 100.15
    assert engine.song_time == pytest.approx(13.05)  # old buffer is still heard
    assert engine.transport_pending
    assert engine.rewind() == 0  # use pending 3, not audible 13.05
    stream.render(dac=100.3)
    stream.time = 100.25
    assert engine.song_time == pytest.approx(3.05)
    stream.time = 100.32
    assert engine.song_time == pytest.approx(0.02) and engine.state == "playing"
    assert not engine.transport_pending
    assert engine.snapshot(seconds_ago=0.15)[1] == "transition"
    engine.close()


def test_rewind_keeps_pending_target_when_pause_is_interleaved():
    engine, stream = positioned_engine(20)
    assert engine.rewind() == 15
    engine.pause()
    assert engine.rewind() == 10
    assert engine.requested_state == "paused"
    stream.render(dac=100.2)
    stream.time = 100.25
    assert engine.state == "paused" and engine.song_time == 10
    engine.play()
    stream.render(dac=100.3)
    stream.time = 100.32
    assert engine.song_time == pytest.approx(10.02)
    engine.close()


def test_later_pause_does_not_resurrect_an_old_rewind_target():
    engine, stream = positioned_engine(20)
    assert engine.rewind() == 15
    stream.render(dac=100.2)
    stream.render(dac=100.3)
    stream.time = 100.35
    assert engine.song_time == pytest.approx(15.15)
    engine.pause()  # pending state command, but the previous seek is already heard
    assert engine.transport_pending
    assert engine.rewind() == pytest.approx(10.15)
    stream.render(dac=100.4)
    stream.time = 100.45
    assert engine.state == "paused" and engine.song_time == pytest.approx(10.15)
    engine.close()


def test_rewind_while_paused_preserves_frozen_clock_until_resume():
    engine, stream = positioned_engine(17)
    engine.pause()
    stream.render(dac=100.2)
    stream.time = 100.25
    assert engine.song_time == pytest.approx(17.1)
    assert engine.rewind() == pytest.approx(12.1)
    stream.render(dac=100.3)
    stream.time = 100.28
    assert engine.song_time == pytest.approx(17.1)
    stream.time = 101.5
    assert engine.state == "paused" and engine.song_time == pytest.approx(12.1)
    assert engine.rewind() == pytest.approx(7.1)
    engine.close()


def test_rewind_from_eof_restarts_at_the_requested_position():
    engine, stream = start_engine(silence(0.1))
    stream.render(dac=100.1)
    stream.time = 100.21
    assert engine.state == "finished"
    assert engine.rewind() == 0
    stream.render(dac=100.3)
    stream.time = 100.32
    assert engine.state == "playing" and engine.song_time == pytest.approx(0.02)
    engine.close()


def test_absolute_seek_and_stop_supersede_pending_rewind_targets():
    engine, stream = positioned_engine(20)
    assert engine.rewind() == 15
    engine.seek(26)
    assert engine.rewind() == 21
    engine.stop()
    assert engine.rewind() == 0 and engine.requested_state == "stopped"
    stream.render(dac=100.2)
    stream.time = 100.25
    assert engine.song_time == 0 and engine.state == "stopped"
    engine.close()
    assert engine._position_request is None and engine._applied_position is None


def test_rewind_target_survives_a_burst_beyond_the_bounded_command_queue():
    engine, stream = positioned_engine(20)
    assert engine.rewind() == 15
    for _ in range(100):
        engine.play()
    assert len(engine._commands) == 64
    stream.render(dac=100.2)
    stream.time = 100.21
    assert engine.song_time == pytest.approx(15.01)
    stream.render(dac=100.3)
    stream.time = 100.31
    assert engine.song_time == pytest.approx(15.11)  # not reset again next callback
    engine.close()


def test_recorder_rewind_removes_future_onsets_and_trims_crossing_hold():
    recorder = LiveRecorder(["f", "j"], 10, notes=[Note(1, 0), Note(2, 1, 2, "Hold"),
                                                  Note(3, 0), Note(3.5, 0), Note(4, 0)])
    assert recorder.rewind_to(3) == 3
    assert recorder.notes == [Note(1, 0), Note(2, 1, 1, "Hold")]
    assert recorder.rewind_count == 1
    recorder.chart(generate_test_track(duration=10, sample_rate=8000)).validate()


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("prefix,kind", [(0.149, "Tap"), (0.150, "Tap"), (0.151, "Hold")])
def test_rewind_hold_prefix_uses_strict_threshold_for_finished_and_active_notes(active, prefix, kind):
    recorder = LiveRecorder(["space"], 5)
    if active:
        recorder.key_down(0, 1)
    else:
        recorder.notes.append(Note(1, 0, 1, "Hold"))
    recorder.rewind_to(1 + prefix)
    assert recorder.notes[0].type == kind
    assert recorder.notes[0].duration == pytest.approx(prefix if kind == "Hold" else 0)
    assert not recorder.pressed
    assert recorder.key_up(0, 3) is None


def test_rewind_cuts_active_holds_and_requires_a_fresh_keydown():
    recorder = LiveRecorder(["d", "f", "j"], 10, notes=[Note(0.2, 0)])
    for lane, start in enumerate((1, 2, 3)):
        recorder.key_down(lane, start)
    recorder.rewind_to(2)
    assert recorder.notes == [Note(0.2, 0), Note(1, 0, 1, "Hold")]
    assert not recorder.pressed
    assert recorder.key_up(0, 4) is None
    assert recorder.key_down(0, 2)
    assert recorder.key_up(0, 2.2).type == "Hold"
    recorder.chart(generate_test_track(duration=10, sample_rate=8000)).validate()


def test_rewind_near_start_erases_all_and_respects_custom_hold_threshold():
    recorder = LiveRecorder(["space"], 5, hold_threshold=0.25, notes=[Note(1, 0, 1, "Hold")])
    recorder.rewind_to(1.2)
    assert recorder.notes == [Note(1, 0)]
    recorder.key_down(0, 2)
    recorder.rewind_to(-0.08)  # raw seek 0 minus a positive 80 ms calibration
    assert not recorder.notes and not recorder.pressed and recorder.rewind_count == 2


@pytest.mark.parametrize("cutoff", [math.nan, math.inf])
def test_invalid_rewind_boundary_does_not_destroy_recording(cutoff):
    recorder = LiveRecorder(["f", "j"], 5, notes=[Note(1, 0)])
    recorder.key_down(1, 2)
    with pytest.raises(ValueError):
        recorder.rewind_to(cutoff)
    assert recorder.notes == [Note(1, 0)] and recorder.pressed == {1: 2} and recorder.rewind_count == 0


def test_rewind_retains_completion_order_for_backspace_undo():
    recorder = LiveRecorder(["f", "j"], 5, notes=[Note(2, 1), Note(1, 0, 2, "Hold")])
    recorder.rewind_to(2.5)
    assert recorder.notes.pop() == Note(1, 0, 1.5, "Hold")
    assert recorder.notes == [Note(2, 1)]


def advance_device(engine, elapsed):
    """Advance monotonically in DEVICE time, even when song time seeks back."""
    stream = engine._stream
    now = 100.1 + elapsed
    next_dac = getattr(stream, "next_dac", 100.1)
    frames = round(engine.track.sample_rate * 0.05)
    while next_dac <= now + 0.025:
        stream.render(frames=frames, dac=next_dac)
        next_dac += frames / engine.track.sample_rate
    stream.next_dac = next_dac
    stream.time = now


@pytest.mark.parametrize("shortcut", ["left", "f2"])
def test_live_rewind_corrects_boundary_and_rejects_pending_and_stale_input(scripted_app, monkeypatch, tmp_path, shortcut):
    import pygame
    import src.chart as chart_module
    from src.live_mapper import run_live_mapper
    app, engines, _, _ = scripted_app
    app.config.audio_offset_ms = 80
    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 10000)
    actions = iter([(0.58, pygame.KEYDOWN, pygame.K_f, None), (0.63, pygame.KEYUP, pygame.K_f, None),
                    (6.08, pygame.KEYDOWN, pygame.K_f, None), (7.03, pygame.KEYDOWN, pygame.K_j, None),
                    (7.33, pygame.KEYUP, pygame.K_j, None), (7.58, pygame.KEYUP, pygame.K_f, None),
                    (12, pygame.KEYDOWN, pygame.key.key_code(shortcut), None),
                    (12.01, pygame.KEYDOWN, pygame.K_f, None),  # seek not yet heard: ignored
                    (12.06, pygame.KEYDOWN, pygame.K_f, None),
                    (12.07, pygame.KEYUP, pygame.K_f, 9900),  # pre-seek epoch: ignored
                    (12.26, pygame.KEYUP, pygame.K_f, None), (12.3, pygame.KEYDOWN, pygame.K_s, None)])
    def events():
        elapsed, kind, key, timestamp = next(actions)
        advance_device(engines[-1], elapsed)
        fields = {"key": key}
        if timestamp is not None:
            fields["timestamp"] = timestamp
        return [pygame.event.Event(kind, **fields)]
    monkeypatch.setattr(app, "events", events)
    run_live_mapper(app, generate_test_track(duration=20, sample_rate=24000), ["f", "j"])
    chart = Chart.load(tmp_path / "test_track_2K.json")
    assert [n.time for n in chart.notes] == pytest.approx([0.5, 6.0, 6.93])
    assert [n.duration for n in chart.notes] == pytest.approx([0, 0.92, 0.20])
    assert all(n.lane == 0 for n in chart.notes)  # old J note at 6.95 was erased
    assert chart.metadata["rewind_count"] == 1 and chart.metadata["audio_offset_ms_used"] == 80


def test_live_repeated_rewind_preserves_pause_then_resumes_with_negative_offset(scripted_app, monkeypatch, tmp_path):
    import pygame
    import src.chart as chart_module
    from src.live_mapper import run_live_mapper
    app, engines, _, _ = scripted_app
    app.config.audio_offset_ms = -40
    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)
    steps = iter([(6.0, [pygame.K_f], pygame.KEYDOWN), (6.3, [pygame.K_f], pygame.KEYUP),
                  (12, [pygame.K_p, pygame.K_LEFT], pygame.KEYDOWN), (12.06, [pygame.K_F2], pygame.KEYDOWN),
                  (12.08, [pygame.K_f], pygame.KEYDOWN), (12.16, [pygame.K_p], pygame.KEYDOWN),
                  (12.21, [pygame.K_f], pygame.KEYDOWN), (12.41, [pygame.K_f], pygame.KEYUP),
                  (12.5, [pygame.K_s], pygame.KEYDOWN)])
    def events():
        elapsed, keys, kind = next(steps)
        advance_device(engines[-1], elapsed)
        if elapsed == 12.16:
            assert engines[-1].state == "paused" and engines[-1].song_time == pytest.approx(2)
        return [pygame.event.Event(kind, key=key) for key in keys]
    monkeypatch.setattr(app, "events", events)
    run_live_mapper(app, generate_test_track(duration=20, sample_rate=24000), ["f"])
    chart = Chart.load(tmp_path / "test_track_1K.json")
    assert len(chart.notes) == 1
    assert chart.notes[0].time == pytest.approx(2.05) and chart.notes[0].duration == pytest.approx(0.2)
    assert chart.metadata["rewind_count"] == 2


def test_custom_left_lane_still_records_while_f2_rewinds(scripted_app, monkeypatch, tmp_path):
    import pygame
    import src.chart as chart_module
    from src.live_mapper import run_live_mapper
    app, engines, _, _ = scripted_app
    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)
    steps = iter([(0.5, pygame.KEYDOWN, pygame.K_LEFT), (0.7, pygame.KEYUP, pygame.K_LEFT),
                  (0.8, pygame.KEYDOWN, pygame.K_F2), (0.86, pygame.KEYDOWN, pygame.K_LEFT),
                  (0.91, pygame.KEYUP, pygame.K_LEFT), (1.0, pygame.KEYDOWN, pygame.K_s)])
    def events():
        elapsed, kind, key = next(steps)
        advance_device(engines[-1], elapsed)
        return [pygame.event.Event(kind, key=key)]
    monkeypatch.setattr(app, "events", events)
    run_live_mapper(app, generate_test_track(duration=2, sample_rate=24000), ["left"])
    chart = Chart.load(tmp_path / "test_track_1K.json")
    assert chart.keys == ["left"] and len(chart.notes) == 1
    assert chart.notes[0].type == "Tap" and chart.notes[0].time == pytest.approx(0.01)
    assert chart.metadata["rewind_count"] == 1


def test_backspace_remains_single_note_undo_not_rewind(scripted_app, monkeypatch, tmp_path):
    import pygame
    import src.chart as chart_module
    from src.live_mapper import run_live_mapper
    app, engines, _, _ = scripted_app
    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)
    steps = iter([(0.2, pygame.KEYDOWN, pygame.K_f), (0.25, pygame.KEYUP, pygame.K_f),
                  (0.5, pygame.KEYDOWN, pygame.K_f), (0.55, pygame.KEYUP, pygame.K_f),
                  (0.6, pygame.KEYDOWN, pygame.K_BACKSPACE), (0.7, pygame.KEYDOWN, pygame.K_s)])
    def events():
        elapsed, kind, key = next(steps)
        advance_device(engines[-1], elapsed)
        return [pygame.event.Event(kind, key=key)]
    monkeypatch.setattr(app, "events", events)
    run_live_mapper(app, generate_test_track(duration=2, sample_rate=24000), ["f"])
    chart = Chart.load(tmp_path / "test_track_1K.json")
    assert len(chart.notes) == 1 and chart.notes[0].time == pytest.approx(0.2)
    assert chart.metadata["rewind_count"] == 0
