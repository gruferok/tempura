"""Offline onset snapping, immutable previews, safe persistence and native UI.

Transient fixtures test digital timing, NOT musical quality or physical latency.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from src.audio_engine import AudioTrack, synth_click
from src.beat_snap import SnapGrid, analyze_beats, save_snap_copy, snap_chart
from src.chart import Chart, Note
from src.ml import Cancelled
from src.storage import Config


def make_chart(notes=None, *, duration=4, keys=None, metadata=None):
    track = AudioTrack(np.zeros((round(duration * 8000), 2), np.float32), 8000)
    return track, Chart(track.reference(), keys or ["f", "j"], notes or [], metadata or {})


def grid_for(chart, targets):
    return SnapGrid(chart.track["id"], chart.track["duration"], tuple(targets))


def click_track(rate=24000, *, antiphase=False):
    attacks = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    pcm = np.zeros((rate * 4, 2), np.float32)
    click = synth_click(rate)
    for attack in attacks:
        start = round(attack * rate)
        pcm[start:start + len(click)] += click
    if antiphase:
        pcm[:, 1] *= -1
    return AudioTrack(pcm, rate), attacks


@pytest.mark.parametrize("rate,antiphase", [(8000, False), (24000, False), (48000, True)])
def test_librosa_attacks_refined_near_real_clicks_and_jitter_reduced(rate, antiphase):
    track, attacks = click_track(rate, antiphase=antiphase)
    grid = analyze_beats(track)
    assert len(grid.onsets) == len(attacks)
    assert np.max(np.abs(np.array(grid.onsets) - attacks)) < 0.005
    assert grid.tempo_bpm == pytest.approx(120, abs=3)
    assert set(grid.beats) <= set(grid.onsets)  # no invented silent beat targets
    jitter = (0.018, -0.021, 0.023, -0.019, 0.017, -0.020)
    chart = Chart(track.reference(), ["space"], [Note(t + j, 0) for t, j in zip(attacks, jitter)])
    result = snap_chart(chart, grid)
    assert len(result.changes) == len(attacks)
    assert max(abs(n.time - t) for n, t in zip(result.chart.notes, attacks)) < 0.005
    assert all(abs(c.delta_ms) <= 25 for c in result.changes)


@pytest.mark.parametrize("duration", [0.000125, 0.1, 4.0])
def test_silence_has_no_synthetic_grid_or_changes(duration):
    track, chart = make_chart([Note(0, 0)], duration=duration)
    result = snap_chart(chart, analyze_beats(track))
    assert result.grid.onsets == result.grid.beats == () and result.grid.tempo_bpm is None
    assert not result.changes and result.chart.notes == chart.notes


def test_analysis_cancellation_checked_before_import_or_work():
    cancel = threading.Event()
    cancel.set()
    track, _ = make_chart()
    with pytest.raises(Cancelled):
        analyze_beats(track, cancel=cancel)


def test_analysis_checks_cancel_between_librosa_stages():
    cancel = threading.Event()
    track, _ = click_track()
    def progress(event):
        if event["phase"] == "onsets":
            cancel.set()
    with pytest.raises(Cancelled):
        analyze_beats(track, progress=progress, cancel=cancel)


@pytest.mark.parametrize("window,count", [(15, 1), (20, 2), (25, 3)])
def test_only_nearby_onsets_move_within_selected_tolerance(window, count):
    _, chart = make_chart([Note(0.510, 0), Note(1.018, 1), Note(1.523, 0), Note(2.030, 1)])
    result = snap_chart(chart, grid_for(chart, [0.5, 1, 1.5, 2]), max_shift_ms=window)
    assert len(result.changes) == count
    assert result.chart.notes[-1].time == 2.030
    assert result.largest_shift_ms <= window + 1e-8
    assert [n.lane for n in result.chart.notes] == [n.lane for n in chart.notes]


@pytest.mark.parametrize("offset", [-120, 0, 80])
def test_saved_corrected_notes_never_have_offsets_applied_again(offset):
    metadata = {"audio_offset_ms_used": offset, "visual_offset_ms": 49, "timestamps_are_corrected": True}
    _, chart = make_chart([Note(1.020, 1)], metadata=metadata)
    result = snap_chart(chart, grid_for(chart, [1]))
    assert result.chart.notes[0].time == 1
    assert result.chart.metadata["audio_offset_ms_used"] == offset
    assert result.chart.metadata["visual_offset_ms"] == 49
    assert result.chart.metadata["timestamps_are_corrected"] is True


def test_exact_window_boundary_and_earlier_tie_are_deterministic():
    _, chart = make_chart([Note(1.025, 0), Note(2, 1)])
    result = snap_chart(chart, grid_for(chart, [1, 1.975, 2.025]))
    assert result.chart.notes[0].time == 1
    assert result.chart.notes[1].time == 1.975
    assert len(result.changes) == 2


def test_holds_translate_without_changing_release_duration_type_or_keys():
    _, chart = make_chart([Note(1.018, 1, 0.651, "Hold"), Note(2.98, 0)], keys=["left", "right"])
    result = snap_chart(chart, grid_for(chart, [1, 3]))
    assert result.chart.keys == chart.keys
    assert result.chart.notes == [Note(1, 1, 0.651, "Hold"), Note(3, 0)]
    assert result.chart.notes[0].end == pytest.approx(chart.notes[0].end - 0.018)


def test_hold_and_its_chord_do_not_shift_past_eof():
    _, chart = make_chart([Note(3.78, 0, 0.22, "Hold"), Note(3.78, 1)])
    result = snap_chart(chart, grid_for(chart, [3.8]))
    assert result.chart.notes == chart.notes
    assert not result.changes and result.blocked_notes == 2


def test_same_lane_taps_never_merge_and_conflicting_exact_chord_reverts_together():
    _, chart = make_chart([Note(0.99, 0), Note(1.01, 0), Note(1.01, 1), Note(2.02, 1)])
    result = snap_chart(chart, grid_for(chart, [1, 2]))
    assert result.chart.notes[:3] == chart.notes[:3]
    assert result.chart.notes[-1] == Note(2, 1)
    assert len(result.chart.notes) == len(chart.notes) and result.blocked_notes == 3
    result.chart.validate()


def test_neighbour_conflicts_rechecked_when_a_hold_shift_is_reverted():
    # Second hold moves forward, which would overlap the unmoved tap at 1.21;
    # reverting it then conflicts with the hold just before it. Both revert.
    _, chart = make_chart([Note(0.48, 0, 0.5, "Hold"), Note(0.98, 0, 0.22, "Hold"), Note(1.21, 0)])
    result = snap_chart(chart, grid_for(chart, [0.5, 1.0]))
    assert result.chart.notes == chart.notes and not result.changes
    assert result.blocked_notes == 2


def test_safe_chords_and_unsorted_completion_order_are_preserved():
    _, chart = make_chart([Note(2.02, 1), Note(1.02, 1), Note(1.02, 0, 0.5, "Hold")])
    result = snap_chart(chart, grid_for(chart, [1, 2]))
    assert result.chart.notes == [Note(2, 1), Note(1, 1), Note(1, 0, 0.5, "Hold")]
    assert [c.index for c in result.changes] == [0, 1, 2]


def test_preview_does_not_mutate_original_or_share_mutable_metadata():
    _, chart = make_chart([Note(1.02, 0)], metadata={"custom": {"nested": [1]}, "human_reviewed": True})
    before = deepcopy(chart.to_dict())
    result = snap_chart(chart, grid_for(chart, [1]))
    assert chart.to_dict() == before
    result.chart.metadata["custom"]["nested"].append(2)
    result.chart.keys.append("k")
    result.chart.track["name"] = "changed"
    assert chart.to_dict() == before
    assert result.chart.metadata["auto_quantized"] is True and result.chart.metadata["human_reviewed"] is False


@pytest.mark.parametrize("window", [0, -15, 51, math.nan, math.inf])
def test_invalid_window_rejected_without_mutating_notes(window):
    _, chart = make_chart([Note(1.02, 0)])
    with pytest.raises(ValueError, match="Окно"):
        snap_chart(chart, grid_for(chart, [1]), max_shift_ms=window)
    assert chart.notes == [Note(1.02, 0)]


@pytest.mark.parametrize("targets", [(-0.01,), (4,), (math.nan,), (math.inf,)])
def test_grid_rejects_invalid_target_timestamps(targets):
    with pytest.raises(ValueError, match="targets"):
        SnapGrid("track", 4, targets)


@pytest.mark.parametrize("identity,duration", [("wrong", 4), ("same", 4.01)])
def test_mismatched_audio_rejected(identity, duration):
    _, chart = make_chart([Note(1.02, 0)])
    grid = SnapGrid(chart.track["id"] if identity == "same" else identity, duration, (1,))
    with pytest.raises(ValueError, match="не совпадает"):
        snap_chart(chart, grid)


@pytest.mark.parametrize("hold", [False, True])
def test_same_audio_with_sample_rounded_duration_uses_original_chart_bounds(hold):
    notes = [Note(1.02, 0), Note(3.98, 1, 0.02 if hold else 0, "Hold" if hold else "Tap")]
    _, chart = make_chart(notes)
    grid = SnapGrid(chart.track["id"], 4.00002, (1, 4.00001 if not hold else 3.98001))
    result = snap_chart(chart, grid)
    assert result.chart.notes[0] == Note(1, 0)
    assert result.chart.notes[1] == chart.notes[1]  # no EOF drift into the longer decoded PCM
    assert result.blocked_notes == 1
    result.chart.validate()


def test_copy_saves_compatible_unique_json_and_leaves_source_and_preview_untouched(tmp_path):
    from src.gameplay import Judge
    _, chart = make_chart([Note(1.02, 0, 0.5, "Hold")])
    source = chart.save(tmp_path / "мой_трек_2K.json")
    original_bytes = source.read_bytes()
    result = snap_chart(chart, grid_for(chart, [1]))
    first = save_snap_copy(result, source)
    first_bytes = first.read_bytes()
    second = save_snap_copy(result, source)
    assert first.name == "мой_трек_2K_snap.json" and second.name == "мой_трек_2K_snap_2.json"
    assert source.read_bytes() == original_bytes and first.read_bytes() == first_bytes
    loaded = Chart.load(second)
    assert loaded.notes == result.chart.notes and len(Judge(loaded).notes) == 1
    provenance = loaded.metadata["beat_snap"]
    assert provenance["source_chart"] == source.name
    assert provenance["source_file_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert "source_chart" not in result.chart.metadata["beat_snap"]
    assert not list(tmp_path.glob("*.tmp"))


def test_noop_or_missing_original_never_writes_a_copy(tmp_path):
    _, chart = make_chart([Note(1.02, 0)])
    with pytest.raises(ValueError, match="не изменил"):
        save_snap_copy(snap_chart(chart, grid_for(chart, [])), tmp_path / "missing.json")
    with pytest.raises(ValueError, match="Сначала сохраните"):
        save_snap_copy(snap_chart(chart, grid_for(chart, [1])), tmp_path / "missing.json")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("reviewed,accepted", [(False, False), ("true", False), (True, True)])
def test_auto_quantized_copies_not_training_labels_without_literal_review(reviewed, accepted):
    from src.ml.dataset import training_exclusion
    _, chart = make_chart([Note(1, 0)], metadata={"auto_quantized": True, "human_reviewed": reviewed})
    reason = training_exclusion(chart)
    assert (reason is None) is accepted
    if reason:
        assert "Beat-Snap" in reason


@pytest.fixture
def snap_ui(monkeypatch, tmp_path):
    import pygame
    import src.snap_screen as screen
    from src.ui import App
    track, chart = make_chart([Note(1.02, 0), Note(2.018, 1)])
    path = chart.save(tmp_path / "song_2K.json")
    app = App(Config())
    monkeypatch.setattr(app, "frame", lambda **_: None)
    class ImmediateJob(screen.BackgroundJob):
        def start(self):
            self._run()
            return self
    monkeypatch.setattr(screen, "BackgroundJob", ImmediateJob)
    try:
        yield app, track, chart, path
    finally:
        pygame.quit()


def test_review_is_opt_in_and_cancel_does_not_analyze_or_save(snap_ui, monkeypatch):
    import pygame
    import src.snap_screen as screen
    app, track, chart, path = snap_ui
    monkeypatch.setattr(screen, "analyze_beats", lambda *_, **__: pytest.fail("Unrequested analysis"))
    keys = iter([pygame.K_RETURN, pygame.K_ESCAPE])
    monkeypatch.setattr(app, "events", lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(keys))])
    assert screen.run_snap_screen(app, track, chart, path) is None
    assert list(path.parent.glob("*.json")) == [path]


def test_review_analyze_window_change_playtest_and_explicit_save(snap_ui, monkeypatch):
    import pygame
    import src.snap_screen as screen
    import src.gameplay as gameplay
    app, track, chart, source = snap_ui
    calls, played = [], []
    def analyze(selected, **kwargs):
        assert selected is track
        calls.append("analyze")
        return grid_for(chart, [1, 2])
    monkeypatch.setattr(screen, "analyze_beats", analyze)
    monkeypatch.setattr(gameplay, "run_gameplay", lambda app, track, chart: played.append(chart))
    actions = iter([pygame.K_b, pygame.K_1, pygame.K_p, pygame.K_2, pygame.K_p, pygame.K_RETURN])
    monkeypatch.setattr(app, "events", lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(actions))])
    result_path = screen.run_snap_screen(app, track, chart, source)
    assert calls == ["analyze"]
    assert len(played) == 1  # 15 ms window has no changes, hence no playtest
    assert Chart.load(result_path).notes == [Note(1, 0), Note(2, 1)]
    assert Chart.load(source).notes == chart.notes
    assert "Оригинал не изменён" in app.toast


def test_discarding_finished_preview_does_not_write(snap_ui, monkeypatch):
    import pygame
    import src.snap_screen as screen
    app, track, chart, source = snap_ui
    monkeypatch.setattr(screen, "analyze_beats", lambda *_, **__: grid_for(chart, [1, 2]))
    actions = iter([pygame.K_b, pygame.K_ESCAPE])
    monkeypatch.setattr(app, "events", lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(actions))])
    assert screen.run_snap_screen(app, track, chart, source) is None
    assert list(source.parent.glob("*.json")) == [source]


def test_review_analysis_errors_do_not_destroy_original_and_can_retry(snap_ui, monkeypatch):
    import pygame
    import src.snap_screen as screen
    app, track, chart, source = snap_ui
    before = source.read_bytes()
    calls = []
    def analyze(*_, **__):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("fixture detector failed")
        return grid_for(chart, [1, 2])
    monkeypatch.setattr(screen, "analyze_beats", analyze)
    actions = iter([pygame.K_b, pygame.K_b, pygame.K_RETURN])
    monkeypatch.setattr(app, "events", lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(actions))])
    assert screen.run_snap_screen(app, track, chart, source).is_file()
    assert len(calls) == 2 and source.read_bytes() == before


def test_review_real_background_worker_renders_and_saves_only_after_acceptance(monkeypatch, tmp_path):
    import pygame
    import src.snap_screen as screen
    from src.ui import App
    track, attacks = click_track()
    chart = Chart(track.reference(), ["space"], [Note(t + 0.02, 0) for t in attacks])
    source = chart.save(tmp_path / "clicks_1K.json")
    original = source.read_bytes()
    app = App(Config())
    state = {"phase": 0, "frames": 0}
    started = time.monotonic()
    real_draw = screen.draw_snap_screen
    def draw(app, review):
        assert time.monotonic() - started < 60, "Beat-Snap worker/UI stalled"
        buttons = real_draw(app, review)
        state["frames"] += 1
        if review.result is not None:
            state["phase"] = 2
        return buttons
    def events():
        assert not list(tmp_path.glob("*_snap*.json")), "Worker saved without explicit acceptance"
        if state["phase"] == 0:
            state["phase"] = 1
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b)]
        if state["phase"] == 2:
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)]
        return []
    monkeypatch.setattr(screen, "draw_snap_screen", draw)
    monkeypatch.setattr(app, "events", events)
    monkeypatch.setattr(app, "frame", lambda **_: time.sleep(0.005))
    try:
        path = screen.run_snap_screen(app, track, chart, source)
        assert path and len(Chart.load(path).notes) == len(attacks)
        assert state["frames"] >= 2 and source.read_bytes() == original
    finally:
        pygame.quit()


def test_beat_snap_can_analyze_and_render_without_torch_in_subprocess():
    code = '''
import sys, importlib.abc
class HideTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("Torch hidden for Beat-Snap test")
sys.meta_path.insert(0, HideTorch())
import numpy as np, pygame
from src.audio_engine import AudioTrack
from src.beat_snap import analyze_beats
from src.snap_screen import SnapState, draw_snap_screen
from src.storage import Config
from src.ui import App
track=AudioTrack(np.zeros((8000,2),np.float32),8000)
assert analyze_beats(track).onsets == ()
app=App(Config())
try:
    draw_snap_screen(app, SnapState())
    assert "torch" not in sys.modules
finally:
    pygame.quit()
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_nonisochronous_attacks_do_not_force_the_notes_to_a_constant_bpm():
    attacks = (0.41, 1.07, 1.68, 2.42, 3.04)
    rate = 24000
    pcm = np.zeros((rate * 4, 2), np.float32)
    click = synth_click(rate)
    for attack in attacks:
        start = round(attack * rate)
        pcm[start:start + len(click)] += click
    track = AudioTrack(pcm, rate)
    chart = Chart(track.reference(), ["space"], [Note(t + 0.02, 0) for t in attacks] + [Note(3.6, 0)])
    result = snap_chart(chart, analyze_beats(track))
    assert len(result.grid.onsets) == len(attacks)
    assert all(abs(n.time - t) < 0.005 for n, t in zip(result.chart.notes, attacks))
    assert result.chart.notes[-1].time == 3.6


@pytest.mark.parametrize("seed", range(4))
def test_random_multilane_snaps_preserve_schema_note_identity_and_maximum_shift(seed):
    rng = np.random.default_rng(seed)
    notes = []
    for lane in range(8):
        starts = [float(t + rng.uniform(-0.023, 0.023)) for t in np.arange(0.15, 3.8, 0.23)]
        for index, start in enumerate(starts):
            next_start = starts[index + 1] if index + 1 < len(starts) else 4.0
            length = min(float(rng.uniform(0.151, 0.21)), next_start - start - 0.001) if rng.random() < 0.5 else 0.0
            notes.append(Note(start, lane, length, "Hold" if length else "Tap"))
    _, chart = make_chart(notes, keys=["a", "b", "c", "d", "e", "f", "g", "h"])
    chart.validate()  # Valid live input, even before attempting any shifts.
    result = snap_chart(chart, grid_for(chart, np.arange(0.15, 3.8, 0.23)))
    result.chart.validate()
    assert len(chart.notes) == len(result.chart.notes)
    for before, after in zip(chart.notes, result.chart.notes):
        assert (before.lane, before.duration, before.type) == (after.lane, after.duration, after.type)
        assert abs(before.time - after.time) <= 0.025 + 1e-10
    assert all(change.before != change.after for change in result.changes)


def test_review_cooperative_cancel_leaves_original_and_no_copy(snap_ui, monkeypatch):
    import pygame
    import src.snap_screen as screen
    from src.ai_screen import BackgroundJob
    from src.ml import check_cancel
    app, track, chart, source = snap_ui
    monkeypatch.setattr(screen, "BackgroundJob", BackgroundJob)
    before = source.read_bytes()
    def work(*_, progress, cancel):
        assert cancel.wait(3)
        check_cancel(cancel)
    monkeypatch.setattr(screen, "analyze_beats", work)
    actions = iter([pygame.K_b, pygame.K_ESCAPE])
    deadline = time.monotonic() + 4
    def events():
        assert time.monotonic() < deadline, "Cancellation did not leave the screen"
        key = next(actions, None)
        return [pygame.event.Event(pygame.KEYDOWN, key=key)] if key is not None else []
    monkeypatch.setattr(app, "events", events)
    monkeypatch.setattr(app, "frame", lambda **_: time.sleep(0.001))
    assert screen.run_snap_screen(app, track, chart, source) is None
    assert source.read_bytes() == before and list(source.parent.glob("*.json")) == [source]


@pytest.mark.parametrize("route", ["record", "saved"])
def test_menu_routes_saved_or_new_live_chart_to_optional_review(snap_ui, monkeypatch, route):
    import pygame
    import main
    import src.live_mapper as mapper
    import src.snap_screen as screen
    import src.ui as ui
    app, track, chart, source = snap_ui
    reviewed = []
    monkeypatch.setattr(ui, "App", lambda _: app)
    monkeypatch.setattr(ui, "choose_keys", lambda _: chart.keys)
    monkeypatch.setattr(ui, "file_picker", lambda *_, **__: source)
    monkeypatch.setattr(main, "generate_test_track", lambda **_: track)
    monkeypatch.setattr(main, "environment_report", lambda: {"audio": {"output_devices": []}})
    monkeypatch.setattr(mapper, "run_live_mapper", lambda *args: (chart, source))
    monkeypatch.setattr(screen, "run_snap_screen", lambda *args: reviewed.append(args))
    actions = iter([pygame.K_2 if route == "record" else pygame.K_b, pygame.K_ESCAPE])
    monkeypatch.setattr(app, "events", lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(actions))])
    main.run_menu(Config())
    assert reviewed == [(app, track, chart, source)]
