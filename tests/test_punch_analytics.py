import pytest

from src.audio_engine import generate_test_track
from src.chart import Chart, Note
from src.gameplay import (Judge, PunchInEditor, error_statistics, save_offset_adjustment)
from src.storage import Config


def fixture_chart():
    track = generate_test_track(duration=8, sample_rate=8000)
    return Chart(track.reference(), ["f", "j"],
                 [Note(1.0, 0), Note(2.0, 1), Note(3.0, 0, 0.4, "Hold"), Note(6.0, 1)], {})


def test_punch_in_replaces_only_onsets_inside_segment_and_preserves_outside():
    editor = PunchInEditor(fixture_chart())
    editor.replace(1.8, 4.0, [Note(2.5, 1), Note(3.5, 0, 0.2, "Hold")])
    assert editor.chart.notes == [Note(1.0, 0), Note(2.5, 1), Note(3.5, 0, 0.2, "Hold"), Note(6.0, 1)]
    editor.chart.validate()


def test_punch_trims_preexisting_hold_crossing_interval_start():
    chart = fixture_chart()
    editor = PunchInEditor(chart)
    editor.replace(3.2, 4.0, [Note(3.5, 0)])
    trimmed = next(n for n in editor.chart.notes if n.time == 3.0 and n.lane == 0)
    assert trimmed.type == "Hold" and trimmed.duration == pytest.approx(0.2)
    assert Note(3.5, 0) in editor.chart.notes


def test_error_mean_std_unstable_rate_and_histogram():
    judge = Judge(fixture_chart())
    judge.key_down(0, 1.010)
    judge.key_down(1, 2.030)
    judge.key_down(0, 3.010)
    stats = error_statistics(judge)
    assert stats["count"] == 3
    assert stats["mean_error_ms"] == pytest.approx(16.6666667)
    assert stats["std_error_ms"] == pytest.approx(9.42809, rel=1e-4)
    assert stats["unstable_rate"] == pytest.approx(stats["std_error_ms"] * 10)
    assert sum(stats["histogram"]) == 3
    assert len(stats["histogram"]) == 18


def test_offset_adjustment_adds_signed_mean_and_saves_config(tmp_path, monkeypatch):
    class Dummy:
        config = Config(audio_offset_ms=12.0)
    dummy = Dummy()
    saved = []
    monkeypatch.setattr(dummy.config, "save", lambda: saved.append(dummy.config.audio_offset_ms))
    assert save_offset_adjustment(dummy, -8.5) == pytest.approx(3.5)
    assert saved == [pytest.approx(3.5)]
    assert save_offset_adjustment(dummy, 900) == 500
