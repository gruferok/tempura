import importlib

import pytest

import main as game


def _fail_import(monkeypatch, name, exc):
    real = importlib.import_module

    def fake(module, *args, **kwargs):
        if module == name:
            raise exc
        return real(module, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake)


def test_import_error_in_packaged_library_exits_2(monkeypatch, capsys):
    _fail_import(monkeypatch, "librosa", ImportError("No module named 'librosa'"))
    assert game.main(["--diagnose"]) == 2
    assert "import librosa failed" in capsys.readouterr().err


def test_import_error_in_numba_exits_2(monkeypatch, capsys):
    _fail_import(monkeypatch, "numba", ImportError("broken numba"))
    assert game.main(["--diagnose"]) == 2


def test_missing_portaudio_is_warning_in_source_checkout(monkeypatch, capsys):
    _fail_import(monkeypatch, "sounddevice", OSError("PortAudio library not found"))
    assert game.main(["--diagnose"]) == 0
    assert "sounddevice runtime unavailable" in capsys.readouterr().out


def test_missing_bundle_charts_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(game, "BUNDLE_ROOT", tmp_path)
    assert game.main(["--diagnose"]) == 2
    assert "bundle has no data/charts" in capsys.readouterr().err


def test_seed_failure_is_reported_not_raised(monkeypatch, capsys):
    def boom():
        raise PermissionError("denied")

    monkeypatch.setattr(game, "seed_from_bundle", boom)
    assert game.main(["--diagnose"]) == 2
    out = capsys.readouterr()
    assert "seed_from_bundle failed" in out.err
    assert '"problems"' in out.out
