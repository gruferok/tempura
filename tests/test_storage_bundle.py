import logging
import sys
import tempfile
from pathlib import Path

import src.storage as storage


def _bundle(tmp_path):
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    (bundle / "data" / "charts" / "nested").mkdir(parents=True)
    (bundle / "data" / "models").mkdir(parents=True)
    (bundle / "config.json").write_text('{"default": true}', encoding="utf-8")
    (bundle / "data" / "charts" / "nested" / "demo.json").write_text("{}", encoding="utf-8")
    (bundle / "data" / "models" / "m.pt").write_bytes(b"x")
    root.mkdir()
    return bundle, root


def test_seed_is_noop_when_bundle_is_root(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "BUNDLE_ROOT", tmp_path)
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    storage.seed_from_bundle()
    assert not any(tmp_path.iterdir())


def test_seed_copies_nested_files_and_never_overwrites(monkeypatch, tmp_path):
    bundle, root = _bundle(tmp_path)
    (root / "config.json").write_text('{"user": true}', encoding="utf-8")
    monkeypatch.setattr(storage, "BUNDLE_ROOT", bundle)
    monkeypatch.setattr(storage, "ROOT", root)
    monkeypatch.setattr(storage, "CONFIG_PATH", root / "config.json")

    storage.seed_from_bundle()
    assert (root / "config.json").read_text(encoding="utf-8") == '{"user": true}'
    assert (root / "data" / "charts" / "nested" / "demo.json").is_file()
    assert not (root / "data" / "models" / "m.pt").exists()  # models are never duplicated into ROOT

    (root / "data" / "charts" / "nested" / "demo.json").write_text('{"edited": 1}', encoding="utf-8")
    storage.seed_from_bundle()  # idempotent: user edits survive a second run
    assert (root / "data" / "charts" / "nested" / "demo.json").read_text(encoding="utf-8") == '{"edited": 1}'


def test_frozen_falls_back_when_exe_dir_not_writable(monkeypatch, tmp_path):
    # The write probe is tempfile.NamedTemporaryFile(dir=exe_dir) with a unique name;
    # denying the probe (and Path.touch, if the probe ever regresses to a fixed file)
    # with PermissionError is the closest emulation of a read-only install folder.
    exe_dir, local = tmp_path / "exe", tmp_path / "local"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "TempuraRhythmLab.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "meipass"), raising=False)
    monkeypatch.delenv("TEMPURA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def denied_for_exe_dir(*args, dir=None, **kwargs):
        if dir is not None and Path(dir).resolve() == exe_dir.resolve():
            raise PermissionError("exe folder is read-only")
        return real_named_temporary_file(*args, dir=dir, **kwargs)

    def touch_denied(self, *args, **kwargs):
        raise PermissionError("exe folder is read-only")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", denied_for_exe_dir)
    monkeypatch.setattr(Path, "touch", touch_denied)
    root, bundle = storage._resolve_roots()
    assert root == local / "Tempura"
    assert bundle == tmp_path / "meipass"


def test_frozen_falls_back_to_tempdir_with_warning_when_nothing_is_writable(monkeypatch, tmp_path, caplog):
    exe_dir = tmp_path / "exe"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "TempuraRhythmLab.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "meipass"), raising=False)
    monkeypatch.delenv("TEMPURA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    def denied(*args, **kwargs):
        raise PermissionError("nothing is writable")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", denied)
    with caplog.at_level(logging.WARNING):
        root, bundle = storage._resolve_roots()
    assert root == Path(tempfile.gettempdir()) / "Tempura"  # last-resort fallback
    assert bundle == tmp_path / "meipass"
    assert any("no writable data folder" in message for message in caplog.messages)


def test_tempura_home_overrides_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TEMPURA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    root, _ = storage._resolve_roots()
    assert root == (tmp_path / "home").resolve()
    assert isinstance(root, Path)


def test_clean_root_gets_config_after_seed_and_loads(monkeypatch, tmp_path):
    bundle, root = _bundle(tmp_path)
    root.rmdir()  # clean machine: ROOT does not exist yet
    monkeypatch.setattr(storage, "BUNDLE_ROOT", bundle)
    monkeypatch.setattr(storage, "ROOT", root)
    monkeypatch.setattr(storage, "CONFIG_PATH", root / "config.json")
    storage.seed_from_bundle()
    assert (root / "config.json").read_text(encoding="utf-8") == '{"default": true}'
    assert storage.Config.load(root / "config.json") is not None


def test_frozen_simulation_seeds_into_tempura_home(monkeypatch, tmp_path):
    bundle, root = _bundle(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "exe" / "TempuraRhythmLab.exe"))
    monkeypatch.setenv("TEMPURA_HOME", str(root))
    resolved_root, resolved_bundle = storage._resolve_roots()
    assert resolved_root == root.resolve() and resolved_bundle == bundle
    monkeypatch.setattr(storage, "ROOT", resolved_root)
    monkeypatch.setattr(storage, "BUNDLE_ROOT", resolved_bundle)
    monkeypatch.setattr(storage, "CONFIG_PATH", resolved_root / "config.json")
    storage.seed_from_bundle()
    assert (root / "config.json").is_file()
    assert (root / "data" / "charts" / "nested" / "demo.json").is_file()
    assert not (root / "data" / "models").exists() or not any((root / "data" / "models").iterdir())


def test_resolve_resource_two_step_for_non_model_resources(monkeypatch, tmp_path):
    bundle, root = _bundle(tmp_path)
    monkeypatch.setattr(storage, "ROOT", root)
    monkeypatch.setattr(storage, "BUNDLE_ROOT", bundle)
    # no user copy: the read-only bundled default (chart) is found
    assert storage.resolve_resource("data/charts/nested/demo.json") == bundle / "data" / "charts" / "nested" / "demo.json"
    # user copy wins (ROOT first)
    target = root / "data" / "charts" / "nested"
    target.mkdir(parents=True)
    (target / "demo.json").write_text('{"user": true}', encoding="utf-8")
    assert storage.resolve_resource("data/charts/nested/demo.json") == target / "demo.json"
    # missing everywhere: the creation default stays in ROOT (writes never go to BUNDLE_ROOT)
    assert storage.resolve_resource("data/audio/click.wav") == root / "data" / "audio" / "click.wav"
