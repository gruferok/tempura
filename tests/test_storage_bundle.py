import sys
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
    exe_dir, local = tmp_path / "exe", tmp_path / "local"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "TempuraRhythmLab.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "meipass"), raising=False)
    monkeypatch.delenv("TEMPURA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    real_writable = storage._writable_dir
    monkeypatch.setattr(storage, "_writable_dir",
                        lambda p: False if p == exe_dir.resolve() else real_writable(p))
    root, bundle = storage._resolve_roots()
    assert root == local / "Tempura"
    assert bundle == tmp_path / "meipass"


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
