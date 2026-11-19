import src.storage as storage


def test_seed_from_bundle_is_noop_when_not_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "BUNDLE_ROOT", tmp_path / "bundle")
    monkeypatch.setattr(storage, "ROOT", tmp_path / "bundle")
    storage.seed_from_bundle()  # must not raise, must not touch anything


def test_seed_from_bundle_copies_defaults_without_overwriting(monkeypatch, tmp_path):
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    (bundle / "data" / "charts").mkdir(parents=True)
    (bundle / "data" / "models").mkdir(parents=True)
    (bundle / "config.json").write_text('{"default": true}', encoding="utf-8")
    (bundle / "data" / "charts" / "demo.json").write_text("{}", encoding="utf-8")
    (bundle / "data" / "models" / "m.pt").write_bytes(b"x")
    root.mkdir()
    (root / "config.json").write_text('{"user": true}', encoding="utf-8")
    monkeypatch.setattr(storage, "BUNDLE_ROOT", bundle)
    monkeypatch.setattr(storage, "ROOT", root)

    storage.seed_from_bundle()

    assert (root / "config.json").read_text(encoding="utf-8") == '{"user": true}'
    assert (root / "data" / "charts" / "demo.json").is_file()
    assert (root / "data" / "models" / "m.pt").is_file()
