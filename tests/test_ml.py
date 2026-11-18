"""Offline ML tests. All numerical tests force CPU; no audio/GPU/display needed.

Loss reduction tests fitting, NOT unseen-song quality or few-shot style transfer.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-ml.txt to run the ML suite")

from src.audio_engine import AudioTrack, generate_test_track
from src.chart import Chart, Note
from src.ml import Cancelled, FEATURE_VERSION, MAX_LANES, check_cancel
from src.ml.dataset import (SequenceDataset, TrainingExample, catalog_charts, encode_targets,
                            load_chart_track, load_examples, make_bootstrap_examples, procedural_track)
from src.ml.features import FEATURE_CONFIG, FEATURE_DIM, extract_features, frame_count
from src.ml.generator import (decode_probabilities, generate_chart, infer_probabilities, peak_pick,
                              threshold_for_sensitivity)
from src.ml.generator_keys import keys_for_lanes
from src.ml.model import ModelConfig, RhythmCRNN, select_device
from src.ml.trainer import (class_weights, fit_model, load_checkpoint, masked_loss, save_checkpoint,
                            split_by_track, train_model)
from src.storage import Config


@pytest.fixture
def tiny_config():
    return ModelConfig(channels=12, hidden_size=12, condition_dim=4, dropout=0)


def silent_track(duration=2, sample_rate=16000):
    return AudioTrack(np.zeros((round(duration * sample_rate), 2), np.float32), sample_rate)


def teaching_chart():
    track = generate_test_track(duration=2.5, sample_rate=16000, lead_in=0.25)
    chart = Chart(track.reference(), ["f", "j"], [Note(0.25, 0), Note(0.75, 1),
                                                 Note(1.25, 0, 0.25, "Hold"), Note(1.75, 1), Note(2.25, 0)])
    return track, chart


@pytest.mark.parametrize("sample_rate,duration", [(16000, 1), (44100, 1.23), (48000, 1.005), (22050, 0.5)])
def test_features_are_exactly_100_hz_independent_of_source_rate(sample_rate, duration):
    samples = round(sample_rate * duration)
    t = np.arange(samples) / sample_rate
    audio = (np.sin(2 * np.pi * 330 * t) * 0.2).astype(np.float32)
    track = AudioTrack(audio, sample_rate)
    sequence = extract_features(track, cache_dir=None)
    assert sequence.values.shape == (math.ceil(track.duration * 100 - 1e-8), 130)
    assert sequence.fps == 100 and sequence.values.dtype == np.float32
    assert np.diff(sequence.times) == pytest.approx(0.01)
    assert sequence.times[0] == 0 and sequence.times[-1] < track.duration
    assert np.isfinite(sequence.values).all()
    assert np.max(np.abs(sequence.values)) <= 8
    assert np.max(np.abs(sequence.values.mean(axis=0))) < 0.5  # clipping may shift the mean slightly


def test_silence_features_finite_and_fft_does_not_shift_grid():
    silence = extract_features(silent_track(), cache_dir=None)
    assert np.all(silence.values == 0)
    audio = np.zeros(16000, np.float32)
    audio[6400] = 0.8  # exact transient at 0.400 seconds
    sequence = extract_features(AudioTrack(audio, 16000), cache_dir=None)
    attack_frame = int(np.argmax(sequence.values[:, 128]))
    assert abs(attack_frame / 100 - 0.4) <= 0.021
    with pytest.raises(ValueError):
        replace(FEATURE_CONFIG, hop_length=161).validate()
    with pytest.raises(ValueError):
        frame_count(math.nan)


def test_feature_cache_is_reused_and_corruption_is_recomputed(tmp_path, monkeypatch):
    import librosa
    track, _ = teaching_chart()
    original = extract_features(track, cache_dir=tmp_path)
    assert not original.cache_hit
    with monkeypatch.context() as m:
        m.setattr(librosa, "stft", lambda *_, **__: pytest.fail("Cache hit recomputed the STFT"))
        restored = extract_features(track, cache_dir=tmp_path)
        assert restored.cache_hit and np.array_equal(original.values, restored.values)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    files[0].write_bytes(b"interrupted cache")
    recomputed = extract_features(track, cache_dir=tmp_path)
    assert not recomputed.cache_hit and np.array_equal(original.values, recomputed.values)
    assert not list(tmp_path.glob("*.tmp"))


def test_gaussian_targets_sigma_fractional_centres_and_hold_interval():
    chart = Chart(silent_track().reference(), ["f", "j"],
                  [Note(0.5, 0), Note(1, 1, 0.3, "Hold")])
    onset, hold = encode_targets(chart)
    assert onset.shape == hold.shape == (200, MAX_LANES)
    assert onset[50, 0] == 1
    assert onset[49, 0] == pytest.approx(math.exp(-0.5 / 1.5 ** 2))
    assert onset[48, 0] == onset[52, 0]
    assert np.all(hold[100:130, 1] == 1) and hold[99, 1] == hold[130, 1] == 0
    assert np.count_nonzero(onset[:, 2:]) == np.count_nonzero(hold[:, 2:]) == 0
    chart.notes = [Note(0.505, 0)]
    onset, _ = encode_targets(chart)
    assert onset[50, 0] == onset[51, 0] == pytest.approx(math.exp(-0.5 * (0.5 / 1.5) ** 2))
    # Overlapping Gaussian labels use maximum, never sum above 1.
    chart.notes = [Note(0.50, 0), Note(0.52, 0)]
    onset, _ = encode_targets(chart)
    assert onset.max() <= 1 and onset[51, 0] == pytest.approx(math.exp(-0.5 / 1.5 ** 2))


def test_window_masks_exclude_padding_and_inactive_lanes():
    track, chart = teaching_chart()
    features = extract_features(track, cache_dir=None).values
    onset, hold = encode_targets(chart)
    example = TrainingExample(features, onset, hold, 2, track.track_id, "test")
    dataset = SequenceDataset([example], window_seconds=1, stride_seconds=0.5)
    assert [start for _, start in dataset.windows] == [0, 50, 100, 150]
    assert torch.equal(dataset[1]["onset_target"], torch.from_numpy(onset[50:150]))
    short = TrainingExample(features[:20], onset[:20], hold[:20], 2, "short", "test")
    padded = SequenceDataset([short], window_seconds=1)[0]
    assert padded["features"].shape == (100, FEATURE_DIM)
    assert padded["mask"].sum() == 40
    assert padded["mask"][20:].sum() == padded["mask"][:, 2:].sum() == 0
    assert padded["length"] == 20


def test_masked_loss_has_no_inactive_or_padded_gradients():
    onset = torch.zeros((1, 10, 8), requires_grad=True)
    hold = torch.zeros_like(onset, requires_grad=True)
    mask = torch.zeros_like(onset)
    mask[:, :5, :2] = 1
    batch = {"onset_target": torch.zeros_like(onset), "hold_target": torch.zeros_like(hold), "mask": mask}
    loss = masked_loss({"onset_logits": onset, "hold_logits": hold}, batch, torch.ones(8), torch.ones(8))
    loss.backward()
    for tensor in (onset, hold):
        assert tensor.grad[:, :5, :2].abs().sum() > 0
        assert tensor.grad[:, 5:].abs().sum() == tensor.grad[:, :, 2:].abs().sum() == 0


def test_model_shapes_conditioning_and_packed_sequences(tiny_config):
    torch.manual_seed(12)
    model = RhythmCRNN(tiny_config)
    features = torch.randn(1, 25, FEATURE_DIM).expand(4, -1, -1).clone()
    outputs = model(features, torch.tensor([1, 2, 4, 8]), torch.tensor([25, 25, 20, 15]))
    assert set(outputs) == {"onset_logits", "hold_logits"}
    assert all(value.shape == (4, 25, 8) and torch.isfinite(value).all() for value in outputs.values())
    assert not torch.allclose(outputs["onset_logits"][0], outputs["onset_logits"][1])
    sum(value.mean() for value in outputs.values()).backward()
    assert model.lane_embedding.weight.grad[1].abs().sum() > 0
    with pytest.raises(ValueError):
        model(features, 9)


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected
    assert select_device("cpu").type == "cpu"
    if not cuda:
        with pytest.raises(ValueError):
            select_device("cuda")


@pytest.mark.parametrize("lanes", range(1, 9))
def test_every_lane_count_has_valid_layout_and_procedural_data(lanes):
    track, chart = procedural_track(100 + lanes, lanes, duration=1.5)
    chart.validate()
    assert chart.lanes == lanes and chart.notes
    assert track.sample_rate == 16000 and np.isfinite(track.pcm).all()
    assert chart.keys == keys_for_lanes(lanes)
    assert all(note.lane < lanes and note.end <= track.duration for note in chart.notes)


def test_catalog_excludes_feedback_loop_examples_empty_and_corrupt(tmp_path):
    track, chart = teaching_chart()
    chart.save(tmp_path / "user.json")
    Chart(chart.track, chart.keys, chart.notes, {"ai_generated": True}).save(tmp_path / "ai.json")
    Chart(chart.track, chart.keys, chart.notes, {"ai_generated": True, "human_reviewed": "false"}).save(tmp_path / "not_reviewed.json")
    Chart(chart.track, chart.keys, chart.notes, {"ai_generated": True, "human_reviewed": True}).save(tmp_path / "reviewed.json")
    Chart(chart.track, chart.keys, chart.notes, {"example_only": True}).save(tmp_path / "example.json")
    Chart(chart.track, chart.keys, chart.notes, {"output_underruns": 1}).save(tmp_path / "underrun.json")
    Chart(chart.track, chart.keys, []).save(tmp_path / "empty.json")
    (tmp_path / "corrupt.json").write_text("not JSON")
    catalog, skipped = catalog_charts(tmp_path)
    assert {p.name for p, _ in catalog} == {"user.json", "reviewed.json"}
    assert len(skipped) == 6
    examples, skipped = load_examples(tmp_path, cache_dir=tmp_path / "cache")
    assert len(examples) == 2 and len(skipped) == 6
    assert all(e.group_id == track.track_id for e in examples)


def test_missing_or_mismatched_audio_is_skipped_not_substituted(tmp_path):
    import soundfile as sf
    audio = tmp_path / "real.wav"
    sf.write(audio, np.zeros(16000, np.float32), 16000)
    track = AudioTrack.from_file(audio, 16000)
    reference = track.reference()
    reference["id"] = "wrong-hash"
    Chart(reference, ["space"], [Note(0.5, 0)]).save(tmp_path / "mismatch.json")
    reference = track.reference()
    reference["audio_path"] = str(tmp_path / "missing.wav")
    Chart(reference, ["space"], [Note(0.5, 0)]).save(tmp_path / "missing.json")
    examples, skipped = load_examples(tmp_path, cache_dir=None)
    assert not examples and len(skipped) == 2
    assert any("SHA-256" in item["reason"] for item in skipped)


def test_validation_groups_never_split_layouts_of_one_track():
    values = np.zeros((10, FEATURE_DIM), np.float32)
    target = np.zeros((10, 8), np.float32)
    examples = [TrainingExample(values, target, target, lanes, group, group)
                for group in ("a", "b", "c") for lanes in (1, 2)]
    train, validation = split_by_track(examples)
    assert train and validation
    assert not {e.group_id for e in train}.intersection(e.group_id for e in validation)
    assert split_by_track(examples[:2])[1] == []
    assert all(torch.isfinite(w).all() for w in class_weights(examples))


@pytest.fixture
def fitted_model(tmp_path, tiny_config):
    _, chart = teaching_chart()
    directory = tmp_path / "charts"
    chart.save(directory / "live.json")
    result = train_model(charts_dir=directory, checkpoint_path=tmp_path / "personal.pt", epochs=8,
                         bootstrap_epochs=0, batch_size=1, learning_rate=0.02, window_seconds=2.5,
                         device="cpu", reset=True, cache_dir=tmp_path / "cache", model_config=tiny_config, seed=31)
    return result, directory


def test_loss_decreases_on_real_chart_and_checkpoint_round_trips(fitted_model):
    result, _ = fitted_model
    history = result.metadata["history"]
    assert history[-1]["train_eval_loss"] < history[0]["train_eval_loss"] * 0.8
    assert result.metadata["personalized"] and result.metadata["current_dataset_maps"] == 1
    assert not result.metadata["validation"]["available"]
    assert result.checkpoint.is_file() and result.checkpoint.with_suffix(".json").is_file()
    model, metadata = load_checkpoint(result.checkpoint, device="cpu")
    assert metadata == result.metadata and next(model.parameters()).device.type == "cpu"
    assert model.config.feature_dim == 130


def test_finetuning_continues_existing_weights_and_cancel_keeps_previous_checkpoint(fitted_model):
    initial, charts = fitted_model
    first_model, _ = load_checkpoint(initial.checkpoint, device="cpu")
    updated = train_model(charts_dir=charts, checkpoint_path=initial.checkpoint, epochs=1, bootstrap_epochs=0,
                          batch_size=1, device="cpu", cache_dir=None, window_seconds=2.5)
    second_model, _ = load_checkpoint(updated.checkpoint, device="cpu")
    assert updated.metadata["training_runs"] == 2
    assert updated.metadata["checkpoint_id"] != initial.metadata["checkpoint_id"]
    assert any(not torch.equal(a, b) for a, b in zip(first_model.parameters(), second_model.parameters()))
    before = hashlib.sha256(updated.checkpoint.read_bytes()).hexdigest()
    sidecar = updated.checkpoint.with_suffix(".json").read_bytes()
    cancel = threading.Event()
    def progress(event):
        if event.get("phase") == "finetune" and event.get("step") == 1:
            cancel.set()
    with pytest.raises(Cancelled):
        train_model(charts_dir=charts, checkpoint_path=updated.checkpoint, epochs=3, bootstrap_epochs=0,
                    device="cpu", cache_dir=None, window_seconds=2.5, progress=progress, cancel=cancel)
    assert hashlib.sha256(updated.checkpoint.read_bytes()).hexdigest() == before
    assert updated.checkpoint.with_suffix(".json").read_bytes() == sidecar
    assert not list(updated.checkpoint.parent.glob("*.tmp"))


def test_peak_pick_boundaries_plateaus_and_minimum_distance():
    probability = np.zeros(40)
    probability[[0, 4, 12, 39]] = [0.99, 0.9, 0.85, 0.8]
    probability[25:28] = 0.95
    assert peak_pick(probability).tolist() == [0, 12, 26, 39]
    probability[0] = 0.7
    assert peak_pick(probability).tolist() == [4, 12, 26, 39]
    assert len(peak_pick(np.zeros(20))) == 0
    for invalid in (np.array([math.nan]), np.array([1.1])):
        with pytest.raises(ValueError):
            peak_pick(invalid)
    assert threshold_for_sensitivity(0) > threshold_for_sensitivity(1)


def test_hold_decoding_is_lane_aware_bridges_short_dips_and_cannot_overlap(tmp_path):
    track = silent_track()
    onset, hold = np.zeros((200, 8), np.float32), np.zeros((200, 8), np.float32)
    onset[[10, 16, 40, 90, 120], 0] = [0.95, 0.8, 0.9, 0.9, 0.9]
    onset[10, 1] = 0.9  # a chord on another lane is not suppressed
    hold[9:35, 0] = 0.9
    hold[20:22, 0] = 0
    hold[90:, 0] = 0.9
    hold[10:25, 1] = 0.9  # exactly 150 ms -> Tap, not Hold
    chart = decode_probabilities(track, onset, hold, 2)
    lane_zero = [n for n in chart.notes if n.lane == 0]
    assert len(lane_zero) == 4
    assert lane_zero[0].type == "Hold" and lane_zero[0].end == pytest.approx(0.35)
    assert lane_zero[1].type == "Tap"
    assert lane_zero[2].end == pytest.approx(lane_zero[3].time)
    assert next(n for n in chart.notes if n.lane == 1).type == "Tap"
    assert all(b.time >= a.end for a, b in zip(lane_zero, lane_zero[1:]))
    path = chart.save(tmp_path / "draft.json")
    restored = Chart.load(path)
    assert restored.to_dict() == chart.to_dict()
    from src.gameplay import Judge
    assert len(Judge(restored).notes) == len(chart.notes)


def test_generation_runs_real_inference_and_outputs_playable_json(tmp_path, tiny_config):
    torch.manual_seed(4)
    model = RhythmCRNN(tiny_config)
    # A constant-probability model is a deterministic structural fixture, not a
    # claim of learned musical quality. Peak-picking centres each plateau.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.onset_head.bias.fill_(2)
        model.hold_head.bias.fill_(-8)
    path = save_checkpoint(model, tmp_path / "model.pt", {"checkpoint_id": "unit-test", "origin": "test_fixture"})
    track = generate_test_track(duration=2, sample_rate=16000, lead_in=0.25)
    result = generate_chart(track, 2, checkpoint_path=path, device="cpu", output_dir=tmp_path,
                            cache_dir=tmp_path / "cache", keys=["g", "h"])
    assert result.path.name == "test_track_ai_2K.json"
    chart = Chart.load(result.path)
    assert len(chart.notes) == 2 and chart.keys == ["g", "h"]
    assert all(n.type == "Tap" for n in chart.notes)
    assert chart.metadata["ai_generated"] and chart.metadata["checkpoint_id"] == "unit-test"
    assert load_chart_track(chart).track_id == track.track_id
    features = extract_features(track, cache_dir=None).values
    onsets, holds = infer_probabilities(model, features, 2, window_frames=100, overlap_frames=20)
    assert onsets.shape == holds.shape == (200, 8)
    assert np.isfinite(onsets).all() and np.all(onsets[:, 2:] == 0)
    assert np.allclose(onsets[:, :2], torch.sigmoid(torch.tensor(2.0)).item(), atol=1e-6)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["feature_version"] = FEATURE_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="несовместимые"):
        load_checkpoint(path, device="cpu")


def test_ai_background_job_bounds_progress_and_cancels_cooperatively():
    from src.ai_screen import BackgroundJob
    def work(progress, cancel):
        for i in range(200):
            progress({"phase": "test", "step": i})
        return {"ok": True}
    job = BackgroundJob(work).start()
    assert job.done.wait(3) and job.result == {"ok": True}
    events = job.poll()
    assert len(events) <= 64 and events[-1]["step"] == 199
    assert not job.error
    with pytest.raises(RuntimeError):
        job.start()
    def wait_for_cancel(progress, cancel):
        assert cancel.wait(3)
        check_cancel(cancel)
    job = BackgroundJob(wait_for_cancel).start()
    job.cancel.set()
    assert job.done.wait(3) and job.cancelled and not job.error


def test_ai_screen_renders_headlessly_without_audio(tmp_path, monkeypatch):
    import pygame
    from src.ai_screen import StudioState, draw_ai_screen
    from src.audio_engine import AudioEngine
    from src.ui import App
    monkeypatch.setattr(AudioEngine, "_open", lambda *_: pytest.fail("AI rendering opened audio"))
    app = App(Config())
    try:
        states = [StudioState(device="CPU"), StudioState(model_exists=True, personalized=True, dataset_count=3,
                  device="CPU", keys=keys_for_lanes(8), busy=True, phase="finetune", epoch=2, epochs=8,
                  loss=0.2345, progress=0.25, elapsed=4.2),
                  StudioState(model_exists=True, generated_path="data/charts/song_ai_2K.json", generated_notes=30)]
        for state in states:
            buttons = draw_ai_screen(app, state)
            assert {"train", "generate", "play", "density"} <= set(buttons)
            pygame.display.flip()
        pygame.image.save(app.screen, tmp_path / "ai.png")
        assert (tmp_path / "ai.png").stat().st_size > 1000
        assert pygame.mixer.get_init() is None
    finally:
        pygame.quit()


def test_ai_studio_actions_train_generate_then_quick_playtest(monkeypatch, tmp_path):
    import pygame
    import src.ai_screen as screen
    import src.ml.generator as generator_module
    import src.ml.trainer as trainer_module
    import src.gameplay as gameplay
    from src.ui import App

    track, chart = teaching_chart()
    played = []
    calls = []
    class ImmediateJob(screen.BackgroundJob):
        def start(self):
            self._run()
            return self
    monkeypatch.setattr(screen, "BackgroundJob", ImmediateJob)
    monkeypatch.setattr(screen, "inspect_studio", lambda *_: {"dataset_count": 1, "skipped_count": 0,
                        "device": "cpu", "model_exists": False, "personalized": False})
    def train(**kwargs):
        calls.append("train")
        kwargs["progress"]({"phase": "finetune", "epoch": 1, "epochs": 1, "loss": 0.2})
        return SimpleNamespace(metadata={"personalized": True, "current_dataset_maps": 1, "skipped_count": 0})
    def generate(selected, lanes, **kwargs):
        calls.append("generate")
        assert selected is track and lanes == 2 and kwargs["keys"] == ["f", "j"]
        return SimpleNamespace(chart=chart, path=tmp_path / "test_ai_2K.json")
    monkeypatch.setattr(trainer_module, "train_model", train)
    monkeypatch.setattr(generator_module, "generate_chart", generate)
    monkeypatch.setattr(gameplay, "run_gameplay", lambda app, audio, data: played.append((audio, data)))
    app = App(Config())
    keys = iter([pygame.K_t, pygame.K_g, pygame.K_RETURN, pygame.K_ESCAPE])
    monkeypatch.setattr(app, "events", lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(keys))])
    monkeypatch.setattr(app, "frame", lambda **_: None)
    try:
        screen.run_ai_screen(app, track, checkpoint_path=tmp_path / "model.pt", charts_dir=tmp_path)
        assert calls == ["train", "generate"] and played == [(track, chart)]
    finally:
        pygame.quit()


def test_core_and_ai_screen_can_render_even_when_torch_is_uninstalled():
    code = '''
import sys, importlib.abc
class HideTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("Torch intentionally hidden for base-game smoke test")
sys.meta_path.insert(0, HideTorch())
from main import headless_smoke
from src.storage import Config
result = headless_smoke(Config())
assert "ai_studio" in result["rendered_screens"]
assert "torch" not in sys.modules
print("base game and AI rendering work without Torch")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_cli_cold_start_and_generation_need_no_audio_or_display(tmp_path):
    import soundfile as sf
    model = tmp_path / "cli.pt"
    charts = tmp_path / "charts"
    charts.mkdir()
    common = [sys.executable, "main.py", "--model", str(model), "--charts-dir", str(charts), "--ml-device", "cpu", "--no-cache"]
    env = {key: value for key, value in os.environ.items() if key not in ("DISPLAY", "WAYLAND_DISPLAY", "SDL_VIDEODRIVER")}
    trained = subprocess.run(common + ["--train", "--bootstrap-samples", "1", "--bootstrap-epochs", "1", "--epochs", "0"],
                             capture_output=True, text=True, env=env, timeout=60)
    assert trained.returncode == 0, trained.stderr
    metadata = json.loads(trained.stdout)["training"]["metadata"]
    assert model.is_file() and metadata["origin"] == "bootstrap_only"
    assert not metadata["personalized"] and metadata["bootstrap"]["external_pretrained"] is False
    assert any(json.loads(line).get("phase") == "bootstrap" for line in trained.stderr.splitlines())
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.1, 16000)
    generated = subprocess.run(common + ["--generate", str(audio), "--lanes", "2", "--sensitivity", "0.8"],
                               capture_output=True, text=True, env=env, timeout=60)
    assert generated.returncode == 0, generated.stderr
    path = Path(json.loads(generated.stdout)["generation"]["chart"])
    assert path.name == "audio_ai_2K.json" and Chart.load(path).lanes == 2
    invalid = subprocess.run(common + ["--generate", str(audio), "--lanes", "9"], capture_output=True, text=True, timeout=15)
    assert invalid.returncode != 0
