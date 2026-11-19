"""CPU/GPU training, procedural cold start, safe checkpoints, and progress events.

No external weights or dataset are downloaded. One to three personal tracks are
supported, but low training loss is NOT evidence of unseen-track style transfer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

from ..storage import CHART_DIR, write_json
from . import (CHECKPOINT_VERSION, FEATURE_CACHE, FEATURE_VERSION, FRAME_RATE, MAX_LANES,
               MODEL_PATH, check_cancel, metadata_path, read_checkpoint, report, require_torch)
from .dataset import SequenceDataset, TrainingExample, load_examples, make_bootstrap_examples
from .features import FEATURE_CONFIG
from .model import ModelConfig, RhythmCRNN, select_device

torch = require_torch()
F = torch.nn.functional


def class_weights(examples: list[TrainingExample]) -> tuple:
    """Soft positive mass, from whole tracks rather than duplicated windows."""
    total = np.zeros(MAX_LANES, dtype=np.float64)
    onset = np.zeros_like(total)
    hold = np.zeros_like(total)
    for example in examples:
        total[:example.num_lanes] += len(example.features)
        onset[:example.num_lanes] += example.onset_target[:, :example.num_lanes].sum(axis=0)
        hold[:example.num_lanes] += example.hold_target[:, :example.num_lanes].sum(axis=0)

    def weight(positive):
        ratio = np.divide(total - positive, positive, out=np.ones_like(total), where=positive > 0)
        return torch.tensor(np.clip(ratio, 1, 30), dtype=torch.float32)
    return weight(onset), weight(hold)


def masked_loss(outputs: dict, batch: dict, onset_weight, hold_weight):
    """No gradients for inactive lanes OR padded time frames."""
    mask = batch["mask"]
    denominator = mask.sum().clamp_min(1)
    onset = F.binary_cross_entropy_with_logits(outputs["onset_logits"], batch["onset_target"],
                                               pos_weight=onset_weight, reduction="none")
    hold = F.binary_cross_entropy_with_logits(outputs["hold_logits"], batch["hold_target"],
                                              pos_weight=hold_weight, reduction="none")
    return ((onset + 0.5 * hold) * mask).sum() / denominator


def _batch_to_device(batch, device):
    # Trim padding shared by the entire batch. Packed GRU also prevents a short
    # item's padded tail from leaking into its backwards recurrent context.
    frames = int(batch["length"].max())
    result = {key: value[:, :frames].to(device) for key, value in batch.items()
              if key in ("features", "onset_target", "hold_target", "mask")}
    result["num_lanes"] = batch["num_lanes"].to(device)
    result["length"] = batch["length"]
    return result


def _evaluate(model, loader, device, weights, cancel):
    model.eval()
    loss_sum, count = 0.0, 0.0
    with torch.inference_mode():
        for raw in loader:
            check_cancel(cancel)
            batch = _batch_to_device(raw, device)
            outputs = model(batch["features"], batch["num_lanes"], batch["length"])
            loss = masked_loss(outputs, batch, *weights)
            size = float(batch["mask"].sum())
            loss_sum += float(loss) * size
            count += size
    return loss_sum / max(1, count)


def fit_model(model: RhythmCRNN, examples: list[TrainingExample], *, epochs: int = 8,
              device="auto", batch_size: int = 4, learning_rate: float = 0.001,
              window_seconds: float = 10.0, validation_examples=None, seed: int = 1234,
              phase: str = "finetune", progress=None, cancel=None) -> list[dict]:
    if not examples or not isinstance(epochs, int) or not 1 <= epochs <= 500:
        raise ValueError("Нужны обучающие примеры и от 1 до 500 эпох.")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
        raise ValueError("batch_size: от 1 до 128.")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.1:
        raise ValueError("learning_rate: больше 0 и не больше 0.1.")
    device = select_device(str(device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model.to(device)
    dataset = SequenceDataset(examples, window_seconds)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=0, generator=generator)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    validation_loader = (torch.utils.data.DataLoader(SequenceDataset(validation_examples, window_seconds),
                                                    batch_size=batch_size, num_workers=0)
                         if validation_examples else None)
    weights = tuple(w.to(device) for w in class_weights(examples))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    baseline = _evaluate(model, eval_loader, device, weights, cancel)
    history = [{"phase": phase, "epoch": 0, "loss": baseline, "train_eval_loss": baseline,
                "validation_loss": None}]
    report(progress, phase=phase, epoch=0, epochs=epochs, step=0, steps=len(loader), loss=baseline,
           message="Начальный loss (без dropout)")
    for epoch in range(1, epochs + 1):
        check_cancel(cancel)
        model.train()
        loss_sum, count = 0.0, 0.0
        for step, raw in enumerate(loader, 1):
            check_cancel(cancel)
            batch = _batch_to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"], batch["num_lanes"], batch["length"])
            loss = masked_loss(outputs, batch, *weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Loss стал неконечным. Чекпоинт не изменён.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            size = float(batch["mask"].sum())
            loss_sum += float(loss.detach()) * size
            count += size
            report(progress, phase=phase, epoch=epoch, epochs=epochs, step=step, steps=len(loader),
                   loss=loss_sum / count, message=f"{phase} · эпоха {epoch}/{epochs}")
        train_eval = _evaluate(model, eval_loader, device, weights, cancel)
        validation_loss = (_evaluate(model, validation_loader, device, weights, cancel)
                           if validation_loader is not None else None)
        metrics = {"phase": phase, "epoch": epoch, "loss": loss_sum / max(1, count),
                   "train_eval_loss": train_eval, "validation_loss": validation_loss}
        history.append(metrics)
        report(progress, **metrics, epochs=epochs, step=len(loader), steps=len(loader),
               message=f"Эпоха {epoch}/{epochs} завершена")
    return history


def split_by_track(examples: list[TrainingExample], fraction: float = 0.2, seed: int = 1234):
    """Never put windows/layouts from the same audio into both train and val."""
    if not math.isfinite(fraction) or not 0 <= fraction < 0.5:
        raise ValueError("validation_fraction: от 0 до 0.5, не включая 0.5.")
    groups = sorted({example.group_id for example in examples})
    if len(groups) < 3 or fraction == 0:
        return examples, []
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    held_out = set(groups[:max(1, round(len(groups) * fraction))])
    return ([e for e in examples if e.group_id not in held_out],
            [e for e in examples if e.group_id in held_out])


def save_checkpoint(model: RhythmCRNN, path: str | Path, metadata: dict, *, cancel=None) -> Path:
    """Atomic weight publication; only tensors and primitive metadata are saved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint_version": CHECKPOINT_VERSION, "feature_version": FEATURE_VERSION,
               "feature_config": asdict(FEATURE_CONFIG), "model_config": model.config_dict(),
               "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "metadata": metadata}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = file.name
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        check_cancel(cancel)
        os.replace(temporary, path)
        write_json(metadata_path(path), metadata)  # small sidecar for UI without loading Torch
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_checkpoint(path: str | Path | None = None, *, device="auto") -> tuple[RhythmCRNN, dict]:
    # Read-side two-step lookup: user model under ROOT first, bundled default second.
    path = read_checkpoint(path)
    if not path.is_file():
        raise ValueError(f"Модель не найдена: {path}. Сначала выполните Train / --train.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Неизвестная версия чекпоинта.")
        if (payload.get("feature_version") != FEATURE_VERSION
                or payload.get("feature_config") != asdict(FEATURE_CONFIG)):
            raise ValueError("Чекпоинт использует несовместимые аудиопризнаки.")
        config = ModelConfig(**payload["model_config"])
        config.validate()
        model = RhythmCRNN(config)
        model.load_state_dict(payload["model_state"], strict=True)
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Нет метаданных модели.")
    except Exception as error:
        raise ValueError(f"Не удалось безопасно загрузить модель {path.name}: {error}") from error
    model.to(select_device(str(device)))
    model.eval()
    return model, metadata


@dataclass
class TrainingResult:
    checkpoint: Path
    metadata: dict
    seconds: float

    def to_dict(self):
        return {"checkpoint": str(self.checkpoint), "seconds": round(self.seconds, 3), "metadata": self.metadata}


def train_model(*, charts_dir: str | Path = CHART_DIR, checkpoint_path: str | Path = MODEL_PATH,
                epochs: int = 8, bootstrap_epochs: int = 3, bootstrap_samples: int = 12,
                batch_size: int = 4, learning_rate: float = 0.001, window_seconds: float = 10.0,
                validation_fraction: float = 0.0, device="auto", reset: bool = False,
                seed: int = 1234, cache_dir=FEATURE_CACHE, model_config: ModelConfig | None = None,
                progress=None, cancel=None) -> TrainingResult:
    started = time.perf_counter()
    if any(not isinstance(n, int) or not 0 <= n <= 500 for n in (epochs, bootstrap_epochs)):
        raise ValueError("epochs и bootstrap_epochs: от 0 до 500.")
    check_cancel(cancel)
    selected = select_device(str(device))
    report(progress, phase="device", message=f"PyTorch {torch.__version__} · {selected}", device=str(selected))
    torch.manual_seed(seed)
    path = Path(checkpoint_path)
    previous = {}
    new_model = reset or not path.is_file()
    if new_model:
        model = RhythmCRNN(model_config)
    else:
        model, previous = load_checkpoint(path, device=str(selected))
        if model_config is not None and model.config != model_config:
            raise ValueError("Параметры модели изменились. Используйте reset=True / --reset-model.")
    examples, skipped = load_examples(charts_dir, cache_dir=cache_dir, progress=progress, cancel=cancel)
    train, validation = split_by_track(examples, validation_fraction, seed)
    history = []
    bootstrap_metadata = previous.get("bootstrap", {})
    if new_model and bootstrap_epochs:
        bootstrap = make_bootstrap_examples(bootstrap_samples, seed=seed, cache_dir=cache_dir,
                                             progress=progress, cancel=cancel)
        history += fit_model(model, bootstrap, epochs=bootstrap_epochs, device=str(selected),
                             batch_size=batch_size, learning_rate=learning_rate, window_seconds=window_seconds,
                             phase="bootstrap", seed=seed, progress=progress, cancel=cancel)
        bootstrap_metadata = {"examples": len(bootstrap), "epochs": bootstrap_epochs, "seed": seed,
                              "lane_counts": sorted({e.num_lanes for e in bootstrap}), "external_pretrained": False}
    personal = bool(train) and epochs > 0
    if personal:
        history += fit_model(model, train, epochs=epochs, device=str(selected), batch_size=batch_size,
                             learning_rate=learning_rate * (0.5 if not new_model else 1.0),
                             window_seconds=window_seconds, validation_examples=validation, phase="finetune",
                             seed=seed, progress=progress, cancel=cancel)
    if not history:
        reasons = "; ".join(f"{Path(item['path']).name}: {item['reason']}" for item in skipped[:3])
        raise ValueError("Нет данных/эпох для обучения. Добавьте Live-карты; для cold start используйте "
                         "bootstrap_epochs > 0 (при существующей модели — --reset-model)."
                         + (f" Пропущено: {reasons}" if reasons else ""))
    check_cancel(cancel)
    personalized = bool(previous.get("personalized")) or personal
    metadata = {"checkpoint_version": CHECKPOINT_VERSION, "feature_version": FEATURE_VERSION,
                "checkpoint_id": uuid.uuid4().hex, "created_utc": datetime.now(timezone.utc).isoformat(),
                "architecture": "lane-conditioned temporal CNN + BiGRU; onset/hold heads",
                "device": str(selected), "torch_version": str(torch.__version__), "frame_rate": FRAME_RATE,
                "gaussian_sigma_frames": 1.5, "max_lanes": MAX_LANES,
                "personalized": personalized, "origin": "personalized" if personalized else "bootstrap_only",
                "training_runs": int(previous.get("training_runs", 0)) + 1,
                "current_dataset_maps": len(examples),
                "user_charts": [e.provenance for e in train] if personal else previous.get("user_charts", []),
                "user_lane_counts": sorted({e.num_lanes for e in train}) if personal else previous.get("user_lane_counts", []),
                "bootstrap": bootstrap_metadata, "skipped_count": len(skipped), "skipped_charts": skipped[:100],
                "validation": {"available": bool(validation) and personal, "grouped_by": "audio track id",
                               "fraction_requested": validation_fraction,
                               "track_ids": sorted({e.group_id for e in validation}) if personal else [],
                               "warning": "By default all personal maps train. Opt-in holdout requires >=3 unique tracks. "
                                          "Training loss alone does not measure style transfer."},
                "last_run": {"seed": seed, "epochs": epochs if personal else 0,
                             "bootstrap_epochs": bootstrap_epochs if new_model else 0,
                             "batch_size": batch_size,
                             "learning_rate": learning_rate * (0.5 if personal and not new_model else 1.0),
                             "bootstrap_learning_rate": learning_rate,
                             "window_seconds": window_seconds, "optimizer": "fresh AdamW per phase/run"},
                "history": (previous.get("history", []) + history)[-1000:]}
    report(progress, phase="saving", message="Атомарное сохранение модели", loss=history[-1]["train_eval_loss"])
    save_checkpoint(model, path, metadata, cancel=cancel)
    elapsed = time.perf_counter() - started
    report(progress, phase="done", message="Модель сохранена", seconds=elapsed, checkpoint=str(path),
           loss=history[-1]["train_eval_loss"], dataset_count=len(examples))
    return TrainingResult(path, metadata, elapsed)
