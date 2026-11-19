"""Offline chart learning. Importing this package never opens audio or loads Torch.

The game remains usable without ML dependencies. Training/inference must not run
in the audio callback; AI Studio runs one cancellable background job at a time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..storage import ROOT

MAX_LANES = 8
FRAME_RATE = 100
FEATURE_VERSION = 1
CHECKPOINT_VERSION = 1
MODEL_DIR = ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "rhythm_model.pt"
FEATURE_CACHE = ROOT / ".cache" / "ml" / "features"
ProgressCallback = Callable[[dict], None]


class Cancelled(RuntimeError):
    """A user cancelled an offline job; the previous checkpoint stays intact."""


def check_cancel(cancel=None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Операция отменена. Предыдущая модель не изменена.")


def report(progress: ProgressCallback | None = None, **event) -> None:
    if progress is not None:
        progress(event)


def require_torch():
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("Для AI Studio нужен PyTorch. Установите requirements-ml.txt; "
                           "CPU: python -m pip install torch==2.6.0 "
                           "--index-url https://download.pytorch.org/whl/cpu. "
                           f"Подробности: {error}") from error
    return torch


def metadata_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).with_suffix(".json")
