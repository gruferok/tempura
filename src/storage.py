"""Small validated configuration and crash-safe JSON persistence helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import sys

_log = logging.getLogger(__name__)


def _writable_dir(path: Path) -> bool:
    """Real write probe (os.access lies for Program Files / UAC-virtualised folders)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".tempura_write_"):
            pass
        return True
    except OSError:
        return False


def _resolve_roots() -> tuple[Path, Path]:
    """Return (ROOT for user data, BUNDLE_ROOT for read-only defaults). TEMPURA_HOME overrides ROOT."""
    override = os.environ.get("TEMPURA_HOME")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        bundle = Path(getattr(sys, "_MEIPASS", exe_dir))
        if override:
            return Path(override).expanduser().resolve(), bundle
        if _writable_dir(exe_dir):
            return exe_dir, bundle
        local = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Tempura"
        if _writable_dir(local):
            return local, bundle
        fallback = Path(tempfile.gettempdir()) / "Tempura"
        _log.warning("Tempura: no writable data folder; using %s", fallback)
        return fallback, bundle
    dev = Path(__file__).resolve().parents[1]
    return (Path(override).expanduser().resolve() if override else dev), dev


ROOT, BUNDLE_ROOT = _resolve_roots()
CONFIG_PATH = ROOT / "config.json"
CHART_DIR = ROOT / "data" / "charts"


def seed_from_bundle() -> None:
    """Copy missing defaults (config.json, data/charts) from BUNDLE_ROOT into ROOT. Never overwrites."""
    import shutil
    if BUNDLE_ROOT.resolve() == ROOT.resolve():
        return
    bundle_cfg = BUNDLE_ROOT / "config.json"
    if bundle_cfg.is_file() and not CONFIG_PATH.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_cfg, CONFIG_PATH)
    # Models are not copied: they can be large and must not be duplicated; the app reads them from ROOT only.
    for rel_dir in ("data/charts",):
        src_dir = BUNDLE_ROOT / rel_dir
        if not src_dir.is_dir():
            continue
        for src in src_dir.rglob("*"):
            if src.is_file():
                target = ROOT / rel_dir / src.relative_to(src_dir)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)


def resolve_resource(rel: str | Path) -> Path:
    """Two-step lookup for non-model resources (charts, audio): ROOT first, then BUNDLE_ROOT.

    Returns the first existing match; when neither exists it returns ROOT / rel,
    the only location user code may create or write files.
    """
    rel = Path(rel)
    for base in (ROOT, BUNDLE_ROOT):
        candidate = base / rel
        if candidate.exists():
            return candidate
    return ROOT / rel


def write_json(path: str | Path, value: Any) -> Path:
    """Write beside the destination, then atomically replace it (never partial JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary = file.name
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON должен содержать объект, а не список или число.")
    return data


@dataclass
class Config:
    schema_version: int = 1
    audio_offset_ms: float = 0.0
    visual_offset_ms: float = 0.0
    calibrated: bool = False
    sample_rate: int = 48000
    output_device: int | str | None = None
    latency: str | float = "low"
    blocksize: int = 0
    volume: float = 0.65
    fps: int = 300
    scroll_seconds: float = 2.0
    orientation: str = "vertical"
    hold_threshold_ms: float = 150.0
    calibration: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Неизвестная версия config.json.")
        for name in ("audio_offset_ms", "visual_offset_ms", "volume", "scroll_seconds",
                     "hold_threshold_ms"):
            if not isinstance(getattr(self, name), (int, float)) or not math.isfinite(getattr(self, name)):
                raise ValueError(f"Некорректное значение {name}.")
        if not -500 <= self.audio_offset_ms <= 500 or not -500 <= self.visual_offset_ms <= 500:
            raise ValueError("Смещения должны быть в пределах ±500 ms.")
        if not 0 <= self.volume <= 1 or not 0.5 <= self.scroll_seconds <= 6:
            raise ValueError("Некорректная громкость или скорость прокрутки.")
        if self.orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation: vertical или horizontal.")
        if self.sample_rate not in (22050, 24000, 44100, 48000, 88200, 96000):
            raise ValueError("Неподдерживаемый sample_rate в config.json.")
        if not isinstance(self.blocksize, int) or not 0 <= self.blocksize <= 8192:
            raise ValueError("blocksize должен быть целым числом от 0 до 8192.")
        if not isinstance(self.fps, int) or not 240 <= self.fps <= 1000:
            raise ValueError("fps должен быть от 240 до 1000.")
        if not 50 <= self.hold_threshold_ms <= 1000:
            raise ValueError("hold_threshold_ms должен быть от 50 до 1000.")
        if isinstance(self.latency, str):
            if self.latency not in ("low", "high"):
                raise ValueError("latency: low, high или секунды.")
        elif not isinstance(self.latency, (int, float)) or not 0 < self.latency <= 1:
            raise ValueError("Некорректное значение latency.")
        if not isinstance(self.calibration, dict):
            raise ValueError("calibration должна быть объектом.")

    @classmethod
    def load(cls, path: str | Path = CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = read_json(path)
        config = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        config.validate()
        return config

    def save(self, path: str | Path = CONFIG_PATH) -> Path:
        self.validate()
        return write_json(path, asdict(self))

    def input_time(self, raw_song_time: float) -> float:
        """Positive audio offset means the player taps late: subtract it once."""
        return raw_song_time - self.audio_offset_ms / 1000.0

    def display_time(self, raw_song_time: float) -> float:
        """Positive visual offset advances the chart, not the audio or judgment."""
        return raw_song_time + self.visual_offset_ms / 1000.0
