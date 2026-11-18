"""Versioned, lane-aware chart format. All times are seconds on the raw track."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import re

from .storage import CHART_DIR, read_json, write_json

DEFAULT_LAYOUTS = {1: ["space"], 2: ["f", "j"], 4: ["d", "f", "j", "k"]}
RESERVED_KEYS = {"escape", "return", "enter", "p", "r", "s", "backspace", "tab", "f1", "f2"}


def validate_keys(keys: list[str]) -> None:
    if not 1 <= len(keys) <= 8 or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("Нужно от 1 до 8 клавиш.")
    if len(set(keys)) != len(keys):
        raise ValueError("Клавиши дорожек должны быть разными.")
    if RESERVED_KEYS.intersection(keys):
        raise ValueError("Esc, Enter, P, R, S, Backspace и Tab зарезервированы.")


@dataclass(frozen=True)
class Note:
    time: float
    lane: int
    duration: float = 0.0
    type: str = "Tap"

    @property
    def end(self) -> float:
        return self.time + self.duration


@dataclass
class Chart:
    track: dict
    keys: list[str]
    notes: list[Note]
    metadata: dict = field(default_factory=dict)
    schema_version: int = 1
    time_unit: str = "seconds"

    @property
    def lanes(self) -> int:
        return len(self.keys)

    def validate(self) -> None:
        if self.schema_version != 1 or self.time_unit != "seconds":
            raise ValueError("Поддерживается chart schema_version=1, time_unit=seconds.")
        validate_keys(self.keys)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata карты должна быть объектом.")
        if "include_in_training" in self.metadata and type(self.metadata["include_in_training"]) is not bool:
            raise ValueError("metadata.include_in_training должен быть boolean.")
        if not isinstance(self.track, dict) or self.track.get("kind") not in ("file", "synthetic"):
            raise ValueError("В карте нет корректного описания трека.")
        if not isinstance(self.track.get("id"), str) or not self.track["id"]:
            raise ValueError("В карте нет идентификатора аудиофайла.")
        duration = self.track.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Некорректная длительность трека.")
        previous_end = [-1.0] * self.lanes
        for note in sorted(self.notes, key=lambda n: (n.time, n.lane)):
            if not isinstance(note.lane, int) or isinstance(note.lane, bool) or not 0 <= note.lane < self.lanes:
                raise ValueError("Номер дорожки за пределами карты.")
            if not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0
                       for t in (note.time, note.duration)):
                raise ValueError("Времена нот должны быть конечными неотрицательными числами.")
            if note.type not in ("Tap", "Hold"):
                raise ValueError("Тип ноты: Tap или Hold.")
            if (note.type == "Tap" and note.duration != 0) or (note.type == "Hold" and note.duration <= 0):
                raise ValueError("Tap не имеет длительности; Hold должен иметь длительность.")
            if note.end > duration + 1e-6:
                raise ValueError("Нота выходит за конец аудио.")
            if note.time < previous_end[note.lane] - 1e-6:
                raise ValueError("Перекрывающиеся ноты на одной дорожке.")
            previous_end[note.lane] = note.end

    def to_dict(self) -> dict:
        self.validate()
        return {"schema_version": self.schema_version, "time_unit": self.time_unit,
                "track": self.track, "lanes": self.lanes, "keys": self.keys,
                "notes": [asdict(n) for n in sorted(self.notes, key=lambda n: (n.time, n.lane))],
                "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Chart":
        try:
            chart = cls(track=data["track"], keys=data["keys"],
                        notes=[Note(**n) for n in data["notes"]],
                        metadata=data.get("metadata", {}),
                        schema_version=data["schema_version"], time_unit=data["time_unit"])
            if data["lanes"] != chart.lanes:
                raise ValueError("lanes не совпадает с количеством клавиш.")
            chart.validate()
            return chart
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"Некорректный формат карты: {error}") from error

    @classmethod
    def load(cls, path: str | Path) -> "Chart":
        return cls.from_dict(read_json(path))

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            name = re.sub(r"[^\w.-]+", "_", str(self.track.get("name", "track")), flags=re.UNICODE)
            name = name.strip("._")[:100] or "track"
            path = CHART_DIR / f"{name}_{self.lanes}K.json"
        return write_json(path, self.to_dict())


def recording_metadata(audio_offset_ms: float, underruns: int, timing_source: str) -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "audio_offset_ms_used": audio_offset_ms, "timestamps_are_corrected": True,
            "timing_source": timing_source, "output_underruns": underruns,
            "note": "Visual offset is not baked into chart times."}
