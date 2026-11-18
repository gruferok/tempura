"""Playable default layouts; custom key order is preserved in generated charts."""
from __future__ import annotations

from ..chart import DEFAULT_LAYOUTS, validate_keys
from . import MAX_LANES

_EXTRA = {3: ["d", "f", "j"], 5: ["d", "f", "g", "j", "k"],
          6: ["d", "f", "g", "h", "j", "k"], 7: ["a", "d", "f", "g", "j", "k", "l"],
          8: ["a", "d", "f", "g", "h", "j", "k", "l"]}


def keys_for_lanes(num_lanes: int, keys: list[str] | None = None) -> list[str]:
    if not isinstance(num_lanes, int) or isinstance(num_lanes, bool) or not 1 <= num_lanes <= MAX_LANES:
        raise ValueError("num_lanes: целое число от 1 до 8.")
    chosen = list(keys) if keys is not None else list((DEFAULT_LAYOUTS | _EXTRA)[num_lanes])
    validate_keys(chosen)
    if len(chosen) != num_lanes:
        raise ValueError("Количество клавиш должно совпадать с num_lanes.")
    return chosen
