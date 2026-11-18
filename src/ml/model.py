"""Small lane-conditioned temporal CNN + bidirectional GRU, with two heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import MAX_LANES, require_torch
from .features import FEATURE_DIM

torch = require_torch()
nn = torch.nn


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = FEATURE_DIM
    channels: int = 48
    hidden_size: int = 48
    condition_dim: int = 12
    max_lanes: int = MAX_LANES
    dropout: float = 0.1

    def validate(self) -> None:
        if self.feature_dim != FEATURE_DIM or self.max_lanes != MAX_LANES:
            raise ValueError("Модель несовместима с feature schema / MAX_LANES.")
        if any(not isinstance(x, int) or not 4 <= x <= 512
               for x in (self.channels, self.hidden_size, self.condition_dim)):
            raise ValueError("Некорректные размеры модели.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Некорректный dropout.")


def select_device(preference: str = "auto") -> torch.device:
    """Availability is checked, not inferred from the OS. No GPU is required."""
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "auto":
        return torch.device("cuda" if cuda else "mps" if mps else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not cuda:
        raise ValueError("CUDA недоступна в этой установке PyTorch/на этом устройстве.")
    if device.type == "mps" and not mps:
        raise ValueError("Apple MPS недоступен на этом устройстве.")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Устройство ML: auto, cuda, mps или cpu.")
    return device


class RhythmCRNN(nn.Module):
    """Inputs [batch, frames, 130], lanes [batch]; logits [batch, frames, 8].

    Frequency bands are Conv1d input channels; temporal kernels learn attacks
    and local timbres. A lane-count embedding is concatenated before the BiGRU.
    The bidirectional layer makes this an OFFLINE generator, not streaming AI.
    """
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        c.validate()
        self.cnn = nn.Sequential(nn.Conv1d(c.feature_dim, c.channels, 5, padding=2), nn.SiLU(),
                                 nn.Conv1d(c.channels, c.channels, 3, padding=2, dilation=2), nn.SiLU())
        self.norm = nn.LayerNorm(c.channels)
        self.dropout = nn.Dropout(c.dropout)
        self.lane_embedding = nn.Embedding(c.max_lanes + 1, c.condition_dim)
        self.context = nn.GRU(c.channels + c.condition_dim, c.hidden_size,
                              batch_first=True, bidirectional=True)
        self.onset_head = nn.Linear(c.hidden_size * 2, c.max_lanes)
        self.hold_head = nn.Linear(c.hidden_size * 2, c.max_lanes)

    def forward(self, features, num_lanes, lengths=None) -> dict:
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim or features.shape[1] < 1:
            raise ValueError("Ожидаются признаки [batch, frames, 130].")
        batch, frames, _ = features.shape
        lanes = torch.as_tensor(num_lanes, dtype=torch.long, device=features.device).reshape(-1)
        if lanes.numel() == 1:
            lanes = lanes.expand(batch)
        if lanes.numel() != batch or bool(((lanes < 1) | (lanes > self.config.max_lanes)).any()):
            raise ValueError("num_lanes должен быть от 1 до 8 для каждого элемента batch.")
        encoded = self.cnn(features.transpose(1, 2)).transpose(1, 2)
        encoded = self.dropout(self.norm(encoded))
        condition = self.lane_embedding(lanes).unsqueeze(1).expand(-1, frames, -1)
        sequence = torch.cat((encoded, condition), dim=-1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths.detach().cpu(),
                                                       batch_first=True, enforce_sorted=False)
            sequence, _ = self.context(packed)
            sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=frames)
        else:
            sequence, _ = self.context(sequence)
        return {"onset_logits": self.onset_head(sequence), "hold_logits": self.hold_head(sequence)}

    def config_dict(self) -> dict:
        return asdict(self.config)
