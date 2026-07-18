from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import GramModelConfig
from .model import GramTransformer


CHECKPOINT_FORMAT_VERSION = 1


def save_clean_checkpoint(
    path: str | Path,
    model: GramTransformer | None,
    model_config: GramModelConfig,
    labels: list[str],
    *,
    state_dict: Mapping[str, Tensor] | None = None,
) -> Path:
    if (model is None) == (state_dict is None):
        raise ValueError("provide exactly one of model or state_dict")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if state_dict is None:
        assert model is not None
        state_dict = model.state_dict()
    clean_state = {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": clean_state,
        "model_config": model_config.__dict__.copy(),
        "labels": list(labels),
    }
    torch.save(payload, target)
    return target


def load_clean_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
        raise ValueError(f"not a clean Keyed-GRAM checkpoint: {source}")
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if any(key in payload for key in ("optimizer", "opts", "scheduler")):
        raise ValueError("deployment checkpoint must not include optimizer state")
    return payload


def build_model_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[GramTransformer, dict[str, Any]]:
    payload = load_clean_checkpoint(path, map_location="cpu")
    config = GramModelConfig(**payload["model_config"])
    model = GramTransformer(config)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device=device, dtype=dtype)
    return model, payload
