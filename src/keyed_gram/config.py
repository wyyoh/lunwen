from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GramModelConfig:
    vocab_size: int = 4096
    context_length: int = 256
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    hidden_size: int = 512
    core_ffn_width: int = 2048
    aux_ffn_width: int = 192
    num_aux: int = 1
    eos_token_id: int = 1
    learnable_aux_scale: bool = False
    aux_scale_init: float = 1.0
    aux_input_stop_gradient: bool = False

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 4:
            raise ValueError("attention head dimension must be divisible by four")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_layers < 2:
            raise ValueError("at least two layers are required")
        if self.num_aux < 1:
            raise ValueError("at least one auxiliary expert is required")
        for name in ("vocab_size", "context_length", "core_ffn_width", "aux_ffn_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.aux_scale_init):
            raise ValueError("aux_scale_init must be finite")


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data/stories"
    private_label: str = "alien-encounters"
    train_fraction: float = 0.1
    eval_sequences_per_core_label: int = 200
    max_private_eval_sequences: int = 10000

    def validate(self) -> None:
        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        if self.eval_sequences_per_core_label <= 0:
            raise ValueError("eval_sequences_per_core_label must be positive")
        if self.max_private_eval_sequences <= 0:
            raise ValueError("max_private_eval_sequences must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    epochs: int = 1
    max_steps: int = 0
    micro_batch_size: int = 8
    accumulation_steps: int = 16
    learning_rate: float = 5e-3
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    decay_fraction: float = 0.1
    aux_route_probability: float = 0.3
    core_robust_probability: float = 0.5
    dtype: str = "bfloat16"
    compile: bool = False
    checkpoint_every: int = 250

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.micro_batch_size <= 0 or self.accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name in ("warmup_fraction", "decay_fraction", "aux_route_probability", "core_robust_probability"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.warmup_fraction + self.decay_fraction > 1:
            raise ValueError("warmup_fraction + decay_fraction must not exceed 1")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "main"
    model: GramModelConfig = field(default_factory=GramModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        config = cls(
            name=value.get("name", "main"),
            model=GramModelConfig(**value.get("model", {})),
            data=DataConfig(**value.get("data", {})),
            training=TrainingConfig(**value.get("training", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)
