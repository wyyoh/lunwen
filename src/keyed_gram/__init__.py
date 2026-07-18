"""Keyed-GRAM research prototype."""

from .config import ExperimentConfig, GramModelConfig
from .keying import AuxPermutationKey, apply_key, generate_key, hash_auxiliary
from .model import GramTransformer

__all__ = [
    "AuxPermutationKey",
    "ExperimentConfig",
    "GramModelConfig",
    "GramTransformer",
    "apply_key",
    "generate_key",
    "hash_auxiliary",
]
