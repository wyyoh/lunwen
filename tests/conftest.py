from __future__ import annotations

import pytest

from keyed_gram.config import GramModelConfig


@pytest.fixture
def tiny_config() -> GramModelConfig:
    return GramModelConfig(
        vocab_size=64,
        context_length=16,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        hidden_size=16,
        core_ffn_width=32,
        aux_ffn_width=16,
        num_aux=1,
        eos_token_id=1,
    )
