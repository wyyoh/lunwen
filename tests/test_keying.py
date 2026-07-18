from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from keyed_gram.keying import (
    apply_key,
    generate_key,
    hash_auxiliary,
    hash_non_auxiliary,
    partial_key,
    validate_key,
    validate_key_labels,
    verify_restoration,
)
from keyed_gram.model import GramTransformer


def test_key_is_self_inverse_and_core_is_untouched(tiny_config):
    torch.manual_seed(0)
    state = GramTransformer(tiny_config).state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    assert len(key.swaps) == 8
    core_hash = hash_non_auxiliary(state)
    original_aux_hash = hash_auxiliary(state)
    locked = apply_key(state, key)
    assert hash_non_auxiliary(locked) == core_hash
    assert hash_auxiliary(locked) != original_aux_hash
    restored = apply_key(locked, key)
    assert verify_restoration(restored, key)
    assert all(torch.equal(state[name], restored[name]) for name in state)


def test_wrong_key_does_not_restore(tiny_config):
    state = GramTransformer(tiny_config).state_dict()
    correct = generate_key(state, "private", group_size=4, seed=42)
    wrong = generate_key(state, "private", group_size=4, seed=1000)
    locked = apply_key(state, correct)
    attacked = apply_key(locked, wrong)
    assert not verify_restoration(attacked, correct)


def test_partial_key_counts(tiny_config):
    state = GramTransformer(tiny_config).state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    assert len(partial_key(key, 0.5, 2000).swaps) == 4
    empty = partial_key(key, 0.0, 2000)
    assert not empty.swaps
    assert all(torch.equal(state[name], apply_key(state, empty)[name]) for name in state)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda key: replace(key, expert_index=0), "auxiliary"),
        (
            lambda key: replace(key, swaps=(key.swaps[0], key.swaps[0]) + key.swaps[2:]),
            "more than one",
        ),
        (
            lambda key: replace(
                key,
                swaps=(((0, 0), (0, 1)),) + key.swaps[1:],
            ),
            "cross layers",
        ),
        (
            lambda key: replace(
                key,
                swaps=(((0, 999), key.swaps[0][1]),) + key.swaps[1:],
            ),
            "out of bounds",
        ),
        (lambda key: replace(key, group_size=3), "divide"),
    ],
)
def test_invalid_keys_are_rejected(tiny_config, mutator, match):
    state = GramTransformer(tiny_config).state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    with pytest.raises(ValueError, match=match):
        validate_key(mutator(key), state)


def test_checkpoint_width_and_core_tampering_are_detectable(tiny_config):
    state = GramTransformer(tiny_config).state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    malformed = {name: value.clone() for name, value in state.items()}
    malformed["blocks.0.moe.experts.1.c_fc.weight"] = malformed[
        "blocks.0.moe.experts.1.c_fc.weight"
    ][:-1]
    with pytest.raises(ValueError, match="same width|width mismatch"):
        validate_key(key, malformed)

    locked = apply_key(state, key)
    locked["embed.weight"][0, 0].add_(1)
    assert hash_non_auxiliary(locked) != hash_non_auxiliary(state)


def test_key_capability_must_match_checkpoint_label(tiny_config):
    state = GramTransformer(tiny_config).state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    validate_key_labels(key, ["core", "private"])
    with pytest.raises(ValueError, match="capability label"):
        validate_key_labels(key, ["core", "different-private-label"])
