from __future__ import annotations

from dataclasses import replace

import torch

from keyed_gram.keying import (
    apply_key,
    generate_key,
    hash_auxiliary,
    hash_non_auxiliary,
)
from keyed_gram.model import GramTransformer


def test_forward_shape_and_loss(tiny_config):
    model = GramTransformer(tiny_config)
    x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.context_length))
    y = torch.randint(0, tiny_config.vocab_size, x.shape)
    logits, loss = model(
        x,
        labels=y,
        fwd_mask=torch.tensor([1, 1]),
        bck_mask=torch.tensor([1, 1]),
    )
    assert logits.shape == (2, tiny_config.context_length, tiny_config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_locked_core_only_logits_are_exact(tiny_config):
    torch.manual_seed(1)
    original = GramTransformer(tiny_config)
    state = original.state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    locked_state = apply_key(state, key)
    locked = GramTransformer(tiny_config)
    locked.load_state_dict(locked_state)
    x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.context_length))
    mask = torch.tensor([1, 0])
    original.eval()
    locked.eval()
    with torch.inference_mode():
        left = original(x, fwd_mask=mask, bck_mask=mask)[0]
        right = locked(x, fwd_mask=mask, bck_mask=mask)[0]
    assert torch.equal(left, right)


def test_keyed_parameter_view_matches_restored_model_without_mutation(tiny_config):
    torch.manual_seed(2)
    original = GramTransformer(tiny_config)
    state = original.state_dict()
    key = generate_key(state, "private", group_size=4, seed=42)
    locked_state = apply_key(state, key)
    keyed = GramTransformer(tiny_config)
    keyed.load_state_dict(locked_state)
    before = hash_auxiliary(keyed.state_dict())
    x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.context_length))
    mask = torch.tensor([1, 1])
    original.eval()
    keyed.eval()
    with torch.inference_mode():
        expected = original(x, fwd_mask=mask, bck_mask=mask)[0]
        actual = keyed(x, fwd_mask=mask, bck_mask=mask, key=key)[0]
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    assert hash_auxiliary(keyed.state_dict()) == before


def test_zero_scaled_residual_is_exactly_core_only_and_diagnostic(tiny_config):
    config = replace(
        tiny_config,
        learnable_aux_scale=True,
        aux_scale_init=0.0,
        aux_input_stop_gradient=True,
    )
    model = GramTransformer(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_length))
    full = torch.tensor([1, 1])
    core = torch.tensor([1, 0])
    model.eval()
    with torch.inference_mode():
        full_logits, _, diagnostics = model(
            x, fwd_mask=full, bck_mask=full, return_diagnostics=True
        )
        core_logits = model(x, fwd_mask=core, bck_mask=core)[0]
    assert torch.equal(full_logits, core_logits)
    assert len(diagnostics["aux_residuals"]) == config.num_layers
    assert all(torch.count_nonzero(value) == 0 for value in diagnostics["aux_residuals"])


def test_aux_scale_is_trainable_but_excluded_from_core_hash(tiny_config):
    config = replace(
        tiny_config,
        learnable_aux_scale=True,
        aux_scale_init=0.01,
        aux_input_stop_gradient=True,
    )
    model = GramTransformer(config)
    core_hash = hash_non_auxiliary(model.state_dict())
    aux_hash = hash_auxiliary(model.state_dict())
    x = torch.randint(0, config.vocab_size, (2, config.context_length))
    y = torch.randint(0, config.vocab_size, x.shape)
    _, loss, diagnostics = model(
        x,
        labels=y,
        fwd_mask=torch.tensor([1, 1]),
        bck_mask=torch.tensor([0, 1]),
        return_diagnostics=True,
    )
    assert loss is not None
    loss.backward()
    assert model.aux_scales is not None and model.aux_scales.grad is not None
    assert any(value.requires_grad for value in diagnostics["aux_residuals"])
    with torch.no_grad():
        model.aux_scales.add_(0.1)
    assert hash_non_auxiliary(model.state_dict()) == core_hash
    assert hash_auxiliary(model.state_dict()) != aux_hash
