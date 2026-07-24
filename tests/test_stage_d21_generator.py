from __future__ import annotations

import torch
import pytest

from keyed_gram.stage_d2_contract import PublicSyntheticValue
from keyed_gram.stage_d21_contract import (
    D21Error,
    GenerationVariant,
    create_generation_payload,
)
from keyed_gram.stage_d21_generator import (
    DeterministicRenderer,
    FrozenMockCopyProbe,
    G0_TEMPLATE,
    GenerationRegistry,
    model_parameter_sha256,
)


def _payload(variant: GenerationVariant):
    return create_generation_payload(
        value=PublicSyntheticValue(b"0123456789ABCDEF"),
        public_instruction="Return only the current value.",
        variant=variant,
        context_id="d21-context-" + "a" * 32,
    )


def test_g0_renders_only_fixed_template() -> None:
    output = DeterministicRenderer().generate(
        _payload(GenerationVariant.G0)
    )
    assert output.text == G0_TEMPLATE.format(
        authorized_value="0123456789ABCDEF"
    )


def test_mock_g1_copies_only_current_value() -> None:
    output = FrozenMockCopyProbe().generate(
        _payload(GenerationVariant.G1)
    )
    assert output.text == "0123456789ABCDEF"


def test_generation_registry_counts_and_destroys_context() -> None:
    registry = GenerationRegistry(
        DeterministicRenderer(),
        FrozenMockCopyProbe(),
    )
    before = registry.snapshot()
    registry.generate(_payload(GenerationVariant.G1))
    after = registry.snapshot()
    delta = after.delta(before)
    assert delta["generator_invocation_delta"] == 1
    assert delta["g1_invocation_delta"] == 1
    assert after.active_context_count == 0


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("model_timeout", "generator_timeout"),
        ("model_exception", "generator_failure"),
        ("renderer_exception", "generator_failure"),
    ],
)
def test_generation_failures_are_fail_closed(mode: str, reason: str) -> None:
    registry = GenerationRegistry(
        DeterministicRenderer(),
        FrozenMockCopyProbe(),
    )
    with pytest.raises(D21Error) as captured:
        registry.generate(
            _payload(GenerationVariant.G1),
            failure_mode=mode,
        )
    assert captured.value.reason.value == reason
    snapshot = registry.snapshot()
    assert snapshot.failure_count == 1
    assert snapshot.active_context_count == 0


def test_parameter_hash_changes_when_parameter_changes() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    before = model_parameter_sha256(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert model_parameter_sha256(model) != before
