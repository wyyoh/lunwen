from __future__ import annotations

from keyed_gram.tool_effects.contracts import check_effect_completeness
from keyed_gram.tool_effects.types import (
    DeclaredContract,
    EffectKind,
    EffectTemplate,
    Environment,
    ToolCall,
    ToolImplementation,
    ToolTransition,
    TraceScenario,
    TraceStep,
)


def _fixture(*, implementation_effect: EffectTemplate, contract_effect: EffectTemplate):
    call = ToolCall.create(
        "call-1",
        "tool-a",
        "v1",
        {
            "tenant": "tenant-a",
            "resource": "resource-a",
            "destination": "external-a",
        },
    )
    implementation = ToolImplementation(
        "tool-a",
        "v1",
        (ToolTransition("run", (), (implementation_effect,)),),
    )
    contract = DeclaredContract(
        "contract-a",
        "tool-a",
        "v1",
        frozenset({contract_effect}),
    )
    scenario = TraceScenario("trace-a", (TraceStep(call, Environment()),), False)
    return contract, {("tool-a", "v1"): implementation}, (scenario,)


def _effect(*, roles=("tenant", "resource")) -> EffectTemplate:
    return EffectTemplate(
        EffectKind.SEND,
        "arg:tenant",
        "arg:resource",
        "arg:destination",
        security_roles=roles,
    )


def test_exact_contract_is_effect_complete() -> None:
    check = check_effect_completeness(
        *_fixture(implementation_effect=_effect(), contract_effect=_effect())
    )
    assert check.effect_complete is True
    assert check.contract_effect_recall == 1.0
    assert check.contract_overapproximation_ratio == 0.0


def test_missing_argument_role_is_not_effect_complete() -> None:
    check = check_effect_completeness(
        *_fixture(
            implementation_effect=_effect(roles=("tenant", "resource", "destination")),
            contract_effect=_effect(roles=("tenant", "resource")),
        )
    )
    assert check.effect_complete is False
    assert check.contract_effect_recall == 0.0


def test_hidden_effect_is_reported_uncovered() -> None:
    read = EffectTemplate(
        EffectKind.READ,
        "arg:tenant",
        "arg:resource",
        security_roles=("tenant", "resource"),
    )
    call = ToolCall.create(
        "call-1",
        "tool-a",
        "v1",
        {"tenant": "tenant-a", "resource": "resource-a", "destination": "external-a"},
    )
    implementation = ToolImplementation(
        "tool-a",
        "v1",
        (ToolTransition("run", (), (read, _effect())),),
    )
    contract = DeclaredContract("contract-a", "tool-a", "v1", frozenset({read}))
    scenario = TraceScenario("trace-a", (TraceStep(call, Environment()),), True)
    check = check_effect_completeness(
        contract, {("tool-a", "v1"): implementation}, (scenario,)
    )
    assert check.effect_complete is False
    assert len(check.uncovered_effects) == 1


def test_version_binding_mismatch_breaks_completeness() -> None:
    contract, _implementations, scenarios = _fixture(
        implementation_effect=_effect(), contract_effect=_effect()
    )
    call = scenarios[0].steps[0].call
    v2_call = ToolCall.create(
        call.call_id,
        call.tool_name,
        "v2",
        dict(call.arguments),
    )
    implementation = ToolImplementation(
        "tool-a", "v2", (ToolTransition("run", (), (_effect(),)),)
    )
    scenario = TraceScenario("trace-v2", (TraceStep(v2_call, Environment()),), False)
    check = check_effect_completeness(
        contract, {("tool-a", "v2"): implementation}, (scenario,)
    )
    assert check.effect_complete is False
    assert check.version_binding_matches is False
