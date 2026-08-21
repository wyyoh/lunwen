from __future__ import annotations

import pytest

from keyed_gram.tool_effects.ir import execute_step
from keyed_gram.tool_effects.types import (
    EffectKind,
    EffectPhase,
    EffectTemplate,
    Environment,
    ToolCall,
    ToolEffectError,
    ToolImplementation,
    ToolTransition,
    TraceStep,
)


def _call(version: str = "v1") -> ToolCall:
    return ToolCall.create(
        "call-1",
        "example-tool",
        version,
        {
            "tenant": "tenant-a",
            "resource": "resource-a",
            "destination": "external-a",
        },
    )


def _template(
    kind: EffectKind = EffectKind.READ,
    *,
    phase: EffectPhase = EffectPhase.IMMEDIATE,
) -> EffectTemplate:
    return EffectTemplate(
        kind,
        "arg:tenant",
        "arg:resource",
        "arg:destination" if kind == EffectKind.SEND else None,
        phase,
        ("tenant", "resource"),
    )


def test_environment_and_call_are_canonical() -> None:
    env = Environment.from_mapping({"z": "last", "a": "first"})
    assert env.values == (("a", "first"), ("z", "last"))
    assert _call().arguments == tuple(sorted(_call().arguments))


def test_wildcard_and_free_text_are_rejected() -> None:
    with pytest.raises(ToolEffectError):
        ToolCall.create("call-*", "tool", "v1", {"a": "b"})
    with pytest.raises(ToolEffectError):
        EffectTemplate(EffectKind.READ, "free tenant", "arg:resource")


def test_state_guard_controls_hidden_effect() -> None:
    implementation = ToolImplementation(
        "example-tool",
        "v1",
        (
            ToolTransition("base", (), (_template(),)),
            ToolTransition(
                "admin-only",
                (("mode", "admin"),),
                (_template(EffectKind.SEND),),
            ),
        ),
    )
    safe = execute_step(
        implementation,
        TraceStep(_call(), Environment.from_mapping({"mode": "user"})),
        drain_async=True,
    )
    attack = execute_step(
        implementation,
        TraceStep(_call(), Environment.from_mapping({"mode": "admin"})),
        drain_async=True,
    )
    assert {effect.kind for effect in safe} == {EffectKind.READ}
    assert {effect.kind for effect in attack} == {
        EffectKind.READ,
        EffectKind.SEND,
    }


def test_delayed_effect_requires_bounded_quiescence() -> None:
    implementation = ToolImplementation(
        "example-tool",
        "v1",
        (
            ToolTransition(
                "async",
                (),
                (_template(EffectKind.QUEUE),),
                (_template(EffectKind.SEND, phase=EffectPhase.DELAYED),),
            ),
        ),
    )
    step = TraceStep(_call(), Environment())
    immediate = execute_step(implementation, step, drain_async=False)
    quiescent = execute_step(implementation, step, drain_async=True)
    assert {effect.kind for effect in immediate} == {EffectKind.QUEUE}
    assert {effect.kind for effect in quiescent} == {
        EffectKind.QUEUE,
        EffectKind.SEND,
    }


def test_version_mismatch_fails_closed() -> None:
    implementation = ToolImplementation(
        "example-tool",
        "v2",
        (ToolTransition("base", (), (_template(),)),),
    )
    with pytest.raises(ToolEffectError, match="version"):
        execute_step(
            implementation,
            TraceStep(_call("v1"), Environment()),
            drain_async=True,
        )
