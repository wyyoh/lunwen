"""ToolEffectIR 的确定性执行、异步排空与 trace oracle。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import (
    EffectAtom,
    EffectPhase,
    ToolEffectError,
    ToolImplementation,
    TraceScenario,
    TraceStep,
)


def execute_step(
    implementation: ToolImplementation,
    step: TraceStep,
    *,
    drain_async: bool,
) -> tuple[EffectAtom, ...]:
    call = step.call
    if call.tool_name != implementation.tool_name:
        raise ToolEffectError("call/tool implementation 不匹配")
    if call.version != implementation.version:
        raise ToolEffectError("call/implementation version 不匹配")
    enabled = [
        transition
        for transition in implementation.transitions
        if transition.enabled(step.environment)
    ]
    if not enabled:
        raise ToolEffectError("当前环境没有可执行 transition")
    effects: set[EffectAtom] = set()
    for transition in enabled:
        effects.update(
            template.instantiate(call) for template in transition.immediate_effects
        )
        if drain_async:
            effects.update(
                template.instantiate(call) for template in transition.delayed_effects
            )
    return tuple(sorted(effects))


def execute_trace(
    implementations: Mapping[tuple[str, str], ToolImplementation],
    scenario: TraceScenario,
    *,
    drain_async: bool = True,
) -> tuple[EffectAtom, ...]:
    effects: set[EffectAtom] = set()
    for step in scenario.steps:
        key = (step.call.tool_name, step.call.version)
        try:
            implementation = implementations[key]
        except KeyError as exc:
            raise ToolEffectError(f"未知 tool/version：{key}") from exc
        effects.update(
            execute_step(
                implementation,
                step,
                drain_async=drain_async,
            )
        )
    return tuple(sorted(effects))


def quiescent_effects(
    implementations: Mapping[tuple[str, str], ToolImplementation],
    scenarios: Sequence[TraceScenario],
) -> frozenset[EffectAtom]:
    """排空所有有限 delayed effect 后返回精确效果集合。"""

    effects: set[EffectAtom] = set()
    for scenario in scenarios:
        effects.update(execute_trace(implementations, scenario, drain_async=True))
    return frozenset(effects)


def immediate_effects(
    implementations: Mapping[tuple[str, str], ToolImplementation],
    scenarios: Sequence[TraceScenario],
) -> frozenset[EffectAtom]:
    effects: set[EffectAtom] = set()
    for scenario in scenarios:
        effects.update(execute_trace(implementations, scenario, drain_async=False))
    return frozenset(effects)


def delayed_effect_count(effects: Sequence[EffectAtom]) -> int:
    return sum(effect.phase == EffectPhase.DELAYED for effect in effects)
