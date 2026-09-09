"""确定的有界 for/guard 语义：逐轮读前状态、原子更新、逐轮排空异步队列。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    ConcreteAssignment,
    ConcreteEffect,
    StateChange,
    StructuredStateDiff,
    SymbolicTerm,
    canonical_json,
)

if TYPE_CHECKING:
    from .hidden_ir import GuardedTransition, HiddenSymbolicImplementation

MAX_LOOP_ITERATIONS = 8
MAX_ASYNC_DEPTH = 2


class UnsupportedLoop(ValueError):
    """越界/不支持只产生 UNKNOWN，不能自动缩小输入域。"""


@dataclass(frozen=True)
class LoopUpdate:
    """有界数学 Int 的增减；delta=0 时支持 Enum/Bool/条件赋值。"""

    field: str
    value: SymbolicTerm
    delta: int = 0

    def __post_init__(self):
        if type(self.delta) is not int:
            raise ValueError("loop delta 必须是数学整数")

    def instantiate(self, assignment):
        value = self.value.evaluate(assignment)
        if self.delta:
            if type(value) is not int:
                raise UnsupportedLoop("非整数不能增减")
            value += self.delta
        return value

    def to_dict(self):
        return {"field": self.field, "value": self.value.to_dict(), "delta": self.delta}


@dataclass(frozen=True)
class BoundedLoop:
    guard: BooleanFormula
    body: tuple[GuardedTransition, ...]
    max_iterations: int | None
    require_guard_false_at_bound: bool = False

    def to_dict(self):
        return {
            "guard": self.guard.to_dict(),
            "body": [t.to_digest_dict() for t in self.body],
            "max_iterations": self.max_iterations,
            "require_guard_false_at_bound": self.require_guard_false_at_bound,
        }


@dataclass(frozen=True)
class ConcreteExecution:
    events: tuple[ConcreteEffect, ...]
    final_state: tuple[tuple[str, Any], ...]
    state_diff: StructuredStateDiff
    iteration_count: int
    termination_reason: str


def validate_program(implementation, *, limit=MAX_LOOP_ITERATIONS):
    if type(limit) is not int or not 0 <= limit <= MAX_LOOP_ITERATIONS:
        raise UnsupportedLoop("configured_loop_limit_invalid")
    loop = implementation.loop
    if implementation.bounded_loop_max is None or (
        loop and loop.max_iterations is None
    ):
        raise UnsupportedLoop("unbounded_loop_not_supported")
    if loop is None:
        if implementation.bounded_loop_max:
            raise UnsupportedLoop("bounded_loop_semantics_not_implemented")
        return
    if type(loop.max_iterations) is not int or not 0 <= loop.max_iterations <= limit:
        raise UnsupportedLoop("loop_bound_exceeds_configured_limit")
    if implementation.bounded_loop_max != loop.max_iterations:
        raise UnsupportedLoop("loop_bound_metadata_mismatch")
    if implementation.delayed_depth > MAX_ASYNC_DEPTH:
        raise UnsupportedLoop("bounded_quiescence_horizon_exceeded")
    for group in (implementation.transitions, loop.body, implementation.after_loop):
        by_slot, by_order = {}, {}
        for t in group:
            for e in t.all_effects():
                if e.slot in by_slot and by_slot[e.slot].signature != e.signature:
                    raise UnsupportedLoop("conflicting_event_slot")
                if e.sequence in by_order and by_order[e.sequence] != e.slot:
                    raise UnsupportedLoop("event_order_not_total")
                by_slot[e.slot], by_order[e.sequence] = e, e.slot
        for e in by_slot.values():
            parent, depth = e.parent_slot, 0
            while parent is not None:
                if parent not in by_slot or by_slot[parent].sequence >= e.sequence:
                    raise UnsupportedLoop("causal_parent_invalid")
                depth += 1
                if depth > MAX_ASYNC_DEPTH:
                    raise UnsupportedLoop("bounded_quiescence_horizon_exceeded")
                parent = by_slot[parent].parent_slot


def group_width(group):
    return max((e.sequence + 1 for t in group for e in t.all_effects()), default=0)


def execute_program(
    implementation: HiddenSymbolicImplementation,
    assignment: ConcreteAssignment,
    *,
    limit=MAX_LOOP_ITERATIONS,
) -> ConcreteExecution:
    validate_program(implementation, limit=limit)
    implementation.schema.validate_assignment(assignment)
    state, events = dict(assignment.state), []

    def group(transitions, prefix, offset):
        current = ConcreteAssignment(assignment.inputs, tuple(state.items()))
        emitted, updates = {}, {}
        for transition in transitions:
            if not transition.guard.evaluate(current):
                continue
            for template in transition.all_effects():
                e = template.instantiate(current)
                e = replace(
                    e,
                    slot=prefix + e.slot,
                    sequence=offset + e.sequence,
                    parent_slot=None
                    if e.parent_slot is None
                    else prefix + e.parent_slot,
                )
                if e.slot in emitted and emitted[e.slot] != e:
                    raise UnsupportedLoop("conflicting_effect_payload")
                emitted[e.slot] = e
            for update in transition.state_updates:
                value = update.instantiate(current)
                if not implementation.schema.field("state", update.field).accepts(
                    value
                ):
                    raise UnsupportedLoop("state_update_out_of_range")
                if update.field in updates and canonical_json(
                    updates[update.field]
                ) != canonical_json(value):
                    raise UnsupportedLoop("conflicting_state_update")
                updates[update.field] = value
        ordered = sorted(emitted.values(), key=lambda e: e.sequence)
        available = {e.slot for e in ordered}
        if any(
            e.parent_slot is not None and e.parent_slot not in available
            for e in ordered
        ):
            raise UnsupportedLoop("causal_parent_not_executed")
        events.extend(ordered)
        state.update(updates)

    group(implementation.transitions, "pre-", 0)
    loop = implementation.loop
    iterations, reason = 0, "ACYCLIC"
    offset = group_width(implementation.transitions)
    if loop is not None:
        reason = "BOUND_REACHED"
        for i in range(loop.max_iterations):
            current = ConcreteAssignment(assignment.inputs, tuple(state.items()))
            if not loop.guard.evaluate(current):
                reason = "GUARD_FALSE"
                break
            group(loop.body, f"loop-{i}-", offset + i * group_width(loop.body))
            iterations += 1
        if (
            reason == "BOUND_REACHED"
            and loop.require_guard_false_at_bound
            and loop.guard.evaluate(
                ConcreteAssignment(assignment.inputs, tuple(state.items()))
            )
        ):
            raise UnsupportedLoop("loop_quiescence_not_established")
        offset += loop.max_iterations * group_width(loop.body)
    group(implementation.after_loop, "post-", offset)
    changes = tuple(
        StateChange("record", "state", name, before, state[name])
        for name, before in assignment.state
        if canonical_json(before) != canonical_json(state[name])
    )
    return ConcreteExecution(
        tuple(events),
        tuple(sorted(state.items())),
        StructuredStateDiff(changes),
        iterations,
        reason,
    )
