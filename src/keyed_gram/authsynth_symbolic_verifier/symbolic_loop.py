"""bounded unrolling + SSA；active 只会变 false，不会在后续轮次重新激活。"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import z3

from keyed_gram.authsynth_symbolic_shared import (
    ConcreteEffect,
    StateChange,
    StructuredStateDiff,
)

from .bounded_loop import (
    MAX_LOOP_ITERATIONS,
    ConcreteExecution,
    UnsupportedLoop,
    group_width,
    validate_program,
)
from .solver import Z3Domain


def equal_expr(left, right):
    if left is None or right is None:
        return z3.BoolVal(left is None and right is None)
    return left == right if left.sort() == right.sort() else z3.BoolVal(False)


@dataclass(frozen=True)
class EncodedEvent:
    signature: tuple
    active: z3.BoolRef
    fields: tuple


@dataclass
class SymbolicExecution:
    domain: Z3Domain
    events: tuple[EncodedEvent, ...]
    final_state: dict
    iteration_count: z3.ArithRef
    bound_reached: z3.BoolRef
    invalid: z3.BoolRef
    has_loop: bool

    def concrete_at(self, assignment):
        """仅用于差分测试：在一个明确赋值下求值，不参与完备性证明。"""
        self.domain.schema.validate_assignment(assignment)
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*self.domain.domain_constraints)
        for f in self.domain.schema.fields:
            solver.add(
                self.domain.variables[(f.source, f.name)]
                == self.domain.scalar(
                    f.source, f.name, assignment.value(f.source, f.name)
                )
            )
        if solver.check() != z3.sat:
            raise UnsupportedLoop("differential_assignment_unknown")
        model = solver.model()
        if z3.is_true(model.eval(self.invalid, model_completion=True)):
            raise UnsupportedLoop("symbolic_execution_invalid")

        def value(expression):
            if expression is None:
                return None
            result = model.eval(expression, model_completion=True)
            if z3.is_bool(result):
                return z3.is_true(result)
            if z3.is_int_value(result):
                return result.as_long()
            return result.as_string()

        events = {
            ConcreteEffect(
                e.signature[0],
                e.signature[1],
                e.signature[2],
                e.signature[3],
                *(value(f) for f in e.fields),
                parent_slot=e.signature[4],
            )
            for e in self.events
            if value(e.active)
        }
        final = tuple(
            sorted((name, value(expr)) for name, expr in self.final_state.items())
        )
        changes = tuple(
            StateChange("record", "state", name, assignment.value("state", name), v)
            for name, v in final
            if assignment.value("state", name) != v
        )
        reason = (
            "ACYCLIC"
            if not self.has_loop
            else ("BOUND_REACHED" if value(self.bound_reached) else "GUARD_FALSE")
        )
        return ConcreteExecution(
            tuple(sorted(events, key=lambda e: e.sequence)),
            final,
            StructuredStateDiff(changes),
            value(self.iteration_count),
            reason,
        )


def symbolic_execute(implementation, *, limit=MAX_LOOP_ITERATIONS):
    validate_program(implementation, limit=limit)
    domain = Z3Domain(implementation.schema)
    state = {
        f.name: domain.variables[("state", f.name)] for f in domain.schema.state_fields
    }
    events, invalid = [], []

    def at_state():
        view = copy(domain)
        view.variables = {
            **domain.variables,
            **{("state", name): value for name, value in state.items()},
        }
        return view

    def group(transitions, enabled, prefix, offset):
        view, updates, emitted = at_state(), {}, []
        for transition in transitions:
            active = z3.And(enabled, view.formula(transition.guard))
            for e in transition.all_effects():
                signature = (
                    prefix + e.slot,
                    offset + e.sequence,
                    e.phase,
                    e.kind,
                    None if e.parent_slot is None else prefix + e.parent_slot,
                )
                encoded = EncodedEvent(
                    signature,
                    active,
                    tuple(
                        view.term(t)
                        for t in (e.tenant, e.resource, e.destination, e.amount)
                    ),
                )
                for other in emitted:
                    if other.signature[0] == signature[0]:
                        same = z3.And(
                            *(
                                equal_expr(a, b)
                                for a, b in zip(
                                    other.fields, encoded.fields, strict=True
                                )
                            )
                        )
                        invalid.append(z3.And(active, other.active, z3.Not(same)))
                emitted.append(encoded)
            for update in transition.state_updates:
                term = view.term(update.value)
                delta = getattr(update, "delta", 0)
                original = state[update.field]
                if (
                    term is None
                    or term.sort() != original.sort()
                    or (delta and not z3.is_int(term))
                ):
                    invalid.append(active)
                    continue
                if delta:
                    term = term + delta
                field = domain.schema.field("state", update.field)
                if field.kind == "int":
                    invalid.append(
                        z3.And(
                            active, z3.Or(term < field.minimum, term > field.maximum)
                        )
                    )
                elif field.kind == "enum":
                    invalid.append(
                        z3.And(
                            active,
                            z3.Not(
                                z3.Or(*(term == z3.StringVal(v) for v in field.values))
                            ),
                        )
                    )
                for other_active, other_term in updates.get(update.field, []):
                    invalid.append(z3.And(active, other_active, term != other_term))
                updates.setdefault(update.field, []).append((active, term))
        for e in emitted:
            if e.signature[4] is not None:
                parents = [
                    p.active for p in emitted if p.signature[0] == e.signature[4]
                ]
                invalid.append(z3.And(e.active, z3.Not(z3.Or(*parents))))
        events.extend(emitted)
        # RHS 全部引用该轮 s_i；再一起建立 s_(i+1)。
        for name, choices in updates.items():
            result = state[name]
            for active, term in choices:
                result = z3.If(active, term, result)
            state[name] = z3.simplify(result)

    group(implementation.transitions, z3.BoolVal(True), "pre-", 0)
    loop, active = implementation.loop, z3.BoolVal(True)
    count, offset = z3.IntVal(0), group_width(implementation.transitions)
    if loop is not None:
        for i in range(loop.max_iterations):
            execute = z3.And(active, at_state().formula(loop.guard))
            group(loop.body, execute, f"loop-{i}-", offset + i * group_width(loop.body))
            count = z3.simplify(count + z3.If(execute, 1, 0))
            active = z3.simplify(execute)
        if loop.require_guard_false_at_bound:
            invalid.append(z3.And(active, at_state().formula(loop.guard)))
        offset += loop.max_iterations * group_width(loop.body)
    group(implementation.after_loop, z3.BoolVal(True), "post-", offset)
    return SymbolicExecution(
        domain, tuple(events), state, count, active, z3.Or(*invalid), loop is not None
    )
