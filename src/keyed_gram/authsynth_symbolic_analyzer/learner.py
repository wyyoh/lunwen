"""仅从公开 schema 与 replay examples 合成 guarded symbolic contract。"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Sequence
from typing import Any

import z3

from keyed_gram.authsynth_symbolic_shared import (
    AnalyzerEvidence,
    BooleanFormula,
    BoundedSchema,
    ConcreteAssignment,
    ConcreteEffect,
    Conjunction,
    GrammarLimits,
    Literal,
    LiteralOperator,
    SymbolicEffectClause,
    SymbolicEffectContract,
    SymbolicEffectTemplate,
    SymbolicStateUpdate,
    SymbolicStateUpdateClause,
    SymbolicTerm,
    TermKind,
    VariableRef,
    canonical_json,
)


class SynthesisTimeout(RuntimeError):
    """仅终止本次候选搜索，不扩大语法或预算。"""


def _remaining_ms(deadline: float) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SynthesisTimeout("synthesis_timeout")
    return max(1, math.ceil(remaining * 1000))


def _same_scalar(left: Any, right: Any) -> bool:
    # Python 中 True == 1，但 IR 的 Bool 与 Int 必须保持不同类型。
    return type(left) is type(right) and left == right


def _literal_pool(schema: BoundedSchema) -> tuple[Literal, ...]:
    result: set[Literal] = set()
    for field in schema.fields:
        ref = VariableRef(field.source, field.name)
        for value in field.values:
            result.add(Literal(ref, LiteralOperator.EQ, value))
            result.add(Literal(ref, LiteralOperator.NE, value))
            if field.kind == "int":
                result.add(Literal(ref, LiteralOperator.LE, value))
                result.add(Literal(ref, LiteralOperator.GE, value))
    return tuple(sorted(result))


def _covers(
    conjunction: Conjunction, values: Sequence[ConcreteAssignment]
) -> frozenset[int]:
    return frozenset(
        index for index, item in enumerate(values) if conjunction.evaluate(item)
    )


def learn_guard(
    schema: BoundedSchema,
    positives: Sequence[ConcreteAssignment],
    negatives: Sequence[ConcreteAssignment],
    limits: GrammarLimits,
    *,
    minimize: bool,
    timeout_ms: int = 3000,
    deadline: float | None = None,
) -> BooleanFormula | None:
    """在候选 cube 池中优化 DNF；不声称对整个语法全局最优。"""

    if timeout_ms < 1:
        raise ValueError("synthesis timeout 必须为正")
    deadline = (
        deadline if deadline is not None else time.monotonic() + timeout_ms / 1000
    )
    _remaining_ms(deadline)
    if not positives:
        return BooleanFormula.false()
    if not negatives:
        return BooleanFormula.true()
    pool = _literal_pool(schema)
    candidates: dict[tuple[frozenset[int], int], Conjunction] = {}
    for positive in positives:
        true_literals = tuple(item for item in pool if item.evaluate(positive))
        found_for_positive = False
        for size in range(1, limits.max_literals_per_conjunction + 1):
            local: list[Conjunction] = []
            for literals in itertools.combinations(true_literals, size):
                _remaining_ms(deadline)
                conjunction = Conjunction(literals)
                if any(conjunction.evaluate(item) for item in negatives):
                    continue
                coverage = _covers(conjunction, positives)
                if not coverage:
                    continue
                local.append(conjunction)
            if local:
                for conjunction in sorted(
                    local, key=lambda item: canonical_json(item.to_dict())
                )[:128]:
                    coverage = _covers(conjunction, positives)
                    key = (coverage, len(conjunction.literals))
                    existing = candidates.get(key)
                    if existing is None or canonical_json(
                        conjunction.to_dict()
                    ) < canonical_json(existing.to_dict()):
                        candidates[key] = conjunction
                found_for_positive = True
                break
        if not found_for_positive:
            return None
    ordered = tuple(
        sorted(candidates.values(), key=lambda item: canonical_json(item.to_dict()))
    )
    if not ordered:
        return None
    if not minimize:
        selected: list[Conjunction] = []
        covered: set[int] = set()
        for index in range(len(positives)):
            if index in covered:
                continue
            options = [item for item in ordered if item.evaluate(positives[index])]
            if not options:
                return None
            chosen = options[0]
            selected.append(chosen)
            covered.update(_covers(chosen, positives))
        formula = BooleanFormula(tuple(selected))
        try:
            formula.validate(schema, limits)
        except ValueError:
            return None
        return formula

    optimizer = z3.Optimize()
    optimizer.set(priority="lex", timeout=_remaining_ms(deadline))
    choose = [z3.Bool(f"cube_{index}") for index in range(len(ordered))]
    for positive_index in range(len(positives)):
        covering = [
            choose[index]
            for index, cube in enumerate(ordered)
            if cube.evaluate(positives[positive_index])
        ]
        if not covering:
            return None
        optimizer.add(z3.Or(*covering))
    optimizer.add(
        z3.Sum([z3.If(item, 1, 0) for item in choose]) <= limits.max_disjuncts
    )
    optimizer.minimize(z3.Sum([z3.If(item, 1, 0) for item in choose]))
    literal_cost = z3.Sum(
        [
            z3.If(choose[index], len(cube.literals), 0)
            for index, cube in enumerate(ordered)
        ]
    )
    optimizer.add(literal_cost <= limits.max_total_literals)
    optimizer.minimize(literal_cost)
    # 固定前两个目标后，二进制权重唯一确定规范化 cube 的选择向量。
    optimizer.maximize(
        z3.Sum(
            [
                z3.If(item, 1 << (len(choose) - index - 1), 0)
                for index, item in enumerate(choose)
            ]
        )
    )
    optimizer.set(timeout=_remaining_ms(deadline))
    status = optimizer.check()
    if status == z3.unknown:
        raise SynthesisTimeout("solver_timeout_or_unknown")
    _remaining_ms(deadline)
    if status != z3.sat:
        return None
    model = optimizer.model()
    selected = tuple(
        cube
        for item, cube in zip(choose, ordered, strict=True)
        if z3.is_true(model.eval(item, model_completion=True))
    )
    formula = BooleanFormula(selected)
    try:
        formula.validate(schema, limits)
    except ValueError:
        return None
    return formula


def _base_term_candidates(
    schema: BoundedSchema,
    assignments: Sequence[ConcreteAssignment],
    values: Sequence[Any],
) -> tuple[SymbolicTerm, ...]:
    if values and all(value is None for value in values):
        return (SymbolicTerm.none(),)
    candidates: list[SymbolicTerm] = []
    for field in schema.fields:
        if all(
            _same_scalar(assignment.value(field.source, field.name), value)
            for assignment, value in zip(assignments, values, strict=True)
        ):
            candidates.append(SymbolicTerm.var(field.source, field.name))
    if values and all(_same_scalar(value, values[0]) for value in values):
        candidates.append(SymbolicTerm.const(values[0]))
    unique = {canonical_json(item.to_dict()): item for item in candidates}
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: (
                0 if unique[item].kind == TermKind.VARIABLE else 1,
                item,
            ),
        )
    )


def infer_term(
    schema: BoundedSchema,
    assignments: Sequence[ConcreteAssignment],
    values: Sequence[Any],
    limits: GrammarLimits,
    *,
    deadline: float | None = None,
) -> SymbolicTerm | None:
    deadline = deadline if deadline is not None else time.monotonic() + 3.0
    _remaining_ms(deadline)
    direct = _base_term_candidates(schema, assignments, values)
    if direct:
        return direct[0]
    if not limits.allow_ite_terms or not assignments:
        return None
    atoms: list[SymbolicTerm] = []
    for field in schema.fields:
        atoms.append(SymbolicTerm.var(field.source, field.name))
    for value in sorted(set(values), key=lambda item: canonical_json(item)):
        if value is None:
            atoms.append(SymbolicTerm.none())
        else:
            atoms.append(SymbolicTerm.const(value))
    atoms = list({canonical_json(item.to_dict()): item for item in atoms}.values())
    candidates: list[SymbolicTerm] = []
    for literal in _literal_pool(schema):
        for then_term, else_term in itertools.product(atoms, repeat=2):
            _remaining_ms(deadline)
            term = SymbolicTerm.ite(literal, then_term, else_term)
            if all(
                _same_scalar(term.evaluate(assignment), value)
                for assignment, value in zip(assignments, values, strict=True)
            ):
                candidates.append(term)
    if not candidates:
        return None
    return min(
        candidates, key=lambda item: (item.ast_nodes, canonical_json(item.to_dict()))
    )


def _template_from_events(
    schema: BoundedSchema,
    assignments: Sequence[ConcreteAssignment],
    events: Sequence[ConcreteEffect],
    limits: GrammarLimits,
    *,
    deadline: float,
) -> SymbolicEffectTemplate | None:
    first = events[0]
    if any(item.signature != first.signature for item in events):
        return None
    terms = []
    for values in (
        [item.tenant for item in events],
        [item.resource for item in events],
        [item.destination for item in events],
        [item.amount for item in events],
    ):
        term = infer_term(schema, assignments, values, limits, deadline=deadline)
        if term is None:
            return None
        terms.append(term)
    return SymbolicEffectTemplate(
        slot=first.slot,
        sequence=first.sequence,
        phase=first.phase,
        kind=first.kind,
        tenant=terms[0],
        resource=terms[1],
        destination=terms[2],
        amount=terms[3],
        parent_slot=first.parent_slot,
    )


class SymbolicContractLearner:
    """examples-only learner；不会请求或遍历完整 assignment 域。"""

    def __init__(self, *, minimize: bool = True) -> None:
        self.minimize = minimize
        self.last_failure_reason: str | None = None

    def fit(
        self,
        *,
        tool_id: str,
        schema: BoundedSchema,
        version_digest: str,
        limits: GrammarLimits,
        evidence: Sequence[AnalyzerEvidence],
        input_schema_digest: str,
        state_schema_digest: str,
        timeout_ms: int = 3000,
    ) -> SymbolicEffectContract | None:
        self.last_failure_reason = None
        if timeout_ms < 1:
            raise ValueError("synthesis timeout 必须为正")
        try:
            return self._fit(
                tool_id=tool_id,
                schema=schema,
                version_digest=version_digest,
                limits=limits,
                evidence=evidence,
                input_schema_digest=input_schema_digest,
                state_schema_digest=state_schema_digest,
                deadline=time.monotonic() + timeout_ms / 1000,
            )
        except SynthesisTimeout:
            self.last_failure_reason = "synthesis_timeout_or_unknown"
            return None

    def _fit(
        self,
        *,
        tool_id: str,
        schema: BoundedSchema,
        version_digest: str,
        limits: GrammarLimits,
        evidence: Sequence[AnalyzerEvidence],
        input_schema_digest: str,
        state_schema_digest: str,
        deadline: float,
    ) -> SymbolicEffectContract | None:
        _remaining_ms(deadline)
        if not evidence:
            return None
        assignments = [item.assignment for item in evidence]
        event_maps = [{event.slot: event for event in item.events} for item in evidence]
        slots = sorted({slot for values in event_maps for slot in values})
        clauses: list[SymbolicEffectClause] = []
        for slot in slots:
            positive_indexes = [
                index for index, values in enumerate(event_maps) if slot in values
            ]
            negative_indexes = [
                index for index, values in enumerate(event_maps) if slot not in values
            ]
            positive_assignments = [assignments[index] for index in positive_indexes]
            negative_assignments = [assignments[index] for index in negative_indexes]
            guard = learn_guard(
                schema,
                positive_assignments,
                negative_assignments,
                limits,
                minimize=self.minimize,
                deadline=deadline,
            )
            if guard is None:
                return None
            template = _template_from_events(
                schema,
                positive_assignments,
                [event_maps[index][slot] for index in positive_indexes],
                limits,
                deadline=deadline,
            )
            if template is None:
                return None
            clauses.append(SymbolicEffectClause(guard, template))

        updates: list[SymbolicStateUpdateClause] = []
        for field in schema.state_fields:
            final_values: list[Any] = []
            changed: list[bool] = []
            for item in evidence:
                changes = {
                    change.field: change.after for change in item.state_diff.changes
                }
                before = item.assignment.value("state", field.name)
                after = changes.get(field.name, before)
                final_values.append(after)
                changed.append(after != before)
            if not any(changed):
                continue
            term = infer_term(
                schema, assignments, final_values, limits, deadline=deadline
            )
            if term is not None:
                updates.append(
                    SymbolicStateUpdateClause(
                        BooleanFormula.true(), SymbolicStateUpdate(field.name, term)
                    )
                )
                continue
            positive_indexes = [index for index, value in enumerate(changed) if value]
            negative_indexes = [
                index for index, value in enumerate(changed) if not value
            ]
            guard = learn_guard(
                schema,
                [assignments[index] for index in positive_indexes],
                [assignments[index] for index in negative_indexes],
                limits,
                minimize=self.minimize,
                deadline=deadline,
            )
            term = infer_term(
                schema,
                [assignments[index] for index in positive_indexes],
                [final_values[index] for index in positive_indexes],
                limits,
                deadline=deadline,
            )
            if guard is None or term is None:
                return None
            updates.append(
                SymbolicStateUpdateClause(guard, SymbolicStateUpdate(field.name, term))
            )

        contract = SymbolicEffectContract(
            tool_id=tool_id,
            implementation_version_digest=version_digest,
            input_schema_digest=input_schema_digest,
            state_schema_digest=state_schema_digest,
            clauses=tuple(clauses),
            state_updates=tuple(updates),
        )
        try:
            contract.validate(schema, limits)
        except ValueError:
            return None
        _remaining_ms(deadline)
        if not self._fits(contract, evidence):
            return None
        return contract

    @staticmethod
    def _fits(
        contract: SymbolicEffectContract, evidence: Sequence[AnalyzerEvidence]
    ) -> bool:
        for item in evidence:
            predicted_events, updates = contract.predict(item.assignment)
            if {event.semantic_key for event in predicted_events} != {
                event.semantic_key for event in item.events
            }:
                return False
            expected_state = dict(item.assignment.state)
            for change in item.state_diff.changes:
                expected_state[change.field] = change.after
            predicted_state = dict(item.assignment.state)
            predicted_state.update(updates)
            if predicted_state != expected_state:
                return False
        return True
