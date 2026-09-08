"""候选 symbolic contract 与内部 ToolSymbolicIR 的 SMT 双向差异检查。"""

from __future__ import annotations

from collections.abc import Sequence

import z3

from keyed_gram.authsynth_symbolic_shared import (
    SymbolicEffectContract,
    SymbolicStateUpdateClause,
    VerificationKind,
    VerificationResult,
    canonical_digest,
)

from .hidden_ir import GuardedTransition, HiddenSymbolicCase
from .solver import Z3Domain, deterministic_model


def _or(values: Sequence[z3.BoolRef]) -> z3.BoolRef:
    return z3.Or(*values) if values else z3.BoolVal(False)


class BoundedSymbolicVerifier:
    """内部持有语义；边界外仅返回 assignment/kind/equivalent/unknown。"""

    solver_name = "Z3"
    solver_version = z3.get_version_string()

    def __init__(self, cases: Sequence[HiddenSymbolicCase], *, timeout_ms: int) -> None:
        self._cases = {item.analyzer_input.case_handle: item for item in cases}
        self._timeout_ms = timeout_ms
        self._check_counts: dict[str, int] = {}

    def check_count(self, case_handle: str) -> int:
        return self._check_counts.get(case_handle, 0)

    def check(
        self, case_handle: str, candidate_contract: SymbolicEffectContract
    ) -> VerificationResult:
        try:
            return self._check(case_handle, candidate_contract)
        except (ValueError, z3.Z3Exception):
            # schema/编码错误不转化成等价结论；也不泄露内部公式或 traceback。
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="unsupported_or_invalid_symbolic_semantics",
            )

    def _check(
        self, case_handle: str, candidate_contract: SymbolicEffectContract
    ) -> VerificationResult:
        try:
            case = self._cases[case_handle]
        except KeyError as exc:
            raise ValueError("未知 symbolic case handle") from exc
        self._check_counts[case_handle] = self.check_count(case_handle) + 1
        implementation = case.implementation
        if candidate_contract.tool_id != implementation.tool_id:
            return VerificationResult(
                VerificationKind.UNKNOWN, reason_code="candidate_tool_identity_mismatch"
            )
        if (
            candidate_contract.implementation_version_digest
            != implementation.version_digest
        ):
            return VerificationResult(
                VerificationKind.VERSION_MISMATCH,
                observed_version_digest=implementation.version_digest,
                reason_code="implementation_version_changed",
            )
        if implementation.proof_obstacle is not None:
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code=implementation.proof_obstacle,
            )
        if implementation.bounded_loop_max is None:
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="unbounded_loop_not_supported",
            )
        if implementation.bounded_loop_max > 0:
            # 当前 IR 尚未展开循环体和逐轮状态；数字上限本身不是完整性证据。
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="bounded_loop_semantics_not_implemented",
            )
        if implementation.delayed_depth > 2:
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="bounded_quiescence_horizon_exceeded",
            )
        if implementation.instrumentation_coverage < 1.0:
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="instrumentation_coverage_incomplete",
            )
        try:
            candidate_contract.validate(
                case.analyzer_input.schema, case.analyzer_input.grammar_limits
            )
        except ValueError:
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="candidate_contract_schema_invalid",
            )
        if candidate_contract.unsupported_regions:
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="candidate_has_unsupported_region",
            )

        domain = Z3Domain(case.analyzer_input.schema)
        for label, updates in (
            (
                "implementation",
                [
                    (transition.guard, update)
                    for transition in implementation.transitions
                    for update in transition.state_updates
                ],
            ),
            (
                "candidate",
                [
                    (clause.guard, clause.update)
                    for clause in candidate_contract.state_updates
                ],
            ),
        ):
            solver = z3.Solver()
            solver.set(timeout=self._timeout_ms)
            solver.add(
                *domain.domain_constraints, self._invalid_updates(domain, updates)
            )
            status = solver.check()
            if status == z3.unknown:
                return VerificationResult(
                    VerificationKind.UNKNOWN, reason_code="solver_timeout_or_unknown"
                )
            if status != z3.unsat:
                return VerificationResult(
                    VerificationKind.UNKNOWN,
                    reason_code=f"{label}_state_semantics_invalid",
                )
        hidden_effects = [
            (domain.formula(transition.guard), effect)
            for transition in implementation.transitions
            for effect in transition.all_effects()
        ]
        candidate_effects = [
            (domain.formula(clause.guard), clause.effect)
            for clause in candidate_contract.clauses
        ]

        state_mismatch = self._state_mismatch(
            domain, implementation.transitions, candidate_contract.state_updates
        )
        outcome = self._counterexample(
            case,
            domain,
            VerificationKind.STATE_UPDATE_MISMATCH,
            state_mismatch,
        )
        if outcome is not None:
            return outcome

        field_mismatch = self._field_mismatch(domain, hidden_effects, candidate_effects)
        outcome = self._counterexample(
            case,
            domain,
            VerificationKind.FIELD_BINDING_MISMATCH,
            field_mismatch,
        )
        if outcome is not None:
            return outcome

        missing = self._missing_effect(domain, hidden_effects, candidate_effects)
        outcome = self._counterexample(
            case, domain, VerificationKind.MISSING_EFFECT, missing
        )
        if outcome is not None:
            return outcome

        spurious = self._spurious_effect(domain, hidden_effects, candidate_effects)
        outcome = self._counterexample(
            case, domain, VerificationKind.SPURIOUS_EFFECT, spurious
        )
        if outcome is not None:
            return outcome
        return VerificationResult(
            VerificationKind.EQUIVALENT,
            observed_version_digest=implementation.version_digest,
            reason_code="smt_bounded_domain_equivalent",
        )

    @staticmethod
    def _invalid_updates(domain: Z3Domain, updates: Sequence[tuple]) -> z3.BoolRef:
        """所有更新按同一前状态求值；冲突与越域均不能被 ITE 覆盖隐藏。"""

        invalid = []
        encoded = []
        for guard, update in updates:
            field = domain.schema.field("state", update.field)
            active = domain.formula(guard)
            term = domain.term(update.value)
            if (
                term is None
                or term.sort() != domain.variables[("state", field.name)].sort()
            ):
                invalid.append(active)
                continue
            if field.kind == "int":
                invalid.append(
                    z3.And(active, z3.Or(term < field.minimum, term > field.maximum))
                )
            elif field.kind == "enum":
                invalid.append(
                    z3.And(
                        active,
                        z3.Not(
                            z3.Or(
                                *[
                                    term == z3.StringVal(value)
                                    for value in field.enum_values
                                ]
                            )
                        ),
                    )
                )
            for other_field, other_active, other_term in encoded:
                if field.name == other_field:
                    invalid.append(z3.And(active, other_active, term != other_term))
            encoded.append((field.name, active, term))
        return _or(invalid)

    def _counterexample(
        self,
        case: HiddenSymbolicCase,
        domain: Z3Domain,
        kind: VerificationKind,
        predicate: z3.BoolRef,
    ) -> VerificationResult | None:
        status, assignment = deterministic_model(
            domain, predicate, timeout_ms=self._timeout_ms
        )
        if status == "unknown":
            return VerificationResult(
                VerificationKind.UNKNOWN,
                reason_code="solver_timeout_or_unknown",
            )
        if status == "unsat":
            return None
        assert assignment is not None
        return VerificationResult(
            kind,
            assignment=assignment,
            counterexample_digest=canonical_digest(
                {
                    "case": case.public_case_id,
                    "kind": kind.value,
                    "assignment": assignment.to_dict(),
                }
            ),
            observed_version_digest=case.implementation.version_digest,
            reason_code="symbolic_difference_witness",
        )

    @staticmethod
    def _same_signature(left: object, right: object) -> bool:
        return left.signature == right.signature

    @staticmethod
    def _fields_equal(domain: Z3Domain, left: object, right: object) -> z3.BoolRef:
        return z3.And(
            domain.equal_terms(left.tenant, right.tenant),
            domain.equal_terms(left.resource, right.resource),
            domain.equal_terms(left.destination, right.destination),
            domain.equal_terms(left.amount, right.amount),
        )

    def _field_mismatch(
        self,
        domain: Z3Domain,
        hidden: Sequence[tuple[z3.BoolRef, object]],
        candidate: Sequence[tuple[z3.BoolRef, object]],
    ) -> z3.BoolRef:
        values: list[z3.BoolRef] = []
        for hidden_guard, hidden_effect in hidden:
            same = [
                item
                for item in candidate
                if self._same_signature(hidden_effect, item[1])
            ]
            if not same:
                continue
            any_active = _or([guard for guard, _ in same])
            exact = _or(
                [
                    z3.And(guard, self._fields_equal(domain, hidden_effect, effect))
                    for guard, effect in same
                ]
            )
            values.append(z3.And(hidden_guard, any_active, z3.Not(exact)))
        for candidate_guard, candidate_effect in candidate:
            same = [
                item
                for item in hidden
                if self._same_signature(candidate_effect, item[1])
            ]
            if not same:
                continue
            any_active = _or([guard for guard, _ in same])
            exact = _or(
                [
                    z3.And(guard, self._fields_equal(domain, effect, candidate_effect))
                    for guard, effect in same
                ]
            )
            values.append(z3.And(candidate_guard, any_active, z3.Not(exact)))
        return _or(values)

    def _missing_effect(
        self,
        domain: Z3Domain,
        hidden: Sequence[tuple[z3.BoolRef, object]],
        candidate: Sequence[tuple[z3.BoolRef, object]],
    ) -> z3.BoolRef:
        del domain
        values = []
        for guard, effect in hidden:
            same = [
                candidate_guard
                for candidate_guard, item in candidate
                if self._same_signature(effect, item)
            ]
            values.append(z3.And(guard, z3.Not(_or(same))))
        return _or(values)

    def _spurious_effect(
        self,
        domain: Z3Domain,
        hidden: Sequence[tuple[z3.BoolRef, object]],
        candidate: Sequence[tuple[z3.BoolRef, object]],
    ) -> z3.BoolRef:
        del domain
        values = []
        for guard, effect in candidate:
            same = [
                hidden_guard
                for hidden_guard, item in hidden
                if self._same_signature(effect, item)
            ]
            values.append(z3.And(guard, z3.Not(_or(same))))
        return _or(values)

    @staticmethod
    def _state_mismatch(
        domain: Z3Domain,
        transitions: Sequence[GuardedTransition],
        candidate_updates: Sequence[SymbolicStateUpdateClause],
    ) -> z3.BoolRef:
        values: list[z3.BoolRef] = []
        for field in domain.schema.state_fields:
            original = domain.variables[("state", field.name)]
            hidden_final = original
            for transition in transitions:
                for update in transition.state_updates:
                    if update.field == field.name:
                        term = domain.term(update.value)
                        assert term is not None
                        hidden_final = z3.If(
                            domain.formula(transition.guard), term, hidden_final
                        )
            candidate_final = original
            for clause in candidate_updates:
                if clause.update.field == field.name:
                    term = domain.term(clause.update.value)
                    assert term is not None
                    candidate_final = z3.If(
                        domain.formula(clause.guard), term, candidate_final
                    )
            values.append(hidden_final != candidate_final)
        return _or(values)
