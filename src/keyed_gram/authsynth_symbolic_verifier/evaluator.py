"""AuthSymbolBench 独立 evaluator 与 B0–B5/A0/U0。"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from keyed_gram.authsynth_symbolic_analyzer import (
    AuthSynthSymbolicAnalyzer,
    SymbolicContractLearner,
)
from keyed_gram.authsynth_symbolic_analyzer.models import SymbolicAnalysisOutcome
from keyed_gram.authsynth_symbolic_shared import (
    AnalyzerEvidence,
    CertificateStatus,
    ConcreteAssignment,
    ConcreteEffect,
    ContractStatus,
    PatchAtom,
    ShieldStatus,
    StateChange,
    StructuredStateDiff,
    SymbolicEffectContract,
    canonical_digest,
    canonical_json,
)

from .benchmark import enumerate_assignments
from .equivalence import BoundedSymbolicVerifier
from .hidden_ir import HiddenSymbolicCase
from .replay_service import SymbolicSandboxReplay


class SymbolicMethod(str, Enum):
    DENY_ALL = "DenyAll"
    DECLARED = "Declared-Contract Monitor"
    RANDOM_TABLE = "Random Replay Table"
    COVERAGE_TABLE = "Coverage-Guided Replay Table"
    PASSIVE = "Passive Symbolic Learner"
    CEGIS_NO_MIN = "CEGIS-No-Minimization"
    AUTHSYNTH = "AuthSynth-Symbolic"
    GOLD = "Gold Symbolic Contract"


METHODS = tuple(SymbolicMethod)


@dataclass(frozen=True)
class MethodExecution:
    method: SymbolicMethod
    contract: SymbolicEffectContract | None
    contract_status: ContractStatus
    shield_status: ShieldStatus
    certificate_status: CertificateStatus
    outcome: SymbolicAnalysisOutcome | None
    replayed_assignments: tuple[ConcreteAssignment, ...]
    trace_table: tuple[
        tuple[str, tuple[ConcreteEffect, ...], tuple[tuple[str, Any], ...]], ...
    ]
    complete_claim_made: bool
    analysis_time_ms: float
    query_count: int
    solver_calls: int


@dataclass(frozen=True)
class SymbolicEvaluationRow:
    method: str
    case_id: str
    split: str
    domain: str
    mutation_category: str
    expected_unknown: bool
    domain_assignment_count: int
    replay_query_count: int
    solver_call_count: int
    complete_claim_made: bool
    false_verified_complete: bool
    analysis_unknown: bool
    unseen_actual_effect_count: int
    unseen_predicted_effect_count: int
    unseen_effect_true_positive_count: int
    unseen_effect_false_positive_count: int
    guard_true_positive_count: int
    guard_false_positive_count: int
    guard_false_negative_count: int
    guard_true_negative_count: int
    guard_exact_match: bool
    tenant_binding_correct: int
    tenant_binding_total: int
    resource_binding_correct: int
    resource_binding_total: int
    destination_binding_correct: int
    destination_binding_total: int
    amount_binding_correct: int
    amount_binding_total: int
    phase_binding_correct: int
    phase_binding_total: int
    state_binding_correct: int
    state_binding_total: int
    forbidden_assignment_count: int
    forbidden_accepted_count: int
    safe_assignment_count: int
    safe_allowed_count: int
    unknown_induced_rejection_count: int
    drift_expected: bool
    drift_detected: bool
    old_certificate_reused_after_drift: bool
    converged: bool
    predicted_patch_types: tuple[str, ...]
    expected_patch_types: tuple[str, ...]
    reason_codes: tuple[str, ...]
    contract_digest: str
    certificate_id: str | None
    iteration_digests: tuple[str, ...]
    analysis_time_ms: float
    synthesis_time_ms: float
    verification_time_ms: float
    contract_clause_count: int
    contract_literal_count: int
    contract_ast_node_count: int
    contract_size_bytes: int
    certificate_size_bytes: int
    predicted_patch_atom_digests: tuple[str, ...] = ()
    expected_patch_atom_digests: tuple[str, ...] = ()
    discovered_omission_atom_digests: tuple[str, ...] = ()
    expected_omission_atom_digests: tuple[str, ...] = ()
    unlocated_patch_record_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _final_state(
    assignment: ConcreteAssignment, updates: dict[str, Any]
) -> dict[str, Any]:
    result = dict(assignment.state)
    result.update(updates)
    return result


def _diff(
    assignment: ConcreteAssignment, updates: dict[str, Any]
) -> StructuredStateDiff:
    changes = []
    for field, after in sorted(updates.items()):
        before = assignment.value("state", field)
        if before != after:
            changes.append(StateChange("record", "state", field, before, after))
    return StructuredStateDiff(tuple(changes))


def _coverage_assignments(
    case: HiddenSymbolicCase, budget: int
) -> tuple[ConcreteAssignment, ...]:
    assignments = tuple(enumerate_assignments(case.analyzer_input.schema))
    universe = {
        (field.source, field.name, value)
        for field in case.analyzer_input.schema.fields
        for value in field.values
    }
    covered: set[tuple[str, str, Any]] = set()
    selected = []
    remaining = list(assignments)
    while remaining and len(selected) < budget:
        best = max(
            remaining,
            key=lambda assignment: (
                len(
                    {
                        (source, name, value)
                        for source, values in (
                            ("input", assignment.inputs),
                            ("state", assignment.state),
                        )
                        for name, value in values
                    }
                    - covered
                ),
                canonical_json(assignment.to_dict()),
            ),
        )
        selected.append(best)
        covered.update(
            (source, name, value)
            for source, values in (("input", best.inputs), ("state", best.state))
            for name, value in values
        )
        remaining.remove(best)
        if covered == universe:
            break
    return tuple(selected)


def _random_assignments(
    case: HiddenSymbolicCase, budget: int
) -> tuple[ConcreteAssignment, ...]:
    return tuple(
        sorted(
            enumerate_assignments(case.analyzer_input.schema),
            key=lambda item: canonical_digest(
                {"case": case.public_case_id, "assignment": item.to_dict()}
            ),
        )[:budget]
    )


def _collect_evidence(
    case: HiddenSymbolicCase,
    replay: SymbolicSandboxReplay,
    assignments: Sequence[ConcreteAssignment],
) -> tuple[AnalyzerEvidence, ...]:
    values = []
    for assignment in assignments:
        result = replay.replay(case.analyzer_input.case_handle, assignment)
        values.append(
            AnalyzerEvidence(
                assignment=assignment,
                events=result.events,
                state_diff=result.state_diff,
                observed_version_digest=result.observed_version_digest,
            )
        )
    return tuple(values)


def _trace_table(
    evidence: Sequence[AnalyzerEvidence],
) -> tuple[tuple[str, tuple[ConcreteEffect, ...], tuple[tuple[str, Any], ...]], ...]:
    rows = []
    for item in evidence:
        updates = tuple(
            (change.field, change.after) for change in item.state_diff.changes
        )
        rows.append((item.assignment.digest, item.events, updates))
    return tuple(sorted(rows, key=lambda item: item[0]))


def _run_method(
    method: SymbolicMethod,
    case: HiddenSymbolicCase,
    *,
    replay_budget: int,
    solver_timeout_ms: int,
) -> MethodExecution:
    started = time.perf_counter_ns()
    if method == SymbolicMethod.DENY_ALL:
        return MethodExecution(
            method,
            None,
            ContractStatus.UNKNOWN,
            ShieldStatus.UNKNOWN,
            CertificateStatus.NOT_ISSUED,
            None,
            (),
            (),
            False,
            (time.perf_counter_ns() - started) / 1_000_000,
            0,
            0,
        )
    if method == SymbolicMethod.DECLARED:
        return MethodExecution(
            method,
            case.analyzer_input.declared_contract,
            ContractStatus.UNKNOWN,
            ShieldStatus.UNKNOWN,
            CertificateStatus.NOT_ISSUED,
            None,
            (),
            (),
            False,
            (time.perf_counter_ns() - started) / 1_000_000,
            0,
            0,
        )
    if method in {SymbolicMethod.RANDOM_TABLE, SymbolicMethod.COVERAGE_TABLE}:
        assignments = (
            _random_assignments(case, replay_budget)
            if method == SymbolicMethod.RANDOM_TABLE
            else _coverage_assignments(case, replay_budget)
        )
        replay = SymbolicSandboxReplay((case,))
        evidence = _collect_evidence(case, replay, assignments)
        return MethodExecution(
            method,
            None,
            ContractStatus.INCOMPLETE,
            ShieldStatus.UNKNOWN,
            CertificateStatus.NOT_ISSUED,
            None,
            tuple(assignments),
            _trace_table(evidence),
            False,
            (time.perf_counter_ns() - started) / 1_000_000,
            len(assignments),
            0,
        )
    if method == SymbolicMethod.PASSIVE:
        assignments = _coverage_assignments(case, replay_budget)
        replay = SymbolicSandboxReplay((case,))
        evidence = _collect_evidence(case, replay, assignments)
        learned = SymbolicContractLearner(minimize=True).fit(
            tool_id=case.analyzer_input.tool_id,
            schema=case.analyzer_input.schema,
            version_digest=case.implementation.version_digest,
            limits=case.analyzer_input.grammar_limits,
            evidence=evidence,
            input_schema_digest=case.analyzer_input.declared_contract.input_schema_digest,
            state_schema_digest=case.analyzer_input.declared_contract.state_schema_digest,
            timeout_ms=solver_timeout_ms,
        )
        return MethodExecution(
            method,
            learned or case.analyzer_input.declared_contract,
            ContractStatus.UNKNOWN,
            ShieldStatus.UNKNOWN,
            CertificateStatus.NOT_ISSUED,
            None,
            tuple(assignments),
            (),
            False,
            (time.perf_counter_ns() - started) / 1_000_000,
            len(assignments),
            0,
        )
    if method in {SymbolicMethod.CEGIS_NO_MIN, SymbolicMethod.AUTHSYNTH}:
        verifier = BoundedSymbolicVerifier((case,), timeout_ms=solver_timeout_ms)
        replay = SymbolicSandboxReplay((case,))
        outcome = AuthSynthSymbolicAnalyzer(
            minimize=method == SymbolicMethod.AUTHSYNTH
        ).analyze(case.analyzer_input, verifier, replay)
        return MethodExecution(
            method,
            outcome.candidate_contract,
            outcome.contract_status,
            outcome.shield_status,
            outcome.certificate_status,
            outcome,
            outcome.replayed_assignments,
            (),
            outcome.complete_claim_made,
            (time.perf_counter_ns() - started) / 1_000_000,
            outcome.replay_query_count,
            outcome.solver_call_count,
        )
    return MethodExecution(
        method,
        case.reference_contract,
        ContractStatus.VERIFIED_COMPLETE,
        ShieldStatus.SAFE,
        CertificateStatus.VALID,
        None,
        (),
        (),
        True,
        (time.perf_counter_ns() - started) / 1_000_000,
        0,
        0,
    )


def _prediction(
    execution: MethodExecution,
    assignment: ConcreteAssignment,
) -> tuple[tuple[ConcreteEffect, ...], dict[str, Any]] | None:
    if execution.contract is not None:
        return execution.contract.predict(assignment)
    table = {
        key: (events, dict(updates)) for key, events, updates in execution.trace_table
    }
    return table.get(assignment.digest)


def _shield_allows(
    execution: MethodExecution,
    case: HiddenSymbolicCase,
    assignment: ConcreteAssignment,
) -> bool:
    if execution.method == SymbolicMethod.DENY_ALL:
        return False
    if (
        execution.method in {SymbolicMethod.CEGIS_NO_MIN, SymbolicMethod.AUTHSYNTH}
        and execution.contract_status != ContractStatus.VERIFIED_COMPLETE
    ):
        return False
    predicted = _prediction(execution, assignment)
    if predicted is None:
        return False
    events, updates = predicted
    return not case.analyzer_input.trusted_safety_spec.violation_codes(
        events, _diff(assignment, updates)
    )


def _evaluate_case_method(
    case: HiddenSymbolicCase,
    method: SymbolicMethod,
    *,
    replay_budget: int,
    solver_timeout_ms: int,
) -> SymbolicEvaluationRow:
    execution = _run_method(
        method,
        case,
        replay_budget=replay_budget,
        solver_timeout_ms=solver_timeout_ms,
    )
    replayed = {item.digest for item in execution.replayed_assignments}
    unseen_actual = unseen_predicted = unseen_tp = unseen_fp = 0
    guard_tp = guard_fp = guard_fn = guard_tn = 0
    guard_exact = True
    binding = {
        name: [0, 0]
        for name in ("tenant", "resource", "destination", "amount", "phase")
    }
    state_correct = state_total = 0
    forbidden = forbidden_accepted = safe = safe_allowed = unknown_reject = 0
    exact_full = True
    # 固定签名全集，才能计入双方均不触发时的 TN；不能使用每次正例的并集。
    guard_signatures = {
        effect.signature
        for transition in case.implementation.transitions
        for effect in transition.all_effects()
    }
    if execution.contract is not None:
        guard_signatures.update(
            clause.effect.signature for clause in execution.contract.clauses
        )
    else:
        guard_signatures.update(
            event.signature
            for _, events, _ in execution.trace_table
            for event in events
        )
    for assignment in enumerate_assignments(case.analyzer_input.schema):
        actual_events, actual_updates = case.implementation.execute(assignment)
        predicted = _prediction(execution, assignment)
        predicted_events, predicted_updates = (
            predicted if predicted is not None else ((), {})
        )
        actual_keys = {item.semantic_key for item in actual_events}
        predicted_keys = {item.semantic_key for item in predicted_events}
        actual_final = _final_state(assignment, actual_updates)
        predicted_final = _final_state(assignment, predicted_updates)
        if actual_keys != predicted_keys or actual_final != predicted_final:
            exact_full = False
        is_unseen = assignment.digest not in replayed
        if is_unseen:
            unseen_actual += len(actual_keys)
            unseen_predicted += len(predicted_keys)
            unseen_tp += len(actual_keys & predicted_keys)
            unseen_fp += len(predicted_keys - actual_keys)
            actual_by_signature = {item.signature: item for item in actual_events}
            predicted_by_signature = {item.signature: item for item in predicted_events}
            for signature in guard_signatures:
                a = signature in actual_by_signature
                p = signature in predicted_by_signature
                guard_tp += int(a and p)
                guard_fp += int(p and not a)
                guard_fn += int(a and not p)
                guard_tn += int(not a and not p)
                if a != p:
                    guard_exact = False
                if a and p:
                    left = actual_by_signature[signature]
                    right = predicted_by_signature[signature]
                    for field in (
                        "tenant",
                        "resource",
                        "destination",
                        "amount",
                        "phase",
                    ):
                        binding[field][1] += 1
                        binding[field][0] += int(
                            getattr(left, field) == getattr(right, field)
                        )
            for field in case.analyzer_input.schema.state_fields:
                state_total += 1
                state_correct += int(
                    actual_final[field.name] == predicted_final[field.name]
                )
        actual_unsafe = bool(
            case.analyzer_input.trusted_safety_spec.violation_codes(
                actual_events, _diff(assignment, actual_updates)
            )
        )
        allowed = _shield_allows(execution, case, assignment)
        if actual_unsafe:
            forbidden += 1
            forbidden_accepted += int(allowed)
        else:
            safe += 1
            safe_allowed += int(allowed)
            unknown_reject += int(
                not allowed and execution.contract_status == ContractStatus.UNKNOWN
            )

    contract = execution.contract
    certificate_size = 0
    if execution.outcome and execution.outcome.certificate:
        certificate_size = len(
            canonical_json(execution.outcome.certificate.to_dict()).encode("utf-8")
        )
    drift_expected = case.mutation_category == "implementation_drift"
    predicted_patches = tuple(
        sorted(
            {
                item.patch_type
                for item in (
                    execution.outcome.patch_records if execution.outcome else ()
                )
            }
        )
    )
    expected_atoms = set(case.omission_atoms)
    if case.implementation.proof_obstacle is not None:
        expected_atoms.add(
            PatchAtom(
                "unsupported_region",
                "tool",
                case.analyzer_input.tool_id,
                case.implementation.proof_obstacle,
            )
        )
    patch_records = execution.outcome.patch_records if execution.outcome else ()
    predicted_atom_ids = {
        record.atom.digest
        if record.atom is not None
        else canonical_digest(
            {
                "unlocated_patch_type": record.patch_type,
                "evidence_digest": record.target_digest,
            }
        )
        for record in patch_records
    }
    old_reuse = False
    if drift_expected and execution.outcome is not None:
        old_id = case.analyzer_input.existing_certificate_id
        old_reuse = bool(
            old_id
            and (
                old_id not in execution.outcome.invalidated_certificate_ids
                or (
                    execution.outcome.certificate is not None
                    and execution.outcome.certificate.certificate_id == old_id
                )
            )
        )
    return SymbolicEvaluationRow(
        method=method.value,
        case_id=case.public_case_id,
        split=case.split,
        domain=case.domain,
        mutation_category=case.mutation_category,
        expected_unknown=case.expected_unknown,
        domain_assignment_count=case.analyzer_input.schema.cardinality,
        replay_query_count=execution.query_count,
        solver_call_count=execution.solver_calls,
        complete_claim_made=execution.complete_claim_made,
        false_verified_complete=execution.complete_claim_made and not exact_full,
        analysis_unknown=execution.contract_status == ContractStatus.UNKNOWN,
        unseen_actual_effect_count=unseen_actual,
        unseen_predicted_effect_count=unseen_predicted,
        unseen_effect_true_positive_count=unseen_tp,
        unseen_effect_false_positive_count=unseen_fp,
        guard_true_positive_count=guard_tp,
        guard_false_positive_count=guard_fp,
        guard_false_negative_count=guard_fn,
        guard_true_negative_count=guard_tn,
        guard_exact_match=guard_exact,
        tenant_binding_correct=binding["tenant"][0],
        tenant_binding_total=binding["tenant"][1],
        resource_binding_correct=binding["resource"][0],
        resource_binding_total=binding["resource"][1],
        destination_binding_correct=binding["destination"][0],
        destination_binding_total=binding["destination"][1],
        amount_binding_correct=binding["amount"][0],
        amount_binding_total=binding["amount"][1],
        phase_binding_correct=binding["phase"][0],
        phase_binding_total=binding["phase"][1],
        state_binding_correct=state_correct,
        state_binding_total=state_total,
        forbidden_assignment_count=forbidden,
        forbidden_accepted_count=forbidden_accepted,
        safe_assignment_count=safe,
        safe_allowed_count=safe_allowed,
        unknown_induced_rejection_count=unknown_reject,
        drift_expected=drift_expected,
        drift_detected=bool(
            execution.outcome and execution.outcome.invalidated_certificate_ids
        ),
        old_certificate_reused_after_drift=old_reuse,
        converged=execution.contract_status == ContractStatus.VERIFIED_COMPLETE,
        predicted_patch_types=predicted_patches,
        expected_patch_types=tuple(
            sorted({atom.patch_type for atom in expected_atoms})
        ),
        reason_codes=(execution.outcome.reason_codes if execution.outcome else ()),
        contract_digest="" if contract is None else contract.digest,
        certificate_id=(
            execution.outcome.certificate.certificate_id
            if execution.outcome and execution.outcome.certificate
            else None
        ),
        iteration_digests=(
            tuple(
                canonical_digest(item.to_dict())
                for item in execution.outcome.iterations
            )
            if execution.outcome
            else ()
        ),
        analysis_time_ms=execution.analysis_time_ms,
        synthesis_time_ms=(
            execution.outcome.synthesis_time_ms if execution.outcome else 0.0
        ),
        verification_time_ms=(
            execution.outcome.verification_time_ms if execution.outcome else 0.0
        ),
        contract_clause_count=0 if contract is None else contract.clause_count,
        contract_literal_count=0 if contract is None else contract.literal_count,
        contract_ast_node_count=0 if contract is None else contract.ast_node_count,
        contract_size_bytes=0
        if contract is None
        else len(canonical_json(contract.to_dict()).encode("utf-8")),
        certificate_size_bytes=certificate_size,
        predicted_patch_atom_digests=tuple(sorted(predicted_atom_ids)),
        expected_patch_atom_digests=tuple(
            sorted(atom.digest for atom in expected_atoms)
        ),
        discovered_omission_atom_digests=tuple(
            sorted(
                atom.digest
                for atom in (
                    execution.outcome.discovered_atoms if execution.outcome else ()
                )
            )
        ),
        expected_omission_atom_digests=tuple(
            sorted(atom.digest for atom in case.omission_atoms)
        ),
        unlocated_patch_record_count=sum(
            record.atom is None for record in patch_records
        ),
    )


def aggregate_metrics(
    rows: Sequence[SymbolicEvaluationRow],
) -> tuple[dict[str, Any], ...]:
    results = []
    for method in METHODS:
        selected = [item for item in rows if item.method == method.value]
        if not selected:
            continue

        def ratio(numerator: int, denominator: int, empty: float = 1.0) -> float:
            return empty if denominator == 0 else numerator / denominator

        actual = sum(item.unseen_actual_effect_count for item in selected)
        predicted = sum(item.unseen_predicted_effect_count for item in selected)
        tp = sum(item.unseen_effect_true_positive_count for item in selected)
        guard_tp = sum(item.guard_true_positive_count for item in selected)
        guard_fp = sum(item.guard_false_positive_count for item in selected)
        guard_fn = sum(item.guard_false_negative_count for item in selected)
        forbidden = sum(item.forbidden_assignment_count for item in selected)
        safe = sum(item.safe_assignment_count for item in selected)
        expected_patches = sum(len(set(item.expected_patch_types)) for item in selected)
        predicted_patches = sum(
            len(set(item.predicted_patch_types)) for item in selected
        )
        patch_tp = sum(
            len(set(item.expected_patch_types) & set(item.predicted_patch_types))
            for item in selected
        )
        expected_atom_count = sum(
            len(set(r.expected_patch_atom_digests)) for r in selected
        )
        predicted_atom_count = sum(
            len(set(r.predicted_patch_atom_digests)) for r in selected
        )
        atom_tp = sum(
            len(
                set(r.expected_patch_atom_digests) & set(r.predicted_patch_atom_digests)
            )
            for r in selected
        )
        omission_count = sum(
            len(set(r.expected_omission_atom_digests)) for r in selected
        )
        discovered_count = sum(
            len(
                set(r.expected_omission_atom_digests)
                & set(r.discovered_omission_atom_digests)
            )
            for r in selected
        )
        drift_expected = sum(item.drift_expected for item in selected)
        result = {
            "method": method.value,
            "case_count": len(selected),
            "false_verified_complete_count": sum(
                item.false_verified_complete for item in selected
            ),
            "complete_claim_count": sum(item.complete_claim_made for item in selected),
            "false_verified_complete_rate": ratio(
                sum(item.false_verified_complete for item in selected),
                sum(item.complete_claim_made for item in selected),
                0.0,
            ),
            "unseen_effect_recall": ratio(tp, actual),
            "unseen_effect_precision": ratio(tp, predicted),
            "guard_precision": ratio(guard_tp, guard_tp + guard_fp),
            "guard_recall": ratio(guard_tp, guard_tp + guard_fn),
            "guard_accuracy": ratio(
                guard_tp + sum(item.guard_true_negative_count for item in selected),
                guard_tp
                + guard_fp
                + guard_fn
                + sum(item.guard_true_negative_count for item in selected),
            ),
            "guard_exact_match_rate": ratio(
                sum(item.guard_exact_match for item in selected), len(selected)
            ),
            "field_binding_accuracy": ratio(
                sum(
                    item.tenant_binding_correct
                    + item.resource_binding_correct
                    + item.destination_binding_correct
                    + item.amount_binding_correct
                    + item.phase_binding_correct
                    for item in selected
                ),
                sum(
                    item.tenant_binding_total
                    + item.resource_binding_total
                    + item.destination_binding_total
                    + item.amount_binding_total
                    + item.phase_binding_total
                    for item in selected
                ),
            ),
            "state_update_binding_accuracy": ratio(
                sum(item.state_binding_correct for item in selected),
                sum(item.state_binding_total for item in selected),
            ),
            "contract_overapproximation_ratio": ratio(
                sum(item.unseen_effect_false_positive_count for item in selected),
                predicted,
                0.0,
            ),
            "forbidden_trace_acceptance_rate": ratio(
                sum(item.forbidden_accepted_count for item in selected), forbidden, 0.0
            ),
            "safe_utility": ratio(
                sum(item.safe_allowed_count for item in selected), safe
            ),
            "maximal_permissiveness_ratio": ratio(
                sum(item.safe_allowed_count for item in selected), safe
            ),
            "replay_query_reduction": 1.0
            - ratio(
                sum(item.replay_query_count for item in selected),
                sum(item.domain_assignment_count for item in selected),
                0.0,
            ),
            "convergence_rate": ratio(
                sum(item.converged for item in selected), len(selected)
            ),
            "unknown_rate": ratio(
                sum(item.analysis_unknown for item in selected), len(selected), 0.0
            ),
            "unknown_expected_precision": ratio(
                sum(
                    item.analysis_unknown and item.expected_unknown for item in selected
                ),
                sum(item.analysis_unknown for item in selected),
            ),
            "unknown_induced_rejection_count": sum(
                item.unknown_induced_rejection_count for item in selected
            ),
            "drift_detection_recall": ratio(
                sum(item.drift_detected for item in selected), drift_expected
            ),
            "old_certificate_reuse_after_drift_count": sum(
                item.old_certificate_reused_after_drift for item in selected
            ),
            "patch_type_precision": ratio(patch_tp, predicted_patches),
            "patch_type_recall": ratio(patch_tp, expected_patches),
            "patch_atom_precision": ratio(atom_tp, predicted_atom_count),
            "patch_atom_recall": ratio(atom_tp, expected_atom_count),
            "predicted_patch_atom_count": predicted_atom_count,
            "expected_patch_atom_count": expected_atom_count,
            "true_positive_patch_atom_count": atom_tp,
            "unlocated_patch_record_count": sum(
                r.unlocated_patch_record_count for r in selected
            ),
            "omission_atom_discovery_recall": ratio(discovered_count, omission_count),
            "expected_omission_atom_count": omission_count,
            "discovered_omission_atom_count": discovered_count,
            "exact_patch_type_set_rate": ratio(
                sum(
                    set(r.predicted_patch_types) == set(r.expected_patch_types)
                    for r in selected
                ),
                len(selected),
            ),
            "exact_patch_set_rate": ratio(
                sum(
                    set(item.predicted_patch_atom_digests)
                    == set(item.expected_patch_atom_digests)
                    for item in selected
                ),
                len(selected),
            ),
            "analysis_latency_mean_ms": ratio(
                int(sum(item.analysis_time_ms for item in selected) * 1_000_000),
                len(selected) * 1_000_000,
                0.0,
            ),
            "synthesis_latency_mean_ms": ratio(
                int(sum(item.synthesis_time_ms for item in selected) * 1_000_000),
                len(selected) * 1_000_000,
                0.0,
            ),
            "verification_latency_mean_ms": ratio(
                int(sum(item.verification_time_ms for item in selected) * 1_000_000),
                len(selected) * 1_000_000,
                0.0,
            ),
            "solver_call_count": sum(item.solver_call_count for item in selected),
            "replay_query_count": sum(item.replay_query_count for item in selected),
            "contract_clause_mean": ratio(
                sum(item.contract_clause_count for item in selected), len(selected), 0.0
            ),
            "contract_literal_mean": ratio(
                sum(item.contract_literal_count for item in selected),
                len(selected),
                0.0,
            ),
            "contract_ast_node_mean": ratio(
                sum(item.contract_ast_node_count for item in selected),
                len(selected),
                0.0,
            ),
            "contract_size_mean_bytes": ratio(
                sum(item.contract_size_bytes for item in selected), len(selected), 0.0
            ),
            "certificate_size_mean_bytes": ratio(
                sum(item.certificate_size_bytes for item in selected),
                len(selected),
                0.0,
            ),
        }
        results.append(result)
    return tuple(results)


def evaluate_cases(
    cases: Sequence[HiddenSymbolicCase],
    *,
    replay_budget: int,
    solver_timeout_ms: int,
    methods: Sequence[SymbolicMethod] = METHODS,
) -> dict[str, Any]:
    rows = tuple(
        _evaluate_case_method(
            case,
            method,
            replay_budget=replay_budget,
            solver_timeout_ms=solver_timeout_ms,
        )
        for method in methods
        for case in cases
    )
    return {
        "rows": rows,
        "metrics": aggregate_metrics(rows),
        "method_case_counts": dict(
            sorted(Counter(item.method for item in rows).items())
        ),
    }
