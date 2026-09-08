"""AuthSynth-Symbolic：query-generating CEGIS 主循环。"""

from __future__ import annotations

import time
from typing import Any

from keyed_gram.authsynth_symbolic_shared import (
    AnalyzerEvidence,
    BoundedCompletenessCertificate,
    CertificateStatus,
    ContractStatus,
    PatchRecord,
    ShieldStatus,
    SymbolicAnalyzerInput,
    SymbolicEffectContract,
    SymbolicReplayClient,
    VerificationKind,
    canonical_digest,
)

from .learner import SymbolicContractLearner
from .models import SymbolicAnalysisOutcome, SymbolicRefinementIteration
from .query_generator import request_counterexample
from .shield import shield_digest

_PATCH_TYPES = {
    VerificationKind.MISSING_EFFECT: "effect_patch",
    VerificationKind.SPURIOUS_EFFECT: "guard_patch",
    VerificationKind.FIELD_BINDING_MISMATCH: "field_binding_patch",
    VerificationKind.STATE_UPDATE_MISMATCH: "state_update_patch",
    VerificationKind.VERSION_MISMATCH: "version_invalidation",
    VerificationKind.UNKNOWN: "unsupported_region",
}


def _derive_patch_types(
    candidate: SymbolicEffectContract,
    observed: Any,
) -> tuple[str, ...]:
    predicted_events, predicted_updates = candidate.predict(observed.assignment)
    actual_by_signature = {item.signature: item for item in observed.events}
    predicted_by_signature = {item.signature: item for item in predicted_events}
    declared_signatures = {item.effect.signature for item in candidate.clauses}
    values: set[str] = set()
    for signature, actual in actual_by_signature.items():
        predicted = predicted_by_signature.get(signature)
        if predicted is None:
            values.add(
                "guard_patch" if signature in declared_signatures else "effect_patch"
            )
        elif predicted.semantic_key != actual.semantic_key:
            values.add("field_binding_patch")
    if set(predicted_by_signature) - set(actual_by_signature):
        values.add("guard_patch")
    expected_state = dict(observed.assignment.state)
    for change in observed.state_diff.changes:
        expected_state[change.field] = change.after
    predicted_state = dict(observed.assignment.state)
    predicted_state.update(predicted_updates)
    if predicted_state != expected_state:
        values.add("state_update_patch")
    return tuple(sorted(values))


def _merge_initial(case: SymbolicAnalyzerInput) -> SymbolicEffectContract:
    contracts = (case.declared_contract, *case.static_hypotheses)
    return SymbolicEffectContract(
        tool_id=case.tool_id,
        implementation_version_digest=case.expected_version_digest,
        input_schema_digest=case.declared_contract.input_schema_digest,
        state_schema_digest=case.declared_contract.state_schema_digest,
        clauses=tuple(clause for item in contracts for clause in item.clauses),
        state_updates=tuple(
            clause for item in contracts for clause in item.state_updates
        ),
        unsupported_regions=tuple(
            region for item in contracts for region in item.unsupported_regions
        ),
    )


class AuthSynthSymbolicAnalyzer:
    """只依赖窄 verifier/replay protocols 的有界符号分析器。"""

    def __init__(self, *, minimize: bool = True) -> None:
        self._learner = SymbolicContractLearner(minimize=minimize)

    def analyze(
        self,
        case: SymbolicAnalyzerInput,
        verifier: Any,
        replay: SymbolicReplayClient,
    ) -> SymbolicAnalysisOutcome:
        candidate = _merge_initial(case)
        candidate.validate(case.schema, case.grammar_limits)
        active_version = case.expected_version_digest
        evidence: list[AnalyzerEvidence] = []
        replayed = []
        seen_assignments: set[str] = set()
        iterations: list[SymbolicRefinementIteration] = []
        patches: list[PatchRecord] = []
        invalidated: list[str] = []
        reason_codes: set[str] = set()
        solver_calls = 0
        synthesis_ms = 0.0
        verification_ms = 0.0
        final_status = ContractStatus.UNKNOWN
        final_shield = ShieldStatus.UNKNOWN
        final_certificate_status = CertificateStatus.NOT_ISSUED
        certificate: BoundedCompletenessCertificate | None = None

        for iteration_number in range(1, case.max_iterations + 1):
            before = candidate.digest
            before_shield = shield_digest(
                candidate, case.trusted_safety_spec, ContractStatus.UNKNOWN
            )
            started_verify = time.perf_counter_ns()
            result = request_counterexample(verifier, case.case_handle, candidate)
            verification_ms += (time.perf_counter_ns() - started_verify) / 1_000_000
            solver_calls += 1

            if result.kind == VerificationKind.EQUIVALENT:
                if result.observed_version_digest != active_version:
                    reason_codes.add("equivalence_version_unbound_or_mismatch")
                    break
                if candidate.unsupported_regions:
                    reason_codes.add("candidate_has_unsupported_region")
                    break
                certificate = BoundedCompletenessCertificate.issue(
                    contract_digest=candidate.digest,
                    tool_id=case.tool_id,
                    implementation_version_digest=active_version,
                    input_schema_digest=candidate.input_schema_digest,
                    state_schema_digest=candidate.state_schema_digest,
                    bounded_domain_cardinality=case.schema.cardinality,
                    solver_name=verifier.solver_name,
                    solver_version=verifier.solver_version,
                )
                final_status = ContractStatus.VERIFIED_COMPLETE
                final_shield = ShieldStatus.SAFE
                final_certificate_status = CertificateStatus.VALID
                reason_codes.add("bounded_domain_equivalent")
                break

            if result.kind == VerificationKind.UNKNOWN:
                patch = PatchRecord(
                    "unsupported_region",
                    canonical_digest(
                        {"reason": result.reason_code or "verifier_unknown"}
                    ),
                )
                patches.append(patch)
                reason_codes.add(result.reason_code or "verifier_unknown")
                iterations.append(
                    SymbolicRefinementIteration(
                        iteration=iteration_number,
                        verification_kind=result.kind,
                        counterexample_digest=None,
                        replay_result_digest=None,
                        patch=patch,
                        contract_before_digest=before,
                        contract_after_digest=before,
                        shield_before_digest=before_shield,
                        shield_after_digest=before_shield,
                        remaining_replay_budget=case.replay_budget - len(replayed),
                    )
                )
                break

            if result.kind == VerificationKind.VERSION_MISMATCH:
                observed = result.observed_version_digest
                assert observed is not None
                old_id = case.existing_certificate_id
                if old_id is not None and old_id not in invalidated:
                    invalidated.append(old_id)
                patch = PatchRecord(
                    "version_invalidation",
                    canonical_digest({"old": active_version, "new": observed}),
                )
                patches.append(patch)
                active_version = observed
                evidence.clear()
                # 漂移清除旧证据，但不能恢复已经消耗的总 replay 预算。
                seen_assignments.clear()
                candidate = _merge_initial(case).with_version(observed)
                after_shield = shield_digest(
                    candidate, case.trusted_safety_spec, ContractStatus.UNKNOWN
                )
                iterations.append(
                    SymbolicRefinementIteration(
                        iteration=iteration_number,
                        verification_kind=result.kind,
                        counterexample_digest=None,
                        replay_result_digest=None,
                        patch=patch,
                        contract_before_digest=before,
                        contract_after_digest=candidate.digest,
                        shield_before_digest=before_shield,
                        shield_after_digest=after_shield,
                        remaining_replay_budget=case.replay_budget - len(replayed),
                    )
                )
                reason_codes.add("old_certificate_invalidated")
                continue

            assignment = result.assignment
            if assignment is None or result.counterexample_digest is None:
                reason_codes.add("malformed_verifier_result")
                break
            if assignment.digest in seen_assignments:
                reason_codes.add("repeated_counterexample_without_progress")
                break
            if len(replayed) >= case.replay_budget:
                reason_codes.add("replay_budget_exhausted")
                break
            case.schema.validate_assignment(assignment)
            observed = replay.replay(case.case_handle, assignment)
            replayed.append(assignment)
            if observed.assignment != assignment:
                reason_codes.add("replay_assignment_mismatch")
                break
            if observed.observed_version_digest != active_version:
                reason_codes.add("replay_version_mismatch")
                break
            if observed.exit_status != "ok":
                reason_codes.add("replay_failed")
                break
            if not observed.bounded_quiescence_reached:
                reason_codes.add("bounded_quiescence_not_reached")
                break
            if observed.instrumentation_coverage < 1.0:
                reason_codes.add("instrumentation_coverage_incomplete")
                break
            seen_assignments.add(assignment.digest)
            evidence.append(
                AnalyzerEvidence(
                    assignment=assignment,
                    events=observed.events,
                    state_diff=observed.state_diff,
                    observed_version_digest=observed.observed_version_digest,
                )
            )
            started_synthesis = time.perf_counter_ns()
            learned = self._learner.fit(
                tool_id=case.tool_id,
                schema=case.schema,
                version_digest=active_version,
                limits=case.grammar_limits,
                evidence=evidence,
                input_schema_digest=candidate.input_schema_digest,
                state_schema_digest=candidate.state_schema_digest,
                timeout_ms=case.solver_timeout_ms,
            )
            synthesis_ms += (time.perf_counter_ns() - started_synthesis) / 1_000_000
            derived = _derive_patch_types(candidate, observed)
            patch_types = derived or (_PATCH_TYPES[result.kind],)
            iteration_patches = tuple(
                PatchRecord(item, result.counterexample_digest) for item in patch_types
            )
            patches.extend(iteration_patches)
            patch = iteration_patches[0]
            if learned is None:
                reason_codes.add(
                    self._learner.last_failure_reason or "contract_grammar_insufficient"
                )
                iterations.append(
                    SymbolicRefinementIteration(
                        iteration=iteration_number,
                        verification_kind=result.kind,
                        counterexample_digest=result.counterexample_digest,
                        replay_result_digest=observed.digest,
                        patch=patch,
                        contract_before_digest=before,
                        contract_after_digest=before,
                        shield_before_digest=before_shield,
                        shield_after_digest=before_shield,
                        remaining_replay_budget=case.replay_budget - len(replayed),
                    )
                )
                break
            candidate = learned
            after_shield = shield_digest(
                candidate, case.trusted_safety_spec, ContractStatus.UNKNOWN
            )
            iterations.append(
                SymbolicRefinementIteration(
                    iteration=iteration_number,
                    verification_kind=result.kind,
                    counterexample_digest=result.counterexample_digest,
                    replay_result_digest=observed.digest,
                    patch=patch,
                    contract_before_digest=before,
                    contract_after_digest=candidate.digest,
                    shield_before_digest=before_shield,
                    shield_after_digest=after_shield,
                    remaining_replay_budget=case.replay_budget - len(replayed),
                )
            )
        else:
            reason_codes.add("iteration_budget_exhausted")

        if final_status != ContractStatus.VERIFIED_COMPLETE:
            final_status = ContractStatus.UNKNOWN
            final_shield = ShieldStatus.UNKNOWN
            final_certificate_status = CertificateStatus.NOT_ISSUED
        return SymbolicAnalysisOutcome(
            public_case_id=case.public_case_id,
            contract_status=final_status,
            shield_status=final_shield,
            certificate_status=final_certificate_status,
            candidate_contract=candidate,
            certificate=certificate,
            invalidated_certificate_ids=tuple(invalidated),
            evidence=tuple(evidence),
            replayed_assignments=tuple(replayed),
            iterations=tuple(iterations),
            patch_records=tuple(patches),
            solver_call_count=solver_calls,
            replay_query_count=len(replayed),
            reason_codes=tuple(sorted(reason_codes)),
            synthesis_time_ms=synthesis_ms,
            verification_time_ms=verification_ms,
        )
