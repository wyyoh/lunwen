from __future__ import annotations

from dataclasses import replace

from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_analyzer import AuthSynthSymbolicAnalyzer
from keyed_gram.authsynth_symbolic_shared import (
    ContractStatus,
    VerificationKind,
    VerificationResult,
)
from keyed_gram.authsynth_symbolic_verifier.equivalence import BoundedSymbolicVerifier
from keyed_gram.authsynth_symbolic_verifier.replay_service import SymbolicSandboxReplay


def _analyze(category: str):
    case = case_for(category)
    outcome = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input,
        BoundedSymbolicVerifier((case,), timeout_ms=3000),
        SymbolicSandboxReplay((case,)),
    )
    return case, outcome


def test_real_cegis_changes_contract_and_shield_until_equivalent() -> None:
    case, outcome = _analyze("multi_branch_effect")
    assert outcome.contract_status == ContractStatus.VERIFIED_COMPLETE
    assert 1 <= outcome.replay_query_count < case.analyzer_input.schema.cardinality
    assert len(outcome.iterations) >= 2
    assert any(
        item.contract_before_digest != item.contract_after_digest
        for item in outcome.iterations
    )
    assert any(
        item.shield_before_digest != item.shield_after_digest
        for item in outcome.iterations
    )
    counterexamples = [
        item.counterexample_digest
        for item in outcome.iterations
        if item.counterexample_digest is not None
    ]
    assert len(counterexamples) == len(set(counterexamples))


def test_query_budget_exhaustion_is_unknown_not_complete() -> None:
    case = case_for("multi_branch_effect")
    limited = replace(
        case,
        analyzer_input=replace(case.analyzer_input, replay_budget=1, max_iterations=5),
    )
    outcome = AuthSynthSymbolicAnalyzer().analyze(
        limited.analyzer_input,
        BoundedSymbolicVerifier((limited,), timeout_ms=3000),
        SymbolicSandboxReplay((limited,)),
    )
    assert outcome.contract_status == ContractStatus.UNKNOWN
    assert "replay_budget_exhausted" in outcome.reason_codes
    assert outcome.certificate is None


def test_solver_unknown_is_fail_closed() -> None:
    case = case_for("hidden_guarded_effect")

    class UnknownVerifier:
        solver_name = "fake"
        solver_version = "1"

        def check(self, _handle, _contract):
            return VerificationResult(
                VerificationKind.UNKNOWN, reason_code="solver_timeout_or_unknown"
            )

    outcome = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input, UnknownVerifier(), SymbolicSandboxReplay((case,))
    )
    assert outcome.contract_status == ContractStatus.UNKNOWN
    assert outcome.certificate is None
    assert "solver_timeout_or_unknown" in outcome.reason_codes


def test_version_drift_invalidates_old_certificate_before_recertification() -> None:
    case, outcome = _analyze("implementation_drift")
    old = case.analyzer_input.existing_certificate_id
    assert old in outcome.invalidated_certificate_ids
    assert outcome.certificate is not None
    assert outcome.certificate.certificate_id != old
    assert (
        outcome.certificate.implementation_version_digest
        == case.implementation.version_digest
    )
