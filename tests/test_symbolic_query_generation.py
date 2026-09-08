from __future__ import annotations

from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_analyzer.query_generator import (
    request_counterexample,
)
from keyed_gram.authsynth_symbolic_shared import VerificationKind
from keyed_gram.authsynth_symbolic_verifier.equivalence import BoundedSymbolicVerifier
from keyed_gram.authsynth_symbolic_verifier.replay_service import SymbolicSandboxReplay


def test_verifier_generates_assignment_not_analyzer_catalog() -> None:
    case = case_for("hidden_guarded_effect")
    verifier = BoundedSymbolicVerifier((case,), timeout_ms=3000)
    first = request_counterexample(
        verifier, case.analyzer_input.case_handle, case.analyzer_input.declared_contract
    )
    assert first.kind == VerificationKind.MISSING_EFFECT
    assert first.assignment is not None
    replay = SymbolicSandboxReplay((case,)).replay(
        case.analyzer_input.case_handle, first.assignment
    )
    assert replay.assignment == first.assignment
    assert "changes" in replay.state_diff.to_dict()
    assert replay.events


def test_second_verifier_query_changes_after_candidate_refinement() -> None:
    from keyed_gram.authsynth_symbolic_analyzer import AuthSynthSymbolicAnalyzer

    case = case_for("multi_branch_effect")
    outcome = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input,
        BoundedSymbolicVerifier((case,), timeout_ms=3000),
        SymbolicSandboxReplay((case,)),
    )
    assignments = [item.digest for item in outcome.replayed_assignments]
    assert len(assignments) >= 2
    assert len(assignments) == len(set(assignments))
