from __future__ import annotations

from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_verifier.evaluator import (
    SymbolicMethod,
    evaluate_cases,
)
from keyed_gram.stage_f2c_metrics import (
    patch_metrics,
    query_efficiency,
    readiness_gate,
    selected_metrics,
)


def _thresholds() -> dict[str, float | int]:
    return {
        "maximum_false_verified_complete_count": 0,
        "maximum_forbidden_trace_acceptance_rate": 0.0,
        "minimum_unseen_effect_recall": 0.95,
        "minimum_unseen_effect_precision": 0.90,
        "minimum_guard_precision": 0.95,
        "minimum_guard_recall": 0.95,
        "minimum_field_binding_accuracy": 0.95,
        "minimum_state_update_binding_accuracy": 0.90,
        "maximum_contract_overapproximation_ratio": 0.10,
        "minimum_safe_utility": 0.90,
        "minimum_maximal_permissiveness_ratio": 0.90,
        "minimum_replay_query_reduction": 0.70,
        "minimum_convergence_rate": 0.85,
        "maximum_unknown_rate": 0.20,
        "minimum_drift_detection_recall": 1.0,
        "maximum_old_certificate_reuse_after_drift_count": 0,
    }


def test_unseen_metrics_exclude_replayed_inputs_and_query_reduction_is_exact() -> None:
    case = case_for("multi_branch_effect")
    bundle = evaluate_cases(
        (case,),
        replay_budget=16,
        solver_timeout_ms=3000,
        methods=(SymbolicMethod.AUTHSYNTH,),
    )
    row = bundle["rows"][0]
    query = query_efficiency((row,))[0]
    assert row.unseen_actual_effect_count > 0
    assert (
        query["replay_query_reduction"]
        == 1 - row.replay_query_count / row.domain_assignment_count
    )


def test_unknown_does_not_count_as_complete() -> None:
    case = case_for("exact_declared_control")
    bundle = evaluate_cases(
        (case,),
        replay_budget=16,
        solver_timeout_ms=3000,
        methods=(SymbolicMethod.AUTHSYNTH,),
    )
    row = bundle["rows"][0]
    assert row.analysis_unknown
    assert not row.complete_claim_made


def test_patch_precision_is_computed_against_omission_atoms_not_defined_as_one() -> (
    None
):
    cases = (case_for("hidden_guarded_effect"), case_for("parameter_role_alias"))
    bundle = evaluate_cases(
        cases,
        replay_budget=16,
        solver_timeout_ms=3000,
        methods=(SymbolicMethod.AUTHSYNTH,),
    )
    rows = patch_metrics(bundle["rows"])
    assert rows
    assert all("true_positive_count" in row for row in rows)
    aggregate = selected_metrics(bundle["metrics"])
    assert aggregate["patch_type_precision"] == (
        sum(row["true_positive_count"] for row in rows)
        / sum(row["predicted_count"] for row in rows)
    )


def test_readiness_gate_does_not_hide_safety_failures() -> None:
    metrics = {
        "false_verified_complete_count": 1,
        "forbidden_trace_acceptance_rate": 0.0,
        "unseen_effect_recall": 1.0,
        "unseen_effect_precision": 1.0,
        "guard_precision": 1.0,
        "guard_recall": 1.0,
        "field_binding_accuracy": 1.0,
        "state_update_binding_accuracy": 1.0,
        "contract_overapproximation_ratio": 0.0,
        "safe_utility": 1.0,
        "maximal_permissiveness_ratio": 1.0,
        "replay_query_reduction": 1.0,
        "convergence_rate": 1.0,
        "unknown_rate": 0.0,
        "drift_detection_recall": 1.0,
        "old_certificate_reuse_after_drift_count": 0,
    }
    gate = readiness_gate(metrics, _thresholds())
    assert gate["false_verified_complete"] is False
