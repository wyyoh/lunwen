from __future__ import annotations

from keyed_gram.stage_f2_metrics import (
    SECURITY_COUNT_FIELDS,
    aggregate_variant_rows,
    security_violation_matrix,
)


def row(**changes):
    value = {
        "split": "development",
        "variant": "A1",
        "oracle_decision": "allow",
        "compiler_outcome": "allow",
        "exact_execution": 1,
        "recoverable_safe": 1,
        "authority_preserved": 1,
        "unnecessary_rejection": 0,
        "confirmation_required": 0,
        "correct_confirmation": 0,
        "compile_latency_ms": 1.0,
        "verify_latency_ms": 0.5,
        "capability_size_bytes": 1200,
        "witness_size_bytes": 600,
        "workflow_state_size_bytes": 0,
        "attack_category": "semantic_ambiguity",
    }
    value.update({field: 0 for field in SECURITY_COUNT_FIELDS})
    value.update(changes)
    return value


def test_safe_utility_and_recoverable_utility_have_explicit_allow_denominator():
    metrics = aggregate_variant_rows(
        [
            row(),
            row(compiler_outcome="needs_explicit_confirmation", exact_execution=0,
                recoverable_safe=1, authority_preserved=0,
                confirmation_required=1, correct_confirmation=1),
            row(oracle_decision="deny", compiler_outcome="deny",
                exact_execution=0, recoverable_safe=0, authority_preserved=0),
        ]
    )
    assert metrics["safe_utility"] == 0.5
    assert metrics["recoverable_safe_utility"] == 1.0
    assert metrics["deny_precision"] == 1.0
    assert metrics["deny_recall"] == 1.0


def test_deny_all_is_not_mistaken_for_useful_security():
    metrics = aggregate_variant_rows(
        [
            row(compiler_outcome="deny", exact_execution=0, recoverable_safe=0,
                authority_preserved=0, unnecessary_rejection=1),
            row(oracle_decision="deny", compiler_outcome="deny",
                exact_execution=0, recoverable_safe=0, authority_preserved=0),
        ]
    )
    assert metrics["safe_utility"] == 0.0
    assert metrics["unnecessary_rejection_rate"] == 1.0
    assert metrics["deny_precision"] == 0.5


def test_security_counts_and_rates_are_aggregated():
    metrics = aggregate_variant_rows(
        [row(authority_amplification_count=1, unauthorized_action_count=1), row()]
    )
    assert metrics["authority_amplification_count"] == 1
    assert metrics["authority_amplification_rate"] == 0.5
    assert metrics["unauthorized_action_rate"] == 0.5


def test_security_violation_matrix_keeps_split_variant_boundaries():
    matrix = security_violation_matrix(
        [
            row(authority_amplification_count=1),
            row(variant="B0", authority_amplification_count=1),
        ]
    )
    assert len(matrix) == 2
    assert all(item["authority_amplification_count"] == 1 for item in matrix)
