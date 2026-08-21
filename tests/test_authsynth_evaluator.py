from __future__ import annotations

import dataclasses

import pytest

from keyed_gram.authsynth_evaluator import (
    InMemorySandboxReplay,
    evaluate_split,
    generate_f2b_benchmark,
)
from keyed_gram.authsynth_evaluator.benchmark import SPLITS, collision_audit


def test_blind_benchmark_size_split_isolation_and_unknown_controls() -> None:
    vault = generate_f2b_benchmark("unit-seed")
    assert len(vault.cases) == 270
    audit = collision_audit(vault.cases)
    assert audit["cross_split_collision_count"] == 0
    assert set(audit["split_case_counts"]) == set(SPLITS)
    evidence_gap_count = sum(
        any(
            result.instrumentation_coverage < 1.0
            or not result.bounded_quiescence_reached
            for _, result in case.concrete_replays
        )
        for case in vault.cases
    )
    assert evidence_gap_count == 24


def test_public_analyzer_schema_excludes_evaluator_labels() -> None:
    vault = generate_f2b_benchmark("schema-seed")
    row = vault.public_rows("locked_test")[0]
    encoded = str(row).casefold()
    for forbidden in (
        "concrete_implementation",
        "hidden_transition",
        "mutation_category",
        "omission_type",
        "expected_spurious_counterexample",
        "attack_trace_is_forbidden",
        "gold_contract",
        "gold_shield",
    ):
        assert forbidden not in encoded


def test_replay_api_returns_only_narrow_result_and_rejects_cross_case() -> None:
    vault = generate_f2b_benchmark("replay-seed")
    first, second = vault.cases[:2]
    service = InMemorySandboxReplay((first, second))
    query = first.analyzer_input.query_catalog[0]
    result = service.replay(first.analyzer_input.case_handle, query)
    assert {field.name for field in dataclasses.fields(result)} == {
        "query_id",
        "events",
        "state_diff_digest",
        "exit_status",
        "bounded_quiescence_reached",
        "instrumentation_coverage",
        "observed_version_digest",
    }
    with pytest.raises(ValueError, match="跨 case"):
        service.replay(second.analyzer_input.case_handle, query)


def test_vault_enforces_one_open_per_split() -> None:
    vault = generate_f2b_benchmark("vault-seed")
    assert vault.opened_count("locked_test") == 0
    assert vault.open_for_evaluation("locked_test")
    assert vault.opened_count("locked_test") == 1
    with pytest.raises(RuntimeError, match="拒绝重复"):
        vault.open_for_evaluation("locked_test")


@pytest.mark.parametrize("split", ["development", "locked_test"])
def test_blind_cgar_fixture_satisfies_preregistered_gate(split: str) -> None:
    vault = generate_f2b_benchmark(f"gate-{split}")
    metrics = evaluate_split(vault.open_for_evaluation(split))["metrics"]
    selected = next(item for item in metrics if item["method"] == "AuthSynth-CGAR")
    assert selected["false_verified_complete_count"] == 0
    assert selected["forbidden_trace_acceptance_rate"] == 0.0
    assert selected["contract_effect_recall"] >= 0.90
    assert selected["policy_omission_discovery_recall"] >= 0.80
    assert selected["counterexample_precision"] >= 0.75
    assert selected["safe_utility"] >= 0.85
    assert selected["maximal_permissiveness_ratio"] >= 0.90
    assert selected["drift_detection_recall"] == 1.0
    assert selected["unknown_rate"] <= 0.25
    assert selected["unknown_expected_precision"] == 1.0


def test_declared_monitor_exposes_contract_completeness_blind_spot() -> None:
    vault = generate_f2b_benchmark("distinguishing-seed")
    metrics = evaluate_split(vault.open_for_evaluation("train"))["metrics"]
    declared = next(
        item for item in metrics if item["method"] == "Declared-Contract Monitor"
    )
    blind = next(item for item in metrics if item["method"] == "AuthSynth-CGAR")
    assert declared["forbidden_trace_acceptance_rate"] > 0.0
    assert blind["forbidden_trace_acceptance_rate"] == 0.0
    assert blind["safe_utility"] > 0.85
