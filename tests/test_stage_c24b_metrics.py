from __future__ import annotations

import json

import pytest
import torch

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_c24b_metrics import (
    EMPTY_DENOMINATOR_POLICY,
    access_control_metrics,
    assert_finite_json,
    compute_readiness,
    fact_retrieval_metrics,
    known_routing_metrics,
    reject_quality_metrics,
    risk_coverage_curve,
    set_valued_routing_metrics,
)


R = RelationId.REGISTRY_ID
C = RelationId.CITY_CODE
A = RelationId.ACCESS_CODE


def _finite_json(value):
    assert_finite_json(value)
    json.dumps(value, allow_nan=False)


def test_known_routing_reports_selective_family_and_directed_confusions():
    metrics = known_routing_metrics(
        [R, None, C, A, R, A],
        [R, R, C, C, A, A],
        [True, False, True, True, True, True],
        ["registry-pair", "registry-pair", "city-pair", "city-pair", "access-a", "access-b"],
        ["f1", "f2", "f1", "f2", "f1", "f2"],
        hard_negative_mask=[False, False, True, True, True, True],
    )

    assert metrics["relation_micro_accuracy"] == 0.5
    assert metrics["known_coverage"] == 5 / 6
    assert metrics["abstention_rate"] == 1 / 6
    assert metrics["accepted_route_accuracy"] == 3 / 5
    assert metrics["accepted_relation_precision"] == 3 / 5
    assert metrics["worst_family_accuracy"] == 0.0
    assert metrics["per_relation_accuracy"] == {
        "registry_id": 0.5,
        "city_code": 0.5,
        "access_code": 0.5,
    }
    assert metrics["access_to_registry_error_rate"] == 0.5
    assert metrics["registry_to_access_error_rate"] == 0.0
    assert metrics["city_to_other_error_rate"] == 0.5
    assert metrics["hard_negative_accuracy"] == 0.5
    assert metrics["hard_negative_rejection_rate"] == 0.0
    assert metrics["confusion_matrix"]["class_order"] == [
        "registry_id",
        "city_code",
        "access_code",
        "reject",
    ]
    assert metrics["confusion_matrix"]["matrix"] == [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
    ]
    assert metrics["same_family_cross_frame_route_agreement"][
        "num_cross_unit_pairs"
    ] == 2
    assert metrics["same_relation_cross_family_route_agreement"][
        "num_cross_unit_pairs"
    ] == 1
    _finite_json(metrics)


def test_set_valued_metrics_distinguish_empty_singleton_and_multi_sets():
    metrics = set_valued_routing_metrics(
        [[R], [C, A], [], [R, A], [R], [], [C]],
        [R, C, A, None, None, None, None],
        ["known", "known", "known", "ambiguous", "ambiguous", "unrelated", "unrelated"],
    )

    assert metrics["average_candidate_set_size"] == 1.0
    assert metrics["singleton_rate"] == 3 / 7
    assert metrics["empty_set_rate"] == 2 / 7
    assert metrics["multi_label_set_rate"] == 2 / 7
    assert metrics["true_relation_inclusion_rate"] == 2 / 3
    assert metrics["singleton_correctness"] == 1.0
    assert metrics["ambiguous_set_rate"] == 0.5
    assert metrics["unknown_set_rate"] == 0.5
    assert metrics["set_coverage"] == 2 / 3
    assert metrics["set_inefficiency"] == 1.0
    assert metrics["known_empty_set_rate"] == 1 / 3
    assert metrics["known_multi_relation_set_rate"] == 1 / 3
    _finite_json(metrics)


def _reject_fixture():
    return {
        "accepted": [True, True, False, False, True, False, True],
        "reasons": [None, None, "unknown", "ambiguous", None, "unknown", None],
        "predictions": [R, C, None, None, A, None, C],
        "targets": [R, R, R, None, None, None, None],
        "types": ["known", "known", "known", "ambiguous", "ambiguous", "unrelated", "unrelated"],
        "families": ["known-a", "known-b", "known-c", "amb-a", "amb-b", "unrel-a", "unrel-b"],
        "scores": [0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.5],
    }


def test_reject_metrics_report_groups_detection_risk_and_error_decomposition():
    values = _reject_fixture()
    metrics = reject_quality_metrics(
        values["accepted"],
        values["reasons"],
        values["predictions"],
        values["targets"],
        values["types"],
        values["families"],
        values["scores"],
    )

    assert metrics["ambiguous_rejection_rate"] == 0.5
    assert metrics["unrelated_rejection_rate"] == 0.5
    assert metrics["combined_rejection_rate"] == 0.5
    assert metrics["ambiguous_false_accept_rate"] == 0.5
    assert metrics["unrelated_false_accept_rate"] == 0.5
    assert metrics["combined_false_accept_rate"] == 0.5
    assert metrics["by_group"]["known"]["accepted_error_rate"] == 0.5
    assert metrics["by_group"]["ambiguous"]["accepted_error_rate"] == 1.0
    assert metrics["by_group"]["unrelated"]["accepted_error_rate"] == 1.0
    assert metrics["by_group"]["ambiguous"]["auroc_defined"] is True
    assert metrics["by_group"]["unrelated"]["aupr_defined"] is True
    assert 0.0 <= metrics["by_group"]["combined"]["auroc"] <= 1.0
    assert 0.0 <= metrics["by_group"]["combined"]["aupr"] <= 1.0
    assert metrics["worst_reject_family_false_accept_rate"] == 1.0
    assert metrics["unknown_to_accept_rate"] == 0.5
    assert metrics["ambiguous_to_accept_rate"] == 0.5
    assert metrics["ambiguous_to_unknown_rate"] == 0.0
    assert metrics["unrelated_to_ambiguous_rate"] == 0.0
    assert metrics["fail_closed_rate"] == 1.0
    assert len(metrics["risk_coverage"]["curve"]) == 7
    assert metrics["coverage_aurc"] == metrics["risk_coverage"]["aurc"]
    _finite_json(metrics)


def test_access_control_primary_rates_use_protocol_denominators():
    values = _reject_fixture()
    metrics = access_control_metrics(
        values["accepted"],
        values["predictions"],
        values["targets"],
        values["types"],
    )

    assert metrics["false_memory_access_rate"] == 0.5
    assert metrics["ambiguous_false_memory_access_rate"] == 0.5
    assert metrics["unrelated_false_memory_access_rate"] == 0.5
    assert metrics["wrong_bucket_access_rate"] == 1 / 3
    assert metrics["safe_coverage"] == 1 / 3
    assert metrics["singleton_acceptance_precision"] == 1 / 4
    assert metrics["counts"]["false_memory_access"] == 2
    assert metrics["counts"]["wrong_bucket_access"] == 1
    assert metrics["counts"]["safe_known_access"] == 1
    _finite_json(metrics)


def test_fact_retrieval_aggregates_only_accepted_known_and_keeps_all_query_top1():
    metrics = fact_retrieval_metrics(
        [True, False, True, True],
        ["known", "known", "known", "ambiguous"],
        [True, None, False, None],
        [True, None, True, None],
        [1.0, None, 0.5, None],
        [0.4, None, -0.1, None],
        [4, None, 4, None],
        [0, None, 0, None],
        oracle_fact_top1=0.9,
    )

    assert metrics["accepted_fact_top1"] == 0.5
    assert metrics["all_query_fact_top1"] == 1 / 3
    assert metrics["row_1nn"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["mean_centroid_margin"] == pytest.approx(0.15)
    assert metrics["oracle_gap"] == pytest.approx(0.9 - 1 / 3)
    assert metrics["relation_bucket_candidate_count"] == {
        "mean": 4.0,
        "min": 4,
        "max": 4,
    }
    assert metrics["cross_relation_candidate_count"] == 0
    assert metrics["cross_relation_candidate_count_max_per_query"] == 0
    _finite_json(metrics)


def test_empty_denominators_are_explicit_finite_and_deterministic():
    sets = set_valued_routing_metrics([[R]], [None], ["ambiguous"])
    assert sets["true_relation_inclusion_rate"] == 0.0
    assert sets["singleton_correctness"] == 0.0
    assert sets["denominators"]["known"] == 0
    assert sets["empty_denominator_policy"] == EMPTY_DENOMINATOR_POLICY

    access = access_control_metrics([False], [None], [None], ["ambiguous"])
    assert access["wrong_bucket_access_rate"] == 0.0
    assert access["safe_coverage"] == 0.0
    assert access["denominators"]["wrong_bucket_access_rate"] == 0

    retrieval = fact_retrieval_metrics(
        [False],
        ["known"],
        [None],
        [None],
        [None],
        [None],
        [None],
        [None],
    )
    assert retrieval["accepted_fact_top1"] == 0.0
    assert retrieval["mrr"] == 0.0
    assert retrieval["relation_bucket_candidate_count"] == {
        "mean": 0.0,
        "min": 0,
        "max": 0,
    }
    for value in (sets, access, retrieval):
        _finite_json(value)


def test_risk_coverage_is_stable_on_ties_and_rejects_nonfinite_scores():
    curve = risk_coverage_curve([0.9, 0.9, 0.1], [True, False, True])
    assert curve["curve"][0]["accuracy"] == 1.0
    assert curve["curve"][1]["risk"] == 0.5
    assert 0.0 <= curve["aurc"] <= 1.0
    with pytest.raises(ValueError, match="finite"):
        risk_coverage_curve([0.5, float("nan")], [True, False])


def test_readiness_can_authorize_only_a_future_pool_never_c3():
    ready = compute_readiness(
        closed_set_selective_router_ready=True,
        open_set_abstention_ready=True,
        discrete_memory_contract_preserved=True,
        public_benchmark_human_reviewed=True,
    )
    assert ready["ready_to_create_new_confirmation_pool"] is True
    assert ready["new_confirmation_pool_created_after_freeze"] is False
    assert ready["c3_eligible"] is False

    blocked = compute_readiness(
        closed_set_selective_router_ready=True,
        open_set_abstention_ready=False,
        discrete_memory_contract_preserved=True,
        public_benchmark_human_reviewed=True,
    )
    assert blocked["ready_to_create_new_confirmation_pool"] is False
    assert blocked["c3_eligible"] is False
    _finite_json(ready)
    _finite_json(blocked)


@pytest.mark.parametrize(
    "payload",
    [
        {"metric": float("nan")},
        {"metric": float("inf")},
        {"metric": torch.tensor(1.0)},
        {"private_answer": "forbidden"},
        {"confirmation_path": "forbidden.jsonl"},
        {"new_confirmation_pool_created_after_freeze": True},
    ],
)
def test_json_validator_rejects_nonfinite_answer_and_confirmation_payloads(payload):
    with pytest.raises((TypeError, ValueError)):
        assert_finite_json(payload)


def test_json_validator_allows_only_explicit_false_sensitive_audit_statuses():
    payload = {
        "contains_private_answer": False,
        "contains_private_answers": False,
        "load_private_answers": False,
        "private_answers_loaded": False,
        "create_confirmation": False,
        "confirmation_created_or_read": False,
        "new_confirmation_pool_created_after_freeze": False,
        "ready_to_create_new_confirmation_pool": False,
        "answer_free": True,
    }
    assert_finite_json(payload)
    json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError):
        assert_finite_json({"private_answers_loaded": True})
    with pytest.raises(ValueError):
        assert_finite_json({"confirmation_created_or_read": True})


def test_metric_schemas_fail_closed_on_invalid_or_sensitive_inputs():
    with pytest.raises(ValueError, match="sample type/target mismatch"):
        access_control_metrics([False], [None], [R], ["ambiguous"])
    with pytest.raises(ValueError, match="private-answer/confirmation"):
        known_routing_metrics([R], [R], [True], ["confirmation-family"], ["f"])
    with pytest.raises(ValueError, match="finite"):
        reject_quality_metrics(
            [True, False],
            [None, "unknown"],
            [R, None],
            [R, None],
            ["known", "unrelated"],
            ["known", "unrelated"],
            [0.1, float("nan")],
        )
    with pytest.raises(ValueError, match="cross-relation"):
        fact_retrieval_metrics(
            [True],
            ["known"],
            [True],
            [True],
            [1.0],
            [0.1],
            [1],
            [-1],
        )
