from __future__ import annotations

import inspect
import json
import math

import torch

from keyed_gram.stage_c24_contract import RELATION_ORDER, RelationId
from keyed_gram.stage_c24_metrics import (
    closed_route_metrics,
    evaluate_reject_guard,
    extract_reject_score_candidates,
    g1_max_softmax_probability,
    g2_top_logit_margin,
    g3_definition_similarity_margin,
    select_calibrated_reject_guard,
)


def _assert_finite_json(value):
    json.dumps(value, allow_nan=False)
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_json(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_closed_route_metrics_include_confusion_and_both_agreements():
    predictions = [
        RelationId.REGISTRY_ID,
        RelationId.REGISTRY_ID,
        RelationId.CITY_CODE,
        RelationId.ACCESS_CODE,
        RelationId.ACCESS_CODE,
        RelationId.ACCESS_CODE,
    ]
    targets = [
        RelationId.REGISTRY_ID,
        RelationId.REGISTRY_ID,
        RelationId.CITY_CODE,
        RelationId.CITY_CODE,
        RelationId.ACCESS_CODE,
        RelationId.ACCESS_CODE,
    ]
    families = ["r1", "r1", "c1", "c1", "a1", "a2"]
    frames = ["f1", "f2", "f1", "f2", "f1", "f2"]
    metrics = closed_route_metrics(predictions, targets, families, frames)
    assert metrics["relation_micro_accuracy"] == 5 / 6
    assert metrics["relation_family_macro_accuracy"] == (1.0 + 0.5 + 1.0 + 1.0) / 4
    assert metrics["worst_family_accuracy"] == 0.5
    assert metrics["confusion_matrix"]["matrix"] == [
        [2, 0, 0],
        [0, 1, 1],
        [0, 0, 2],
    ]
    assert metrics["same_family_cross_frame_route_agreement"][
        "per_group_agreement"
    ]["c1"] == 0.0
    assert metrics["same_relation_cross_phrase_family_route_agreement"][
        "per_group_agreement"
    ]["access_code"] == 1.0


def test_score_extractors_reorder_classes_and_compute_g1_g2_g3():
    source_order = (
        RelationId.ACCESS_CODE,
        RelationId.REGISTRY_ID,
        RelationId.CITY_CODE,
    )
    head = torch.tensor([[1.0, 4.0, 2.0]])
    definition = torch.tensor([[5.0, 1.0, 3.0]])
    extracted = extract_reject_score_candidates(
        head,
        definition,
        head_class_order=source_order,
        definition_class_order=source_order,
    )
    assert extracted["relation_order"] == RELATION_ORDER
    assert extracted["predicted_relations"] == (RelationId.REGISTRY_ID,)
    assert torch.allclose(
        extracted["scores"]["G1_max_softmax"],
        g1_max_softmax_probability(head, class_order=source_order),
    )
    assert g2_top_logit_margin(head, class_order=source_order).tolist() == [2.0]
    assert g3_definition_similarity_margin(
        definition, class_order=source_order
    ).tolist() == [2.0]


def test_calibration_selection_api_has_no_development_or_locked_input():
    parameters = inspect.signature(select_calibrated_reject_guard).parameters
    assert all(
        token not in name.lower()
        for name in parameters
        for token in ("development", "locked", "audit")
    )

    # Known definition margins are large while all open-set definition margins
    # are small, making G3 deterministically preferable on calibration alone.
    head = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
            [0.0, 1.0, 5.0],
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    definitions = torch.tensor(
        [
            [8.0, 1.0, 0.0],
            [0.0, 8.0, 1.0],
            [0.0, 1.0, 8.0],
            [7.0, 1.0, 0.0],
            [0.0, 7.0, 1.0],
            [0.0, 1.0, 7.0],
            [1.00, 0.99, 0.98],
            [0.99, 1.00, 0.98],
            [1.00, 0.98, 0.99],
            [0.98, 1.00, 0.99],
        ]
    )
    targets = [
        RelationId.REGISTRY_ID,
        RelationId.CITY_CODE,
        RelationId.ACCESS_CODE,
        RelationId.REGISTRY_ID,
        RelationId.CITY_CODE,
        RelationId.ACCESS_CODE,
        None,
        None,
        None,
        None,
    ]
    selection = select_calibrated_reject_guard(
        head,
        definitions,
        targets,
        ["known"] * 6 + ["ambiguous", "ambiguous", "unrelated", "unrelated"],
        ["known"] * 6 + ["amb-a", "amb-b", "unrel-a", "unrel-b"],
        head_class_order=RELATION_ORDER,
        definition_class_order=RELATION_ORDER,
        temperature_grid=(0.5, 1.0, 2.0),
    )
    assert selection["selected_score"] == "G3_definition_margin"
    provenance = selection["selection_provenance"]
    assert provenance["selection_split"] == "public_calibration"
    assert provenance["development_used_for_selection"] is False
    assert provenance["locked_audit_used_for_selection"] is False
    assert selection["selected_candidate"]["calibration_metrics"]["known_coverage"] >= 0.95
    _assert_finite_json(selection)


def test_reject_evaluation_fails_closed_on_nan_and_reports_grouped_far():
    metrics = evaluate_reject_guard(
        torch.tensor([0.9, float("nan"), 0.8, 0.7, 0.1, 0.2, 0.6]),
        [
            RelationId.REGISTRY_ID,
            RelationId.CITY_CODE,
            RelationId.CITY_CODE,
            RelationId.ACCESS_CODE,
            RelationId.REGISTRY_ID,
            RelationId.CITY_CODE,
            RelationId.ACCESS_CODE,
        ],
        [
            RelationId.REGISTRY_ID,
            RelationId.CITY_CODE,
            RelationId.CITY_CODE,
            None,
            None,
            None,
            None,
        ],
        ["known", "known", "known", "ambiguous", "ambiguous", "unrelated", "unrelated"],
        ["known"] * 3 + ["amb-a", "amb-b", "unrel-a", "unrel-b"],
        threshold=0.5,
        calibration_confidence=torch.tensor([0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.7]),
    )
    assert metrics["known_coverage"] == 2 / 3
    assert metrics["ambiguous_false_accept_rate"] == 0.5
    assert metrics["unrelated_false_accept_rate"] == 0.5
    assert metrics["false_accept_rate"] == 0.5
    assert metrics["worst_reject_family_false_accept_rate"] == 1.0
    assert metrics["num_invalid_scores_or_routes"] == 1
    assert metrics["invalid_score_rejection_rate"] == 1.0
    assert metrics["fail_closed_rate"] == 1.0
    assert metrics["nan_or_invalid_score_behavior"] == "reject"
    assert set(metrics["by_open_set_kind"]) == {
        "known",
        "ambiguous",
        "unrelated",
        "ambiguous_unrelated_overall",
    }
    assert metrics["by_open_set_kind"]["ambiguous"][
        "false_accept_rate_at_frozen_calibration_threshold"
    ] == 0.5
    assert metrics["tnr_at_frozen_calibration_threshold"] == 0.5
    _assert_finite_json(metrics)
