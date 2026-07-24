from __future__ import annotations

import inspect
import json
from dataclasses import replace

import pytest
import torch

from keyed_gram.stage_c24_contract import AcceptedRoute, RelationId
from keyed_gram.stage_c24b_calibration import (
    ClassConditionalConformalCalibrator,
    PairwiseRidgeEvidenceHead,
    build_pairwise_relation_features,
    fit_class_conditional_conformal,
    fit_pairwise_ridge_head,
    freeze_router_selection,
    higher_conformal_quantile,
    select_router_on_calibration,
    verify_frozen_router_selection,
)
from keyed_gram.stage_c24b_router import RejectedRoute


def _candidate(**overrides):
    values = {
        "known_coverage": 0.92,
        "accepted_relation_precision": 0.98,
        "singleton_acceptance_precision": 0.98,
        "safe_coverage": 0.90,
        "false_memory_access_rate": 0.08,
        "ambiguous_rejection_rate": 0.85,
        "unrelated_rejection_rate": 0.97,
        "ambiguous_false_memory_access_rate": 0.15,
        "unrelated_false_memory_access_rate": 0.03,
        "wrong_bucket_access_rate": 0.02,
        "worst_reject_family_false_accept_rate": 0.25,
        "family_macro_accuracy": 0.96,
        "worst_family_accuracy": 0.86,
        "true_relation_inclusion_rate": 0.90,
        "average_candidate_set_size": 1.1,
        "known_multi_relation_set_rate": 0.10,
        "known_empty_set_rate": 0.05,
    }
    values.update(overrides)
    return values


def _calibration_rows():
    rows = []
    targets = []
    for relation in RelationId:
        for score in (0.80, 0.85, 0.90):
            rows.append(
                {
                    current: (
                        score if current is relation else 0.15
                    )
                    for current in RelationId
                }
            )
            targets.append(relation)
    rows.extend(
        [
            {relation: 0.1 for relation in RelationId},
            {relation: 0.4 for relation in RelationId},
        ]
    )
    targets.extend([None, None])
    return rows, targets


def test_pairwise_feature_builder_is_relation_specific_and_answer_free_shape():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    definitions = {
        RelationId.REGISTRY_ID: torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        RelationId.CITY_CODE: torch.tensor([[0.0, 1.0], [-1.0, 1.0]]),
        RelationId.ACCESS_CODE: torch.tensor([[-1.0, 0.0], [-1.0, -1.0]]),
    }
    features = build_pairwise_relation_features(queries, definitions)
    assert set(features) == set(RelationId)
    # 四组 2-D 特征加一个 cosine。
    assert {tuple(value.shape) for value in features.values()} == {(2, 9)}
    assert not torch.equal(
        features[RelationId.REGISTRY_ID], features[RelationId.ACCESS_CODE]
    )
    assert all(torch.isfinite(value).all() for value in features.values())


def test_r2_fits_independent_binary_ridge_heads_and_round_trips_public_state():
    targets = [
        RelationId.REGISTRY_ID,
        RelationId.REGISTRY_ID,
        RelationId.CITY_CODE,
        RelationId.CITY_CODE,
        RelationId.ACCESS_CODE,
        RelationId.ACCESS_CODE,
    ]
    features = {}
    for relation in RelationId:
        positive = torch.tensor(
            [1.0 if target is relation else -1.0 for target in targets]
        )
        secondary = torch.linspace(-0.3, 0.3, len(targets))
        features[relation] = torch.stack([positive, secondary], dim=1)
    head = fit_pairwise_ridge_head(features, targets, ridge_strength=0.01)
    assert isinstance(head, PairwiseRidgeEvidenceHead)
    scores = head.score_batch(features)
    for index, target in enumerate(targets):
        assert scores[target][index] > 0.8
        for other in RelationId:
            if other is not target:
                assert scores[other][index] < 0.2

    state = head.to_state_dict()
    assert state["optimizer_state_present"] is False
    assert state["private_data_present"] is False
    restored = PairwiseRidgeEvidenceHead.from_state_dict(state)
    restored_scores = restored.score_batch(features)
    for relation in RelationId:
        assert torch.allclose(scores[relation], restored_scores[relation])
    json.dumps(head.to_json_dict(), allow_nan=False)
    assert "answer" not in json.dumps(head.to_json_dict()).casefold()


def test_r2_unknown_targets_nonfinite_features_and_private_state_fail_closed():
    features = {relation: torch.eye(3) for relation in RelationId}
    with pytest.raises(ValueError, match="unknown relation"):
        fit_pairwise_ridge_head(
            features,
            [RelationId.REGISTRY_ID, RelationId.CITY_CODE, "unknown"],
        )
    broken = dict(features)
    broken[RelationId.ACCESS_CODE] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]]
    )
    with pytest.raises(ValueError, match="finite"):
        fit_pairwise_ridge_head(
            broken,
            [
                RelationId.REGISTRY_ID,
                RelationId.CITY_CODE,
                RelationId.ACCESS_CODE,
            ],
        )


def test_higher_conformal_quantile_uses_finite_sample_ceiling_rule():
    values = torch.tensor([0.1, 0.2, 0.3, 0.4])
    quantile, rank = higher_conformal_quantile(values, alpha=0.20)
    assert rank == 4  # ceil((4 + 1) * 0.8)
    assert quantile == pytest.approx(0.4)
    clipped, clipped_rank = higher_conformal_quantile(values, alpha=0.10)
    assert clipped_rank == 4
    assert clipped == pytest.approx(0.4)


def test_r4_class_conditional_sets_produce_empty_singleton_and_multiple_routes():
    rows, targets = _calibration_rows()
    calibrator = fit_class_conditional_conformal(rows, targets, alpha=0.20)
    assert isinstance(calibrator, ClassConditionalConformalCalibrator)
    assert calibrator.candidate_set(
        {relation: 0.1 for relation in RelationId}
    ) == frozenset()
    registry = {
        RelationId.REGISTRY_ID: 0.85,
        RelationId.CITY_CODE: 0.1,
        RelationId.ACCESS_CODE: 0.1,
    }
    assert calibrator.route(registry) == AcceptedRoute(RelationId.REGISTRY_ID)
    ambiguous = {
        RelationId.REGISTRY_ID: 0.85,
        RelationId.CITY_CODE: 0.85,
        RelationId.ACCESS_CODE: 0.1,
    }
    assert calibrator.route(ambiguous) == RejectedRoute("ambiguous")
    assert calibrator.route(
        {relation: 0.1 for relation in RelationId}
    ) == RejectedRoute("unknown")
    assert calibrator.route(
        {
            RelationId.REGISTRY_ID: float("nan"),
            RelationId.CITY_CODE: 0.1,
            RelationId.ACCESS_CODE: 0.1,
        }
    ) == RejectedRoute("invalid")


def test_r4_metadata_states_scope_and_distribution_shift_limitation_and_round_trips():
    rows, targets = _calibration_rows()
    calibrator = fit_class_conditional_conformal(rows, targets, alpha=0.10)
    payload = calibrator.to_dict()
    assert payload["nonconformity"] == "negative_independent_relation_support"
    assert "ceil((n+1)*(1-alpha))" in payload["metadata"]["quantile_rule"]
    assert "exchangeability" in payload["metadata"]["finite_sample_scope"]
    assert any(
        "distribution shift" in item
        for item in payload["metadata"]["limitations"]
    )
    json.dumps(payload, allow_nan=False)
    restored = ClassConditionalConformalCalibrator.from_dict(payload)
    assert restored.to_dict() == payload


def test_router_selection_api_has_no_post_calibration_inputs_and_is_deterministic():
    parameters = inspect.signature(select_router_on_calibration).parameters
    assert all(
        term not in name.casefold()
        for name in parameters
        for term in ("development", "locked", "audit")
    )
    selection = select_router_on_calibration(
        {
            "R3": _candidate(false_memory_access_rate=0.11),
            "R4": _candidate(),
        }
    )
    assert selection["selected_router"] == "R4"
    assert selection["selection_provenance"] == {
        "selection_split": "public_calibration_v2",
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
    }
    with pytest.raises(ValueError, match="post-calibration"):
        select_router_on_calibration(
            {"R4": {**_candidate(), "locked_score": 1.0}}
        )


def test_frozen_router_selection_binds_parameters_sources_models_and_sha():
    frozen = freeze_router_selection(
        "R4",
        {
            "alpha": 0.1,
            "quantiles": {relation: -0.8 for relation in RelationId},
        },
        public_train_sha256="a" * 64,
        public_calibration_sha256="b" * 64,
        review_manifest_sha256="c" * 64,
        git_commit="1234567890abcdef",
        model_manifests={
            "small_encoder": {
                "model_id": "example/small",
                "revision": "fixed-revision",
                "manifest_sha256": "d" * 64,
            }
        },
        selection_protocol={
            "aggregation_selected_on": "public_calibration_v2",
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
        },
    )
    assert verify_frozen_router_selection(frozen)
    assert verify_frozen_router_selection(frozen, expected_sha256=frozen.sha256)
    payload = frozen.to_dict()
    assert payload["router_parameters"]["quantiles"]["registry_id"] == -0.8
    assert payload["selection_protocol"]["parameters_frozen"] is True
    assert payload["selection_protocol"]["development_used_for_selection"] is False
    assert payload["selection_protocol"]["locked_audit_used_for_selection"] is False
    json.dumps(payload, allow_nan=False)

    tampered = replace(frozen, canonical_json=frozen.canonical_json + " ")
    assert verify_frozen_router_selection(tampered) is False
    with pytest.raises(ValueError, match="non-finite"):
        freeze_router_selection(
            "R3",
            {"threshold": float("nan")},
            public_train_sha256="a" * 64,
            public_calibration_sha256="b" * 64,
            review_manifest_sha256="c" * 64,
            git_commit="1234567",
            model_manifests={},
        )
    with pytest.raises(ValueError, match="forbidden state"):
        freeze_router_selection(
            "R2",
            {"private_answer": "must-not-enter"},
            public_train_sha256="a" * 64,
            public_calibration_sha256="b" * 64,
            review_manifest_sha256="c" * 64,
            git_commit="1234567",
            model_manifests={},
        )
