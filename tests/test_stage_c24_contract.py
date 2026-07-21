from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

import keyed_gram.stage_c24_contract as contract
from keyed_gram.stage_c24_contract import (
    AcceptedRoute,
    DiscreteMemoryInput,
    RelationId,
    RejectedRoute,
    closed_set_route,
    hard_one_hot_from_logits,
    memory_contract_exposure_flags,
    memory_input,
    retrieve_for_route,
    retrieve_from_bucket,
    route_with_reject_guard,
)


def _buckets() -> dict[RelationId, torch.Tensor]:
    return {
        RelationId.REGISTRY_ID: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        RelationId.CITY_CODE: torch.tensor([[-1.0, 0.0]]),
        RelationId.ACCESS_CODE: torch.tensor([[0.0, -1.0]]),
    }


def test_accepted_route_has_only_status_and_discrete_relation_id():
    route = AcceptedRoute(RelationId.CITY_CODE)
    assert route.status == "accept"
    assert route.relation_id is RelationId.CITY_CODE
    assert {item.name for item in fields(route)} == {"status", "relation_id"}
    assert not hasattr(route, "confidence")
    with pytest.raises(TypeError, match="RelationId"):
        AcceptedRoute("city_code")  # type: ignore[arg-type]


def test_fail_closed_route_rejects_nonfinite_malformed_and_unknown_relations():
    assert closed_set_route(torch.tensor([0.0, 2.0, 1.0])) == AcceptedRoute(
        RelationId.CITY_CODE
    )
    for logits in (
        torch.tensor([0.0, float("nan"), 1.0]),
        torch.tensor([0.0, float("inf"), 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0.0, 2.0, 1.0]]),
    ):
        result = closed_set_route(logits)
        assert isinstance(result, RejectedRoute)
        assert result.reason == "invalid_relation_logits"

    result = closed_set_route(
        torch.tensor([0.0, 2.0, 1.0]),
        relation_order=(
            RelationId.REGISTRY_ID,
            RelationId.CITY_CODE,
            "unknown_relation",  # type: ignore[arg-type]
        ),
    )
    assert result == RejectedRoute("unknown_relation_enum")


def test_reject_guard_is_fail_closed_for_nan_invalid_scores_and_thresholds():
    logits = torch.tensor([3.0, 2.0, 1.0])
    assert route_with_reject_guard(
        logits, reject_score=0.75, threshold=0.75
    ) == AcceptedRoute(RelationId.REGISTRY_ID)
    assert route_with_reject_guard(
        logits, reject_score=0.74, threshold=0.75
    ) == RejectedRoute("below_reject_threshold")
    assert route_with_reject_guard(
        logits, reject_score=float("nan"), threshold=0.75
    ) == RejectedRoute("invalid_reject_score")
    assert route_with_reject_guard(
        logits, reject_score=object(), threshold=0.75  # type: ignore[arg-type]
    ) == RejectedRoute("invalid_reject_score")
    assert route_with_reject_guard(
        logits, reject_score=0.8, threshold=float("inf")
    ) == RejectedRoute("invalid_reject_threshold")


def test_d1_hard_one_hot_contains_one_literal_one_per_query():
    logits = torch.tensor(
        [[0.1, 0.2, 0.3], [5.0, -1.0, 2.0], [1.0, 1.0, 1.0]]
    )
    encoded = hard_one_hot_from_logits(logits)
    assert encoded.dtype == torch.float32
    assert encoded.tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert torch.equal(encoded.sum(dim=1), torch.ones(3))
    assert set(encoded.unique().tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError, match="finite"):
        hard_one_hot_from_logits(torch.tensor([1.0, float("nan"), 0.0]))


def test_memory_api_has_no_continuous_or_text_side_channel():
    signature = inspect.signature(retrieve_from_bucket)
    assert tuple(signature.parameters) == (
        "entity_embedding",
        "relation_id",
        "bucket_prototypes",
    )
    forbidden = {
        "logits",
        "probabilities",
        "confidence",
        "margin",
        "semantic_embedding",
        "raw_prompt",
        "relation_phrase",
        "family_id",
    }
    assert forbidden.isdisjoint(signature.parameters)
    assert {item.name for item in fields(DiscreteMemoryInput)} == {
        "entity_embedding",
        "relation_id",
    }
    flags = memory_contract_exposure_flags()
    assert all(value is False for value in flags.values())
    with pytest.raises(TypeError, match="unexpected keyword"):
        retrieve_from_bucket(
            torch.tensor([1.0, 0.0]),
            RelationId.REGISTRY_ID,
            _buckets(),
            confidence=0.99,  # type: ignore[call-arg]
        )


def test_memory_input_is_invariant_to_upstream_confidence_and_rejects_it_as_input():
    entity = torch.tensor([1.0, 2.0])
    low_confidence_route = route_with_reject_guard(
        torch.tensor([3.0, 2.0, 1.0]), reject_score=0.51, threshold=0.5
    )
    high_confidence_route = route_with_reject_guard(
        torch.tensor([3.0, 2.0, 1.0]), reject_score=0.99, threshold=0.5
    )
    assert isinstance(low_confidence_route, AcceptedRoute)
    assert isinstance(high_confidence_route, AcceptedRoute)
    assert memory_input(entity, low_confidence_route.relation_id) == memory_input(
        entity, high_confidence_route.relation_id
    )
    assert "confidence" not in inspect.signature(memory_input).parameters
    with pytest.raises(TypeError, match="unexpected keyword"):
        memory_input(
            entity,
            RelationId.REGISTRY_ID,
            confidence=0.51,  # type: ignore[call-arg]
        )


def test_d2_searches_only_selected_bucket_and_reports_zero_cross_candidates():
    buckets = _buckets()
    result = retrieve_from_bucket(
        torch.tensor([0.0, 1.0]), RelationId.REGISTRY_ID, buckets
    )
    assert result.relation_id is RelationId.REGISTRY_ID
    assert result.candidate_index == 1
    assert result.candidate_count == 2
    assert result.cross_relation_candidate_count == 0
    assert result.memory_access_count == 1

    # A much closer vector in another relation is never a candidate.
    changed = {**buckets, RelationId.CITY_CODE: torch.tensor([[0.0, 1000.0]])}
    assert retrieve_from_bucket(
        torch.tensor([0.0, 1.0]), RelationId.REGISTRY_ID, changed
    ) == result


def test_rejected_route_cannot_trigger_retrieval_or_inspect_memory(monkeypatch):
    accessed = False

    def forbidden_retrieval(*args, **kwargs):
        nonlocal accessed
        accessed = True
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(contract, "retrieve_from_bucket", forbidden_retrieval)
    execution = retrieve_for_route(
        object(),  # type: ignore[arg-type]
        RejectedRoute("ambiguous_request"),
        object(),  # type: ignore[arg-type]
    )
    assert accessed is False
    assert execution.retrieval is None
    assert execution.memory_access_count == 0
    assert execution.route.status == "reject"


def test_passing_rejected_route_to_bucket_api_fails_before_mapping_access():
    class MustNotBeRead(dict):
        def __contains__(self, key):
            raise AssertionError("memory mapping was inspected")

        def __getitem__(self, key):
            raise AssertionError("memory mapping was inspected")

    with pytest.raises(TypeError, match="accepted RelationId"):
        retrieve_from_bucket(
            torch.tensor([1.0, 0.0]),
            RejectedRoute("unrelated_request"),  # type: ignore[arg-type]
            MustNotBeRead(),
        )


def test_retrieval_validates_finite_nonempty_selected_bucket():
    with pytest.raises(ValueError, match="finite"):
        retrieve_from_bucket(
            torch.tensor([1.0, 0.0]),
            RelationId.REGISTRY_ID,
            {RelationId.REGISTRY_ID: torch.tensor([[float("nan"), 0.0]])},
        )
    with pytest.raises(ValueError, match="empty or has the wrong width"):
        retrieve_from_bucket(
            torch.tensor([1.0, 0.0]),
            RelationId.REGISTRY_ID,
            {RelationId.REGISTRY_ID: torch.empty(0, 2)},
        )
    with pytest.raises(ValueError, match="non-zero norm"):
        retrieve_from_bucket(
            torch.tensor([1.0, 0.0]),
            RelationId.REGISTRY_ID,
            {RelationId.REGISTRY_ID: torch.zeros(1, 2)},
        )
