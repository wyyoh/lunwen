from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

import keyed_gram.stage_c24b_router as router_module
from keyed_gram.stage_c24_contract import (
    AcceptedRoute,
    RelationId,
    RetrievalResult,
    retrieve_from_bucket,
)
from keyed_gram.stage_c24b_router import (
    RejectedRoute,
    candidate_set_from_scores,
    execute_selective_route,
    r0_frozen_argmax_route,
    r1_definition_ensemble_scores,
    r3_set_valued_route,
    route_from_candidate_set,
)


def _scores(registry: float, city: float, access: float):
    return {
        RelationId.REGISTRY_ID: registry,
        RelationId.CITY_CODE: city,
        RelationId.ACCESS_CODE: access,
    }


def test_accepted_route_and_memory_api_expose_only_the_discrete_contract():
    route = AcceptedRoute(RelationId.REGISTRY_ID)
    assert {item.name for item in fields(route)} == {"status", "relation_id"}
    assert isinstance(route.relation_id, RelationId)
    with pytest.raises(TypeError, match="RelationId"):
        AcceptedRoute("registry_id")  # type: ignore[arg-type]

    signature = inspect.signature(retrieve_from_bucket)
    assert tuple(signature.parameters) == (
        "entity_embedding",
        "relation_id",
        "bucket_prototypes",
    )
    forbidden = {
        "confidence",
        "logits",
        "probabilities",
        "margin",
        "semantic_embedding",
        "raw_text",
        "relation_phrase",
        "family_id",
        "candidate_relation_set",
    }
    assert forbidden.isdisjoint(signature.parameters)


def test_c24b_rejected_route_has_only_three_fail_closed_reasons():
    for reason in ("unknown", "ambiguous", "invalid"):
        route = RejectedRoute(reason)  # type: ignore[arg-type]
        assert route.status == "reject"
        assert route.reason == reason
        assert {item.name for item in fields(route)} == {"status", "reason"}
    with pytest.raises(ValueError, match="unknown, ambiguous, or invalid"):
        RejectedRoute("below_threshold")  # type: ignore[arg-type]


def test_r0_reproduces_frozen_argmax_with_explicit_class_order_and_fails_closed():
    order = (
        RelationId.ACCESS_CODE,
        RelationId.CITY_CODE,
        RelationId.REGISTRY_ID,
    )
    assert r0_frozen_argmax_route(
        torch.tensor([8.0, 1.0, 0.0]), relation_order=order
    ) == AcceptedRoute(RelationId.ACCESS_CODE)
    assert r0_frozen_argmax_route(torch.tensor([8.0, 1.0, 0.0])) == AcceptedRoute(
        RelationId.ACCESS_CODE
    )
    assert r0_frozen_argmax_route(
        torch.tensor([0.0, 1.0, 8.0]), relation_order=order
    ) == AcceptedRoute(RelationId.REGISTRY_ID)
    for malformed in (
        torch.tensor([1.0, float("nan"), 0.0]),
        torch.tensor([1.0, float("inf"), 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    ):
        assert r0_frozen_argmax_route(malformed) == RejectedRoute("invalid")
    assert r0_frozen_argmax_route(
        torch.tensor([1.0, 0.0, 0.0]),
        relation_order=("unknown", "city_code", "access_code"),
    ) == RejectedRoute("invalid")


def test_r1_definition_ensemble_supports_mean_max_and_top_k_without_softmax():
    query = torch.tensor([1.0, 0.0])
    definitions = {
        RelationId.REGISTRY_ID: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        RelationId.CITY_CODE: torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
        RelationId.ACCESS_CODE: torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
    }
    mean = r1_definition_ensemble_scores(query, definitions, aggregation="mean")
    maximum = r1_definition_ensemble_scores(query, definitions, aggregation="max")
    top_one = r1_definition_ensemble_scores(
        query, definitions, aggregation="top-k-mean", top_k=1
    )
    assert mean[RelationId.REGISTRY_ID] == pytest.approx(0.5)
    assert maximum[RelationId.REGISTRY_ID] == pytest.approx(1.0)
    assert top_one[RelationId.REGISTRY_ID] == pytest.approx(1.0)
    assert mean[RelationId.CITY_CODE] == pytest.approx(-0.5)
    assert maximum[RelationId.CITY_CODE] == pytest.approx(0.0)
    assert mean[RelationId.ACCESS_CODE] == pytest.approx(2**-0.5)
    # 三个 raw cosine evidence 不被归一化为概率和 1。
    assert sum(mean.values()) != pytest.approx(1.0)


def test_r1_rejects_nonfinite_unknown_or_malformed_definition_evidence():
    definitions = {
        RelationId.REGISTRY_ID: torch.eye(2),
        RelationId.CITY_CODE: torch.eye(2),
        RelationId.ACCESS_CODE: torch.eye(2),
    }
    with pytest.raises(ValueError, match="finite"):
        r1_definition_ensemble_scores(
            torch.tensor([float("nan"), 0.0]), definitions
        )
    with pytest.raises(ValueError, match="unknown relation|frozen relation set"):
        r1_definition_ensemble_scores(
            torch.tensor([1.0, 0.0]),
            {**definitions, "unknown": torch.eye(2)},
        )
    with pytest.raises(ValueError, match="top_k"):
        r1_definition_ensemble_scores(
            torch.tensor([1.0, 0.0]), definitions, aggregation="top_k_mean"
        )


def test_r3_candidate_set_empty_singleton_multiple_and_margin_semantics():
    assert candidate_set_from_scores(_scores(0.1, 0.2, 0.3), 0.5) == frozenset()
    assert candidate_set_from_scores(_scores(0.8, 0.2, 0.3), 0.5) == frozenset(
        {RelationId.REGISTRY_ID}
    )
    assert route_from_candidate_set(()) == RejectedRoute("unknown")
    assert route_from_candidate_set((RelationId.CITY_CODE,)) == AcceptedRoute(
        RelationId.CITY_CODE
    )
    assert route_from_candidate_set(
        (RelationId.CITY_CODE, RelationId.ACCESS_CODE)
    ) == RejectedRoute("ambiguous")

    assert r3_set_valued_route(_scores(0.80, 0.75, 0.10), 0.78) == AcceptedRoute(
        RelationId.REGISTRY_ID
    )
    assert r3_set_valued_route(
        _scores(0.80, 0.75, 0.10), 0.78, minimum_margin=0.10
    ) == RejectedRoute("ambiguous")
    # 唯一过阈值候选并非 raw top-1 时，margin 必须相对该候选计算，
    # 不能借用候选之外两个分数的正差而错误接受。
    assert r3_set_valued_route(
        _scores(0.40, 0.90, 0.10),
        {
            RelationId.REGISTRY_ID: 0.30,
            RelationId.CITY_CODE: 1.00,
            RelationId.ACCESS_CODE: 0.50,
        },
        minimum_margin=0.20,
    ) == RejectedRoute("ambiguous")


def test_r3_nan_infinity_unknown_relation_and_threshold_fail_closed():
    for invalid in (
        _scores(float("nan"), 0.1, 0.1),
        _scores(float("inf"), 0.1, 0.1),
        {"registry_id": 0.9, "city_code": 0.1, "unknown": 0.2},
    ):
        assert r3_set_valued_route(invalid, 0.5) == RejectedRoute("invalid")
    assert r3_set_valued_route(
        _scores(0.9, 0.1, 0.1), float("nan")
    ) == RejectedRoute("invalid")
    assert r3_set_valued_route(
        _scores(0.9, 0.1, 0.1), 0.5, minimum_margin=-0.1
    ) == RejectedRoute("invalid")


def test_rejected_route_does_not_inspect_entity_or_memory(monkeypatch):
    class MustNotBeRead(dict):
        def __getitem__(self, key):
            raise AssertionError("bucket mapping must not be inspected")

    def forbidden_retrieval(*args, **kwargs):
        raise AssertionError("memory retrieval must not run")

    monkeypatch.setattr(router_module, "retrieve_from_bucket", forbidden_retrieval)
    execution = execute_selective_route(
        object(),  # type: ignore[arg-type]
        RejectedRoute("ambiguous"),
        MustNotBeRead(),
    )
    assert execution.memory_access_count == 0
    assert execution.retrieval is None


def test_accepted_route_passes_exactly_one_relation_bucket(monkeypatch):
    observed = {}

    def observed_retrieval(entity_embedding, relation_id, bucket_prototypes):
        observed["entity"] = entity_embedding
        observed["relation"] = relation_id
        observed["keys"] = tuple(bucket_prototypes)
        return RetrievalResult(
            relation_id=relation_id,
            candidate_index=0,
            similarity=1.0,
            candidate_count=1,
        )

    monkeypatch.setattr(router_module, "retrieve_from_bucket", observed_retrieval)
    buckets = {
        RelationId.REGISTRY_ID: torch.tensor([[1.0, 0.0]]),
        RelationId.CITY_CODE: torch.tensor([[0.0, 1.0]]),
        RelationId.ACCESS_CODE: torch.tensor([[-1.0, 0.0]]),
    }
    entity = torch.tensor([1.0, 0.0])
    execution = execute_selective_route(
        entity, AcceptedRoute(RelationId.CITY_CODE), buckets
    )
    assert observed == {
        "entity": entity,
        "relation": RelationId.CITY_CODE,
        "keys": (RelationId.CITY_CODE,),
    }
    assert execution.memory_access_count == 1
    assert execution.retrieval is not None
    assert execution.retrieval.cross_relation_candidate_count == 0
