"""Typed, fail-closed memory contract for Stage C2.4.

The relation classifier and reject guard live *upstream* of this module's memory
surface.  An accepted request crosses that boundary as a discrete
``RelationId`` only.  In particular, classifier logits, probabilities, margins,
confidence values, semantic relation embeddings, prompt text, and phrase-family
metadata are deliberately absent from every memory-facing type and function.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Literal, Mapping, Sequence, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor


class RelationId(str, Enum):
    """The complete closed-set relation vocabulary frozen for Stage C2.4."""

    REGISTRY_ID = "registry_id"
    CITY_CODE = "city_code"
    ACCESS_CODE = "access_code"


RELATION_ORDER: tuple[RelationId, ...] = tuple(RelationId)


@dataclass(frozen=True, slots=True)
class AcceptedRoute:
    """The only route payload permitted to cross the memory boundary."""

    relation_id: RelationId
    status: Literal["accept"] = field(default="accept", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.relation_id, RelationId):
            raise TypeError("AcceptedRoute requires a RelationId enum value")


@dataclass(frozen=True, slots=True)
class RejectedRoute:
    """A fail-closed decision that is never allowed to access memory."""

    reason: str
    status: Literal["reject"] = field(default="reject", init=False)

    def __post_init__(self) -> None:
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("RejectedRoute reason must be non-empty")
        object.__setattr__(self, "reason", reason)


RouteResult: TypeAlias = AcceptedRoute | RejectedRoute


@dataclass(frozen=True, slots=True, eq=False)
class DiscreteMemoryInput:
    """The complete input visible to exact relation-bucket retrieval.

    ``eq`` is intentionally tensor-aware so contract invariance can be asserted
    directly without exposing any upstream confidence value.
    """

    entity_embedding: Tensor
    relation_id: RelationId

    def __eq__(self, other: object) -> bool:
        return bool(
            isinstance(other, DiscreteMemoryInput)
            and self.relation_id is other.relation_id
            and torch.equal(self.entity_embedding, other.entity_embedding)
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A nearest-prototype result produced from exactly one relation bucket."""

    relation_id: RelationId
    candidate_index: int
    similarity: float
    candidate_count: int
    cross_relation_candidate_count: Literal[0] = 0
    memory_access_count: Literal[1] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.relation_id, RelationId):
            raise TypeError("RetrievalResult requires a RelationId enum value")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not 0 <= self.candidate_index < self.candidate_count:
            raise ValueError("candidate_index is outside the selected bucket")
        if not math.isfinite(float(self.similarity)):
            raise ValueError("retrieval similarity must be finite")
        if self.cross_relation_candidate_count != 0:
            raise ValueError("relation-bucket retrieval cannot expose cross relations")
        if self.memory_access_count != 1:
            raise ValueError("a successful retrieval records exactly one memory access")


@dataclass(frozen=True, slots=True)
class RouteExecution:
    """Auditable result of applying a route before any possible memory access."""

    route: RouteResult
    retrieval: RetrievalResult | None
    memory_access_count: Literal[0, 1]

    def __post_init__(self) -> None:
        if isinstance(self.route, RejectedRoute):
            if self.retrieval is not None or self.memory_access_count != 0:
                raise ValueError("rejected routes must have zero memory access")
        elif isinstance(self.route, AcceptedRoute):
            if self.retrieval is None or self.memory_access_count != 1:
                raise ValueError("accepted routes must contain one retrieval")
        else:  # pragma: no cover - guarded by the constructor's public helpers
            raise TypeError("route must be AcceptedRoute or RejectedRoute")


def _validated_relation_order(
    relation_order: Sequence[RelationId], *, expected_width: int
) -> tuple[RelationId, ...] | None:
    order = tuple(relation_order)
    if len(order) != expected_width or len(set(order)) != len(order):
        return None
    if any(not isinstance(relation, RelationId) for relation in order):
        return None
    # C2.4 is a fixed closed-set audit.  A subset or an extra enum would silently
    # change the contract rather than merely changing display order.
    if set(order) != set(RELATION_ORDER):
        return None
    return order


def _validated_single_logits(logits: Tensor) -> Tensor | None:
    if not isinstance(logits, Tensor) or logits.ndim != 1:
        return None
    values = logits.detach().float()
    if values.numel() != len(RELATION_ORDER) or not bool(torch.isfinite(values).all()):
        return None
    return values


def closed_set_route(
    logits: Tensor,
    *,
    relation_order: Sequence[RelationId] = RELATION_ORDER,
) -> RouteResult:
    """Convert one finite closed-set logit vector into a discrete route.

    Malformed logits or an unknown relation vocabulary return ``RejectedRoute``
    instead of guessing a bucket.
    """

    values = _validated_single_logits(logits)
    if values is None:
        return RejectedRoute("invalid_relation_logits")
    order = _validated_relation_order(relation_order, expected_width=values.numel())
    if order is None:
        return RejectedRoute("unknown_relation_enum")
    index = int(torch.argmax(values).item())
    return AcceptedRoute(order[index])


def route_with_reject_guard(
    logits: Tensor,
    *,
    reject_score: float,
    threshold: float,
    relation_order: Sequence[RelationId] = RELATION_ORDER,
) -> RouteResult:
    """Apply a finite reject score before emitting a closed-set route.

    The score is consumed here and is not carried by ``AcceptedRoute``.  Invalid
    scores and thresholds reject by default.  A score equal to the frozen
    threshold is accepted.
    """

    try:
        score_value = float(reject_score)
    except (TypeError, ValueError, OverflowError):
        return RejectedRoute("invalid_reject_score")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError):
        return RejectedRoute("invalid_reject_threshold")
    if not math.isfinite(score_value):
        return RejectedRoute("invalid_reject_score")
    if not math.isfinite(threshold_value):
        return RejectedRoute("invalid_reject_threshold")
    if score_value < threshold_value:
        return RejectedRoute("below_reject_threshold")
    return closed_set_route(logits, relation_order=relation_order)


def hard_one_hot_from_logits(logits: Tensor) -> Tensor:
    """Return D1's exact hard one-hot relation representation.

    Every row contains exactly one literal ``1`` and otherwise literal ``0``.
    Continuous probabilities are never computed by this helper.
    """

    if not isinstance(logits, Tensor) or logits.ndim not in (1, 2):
        raise ValueError("relation logits must be a vector or matrix")
    values = logits.detach().float()
    if values.shape[-1] != len(RELATION_ORDER):
        raise ValueError("relation logit width does not match the fixed relation set")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("relation logits must be finite")
    indices = torch.argmax(values, dim=-1)
    return F.one_hot(indices, num_classes=len(RELATION_ORDER)).to(torch.float32)


def memory_input(
    entity_embedding: Tensor,
    relation_id: RelationId,
) -> DiscreteMemoryInput:
    """Build the complete memory input; upstream confidence is not an argument."""

    if not isinstance(relation_id, RelationId):
        raise TypeError("memory input requires a RelationId enum value")
    if not isinstance(entity_embedding, Tensor) or entity_embedding.ndim != 1:
        raise ValueError("entity_embedding must be a one-dimensional tensor")
    embedding = entity_embedding.detach().float().clone().contiguous()
    if embedding.numel() == 0 or not bool(torch.isfinite(embedding).all()):
        raise ValueError("entity_embedding must be non-empty and finite")
    if float(torch.linalg.vector_norm(embedding).item()) <= 0.0:
        raise ValueError("entity_embedding must have non-zero norm")
    return DiscreteMemoryInput(
        entity_embedding=embedding,
        relation_id=relation_id,
    )


def retrieve_from_bucket(
    entity_embedding: Tensor,
    relation_id: RelationId,
    bucket_prototypes: Mapping[RelationId, Tensor],
) -> RetrievalResult:
    """Search only the exact bucket selected by ``relation_id``.

    This deliberately narrow public signature is the memory API audited by
    Stage C2.4.  It contains no optional ``**kwargs`` escape hatch.
    """

    # Validate the route before touching either the query or the bucket mapping,
    # so accidentally passing a RejectedRoute cannot cause a memory read.
    if not isinstance(relation_id, RelationId):
        raise TypeError("retrieval requires an accepted RelationId")
    request = memory_input(entity_embedding, relation_id)
    if not isinstance(bucket_prototypes, Mapping):
        raise TypeError("bucket_prototypes must be a relation-to-tensor mapping")
    if relation_id not in bucket_prototypes:
        raise KeyError(f"missing prototype bucket for {relation_id.value}")
    prototypes = bucket_prototypes[relation_id]
    if not isinstance(prototypes, Tensor) or prototypes.ndim != 2:
        raise ValueError("the selected prototype bucket must be a matrix")
    selected = prototypes.detach().float()
    if selected.shape[0] == 0 or selected.shape[1] != request.entity_embedding.numel():
        raise ValueError("selected prototype bucket is empty or has the wrong width")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("selected prototype bucket must be finite")
    prototype_norms = torch.linalg.vector_norm(selected, dim=1)
    if bool((prototype_norms <= 0).any()):
        raise ValueError("selected prototype vectors must have non-zero norm")

    query = request.entity_embedding.to(device=selected.device)
    similarities = F.normalize(selected, dim=1) @ F.normalize(query, dim=0)
    candidate_index = int(torch.argmax(similarities).item())
    return RetrievalResult(
        relation_id=relation_id,
        candidate_index=candidate_index,
        similarity=float(similarities[candidate_index].item()),
        candidate_count=int(selected.shape[0]),
        cross_relation_candidate_count=0,
        memory_access_count=1,
    )


def retrieve_for_route(
    entity_embedding: Tensor,
    route: RouteResult,
    bucket_prototypes: Mapping[RelationId, Tensor],
) -> RouteExecution:
    """Enforce reject-before-memory ordering and record access accounting."""

    if isinstance(route, RejectedRoute):
        # Do not validate or inspect either remaining argument on this branch.
        return RouteExecution(route=route, retrieval=None, memory_access_count=0)
    if not isinstance(route, AcceptedRoute):
        rejected = RejectedRoute("invalid_route_result")
        return RouteExecution(route=rejected, retrieval=None, memory_access_count=0)
    retrieval = retrieve_from_bucket(
        entity_embedding,
        route.relation_id,
        bucket_prototypes,
    )
    return RouteExecution(route=route, retrieval=retrieval, memory_access_count=1)


_FORBIDDEN_MEMORY_SURFACE_NAMES = frozenset(
    {
        "logit",
        "logits",
        "probability",
        "probabilities",
        "softmax",
        "confidence",
        "margin",
        "semantic_embedding",
        "raw_prompt",
        "raw_text",
        "prompt",
        "relation_phrase",
        "phrase_family",
        "family_id",
    }
)


def assert_discrete_memory_contract() -> None:
    """Fail if a future edit widens the audited memory-facing surface."""

    memory_fields = {item.name for item in fields(DiscreteMemoryInput)}
    expected_fields = {"entity_embedding", "relation_id"}
    if memory_fields != expected_fields:
        raise RuntimeError(
            f"DiscreteMemoryInput fields changed: expected {expected_fields}, got {memory_fields}"
        )
    parameters = inspect.signature(retrieve_from_bucket).parameters
    expected_parameters = (
        "entity_embedding",
        "relation_id",
        "bucket_prototypes",
    )
    if tuple(parameters) != expected_parameters:
        raise RuntimeError(
            "retrieve_from_bucket signature changed outside the discrete contract"
        )
    exposed_names = memory_fields.union(parameters)
    forbidden = sorted(exposed_names.intersection(_FORBIDDEN_MEMORY_SURFACE_NAMES))
    if forbidden:
        raise RuntimeError(f"forbidden memory contract fields exposed: {forbidden}")
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters.values()
    ):
        raise RuntimeError("memory retrieval cannot expose variadic arguments")


def memory_contract_exposure_flags() -> dict[str, bool]:
    """Return machine-readable evidence for the D2/D3 contract audit."""

    assert_discrete_memory_contract()
    return {
        "continuous_relation_values_exposed_to_memory": False,
        "confidence_exposed_to_memory": False,
        "semantic_embedding_exposed_to_memory": False,
        "raw_text_exposed_to_memory": False,
        "phrase_family_exposed_to_memory": False,
    }


# Validate the public contract as soon as this module is imported.  This turns a
# future accidental signature expansion into a local, deterministic failure.
assert_discrete_memory_contract()
