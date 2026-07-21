"""Stage C2.4b 的选择性离散 relation 路由器。

本模块位于 semantic evidence 与 memory 之间。连续 evidence、threshold、
candidate set 和 margin 都在本模块内终止；一旦路由被接受，memory 侧只会
收到 C2.4 已冻结的 ``RelationId``、entity embedding 和单个 relation bucket。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from .stage_c24_contract import (
    AcceptedRoute,
    RelationId,
    RetrievalResult,
    retrieve_from_bucket,
)


Aggregation = Literal["mean", "max", "top_k_mean"]
RejectReason = Literal["unknown", "ambiguous", "invalid"]
C24_FROZEN_HEAD_ORDER: tuple[RelationId, ...] = (
    RelationId.ACCESS_CODE,
    RelationId.CITY_CODE,
    RelationId.REGISTRY_ID,
)


@dataclass(frozen=True, slots=True)
class RejectedRoute:
    """C2.4b 的严格 fail-closed 路由结果。"""

    reason: RejectReason
    status: Literal["reject"] = field(default="reject", init=False)

    def __post_init__(self) -> None:
        if self.reason not in {"unknown", "ambiguous", "invalid"}:
            raise ValueError(
                "C2.4b RejectedRoute reason must be unknown, ambiguous, or invalid"
            )


RouteResult: TypeAlias = AcceptedRoute | RejectedRoute


@dataclass(frozen=True, slots=True)
class SelectiveRouteExecution:
    """记录选择性路由是否触发了一次且仅一次 bucket 访问。"""

    route: RouteResult
    retrieval: RetrievalResult | None
    memory_access_count: Literal[0, 1]

    def __post_init__(self) -> None:
        if isinstance(self.route, RejectedRoute):
            if self.retrieval is not None or self.memory_access_count != 0:
                raise ValueError("rejected routes must not access memory")
        elif isinstance(self.route, AcceptedRoute):
            if self.retrieval is None or self.memory_access_count != 1:
                raise ValueError("accepted routes must perform exactly one retrieval")
        else:  # pragma: no cover - dataclass 的公开构造路径已覆盖
            raise TypeError("route must be AcceptedRoute or C2.4b RejectedRoute")


def _as_relation(value: RelationId | str) -> RelationId:
    if isinstance(value, RelationId):
        return value
    try:
        return RelationId(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown relation {value!r}") from exc


def _canonical_relation_mapping(
    values: Mapping[RelationId | str, object],
    *,
    label: str,
) -> dict[RelationId, object]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a relation mapping")
    output: dict[RelationId, object] = {}
    for raw_relation, value in values.items():
        relation = _as_relation(raw_relation)
        if relation in output:
            raise ValueError(f"{label} contains a duplicate relation")
        output[relation] = value
    if set(output) != set(RelationId):
        raise ValueError(f"{label} must contain exactly the frozen relation set")
    return output


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def canonical_evidence_scores(
    scores: Mapping[RelationId | str, float],
) -> dict[RelationId, float]:
    """验证三个独立支持度；不执行 softmax 或跨 relation 归一化。"""

    raw = _canonical_relation_mapping(scores, label="relation evidence")
    return {
        relation: _finite_float(value, label=f"evidence[{relation.value}]")
        for relation, value in raw.items()
    }


def r0_frozen_argmax_route(
    logits: Tensor,
    *,
    relation_order: Sequence[RelationId | str] = C24_FROZEN_HEAD_ORDER,
) -> RouteResult:
    """复现冻结三分类 head 的 argmax，但将任何非法输入映射为 ``invalid``。"""

    if not isinstance(logits, Tensor) or logits.ndim != 1:
        return RejectedRoute("invalid")
    values = logits.detach().cpu().float()
    if values.numel() != len(RelationId) or not bool(torch.isfinite(values).all()):
        return RejectedRoute("invalid")
    try:
        order = tuple(_as_relation(value) for value in relation_order)
    except (TypeError, ValueError):
        return RejectedRoute("invalid")
    if len(order) != len(RelationId) or len(set(order)) != len(order):
        return RejectedRoute("invalid")
    if set(order) != set(RelationId):
        return RejectedRoute("invalid")
    return AcceptedRoute(order[int(torch.argmax(values).item())])


def _normalized_vector(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{label} must be a non-empty vector")
    vector = value.detach().cpu().float()
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label} must be finite")
    if float(torch.linalg.vector_norm(vector).item()) <= 0.0:
        raise ValueError(f"{label} must have non-zero norm")
    return F.normalize(vector, p=2, dim=0)


def _normalized_matrix(value: Tensor, *, width: int, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] != width
    ):
        raise ValueError(f"{label} must be a non-empty matrix with matching width")
    matrix = value.detach().cpu().float()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{label} must be finite")
    if bool((torch.linalg.vector_norm(matrix, dim=1) <= 0.0).any()):
        raise ValueError(f"{label} rows must have non-zero norm")
    return F.normalize(matrix, p=2, dim=1)


def _aggregation_name(value: str) -> Aggregation:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"topk", "topk_mean"}:
        normalized = "top_k_mean"
    if normalized not in {"mean", "max", "top_k_mean"}:
        raise ValueError("definition aggregation must be mean, max, or top-k mean")
    return normalized  # type: ignore[return-value]


def r1_definition_ensemble_scores(
    query_embedding: Tensor,
    relation_definitions: Mapping[RelationId | str, Tensor],
    *,
    aggregation: str = "mean",
    top_k: int | None = None,
) -> dict[RelationId, float]:
    """用多个公开定义原型产生三个互相独立的 cosine evidence。"""

    query = _normalized_vector(query_embedding, label="query embedding")
    raw_definitions = _canonical_relation_mapping(
        relation_definitions, label="relation definitions"
    )
    method = _aggregation_name(aggregation)
    if method == "top_k_mean":
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top-k mean requires a positive integer top_k")
    elif top_k is not None:
        raise ValueError("top_k is only valid for top-k mean aggregation")

    scores: dict[RelationId, float] = {}
    for relation in RelationId:
        definitions = _normalized_matrix(
            raw_definitions[relation],  # type: ignore[arg-type]
            width=query.numel(),
            label=f"definitions[{relation.value}]",
        )
        similarities = definitions @ query
        if method == "mean":
            score = similarities.mean()
        elif method == "max":
            score = similarities.max()
        else:
            assert top_k is not None
            if top_k > len(similarities):
                raise ValueError("top_k exceeds a relation's definition count")
            score = similarities.topk(top_k).values.mean()
        scores[relation] = float(score.item())
    return scores


def _canonical_thresholds(
    relation_threshold: float | Mapping[RelationId | str, float],
) -> dict[RelationId, float]:
    if isinstance(relation_threshold, Mapping):
        raw = _canonical_relation_mapping(
            relation_threshold, label="relation thresholds"
        )
        return {
            relation: _finite_float(value, label=f"threshold[{relation.value}]")
            for relation, value in raw.items()
        }
    threshold = _finite_float(relation_threshold, label="global relation threshold")
    return {relation: threshold for relation in RelationId}


def candidate_set_from_scores(
    scores: Mapping[RelationId | str, float],
    relation_threshold: float | Mapping[RelationId | str, float],
) -> frozenset[RelationId]:
    """按 relation-specific 或全局 threshold 构造集合，不执行 argmax。"""

    evidence = canonical_evidence_scores(scores)
    thresholds = _canonical_thresholds(relation_threshold)
    return frozenset(
        relation
        for relation in RelationId
        if evidence[relation] >= thresholds[relation]
    )


def route_from_candidate_set(
    candidate_set: Sequence[RelationId | str] | frozenset[RelationId],
    *,
    top1_top2_margin: float | None = None,
    minimum_margin: float | None = None,
) -> RouteResult:
    """将 0/1/>1 候选严格映射到 unknown/accept/ambiguous。"""

    required: float | None = None
    if minimum_margin is not None:
        try:
            required = _finite_float(minimum_margin, label="minimum margin")
        except ValueError:
            return RejectedRoute("invalid")
        if required < 0.0:
            return RejectedRoute("invalid")
    try:
        candidates = frozenset(_as_relation(value) for value in candidate_set)
    except (TypeError, ValueError):
        return RejectedRoute("invalid")
    if len(candidates) == 0:
        return RejectedRoute("unknown")
    if len(candidates) > 1:
        return RejectedRoute("ambiguous")
    if required is not None:
        try:
            observed = _finite_float(top1_top2_margin, label="top1-top2 margin")
        except ValueError:
            return RejectedRoute("invalid")
        if observed < required:
            return RejectedRoute("ambiguous")
    return AcceptedRoute(next(iter(candidates)))


def r3_set_valued_route(
    scores: Mapping[RelationId | str, float],
    relation_threshold: float | Mapping[RelationId | str, float],
    *,
    minimum_margin: float | None = None,
) -> RouteResult:
    """基于独立 evidence 进行 0/1/多候选选择性路由。"""

    try:
        evidence = canonical_evidence_scores(scores)
        candidates = candidate_set_from_scores(evidence, relation_threshold)
        # margin 必须围绕唯一候选计算。relation-specific threshold 可能让
        # raw top-1 未过自己的高阈值，而另一 relation 成为唯一候选；此时用
        # 全局 top1-top2 会错误地把候选之外的正 margin 记到候选上。
        if len(candidates) == 1:
            only = next(iter(candidates))
            margin = evidence[only] - max(
                score for relation, score in evidence.items() if relation is not only
            )
        else:
            ordered = sorted(evidence.values(), reverse=True)
            margin = ordered[0] - ordered[1]
    except (TypeError, ValueError, OverflowError):
        return RejectedRoute("invalid")
    return route_from_candidate_set(
        candidates,
        top1_top2_margin=margin,
        minimum_margin=minimum_margin,
    )


def execute_selective_route(
    entity_embedding: Tensor,
    route: RouteResult,
    bucket_prototypes: Mapping[RelationId, Tensor],
) -> SelectiveRouteExecution:
    """先执行 reject；accept 时仅向 C2.4 memory API 传一个 bucket。"""

    if isinstance(route, RejectedRoute):
        # 该分支故意不验证、不读取其余两个参数。
        return SelectiveRouteExecution(route, None, 0)
    if not isinstance(route, AcceptedRoute):
        return SelectiveRouteExecution(RejectedRoute("invalid"), None, 0)
    # 只读取被接受的离散 relation，并构造单项 mapping 后跨越 memory 边界。
    selected_bucket = bucket_prototypes[route.relation_id]
    retrieval = retrieve_from_bucket(
        entity_embedding,
        route.relation_id,
        {route.relation_id: selected_bucket},
    )
    return SelectiveRouteExecution(route, retrieval, 1)


# 清晰的短别名便于 orchestration 与实验表使用 R0/R1/R3 命名。
r0_route = r0_frozen_argmax_route
r1_scores = r1_definition_ensemble_scores
r3_route = r3_set_valued_route
execute_route = execute_selective_route


__all__ = [
    "AcceptedRoute",
    "Aggregation",
    "C24_FROZEN_HEAD_ORDER",
    "RejectReason",
    "RejectedRoute",
    "RelationId",
    "RouteResult",
    "SelectiveRouteExecution",
    "candidate_set_from_scores",
    "canonical_evidence_scores",
    "execute_route",
    "execute_selective_route",
    "r0_frozen_argmax_route",
    "r0_route",
    "r1_definition_ensemble_scores",
    "r1_scores",
    "r3_route",
    "r3_set_valued_route",
    "route_from_candidate_set",
]
