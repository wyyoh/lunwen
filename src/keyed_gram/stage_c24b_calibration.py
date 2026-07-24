"""Stage C2.4b 的独立 evidence head、conformal calibration 与冻结工具。

所有拟合和选择函数只接收 public train/calibration 输入。模块没有
development 或 locked-audit 参数，也不会加载 private answer、optimizer state
或 confirmation 内容。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .stage_c24_contract import RelationId
from .stage_c24b_router import (
    RejectedRoute,
    RouteResult,
    canonical_evidence_scores,
    route_from_candidate_set,
)


PAIRWISE_HEAD_SCHEMA_VERSION = 1
CONFORMAL_SCHEMA_VERSION = 1
FROZEN_SELECTION_SCHEMA_VERSION = 1
NONCONFORMITY_DEFINITION = "negative_independent_relation_support"
_POST_CALIBRATION_TERMS = ("development", "locked", "audit")
_FORBIDDEN_STATE_TERMS = (
    "optimizer",
    "private_answer",
    "private_value",
    "confirmation",
    "permutation",
)


def _as_relation(value: RelationId | str) -> RelationId:
    if isinstance(value, RelationId):
        return value
    try:
        return RelationId(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown relation {value!r}") from exc


def _relation_tensor_mapping(
    values: Mapping[RelationId | str, Tensor],
    *,
    label: str,
    ndim: int,
) -> dict[RelationId, Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a relation mapping")
    output: dict[RelationId, Tensor] = {}
    for raw_relation, raw_value in values.items():
        relation = _as_relation(raw_relation)
        if relation in output:
            raise ValueError(f"{label} contains a duplicate relation")
        if not isinstance(raw_value, Tensor) or raw_value.ndim != ndim:
            raise ValueError(f"{label}[{relation.value}] has an invalid shape")
        value = raw_value.detach().cpu().double().contiguous()
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label}[{relation.value}] must be non-empty and finite")
        output[relation] = value
    if set(output) != set(RelationId):
        raise ValueError(f"{label} must contain exactly the frozen relation set")
    return output


def _target_relations(
    values: Sequence[RelationId | str],
) -> tuple[RelationId, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("relation targets must be a non-empty sequence")
    return tuple(_as_relation(value) for value in values)


def build_pairwise_relation_features(
    query_embeddings: Tensor,
    relation_definition_embeddings: Mapping[RelationId | str, Tensor],
) -> dict[RelationId, Tensor]:
    """构造冻结 query/definition 对特征，不混合不同 relation 的分数。

    每个 relation 的特征为 ``[q, d, |q-d|, q*d, cosine(q,d)]``，其中 ``d``
    是该 relation 多个公开定义 embedding 的归一化均值原型。
    """

    if not isinstance(query_embeddings, Tensor) or query_embeddings.ndim not in (1, 2):
        raise ValueError("query embeddings must be a vector or matrix")
    queries = query_embeddings.detach().cpu().float()
    single = queries.ndim == 1
    if single:
        queries = queries.unsqueeze(0)
    if queries.shape[0] == 0 or queries.shape[1] == 0:
        raise ValueError("query embeddings cannot be empty")
    if not bool(torch.isfinite(queries).all()):
        raise ValueError("query embeddings must be finite")
    norms = torch.linalg.vector_norm(queries, dim=1)
    if bool((norms <= 0.0).any()):
        raise ValueError("query embeddings must have non-zero norm")
    queries = F.normalize(queries, p=2, dim=1)

    if not isinstance(relation_definition_embeddings, Mapping):
        raise TypeError("relation definition embeddings must be a mapping")
    definitions: dict[RelationId, Tensor] = {}
    for raw_relation, raw_value in relation_definition_embeddings.items():
        relation = _as_relation(raw_relation)
        if relation in definitions:
            raise ValueError("relation definition embeddings contain duplicates")
        if not isinstance(raw_value, Tensor) or raw_value.ndim not in (1, 2):
            raise ValueError("definition embeddings must be vectors or matrices")
        matrix = raw_value.detach().cpu().float()
        if matrix.ndim == 1:
            matrix = matrix.unsqueeze(0)
        if matrix.shape[0] == 0 or matrix.shape[1] != queries.shape[1]:
            raise ValueError("definition embeddings have an incompatible shape")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("definition embeddings must be finite")
        if bool((torch.linalg.vector_norm(matrix, dim=1) <= 0.0).any()):
            raise ValueError("definition embeddings must have non-zero norm")
        normalized = F.normalize(matrix, p=2, dim=1)
        definition_mean = normalized.mean(dim=0)
        if float(torch.linalg.vector_norm(definition_mean).item()) <= 0.0:
            raise ValueError("definition ensemble mean must have non-zero norm")
        prototype = F.normalize(definition_mean, p=2, dim=0)
        definitions[relation] = prototype
    if set(definitions) != set(RelationId):
        raise ValueError("definition embeddings must cover the frozen relation set")

    output: dict[RelationId, Tensor] = {}
    for relation in RelationId:
        prototype = definitions[relation].expand(len(queries), -1)
        cosine = (queries * prototype).sum(dim=1, keepdim=True)
        features = torch.cat(
            [
                queries,
                prototype,
                torch.abs(queries - prototype),
                queries * prototype,
                cosine,
            ],
            dim=1,
        )
        output[relation] = features[0] if single else features
    return output


@dataclass(frozen=True, slots=True)
class PairwiseRidgeEvidenceHead:
    """三个互相独立的一对其余公共 ridge evidence head。"""

    relation_order: tuple[RelationId, ...]
    feature_mean: Mapping[RelationId, Tensor]
    feature_scale: Mapping[RelationId, Tensor]
    weights: Mapping[RelationId, Tensor]
    regularizers: Mapping[RelationId, float]

    def __post_init__(self) -> None:
        if tuple(self.relation_order) != tuple(RelationId):
            raise ValueError("pairwise head relation order must be the frozen enum order")
        for name, mapping in (
            ("feature_mean", self.feature_mean),
            ("feature_scale", self.feature_scale),
            ("weights", self.weights),
            ("regularizers", self.regularizers),
        ):
            if set(mapping) != set(RelationId):
                raise ValueError(f"pairwise head {name} has an invalid relation set")
        for relation in RelationId:
            mean = self.feature_mean[relation]
            scale = self.feature_scale[relation]
            weights = self.weights[relation]
            if (
                not isinstance(mean, Tensor)
                or mean.ndim != 1
                or not isinstance(scale, Tensor)
                or scale.shape != mean.shape
                or not isinstance(weights, Tensor)
                or weights.shape != (mean.numel() + 1,)
            ):
                raise ValueError("pairwise head tensors have incompatible shapes")
            if not all(
                bool(torch.isfinite(value).all()) for value in (mean, scale, weights)
            ):
                raise ValueError("pairwise head tensors must be finite")
            if bool((scale <= 0.0).any()):
                raise ValueError("pairwise feature scales must be positive")
            regularizer = float(self.regularizers[relation])
            if not math.isfinite(regularizer) or regularizer <= 0.0:
                raise ValueError("pairwise ridge regularizers must be finite and positive")

    def score_batch(
        self,
        pair_features: Mapping[RelationId | str, Tensor],
    ) -> dict[RelationId, Tensor]:
        matrices = _relation_tensor_mapping(
            pair_features, label="pair features", ndim=2
        )
        row_counts = {len(value) for value in matrices.values()}
        if len(row_counts) != 1:
            raise ValueError("pair feature matrices must have aligned rows")
        output: dict[RelationId, Tensor] = {}
        for relation in RelationId:
            matrix = matrices[relation]
            mean = self.feature_mean[relation].detach().cpu().double()
            scale = self.feature_scale[relation].detach().cpu().double()
            weights = self.weights[relation].detach().cpu().double()
            if matrix.shape[1] != mean.numel():
                raise ValueError("pair feature width differs from the fitted head")
            standardized = (matrix - mean) / scale
            augmented = torch.cat(
                [
                    standardized,
                    torch.ones((len(matrix), 1), dtype=standardized.dtype),
                ],
                dim=1,
            )
            values = (augmented @ weights).float()
            if not bool(torch.isfinite(values).all()):
                raise ValueError("pairwise evidence head produced non-finite scores")
            output[relation] = values
        return output

    def score_one(
        self,
        pair_features: Mapping[RelationId | str, Tensor],
    ) -> dict[RelationId, float]:
        if not isinstance(pair_features, Mapping):
            raise TypeError("pair features must be a relation mapping")
        batched = {
            key: value.unsqueeze(0) if isinstance(value, Tensor) and value.ndim == 1 else value
            for key, value in pair_features.items()
        }
        scores = self.score_batch(batched)  # type: ignore[arg-type]
        if any(len(value) != 1 for value in scores.values()):
            raise ValueError("score_one requires one row per relation")
        return {relation: float(scores[relation][0]) for relation in RelationId}

    def to_state_dict(self) -> dict[str, Any]:
        """返回仅含公开拟合张量和标量的可序列化状态。"""

        return {
            "schema_version": PAIRWISE_HEAD_SCHEMA_VERSION,
            "stage": "C2.4b-R2-public-pairwise-ridge-evidence-head",
            "relation_order": [relation.value for relation in self.relation_order],
            "feature_mean": {
                relation.value: self.feature_mean[relation].detach().cpu().clone()
                for relation in RelationId
            },
            "feature_scale": {
                relation.value: self.feature_scale[relation].detach().cpu().clone()
                for relation in RelationId
            },
            "weights": {
                relation.value: self.weights[relation].detach().cpu().clone()
                for relation in RelationId
            },
            "regularizers": {
                relation.value: float(self.regularizers[relation])
                for relation in RelationId
            },
            "optimizer_state_present": False,
            "private_data_present": False,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return _json_ready(self.to_state_dict())

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "PairwiseRidgeEvidenceHead":
        expected = {
            "schema_version",
            "stage",
            "relation_order",
            "feature_mean",
            "feature_scale",
            "weights",
            "regularizers",
            "optimizer_state_present",
            "private_data_present",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("pairwise head state has an invalid schema")
        if (
            payload["schema_version"] != PAIRWISE_HEAD_SCHEMA_VERSION
            or payload["stage"] != "C2.4b-R2-public-pairwise-ridge-evidence-head"
            or payload["optimizer_state_present"] is not False
            or payload["private_data_present"] is not False
        ):
            raise ValueError("pairwise head state metadata is invalid")
        order = tuple(_as_relation(value) for value in payload["relation_order"])

        def tensors(name: str) -> dict[RelationId, Tensor]:
            raw = payload[name]
            if not isinstance(raw, Mapping):
                raise ValueError(f"pairwise head {name} is malformed")
            output = {}
            for key, value in raw.items():
                relation = _as_relation(key)
                output[relation] = (
                    value.detach().cpu().double().clone()
                    if isinstance(value, Tensor)
                    else torch.tensor(value, dtype=torch.float64)
                )
            return output

        raw_regularizers = payload["regularizers"]
        if not isinstance(raw_regularizers, Mapping):
            raise ValueError("pairwise head regularizers are malformed")
        regularizers = {
            _as_relation(key): float(value)
            for key, value in raw_regularizers.items()
        }
        return cls(
            relation_order=order,
            feature_mean=tensors("feature_mean"),
            feature_scale=tensors("feature_scale"),
            weights=tensors("weights"),
            regularizers=regularizers,
        )


def fit_pairwise_ridge_head(
    train_pair_features: Mapping[RelationId | str, Tensor],
    train_targets: Sequence[RelationId | str],
    *,
    ridge_strength: float = 0.01,
) -> PairwiseRidgeEvidenceHead:
    """仅用 public train 拟合三个独立二分类 ridge evidence head。"""

    strength = float(ridge_strength)
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError("ridge strength must be finite and positive")
    features = _relation_tensor_mapping(
        train_pair_features, label="train pair features", ndim=2
    )
    targets = _target_relations(train_targets)
    if any(len(matrix) != len(targets) for matrix in features.values()):
        raise ValueError("train pair features and targets must be aligned")

    means: dict[RelationId, Tensor] = {}
    scales: dict[RelationId, Tensor] = {}
    weights: dict[RelationId, Tensor] = {}
    regularizers: dict[RelationId, float] = {}
    for relation in RelationId:
        matrix = features[relation]
        binary = torch.tensor(
            [1.0 if target is relation else 0.0 for target in targets],
            dtype=torch.float64,
        )
        if int(binary.sum()) == 0 or int(binary.sum()) == len(binary):
            raise ValueError("every relation needs positive and negative public rows")
        mean = matrix.mean(dim=0)
        scale = matrix.std(dim=0, unbiased=False).clamp_min(1e-5)
        standardized = (matrix - mean) / scale
        augmented = torch.cat(
            [
                standardized,
                torch.ones((len(matrix), 1), dtype=standardized.dtype),
            ],
            dim=1,
        )
        kernel = augmented @ augmented.T
        regularizer = max(float(kernel.diagonal().mean()) * strength, 1e-8)
        kernel.diagonal().add_(regularizer)
        coefficients = torch.linalg.solve(kernel, binary)
        relation_weights = augmented.T @ coefficients
        means[relation] = mean
        scales[relation] = scale
        weights[relation] = relation_weights
        regularizers[relation] = regularizer
    return PairwiseRidgeEvidenceHead(
        relation_order=tuple(RelationId),
        feature_mean=means,
        feature_scale=scales,
        weights=weights,
        regularizers=regularizers,
    )


def higher_conformal_quantile(
    nonconformity_scores: Tensor,
    *,
    alpha: float,
) -> tuple[float, int]:
    """有限样本 split-conformal higher quantile。

    rank 为 ``ceil((n + 1) * (1 - alpha))``，并在有限 calibration 样本数
    ``n`` 处截断；返回 1-based rank。
    """

    level = float(alpha)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("conformal alpha must be in (0, 1)")
    if not isinstance(nonconformity_scores, Tensor):
        raise TypeError("nonconformity scores must be a tensor")
    values = nonconformity_scores.detach().cpu().double().flatten()
    if len(values) == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("nonconformity scores must be non-empty and finite")
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * (1.0 - level))))
    quantile = float(values.sort().values[rank - 1])
    return quantile, rank


def _score_rows(
    values: Tensor | Sequence[Mapping[RelationId | str, float]],
    *,
    relation_order: Sequence[RelationId | str],
) -> Tensor:
    order = tuple(_as_relation(value) for value in relation_order)
    if len(order) != len(RelationId) or len(set(order)) != len(order):
        raise ValueError("score relation order must contain three unique relations")
    if set(order) != set(RelationId):
        raise ValueError("score relation order differs from the frozen relation set")
    if isinstance(values, Tensor):
        matrix = values.detach().cpu().double()
        if matrix.ndim != 2 or matrix.shape[1] != len(order) or len(matrix) == 0:
            raise ValueError("evidence scores must be a non-empty [rows, relations] matrix")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("evidence scores must be finite")
        canonical_indices = [order.index(relation) for relation in RelationId]
        return matrix[:, canonical_indices]
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("evidence score rows cannot be empty")
    rows = [canonical_evidence_scores(row) for row in values]
    return torch.tensor(
        [[row[relation] for relation in RelationId] for row in rows],
        dtype=torch.float64,
    )


@dataclass(frozen=True, slots=True)
class ClassConditionalConformalCalibrator:
    """R4 的 class-conditional split-conformal prediction-set calibrator。"""

    alpha: float
    relation_order: tuple[RelationId, ...]
    nonconformity: str
    quantiles: Mapping[RelationId, float]
    calibration_counts: Mapping[RelationId, int]
    quantile_ranks: Mapping[RelationId, int]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.alpha)) or not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("conformal alpha must be in (0, 1)")
        if self.relation_order != tuple(RelationId):
            raise ValueError("conformal relation order must be the frozen enum order")
        if self.nonconformity != NONCONFORMITY_DEFINITION:
            raise ValueError("unknown conformal nonconformity definition")
        for mapping in (self.quantiles, self.calibration_counts, self.quantile_ranks):
            if set(mapping) != set(RelationId):
                raise ValueError("conformal state has an invalid relation set")
        for relation in RelationId:
            if not math.isfinite(float(self.quantiles[relation])):
                raise ValueError("conformal quantiles must be finite")
            count = int(self.calibration_counts[relation])
            rank = int(self.quantile_ranks[relation])
            if count <= 0 or not 1 <= rank <= count:
                raise ValueError("conformal calibration count/rank is invalid")

    def candidate_set(
        self,
        scores: Mapping[RelationId | str, float],
    ) -> frozenset[RelationId]:
        evidence = canonical_evidence_scores(scores)
        return frozenset(
            relation
            for relation in RelationId
            if -evidence[relation] <= float(self.quantiles[relation])
        )

    def route(
        self,
        scores: Mapping[RelationId | str, float],
    ) -> RouteResult:
        try:
            candidates = self.candidate_set(scores)
        except (TypeError, ValueError, OverflowError):
            return RejectedRoute("invalid")
        return route_from_candidate_set(candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFORMAL_SCHEMA_VERSION,
            "method": "class_conditional_split_conformal_prediction_set",
            "alpha": float(self.alpha),
            "target_coverage": 1.0 - float(self.alpha),
            "relation_order": [relation.value for relation in self.relation_order],
            "nonconformity": self.nonconformity,
            "quantiles": {
                relation.value: float(self.quantiles[relation])
                for relation in RelationId
            },
            "calibration_counts": {
                relation.value: int(self.calibration_counts[relation])
                for relation in RelationId
            },
            "quantile_ranks": {
                relation.value: int(self.quantile_ranks[relation])
                for relation in RelationId
            },
            "metadata": _json_ready(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ClassConditionalConformalCalibrator":
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != CONFORMAL_SCHEMA_VERSION
            or payload.get("method")
            != "class_conditional_split_conformal_prediction_set"
        ):
            raise ValueError("conformal payload metadata is invalid")
        return cls(
            alpha=float(payload["alpha"]),
            relation_order=tuple(
                _as_relation(value) for value in payload["relation_order"]
            ),
            nonconformity=str(payload["nonconformity"]),
            quantiles={
                _as_relation(key): float(value)
                for key, value in payload["quantiles"].items()
            },
            calibration_counts={
                _as_relation(key): int(value)
                for key, value in payload["calibration_counts"].items()
            },
            quantile_ranks={
                _as_relation(key): int(value)
                for key, value in payload["quantile_ranks"].items()
            },
            metadata=dict(payload["metadata"]),
        )


def fit_class_conditional_conformal(
    calibration_scores: Tensor | Sequence[Mapping[RelationId | str, float]],
    calibration_targets: Sequence[RelationId | str | None],
    *,
    alpha: float = 0.10,
    relation_order: Sequence[RelationId | str] = tuple(RelationId),
) -> ClassConditionalConformalCalibrator:
    """只用 calibration known rows 拟合每个 relation 的 higher quantile。"""

    matrix = _score_rows(calibration_scores, relation_order=relation_order)
    if len(matrix) != len(calibration_targets):
        raise ValueError("calibration scores and targets must be aligned")
    targets: list[RelationId | None] = []
    for value in calibration_targets:
        targets.append(None if value is None else _as_relation(value))
    quantiles: dict[RelationId, float] = {}
    counts: dict[RelationId, int] = {}
    ranks: dict[RelationId, int] = {}
    for column, relation in enumerate(RelationId):
        indices = [index for index, target in enumerate(targets) if target is relation]
        if not indices:
            raise ValueError(f"calibration has no known rows for {relation.value}")
        nonconformity = -matrix[indices, column]
        quantile, rank = higher_conformal_quantile(nonconformity, alpha=alpha)
        quantiles[relation] = quantile
        counts[relation] = len(indices)
        ranks[relation] = rank
    metadata = {
        "calibration_rows_used": "known_rows_grouped_by_true_relation_only",
        "open_set_rows_used_for_quantiles": False,
        "quantile_rule": "ceil((n+1)*(1-alpha)) higher empirical quantile, clipped at n",
        "finite_sample_scope": (
            "class-conditional marginal coverage requires exchangeability and a "
            "scorer/evidence source/alpha fixed independently before this calibration set"
        ),
        "nominal_finite_sample_guarantee_claimed": False,
        "limitations": [
            "selecting the evidence source or alpha on these same calibration rows "
            "breaks the standard nominal split-conformal guarantee",
            "no strict coverage guarantee under lexical or semantic distribution shift",
            "no per-family conditional coverage guarantee",
            "independent evidence scores are not softmax probabilities",
            "prediction-set validity does not establish deployment or cryptographic safety",
        ],
        "post_calibration_parameter_adjustment_allowed": False,
    }
    return ClassConditionalConformalCalibrator(
        alpha=float(alpha),
        relation_order=tuple(RelationId),
        nonconformity=NONCONFORMITY_DEFINITION,
        quantiles=quantiles,
        calibration_counts=counts,
        quantile_ranks=ranks,
        metadata=metadata,
    )


def _reject_post_calibration_inputs(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(term in lowered for term in _POST_CALIBRATION_TERMS):
                raise ValueError(f"post-calibration input is prohibited at {path}.{key}")
            _reject_post_calibration_inputs(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_post_calibration_inputs(child, path=f"{path}[{index}]")


def _validate_frozen_protocol(value: Any, *, path: str) -> None:
    """冻结记录可以保存 post-calibration 未参与选择的显式 false 证明。"""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(term in lowered for term in _POST_CALIBRATION_TERMS):
                if child is not False:
                    raise ValueError(
                        f"post-calibration participation must be false at {path}.{key}"
                    )
                continue
            _validate_frozen_protocol(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_frozen_protocol(child, path=f"{path}[{index}]")


def select_router_on_calibration(
    calibration_candidates: Mapping[str, Mapping[str, Any]],
    *,
    gate_targets: Mapping[str, float] | None = None,
    preference_order: Sequence[str] = ("R4", "R3", "R2", "R1", "R0"),
) -> dict[str, Any]:
    """按预注册 access-control 指标确定性选择 router，仅接受 calibration 结果。"""

    if any(
        token in name.casefold()
        for name in inspect.signature(select_router_on_calibration).parameters
        for token in _POST_CALIBRATION_TERMS
    ):
        raise RuntimeError("router selection API exposed a post-calibration argument")
    if not isinstance(calibration_candidates, Mapping) or not calibration_candidates:
        raise ValueError("calibration candidates cannot be empty")
    _reject_post_calibration_inputs(calibration_candidates, path="candidates")
    targets = {
        "known_coverage": 0.90,
        "accepted_relation_precision": 0.97,
        "singleton_acceptance_precision": 0.97,
        "safe_coverage": 0.88,
        "false_memory_access_rate": 0.10,
        "ambiguous_rejection_rate": 0.80,
        "unrelated_rejection_rate": 0.95,
        "ambiguous_false_memory_access_rate": 0.20,
        "unrelated_false_memory_access_rate": 0.05,
        "wrong_bucket_access_rate": 0.03,
        "worst_reject_family_false_accept_rate": 0.30,
        "family_macro_accuracy": 0.95,
        "worst_family_accuracy": 0.85,
        "true_relation_inclusion_rate": 0.85,
        "average_candidate_set_size": 1.30,
        "known_multi_relation_set_rate": 0.15,
        "known_empty_set_rate": 0.10,
    }
    if gate_targets is not None:
        unknown = set(gate_targets) - set(targets)
        if unknown:
            raise ValueError(f"unknown calibration gate targets: {sorted(unknown)}")
        targets.update({key: float(value) for key, value in gate_targets.items()})
    if any(not math.isfinite(value) for value in targets.values()):
        raise ValueError("calibration gate targets must be finite")
    preference = {name: len(preference_order) - index for index, name in enumerate(preference_order)}
    if len(preference) != len(preference_order):
        raise ValueError("router preference order contains duplicates")

    ranked: list[tuple[tuple[float, ...], str]] = []
    rows: dict[str, dict[str, Any]] = {}
    required = (
        "known_coverage",
        "accepted_relation_precision",
        "singleton_acceptance_precision",
        "safe_coverage",
        "false_memory_access_rate",
        "ambiguous_rejection_rate",
        "unrelated_rejection_rate",
        "ambiguous_false_memory_access_rate",
        "unrelated_false_memory_access_rate",
        "wrong_bucket_access_rate",
        "worst_reject_family_false_accept_rate",
        "family_macro_accuracy",
        "worst_family_accuracy",
        "true_relation_inclusion_rate",
        "average_candidate_set_size",
        "known_multi_relation_set_rate",
        "known_empty_set_rate",
    )
    for name, raw_metrics in calibration_candidates.items():
        if not isinstance(raw_metrics, Mapping) or not set(required).issubset(raw_metrics):
            raise ValueError(f"calibration candidate {name} lacks required metrics")
        metrics = {
            key: float(raw_metrics[key])
            for key in required
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"calibration candidate {name} has non-finite metrics")
        gates = {
            "known_coverage": metrics["known_coverage"] >= targets["known_coverage"],
            "accepted_relation_precision": metrics["accepted_relation_precision"]
            >= targets["accepted_relation_precision"],
            "singleton_acceptance_precision": metrics[
                "singleton_acceptance_precision"
            ]
            >= targets["singleton_acceptance_precision"],
            "safe_coverage": metrics["safe_coverage"] >= targets["safe_coverage"],
            "false_memory_access_rate": metrics["false_memory_access_rate"]
            <= targets["false_memory_access_rate"],
            "wrong_bucket_access_rate": metrics["wrong_bucket_access_rate"]
            <= targets["wrong_bucket_access_rate"],
            "ambiguous_rejection_rate": metrics["ambiguous_rejection_rate"]
            >= targets["ambiguous_rejection_rate"],
            "unrelated_rejection_rate": metrics["unrelated_rejection_rate"]
            >= targets["unrelated_rejection_rate"],
            "ambiguous_false_memory_access_rate": metrics[
                "ambiguous_false_memory_access_rate"
            ]
            <= targets["ambiguous_false_memory_access_rate"],
            "unrelated_false_memory_access_rate": metrics[
                "unrelated_false_memory_access_rate"
            ]
            <= targets["unrelated_false_memory_access_rate"],
            "worst_reject_family_false_accept_rate": metrics[
                "worst_reject_family_false_accept_rate"
            ]
            <= targets["worst_reject_family_false_accept_rate"],
            "family_macro_accuracy": metrics["family_macro_accuracy"]
            >= targets["family_macro_accuracy"],
            "worst_family_accuracy": metrics["worst_family_accuracy"]
            >= targets["worst_family_accuracy"],
            "true_relation_inclusion_rate": metrics[
                "true_relation_inclusion_rate"
            ]
            >= targets["true_relation_inclusion_rate"],
            "average_candidate_set_size": metrics["average_candidate_set_size"]
            <= targets["average_candidate_set_size"],
            "known_multi_relation_set_rate": metrics[
                "known_multi_relation_set_rate"
            ]
            <= targets["known_multi_relation_set_rate"],
            "known_empty_set_rate": metrics["known_empty_set_rate"]
            <= targets["known_empty_set_rate"],
        }
        gate_count = sum(gates.values())
        objective = (
            float(all(gates.values())),
            float(gate_count),
            metrics["safe_coverage"],
            metrics["singleton_acceptance_precision"],
            metrics["accepted_relation_precision"],
            metrics["ambiguous_rejection_rate"],
            metrics["unrelated_rejection_rate"],
            -metrics["false_memory_access_rate"],
            -metrics["wrong_bucket_access_rate"],
            metrics["family_macro_accuracy"],
            metrics["worst_family_accuracy"],
            -metrics["average_candidate_set_size"],
            float(preference.get(str(name), 0)),
        )
        rows[str(name)] = {
            "router": str(name),
            "metrics": metrics,
            "gates": gates,
            "gate_count": gate_count,
            "all_gates_passed": all(gates.values()),
            "selection_objective": list(objective),
        }
        ranked.append((objective, str(name)))
    selected = max(ranked)[1]
    return {
        "selected_router": selected,
        "selected_candidate": rows[selected],
        "candidates": rows,
        "gate_targets": targets,
        "selection_objective_order": [
            "all_access_control_gates_passed",
            "gate_count",
            "higher_safe_coverage",
            "higher_singleton_acceptance_precision",
            "higher_accepted_relation_precision",
            "higher_ambiguous_rejection_rate",
            "higher_unrelated_rejection_rate",
            "lower_false_memory_access_rate",
            "lower_wrong_bucket_access_rate",
            "higher_family_macro_accuracy",
            "higher_worst_family_accuracy",
            "lower_average_candidate_set_size",
            "preregistered_router_preference",
        ],
        "selection_provenance": {
            "selection_split": "public_calibration_v2",
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
        },
    }


def _json_ready(value: Any, *, path: str = "root") -> Any:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite tensor at {path}")
        return tensor.tolist()
    if isinstance(value, RelationId):
        return value.value
    if isinstance(value, Mapping):
        output = {}
        for key, child in value.items():
            normalized_key = key.value if isinstance(key, RelationId) else str(key)
            lowered = normalized_key.casefold()
            if any(term in lowered for term in _FORBIDDEN_STATE_TERMS):
                # 明确为 false 的审计布尔值可以保留，但绝不能携带对应状态。
                if child is not False:
                    raise ValueError(f"forbidden state at {path}.{normalized_key}")
            output[normalized_key] = _json_ready(
                child, path=f"{path}.{normalized_key}"
            )
        return output
    if isinstance(value, (list, tuple)):
        return [_json_ready(child, path=f"{path}[{index}]") for index, child in enumerate(value)]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
        return value
    raise TypeError(f"unsupported frozen value at {path}: {type(value).__name__}")


def canonical_selection_sha256(value: Any) -> str:
    ready = _json_ready(value)
    encoded = json.dumps(
        ready,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenRouterSelection:
    """以 canonical JSON 字节保存的不可变 router 冻结记录。"""

    canonical_json: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise ValueError("frozen router payload is not an object")
        return value

    def verify(self, expected_sha256: str | None = None) -> bool:
        observed = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return observed == self.sha256 and (
            expected_sha256 is None or observed == str(expected_sha256)
        )


def freeze_router_selection(
    selected_router: str,
    router_parameters: Mapping[str, Any],
    *,
    public_train_sha256: str,
    public_calibration_sha256: str,
    review_manifest_sha256: str,
    git_commit: str,
    model_manifests: Mapping[str, Any],
    selection_protocol: Mapping[str, Any] | None = None,
) -> FrozenRouterSelection:
    """冻结模型、threshold/conformal 参数与其全部选择来源。"""

    if not str(selected_router).strip():
        raise ValueError("selected router cannot be empty")
    hashes = {
        "public_train_v2": str(public_train_sha256),
        "public_calibration_v2": str(public_calibration_sha256),
        "review_manifest": str(review_manifest_sha256),
    }
    if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in hashes.values()):
        raise ValueError("frozen data hashes must be lowercase SHA-256 strings")
    if len(str(git_commit)) < 7:
        raise ValueError("frozen git commit is malformed")
    protocol = dict(selection_protocol or {})
    _validate_frozen_protocol(protocol, path="selection_protocol")
    payload = _json_ready(
        {
            "schema_version": FROZEN_SELECTION_SCHEMA_VERSION,
            "stage": "C2.4b-selective-discrete-relation-router",
            "selected_router": str(selected_router),
            "router_parameters": router_parameters,
            "data_sha256": hashes,
            "git_commit": str(git_commit),
            "model_manifests": model_manifests,
            "selection_protocol": {
                **protocol,
                "selection_split": "public_calibration_v2",
                "development_used_for_selection": False,
                "locked_audit_used_for_selection": False,
                "parameters_frozen": True,
            },
        }
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrozenRouterSelection(canonical_json=canonical, sha256=digest)


def verify_frozen_router_selection(
    frozen: FrozenRouterSelection,
    *,
    expected_sha256: str | None = None,
) -> bool:
    if not isinstance(frozen, FrozenRouterSelection):
        return False
    return frozen.verify(expected_sha256)


# 实验表中的简短别名。
fit_r2_pairwise_head = fit_pairwise_ridge_head
fit_r4_conformal = fit_class_conditional_conformal
select_calibrated_router = select_router_on_calibration


__all__ = [
    "ClassConditionalConformalCalibrator",
    "FrozenRouterSelection",
    "NONCONFORMITY_DEFINITION",
    "PAIRWISE_HEAD_SCHEMA_VERSION",
    "PairwiseRidgeEvidenceHead",
    "build_pairwise_relation_features",
    "canonical_selection_sha256",
    "fit_class_conditional_conformal",
    "fit_pairwise_ridge_head",
    "fit_r2_pairwise_head",
    "fit_r4_conformal",
    "freeze_router_selection",
    "higher_conformal_quantile",
    "select_calibrated_router",
    "select_router_on_calibration",
    "verify_frozen_router_selection",
]
