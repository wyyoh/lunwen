"""Stage C2.5 的任意 intent R0、R2、max-threshold 与集合式 R3。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .stage_c25_protocol import C25ProtocolError, canonical_sha256


@dataclass(frozen=True, slots=True)
class IntentDecision:
    status: str
    predicted_intent: str | None
    reason: str | None
    candidate_set: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"accept", "reject"}:
            raise ValueError("decision status 必须为 accept/reject")
        if self.status == "accept":
            if (
                not self.predicted_intent
                or self.reason is not None
                or len(self.candidate_set) != 1
                or self.candidate_set[0] != self.predicted_intent
            ):
                raise ValueError("accepted decision schema 非法")
        elif (
            self.predicted_intent is not None
            or self.reason not in {"unknown", "ambiguous", "invalid"}
        ):
            raise ValueError("rejected decision schema 非法")


@dataclass(frozen=True, slots=True)
class MulticlassRidge:
    classes: tuple[str, ...]
    feature_mean: Tensor
    feature_scale: Tensor
    weights: Tensor
    regularizer: float

    def score(self, features: Tensor) -> Tensor:
        matrix = _finite_matrix(features, "R0 features").double()
        standardized = (matrix - self.feature_mean) / self.feature_scale
        augmented = torch.cat(
            (
                standardized,
                torch.ones((len(matrix), 1), dtype=torch.float64),
            ),
            dim=1,
        )
        return (augmented @ self.weights).float()

    def state_sha256(self) -> str:
        return _tensor_state_sha(
            {
                "classes": self.classes,
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "weights": self.weights,
                "regularizer": self.regularizer,
            }
        )


@dataclass(frozen=True, slots=True)
class PairwiseEvidenceModel:
    intents: tuple[str, ...]
    centroids: Tensor
    prototypes: Tensor
    label_embeddings: Tensor
    feature_mean: Tensor
    feature_scale: Tensor
    weights: Tensor
    regularizer: float
    prototype_top_k: int

    def score(self, query_embeddings: Tensor) -> Tensor:
        features = pairwise_features(
            query_embeddings,
            self.centroids,
            self.prototypes,
            self.label_embeddings,
            top_k=self.prototype_top_k,
        )
        standardized = (
            features.double() - self.feature_mean
        ) / self.feature_scale
        augmented = torch.cat(
            (
                standardized,
                torch.ones((*standardized.shape[:2], 1), dtype=torch.float64),
            ),
            dim=2,
        )
        return torch.einsum("ncf,f->nc", augmented, self.weights).float()

    def state_sha256(self) -> str:
        return _tensor_state_sha(
            {
                "intents": self.intents,
                "centroids": self.centroids,
                "prototypes": self.prototypes,
                "label_embeddings": self.label_embeddings,
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "weights": self.weights,
                "regularizer": self.regularizer,
                "prototype_top_k": self.prototype_top_k,
            }
        )


def _finite_matrix(value: Tensor, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or not value.numel():
        raise ValueError(f"{label} 必须是非空二维 tensor")
    matrix = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{label} 含 NaN/Infinity")
    if bool((torch.linalg.vector_norm(matrix, dim=1) <= 0).any()):
        raise ValueError(f"{label} 含零向量")
    return matrix


def _tensor_state_sha(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    metadata: dict[str, Any] = {}
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            metadata[key] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        else:
            metadata[key] = value
    digest.update(canonical_sha256(metadata).encode("ascii"))
    return digest.hexdigest()


def fit_multiclass_ridge(
    embeddings: Tensor,
    labels: Sequence[str],
    *,
    ridge_strength: float,
    classes: Sequence[str] | None = None,
) -> MulticlassRidge:
    matrix = _finite_matrix(embeddings, "R0 train embeddings").double()
    if len(matrix) != len(labels):
        raise ValueError("R0 train embeddings/labels 不对齐")
    names = tuple(sorted(set(labels)) if classes is None else classes)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("R0 至少需要两个唯一 intent")
    lookup = {label: index for index, label in enumerate(names)}
    if any(label not in lookup for label in labels):
        raise ValueError("R0 label 不在 class list")
    strength = float(ridge_strength)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("R0 ridge strength 非法")
    targets = F.one_hot(
        torch.tensor([lookup[label] for label in labels]),
        num_classes=len(names),
    ).double()
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False).clamp_min(1e-5)
    standardized = (matrix - mean) / scale
    augmented = torch.cat(
        (standardized, torch.ones((len(matrix), 1), dtype=torch.float64)),
        dim=1,
    )
    gram = augmented.T @ augmented
    regularizer = max(float(gram.diagonal().mean()) * strength, 1e-8)
    gram.diagonal().add_(regularizer)
    weights = torch.linalg.solve(gram, augmented.T @ targets)
    return MulticlassRidge(names, mean, scale, weights, regularizer)


def _farthest_prototypes(
    embeddings: Tensor, count: int
) -> Tensor:
    matrix = F.normalize(embeddings.float(), p=2, dim=1)
    centroid = F.normalize(matrix.mean(dim=0), p=2, dim=0)
    first = int((matrix @ centroid).argmax())
    selected = [first]
    minimum_distance = 1.0 - matrix @ matrix[first]
    while len(selected) < min(count, len(matrix)):
        minimum_distance[selected] = -1.0
        index = int(minimum_distance.argmax())
        selected.append(index)
        distance = 1.0 - matrix @ matrix[index]
        minimum_distance = torch.minimum(minimum_distance, distance)
    if len(selected) < count:
        selected.extend([selected[-1]] * (count - len(selected)))
    return matrix[selected]


def build_pairwise_prototypes(
    embeddings: Tensor,
    labels: Sequence[str],
    intents: Sequence[str],
    label_embeddings: Tensor,
    *,
    prototype_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    matrix = F.normalize(
        _finite_matrix(embeddings, "R2 train embeddings"), p=2, dim=1
    )
    if len(matrix) != len(labels):
        raise ValueError("R2 train embeddings/labels 不对齐")
    names = tuple(intents)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("R2 intents 非法")
    label_matrix = F.normalize(
        _finite_matrix(label_embeddings, "R2 label embeddings"), p=2, dim=1
    )
    if label_matrix.shape != (len(names), matrix.shape[1]):
        raise ValueError("R2 label embeddings shape 非法")
    centroids, prototypes = [], []
    for intent in names:
        indices = [index for index, label in enumerate(labels) if label == intent]
        if not indices:
            raise ValueError(f"R2 intent 缺少 train rows：{intent}")
        rows = matrix[indices]
        centroids.append(F.normalize(rows.mean(dim=0), p=2, dim=0))
        prototypes.append(_farthest_prototypes(rows, prototype_count))
    return torch.stack(centroids), torch.stack(prototypes), label_matrix


def pairwise_features(
    query_embeddings: Tensor,
    centroids: Tensor,
    prototypes: Tensor,
    label_embeddings: Tensor,
    *,
    top_k: int,
) -> Tensor:
    queries = F.normalize(
        _finite_matrix(query_embeddings, "R2 query embeddings"), p=2, dim=1
    )
    centers = F.normalize(_finite_matrix(centroids, "R2 centroids"), p=2, dim=1)
    labels = F.normalize(
        _finite_matrix(label_embeddings, "R2 label embeddings"), p=2, dim=1
    )
    if prototypes.ndim != 3 or prototypes.shape[:2] == (0, 0):
        raise ValueError("R2 prototypes shape 非法")
    samples = F.normalize(prototypes.detach().cpu().float(), p=2, dim=2)
    if (
        centers.shape != labels.shape
        or samples.shape[0] != len(centers)
        or samples.shape[2] != centers.shape[1]
        or queries.shape[1] != centers.shape[1]
    ):
        raise ValueError("R2 pairwise tensors shape 不兼容")
    k = min(int(top_k), samples.shape[1])
    if k <= 0:
        raise ValueError("R2 top_k 必须为正")
    centroid_cosine = queries @ centers.T
    label_cosine = queries @ labels.T
    sample_cosine = torch.einsum("nd,ckd->nck", queries, samples)
    nearest = sample_cosine.max(dim=2).values
    top_mean = sample_cosine.topk(k, dim=2).values.mean(dim=2)
    return torch.stack(
        (
            centroid_cosine,
            label_cosine,
            nearest,
            top_mean,
            torch.abs(centroid_cosine - label_cosine),
            nearest - top_mean,
        ),
        dim=2,
    )


def fit_pairwise_evidence(
    embeddings: Tensor,
    labels: Sequence[str],
    intents: Sequence[str],
    label_embeddings: Tensor,
    *,
    prototype_count: int,
    top_k: int,
    negative_pairs_per_positive: int,
    ridge_strength: float,
) -> PairwiseEvidenceModel:
    names = tuple(intents)
    centroids, prototypes, label_matrix = build_pairwise_prototypes(
        embeddings,
        labels,
        names,
        label_embeddings,
        prototype_count=prototype_count,
    )
    all_features = pairwise_features(
        embeddings, centroids, prototypes, label_matrix, top_k=top_k
    )
    lookup = {label: index for index, label in enumerate(names)}
    selected_features, binary_targets = [], []
    negative_count = int(negative_pairs_per_positive)
    if negative_count <= 0 or negative_count >= len(names):
        raise ValueError("R2 negative pair count 非法")
    for row_index, label in enumerate(labels):
        positive = lookup[label]
        selected_features.append(all_features[row_index, positive])
        binary_targets.append(1.0)
        hard_order = torch.argsort(
            all_features[row_index, :, 0], descending=True
        ).tolist()
        negatives = [index for index in hard_order if index != positive][
            :negative_count
        ]
        selected_features.extend(
            all_features[row_index, index] for index in negatives
        )
        binary_targets.extend([0.0] * len(negatives))
    matrix = torch.stack(selected_features).double()
    targets = torch.tensor(binary_targets, dtype=torch.float64)
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False).clamp_min(1e-5)
    standardized = (matrix - mean) / scale
    augmented = torch.cat(
        (standardized, torch.ones((len(matrix), 1), dtype=torch.float64)),
        dim=1,
    )
    strength = float(ridge_strength)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("R2 ridge strength 非法")
    gram = augmented.T @ augmented
    regularizer = max(float(gram.diagonal().mean()) * strength, 1e-8)
    gram.diagonal().add_(regularizer)
    weights = torch.linalg.solve(gram, augmented.T @ targets)
    return PairwiseEvidenceModel(
        names,
        centroids,
        prototypes,
        label_matrix,
        mean,
        scale,
        weights,
        regularizer,
        int(top_k),
    )


def forced_argmax(scores: Tensor, intents: Sequence[str]) -> list[IntentDecision]:
    matrix = scores.detach().cpu().float()
    names = tuple(intents)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("forced argmax score shape 非法")
    output = []
    for row in matrix:
        if not bool(torch.isfinite(row).all()):
            output.append(IntentDecision("reject", None, "invalid", ()))
            continue
        intent = names[int(row.argmax())]
        output.append(IntentDecision("accept", intent, None, (intent,)))
    return output


def max_threshold_route(
    scores: Tensor, intents: Sequence[str], *, threshold: float
) -> list[IntentDecision]:
    matrix = scores.detach().cpu().float()
    names = tuple(intents)
    output = []
    for row in matrix:
        if (
            row.ndim != 1
            or len(row) != len(names)
            or not bool(torch.isfinite(row).all())
        ):
            output.append(IntentDecision("reject", None, "invalid", ()))
            continue
        index = int(row.argmax())
        if float(row[index]) >= float(threshold):
            intent = names[index]
            output.append(IntentDecision("accept", intent, None, (intent,)))
        else:
            output.append(IntentDecision("reject", None, "unknown", ()))
    return output


def set_valued_route(
    scores: Tensor,
    intents: Sequence[str],
    *,
    threshold: float,
    minimum_margin: float,
) -> list[IntentDecision]:
    matrix = scores.detach().cpu().float()
    names = tuple(intents)
    output = []
    for row in matrix:
        if (
            row.ndim != 1
            or len(row) != len(names)
            or not bool(torch.isfinite(row).all())
        ):
            output.append(IntentDecision("reject", None, "invalid", ()))
            continue
        candidates = tuple(
            names[index]
            for index in range(len(names))
            if float(row[index]) >= float(threshold)
        )
        if not candidates:
            output.append(IntentDecision("reject", None, "unknown", ()))
        elif len(candidates) > 1:
            output.append(
                IntentDecision("reject", None, "ambiguous", candidates)
            )
        else:
            ordered = row.topk(2).values
            margin = float(ordered[0] - ordered[1])
            if margin < float(minimum_margin):
                output.append(
                    IntentDecision("reject", None, "ambiguous", candidates)
                )
            else:
                output.append(
                    IntentDecision("accept", candidates[0], None, candidates)
                )
    return output


def assert_external_memory_boundary() -> None:
    """C2.5 只模拟 bucket access 计数，不导入或扩展 task-specific memory API。"""

    if "private" in IntentDecision.__annotations__:
        raise C25ProtocolError("C2.5 decision 意外包含 private 字段")

