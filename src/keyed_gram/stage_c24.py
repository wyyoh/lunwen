"""Stage C2.4 discrete relation-contract audit.

This module freezes the selected C2.3 S5 encoder/view/ridge head and compares
four query contracts.  It never trains a private value memory and never reads
or serializes private answers.  D0 is the historical continuous reference, D1
uses a hard one-hot in the historical global index, D2 uses an exact relation
bucket, and D3 places a calibration-frozen reject guard before D2.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from .canonicalizer import load_canonicalizer_checkpoint
from .phase_a import _sha256_file
from .stage_c1 import cosine_silhouette, representation_geometry, ridge_probe_accuracy
from .stage_c21 import _load_feature_cache
from .stage_c23 import (
    _extract_frozen_entity_embeddings,
    answer_free_query_metrics,
    load_c23_protocol_status,
    validate_answer_free_private_feature_cache,
    validate_oracle_sources,
)
from .stage_c23_audit import (
    _EmbeddingStore,
    fuse_predicted_relation_queries,
    sanitize_private_metadata,
)
from .stage_c23_semantic import (
    SEMANTIC_ENCODER_SPECS,
    LoadedSemanticEncoder,
    RidgeLinearHead,
    build_model_file_manifest,
    build_semantic_view_texts,
    encode_semantic_texts,
    load_semantic_encoder,
    semantic_text_sha256,
)
from .stage_c24_contract import (
    AcceptedRoute,
    RelationId,
    RejectedRoute,
    hard_one_hot_from_logits,
    memory_contract_exposure_flags,
    retrieve_for_route,
    route_with_reject_guard,
)
from .stage_c24_benchmark import (
    PublicSplit,
    REVIEW_FIELDS,
    build_locked_audit_review_rows,
    build_public_benchmark,
    audit_phrase_collections,
    strict_json_dumps,
)
from .stage_c23_benchmark import normalize_text
from .stage_c24_metrics import (
    closed_route_metrics,
    evaluate_reject_guard,
    extract_reject_score_candidates,
    select_calibrated_reject_guard,
)
from .train import collect_environment, resolve_device


STAGE_C24_SCHEMA_VERSION = 1
STAGE_NAME = "C2.4-discrete-relation-contract-audit"
_SENSITIVE_INPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_index",
        "answer_text",
        "answer_value",
        "candidate_answer",
        "candidate_answers",
        "completion",
        "completions",
        "ground_truth_value",
        "candidates",
        "private_answer",
        "private_answers",
        "private_value",
        "private_values",
        "response_value",
        "target_value",
    }
)
_FORBIDDEN_INPUT_TERMS = ("confirmation", "seal")
_SENSITIVE_OUTPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_index",
        "answer_text",
        "answer_value",
        "candidate_answer",
        "candidate_answers",
        "completion",
        "completions",
        "ground_truth_value",
        "private_answer",
        "private_answers",
        "private_value",
        "private_values",
        "response_value",
        "target_value",
    }
)


class ProtocolViolation(RuntimeError):
    """A violation that invalidates the current audit before evaluation."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if (
            key in output
            and isinstance(output[key], Mapping)
            and isinstance(value, Mapping)
        ):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = value
    return output


def load_stage_c24_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one optional ``extends`` layer without accepting unsafe inputs."""

    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ProtocolViolation("Stage C2.4 config must be a mapping")
    raw = dict(raw)
    parent_name = raw.pop("extends", None)
    source_files = [source]
    if parent_name is not None:
        parent_path = Path(str(parent_name))
        if not parent_path.is_absolute():
            candidate = source.parent / parent_path
            parent_path = candidate if candidate.exists() else parent_path
        parent_raw = yaml.safe_load(parent_path.read_text(encoding="utf-8")) or {}
        if not isinstance(parent_raw, Mapping) or "extends" in parent_raw:
            raise ProtocolViolation("Stage C2.4 supports exactly one config parent")
        values = _deep_merge(parent_raw, raw)
        source_files.insert(0, parent_path)
    else:
        values = raw
    _reject_unsafe_input(values)
    expected_top_level = {
        "schema_version",
        "stage",
        "fixed_sources",
        "fixed_s5",
        "fixed_g3",
        "run",
        "gates",
        "benchmark",
    }
    if set(values) != expected_top_level:
        raise ProtocolViolation(
            "Stage C2.4 config top-level fields differ from the strict schema"
        )
    if int(values.get("schema_version", -1)) != STAGE_C24_SCHEMA_VERSION:
        raise ProtocolViolation("unsupported Stage C2.4 config schema")
    if not str(values.get("stage", "")).startswith(STAGE_NAME):
        raise ProtocolViolation("config is not a Stage C2.4 audit")
    expected_scores = (
        "G1_max_softmax",
        "G2_top_logit_margin",
        "G3_definition_margin",
    )
    if tuple(values.get("run", {}).get("reject_scores", ())) != expected_scores:
        raise ProtocolViolation("reject score candidates differ from preregistration")
    metadata = {
        "source_files": [str(item) for item in source_files],
        "source_sha256": {
            str(item): _sha256_file(item) for item in source_files
        },
        "resolved_config_sha256": canonical_sha256(values),
    }
    return values, metadata


def _reject_unsafe_input(value: Any, *, location: str = "config") -> None:
    """Reject answer-bearing or forbidden evaluation-pool inputs recursively."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _SENSITIVE_INPUT_KEYS:
                raise ProtocolViolation(
                    f"answer-bearing input field is forbidden at {location}.{key}"
                )
            if any(term in lowered for term in _FORBIDDEN_INPUT_TERMS):
                raise ProtocolViolation(
                    f"forbidden evaluation-pool input at {location}.{key}"
                )
            if lowered in {"contains_private_answer", "contains_private_answers"}:
                if item is not False:
                    raise ProtocolViolation(
                        f"answer-free invariant must be false at {location}.{key}"
                    )
            _reject_unsafe_input(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_unsafe_input(item, location=f"{location}[{index}]")
    elif isinstance(value, (str, Path)):
        lowered = str(value).lower()
        if any(term in lowered for term in _FORBIDDEN_INPUT_TERMS):
            raise ProtocolViolation(f"forbidden input path/value at {location}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _assert_finite_json(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _SENSITIVE_OUTPUT_KEYS:
                raise ValueError(f"answer-bearing JSON field at {location}.{key}")
            _assert_finite_json(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {location}")
    elif isinstance(value, Tensor):
        raise TypeError(f"tensor cannot be serialized at {location}")


def _write_json(path: str | Path, value: Any) -> Path:
    _assert_finite_json(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return target


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("CSV output cannot be empty")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows have inconsistent columns")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _git_state() -> dict[str, Any]:
    executable = "git"
    try:
        commit = subprocess.check_output(
            [executable, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            [executable, "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_path = Path(r"C:\Program Files\Git\cmd\git.exe")
        if not git_path.exists():
            return {"commit": "unavailable", "dirty": True}
        commit = subprocess.check_output(
            [str(git_path), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            [str(git_path), "status", "--porcelain"], text=True
        )
    return {"commit": commit, "dirty": bool(status.strip())}


def _tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_file_hash(path: str | Path, expected: str, *, label: str) -> Path:
    source = Path(path)
    observed = _sha256_file(source)
    if observed != str(expected):
        raise ProtocolViolation(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return source


def load_fixed_s5_head(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_variant: str,
    expected_view: str,
    expected_classes: Sequence[str],
) -> RidgeLinearHead:
    """Load, never refit, the selected C2.3 S5 ridge relation head."""

    source = _validate_file_hash(path, expected_sha256, label="fixed S5 head")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected_metadata = {
        "format_version": 1,
        "stage": "C2.3-S5-public-ridge-relation-head",
        "base_encoder_variant": expected_variant,
        "view": expected_view,
        "classes": list(expected_classes),
    }
    if any(payload.get(key) != value for key, value in expected_metadata.items()):
        raise ProtocolViolation("fixed S5 ridge-head metadata mismatch")
    mean = payload.get("feature_mean")
    scale = payload.get("feature_scale")
    weights = payload.get("weights")
    if not all(isinstance(value, Tensor) for value in (mean, scale, weights)):
        raise ProtocolViolation("fixed S5 ridge-head tensors are missing")
    if mean.ndim != 1 or scale.shape != mean.shape:
        raise ProtocolViolation("fixed S5 ridge-head normalization is malformed")
    if weights.shape != (mean.numel() + 1, len(expected_classes)):
        raise ProtocolViolation("fixed S5 ridge-head weights are malformed")
    if not all(bool(torch.isfinite(value).all()) for value in (mean, scale, weights)):
        raise ProtocolViolation("fixed S5 ridge-head contains non-finite values")
    regularizer = float(payload.get("regularizer", float("nan")))
    if not math.isfinite(regularizer) or regularizer <= 0:
        raise ProtocolViolation("fixed S5 ridge-head regularizer is invalid")
    return RidgeLinearHead(
        classes=tuple(str(value) for value in payload["classes"]),
        feature_mean=mean.detach().cpu().double(),
        feature_scale=scale.detach().cpu().double(),
        weights=weights.detach().cpu().double(),
        regularizer=regularizer,
    )


def _row_ids(rows: Sequence[Mapping[str, Any]], *, prefix: str) -> list[str]:
    output = []
    for index, row in enumerate(rows):
        value = row.get("example_id")
        if value is None:
            value = f"{prefix}:{index}:{row['fact_id']}:{row['template_id']}"
        output.append(str(value))
    return output


def _definition_embeddings(
    variant: str,
    encoder: LoadedSemanticEncoder,
    model_manifest: Mapping[str, Any],
    definitions: Mapping[str, Sequence[str]],
    store: _EmbeddingStore,
) -> dict[str, Tensor]:
    relations = tuple(relation.value for relation in RelationId)
    if set(definitions) != set(relations):
        raise ProtocolViolation("public definitions must cover the relation enum")
    texts: list[str] = []
    row_ids: list[str] = []
    counts: dict[str, int] = {}
    for relation in relations:
        values = [str(value).strip() for value in definitions[relation]]
        if not values or any(not value for value in values):
            raise ProtocolViolation("public relation definitions cannot be empty")
        counts[relation] = len(values)
        texts.extend(values)
        row_ids.extend(
            f"definition:{relation}:{index}" for index in range(len(values))
        )
    encoded = store.get(
        variant,
        encoder,
        model_manifest,
        dataset="public_definitions",
        view="definition",
        texts=texts,
        row_ids=row_ids,
        e5_input_type="query",
    )
    output: dict[str, Tensor] = {}
    offset = 0
    for relation in relations:
        output[relation] = encoded[offset : offset + counts[relation]]
        offset += counts[relation]
    return output


def _definition_logits(
    embeddings: Tensor,
    definitions: Mapping[str, Tensor],
    classes: Sequence[str],
) -> Tensor:
    queries = F.normalize(embeddings.float(), p=2, dim=-1)
    scores = []
    for relation in classes:
        values = F.normalize(definitions[str(relation)].float(), p=2, dim=-1)
        prototype = F.normalize(values.mean(dim=0), p=2, dim=0)
        scores.append(queries @ prototype)
    return torch.stack(scores, dim=1)


def _prediction_strings(logits: Tensor, classes: Sequence[str]) -> list[str]:
    labels = tuple(str(value) for value in classes)
    return [labels[int(index)] for index in logits.argmax(dim=1)]


def _hard_fused_queries(
    entity_embeddings: Tensor,
    logits: Tensor,
    *,
    alpha: float,
    class_order: Sequence[str | RelationId] = tuple(RelationId),
) -> Tensor:
    if float(alpha) != 0.5:
        raise ProtocolViolation("D1 is frozen at alpha=0.5")
    if len(entity_embeddings) != len(logits):
        raise ValueError("D1 entity and relation inputs are incompatible")
    try:
        source_order = tuple(
            value if isinstance(value, RelationId) else RelationId(str(value))
            for value in class_order
        )
    except ValueError as exc:
        raise ProtocolViolation("D1 class order contains an unknown relation") from exc
    if len(source_order) != len(RelationId) or set(source_order) != set(RelationId):
        raise ProtocolViolation("D1 class order differs from the relation enum")
    canonical_indices = [source_order.index(relation) for relation in RelationId]
    canonical_logits = logits[:, canonical_indices]
    one_hot = hard_one_hot_from_logits(canonical_logits)
    if not torch.equal(one_hot.sum(dim=1), torch.ones(len(one_hot))):
        raise RuntimeError("D1 hard one-hot invariant failed")
    entity = F.normalize(entity_embeddings.float(), p=2, dim=-1)
    return F.normalize(torch.cat([entity, float(alpha) * one_hot], dim=1), p=2, dim=-1)


def d0_continuous_queries(
    entity_embeddings: Tensor, relation_logits: Tensor, *, alpha: float = 0.5
) -> Tensor:
    """Expose the frozen C2.3 continuous baseline as an auditable D0 helper."""

    return fuse_predicted_relation_queries(
        entity_embeddings, relation_logits, alpha=alpha
    )


@dataclass(frozen=True)
class FactBuckets:
    prototypes: dict[RelationId, Tensor]
    fact_ids: dict[RelationId, tuple[str, ...]]
    entity_ids: dict[RelationId, tuple[str, ...]]
    row_embeddings: dict[RelationId, Tensor]
    row_fact_ids: dict[RelationId, tuple[str, ...]]


def build_fact_buckets(
    train_entity_embeddings: Tensor,
    train_rows: Sequence[Mapping[str, Any]],
) -> FactBuckets:
    if train_entity_embeddings.ndim != 2 or len(train_entity_embeddings) != len(
        train_rows
    ):
        raise ValueError("fact bucket inputs are incompatible")
    prototypes: dict[RelationId, Tensor] = {}
    fact_ids: dict[RelationId, tuple[str, ...]] = {}
    entity_ids: dict[RelationId, tuple[str, ...]] = {}
    row_embeddings: dict[RelationId, Tensor] = {}
    row_fact_ids: dict[RelationId, tuple[str, ...]] = {}
    for relation in RelationId:
        indices = [
            index
            for index, row in enumerate(train_rows)
            if str(row["attribute"]) == relation.value
        ]
        if not indices:
            raise ValueError(f"training facts are missing {relation.value}")
        grouped: dict[str, list[int]] = defaultdict(list)
        entities: dict[str, str] = {}
        for index in indices:
            fact = str(train_rows[index]["fact_id"])
            grouped[fact].append(index)
            entity = str(train_rows[index]["entity"])
            previous = entities.setdefault(fact, entity)
            if previous != entity:
                raise ValueError("one fact maps to multiple entities")
        ordered = tuple(sorted(grouped))
        fact_ids[relation] = ordered
        entity_ids[relation] = tuple(entities[fact] for fact in ordered)
        prototypes[relation] = torch.stack(
            [
                F.normalize(
                    train_entity_embeddings[grouped[fact]].float().mean(dim=0),
                    p=2,
                    dim=0,
                )
                for fact in ordered
            ]
        )
        row_embeddings[relation] = train_entity_embeddings[indices].float()
        row_fact_ids[relation] = tuple(
            str(train_rows[index]["fact_id"]) for index in indices
        )
    return FactBuckets(
        prototypes=prototypes,
        fact_ids=fact_ids,
        entity_ids=entity_ids,
        row_embeddings=row_embeddings,
        row_fact_ids=row_fact_ids,
    )


def evaluate_bucket_retrieval(
    buckets: FactBuckets,
    evaluation_entity_embeddings: Tensor,
    evaluation_rows: Sequence[Mapping[str, Any]],
    relation_logits: Tensor,
    *,
    relation_order: Sequence[RelationId],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate actual D2 routing; a misrouted true fact receives rank zero."""

    if not (
        evaluation_entity_embeddings.ndim == 2
        and relation_logits.ndim == 2
        and len(evaluation_rows)
        == len(evaluation_entity_embeddings)
        == len(relation_logits)
    ):
        raise ValueError("D2 evaluation inputs are incompatible")
    order = tuple(relation_order)
    if len(order) != relation_logits.size(1) or set(order) != set(RelationId):
        raise ValueError("D2 relation order differs from the fixed head")
    predictions: list[dict[str, Any]] = []
    centroid_hits: list[bool] = []
    row_hits: list[bool] = []
    reciprocal_ranks: list[float] = []
    margins: list[float] = []
    true_similarities: list[float] = []
    wrong_similarities: list[float] = []
    misrouted = 0
    total_accesses = 0
    candidate_counts = []
    for index, row in enumerate(evaluation_rows):
        route_index = int(relation_logits[index].argmax().item())
        relation = order[route_index]
        route = AcceptedRoute(relation)
        # Passing a single-entry mapping makes the corresponding relation bucket
        # the only prototype collection visible at the memory boundary.
        execution = retrieve_for_route(
            evaluation_entity_embeddings[index],
            route,
            {relation: buckets.prototypes[relation]},
        )
        if execution.retrieval is None:
            raise RuntimeError("an accepted D2 route did not retrieve")
        result = execution.retrieval
        total_accesses += execution.memory_access_count
        candidate_counts.append(result.candidate_count)
        predicted_fact = buckets.fact_ids[relation][result.candidate_index]
        true_fact = str(row["fact_id"])
        routed_correctly = str(row["attribute"]) == relation.value
        if not routed_correctly:
            misrouted += 1

        centroid_matrix = F.normalize(buckets.prototypes[relation].float(), p=2, dim=1)
        query = F.normalize(evaluation_entity_embeddings[index].float(), p=2, dim=0)
        centroid_similarity = centroid_matrix @ query
        ranking = centroid_similarity.argsort(descending=True)
        if true_fact in buckets.fact_ids[relation]:
            true_index = buckets.fact_ids[relation].index(true_fact)
            rank = int((ranking == true_index).nonzero(as_tuple=False)[0].item()) + 1
            true_similarity = float(centroid_similarity[true_index])
            wrong_mask = torch.ones(len(centroid_similarity), dtype=torch.bool)
            wrong_mask[true_index] = False
            best_wrong = float(centroid_similarity[wrong_mask].max())
            margin = true_similarity - best_wrong
        else:
            rank = 0
            true_similarity = -1.0
            best_wrong = float(centroid_similarity.max())
            margin = -1.0 - best_wrong

        row_matrix = F.normalize(buckets.row_embeddings[relation].float(), p=2, dim=1)
        row_index = int((row_matrix @ query).argmax().item())
        row_prediction = buckets.row_fact_ids[relation][row_index]
        centroid_hit = predicted_fact == true_fact
        row_hit = row_prediction == true_fact
        centroid_hits.append(centroid_hit)
        row_hits.append(row_hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        margins.append(margin)
        true_similarities.append(true_similarity)
        wrong_similarities.append(best_wrong)
        predictions.append(
            {
                "row_id": str(row.get("example_id", f"private:{index}")),
                "fact_id": true_fact,
                "entity": str(row["entity"]),
                "true_relation_id": str(row["attribute"]),
                "predicted_relation_id": relation.value,
                "predicted_fact_id": predicted_fact,
                "row_1nn_prediction": row_prediction,
                "centroid_correct": centroid_hit,
                "row_1nn_correct": row_hit,
                "centroid_rank": rank,
                "centroid_margin": margin,
                "candidate_count": result.candidate_count,
                "cross_relation_candidate_count": result.cross_relation_candidate_count,
                "memory_access_count": execution.memory_access_count,
            }
        )
    count = len(evaluation_rows)
    return {
        "num_database_facts": sum(len(values) for values in buckets.fact_ids.values()),
        "num_queries": count,
        "centroid_top1_accuracy": sum(centroid_hits) / count,
        "row_1nn_accuracy": sum(row_hits) / count,
        "centroid_mrr": sum(reciprocal_ranks) / count,
        "mean_centroid_margin": sum(margins) / count,
        "mean_true_centroid_similarity": sum(true_similarities) / count,
        "mean_best_wrong_centroid_similarity": sum(wrong_similarities) / count,
        "misrouted_query_count": misrouted,
        "misrouted_query_rate": misrouted / count,
        "memory_access_count": total_accesses,
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "cross_relation_candidate_count": 0,
    }, predictions


def bucket_geometry_diagnostic(
    train_entity_embeddings: Tensor,
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_entity_embeddings: Tensor,
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Geometry inside oracle relation partitions, outside the memory contract."""

    silhouettes = []
    margins = []
    for relation in RelationId:
        train_indices = [
            index
            for index, row in enumerate(train_rows)
            if str(row["attribute"]) == relation.value
        ]
        evaluation_indices = [
            index
            for index, row in enumerate(evaluation_rows)
            if str(row["attribute"]) == relation.value
        ]
        train_features = train_entity_embeddings[train_indices]
        eval_features = evaluation_entity_embeddings[evaluation_indices]
        train_clean = [
            {
                "fact_id": str(train_rows[index]["fact_id"]),
                "entity": str(train_rows[index]["entity"]),
                "attribute": str(train_rows[index]["attribute"]),
                "template_id": str(train_rows[index]["template_id"]),
            }
            for index in train_indices
        ]
        eval_clean = [
            {
                "fact_id": str(evaluation_rows[index]["fact_id"]),
                "entity": str(evaluation_rows[index]["entity"]),
                "attribute": str(evaluation_rows[index]["attribute"]),
                "template_id": str(evaluation_rows[index]["template_id"]),
            }
            for index in evaluation_indices
        ]
        combined = torch.cat([train_features, eval_features], dim=0)
        fact_labels = [
            str(row["fact_id"]) for row in [*train_clean, *eval_clean]
        ]
        silhouettes.append(cosine_silhouette(combined, fact_labels))
        geometry = representation_geometry(
            train_features, train_clean, eval_features, eval_clean
        )
        margins.append(float(geometry["fact_over_template_margin"]))
    return {
        "scope": "evaluation_only_true_relation_partitions_outside_memory_contract",
        "fact_silhouette": sum(silhouettes) / len(silhouettes),
        "fact_over_template_margin": sum(margins) / len(margins),
        "per_relation_fact_silhouette": {
            relation.value: value for relation, value in zip(RelationId, silhouettes)
        },
        "per_relation_fact_over_template_margin": {
            relation.value: value for relation, value in zip(RelationId, margins)
        },
    }


def compute_eligibility(
    *,
    closed_set_discrete_contract_ready: bool,
    open_set_reject_guard_ready: bool,
    public_benchmark_human_reviewed: bool,
    new_pool_created_after_freeze: bool,
) -> dict[str, bool]:
    c3 = bool(
        closed_set_discrete_contract_ready
        and open_set_reject_guard_ready
        and public_benchmark_human_reviewed
        and new_pool_created_after_freeze
    )
    return {
        "closed_set_discrete_contract_ready": bool(
            closed_set_discrete_contract_ready
        ),
        "open_set_reject_guard_ready": bool(open_set_reject_guard_ready),
        "public_benchmark_human_reviewed": bool(public_benchmark_human_reviewed),
        "new_confirmation_pool_created_after_freeze": bool(
            new_pool_created_after_freeze
        ),
        "c3_eligible": c3,
    }


def _gate(value: float | int, threshold: float | int, *, lower: bool = False) -> dict[str, Any]:
    observed = float(value)
    target = float(threshold)
    return {
        "value": observed,
        "threshold": target,
        "direction": "<=" if lower else ">=",
        "passed": observed <= target if lower else observed >= target,
    }


def _artifact_manifest(output_dir: Path, provenance: Mapping[str, Any]) -> Path:
    target = output_dir / "artifact_sha256_manifest.json"
    files = []
    for path in sorted(
        (item for item in output_dir.rglob("*") if item.is_file() and item != target),
        key=lambda item: item.relative_to(output_dir).as_posix(),
    ):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return _write_json(
        target,
        {
            "schema_version": STAGE_C24_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "provenance": dict(provenance),
            "self_excluded": True,
            "files": files,
        },
    )


def _base_provenance(
    *,
    config_metadata: Mapping[str, Any],
    values: Mapping[str, Any],
    device: torch.device,
    model_manifest: Mapping[str, Any] | None = None,
    g3_model_manifest: Mapping[str, Any] | None = None,
    data_sha256: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fixed = values["fixed_s5"]
    fixed_g3 = values["fixed_g3"]
    return {
        "schema_version": STAGE_C24_SCHEMA_VERSION,
        "git": _git_state(),
        "environment": collect_environment(device, torch.float32),
        "model": {
            "id": str(fixed["model_id"]),
            "revision": str(fixed["revision"]),
            "manifest_sha256": str(fixed["model_manifest_sha256"]),
            "observed_manifest_sha256": (
                str(model_manifest["manifest_sha256"])
                if model_manifest is not None
                else None
            ),
        },
        "models": {
            "S5_relation_head_encoder": {
                "id": str(fixed["model_id"]),
                "revision": str(fixed["revision"]),
                "manifest_sha256": str(fixed["model_manifest_sha256"]),
                "observed_manifest_sha256": (
                    str(model_manifest["manifest_sha256"])
                    if model_manifest is not None
                    else None
                ),
            },
            "G3_definition_encoder": {
                "id": str(fixed_g3["model_id"]),
                "revision": str(fixed_g3["revision"]),
                "manifest_sha256": str(fixed_g3["model_manifest_sha256"]),
                "observed_manifest_sha256": (
                    str(g3_model_manifest["manifest_sha256"])
                    if g3_model_manifest is not None
                    else None
                ),
                "view": str(fixed_g3["view"]),
            },
        },
        "config": dict(config_metadata),
        "data_sha256": dict(data_sha256 or {}),
        "seed": int(values["run"]["seed"]),
    }


def _write_review_csv_strict(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    if not rows or any(tuple(row) != REVIEW_FIELDS for row in rows):
        raise ProtocolViolation("locked-audit review rows have an invalid schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _historical_c23_locked_novelty_audit(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove locked wording/IDs/entities/frames are new versus all C2.3 public data."""

    fixed_sources = values["fixed_sources"]
    source = _validate_file_hash(
        fixed_sources["stage_c23_config"],
        fixed_sources["stage_c23_config_sha256"],
        label="C2.3 config",
    )
    historical = yaml.safe_load(source.read_text(encoding="utf-8"))["benchmark"]
    locked = values["benchmark"]["splits"][PublicSplit.LOCKED_AUDIT.value]

    historical_phrases: dict[str, str] = {}
    historical_family_ids: set[str] = set()
    for group_name in ("train_families", "validation_families"):
        for relation, families in historical[group_name].items():
            for family_id, phrase in families.items():
                historical_family_ids.add(str(family_id))
                historical_phrases[f"{group_name}:{relation}:{family_id}"] = str(phrase)
    for kind, families in historical["rejection_families"].items():
        for family_id, phrase in families.items():
            historical_family_ids.add(str(family_id))
            historical_phrases[f"rejection:{kind}:{family_id}"] = str(phrase)
    for split, relations in historical["private_phrase_views"].items():
        for relation, phrases in relations.items():
            for index, phrase in enumerate(phrases):
                historical_phrases[
                    f"private_phrase_view:{split}:{relation}:{index}"
                ] = str(phrase)

    locked_phrases: dict[str, str] = {}
    locked_family_ids: set[str] = set()
    for group_name in ("known_families", "rejection_families"):
        for relation, families in locked[group_name].items():
            for family_id, phrase in families.items():
                locked_family_ids.add(str(family_id))
                locked_phrases[f"{group_name}:{relation}:{family_id}"] = str(phrase)
    phrase_audit = audit_phrase_collections(
        historical_phrases,
        locked_phrases,
        left_name="all_C2.3_public_and_private_phrase_views",
        right_name="C2.4_public_locked_audit",
        lemma_bigram_threshold=float(values["benchmark"].get("lemma_bigram_threshold", 0.8)),
        minimum_containment_tokens=1,
    )
    historical_entities = {
        normalize_text(value)
        for field in ("train_entities", "validation_entities")
        for value in historical[field]
    }
    locked_entities = {normalize_text(value) for value in locked["entities"]}
    historical_frames = {
        normalize_text(value)
        for field in ("train_frames", "validation_frames")
        for value in historical[field]
    }
    locked_frames = {normalize_text(value) for value in locked["frames"]}
    family_overlap = sorted(historical_family_ids & locked_family_ids)
    entity_overlap = sorted(historical_entities & locked_entities)
    frame_overlap = sorted(historical_frames & locked_frames)
    result = {
        "schema_version": STAGE_C24_SCHEMA_VERSION,
        "historical_source": source.as_posix(),
        "historical_source_sha256": _sha256_file(source),
        "phrase_audit": phrase_audit,
        "phrase_family_id_collisions": family_overlap,
        "entity_collisions": entity_overlap,
        "frame_collisions": frame_overlap,
        "passed": phrase_audit["passed"]
        and not family_overlap
        and not entity_overlap
        and not frame_overlap,
    }
    if not result["passed"]:
        raise ProtocolViolation("locked audit is not novel versus all C2.3 public data")
    return result


def prepare_stage_c24(config_path: str | Path) -> dict[str, Any]:
    """Materialize only the answer-free public protocol and review template."""

    values, config_metadata = load_stage_c24_config(config_path)
    benchmark_config = values.get("benchmark")
    if not isinstance(benchmark_config, Mapping):
        raise ProtocolViolation("Stage C2.4 config is missing benchmark settings")
    data_dir = Path(str(benchmark_config["public_data_dir"]))
    manifest_path = Path(str(benchmark_config["manifest"]))
    review_path = Path(str(benchmark_config["review_csv"]))
    planned_data_paths = [
        data_dir / f"{split.value}.jsonl" for split in PublicSplit
    ]
    existing = [
        path
        for path in (
            manifest_path,
            review_path,
            manifest_path.parent / "protocol_incident.json",
            *planned_data_paths,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Stage C2.4 preparation outputs already exist and cannot be overwritten"
        )
    try:
        rows, audit = build_public_benchmark(benchmark_config)
        historical_novelty = _historical_c23_locked_novelty_audit(values)
    except (ProtocolViolation, ValueError, KeyError, TypeError) as exc:
        _record_protocol_incident(manifest_path.parent, str(exc))
        raise
    paths = {
        split: data_dir / f"{split}.jsonl" for split in sorted(rows)
    }
    # Capture repository state before creating any tracked preparation output.
    provenance = _base_provenance(
        config_metadata=config_metadata,
        values=values,
        device=torch.device("cpu"),
        data_sha256={
            split: _jsonl_sha256(split_rows)
            for split, split_rows in rows.items()
        },
    )
    for split, split_rows in rows.items():
        target = paths[split]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in split_rows:
                handle.write(strict_json_dumps(row) + "\n")
    review_rows = build_locked_audit_review_rows(
        rows[PublicSplit.LOCKED_AUDIT.value]
    )
    _write_review_csv_strict(review_path, review_rows)
    manifest = {
        "schema_version": STAGE_C24_SCHEMA_VERSION,
        "stage": "C2.4-public-discrete-relation-contract-benchmark",
        "benchmark_version": str(benchmark_config["version"]),
        "preprocessing_version": str(benchmark_config["preprocessing_version"]),
        "review_status": str(benchmark_config["review_status"]),
        "answer_free": True,
        "contains_private_answers": False,
        "public_benchmark_human_reviewed": False,
        "provisional_locked_audit": True,
        "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "row_counts_by_kind": {
            split: dict(sorted(Counter(row["kind"] for row in split_rows).items()))
            for split, split_rows in rows.items()
        },
        "family_counts": {
            split: len({str(row["family_id"]) for row in split_rows})
            for split, split_rows in rows.items()
        },
        "audit": audit,
        "historical_c23_novelty_audit": historical_novelty,
        "files": {
            split: {
                "path": path.as_posix(),
                "sha256": _sha256_file(path),
                "row_sha256": _jsonl_sha256(rows[split]),
            }
            for split, path in paths.items()
        },
        "human_review_file": {
            "path": review_path.as_posix(),
            "sha256": _sha256_file(review_path),
            "row_count": len(review_rows),
            "all_rows_pending": all(
                row["review_status"] == "pending" for row in review_rows
            ),
        },
        "provenance": provenance,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolViolation(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def _load_prepared_benchmark(
    benchmark_config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load train/calibration only; locked rows remain unopened until freeze."""

    manifest_path = Path(str(benchmark_config["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("public_benchmark_human_reviewed") is not False:
        raise ProtocolViolation("locked audit review status was altered")
    if manifest.get("review_status") != (
        "curated_draft_pending_independent_human_review"
    ):
        raise ProtocolViolation("locked audit is not a pending curated draft")
    data_dir = Path(str(benchmark_config["public_data_dir"]))
    early_splits = (PublicSplit.TRAIN, PublicSplit.CALIBRATION)
    rows = {
        split.value: _read_jsonl(data_dir / f"{split.value}.jsonl")
        for split in early_splits
    }
    generated, audit = build_public_benchmark(benchmark_config)
    if any(rows[split.value] != generated[split.value] for split in early_splits):
        raise ProtocolViolation("materialized train/calibration differs from frozen config")
    if not audit["split_isolation"]["passed"]:
        raise ProtocolViolation("public split-isolation audit failed")
    for split, split_rows in rows.items():
        details = manifest.get("files", {}).get(split, {})
        path = data_dir / f"{split}.jsonl"
        if details.get("sha256") != _sha256_file(path):
            raise ProtocolViolation(f"prepared {split} file hash mismatch")
        if details.get("row_sha256") != _jsonl_sha256(split_rows):
            raise ProtocolViolation(f"prepared {split} row hash mismatch")
    return rows, manifest


def _load_locked_audit_after_freeze(
    benchmark_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Open and validate provisional locked rows only after development."""

    data_dir = Path(str(benchmark_config["public_data_dir"]))
    path = data_dir / f"{PublicSplit.LOCKED_AUDIT.value}.jsonl"
    rows = _read_jsonl(path)
    generated, audit = build_public_benchmark(benchmark_config)
    if not audit["split_isolation"]["passed"]:
        raise ProtocolViolation("public split-isolation audit failed")
    if rows != generated[PublicSplit.LOCKED_AUDIT.value]:
        raise ProtocolViolation("materialized locked audit differs from frozen config")
    details = manifest.get("files", {}).get(PublicSplit.LOCKED_AUDIT.value, {})
    if details.get("sha256") != _sha256_file(path):
        raise ProtocolViolation("prepared locked-audit file hash mismatch")
    if details.get("row_sha256") != _jsonl_sha256(rows):
        raise ProtocolViolation("prepared locked-audit row hash mismatch")
    review_path = Path(str(benchmark_config["review_csv"]))
    if manifest.get("human_review_file", {}).get("sha256") != _sha256_file(
        review_path
    ):
        raise ProtocolViolation("locked-audit review file hash mismatch")
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if not review_rows or any(row.get("review_status") != "pending" for row in review_rows):
        raise ProtocolViolation("locked-audit human review is not entirely pending")
    if any(
        row.get(field, "")
        for row in review_rows
        for field in ("reviewer_1_label", "reviewer_2_label", "adjudicated_label")
    ):
        raise ProtocolViolation("audit cannot consume completed human labels")
    return rows


def _public_parts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    known = [row for row in rows if row["kind"] == "known"]
    rejected = [row for row in rows if row["kind"] != "known"]
    if not known or not rejected:
        raise ProtocolViolation("open-set split requires known and rejection rows")
    return known, rejected


def _route_metrics_for_rows(
    logits: Tensor,
    rows: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    known_indices = [index for index, row in enumerate(rows) if row["kind"] == "known"]
    predictions = _prediction_strings(logits[known_indices], classes)
    known_rows = [rows[index] for index in known_indices]
    return closed_route_metrics(
        predictions,
        [str(row["attribute"]) for row in known_rows],
        [str(row["family_id"]) for row in known_rows],
        [str(row["frame_id"]) for row in known_rows],
        relation_order=[RelationId(value) for value in classes],
    )


def _private_route_metrics(
    logits: Tensor,
    rows: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    predictions = _prediction_strings(logits, classes)
    targets = [str(row["attribute"]) for row in rows]
    accuracy = sum(left == right for left, right in zip(predictions, targets)) / len(rows)
    return {
        "relation_accuracy": accuracy,
        "num_rows": len(rows),
        "class_order": list(classes),
    }


def _guard_inputs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[RelationId | None], list[str], list[str]]:
    targets = [
        RelationId(str(row["attribute"])) if row["kind"] == "known" else None
        for row in rows
    ]
    groups = ["known" if row["kind"] == "known" else str(row["kind"]) for row in rows]
    families = [str(row["family_id"]) for row in rows]
    return targets, groups, families


def _evaluate_guard_split(
    head_logits: Tensor,
    definition_logits: Tensor,
    rows: Sequence[Mapping[str, Any]],
    *,
    head_classes: Sequence[str],
    selected_score: str,
    temperature: float,
    threshold: float,
    known_coverage_target: float,
    ece_bins: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted = extract_reject_score_candidates(
        head_logits,
        definition_logits,
        head_class_order=head_classes,
        definition_class_order=head_classes,
        temperature=temperature,
    )
    targets, groups, families = _guard_inputs(rows)
    metrics = evaluate_reject_guard(
        extracted["scores"][selected_score],
        extracted["predicted_relations"],
        targets,
        groups,
        families,
        threshold=threshold,
        calibration_confidence=extracted["calibration_confidence"][selected_score],
        known_coverage_target=known_coverage_target,
        num_ece_bins=ece_bins,
    )
    route_rows = []
    scores = extracted["scores"][selected_score]
    relation_order = tuple(RelationId(value) for value in head_classes)
    rejected_memory_access_count = 0
    for index, row in enumerate(rows):
        score = float(scores[index])
        route = route_with_reject_guard(
            head_logits[index],
            reject_score=score,
            threshold=threshold,
            relation_order=relation_order,
        )
        if isinstance(route, RejectedRoute):
            execution = retrieve_for_route(object(), route, object())
            rejected_memory_access_count += execution.memory_access_count
            prediction = None
            access_count: int | None = execution.memory_access_count
            access_reason = "rejected_before_memory"
        else:
            prediction = route.relation_id
            # Public benchmark identities deliberately have no private-memory
            # features.  Acceptance permits D2 access, but no memory is
            # instantiated or touched for this relation-only guard audit.
            access_count = None
            access_reason = "not_executed_relation_only_public_benchmark"
        route_rows.append(
            {
                "row_id": str(row["example_id"]),
                "split": str(row["split"]),
                "kind": str(row["kind"]),
                "family_id": str(row["family_id"]),
                "true_relation_id": row["attribute"],
                "predicted_relation_id": (
                    prediction.value if prediction is not None else None
                ),
                "route_status": route.status,
                "reject_reason": (
                    route.reason if isinstance(route, RejectedRoute) else None
                ),
                "reject_score": score if math.isfinite(score) else None,
                "memory_access_count": access_count,
                "memory_access_permitted": isinstance(route, AcceptedRoute),
                "memory_access_reason": access_reason,
            }
        )
    return metrics, {
        "rows": route_rows,
        "extracted": extracted,
        "rejected_query_memory_access_count": rejected_memory_access_count,
    }


def _fact_metrics_view(value: Mapping[str, Any]) -> dict[str, float]:
    retrieval = value["retrieval"]
    return {
        "fact_centroid_top1": float(retrieval["centroid_top1_accuracy"]),
        "row_1nn": float(retrieval["row_1nn_accuracy"]),
        "mrr": float(retrieval["centroid_mrr"]),
        "mean_centroid_margin": float(retrieval["mean_centroid_margin"]),
        "fact_silhouette": float(value["silhouette"]["fact_id"]),
        "fact_over_template_margin": float(
            value["geometry"]["fact_over_template_margin"]
        ),
        "entity_probe": float(value["linear_probes"]["entity_id"]["accuracy"]),
    }


def _d2_fact_metrics_view(
    retrieval: Mapping[str, Any],
    geometry: Mapping[str, Any],
    entity_probe: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "fact_centroid_top1": float(retrieval["centroid_top1_accuracy"]),
        "row_1nn": float(retrieval["row_1nn_accuracy"]),
        "mrr": float(retrieval["centroid_mrr"]),
        "mean_centroid_margin": float(retrieval["mean_centroid_margin"]),
        "fact_silhouette": float(geometry["fact_silhouette"]),
        "fact_over_template_margin": float(
            geometry["fact_over_template_margin"]
        ),
        "entity_probe": float(entity_probe["accuracy"]),
    }


def _d0_reproduction(
    validation: Mapping[str, Any],
    development: Mapping[str, Any],
    references: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    observed = {
        "validation_fact_centroid_top1": validation["retrieval"][
            "centroid_top1_accuracy"
        ],
        "validation_row_1nn": validation["retrieval"]["row_1nn_accuracy"],
        "validation_mrr": validation["retrieval"]["centroid_mrr"],
        "development_fact_centroid_top1": development["retrieval"][
            "centroid_top1_accuracy"
        ],
        "development_row_1nn": development["retrieval"]["row_1nn_accuracy"],
        "development_mrr": development["retrieval"]["centroid_mrr"],
    }
    details = {
        name: {
            "observed": float(value),
            "reference": float(references[name]),
            "absolute_error": abs(float(value) - float(references[name])),
            "passed": abs(float(value) - float(references[name])) <= tolerance,
        }
        for name, value in observed.items()
    }
    return {
        "tolerance": float(tolerance),
        "all_metrics_reproduced": all(item["passed"] for item in details.values()),
        "metrics": details,
    }


def _oracle_retrieval_values(
    path: str | Path, expected_sha256: str
) -> dict[str, float]:
    source = _validate_file_hash(
        path, expected_sha256, label="C2.3 S0 oracle summary"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("stage") != "C2.3-S0-ground-truth-relation-oracle":
        raise ProtocolViolation("C2.3 oracle summary stage mismatch")
    return {
        split: float(
            payload["evaluation"][split]["query_space"]["retrieval"][
                "centroid_top1_accuracy"
            ]
        )
        for split in ("validation", "development")
    }


def _private_d3_retrieval(
    buckets: FactBuckets,
    entity_embeddings: Tensor,
    rows: Sequence[Mapping[str, Any]],
    head_logits: Tensor,
    definition_logits: Tensor,
    *,
    head_classes: Sequence[str],
    selected_score: str,
    temperature: float,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extracted = extract_reject_score_candidates(
        head_logits,
        definition_logits,
        head_class_order=head_classes,
        definition_class_order=head_classes,
        temperature=temperature,
    )
    scores: Tensor = extracted["scores"][selected_score]
    relation_order = tuple(RelationId(value) for value in head_classes)
    predictions: list[dict[str, Any]] = []
    accepted_count = 0
    memory_access_count = 0
    correct_count = 0
    rejected_access_count = 0
    for index, row in enumerate(rows):
        route = route_with_reject_guard(
            head_logits[index],
            reject_score=float(scores[index]),
            threshold=threshold,
            relation_order=relation_order,
        )
        if isinstance(route, RejectedRoute):
            execution = retrieve_for_route(
                object(),  # rejected branch must not inspect this argument
                route,
                object(),  # rejected branch must not inspect this argument
            )
            rejected_access_count += execution.memory_access_count
            predictions.append(
                {
                    "row_id": str(row.get("example_id", f"private:{index}")),
                    "fact_id": str(row["fact_id"]),
                    "true_relation_id": str(row["attribute"]),
                    "predicted_relation_id": None,
                    "route_status": "reject",
                    "reject_reason": route.reason,
                    "memory_access_count": execution.memory_access_count,
                    "centroid_correct": False,
                }
            )
            continue
        accepted_count += 1
        single_bucket = {route.relation_id: buckets.prototypes[route.relation_id]}
        execution = retrieve_for_route(
            entity_embeddings[index], route, single_bucket
        )
        if execution.retrieval is None:
            raise RuntimeError("accepted D3 route did not retrieve")
        memory_access_count += execution.memory_access_count
        result = execution.retrieval
        predicted_fact = buckets.fact_ids[route.relation_id][result.candidate_index]
        correct = predicted_fact == str(row["fact_id"])
        correct_count += correct
        predictions.append(
            {
                "row_id": str(row.get("example_id", f"private:{index}")),
                "fact_id": str(row["fact_id"]),
                "true_relation_id": str(row["attribute"]),
                "predicted_relation_id": route.relation_id.value,
                "predicted_fact_id": predicted_fact,
                "route_status": "accept",
                "memory_access_count": execution.memory_access_count,
                "cross_relation_candidate_count": result.cross_relation_candidate_count,
                "centroid_correct": correct,
            }
        )
    total = len(rows)
    return {
        "num_queries": total,
        "known_coverage": accepted_count / total,
        "known_accepted_fact_accuracy": (
            correct_count / accepted_count if accepted_count else 0.0
        ),
        "all_query_fact_top1": correct_count / total,
        "memory_access_count": memory_access_count,
        "rejected_query_memory_access_count": rejected_access_count,
        "cross_relation_candidate_count": 0,
    }, predictions


def _grade_closed_contract(
    d2: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    public = d2["route_metrics"]["public_locked_audit"]
    private_validation = d2["route_metrics"]["private_validation"]
    development = d2["route_metrics"]["development"]
    fact_validation = d2["fact_retrieval"]["validation"]["summary"]
    fact_development = d2["fact_retrieval"]["development"]["summary"]
    contract = d2["contract_metrics"]
    checks = {
        "public_family_macro_accuracy": _gate(
            public["relation_family_macro_accuracy"],
            gates["public_family_macro_accuracy"],
        ),
        "worst_family_accuracy": _gate(
            public["worst_family_accuracy"], gates["worst_family_accuracy"]
        ),
        "private_validation_relation_accuracy": _gate(
            private_validation["relation_accuracy"],
            gates["private_validation_relation_accuracy"],
        ),
        "development_relation_accuracy": _gate(
            development["relation_accuracy"],
            gates["development_relation_accuracy"],
        ),
        "fact_centroid_top1": _gate(
            min(
                fact_validation["fact_centroid_top1"],
                fact_development["fact_centroid_top1"],
            ),
            gates["fact_centroid_top1"],
        ),
        "mrr": _gate(
            min(fact_validation["mrr"], fact_development["mrr"]),
            gates["mrr"],
        ),
        "oracle_gap": _gate(
            max(
                fact_validation["oracle_gap"], fact_development["oracle_gap"]
            ),
            gates["oracle_gap"],
            lower=True,
        ),
        "cross_relation_candidate_count": {
            "value": int(contract["cross_relation_candidate_count"]),
            "threshold": 0,
            "direction": "==",
            "passed": int(contract["cross_relation_candidate_count"]) == 0,
        },
        "continuous_relation_values_exposed_to_memory": {
            "value": bool(
                contract["continuous_relation_values_exposed_to_memory"]
            ),
            "threshold": False,
            "direction": "==",
            "passed": not bool(
                contract["continuous_relation_values_exposed_to_memory"]
            ),
        },
        "confidence_exposed_to_memory": {
            "value": bool(contract["confidence_exposed_to_memory"]),
            "threshold": False,
            "direction": "==",
            "passed": not bool(contract["confidence_exposed_to_memory"]),
        },
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "passed_count": sum(bool(item["passed"]) for item in checks.values()),
        "total": len(checks),
        "provisional_due_to_pending_human_review": True,
        "checks": checks,
    }


def _grade_open_guard(
    calibration: Mapping[str, Any],
    locked: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    specifications = {
        "known_detection_auroc": (
            min(
                float(calibration["known_detection_auroc"]),
                float(locked["known_detection_auroc"]),
            ),
            gates["known_detection_auroc"],
            False,
        ),
        "tnr_at_95_known_coverage": (
            min(
                float(calibration["tnr_at_95_known_coverage"]),
                float(locked["tnr_at_95_known_coverage"]),
            ),
            gates["tnr_at_95_known_coverage"],
            False,
        ),
        "ambiguous_false_accept_rate": (
            max(
                float(calibration["ambiguous_false_accept_rate"]),
                float(locked["ambiguous_false_accept_rate"]),
            ),
            gates["ambiguous_false_accept_rate"],
            True,
        ),
        "unrelated_false_accept_rate": (
            max(
                float(calibration["unrelated_false_accept_rate"]),
                float(locked["unrelated_false_accept_rate"]),
            ),
            gates["unrelated_false_accept_rate"],
            True,
        ),
        "worst_reject_family_false_accept_rate": (
            max(
                float(calibration["worst_reject_family_false_accept_rate"]),
                float(locked["worst_reject_family_false_accept_rate"]),
            ),
            gates["worst_reject_family_false_accept_rate"],
            True,
        ),
        "ece": (
            max(float(calibration["known_ece"]), float(locked["known_ece"])),
            gates["ece"],
            True,
        ),
        "known_coverage": (
            min(
                float(calibration["known_coverage"]),
                float(locked["known_coverage"]),
            ),
            gates["known_coverage"],
            False,
        ),
    }
    checks = {
        name: _gate(value, threshold, lower=lower)
        for name, (value, threshold, lower) in specifications.items()
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "passed_count": sum(bool(item["passed"]) for item in checks.values()),
        "total": len(checks),
        "evaluated_on": "public_calibration_and_provisional_public_locked_audit",
        "provisional_due_to_pending_human_review": True,
        "checks": checks,
    }


def _record_protocol_incident(output_dir: Path, message: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "protocol_incident.json"
    if base.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = output_dir / f"protocol_incident_{stamp}.json"
    return _write_json(
        base,
        {
            "schema_version": STAGE_C24_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "status": "stopped_before_valid_completion",
            "incident": str(message),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "historical_results_overwritten": False,
            "provenance": {
                "git": _git_state(),
                "python": sys.version,
                "platform": platform.platform(),
            },
        },
    )


def run_stage_c24_audit(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
    encoder_loader: Callable[..., LoadedSemanticEncoder] = load_semantic_encoder,
    encode_fn: Callable[..., Tensor] = encode_semantic_texts,
) -> dict[str, Any]:
    """Run D0-D3 in the preregistered order and write finite audit artifacts."""

    target = Path(output_dir)
    if (target / "stage_c24_summary.json").exists():
        raise FileExistsError("a completed Stage C2.4 result cannot be overwritten")
    if target.exists() and any(target.glob("protocol_incident*.json")):
        raise FileExistsError(
            "an incident-tainted Stage C2.4 directory cannot be reused"
        )
    try:
        return _run_stage_c24_audit(
            config_path,
            target,
            device_name=device_name,
            encoder_loader=encoder_loader,
            encode_fn=encode_fn,
        )
    except (ProtocolViolation, ValueError, KeyError, TypeError) as exc:
        _record_protocol_incident(target, str(exc))
        raise


def _run_stage_c24_audit(
    config_path: str | Path,
    output_dir: Path,
    *,
    device_name: str | None,
    encoder_loader: Callable[..., LoadedSemanticEncoder],
    encode_fn: Callable[..., Tensor],
) -> dict[str, Any]:
    started = time.perf_counter()
    values, config_metadata = load_stage_c24_config(config_path)
    if (output_dir / "stage_c24_summary.json").exists():
        raise ProtocolViolation("a completed Stage C2.4 result cannot be overwritten")
    fixed_sources = values["fixed_sources"]
    fixed_s5 = values["fixed_s5"]
    fixed_g3 = values["fixed_g3"]
    run = values["run"]
    benchmark_config = values["benchmark"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # The accounting record is read through its bounded C2.3 parser; no path
    # referenced by the historical incident is followed.
    incident_path = _validate_file_hash(
        fixed_sources["protocol_incident"],
        fixed_sources["protocol_incident_sha256"],
        label="C2.3 protocol incident",
    )
    protocol = load_c23_protocol_status(
        incident_path, run_confirmation_data_read=False
    )
    public_rows, benchmark_manifest = _load_prepared_benchmark(benchmark_config)

    for field, hash_field, label in (
        ("private_feature_cache", "private_feature_cache_sha256", "answer-free cache"),
        ("entity_checkpoint", "entity_checkpoint_sha256", "R3 checkpoint"),
        ("stage_c23_config", "stage_c23_config_sha256", "C2.3 config"),
        ("stage_c23_summary", "stage_c23_summary_sha256", "C2.3 audit summary"),
    ):
        _validate_file_hash(
            fixed_sources[field], fixed_sources[hash_field], label=label
        )
    private_cache_file_sha_before = _sha256_file(
        Path(fixed_sources["private_feature_cache"])
    )
    entity_checkpoint_file_sha_before = _sha256_file(
        Path(fixed_sources["entity_checkpoint"])
    )
    c23_summary = json.loads(
        Path(fixed_sources["stage_c23_summary"]).read_text(encoding="utf-8")
    )
    selected_s5 = c23_summary.get("s5", {})
    if (
        c23_summary.get("selected_variant") != fixed_s5["base_encoder_variant"]
        or selected_s5.get("base_encoder_variant")
        != fixed_s5["base_encoder_variant"]
        or selected_s5.get("view") != fixed_s5["view"]
        or float(selected_s5.get("fact_geometry", {}).get("alpha", -1.0))
        != float(fixed_s5["alpha"])
    ):
        raise ProtocolViolation("C2.3 selected S5 record differs from frozen C2.4 inputs")
    if c23_summary.get("zero_shot_definition_matching", {}).get(
        "selected_candidate"
    ) != fixed_g3["selected_candidate"]:
        raise ProtocolViolation("C2.3 selected zero-shot definition candidate changed")
    historical_validation = selected_s5["fact_geometry"]["validation"]["retrieval"]
    historical_development = selected_s5["fact_geometry"][
        "development_diagnostic"
    ]["retrieval"]
    historical_d0_reference = {
        "validation_fact_centroid_top1": historical_validation[
            "centroid_top1_accuracy"
        ],
        "validation_row_1nn": historical_validation["row_1nn_accuracy"],
        "validation_mrr": historical_validation["centroid_mrr"],
        "development_fact_centroid_top1": historical_development[
            "centroid_top1_accuracy"
        ],
        "development_row_1nn": historical_development["row_1nn_accuracy"],
        "development_mrr": historical_development["centroid_mrr"],
    }
    if any(
        float(run["c23_d0_reference"].get(name, float("nan"))) != float(value)
        for name, value in historical_d0_reference.items()
    ):
        raise ProtocolViolation("configured D0 references differ from hashed C2.3 summary")
    device = resolve_device(device_name)
    raw_cache = _load_feature_cache(fixed_sources["private_feature_cache"])
    private_counts = validate_answer_free_private_feature_cache(raw_cache)
    if str(raw_cache.get("source_core_sha256")) != str(
        fixed_sources["source_core_sha256"]
    ):
        raise ProtocolViolation("frozen source core hash differs from C2.3")
    system, checkpoint_payload = load_canonicalizer_checkpoint(
        fixed_sources["entity_checkpoint"], device=device
    )
    labels = validate_oracle_sources(raw_cache, system, checkpoint_payload)
    if set(labels["relations"]) != {relation.value for relation in RelationId}:
        raise ProtocolViolation("private relation labels differ from C2.4 enum")
    private = sanitize_private_metadata(
        raw_cache["metadata"], benchmark_config["private_phrase_views"]
    )
    private_features = {
        split: raw_cache["features"][split]["pools"].float()
        for split in ("train", "validation", "test")
    }
    source_core_before = str(raw_cache["source_core_sha256"])
    del raw_cache
    r3_state_before = _tensor_state_sha256(system.state_dict())

    head = load_fixed_s5_head(
        fixed_sources["stage_c23_head"],
        expected_sha256=fixed_sources["stage_c23_head_sha256"],
        expected_variant=fixed_s5["base_encoder_variant"],
        expected_view=fixed_s5["view"],
        expected_classes=fixed_s5["classes"],
    )
    if float(fixed_s5["alpha"]) != 0.5:
        raise ProtocolViolation("Stage C2.4 must retain C2.3 alpha=0.5")
    spec = SEMANTIC_ENCODER_SPECS[str(fixed_s5["semantic_encoder"])]
    if spec.model_id != fixed_s5["model_id"] or spec.revision != fixed_s5["revision"]:
        raise ProtocolViolation("fixed S5 semantic encoder metadata mismatch")
    encoder = encoder_loader(
        spec, cache_dir=fixed_s5["model_cache_dir"], device=device
    )
    model_manifest = build_model_file_manifest(encoder.snapshot_path, spec)
    if model_manifest["manifest_sha256"] != fixed_s5["model_manifest_sha256"]:
        raise ProtocolViolation("fixed BGE model file manifest mismatch")
    model_weights = [
        item for item in model_manifest["files"] if item["path"] == "model.safetensors"
    ]
    if len(model_weights) != 1 or model_weights[0]["sha256"] != fixed_s5[
        "model_safetensors_sha256"
    ]:
        raise ProtocolViolation("fixed BGE model weight hash mismatch")

    g3_spec = SEMANTIC_ENCODER_SPECS[str(fixed_g3["semantic_encoder"])]
    if (
        g3_spec.model_id != fixed_g3["model_id"]
        or g3_spec.revision != fixed_g3["revision"]
        or fixed_g3["view"] != "phrase_only"
    ):
        raise ProtocolViolation("fixed G3 zero-shot encoder/view metadata mismatch")
    g3_encoder = encoder_loader(
        g3_spec, cache_dir=fixed_g3["model_cache_dir"], device=device
    )
    g3_model_manifest = build_model_file_manifest(
        g3_encoder.snapshot_path, g3_spec
    )
    if g3_model_manifest["manifest_sha256"] != fixed_g3["model_manifest_sha256"]:
        raise ProtocolViolation("fixed MiniLM model file manifest mismatch")
    g3_weights = [
        item
        for item in g3_model_manifest["files"]
        if item["path"] == "model.safetensors"
    ]
    if len(g3_weights) != 1 or g3_weights[0]["sha256"] != fixed_g3[
        "model_safetensors_sha256"
    ]:
        raise ProtocolViolation("fixed MiniLM model weight hash mismatch")

    public_store = _EmbeddingStore(
        run["embedding_cache_dir"], encode_fn=encode_fn, batch_size=int(run["batch_size"])
    )
    private_store = _EmbeddingStore(
        fixed_s5["private_embedding_cache_dir"],
        encode_fn=encode_fn,
        batch_size=int(run["batch_size"]),
    )
    g3_public_store = _EmbeddingStore(
        run["embedding_cache_dir"],
        encode_fn=encode_fn,
        batch_size=int(run["batch_size"]),
    )
    g3_private_store = _EmbeddingStore(
        fixed_g3["embedding_cache_dir"],
        encode_fn=encode_fn,
        batch_size=int(run["batch_size"]),
    )
    view = str(fixed_s5["view"])
    g3_view = str(fixed_g3["view"])

    def public_embedding(split: str) -> Tensor:
        rows = public_rows[split]
        texts = build_semantic_view_texts(rows, view)
        if (
            split == PublicSplit.TRAIN.value
            and semantic_text_sha256(texts)
            != fixed_s5["public_train_semantic_text_sha256"]
        ):
            raise ProtocolViolation(
                "public_train semantic texts differ from fixed C2.3 S5 training data"
            )
        return public_store.get(
            "S4",
            encoder,
            model_manifest,
            dataset=split,
            view=view,
            texts=texts,
            row_ids=_row_ids(rows, prefix=split),
        )

    def public_g3_embedding(split: str) -> Tensor:
        rows = public_rows[split]
        texts = build_semantic_view_texts(rows, g3_view)
        return g3_public_store.get(
            "S2",
            g3_encoder,
            g3_model_manifest,
            dataset=split,
            view=g3_view,
            texts=texts,
            row_ids=_row_ids(rows, prefix=split),
        )

    # Public calibration is the only source for score, temperature and threshold.
    train_public_embedding = public_embedding(PublicSplit.TRAIN.value)
    calibration_embedding = public_embedding(PublicSplit.CALIBRATION.value)
    calibration_g3_embedding = public_g3_embedding(PublicSplit.CALIBRATION.value)
    definitions = _definition_embeddings(
        "S2",
        g3_encoder,
        g3_model_manifest,
        benchmark_config["definitions"],
        g3_private_store,
    )
    train_public_logits = head.logits(train_public_embedding)
    calibration_head_logits = head.logits(calibration_embedding)
    calibration_definition_logits = _definition_logits(
        calibration_g3_embedding, definitions, head.classes
    )
    calibration_targets, calibration_groups, calibration_families = _guard_inputs(
        public_rows[PublicSplit.CALIBRATION.value]
    )
    selection = select_calibrated_reject_guard(
        calibration_head_logits,
        calibration_definition_logits,
        calibration_targets,
        calibration_groups,
        calibration_families,
        head_class_order=head.classes,
        definition_class_order=head.classes,
        temperature_grid=run["temperature_grid"],
        known_coverage_target=float(run["known_coverage_target"]),
        num_ece_bins=int(run["ece_bins"]),
    )
    selected_score = str(selection["selected_score"])
    selected_temperature = float(selection["selected_temperature"])
    selected_threshold = float(selection["selected_threshold"])
    frozen_guard_sha256 = canonical_sha256(
        {
            "score": selected_score,
            "temperature": selected_temperature,
            "threshold": selected_threshold,
            "classes": list(head.classes),
            "head_sha256": fixed_sources["stage_c23_head_sha256"],
            "g3_candidate": fixed_g3["selected_candidate"],
            "g3_model_manifest_sha256": fixed_g3["model_manifest_sha256"],
            "calibration_rows_sha256": _jsonl_sha256(
                public_rows[PublicSplit.CALIBRATION.value]
            ),
        }
    )

    # Everything below observes the frozen setting.  Calibration is reported
    # with the already selected threshold; it is not selected a second time.
    calibration_guard, calibration_runtime = _evaluate_guard_split(
        calibration_head_logits,
        calibration_definition_logits,
        public_rows[PublicSplit.CALIBRATION.value],
        head_classes=head.classes,
        selected_score=selected_score,
        temperature=selected_temperature,
        threshold=selected_threshold,
        known_coverage_target=float(run["known_coverage_target"]),
        ece_bins=int(run["ece_bins"]),
    )
    # Development is the first post-freeze evaluation.  Its four variants are
    # materialized in this single audit pass before locked-audit data is opened.
    private_embeddings: dict[str, Tensor] = {}
    private_semantic: dict[str, Tensor] = {}
    private_g3_semantic: dict[str, Tensor] = {}
    dataset_names = {
        "train": "private_train",
        "validation": "private_validation",
        "test": "private_development",
    }
    for split in ("train", "validation", "test"):
        private_embeddings[split] = _extract_frozen_entity_embeddings(
            system,
            private_features[split],
            batch_size=int(run["batch_size"]),
            device=device,
        )
        texts = build_semantic_view_texts(private[split], view)
        private_semantic[split] = private_store.get(
            "S4",
            encoder,
            model_manifest,
            dataset=dataset_names[split],
            view=view,
            texts=texts,
            row_ids=_row_ids(private[split], prefix=dataset_names[split]),
        )
        g3_texts = build_semantic_view_texts(private[split], g3_view)
        private_g3_semantic[split] = g3_private_store.get(
            "S2",
            g3_encoder,
            g3_model_manifest,
            dataset=dataset_names[split],
            view=g3_view,
            texts=g3_texts,
            row_ids=_row_ids(private[split], prefix=dataset_names[split]),
        )
    private_logits = {
        split: head.logits(embeddings)
        for split, embeddings in private_semantic.items()
    }
    private_definition_logits = {
        split: _definition_logits(embeddings, definitions, head.classes)
        for split, embeddings in private_g3_semantic.items()
    }
    relation_order = tuple(RelationId(value) for value in head.classes)
    alpha = float(fixed_s5["alpha"])
    ridge_strength = float(run["ridge_strength"])

    d0_queries = {
        split: d0_continuous_queries(
            private_embeddings[split], private_logits[split], alpha=alpha
        )
        for split in ("train", "validation", "test")
    }
    d1_queries = {
        split: _hard_fused_queries(
            private_embeddings[split],
            private_logits[split],
            alpha=alpha,
            class_order=head.classes,
        )
        for split in ("train", "validation", "test")
    }
    d0_validation, d0_validation_predictions = answer_free_query_metrics(
        d0_queries["train"],
        private["train"],
        d0_queries["validation"],
        private["validation"],
        ridge_strength=ridge_strength,
    )
    d0_development, d0_development_predictions = answer_free_query_metrics(
        d0_queries["train"],
        private["train"],
        d0_queries["test"],
        private["test"],
        ridge_strength=ridge_strength,
    )
    reproduction = _d0_reproduction(
        d0_validation,
        d0_development,
        run["c23_d0_reference"],
        tolerance=float(run["d0_reproduction_tolerance"]),
    )
    if not reproduction["all_metrics_reproduced"]:
        raise ProtocolViolation("D0 did not reproduce the frozen C2.3 baseline")

    d1_validation, d1_validation_predictions = answer_free_query_metrics(
        d1_queries["train"],
        private["train"],
        d1_queries["validation"],
        private["validation"],
        ridge_strength=ridge_strength,
    )
    d1_development, d1_development_predictions = answer_free_query_metrics(
        d1_queries["train"],
        private["train"],
        d1_queries["test"],
        private["test"],
        ridge_strength=ridge_strength,
    )
    buckets = build_fact_buckets(private_embeddings["train"], private["train"])
    d2_validation_retrieval, d2_validation_predictions = evaluate_bucket_retrieval(
        buckets,
        private_embeddings["validation"],
        private["validation"],
        private_logits["validation"],
        relation_order=relation_order,
    )
    d2_development_retrieval, d2_development_predictions = evaluate_bucket_retrieval(
        buckets,
        private_embeddings["test"],
        private["test"],
        private_logits["test"],
        relation_order=relation_order,
    )
    d2_validation_geometry = bucket_geometry_diagnostic(
        private_embeddings["train"],
        private["train"],
        private_embeddings["validation"],
        private["validation"],
    )
    d2_development_geometry = bucket_geometry_diagnostic(
        private_embeddings["train"],
        private["train"],
        private_embeddings["test"],
        private["test"],
    )
    entity_probe_validation = ridge_probe_accuracy(
        private_embeddings["train"],
        [str(row["entity"]) for row in private["train"]],
        private_embeddings["validation"],
        [str(row["entity"]) for row in private["validation"]],
        ridge_strength=ridge_strength,
    )
    entity_probe_development = ridge_probe_accuracy(
        private_embeddings["train"],
        [str(row["entity"]) for row in private["train"]],
        private_embeddings["test"],
        [str(row["entity"]) for row in private["test"]],
        ridge_strength=ridge_strength,
    )
    oracle_values = _oracle_retrieval_values(
        fixed_sources["stage_c23_oracle_summary"],
        fixed_sources["stage_c23_oracle_summary_sha256"],
    )

    d3_validation, d3_validation_predictions = _private_d3_retrieval(
        buckets,
        private_embeddings["validation"],
        private["validation"],
        private_logits["validation"],
        private_definition_logits["validation"],
        head_classes=head.classes,
        selected_score=selected_score,
        temperature=selected_temperature,
        threshold=selected_threshold,
    )
    d3_development, d3_development_predictions = _private_d3_retrieval(
        buckets,
        private_embeddings["test"],
        private["test"],
        private_logits["test"],
        private_definition_logits["test"],
        head_classes=head.classes,
        selected_score=selected_score,
        temperature=selected_temperature,
        threshold=selected_threshold,
    )

    # The provisional locked audit is opened only after all frozen development
    # variants have completed.  Its result cannot alter the guard or any model.
    public_rows[PublicSplit.LOCKED_AUDIT.value] = _load_locked_audit_after_freeze(
        benchmark_config, benchmark_manifest
    )
    locked_embedding = public_embedding(PublicSplit.LOCKED_AUDIT.value)
    locked_g3_embedding = public_g3_embedding(PublicSplit.LOCKED_AUDIT.value)
    locked_head_logits = head.logits(locked_embedding)
    locked_definition_logits = _definition_logits(
        locked_g3_embedding, definitions, head.classes
    )
    locked_guard, locked_runtime = _evaluate_guard_split(
        locked_head_logits,
        locked_definition_logits,
        public_rows[PublicSplit.LOCKED_AUDIT.value],
        head_classes=head.classes,
        selected_score=selected_score,
        temperature=selected_temperature,
        threshold=selected_threshold,
        known_coverage_target=float(run["known_coverage_target"]),
        ece_bins=int(run["ece_bins"]),
    )
    locked_guard["provisional"] = True
    locked_guard["human_review_status"] = (
        "curated_draft_pending_independent_human_review"
    )

    public_route = {
        "public_train": _route_metrics_for_rows(
            train_public_logits,
            public_rows[PublicSplit.TRAIN.value],
            head.classes,
        ),
        "public_calibration": _route_metrics_for_rows(
            calibration_head_logits,
            public_rows[PublicSplit.CALIBRATION.value],
            head.classes,
        ),
        "public_locked_audit": _route_metrics_for_rows(
            locked_head_logits,
            public_rows[PublicSplit.LOCKED_AUDIT.value],
            head.classes,
        ),
        "private_validation": _private_route_metrics(
            private_logits["validation"], private["validation"], head.classes
        ),
        "development": _private_route_metrics(
            private_logits["test"], private["test"], head.classes
        ),
    }
    global_cross_candidates = len(labels["facts"]) - len(labels["entities"])
    base_contract = {
        "confidence_exposed_to_memory": False,
        "semantic_embedding_exposed_to_memory": False,
        "raw_text_exposed_to_memory": False,
        "phrase_family_exposed_to_memory": False,
        "rejected_query_memory_access_count": 0,
    }
    d0 = {
        "variant": "D0",
        "contract": "continuous_softmax_global_retrieval",
        "route_metrics": public_route,
        "fact_retrieval": {
            "validation": {
                "summary": {
                    **_fact_metrics_view(d0_validation),
                    "oracle_gap": oracle_values["validation"]
                    - d0_validation["retrieval"]["centroid_top1_accuracy"],
                },
                "detail": d0_validation,
            },
            "development": {
                "summary": {
                    **_fact_metrics_view(d0_development),
                    "oracle_gap": oracle_values["development"]
                    - d0_development["retrieval"]["centroid_top1_accuracy"],
                },
                "detail": d0_development,
            },
        },
        "contract_metrics": {
            **base_contract,
            "continuous_relation_values_exposed_to_memory": True,
            "cross_relation_candidate_count": global_cross_candidates,
        },
        "c23_numeric_reproduction": reproduction,
    }
    d1 = {
        "variant": "D1",
        "contract": "hard_one_hot_global_retrieval",
        "route_metrics": public_route,
        "fact_retrieval": {
            "validation": {
                "summary": {
                    **_fact_metrics_view(d1_validation),
                    "oracle_gap": oracle_values["validation"]
                    - d1_validation["retrieval"]["centroid_top1_accuracy"],
                },
                "detail": d1_validation,
            },
            "development": {
                "summary": {
                    **_fact_metrics_view(d1_development),
                    "oracle_gap": oracle_values["development"]
                    - d1_development["retrieval"]["centroid_top1_accuracy"],
                },
                "detail": d1_development,
            },
        },
        "contract_metrics": {
            **base_contract,
            "continuous_relation_values_exposed_to_memory": False,
            "cross_relation_candidate_count": global_cross_candidates,
        },
    }
    discrete_flags = memory_contract_exposure_flags()
    d2 = {
        "variant": "D2",
        "contract": "exact_relation_bucket_retrieval",
        "route_metrics": public_route,
        "fact_retrieval": {
            "validation": {
                "summary": {
                    **_d2_fact_metrics_view(
                        d2_validation_retrieval,
                        d2_validation_geometry,
                        entity_probe_validation,
                    ),
                    "oracle_gap": oracle_values["validation"]
                    - d2_validation_retrieval["centroid_top1_accuracy"],
                },
                "retrieval": d2_validation_retrieval,
                "geometry": d2_validation_geometry,
            },
            "development": {
                "summary": {
                    **_d2_fact_metrics_view(
                        d2_development_retrieval,
                        d2_development_geometry,
                        entity_probe_development,
                    ),
                    "oracle_gap": oracle_values["development"]
                    - d2_development_retrieval["centroid_top1_accuracy"],
                },
                "retrieval": d2_development_retrieval,
                "geometry": d2_development_geometry,
            },
        },
        "contract_metrics": {
            **discrete_flags,
            "cross_relation_candidate_count": 0,
            "rejected_query_memory_access_count": 0,
            "memory_input_fields": ["entity_embedding", "relation_id"],
            "bucket_mapping_entries_visible_per_call": 1,
        },
    }
    d3 = {
        "variant": "D3",
        "contract": "D2_plus_independent_pre_memory_reject_guard",
        "route_metrics": public_route,
        "reject_guard": {
            "selected_score": selected_score,
            "temperature": selected_temperature,
            "threshold": selected_threshold,
            "frozen_guard_sha256": frozen_guard_sha256,
            "selection": selection,
            "public_calibration": calibration_guard,
            "public_locked_audit": locked_guard,
        },
        "fact_retrieval": {
            "validation": d3_validation,
            "development": d3_development,
        },
        "contract_metrics": {
            **discrete_flags,
            "cross_relation_candidate_count": 0,
            "rejected_query_memory_access_count": (
                d3_validation["rejected_query_memory_access_count"]
                + d3_development["rejected_query_memory_access_count"]
                + calibration_runtime["rejected_query_memory_access_count"]
                + locked_runtime["rejected_query_memory_access_count"]
            ),
            "reject_gate_precedes_memory_access": True,
            "confidence_lifetime": "reject_gate_internal_only",
        },
    }
    closed_grade = _grade_closed_contract(d2, values["gates"]["closed_set"])
    open_grade = _grade_open_guard(
        calibration_guard, locked_guard, values["gates"]["open_set"]
    )
    eligibility = compute_eligibility(
        closed_set_discrete_contract_ready=closed_grade["passed"],
        open_set_reject_guard_ready=open_grade["passed"],
        public_benchmark_human_reviewed=False,
        new_pool_created_after_freeze=False,
    )

    system.to("cpu")
    r3_state_after = _tensor_state_sha256(system.state_dict())
    private_cache_file_sha_after = _sha256_file(
        Path(fixed_sources["private_feature_cache"])
    )
    entity_checkpoint_file_sha_after = _sha256_file(
        Path(fixed_sources["entity_checkpoint"])
    )
    core_integrity = {
        "core_model_loaded": False,
        "source_core_provenance_sha256": source_core_before,
        "source_core_provenance_matches_expected": source_core_before
        == str(fixed_sources["source_core_sha256"]),
        "source_core_runtime_rehash_available": False,
        "private_answer_free_cache_sha256_before": private_cache_file_sha_before,
        "private_answer_free_cache_sha256_after": private_cache_file_sha_after,
        "private_answer_free_cache_sha256_equal": private_cache_file_sha_before
        == private_cache_file_sha_after,
        "entity_checkpoint_file_sha256_before": entity_checkpoint_file_sha_before,
        "entity_checkpoint_file_sha256_after": entity_checkpoint_file_sha_after,
        "entity_checkpoint_file_sha256_equal": entity_checkpoint_file_sha_before
        == entity_checkpoint_file_sha_after,
        "r3_state_sha256_before": r3_state_before,
        "r3_state_sha256_after": r3_state_after,
        "r3_state_sha256_equal": r3_state_before == r3_state_after,
    }
    if not all(
        bool(core_integrity[name])
        for name in (
            "source_core_provenance_matches_expected",
            "private_answer_free_cache_sha256_equal",
            "entity_checkpoint_file_sha256_equal",
            "r3_state_sha256_equal",
        )
    ):
        raise ProtocolViolation("core or frozen R3 state changed during audit")

    data_hashes = {
        "public": {
            split: _jsonl_sha256(rows) for split, rows in public_rows.items()
        },
        "private_answer_free_cache": fixed_sources["private_feature_cache_sha256"],
        "entity_checkpoint": fixed_sources["entity_checkpoint_sha256"],
        "fixed_ridge_head": fixed_sources["stage_c23_head_sha256"],
        "protocol_incident": fixed_sources["protocol_incident_sha256"],
    }
    provenance = _base_provenance(
        config_metadata=config_metadata,
        values=values,
        device=device,
        model_manifest=model_manifest,
        g3_model_manifest=g3_model_manifest,
        data_sha256=data_hashes,
    )
    protocol_status = {
        "schema_version": STAGE_C24_SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "provenance": provenance,
        "selection_order": [
            "fixed_C2.3_S5_encoder_view_head",
            "public_calibration_score_temperature_threshold",
            "freeze_guard",
            "development_evaluation_once",
            "public_locked_audit_evaluation_once",
        ],
        "fixed_guard_sha256": frozen_guard_sha256,
        "development_evaluation_count": 1,
        "locked_audit_evaluation_count": 1,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "run_confirmation_data_read": False,
        "retired_confirmation_template_read_count": (
            protocol.retired_confirmation_template_read_count
        ),
        "confirmation_evaluation_count": protocol.confirmation_evaluation_count,
        "new_confirmation_pool_created_after_freeze": False,
        "old_confirmation_seal_reused": False,
        "public_benchmark_human_reviewed": False,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "answer_injection_enabled": False,
        "key_or_attack_experiments_run": False,
        "core_integrity": core_integrity,
        "private_answer_free_row_counts": private_counts,
        **eligibility,
    }

    evaluation_paths = {}
    for name, evaluation in (("D0", d0), ("D1", d1), ("D2", d2), ("D3", d3)):
        evaluation_paths[name] = _write_json(
            output_dir / name / "evaluation.json",
            {
                "schema_version": STAGE_C24_SCHEMA_VERSION,
                "stage": STAGE_NAME,
                "provenance": provenance,
                "evaluation": evaluation,
            },
        )
    ablation_rows = []
    for name, evaluation in (("D0", d0), ("D1", d1), ("D2", d2)):
        validation_summary = evaluation["fact_retrieval"]["validation"]["summary"]
        development_summary = evaluation["fact_retrieval"]["development"]["summary"]
        contract = evaluation["contract_metrics"]
        ablation_rows.append(
            {
                "variant": name,
                "public_locked_family_macro_accuracy": public_route[
                    "public_locked_audit"
                ]["relation_family_macro_accuracy"],
                "public_locked_worst_family_accuracy": public_route[
                    "public_locked_audit"
                ]["worst_family_accuracy"],
                "private_validation_relation_accuracy": public_route[
                    "private_validation"
                ]["relation_accuracy"],
                "development_relation_accuracy": public_route["development"][
                    "relation_accuracy"
                ],
                "validation_fact_centroid_top1": validation_summary[
                    "fact_centroid_top1"
                ],
                "development_fact_centroid_top1": development_summary[
                    "fact_centroid_top1"
                ],
                "validation_mrr": validation_summary["mrr"],
                "development_mrr": development_summary["mrr"],
                "mrr_status": "measured",
                "max_oracle_gap": max(
                    validation_summary["oracle_gap"],
                    development_summary["oracle_gap"],
                ),
                "continuous_relation_values_exposed_to_memory": contract[
                    "continuous_relation_values_exposed_to_memory"
                ],
                "confidence_exposed_to_memory": contract[
                    "confidence_exposed_to_memory"
                ],
                "cross_relation_candidate_count": contract[
                    "cross_relation_candidate_count"
                ],
            }
        )
    ablation_rows.append(
        {
            "variant": "D3",
            "public_locked_family_macro_accuracy": public_route[
                "public_locked_audit"
            ]["relation_family_macro_accuracy"],
            "public_locked_worst_family_accuracy": public_route[
                "public_locked_audit"
            ]["worst_family_accuracy"],
            "private_validation_relation_accuracy": public_route[
                "private_validation"
            ]["relation_accuracy"],
            "development_relation_accuracy": public_route["development"][
                "relation_accuracy"
            ],
            "validation_fact_centroid_top1": d3_validation["all_query_fact_top1"],
            "development_fact_centroid_top1": d3_development[
                "all_query_fact_top1"
            ],
            "validation_mrr": "",
            "development_mrr": "",
            "mrr_status": "not_evaluated_for_D3",
            "max_oracle_gap": max(
                oracle_values["validation"] - d3_validation["all_query_fact_top1"],
                oracle_values["development"]
                - d3_development["all_query_fact_top1"],
            ),
            "continuous_relation_values_exposed_to_memory": False,
            "confidence_exposed_to_memory": False,
            "cross_relation_candidate_count": 0,
        }
    )
    _write_csv(output_dir / "stage_c24_ablation.csv", ablation_rows)

    candidate_rows = []
    for score_name, candidate in selection["candidates"].items():
        metrics = candidate["calibration_metrics"]
        candidate_rows.append(
            {
                "score": score_name,
                "selected": score_name == selected_score,
                "temperature": candidate["temperature"],
                "threshold": candidate["threshold"],
                "known_detection_auroc": metrics["known_detection_auroc"],
                "known_detection_aupr": metrics["known_detection_aupr"],
                "known_ece": metrics["known_ece"],
                "tnr_at_95_known_coverage": metrics[
                    "tnr_at_95_known_coverage"
                ],
                "known_coverage": metrics["known_coverage"],
                "ambiguous_false_accept_rate": metrics[
                    "ambiguous_false_accept_rate"
                ],
                "unrelated_false_accept_rate": metrics[
                    "unrelated_false_accept_rate"
                ],
                "worst_reject_family_false_accept_rate": metrics[
                    "worst_reject_family_false_accept_rate"
                ],
                "research_gate_count": candidate["research_gate_count"],
            }
        )
    _write_csv(output_dir / "reject_score_candidates.csv", candidate_rows)

    retrieval_prediction_rows = []
    for name, split, rows in (
        ("D0", "validation", d0_validation_predictions),
        ("D0", "development", d0_development_predictions),
        ("D1", "validation", d1_validation_predictions),
        ("D1", "development", d1_development_predictions),
        ("D2", "validation", d2_validation_predictions),
        ("D2", "development", d2_development_predictions),
        ("D3", "validation", d3_validation_predictions),
        ("D3", "development", d3_development_predictions),
    ):
        retrieval_prediction_rows.extend(
            {"variant": name, "query_split": split, **row} for row in rows
        )
    _write_json(
        output_dir / "retrieval_predictions.json",
        {
            "schema_version": STAGE_C24_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "provenance": provenance,
            "answer_free": True,
            "rows": retrieval_prediction_rows,
        },
    )
    route_prediction_rows = [
        *calibration_runtime["rows"],
        *locked_runtime["rows"],
    ]
    _write_json(
        output_dir / "route_predictions.json",
        {
            "schema_version": STAGE_C24_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "provenance": provenance,
            "continuous_scores_are_upstream_reject_audit_only": True,
            "rows": route_prediction_rows,
        },
    )
    _write_csv(output_dir / "route_predictions.csv", route_prediction_rows)
    _write_json(
        output_dir / "resolved_config.json",
        {
            "schema_version": STAGE_C24_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "provenance": provenance,
            "resolved_config": values,
        },
    )
    _write_json(output_dir / "protocol_status.json", protocol_status)

    summary = {
        "schema_version": STAGE_C24_SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "completed_provisional_locked_audit",
        "provenance": provenance,
        "fixed_s5": {
            "encoder": fixed_s5["model_id"],
            "revision": fixed_s5["revision"],
            "view": fixed_s5["view"],
            "ridge_head_sha256": fixed_sources["stage_c23_head_sha256"],
            "alpha": alpha,
            "head_refit": False,
            "encoder_reselected": False,
            "view_reselected": False,
        },
        "fixed_g3": {
            "selected_candidate": fixed_g3["selected_candidate"],
            "encoder": fixed_g3["model_id"],
            "revision": fixed_g3["revision"],
            "view": fixed_g3["view"],
            "definition_aggregation": fixed_g3["definition_aggregation"],
            "candidate_reselected_outside_public_calibration": False,
        },
        "public_benchmark": {
            "row_counts": benchmark_manifest["row_counts"],
            "review_status": benchmark_manifest["review_status"],
            "public_benchmark_human_reviewed": False,
            "locked_metrics_provisional": True,
            "split_isolation_passed": benchmark_manifest["audit"][
                "split_isolation"
            ]["passed"],
        },
        "reject_guard": {
            "selected_score": selected_score,
            "temperature": selected_temperature,
            "threshold": selected_threshold,
            "frozen_guard_sha256": frozen_guard_sha256,
            "public_calibration": calibration_guard,
            "public_locked_audit": locked_guard,
        },
        "variants": {
            "D0": {
                "validation": d0["fact_retrieval"]["validation"]["summary"],
                "development": d0["fact_retrieval"]["development"]["summary"],
                "contract_metrics": d0["contract_metrics"],
                "c23_numeric_reproduction": reproduction,
            },
            "D1": {
                "validation": d1["fact_retrieval"]["validation"]["summary"],
                "development": d1["fact_retrieval"]["development"]["summary"],
                "contract_metrics": d1["contract_metrics"],
            },
            "D2": {
                "validation": d2["fact_retrieval"]["validation"]["summary"],
                "development": d2["fact_retrieval"]["development"]["summary"],
                "contract_metrics": d2["contract_metrics"],
            },
            "D3": {
                "validation": d3_validation,
                "development": d3_development,
                "contract_metrics": d3["contract_metrics"],
            },
        },
        "closed_set_grade": closed_grade,
        "open_set_grade": open_grade,
        "eligibility": eligibility,
        "core_integrity": core_integrity,
        "private_answer_free_row_counts": private_counts,
        "protocol": {
            "private_value_memory_trained": False,
            "private_answers_loaded": False,
            "answer_injection_enabled": False,
            "key_or_attack_experiments_run": False,
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
            "new_confirmation_pool_created_after_freeze": False,
            "c3_eligible": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "stage_c24_summary.json", summary)
    _artifact_manifest(output_dir, provenance)
    return summary


prepare_stage_c24_from_config = prepare_stage_c24
