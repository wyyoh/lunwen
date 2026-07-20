from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from .canonicalizer import CanonicalizerSystem, load_canonicalizer_checkpoint
from .phase_a import _sha256_file
from .stage_c1 import (
    cosine_silhouette,
    cross_template_retrieval,
    representation_geometry,
    ridge_probe_accuracy,
)
from .stage_c2 import _template_leakage_probe
from .stage_c21 import _load_feature_cache, relation_geometry_margin
from .train import collect_environment, resolve_device


STAGE_C23_SCHEMA_VERSION = 1
DEFAULT_ORACLE_ALPHA_GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

_METRIC_ROW_FIELDS = ("fact_id", "entity", "attribute", "template_id")
_ANSWER_FREE_PRIVATE_METADATA_FIELDS = (
    "fact_id",
    "entity",
    "attribute",
    "template_id",
    "prompt",
)
_PRIVATE_CACHE_TOP_LEVEL_FIELDS = (
    "format_version",
    "source_core_sha256",
    "selected_layers_one_based",
    "selected_layer_indices",
    "q0_baseline_layer_index",
    "pool_names",
    "features",
)
_SENSITIVE_OUTPUT_KEYS = {
    "answer",
    "answers",
    "answer_index",
    "candidates",
    "private_answer",
    "private_value",
}


@dataclass(frozen=True)
class C23ProtocolStatus:
    """Auditable protocol state carried into the S0 run.

    The incident record acknowledges the retired C2.1 confirmation-template read.
    It is not a confirmation dataset and its referenced old seal is never opened.
    """

    run_confirmation_data_read: bool
    retired_confirmation_template_read_count: int
    confirmation_evaluation_count: int
    confirmation_metric_access_count: int
    new_confirmation_status: str
    incident_record: str
    incident_record_sha256: str


def _assert_answer_free_payload(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _SENSITIVE_OUTPUT_KEYS:
                raise ValueError(
                    f"answer-bearing field {key!r} cannot be serialized at {location}"
                )
            _assert_answer_free_payload(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_answer_free_payload(item, location=f"{location}[{index}]")


def write_answer_free_json(path: str | Path, value: Any) -> Path:
    """Write JSON only after recursively rejecting private-answer fields."""

    _assert_answer_free_payload(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return target


def validate_answer_free_private_feature_cache(
    cache: Mapping[str, Any],
) -> dict[str, int]:
    """Require the C2.3 runtime cache to expose only answer-free metadata."""

    metadata = cache.get("metadata")
    features = cache.get("features")
    if not isinstance(metadata, Mapping) or not isinstance(features, Mapping):
        raise ValueError("C2.3 private feature cache is missing metadata or features")
    expected_fields = set(_ANSWER_FREE_PRIVATE_METADATA_FIELDS)
    row_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = metadata.get(split)
        split_features = features.get(split)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"C2.3 private feature cache has invalid {split} rows")
        if not isinstance(split_features, Mapping) or "pools" not in split_features:
            raise ValueError(f"C2.3 private feature cache has invalid {split} features")
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != expected_fields:
                raise ValueError(
                    "C2.3 runtime requires the prepared answer-free private "
                    f"feature cache; {split} metadata fields do not match the whitelist"
                )
        if len(rows) != len(split_features["pools"]):
            raise ValueError(f"C2.3 {split} metadata/features length mismatch")
        row_counts[split] = len(rows)
    return row_counts


def prepare_answer_free_private_feature_cache(
    source_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Create the only private feature-cache shape accepted by C2.3 runtimes."""

    source = Path(source_path)
    target = Path(output_path)
    manifest_target = Path(manifest_path)
    if source.resolve() == target.resolve():
        raise ValueError("answer-free cache output must differ from its source")
    raw = _load_feature_cache(source)
    missing_top_level = [
        name for name in _PRIVATE_CACHE_TOP_LEVEL_FIELDS if name not in raw
    ]
    if missing_top_level:
        raise ValueError(
            f"source private feature cache is missing fields {missing_top_level}"
        )
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("source private feature cache has no metadata")
    stripped_fields: set[str] = set()
    safe_metadata: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        rows = metadata.get(split)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"source private feature cache has invalid {split} rows")
        safe_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"source private feature cache has invalid {split} row")
            missing = [
                name for name in _ANSWER_FREE_PRIVATE_METADATA_FIELDS if name not in row
            ]
            if missing:
                raise ValueError(f"source {split} row is missing fields {missing}")
            stripped_fields.update(set(row) - set(_ANSWER_FREE_PRIVATE_METADATA_FIELDS))
            safe_rows.append(
                {name: row[name] for name in _ANSWER_FREE_PRIVATE_METADATA_FIELDS}
            )
        safe_metadata[split] = safe_rows
    payload = {name: raw[name] for name in _PRIVATE_CACHE_TOP_LEVEL_FIELDS}
    payload["metadata"] = safe_metadata
    row_counts = validate_answer_free_private_feature_cache(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    reloaded = _load_feature_cache(target)
    if validate_answer_free_private_feature_cache(reloaded) != row_counts:
        raise RuntimeError("answer-free private feature cache verification failed")
    manifest = {
        "schema_version": 1,
        "stage": "C2.3-answer-free-private-feature-cache",
        "source": {
            "path": str(source.resolve()),
            "sha256": _sha256_file(source),
        },
        "output": {
            "path": str(target.resolve()),
            "sha256": _sha256_file(target),
        },
        "row_counts": row_counts,
        "metadata_fields": list(_ANSWER_FREE_PRIVATE_METADATA_FIELDS),
        "stripped_field_names": sorted(stripped_fields),
        "preparation_deserialized_source_metadata": True,
        "runtime_cache_answer_free": True,
    }
    write_answer_free_json(manifest_target, manifest)
    return manifest


def prepare_answer_free_private_feature_cache_from_config(
    config_path: str | Path,
) -> dict[str, Any]:
    source = Path(config_path)
    values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    required = (
        "private_feature_cache_source",
        "private_feature_cache",
        "private_feature_cache_manifest",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Stage C2.3 config is missing fields {missing}")
    return prepare_answer_free_private_feature_cache(
        values["private_feature_cache_source"],
        values["private_feature_cache"],
        values["private_feature_cache_manifest"],
    )


def load_c23_protocol_status(
    protocol_incident_path: str | Path,
    *,
    run_confirmation_data_read: bool,
) -> C23ProtocolStatus:
    """Load accounting from a protocol incident, never from confirmation data."""

    if run_confirmation_data_read:
        raise ValueError("Stage C2.3 S0 must not read confirmation data")
    source = Path(protocol_incident_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("incident") != (
        "local_confirmation_template_selection_read_during_read_only_audit"
    ):
        raise ValueError("unrecognized C2.3 protocol incident record")
    scope = payload.get("scope")
    accounting = payload.get("access_accounting")
    if not isinstance(scope, Mapping) or not isinstance(accounting, Mapping):
        raise ValueError("protocol incident is missing scope or access accounting")
    if not bool(scope.get("templates_read")):
        raise ValueError("protocol incident does not acknowledge the retired read")
    for field in (
        "rows_read",
        "private_answers_read",
        "model_evaluation_run",
        "metrics_computed",
    ):
        if bool(scope.get(field, True)):
            raise ValueError(f"protocol incident has incompatible scope field {field}")

    retired_count = int(accounting.get("confirmation_template_read_count", -1))
    evaluation_count = int(accounting.get("confirmation_evaluation_count", -1))
    metric_count = int(accounting.get("confirmation_metric_access_count", -1))
    if retired_count != 1:
        raise ValueError("C2.3 must acknowledge exactly one retired template read")
    if evaluation_count != 0 or metric_count != 0:
        raise ValueError("S0 cannot start after a confirmation evaluation or metric read")
    new_status = str(payload.get("new_confirmation_status", ""))
    if new_status != "not_created":
        raise ValueError("S0 expects the replacement confirmation set to be uncreated")
    old_seal = payload.get("old_seal")
    if not isinstance(old_seal, Mapping) or old_seal.get("status") != (
        "retired_compromised_for_future_confirmation"
    ):
        raise ValueError("protocol incident does not retire the old confirmation seal")

    return C23ProtocolStatus(
        run_confirmation_data_read=False,
        retired_confirmation_template_read_count=retired_count,
        confirmation_evaluation_count=evaluation_count,
        confirmation_metric_access_count=metric_count,
        new_confirmation_status=new_status,
        incident_record=str(source.resolve()),
        incident_record_sha256=_sha256_file(source),
    )


def oracle_relation_one_hot(
    rows: Sequence[Mapping[str, Any]], relation_labels: Sequence[str]
) -> Tensor:
    labels = [str(value) for value in relation_labels]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("oracle relation labels must be non-empty and unique")
    lookup = {value: index for index, value in enumerate(labels)}
    observed = [str(row["attribute"]) for row in rows]
    unknown = sorted(set(observed).difference(lookup))
    if unknown:
        raise ValueError(f"oracle rows contain unknown relation labels {unknown}")
    indices = torch.tensor([lookup[value] for value in observed], dtype=torch.long)
    return F.one_hot(indices, num_classes=len(labels)).float()


def fuse_oracle_queries(
    entity_embeddings: Tensor, relation_embeddings: Tensor, *, alpha: float
) -> Tensor:
    if (
        entity_embeddings.ndim != 2
        or relation_embeddings.ndim != 2
        or len(entity_embeddings) != len(relation_embeddings)
    ):
        raise ValueError("oracle fusion inputs must be compatible matrices")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("oracle relation scale must be finite and non-negative")
    entity = F.normalize(entity_embeddings.float(), dim=-1)
    relation = F.normalize(relation_embeddings.float(), dim=-1)
    return F.normalize(torch.cat([entity, alpha * relation], dim=-1), dim=-1)


def _metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    sanitized = []
    for row in rows:
        missing = [field for field in _METRIC_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"query row is missing answer-free fields {missing}")
        sanitized.append(
            {
                **{field: str(row[field]) for field in _METRIC_ROW_FIELDS},
                # The legacy retrieval helper uses this only when constructing its
                # prediction row. It never contributes to similarities or labels.
                "answer": "<redacted>",
            }
        )
    return sanitized


def answer_free_query_metrics(
    database_embeddings: Tensor,
    database_rows: Sequence[Mapping[str, Any]],
    evaluation_embeddings: Tensor,
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    ridge_strength: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate fact geometry without probing or serializing answer values."""

    database_metric_rows = _metric_rows(database_rows)
    evaluation_metric_rows = _metric_rows(evaluation_rows)
    retrieval, predictions = cross_template_retrieval(
        database_embeddings,
        database_metric_rows,
        evaluation_embeddings,
        evaluation_metric_rows,
    )
    for prediction in predictions:
        prediction.pop("answer", None)
    geometry = representation_geometry(
        database_embeddings,
        database_metric_rows,
        evaluation_embeddings,
        evaluation_metric_rows,
    )
    combined = torch.cat([database_embeddings, evaluation_embeddings], dim=0)
    combined_rows = [*database_metric_rows, *evaluation_metric_rows]
    probes = {
        name: ridge_probe_accuracy(
            database_embeddings,
            [str(row[field]) for row in database_metric_rows],
            evaluation_embeddings,
            [str(row[field]) for row in evaluation_metric_rows],
            ridge_strength=ridge_strength,
        )
        for name, field in (("entity_id", "entity"), ("relation_id", "attribute"))
    }
    silhouette = {
        name: cosine_silhouette(
            combined, [str(row[field]) for row in combined_rows]
        )
        for name, field in (
            ("fact_id", "fact_id"),
            ("entity_id", "entity"),
            ("relation_id", "attribute"),
        )
    }
    return {
        "answer_free": True,
        "retrieval": retrieval,
        "geometry": geometry,
        "linear_probes": probes,
        "silhouette": silhouette,
    }, predictions


def _oracle_gate(
    value: float, threshold: float, *, lower_is_better: bool = False
) -> dict[str, Any]:
    value = float(value)
    threshold = float(threshold)
    return {
        "value": value,
        "threshold": threshold,
        "direction": "<=" if lower_is_better else ">=",
        "passed": value <= threshold if lower_is_better else value >= threshold,
    }


def _oracle_grade(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    query = evaluation["query_space"]
    retrieval = query["retrieval"]
    template = evaluation["oracle_relation_template_probe"]
    gates = {
        "fact_centroid_top1": _oracle_gate(
            retrieval["centroid_top1_accuracy"], 0.90
        ),
        "row_1nn": _oracle_gate(retrieval["row_1nn_accuracy"], 0.75),
        "centroid_mrr": _oracle_gate(retrieval["centroid_mrr"], 0.85),
        "mean_centroid_margin": _oracle_gate(
            retrieval["mean_centroid_margin"], 0.10
        ),
        "fact_silhouette": _oracle_gate(query["silhouette"]["fact_id"], 0.40),
        "fact_template_margin": _oracle_gate(
            query["geometry"]["fact_over_template_margin"], 0.30
        ),
        "entity_branch_probe": _oracle_gate(
            evaluation["entity_branch_probe"]["accuracy"], 0.90
        ),
        "oracle_relation_probe": _oracle_gate(
            evaluation["oracle_relation_probe"]["accuracy"], 1.0
        ),
        "oracle_relation_template_margin": _oracle_gate(
            evaluation["oracle_relation_geometry"]["relation_template_margin"],
            0.15,
        ),
        "oracle_relation_template_probe": _oracle_gate(
            template["accuracy"],
            float(template["chance_accuracy"]) + 0.10,
            lower_is_better=True,
        ),
    }
    passed_count = sum(bool(gate["passed"]) for gate in gates.values())
    return {
        "passed": passed_count == len(gates),
        "passed_count": passed_count,
        "total": len(gates),
        "gates": gates,
    }


def evaluate_oracle_split(
    train_entity_embeddings: Tensor,
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_entity_embeddings: Tensor,
    evaluation_rows: Sequence[Mapping[str, Any]],
    relation_labels: Sequence[str],
    *,
    alpha: float,
    ridge_strength: float,
    template_probe_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_relation = oracle_relation_one_hot(train_rows, relation_labels)
    evaluation_relation = oracle_relation_one_hot(evaluation_rows, relation_labels)
    train_query = fuse_oracle_queries(
        train_entity_embeddings, train_relation, alpha=alpha
    )
    evaluation_query = fuse_oracle_queries(
        evaluation_entity_embeddings, evaluation_relation, alpha=alpha
    )
    query_space, predictions = answer_free_query_metrics(
        train_query,
        train_rows,
        evaluation_query,
        evaluation_rows,
        ridge_strength=ridge_strength,
    )
    counts = Counter(str(row["fact_id"]) for row in train_rows)
    evaluation = {
        "alpha": float(alpha),
        "answer_free": True,
        "query_space": query_space,
        "entity_branch_probe": ridge_probe_accuracy(
            train_entity_embeddings,
            [str(row["entity"]) for row in train_rows],
            evaluation_entity_embeddings,
            [str(row["entity"]) for row in evaluation_rows],
            ridge_strength=ridge_strength,
        ),
        "oracle_relation_probe": ridge_probe_accuracy(
            train_relation,
            [str(row["attribute"]) for row in train_rows],
            evaluation_relation,
            [str(row["attribute"]) for row in evaluation_rows],
            ridge_strength=ridge_strength,
        ),
        "oracle_relation_geometry": relation_geometry_margin(
            evaluation_relation, evaluation_rows
        ),
        "oracle_relation_template_probe": _template_leakage_probe(
            train_relation,
            train_rows,
            ridge_strength=ridge_strength,
            seed=template_probe_seed,
        ),
        # Carried as a diagnostic rather than an S0 gate because contextual
        # template information originates in the already-frozen entity branch.
        "query_template_probe_diagnostic": _template_leakage_probe(
            train_query,
            train_rows,
            ridge_strength=ridge_strength,
            seed=template_probe_seed,
        ),
        "fact_centroid_manifest": {
            "source_split": "train",
            "num_facts": len(counts),
            "minimum_rows_per_fact": min(counts.values()),
            "maximum_rows_per_fact": max(counts.values()),
            "prototype_rule": (
                "mean unit train-query vectors by fact_id, then L2 normalize"
            ),
            "contains_private_answers": False,
        },
    }
    evaluation["strict_oracle_geometry"] = _oracle_grade(evaluation)
    return evaluation, predictions


def oracle_selection_score(
    evaluation: Mapping[str, Any], *, alpha: float
) -> tuple[float, ...]:
    """Validation-only score with retrieval margin before rank-only tie-breakers."""

    grade = evaluation["strict_oracle_geometry"]
    query = evaluation["query_space"]
    retrieval = query["retrieval"]
    return (
        float(grade["passed_count"]),
        float(retrieval["centroid_top1_accuracy"]),
        float(retrieval["row_1nn_accuracy"]),
        float(retrieval["mean_centroid_margin"]),
        float(retrieval["centroid_mrr"]),
        float(query["silhouette"]["fact_id"]),
        float(query["geometry"]["fact_over_template_margin"]),
        float(evaluation["entity_branch_probe"]["accuracy"]),
        -float(alpha),
    )


def select_oracle_alpha(
    validation_evaluations: Mapping[float, Mapping[str, Any]],
) -> tuple[float, tuple[float, ...]]:
    """Select solely from validation results; no development input is accepted."""

    if not validation_evaluations:
        raise ValueError("oracle alpha selection needs validation evaluations")
    scored = {
        float(alpha): oracle_selection_score(evaluation, alpha=float(alpha))
        for alpha, evaluation in validation_evaluations.items()
    }
    selected = max(scored, key=lambda alpha: scored[alpha])
    return selected, scored[selected]


def _validate_alpha_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("oracle alpha grid cannot be empty")
    if any(not math.isfinite(value) or value < 0 for value in grid):
        raise ValueError("oracle alpha grid must contain finite non-negative values")
    if len(set(grid)) != len(grid):
        raise ValueError("oracle alpha grid contains duplicate values")
    return grid


def validate_oracle_sources(
    cache: Mapping[str, Any],
    system: CanonicalizerSystem,
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Validate the R3/cache provenance and the full Cartesian label schema."""

    if checkpoint_payload.get("variant") != "R3":
        raise ValueError("S0 entity branch must come from the selected C2.1 R3")
    if system.config.architecture != "factorized":
        raise ValueError("S0 needs a factorized R3 canonicalizer")
    if checkpoint_payload.get("source_core_sha256") != cache.get(
        "source_core_sha256"
    ):
        raise ValueError("R3 checkpoint and feature cache core hashes differ")
    if tuple(checkpoint_payload.get("selected_core_layers", ())) != tuple(
        cache.get("selected_layer_indices", ())
    ):
        raise ValueError("R3 checkpoint and feature cache layer selections differ")
    if system.config.num_input_layers != len(cache["selected_layer_indices"]):
        raise ValueError("R3 input-layer count does not match the feature cache")

    raw_labels = checkpoint_payload.get("labels")
    if not isinstance(raw_labels, Mapping):
        raise ValueError("R3 checkpoint has no label schema")
    required = ("entities", "relations", "templates", "facts")
    if any(name not in raw_labels for name in required):
        raise ValueError("R3 checkpoint label schema is incomplete")
    labels = {name: [str(value) for value in raw_labels[name]] for name in required}
    if any(not values or len(set(values)) != len(values) for values in labels.values()):
        raise ValueError("R3 checkpoint labels must be non-empty and unique")

    rows_by_split = {
        split: list(cache["metadata"][split])
        for split in ("train", "validation", "test")
    }
    for split, rows in rows_by_split.items():
        for row in rows:
            missing = [field for field in _METRIC_ROW_FIELDS if field not in row]
            if missing:
                raise ValueError(f"{split} row is missing fields {missing}")
    train_rows = rows_by_split["train"]
    expected = {
        "entities": {str(row["entity"]) for row in train_rows},
        "relations": {str(row["attribute"]) for row in train_rows},
        "templates": {str(row["template_id"]) for row in train_rows},
        "facts": {str(row["fact_id"]) for row in train_rows},
    }
    for name in required:
        if set(labels[name]) != expected[name]:
            raise ValueError(f"R3 checkpoint {name} labels do not match training data")
    if system.config.num_entities != len(labels["entities"]):
        raise ValueError("R3 entity-head size does not match labels")
    if system.config.num_relations != len(labels["relations"]):
        raise ValueError("R3 relation-head size does not match labels")
    if system.config.num_templates != len(labels["templates"]):
        raise ValueError("R3 template-head size does not match labels")

    fact_mapping: dict[str, tuple[str, str]] = {}
    for split, rows in rows_by_split.items():
        if {str(row["fact_id"]) for row in rows} != expected["facts"]:
            raise ValueError(f"{split} does not contain the same fact set as train")
        if {str(row["entity"]) for row in rows} != expected["entities"]:
            raise ValueError(f"{split} does not contain the same entity set as train")
        if {str(row["attribute"]) for row in rows} != expected["relations"]:
            raise ValueError(f"{split} does not contain the same relation set as train")
        for row in rows:
            fact = str(row["fact_id"])
            identity = (str(row["entity"]), str(row["attribute"]))
            if fact in fact_mapping and fact_mapping[fact] != identity:
                raise ValueError("a fact_id maps to multiple entity-relation pairs")
            fact_mapping[fact] = identity
    expected_pairs = {
        (entity, relation)
        for entity in expected["entities"]
        for relation in expected["relations"]
    }
    if set(fact_mapping.values()) != expected_pairs:
        raise ValueError("private facts do not form the complete entity-relation product")
    return labels


@torch.inference_mode()
def _extract_frozen_entity_embeddings(
    system: CanonicalizerSystem,
    features: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if batch_size <= 0 or not len(features):
        raise ValueError("entity extraction needs a positive batch and non-empty split")
    if system.canonicalizer.entity_encoder is None:
        raise ValueError("S0 needs the explicit R3 entity encoder")
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    collected = []
    for offset in range(0, len(features), batch_size):
        pools = system.canonicalizer.mixed_pools(
            features[offset : offset + batch_size].to(device)
        )
        entity = system.canonicalizer.entity_encoder(pools[:, 0])
        collected.append(F.normalize(entity.float(), dim=-1).cpu())
    return torch.cat(collected, dim=0)


def run_stage_c23_oracle(
    feature_cache_path: str | Path,
    entity_checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    protocol_incident_path: str | Path,
    run_confirmation_data_read: bool,
    alpha_grid: Sequence[float] = DEFAULT_ORACLE_ALPHA_GRID,
    evaluation_batch_size: int = 128,
    ridge_strength: float = 0.01,
    template_probe_seed: int = 91,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the zero-training S0 oracle with validation-only scale selection."""

    started = time.perf_counter()
    protocol = load_c23_protocol_status(
        protocol_incident_path,
        run_confirmation_data_read=run_confirmation_data_read,
    )
    grid = _validate_alpha_grid(alpha_grid)
    if evaluation_batch_size <= 0 or ridge_strength <= 0:
        raise ValueError("evaluation batch size and ridge strength must be positive")
    cache_path = Path(feature_cache_path)
    checkpoint_path = Path(entity_checkpoint_path)
    cache = _load_feature_cache(cache_path)
    validate_answer_free_private_feature_cache(cache)
    device = resolve_device(device_name)
    system, checkpoint_payload = load_canonicalizer_checkpoint(
        checkpoint_path, device=device
    )
    labels = validate_oracle_sources(cache, system, checkpoint_payload)
    split_rows = {
        split: list(cache["metadata"][split])
        for split in ("train", "validation", "test")
    }
    entity_embeddings = {
        split: _extract_frozen_entity_embeddings(
            system,
            cache["features"][split]["pools"].float(),
            batch_size=evaluation_batch_size,
            device=device,
        )
        for split in split_rows
    }
    system.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_evaluations: dict[float, dict[str, Any]] = {}
    validation_predictions: dict[float, list[dict[str, Any]]] = {}
    sweep = []
    for alpha in grid:
        evaluation, predictions = evaluate_oracle_split(
            entity_embeddings["train"],
            split_rows["train"],
            entity_embeddings["validation"],
            split_rows["validation"],
            labels["relations"],
            alpha=alpha,
            ridge_strength=ridge_strength,
            template_probe_seed=template_probe_seed,
        )
        score = oracle_selection_score(evaluation, alpha=alpha)
        validation_evaluations[alpha] = evaluation
        validation_predictions[alpha] = predictions
        sweep.append(
            {
                "alpha": alpha,
                "selection_split": "validation",
                "selection_score": list(score),
                "evaluation": evaluation,
            }
        )
    selected_alpha, selected_score = select_oracle_alpha(validation_evaluations)
    selected_validation = validation_evaluations[selected_alpha]

    # Development metrics are computed only after the validation sweep is
    # complete and the selected alpha has been frozen.
    development, development_predictions = evaluate_oracle_split(
        entity_embeddings["train"],
        split_rows["train"],
        entity_embeddings["test"],
        split_rows["test"],
        labels["relations"],
        alpha=selected_alpha,
        ridge_strength=ridge_strength,
        template_probe_seed=template_probe_seed,
    )
    oracle_passed = bool(
        selected_validation["strict_oracle_geometry"]["passed"]
        and development["strict_oracle_geometry"]["passed"]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = [
        {**row, "query_split": "validation", "alpha": selected_alpha}
        for row in validation_predictions[selected_alpha]
    ] + [
        {**row, "query_split": "development", "alpha": selected_alpha}
        for row in development_predictions
    ]
    sweep_path = write_answer_free_json(
        output_dir / "oracle_alpha_sweep.json", sweep
    )
    predictions_path = write_answer_free_json(
        output_dir / "retrieval_predictions.json", predictions
    )
    summary = {
        "schema_version": STAGE_C23_SCHEMA_VERSION,
        "stage": "C2.3-S0-ground-truth-relation-oracle",
        "status": "oracle_upper_bound_passed" if oracle_passed else "oracle_upper_bound_failed",
        "zero_training": True,
        "core_frozen": True,
        "entity_branch_frozen": True,
        "oracle_relation_labels_used": True,
        "fusion": "l2_normalize(concat(l2_normalize(z_e), alpha * one_hot(r)))",
        "memory_attached": False,
        "answer_injection_enabled": False,
        "private_answers_used_as_training_targets": False,
        "private_answers_deserialized_by_runtime": False,
        "private_answers_serialized": False,
        "run_confirmation_data_read": protocol.run_confirmation_data_read,
        "retired_confirmation_template_read_count": (
            protocol.retired_confirmation_template_read_count
        ),
        "confirmation_evaluation_count": protocol.confirmation_evaluation_count,
        "protocol_status": asdict(protocol),
        "c3_eligible": False,
        "c3_eligibility_reason": (
            "S0 is only a relation-oracle upper bound; public semantic encoder "
            "audits and a replacement confirmation protocol remain required"
        ),
        "oracle_upper_bound_passed": oracle_passed,
        "selected_alpha": selected_alpha,
        "selected_alpha_validation_score": list(selected_score),
        "selection_split": "validation",
        "selection_protocol": (
            "all alpha candidates are ranked only on private validation; the "
            "legacy test split is evaluated once as development after alpha freezes"
        ),
        "development_evaluated_after_selection": True,
        "relation_label_order": labels["relations"],
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "evaluation": {
            "validation": selected_validation,
            "development": development,
        },
        "source": {
            "feature_cache": str(cache_path.resolve()),
            "feature_cache_sha256": _sha256_file(cache_path),
            "entity_checkpoint": str(checkpoint_path.resolve()),
            "entity_checkpoint_sha256": _sha256_file(checkpoint_path),
            "source_core_sha256": cache["source_core_sha256"],
            "selected_layer_indices": list(cache["selected_layer_indices"]),
            "checkpoint_variant": checkpoint_payload["variant"],
        },
        "environment": collect_environment(device, torch.float32),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "alpha_sweep": str(sweep_path.resolve()),
            "retrieval_predictions": str(predictions_path.resolve()),
        },
    }
    summary_path = output_dir / "stage_c23_s0_summary.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    write_answer_free_json(summary_path, summary)
    return summary


def run_stage_c23_oracle_from_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the preregistered S0 configuration without a confirmation-data option."""

    source = Path(config_path)
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Stage C2.3 config must contain a mapping")
    if int(values.get("schema_version", -1)) != STAGE_C23_SCHEMA_VERSION:
        raise ValueError("unsupported Stage C2.3 config schema")
    if values.get("stage") != "C2.3-public-semantic-encoder-audit":
        raise ValueError("unexpected Stage C2.3 config stage")
    oracle = values.get("oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError("Stage C2.3 config is missing oracle settings")
    return run_stage_c23_oracle(
        values["private_feature_cache"],
        values["entity_checkpoint"],
        output_dir,
        protocol_incident_path=values["protocol_incident"],
        run_confirmation_data_read=False,
        alpha_grid=oracle.get("alpha_grid", DEFAULT_ORACLE_ALPHA_GRID),
        evaluation_batch_size=int(oracle.get("evaluation_batch_size", 128)),
        ridge_strength=float(oracle.get("ridge_strength", 0.01)),
        template_probe_seed=int(oracle.get("template_probe_seed", 91)),
        device_name=device_name,
    )
