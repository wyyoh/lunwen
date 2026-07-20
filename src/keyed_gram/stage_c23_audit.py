from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from .canonicalizer import load_canonicalizer_checkpoint
from .phase_a import _sha256_file
from .stage_c1 import ridge_probe_accuracy
from .stage_c23 import (
    _extract_frozen_entity_embeddings,
    answer_free_query_metrics,
    load_c23_protocol_status,
    validate_oracle_sources,
    write_answer_free_json,
)
from .stage_c23_benchmark import (
    RELATIONS,
    build_public_lexical_benchmark,
    prepare_public_lexical_benchmark,
)
from .stage_c23_semantic import (
    SEMANTIC_ENCODER_SPECS,
    SEMANTIC_VIEWS,
    LoadedSemanticEncoder,
    SemanticEncoderSpec,
    build_model_file_manifest,
    build_semantic_view_texts,
    definition_prototype_matching,
    encode_semantic_texts,
    family_bootstrap_accuracy_ci,
    family_macro_accuracy,
    fit_ridge_linear_head,
    leave_one_family_out_relation_margin,
    load_semantic_encoder,
    open_set_rejection_metrics,
    semantic_text_sha256,
)
from .stage_c21 import _load_feature_cache
from .train import collect_environment, resolve_device


STAGE_C23_AUDIT_SCHEMA_VERSION = 1
VARIANT_TO_SPEC = {
    "S2": "minilm",
    "S3": "e5-small",
    "S4": "bge-small",
    "S6": "e5-base",
}
SMALL_VARIANTS = ("S2", "S3", "S4")
PRIVATE_SPLIT_ALIASES = {
    "train": "train",
    "validation": "validation",
    "test": "development",
}
SANITIZED_PRIVATE_FIELDS = frozenset(
    {
        "prompt",
        "entity",
        "attribute",
        "fact_id",
        "template_id",
        "relation_phrase",
    }
)


def _canonical_sha256(value: Any) -> str:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                rows.append(value)
    return rows


def sanitize_private_metadata(
    metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    private_phrase_views: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, list[dict[str, str]]]:
    """Copy only query identity fields, then attach the preregistered phrase."""

    if set(metadata) != set(PRIVATE_SPLIT_ALIASES):
        raise ValueError("private metadata must contain train, validation, and test")
    sanitized: dict[str, list[dict[str, str]]] = {}
    required = ("prompt", "entity", "attribute", "fact_id", "template_id")
    for source_split, protocol_split in PRIVATE_SPLIT_ALIASES.items():
        rows = list(metadata[source_split])
        configured = private_phrase_views.get(protocol_split)
        if not isinstance(configured, Mapping):
            raise ValueError(f"private phrase views are missing {protocol_split}")
        observed_relations = {str(row.get("attribute")) for row in rows}
        if set(configured) != observed_relations:
            raise ValueError(
                f"private phrase-view relations differ for {protocol_split}"
            )
        phrase_by_relation_template: dict[tuple[str, str], str] = {}
        for relation in sorted(observed_relations):
            templates = sorted(
                {
                    str(row["template_id"])
                    for row in rows
                    if str(row.get("attribute")) == relation
                }
            )
            phrases = [str(value).strip() for value in configured[relation]]
            if len(phrases) != len(templates) or any(not value for value in phrases):
                raise ValueError(
                    f"private phrase views do not align with {protocol_split}/{relation}"
                )
            for template, phrase in zip(templates, phrases):
                phrase_by_relation_template[(relation, template)] = phrase

        split_rows = []
        for row in rows:
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(f"private row is missing fields {missing}")
            relation = str(row["attribute"])
            template = str(row["template_id"])
            clean = {
                "prompt": str(row["prompt"]),
                "entity": str(row["entity"]),
                "attribute": relation,
                "fact_id": str(row["fact_id"]),
                "template_id": template,
                "relation_phrase": phrase_by_relation_template[(relation, template)],
            }
            if set(clean) != SANITIZED_PRIVATE_FIELDS:
                raise RuntimeError("internal private sanitization schema mismatch")
            split_rows.append(clean)
        sanitized[source_split] = split_rows
    return sanitized


def _validate_public_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    if set(rows) != {"train", "validation", "reject"}:
        raise ValueError("public rows must contain train, validation, and reject")
    output: dict[str, list[dict[str, Any]]] = {}
    for split, split_rows in rows.items():
        copied = []
        for row in split_rows:
            if any(key in row for key in ("answer", "answers", "candidates", "answer_index")):
                raise ValueError("public semantic rows contain an answer-bearing field")
            required = {
                "prompt",
                "entity",
                "relation_phrase",
                "family_id",
                "template_id",
            }
            if required.difference(row):
                raise ValueError("public semantic row has an incomplete schema")
            attribute = row.get("attribute")
            if split == "reject":
                if attribute is not None:
                    raise ValueError("rejection rows must not carry relation targets")
            elif str(attribute) not in RELATIONS:
                raise ValueError("known public row has an invalid relation target")
            copied.append(dict(row))
        if not copied:
            raise ValueError(f"public {split} rows cannot be empty")
        output[split] = copied
    return output


def load_or_build_public_rows(
    config_path: str | Path,
    values: Mapping[str, Any],
    *,
    supplied_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if supplied_rows is not None:
        rows = _validate_public_rows(supplied_rows)
        return rows, {
            "source": "injected_rows",
            "answer_free": True,
            "contains_private_answers": False,
            "row_counts": {split: len(items) for split, items in rows.items()},
            "row_sha256": {
                split: _jsonl_sha256(items) for split, items in rows.items()
            },
        }

    benchmark = values.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("C2.3 config is missing benchmark settings")
    generated, audit = build_public_lexical_benchmark(benchmark)
    manifest = prepare_public_lexical_benchmark(config_path)
    data_dir = Path(benchmark.get("public_data_dir", "data/stage_c23"))
    loaded = {
        split: _read_jsonl(data_dir / f"{split}.jsonl") for split in generated
    }
    if loaded != generated:
        raise ValueError("materialized public benchmark differs from generated rows")
    rows = _validate_public_rows(loaded)
    return rows, {**manifest, "runtime_audit": audit, "source": "generated_and_read"}


def _reject_confirmation_inputs(value: Any, *, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if "confirmation" in name:
                raise ValueError(
                    f"C2.3 semantic audit rejects confirmation input at {location}.{key}"
                )
            _reject_confirmation_inputs(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_confirmation_inputs(item, location=f"{location}[{index}]")


def _resolve_variants(
    values: Mapping[str, Any],
    variants: Sequence[str] | None,
    supplied_specs: Mapping[str, SemanticEncoderSpec] | None,
) -> tuple[tuple[str, ...], dict[str, SemanticEncoderSpec]]:
    audit = values.get("semantic_audit", {})
    requested = tuple(
        str(value)
        for value in (
            variants
            if variants is not None
            else audit.get("small_variants", SMALL_VARIANTS)
        )
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("semantic audit variants must be non-empty and unique")
    unknown = sorted(set(requested).difference(VARIANT_TO_SPEC))
    if unknown:
        raise ValueError(f"unknown C2.3 semantic variants {unknown}")
    specs = {}
    configured_models = values.get("models", {})
    for variant in requested:
        if supplied_specs is not None and variant in supplied_specs:
            spec = supplied_specs[variant]
            spec.validate()
        else:
            spec = SEMANTIC_ENCODER_SPECS[VARIANT_TO_SPEC[variant]]
            configured = configured_models.get(variant)
            if not isinstance(configured, Mapping):
                raise ValueError(f"config is missing pinned model {variant}")
            expected = {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "dimension": spec.dimension,
            }
            if any(configured.get(key) != value for key, value in expected.items()):
                raise ValueError(f"configured model {variant} differs from pinned spec")
        specs[variant] = spec
    return requested, specs


class _EmbeddingStore:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        encode_fn: Callable[..., Tensor],
        batch_size: int,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encode_fn = encode_fn
        self.batch_size = int(batch_size)
        self.entries: list[dict[str, Any]] = []

    def get(
        self,
        variant: str,
        encoder: LoadedSemanticEncoder,
        model_manifest: Mapping[str, Any],
        *,
        dataset: str,
        view: str,
        texts: Sequence[str],
        row_ids: Sequence[str],
        e5_input_type: str = "query",
    ) -> Tensor:
        if len(texts) != len(row_ids):
            raise ValueError("embedding texts and row IDs are incompatible")
        text_sha = semantic_text_sha256(texts)
        row_ids_sha = _canonical_sha256([str(value) for value in row_ids])
        cache_key = hashlib.sha256(
            f"{text_sha}:{row_ids_sha}".encode("utf-8")
        ).hexdigest()
        safe_dataset = dataset.replace("/", "-").replace("\\", "-")
        safe_view = view.replace("/", "-").replace("\\", "-")
        path = self.cache_dir / (
            f"{variant}-{safe_dataset}-{safe_view}-{cache_key[:16]}.pt"
        )
        expected = {
            "format_version": 1,
            "variant": variant,
            "model_id": encoder.spec.model_id,
            "revision": encoder.spec.revision,
            "dimension": encoder.spec.dimension,
            "dataset": dataset,
            "view": view,
            "text_sha256": text_sha,
            "model_manifest_sha256": str(model_manifest["manifest_sha256"]),
            "row_ids_sha256": row_ids_sha,
            "row_count": len(texts),
        }
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if any(payload.get(key) != value for key, value in expected.items()):
                raise ValueError("semantic embedding cache provenance mismatch")
            embeddings = payload.get("embeddings")
        else:
            embeddings = self.encode_fn(
                encoder,
                texts,
                batch_size=self.batch_size,
                e5_input_type=e5_input_type,
            )
            embeddings = F.normalize(embeddings.detach().cpu().float(), p=2, dim=-1)
            if embeddings.shape != (len(texts), encoder.spec.dimension):
                raise ValueError("semantic encoder returned an unexpected shape")
            torch.save({**expected, "embeddings": embeddings}, path)
        if not isinstance(embeddings, Tensor) or embeddings.shape != (
            len(texts),
            encoder.spec.dimension,
        ):
            raise ValueError("cached semantic embeddings are malformed")
        entry = {
            **expected,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if not any(current["path"] == entry["path"] for current in self.entries):
            self.entries.append(entry)
        return F.normalize(embeddings.float(), p=2, dim=-1)


def _view_texts(rows: Sequence[Mapping[str, Any]], view: str) -> list[str]:
    return build_semantic_view_texts(rows, view)


def _row_ids(
    rows: Sequence[Mapping[str, Any]], *, prefix: str
) -> list[str]:
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
    if set(definitions) != set(RELATIONS):
        raise ValueError("public definitions must cover exactly the three relations")
    texts = []
    row_ids = []
    counts = {}
    for relation in RELATIONS:
        values = [str(value).strip() for value in definitions[relation]]
        if not values or any(not value for value in values):
            raise ValueError("public relation definitions cannot be empty")
        counts[relation] = len(values)
        texts.extend(values)
        row_ids.extend(f"definition:{relation}:{index}" for index in range(len(values)))
    encoded = store.get(
        variant,
        encoder,
        model_manifest,
        dataset="public_definitions",
        view="definition",
        texts=texts,
        row_ids=row_ids,
        # E5's official symmetric similarity/classification protocol prefixes
        # both queries and definition prototypes as queries.  ``passage:`` is
        # reserved for the document side of asymmetric retrieval.
        e5_input_type="query",
    )
    output = {}
    offset = 0
    for relation in RELATIONS:
        output[relation] = encoded[offset : offset + counts[relation]]
        offset += counts[relation]
    return output


def _accuracy(predictions: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> float:
    if len(predictions) != len(rows) or not rows:
        raise ValueError("relation predictions and rows are incompatible")
    return sum(
        prediction == str(row["attribute"])
        for prediction, row in zip(predictions, rows)
    ) / len(rows)


def _candidate_metrics(
    public_embeddings: Tensor,
    public_rows: Sequence[Mapping[str, Any]],
    reject_embeddings: Tensor,
    reject_rows: Sequence[Mapping[str, Any]],
    private_embeddings: Tensor,
    private_rows: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Tensor],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ridge_strength: float,
) -> dict[str, Any]:
    public_match = definition_prototype_matching(
        public_embeddings, definitions, aggregation="mean"
    )
    private_match = definition_prototype_matching(
        private_embeddings, definitions, aggregation="mean"
    )
    reject_match = definition_prototype_matching(
        reject_embeddings, definitions, aggregation="mean"
    )
    public_targets = [str(row["attribute"]) for row in public_rows]
    families = [str(row["family_id"]) for row in public_rows]
    family = family_macro_accuracy(
        public_match["predictions"], public_targets, families
    )
    relation_lookup = {
        relation: index for index, relation in enumerate(public_match["classes"])
    }
    open_targets = torch.tensor(
        [relation_lookup[value] for value in public_targets]
        + [-1] * len(reject_rows),
        dtype=torch.long,
    )
    return {
        "definition_aggregation": "mean",
        "public_validation_relation_accuracy": _accuracy(
            public_match["predictions"], public_rows
        ),
        "public_validation_family_macro": family,
        "public_validation_family_bootstrap_ci": family_bootstrap_accuracy_ci(
            public_match["predictions"],
            public_targets,
            families,
            num_resamples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "private_validation_relation_accuracy": _accuracy(
            private_match["predictions"], private_rows
        ),
        "public_validation_lofo_relation_geometry": (
            leave_one_family_out_relation_margin(
                public_embeddings, public_targets, families
            )
        ),
        "raw_encoder_family_leakage_diagnostic": (
            relation_conditioned_projected_family_leakage(
                public_embeddings,
                public_rows,
                ridge_strength=ridge_strength,
                projection_label="raw_frozen_encoder_embedding",
            )
        ),
        "open_set": _json_open_set(
            open_set_rejection_metrics(
                torch.cat([public_match["logits"], reject_match["logits"]], dim=0),
                open_targets,
            )
        ),
    }


def _candidate_score(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> tuple[float, ...]:
    public_macro = float(
        metrics["public_validation_family_macro"]["macro_accuracy"]
    )
    private = float(metrics["private_validation_relation_accuracy"])
    margin = float(
        metrics["public_validation_lofo_relation_geometry"]["macro_margin"]
    )
    checks = (
        public_macro >= float(gates.get("public_family_macro_accuracy", 0.85)),
        private >= float(gates.get("private_validation_relation_accuracy", 0.85)),
        margin >= float(gates.get("relation_family_margin", 0.15)),
    )
    return (
        float(sum(bool(value) for value in checks)),
        min(public_macro, private),
        public_macro,
        private,
        margin,
    )


def relation_conditioned_projected_family_leakage(
    projected_embeddings: Tensor,
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge_strength: float,
    projection_label: str = "softmax_ridge_relation_logits",
) -> dict[str, Any]:
    """Probe family identity within each true relation using held-out frames."""

    if projected_embeddings.ndim != 2 or len(projected_embeddings) != len(rows):
        raise ValueError("projected-family leakage inputs are incompatible")
    per_relation = {}
    for relation in sorted({str(row["attribute"]) for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if str(row["attribute"]) == relation
        ]
        frames = sorted({str(rows[index].get("frame_id", rows[index]["template_id"])) for index in indices})
        if len(frames) < 2:
            raise ValueError("projected-family leakage needs at least two frames")
        split = max(1, math.ceil(2 * len(frames) / 3))
        split = min(split, len(frames) - 1)
        train_frames = set(frames[:split])
        train_indices = [
            index
            for index in indices
            if str(rows[index].get("frame_id", rows[index]["template_id"])) in train_frames
        ]
        evaluation_indices = [index for index in indices if index not in train_indices]
        families = sorted({str(rows[index]["family_id"]) for index in indices})
        train_families = [str(rows[index]["family_id"]) for index in train_indices]
        evaluation_families = [
            str(rows[index]["family_id"]) for index in evaluation_indices
        ]
        if set(train_families) != set(families) or set(evaluation_families) != set(
            families
        ):
            raise ValueError("frame split does not preserve all lexical families")
        head = fit_ridge_linear_head(
            projected_embeddings[train_indices],
            train_families,
            ridge_strength=ridge_strength,
            classes=families,
        )
        logits = head.logits(projected_embeddings[evaluation_indices])
        predictions = [families[int(index)] for index in logits.argmax(dim=1)]
        accuracy = sum(
            prediction == target
            for prediction, target in zip(predictions, evaluation_families)
        ) / len(evaluation_families)
        per_relation[relation] = {
            "accuracy": accuracy,
            "chance_accuracy": 1.0 / len(families),
            "num_families": len(families),
            "train_frames": sorted(train_frames),
            "evaluation_frames": sorted(set(frames).difference(train_frames)),
            "train_rows": len(train_indices),
            "evaluation_rows": len(evaluation_indices),
        }
    return {
        "macro_accuracy": sum(value["accuracy"] for value in per_relation.values())
        / len(per_relation),
        "macro_chance_accuracy": sum(
            value["chance_accuracy"] for value in per_relation.values()
        )
        / len(per_relation),
        "per_relation": per_relation,
        "conditioning": "true_relation",
        "projection": str(projection_label),
        "split": "frame-held-out-within-public-validation",
    }


def fuse_predicted_relation_queries(
    entity_embeddings: Tensor, relation_logits: Tensor, *, alpha: float = 0.5
) -> Tensor:
    if (
        entity_embeddings.ndim != 2
        or relation_logits.ndim != 2
        or len(entity_embeddings) != len(relation_logits)
    ):
        raise ValueError("predicted relation fusion inputs are incompatible")
    if float(alpha) != 0.5:
        raise ValueError("C2.3 preregisters alpha=0.5 for encoder fact geometry")
    probabilities = relation_logits.float().softmax(dim=1)
    entity = F.normalize(entity_embeddings.float(), p=2, dim=-1)
    return F.normalize(torch.cat([entity, float(alpha) * probabilities], dim=1), p=2, dim=-1)


def _prediction_strings(logits: Tensor, classes: Sequence[str]) -> list[str]:
    labels = tuple(str(value) for value in classes)
    return [labels[int(index)] for index in logits.argmax(dim=1)]


def _json_open_set(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"confidence", "predictions"}
    }


def _validation_only_grade(
    s5: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    """Grade an S5 encoder/view without consulting development metrics."""

    leakage = s5["relation_conditioned_projected_family_leakage"]
    leakage_threshold = float(leakage["macro_chance_accuracy"]) + float(
        gates.get("projected_family_probe_chance_margin", 0.10)
    )
    specifications = {
        "public_family_macro_accuracy": (
            s5["public_validation_family_macro"]["macro_accuracy"],
            float(gates.get("public_family_macro_accuracy", 0.85)),
            False,
        ),
        "private_validation_relation_accuracy": (
            s5["private_validation_relation_accuracy"],
            float(gates.get("private_validation_relation_accuracy", 0.85)),
            False,
        ),
        "relation_family_margin": (
            # S5 is selected on its projected relation representation.  The
            # raw frozen-encoder LOFO result remains a diagnostic only.
            s5["public_validation_lofo_relation_geometry"]["macro_margin"],
            float(gates.get("relation_family_margin", 0.15)),
            False,
        ),
        "projected_family_probe": (
            leakage["macro_accuracy"],
            leakage_threshold,
            True,
        ),
        "encoder_fact_centroid_top1": (
            s5["fact_geometry"]["validation"]["retrieval"][
                "centroid_top1_accuracy"
            ],
            float(gates.get("encoder_fact_centroid_top1", 0.80)),
            False,
        ),
        "entity_probe": (
            s5["entity_probe"]["validation"]["accuracy"],
            float(gates.get("entity_probe", 0.90)),
            False,
        ),
    }
    output = {}
    for name, (raw_value, threshold, lower) in specifications.items():
        value = float(raw_value)
        output[name] = {
            "value": value,
            "threshold": threshold,
            "direction": "<=" if lower else ">=",
            "passed": value <= threshold if lower else value >= threshold,
        }
    passed_count = sum(bool(value["passed"]) for value in output.values())
    return {
        "passed": passed_count == len(output),
        "passed_count": passed_count,
        "total": len(output),
        "gates": output,
        "development_consulted": False,
        "relation_family_representation": "softmax_fixed_ridge_logits",
    }


def _s5_candidate_score(s5: Mapping[str, Any]) -> tuple[float, ...]:
    """Validation-only lexicographic selection with gate count first."""

    grade = s5["validation_only_readiness"]
    public_macro = float(
        s5["public_validation_family_macro"]["macro_accuracy"]
    )
    private_accuracy = float(s5["private_validation_relation_accuracy"])
    fact_top1 = float(
        s5["fact_geometry"]["validation"]["retrieval"][
            "centroid_top1_accuracy"
        ]
    )
    projected_margin = float(
        s5["public_validation_lofo_relation_geometry"]["macro_margin"]
    )
    leakage = float(
        s5["relation_conditioned_projected_family_leakage"]["macro_accuracy"]
    )
    open_auroc = float(s5["open_set"]["known_detection_auroc"])
    return (
        float(grade["passed_count"]),
        min(public_macro, private_accuracy, fact_top1),
        public_macro,
        private_accuracy,
        fact_top1,
        projected_margin,
        -leakage,
        open_auroc,
    )


def _strict_grade(
    s5: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    """Add the one-time selected development diagnostic to validation gates."""

    validation = _validation_only_grade(s5, gates)
    output = {name: dict(value) for name, value in validation["gates"].items()}
    entity_threshold = float(gates.get("entity_probe", 0.90))
    entity_value = min(
        float(s5["entity_probe"]["validation"]["accuracy"]),
        float(s5["entity_probe"]["development"]["accuracy"]),
    )
    output["entity_probe"] = {
        "value": entity_value,
        "threshold": entity_threshold,
        "direction": ">=",
        "passed": entity_value >= entity_threshold,
    }
    development_value = float(s5["development_relation_accuracy"])
    development_threshold = float(
        gates.get("development_relation_accuracy", 0.85)
    )
    output["development_relation_accuracy"] = {
        "value": development_value,
        "threshold": development_threshold,
        "direction": ">=",
        "passed": development_value >= development_threshold,
    }
    passed_count = sum(bool(value["passed"]) for value in output.values())
    return {
        "passed": passed_count == len(output),
        "passed_count": passed_count,
        "total": len(output),
        "gates": output,
        "development_consulted": True,
        "relation_family_representation": "softmax_fixed_ridge_logits",
    }


def _write_candidate_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_stage_c23_audit(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    variants: Sequence[str] | None = None,
    device_name: str | None = None,
    encoder_loader: Callable[..., LoadedSemanticEncoder] = load_semantic_encoder,
    encode_fn: Callable[..., Tensor] = encode_semantic_texts,
    variant_specs: Mapping[str, SemanticEncoderSpec] | None = None,
    public_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    model_cache_dir: str | Path | None = None,
    embedding_cache_dir: str | Path | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    """Run S2-S5 without exposing development to encoder/view selection."""

    started = time.perf_counter()
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if int(values.get("schema_version", -1)) != STAGE_C23_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported C2.3 semantic-audit config")
    if values.get("stage") != "C2.3-public-semantic-encoder-audit":
        raise ValueError("unexpected C2.3 semantic-audit stage")
    # The incident record is the only protocol-accounting input allowed to use
    # confirmation terminology. No confirmation data/seal path is accepted.
    config_without_incident = {
        key: item for key, item in values.items() if key != "protocol_incident"
    }
    _reject_confirmation_inputs(config_without_incident)
    protocol = load_c23_protocol_status(
        values["protocol_incident"], run_confirmation_data_read=False
    )
    audit_config = values.get("semantic_audit")
    benchmark_config = values.get("benchmark")
    if not isinstance(audit_config, Mapping) or not isinstance(
        benchmark_config, Mapping
    ):
        raise ValueError("C2.3 config is missing semantic_audit or benchmark")
    input_views = tuple(str(value) for value in audit_config.get("input_views", ()))
    if input_views != SEMANTIC_VIEWS:
        raise ValueError("C2.3 requires all three preregistered semantic views")
    relation_alpha = float(audit_config.get("oracle_alpha", 0.5))
    if relation_alpha != 0.5:
        raise ValueError("C2.3 encoder fact geometry fixes alpha at 0.5")
    requested, specs = _resolve_variants(values, variants, variant_specs)
    public, public_manifest = load_or_build_public_rows(
        config_path, values, supplied_rows=public_rows
    )

    device = resolve_device(device_name)
    raw_private_cache = _load_feature_cache(values["private_feature_cache"])
    system, checkpoint_payload = load_canonicalizer_checkpoint(
        values["entity_checkpoint"], device=device
    )
    labels = validate_oracle_sources(raw_private_cache, system, checkpoint_payload)
    private = sanitize_private_metadata(
        raw_private_cache["metadata"], benchmark_config["private_phrase_views"]
    )
    private_features = {
        split: raw_private_cache["features"][split]["pools"].float()
        for split in PRIVATE_SPLIT_ALIASES
    }
    source_core_sha256 = str(raw_private_cache["source_core_sha256"])
    selected_layer_indices = list(raw_private_cache["selected_layer_indices"])
    # Drop the answer-bearing metadata container immediately after the strict
    # whitelist copy above. Only feature tensors and six-field rows remain.
    del raw_private_cache
    evaluation_batch_size = int(audit_config.get("batch_size", 64))
    entity_embeddings = {
        split: _extract_frozen_entity_embeddings(
            system,
            private_features[split],
            batch_size=evaluation_batch_size,
            device=device,
        )
        for split in ("train", "validation")
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cache = Path(
        model_cache_dir
        if model_cache_dir is not None
        else ".downloads/hf"
    )
    embedding_cache = Path(
        embedding_cache_dir
        if embedding_cache_dir is not None
        else output_dir / "embedding_cache"
    )
    store = _EmbeddingStore(
        embedding_cache,
        encode_fn=encode_fn,
        batch_size=evaluation_batch_size,
    )
    model_manifests: dict[str, dict[str, Any]] = {}
    encoders: dict[str, LoadedSemanticEncoder] = {}
    zero_shot_details: dict[str, dict[str, Any]] = {}
    zero_shot_csv = []
    s5_candidate_details: dict[str, dict[str, Any]] = {}
    s5_candidate_csv = []
    s5_runtime: dict[str, dict[str, Any]] = {}
    gates = audit_config.get("gates", {})
    definitions = benchmark_config.get("definitions")
    if not isinstance(definitions, Mapping):
        raise ValueError("C2.3 benchmark definitions are missing")
    resamples = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else audit_config.get("bootstrap_samples", 10000)
    )
    bootstrap_seed = int(audit_config.get("bootstrap_seed", 2301))
    ridge_strength = float(audit_config.get("ridge_strength", 0.01))
    relation_labels = list(labels["relations"])
    entity_probe_validation = ridge_probe_accuracy(
        entity_embeddings["train"],
        [str(row["entity"]) for row in private["train"]],
        entity_embeddings["validation"],
        [str(row["entity"]) for row in private["validation"]],
        ridge_strength=ridge_strength,
    )

    # Zero-shot definition matching and S5 linear-head selection are two
    # independent reports. Both stop at validation; development is untouched.
    for variant in requested:
        spec = specs[variant]
        encoder = encoder_loader(spec, cache_dir=model_cache, device=device)
        if encoder.spec != spec:
            raise ValueError("loaded semantic encoder does not match its pinned spec")
        manifest = build_model_file_manifest(encoder.snapshot_path, spec)
        model_manifests[variant] = manifest
        encoders[variant] = encoder
        definition_embeddings = _definition_embeddings(
            variant, encoder, manifest, definitions, store
        )
        for view in input_views:
            split_rows = {
                "public_train": public["train"],
                "public_validation": public["validation"],
                "public_reject": public["reject"],
                "private_train": private["train"],
                "private_validation": private["validation"],
            }
            embeddings = {}
            for dataset, rows in split_rows.items():
                texts = _view_texts(rows, view)
                embeddings[dataset] = store.get(
                    variant,
                    encoder,
                    manifest,
                    dataset=dataset,
                    view=view,
                    texts=texts,
                    row_ids=_row_ids(rows, prefix=dataset),
                    e5_input_type="query",
                )

            zero_metrics = _candidate_metrics(
                embeddings["public_validation"],
                public["validation"],
                embeddings["public_reject"],
                public["reject"],
                embeddings["private_validation"],
                private["validation"],
                definition_embeddings,
                bootstrap_samples=resamples,
                bootstrap_seed=bootstrap_seed,
                ridge_strength=ridge_strength,
            )
            zero_score = _candidate_score(zero_metrics, gates)
            key = f"{variant}:{view}"
            zero_shot_details[key] = {
                "variant": variant,
                "view": view,
                "selection_split": "public_validation+private_validation",
                "selection_score": list(zero_score),
                "metrics": zero_metrics,
            }
            zero_shot_csv.append(
                {
                    "variant": variant,
                    "view": view,
                    "selection_gate_count": int(zero_score[0]),
                    "public_family_macro_accuracy": zero_metrics[
                        "public_validation_family_macro"
                    ]["macro_accuracy"],
                    "public_relation_accuracy": zero_metrics[
                        "public_validation_relation_accuracy"
                    ],
                    "private_validation_relation_accuracy": zero_metrics[
                        "private_validation_relation_accuracy"
                    ],
                    "raw_lofo_relation_margin": zero_metrics[
                        "public_validation_lofo_relation_geometry"
                    ]["macro_margin"],
                    "known_detection_auroc": zero_metrics["open_set"][
                        "known_detection_auroc"
                    ],
                }
            )
            head = fit_ridge_linear_head(
                embeddings["public_train"],
                [str(row["attribute"]) for row in public["train"]],
                ridge_strength=ridge_strength,
                classes=relation_labels,
            )
            logits = {
                dataset: head.logits(value) for dataset, value in embeddings.items()
            }
            predictions = {
                dataset: _prediction_strings(value, head.classes)
                for dataset, value in logits.items()
            }
            public_targets = [
                str(row["attribute"]) for row in public["validation"]
            ]
            public_families = [
                str(row["family_id"]) for row in public["validation"]
            ]
            projected = logits["public_validation"].softmax(dim=1)
            projected_lofo = leave_one_family_out_relation_margin(
                projected, public_targets, public_families
            )
            raw_lofo = leave_one_family_out_relation_margin(
                embeddings["public_validation"], public_targets, public_families
            )
            leakage = relation_conditioned_projected_family_leakage(
                projected,
                public["validation"],
                ridge_strength=ridge_strength,
            )
            relation_lookup = {
                value: index for index, value in enumerate(head.classes)
            }
            known_targets = torch.tensor(
                [relation_lookup[value] for value in public_targets]
                + [-1] * len(public["reject"]),
                dtype=torch.long,
            )
            open_set = _json_open_set(
                open_set_rejection_metrics(
                    torch.cat(
                        [logits["public_validation"], logits["public_reject"]],
                        dim=0,
                    ),
                    known_targets,
                )
            )
            private_train_query = fuse_predicted_relation_queries(
                entity_embeddings["train"],
                logits["private_train"],
                alpha=relation_alpha,
            )
            private_validation_query = fuse_predicted_relation_queries(
                entity_embeddings["validation"],
                logits["private_validation"],
                alpha=relation_alpha,
            )
            validation_fact, validation_fact_predictions = answer_free_query_metrics(
                private_train_query,
                private["train"],
                private_validation_query,
                private["validation"],
                ridge_strength=ridge_strength,
            )
            public_family = family_macro_accuracy(
                predictions["public_validation"], public_targets, public_families
            )
            s5_metrics = {
                "public_validation_family_macro": public_family,
                "public_validation_family_bootstrap_ci": (
                    family_bootstrap_accuracy_ci(
                        predictions["public_validation"],
                        public_targets,
                        public_families,
                        num_resamples=resamples,
                        seed=bootstrap_seed,
                    )
                ),
                "public_validation_relation_accuracy": _accuracy(
                    predictions["public_validation"], public["validation"]
                ),
                "private_validation_relation_accuracy": _accuracy(
                    predictions["private_validation"], private["validation"]
                ),
                "public_validation_lofo_relation_geometry": projected_lofo,
                "raw_encoder_lofo_relation_geometry_diagnostic": raw_lofo,
                "relation_conditioned_projected_family_leakage": leakage,
                "open_set": open_set,
                "fact_geometry": {
                    "alpha": relation_alpha,
                    "relation_representation": "softmax_fixed_ridge_logits",
                    "validation": validation_fact,
                },
                "entity_probe": {"validation": entity_probe_validation},
            }
            s5_metrics["validation_only_readiness"] = _validation_only_grade(
                s5_metrics, gates
            )
            s5_score = _s5_candidate_score(s5_metrics)
            s5_candidate_details[key] = {
                "variant": variant,
                "view": view,
                "training_scope": "ridge_head_on_answer_free_public_train",
                "selection_split": "public_validation+private_validation",
                "development_materialized": False,
                "selection_score": list(s5_score),
                "metrics": s5_metrics,
            }
            s5_candidate_csv.append(
                {
                    "variant": variant,
                    "view": view,
                    "validation_gate_count": int(s5_score[0]),
                    "public_family_macro_accuracy": public_family[
                        "macro_accuracy"
                    ],
                    "private_validation_relation_accuracy": s5_metrics[
                        "private_validation_relation_accuracy"
                    ],
                    "projected_lofo_relation_margin": projected_lofo[
                        "macro_margin"
                    ],
                    "raw_lofo_relation_margin_diagnostic": raw_lofo[
                        "macro_margin"
                    ],
                    "projected_family_leakage": leakage["macro_accuracy"],
                    "validation_fact_centroid_top1": validation_fact["retrieval"][
                        "centroid_top1_accuracy"
                    ],
                    "known_detection_auroc": open_set["known_detection_auroc"],
                }
            )
            s5_runtime[key] = {
                "head": head,
                "private_train_query": private_train_query,
                "validation_fact_predictions": validation_fact_predictions,
            }

    zero_shot_selected_key = max(
        zero_shot_details,
        key=lambda key: tuple(zero_shot_details[key]["selection_score"]),
    )
    selected_key = max(
        s5_candidate_details,
        key=lambda key: tuple(s5_candidate_details[key]["selection_score"]),
    )
    selected_variant, selected_view = selected_key.split(":", 1)
    selected_encoder = encoders[selected_variant]
    selected_manifest = model_manifests[selected_variant]
    selected_runtime = s5_runtime[selected_key]
    head = selected_runtime["head"]

    # The selected S5 candidate is now immutable. This is the first and only
    # materialization/evaluation of the private development split.
    development_texts = _view_texts(private["test"], selected_view)
    development_embedding = store.get(
        selected_variant,
        selected_encoder,
        selected_manifest,
        dataset="private_development",
        view=selected_view,
        texts=development_texts,
        row_ids=_row_ids(private["test"], prefix="private_development"),
        e5_input_type="query",
    )
    development_logits = head.logits(development_embedding)
    development_predictions = _prediction_strings(development_logits, head.classes)
    entity_embeddings["test"] = _extract_frozen_entity_embeddings(
        system,
        private_features["test"],
        batch_size=evaluation_batch_size,
        device=device,
    )
    system.to("cpu")
    development_query = fuse_predicted_relation_queries(
        entity_embeddings["test"], development_logits, alpha=relation_alpha
    )
    development_fact, development_fact_predictions = answer_free_query_metrics(
        selected_runtime["private_train_query"],
        private["train"],
        development_query,
        private["test"],
        ridge_strength=ridge_strength,
    )
    entity_probe_development = ridge_probe_accuracy(
        entity_embeddings["train"],
        [str(row["entity"]) for row in private["train"]],
        entity_embeddings["test"],
        [str(row["entity"]) for row in private["test"]],
        ridge_strength=ridge_strength,
    )
    selected_validation = s5_candidate_details[selected_key]["metrics"]
    s5 = {
        "variant": "S5",
        "base_encoder_variant": selected_variant,
        "view": selected_view,
        "training_scope": "fixed_ridge_head_on_answer_free_public_train",
        **selected_validation,
        "development_relation_accuracy": _accuracy(
            development_predictions, private["test"]
        ),
        "development_materialized_after_selection": True,
        "development_evaluation_count": 1,
    }
    s5["fact_geometry"] = {
        **selected_validation["fact_geometry"],
        "development_diagnostic": development_fact,
    }
    s5["entity_probe"] = {
        "validation": entity_probe_validation,
        "development": entity_probe_development,
    }
    s5["strict_readiness"] = _strict_grade(s5, gates)

    head_path = output_dir / "S5" / "ridge_relation_head.pt"
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "stage": "C2.3-S5-public-ridge-relation-head",
            "base_encoder_variant": selected_variant,
            "view": selected_view,
            "classes": list(head.classes),
            "feature_mean": head.feature_mean,
            "feature_scale": head.feature_scale,
            "weights": head.weights,
            "regularizer": head.regularizer,
        },
        head_path,
    )
    predictions_path = write_answer_free_json(
        output_dir / "S5" / "fact_retrieval_predictions.json",
        [
            {**row, "query_split": "validation"}
            for row in selected_runtime["validation_fact_predictions"]
        ]
        + [
            {**row, "query_split": "development"}
            for row in development_fact_predictions
        ],
    )
    zero_shot_csv_path = _write_candidate_csv(
        output_dir / "zero_shot_candidates.csv", zero_shot_csv
    )
    s5_candidate_csv_path = _write_candidate_csv(
        output_dir / "s5_linear_candidates.csv", s5_candidate_csv
    )
    model_manifest_path = write_answer_free_json(
        output_dir / "model_file_manifest.json",
        {
            "schema_version": 1,
            "models": model_manifests,
        },
    )
    embedding_manifest_path = write_answer_free_json(
        output_dir / "embedding_cache_manifest.json",
        {
            "schema_version": 1,
            "cache_entries": sorted(store.entries, key=lambda row: row["path"]),
            "s5_head": {
                "path": str(head_path.resolve()),
                "size_bytes": head_path.stat().st_size,
                "sha256": _sha256_file(head_path),
            },
        },
    )

    small_executed = any(value in SMALL_VARIANTS for value in requested)
    strict_passed = bool(s5["strict_readiness"]["passed"])
    validation_only_passed = bool(s5["validation_only_readiness"]["passed"])
    if "S6" in requested:
        s6_status = "executed_explicitly"
    elif validation_only_passed and small_executed:
        s6_status = "skipped_small_model_validation_gates_passed"
    else:
        s6_status = (
            "requested_small_models_failed_validation_gates_not_automatically_downloaded"
        )
    summary = {
        "schema_version": STAGE_C23_AUDIT_SCHEMA_VERSION,
        "stage": "C2.3-public-semantic-encoder-audit",
        "status": "small_model_passed" if strict_passed else "strict_gates_failed",
        "requested_variants": list(requested),
        "executed_variants": list(requested),
        "selected_candidate": selected_key,
        "selected_variant": selected_variant,
        "selected_view": selected_view,
        "selection_protocol": (
            "zero-shot definition matching is reported independently; S5 fits one "
            "public-train ridge head for every requested encoder/view and selects "
            "only by validation gate count and score; development is evaluated once "
            "after S5 selection"
        ),
        "development_used_for_selection": False,
        "development_materialized_candidates": [selected_key],
        "private_cache_metadata_sanitized_immediately": True,
        "private_sanitized_fields": sorted(SANITIZED_PRIVATE_FIELDS),
        "public_benchmark": public_manifest,
        "zero_shot_definition_matching": {
            "selected_candidate": zero_shot_selected_key,
            "selection_independent_of_s5": True,
            "trials": zero_shot_details,
        },
        "semantic_encoder_trials": zero_shot_details,
        "s5_linear_head_trials": s5_candidate_details,
        "s5": s5,
        "small_model_strict_passed": strict_passed and small_executed,
        "small_model_validation_only_passed": (
            validation_only_passed and small_executed
        ),
        "s6_status": s6_status,
        "c3_eligible": False,
        "c3_eligibility_reason": (
            "C2.3 audits query geometry only; a new independent confirmation pool "
            "must be created after the protocol and selected model are frozen"
        ),
        "run_confirmation_data_read": protocol.run_confirmation_data_read,
        "retired_confirmation_template_read_count": (
            protocol.retired_confirmation_template_read_count
        ),
        "confirmation_evaluation_count": protocol.confirmation_evaluation_count,
        "protocol_status": asdict(protocol),
        "source": {
            "config": str(config_path.resolve()),
            "config_sha256": _sha256_file(config_path),
            "private_feature_cache": str(Path(values["private_feature_cache"]).resolve()),
            "private_feature_cache_sha256": _sha256_file(
                Path(values["private_feature_cache"])
            ),
            "entity_checkpoint": str(Path(values["entity_checkpoint"]).resolve()),
            "entity_checkpoint_sha256": _sha256_file(
                Path(values["entity_checkpoint"])
            ),
            "source_core_sha256": source_core_sha256,
            "selected_layer_indices": selected_layer_indices,
        },
        "environment": collect_environment(device, torch.float32),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "zero_shot_candidate_csv": str(zero_shot_csv_path.resolve()),
            "s5_candidate_csv": str(s5_candidate_csv_path.resolve()),
            "model_file_manifest": str(model_manifest_path.resolve()),
            "embedding_cache_manifest": str(embedding_manifest_path.resolve()),
            "s5_head": str(head_path.resolve()),
            "fact_predictions": str(predictions_path.resolve()),
        },
    }
    summary_path = output_dir / "stage_c23_audit_summary.json"
    artifact_manifest_path = output_dir / "artifact_sha256_manifest.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    summary["artifacts"]["artifact_sha256_manifest"] = str(
        artifact_manifest_path.resolve()
    )
    write_answer_free_json(summary_path, summary)
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in (
                zero_shot_csv_path,
                s5_candidate_csv_path,
                model_manifest_path,
                embedding_manifest_path,
                head_path,
                predictions_path,
                summary_path,
            )
        ],
    }
    write_answer_free_json(artifact_manifest_path, artifact_manifest)
    return summary


run_stage_c23_semantic_audit = run_stage_c23_audit
