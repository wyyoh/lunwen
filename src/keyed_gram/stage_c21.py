from __future__ import annotations

import csv
import itertools
import json
import random
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm.auto import tqdm

from .canonicalizer import (
    POOL_NAMES,
    CanonicalizerConfig,
    CanonicalizerSystem,
    StructuredFactBatchSampler,
    load_canonicalizer_checkpoint,
    save_canonicalizer_checkpoint,
    supervised_contrastive_loss,
)
from .phase_a import _sha256_file
from .stage_c1 import ridge_probe_accuracy
from .stage_c2 import (
    _clone_state,
    _label_map,
    _lr_scale,
    _query_space_metrics,
    _targets,
    _template_leakage_probe,
)
from .train import collect_environment, resolve_device, set_seed


STAGE_C21_SCHEMA_VERSION = 1


CONFIRMATION_RELATION_PHRASES = {
    "registry_id": (
        "enrollment identifier",
        "archival locator",
        "census reference",
        "record locator",
    ),
    "city_code": (
        "municipal identifier",
        "geographic locator",
        "urban reference",
        "locality marker",
    ),
    "access_code": (
        "authorization token",
        "entry marker",
        "admission secret",
        "permission token",
    ),
}

CONFIRMATION_FRAMES = (
    "Which {relation_phrase} belongs to {entity}? Reply:",
    "For {entity}, provide the {relation_phrase}. Answer:",
    "Locate {entity}'s {relation_phrase}:",
    "What is the {relation_phrase} assigned to {entity}? Value:",
)


@dataclass(frozen=True)
class RelationSource:
    source_id: str
    family: str
    pool_indices: tuple[int, ...]
    layer_offsets: tuple[int, ...]
    layers_one_based: tuple[int, ...]
    input_dim: int
    raw_core_source: bool = True


@dataclass(frozen=True)
class C21Variant:
    name: str
    import_q3: bool
    relation_ce_weight: float
    relation_supcon_weight: float
    entity_ce_weight: float
    template_adversary_weight: float
    relation_first: bool
    normalized_gated_fusion: bool
    relation_encoder_joint_lr_scale: float

    def validate(self) -> None:
        for name in (
            "relation_ce_weight",
            "relation_supcon_weight",
            "entity_ce_weight",
            "template_adversary_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < self.relation_encoder_joint_lr_scale <= 1:
            raise ValueError("relation_encoder_joint_lr_scale must be in (0, 1]")
        if self.import_q3 and self.name != "R0":
            raise ValueError("only R0 may import the exact Stage-C2 Q3 baseline")


DEFAULT_C21_VARIANTS = (
    C21Variant("R0", True, 0.2, 0.0, 0.2, 0.0, False, False, 1.0),
    C21Variant("R1", False, 0.5, 0.2, 0.2, 0.0, False, False, 1.0),
    C21Variant("R2", False, 0.5, 0.2, 0.2, 0.05, False, False, 1.0),
    C21Variant("R3", False, 0.5, 0.2, 0.2, 0.05, True, False, 0.2),
    C21Variant("R4", False, 0.5, 0.2, 0.2, 0.05, True, True, 0.2),
)


def parse_c21_variants(raw: Any) -> tuple[C21Variant, ...]:
    defaults = {variant.name: variant for variant in DEFAULT_C21_VARIANTS}
    if not raw:
        return DEFAULT_C21_VARIANTS
    if set(raw) != set(defaults):
        raise ValueError("Stage C2.1 requires exactly R0, R1, R2, R3, and R4")
    variants = []
    for name in ("R0", "R1", "R2", "R3", "R4"):
        default = defaults[name]
        values = raw[name] or {}
        variant = C21Variant(
            name=name,
            import_q3=bool(values.get("import_q3", default.import_q3)),
            relation_ce_weight=float(
                values.get("relation_ce_weight", default.relation_ce_weight)
            ),
            relation_supcon_weight=float(
                values.get("relation_supcon_weight", default.relation_supcon_weight)
            ),
            entity_ce_weight=float(
                values.get("entity_ce_weight", default.entity_ce_weight)
            ),
            template_adversary_weight=float(
                values.get(
                    "template_adversary_weight",
                    default.template_adversary_weight,
                )
            ),
            relation_first=bool(
                values.get("relation_first", default.relation_first)
            ),
            normalized_gated_fusion=bool(
                values.get(
                    "normalized_gated_fusion",
                    default.normalized_gated_fusion,
                )
            ),
            relation_encoder_joint_lr_scale=float(
                values.get(
                    "relation_encoder_joint_lr_scale",
                    default.relation_encoder_joint_lr_scale,
                )
            ),
        )
        variant.validate()
        variants.append(variant)
    if not variants[0].import_q3 or any(value.import_q3 for value in variants[1:]):
        raise ValueError("R0 must be the only imported Q3 baseline")
    return tuple(variants)


def relation_geometry_margin(
    embeddings: Tensor, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Contrast relation invariance with a same-template hard negative."""

    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise ValueError("relation geometry inputs have incompatible shapes")
    unit = F.normalize(embeddings.float(), dim=-1)
    similarity = unit @ unit.T
    positive: list[float] = []
    negative: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            left_row = rows[left]
            right_row = rows[right]
            same_relation = str(left_row["attribute"]) == str(
                right_row["attribute"]
            )
            same_entity = str(left_row["entity"]) == str(right_row["entity"])
            same_template = str(left_row["template_id"]) == str(
                right_row["template_id"]
            )
            value = float(similarity[left, right])
            if same_relation and not same_entity and not same_template:
                positive.append(value)
            elif not same_relation and same_entity and same_template:
                negative.append(value)
    if not positive or not negative:
        raise ValueError("relation geometry needs both positive and hard-negative pairs")
    positive_mean = float(np.mean(positive))
    negative_mean = float(np.mean(negative))
    return {
        "same_relation_different_entity_template_cosine": positive_mean,
        "different_relation_same_entity_template_cosine": negative_mean,
        "relation_template_margin": positive_mean - negative_mean,
        "positive_pair_count": len(positive),
        "hard_negative_pair_count": len(negative),
    }


@torch.inference_mode()
def _embed_branch_outputs(
    system: CanonicalizerSystem,
    features: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Tensor]:
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    if system.config.architecture != "factorized":
        raise ValueError("Stage C2.1 requires a factorized canonicalizer")
    system.eval()
    collected: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "query",
            "z_e",
            "z_r",
            "relation_logits",
            "entity_logits",
            "entity_contribution",
            "relation_contribution",
            "interaction_contribution",
        )
    }
    fusion_scales: Tensor | None = None
    for offset in range(0, len(features), batch_size):
        output = system(features[offset : offset + batch_size].to(device))
        mapping = {
            "query": output["query"],
            "z_e": output["entity_unit"],
            "z_r": output["relation_unit"],
            "relation_logits": output["relation_logits"],
            "entity_logits": output["entity_logits"],
            "entity_contribution": output["entity_contribution"],
            "relation_contribution": output["relation_contribution"],
            "interaction_contribution": output["interaction_contribution"],
        }
        for name, value in mapping.items():
            collected[name].append(value.float().cpu())
        current_scales = output["fusion_scales"].float().cpu()
        if fusion_scales is not None and not torch.equal(
            fusion_scales, current_scales
        ):
            raise RuntimeError("fusion scales changed during inference")
        fusion_scales = current_scales
    if fusion_scales is None:
        raise ValueError("cannot embed an empty feature split")
    return {
        **{name: torch.cat(values, dim=0) for name, values in collected.items()},
        "fusion_scales": fusion_scales,
    }


def _fusion_contribution_metrics(outputs: dict[str, Tensor]) -> dict[str, Any]:
    names = ("entity", "relation", "interaction")
    means = {}
    standard_deviations = {}
    for name in names:
        norms = outputs[f"{name}_contribution"].norm(dim=-1)
        means[name] = float(norms.mean())
        standard_deviations[name] = float(norms.std(unbiased=False))
    total = sum(means.values())
    shares = {
        name: (means[name] / total if total > 0 else 0.0) for name in names
    }
    return {
        "mean_l2_norm": means,
        "standard_deviation_l2_norm": standard_deviations,
        "norm_share": shares,
        "learned_scales": [float(value) for value in outputs["fusion_scales"]],
    }


def _head_accuracies(
    outputs: dict[str, Tensor],
    rows: Sequence[dict[str, Any]],
    labels: dict[str, Sequence[str]],
) -> dict[str, float]:
    relation_lookup = {
        str(value): index for index, value in enumerate(labels["relations"])
    }
    entity_lookup = {
        str(value): index for index, value in enumerate(labels["entities"])
    }
    relation_targets = torch.tensor(
        [relation_lookup[str(row["attribute"])] for row in rows]
    )
    entity_targets = torch.tensor(
        [entity_lookup[str(row["entity"])] for row in rows]
    )
    return {
        "relation_accuracy": float(
            outputs["relation_logits"]
            .argmax(dim=1)
            .eq(relation_targets)
            .float()
            .mean()
        ),
        "entity_accuracy": float(
            outputs["entity_logits"]
            .argmax(dim=1)
            .eq(entity_targets)
            .float()
            .mean()
        ),
    }


def _branch_information_matrix(
    database_outputs: dict[str, Tensor],
    database_rows: Sequence[dict[str, Any]],
    evaluation_outputs: dict[str, Tensor],
    evaluation_rows: Sequence[dict[str, Any]],
    *,
    ridge_strength: float,
    template_probe_seed: int,
) -> dict[str, Any]:
    matrix = {}
    for output_name in ("z_e", "z_r"):
        database = database_outputs[output_name]
        evaluation = evaluation_outputs[output_name]
        matrix[output_name] = {
            "entity": ridge_probe_accuracy(
                database,
                [str(row["entity"]) for row in database_rows],
                evaluation,
                [str(row["entity"]) for row in evaluation_rows],
                ridge_strength=ridge_strength,
            ),
            "relation": ridge_probe_accuracy(
                database,
                [str(row["attribute"]) for row in database_rows],
                evaluation,
                [str(row["attribute"]) for row in evaluation_rows],
                ridge_strength=ridge_strength,
            ),
            # Template vocabularies are intentionally disjoint across splits.
            # Measure leakage within train using an entity-held-out probe instead
            # of creating a meaningless train-template -> OOD-template classifier.
            "template": _template_leakage_probe(
                database,
                database_rows,
                ridge_strength=ridge_strength,
                seed=template_probe_seed,
            ),
        }
    return matrix


def _strict_c21_grade(evaluation: dict[str, Any]) -> dict[str, Any]:
    query = evaluation["query_space"]
    retrieval = query["retrieval"]
    geometry = query["geometry"]
    template = evaluation["query_template_probe"]
    matrix = evaluation["branch_information_matrix"]
    template_threshold = float(template["chance_accuracy"]) + 0.10
    specifications = {
        "relation_head": (
            evaluation["auxiliary_heads"]["relation_accuracy"],
            0.85,
            False,
        ),
        "relation_probe": (
            matrix["z_r"]["relation"]["accuracy"],
            0.85,
            False,
        ),
        "fact_centroid_top1": (
            retrieval["centroid_top1_accuracy"],
            0.80,
            False,
        ),
        "row_1nn": (retrieval["row_1nn_accuracy"], 0.75, False),
        "centroid_mrr": (retrieval["centroid_mrr"], 0.85, False),
        "fact_silhouette": (query["silhouette"]["fact_id"], 0.40, False),
        "fact_template_margin": (
            geometry["fact_over_template_margin"],
            0.30,
            False,
        ),
        "entity_probe": (
            query["linear_probes"]["entity_id"]["accuracy"],
            0.90,
            False,
        ),
        "template_probe": (template["accuracy"], template_threshold, True),
        "relation_template_margin": (
            evaluation["relation_geometry"]["relation_template_margin"],
            0.15,
            False,
        ),
    }
    gates = {}
    for name, (value, threshold, lower_is_better) in specifications.items():
        value = float(value)
        threshold = float(threshold)
        gates[name] = {
            "value": value,
            "threshold": threshold,
            "direction": "<=" if lower_is_better else ">=",
            "passed": value <= threshold if lower_is_better else value >= threshold,
        }
    passed_count = sum(bool(value["passed"]) for value in gates.values())
    return {
        "passed": passed_count == len(gates),
        "passed_count": passed_count,
        "total": len(gates),
        "gates": gates,
    }


def _selection_score(evaluation: dict[str, Any]) -> tuple[float, ...]:
    """Validation-only lexicographic score; development never calls selection."""

    grade = evaluation["strict_readiness"]
    query = evaluation["query_space"]
    retrieval = query["retrieval"]
    geometry = query["geometry"]
    matrix = evaluation["branch_information_matrix"]
    return (
        float(grade["passed_count"]),
        float(retrieval["centroid_top1_accuracy"]),
        float(retrieval["row_1nn_accuracy"]),
        float(retrieval["centroid_mrr"]),
        float(geometry["fact_over_template_margin"]),
        float(query["silhouette"]["fact_id"]),
        float(evaluation["auxiliary_heads"]["relation_accuracy"]),
        float(matrix["z_r"]["relation"]["accuracy"]),
        float(evaluation["relation_geometry"]["relation_template_margin"]),
        float(query["linear_probes"]["entity_id"]["accuracy"]),
        -float(evaluation["query_template_probe"]["accuracy"]),
    )


def _evaluate_c21_split(
    outputs: dict[str, dict[str, Tensor]],
    split_rows: dict[str, list[dict[str, Any]]],
    labels: dict[str, Sequence[str]],
    evaluation_split: str,
    *,
    ridge_strength: float,
    template_probe_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if evaluation_split == "train":
        raise ValueError("evaluation split must differ from the retrieval database")
    query_metrics, predictions = _query_space_metrics(
        outputs["train"]["query"],
        split_rows["train"],
        outputs[evaluation_split]["query"],
        split_rows[evaluation_split],
        ridge_strength=ridge_strength,
    )
    evaluation = {
        "split": "development" if evaluation_split == "test" else evaluation_split,
        "source_split": evaluation_split,
        "query_space": query_metrics,
        "query_template_probe": _template_leakage_probe(
            outputs["train"]["query"],
            split_rows["train"],
            ridge_strength=ridge_strength,
            seed=template_probe_seed,
        ),
        "auxiliary_heads": _head_accuracies(
            outputs[evaluation_split], split_rows[evaluation_split], labels
        ),
        "relation_geometry": relation_geometry_margin(
            outputs[evaluation_split]["z_r"], split_rows[evaluation_split]
        ),
        "branch_information_matrix": _branch_information_matrix(
            outputs["train"],
            split_rows["train"],
            outputs[evaluation_split],
            split_rows[evaluation_split],
            ridge_strength=ridge_strength,
            template_probe_seed=template_probe_seed,
        ),
        "fusion_contributions": _fusion_contribution_metrics(
            outputs[evaluation_split]
        ),
    }
    evaluation["strict_readiness"] = _strict_c21_grade(evaluation)
    return evaluation, predictions


def _load_feature_cache(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    # Stage-C2 caches are locally generated research artifacts and contain row
    # metadata in addition to tensors, so weights_only cannot be used here.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "source_core_sha256",
        "selected_layers_one_based",
        "selected_layer_indices",
        "pool_names",
        "features",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Stage-C2 feature cache is missing {sorted(missing)}")
    if tuple(payload["pool_names"]) != POOL_NAMES:
        raise ValueError("Stage-C2 feature cache uses an unexpected pool layout")
    for split in ("train", "validation", "test"):
        if split not in payload["features"] or split not in payload["metadata"]:
            raise ValueError(f"Stage-C2 feature cache is missing split {split}")
        pools = payload["features"][split]["pools"]
        if pools.ndim != 4 or pools.size(1) != len(POOL_NAMES):
            raise ValueError(f"invalid pooled feature layout for split {split}")
        if len(pools) != len(payload["metadata"][split]):
            raise ValueError(f"feature and metadata counts differ for split {split}")
    return payload


def _raw_relation_sources(cache: dict[str, Any]) -> list[RelationSource]:
    layers = tuple(int(value) for value in cache["selected_layers_one_based"])
    hidden_size = int(cache["features"]["train"]["pools"].size(-1))
    families = (
        ("question_without_entity", (POOL_NAMES.index("question_without_entity"),)),
        ("answer_position", (POOL_NAMES.index("answer_position"),)),
        (
            "question_without_entity+answer_position",
            (
                POOL_NAMES.index("question_without_entity"),
                POOL_NAMES.index("answer_position"),
            ),
        ),
    )
    sources = []
    for family, pools in families:
        for count in range(1, len(layers) + 1):
            for offsets in itertools.combinations(range(len(layers)), count):
                selected_layers = tuple(layers[index] for index in offsets)
                suffix = "-".join(str(value) for value in selected_layers)
                sources.append(
                    RelationSource(
                        source_id=f"{family}:layers-{suffix}",
                        family=family,
                        pool_indices=tuple(pools),
                        layer_offsets=tuple(offsets),
                        layers_one_based=selected_layers,
                        input_dim=hidden_size * len(pools) * len(offsets),
                    )
                )
    return sources


def _materialize_raw_source(
    pools: Tensor, source: RelationSource
) -> Tensor:
    return torch.cat(
        [
            pools[:, pool_index, layer_offset, :]
            for pool_index in source.pool_indices
            for layer_offset in source.layer_offsets
        ],
        dim=-1,
    ).float()


def _probe_source(
    split_features: dict[str, Tensor],
    split_rows: dict[str, Sequence[dict[str, Any]]],
    *,
    ridge_strength: float,
) -> dict[str, Any]:
    train_labels = [str(row["attribute"]) for row in split_rows["train"]]
    result: dict[str, Any] = {}
    for output_name, split in (("validation", "validation"), ("development", "test")):
        result[output_name] = ridge_probe_accuracy(
            split_features["train"],
            train_labels,
            split_features[split],
            [str(row["attribute"]) for row in split_rows[split]],
            ridge_strength=ridge_strength,
        )
    return result


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        min(float(row["validation_accuracy"]), float(row["development_accuracy"])),
        float(row["development_accuracy"]),
        float(row["validation_accuracy"]),
        -float(row["input_dim"]),
        -float(row["num_layers"]),
        -float(row["num_pools"]),
    )


def run_relation_source_audit(
    feature_cache: str | Path,
    output_dir: str | Path,
    *,
    canonicalizer_checkpoint: str | Path | None = None,
    ridge_strength: float = 0.01,
) -> dict[str, Any]:
    """Audit frozen-core relation sources without updating any parameter."""

    if ridge_strength <= 0:
        raise ValueError("ridge_strength must be positive")
    feature_cache = Path(feature_cache)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_feature_cache(feature_cache)
    split_rows = {
        split: list(cache["metadata"][split])
        for split in ("train", "validation", "test")
    }
    split_pools = {
        split: cache["features"][split]["pools"].float()
        for split in split_rows
    }

    rows: list[dict[str, Any]] = []
    source_specs: dict[str, dict[str, Any]] = {}
    for source in _raw_relation_sources(cache):
        features = {
            split: _materialize_raw_source(pools, source)
            for split, pools in split_pools.items()
        }
        probes = _probe_source(
            features, split_rows, ridge_strength=ridge_strength
        )
        row = {
            "source_id": source.source_id,
            "family": source.family,
            "raw_core_source": True,
            "layers_one_based": ",".join(
                str(value) for value in source.layers_one_based
            ),
            "num_layers": len(source.layer_offsets),
            "num_pools": len(source.pool_indices),
            "input_dim": source.input_dim,
            "validation_accuracy": probes["validation"]["accuracy"],
            "development_accuracy": probes["development"]["accuracy"],
            "validation_train_accuracy": probes["validation"]["train_accuracy"],
            "development_train_accuracy": probes["development"]["train_accuracy"],
        }
        rows.append(row)
        source_specs[source.source_id] = asdict(source)

    learned_details: dict[str, Any] = {}
    if canonicalizer_checkpoint is not None:
        checkpoint = Path(canonicalizer_checkpoint)
        system, payload = load_canonicalizer_checkpoint(checkpoint)
        if payload.get("source_core_sha256") != cache["source_core_sha256"]:
            raise ValueError("canonicalizer and feature cache core hashes differ")
        if tuple(payload.get("selected_core_layers", ())) != tuple(
            cache["selected_layer_indices"]
        ):
            raise ValueError("canonicalizer and feature cache layer selections differ")
        if system.config.architecture != "factorized":
            raise ValueError("relation-source audit needs a factorized canonicalizer")
        system.eval()
        learned_features = {
            "learned_relation_input": {},
            "learned_relation_output": {},
        }
        with torch.inference_mode():
            for split, pools in split_pools.items():
                mixed = system.canonicalizer.mixed_pools(pools)
                learned_features["learned_relation_input"][split] = torch.cat(
                    [mixed[:, 1], mixed[:, 2]], dim=-1
                ).float()
                learned_features["learned_relation_output"][split] = system(
                    pools
                )["relation"].float()
        for name, features in learned_features.items():
            probes = _probe_source(
                features, split_rows, ridge_strength=ridge_strength
            )
            row = {
                "source_id": name,
                "family": "learned",
                "raw_core_source": False,
                "layers_one_based": "learned-mixture",
                "num_layers": len(cache["selected_layers_one_based"]),
                "num_pools": 2,
                "input_dim": int(features["train"].size(1)),
                "validation_accuracy": probes["validation"]["accuracy"],
                "development_accuracy": probes["development"]["accuracy"],
                "validation_train_accuracy": probes["validation"]["train_accuracy"],
                "development_train_accuracy": probes["development"]["train_accuracy"],
            }
            rows.append(row)
            learned_details[name] = probes

    raw_rows = [row for row in rows if bool(row["raw_core_source"])]
    selected = max(raw_rows, key=_selection_key)
    selected_spec = source_specs[str(selected["source_id"])]
    development_accuracy = float(selected["development_accuracy"])
    if development_accuracy >= 0.80:
        interpretation = "raw_anchor_ready"
    elif development_accuracy < 0.75:
        interpretation = "public_relation_paraphrase_supervision_required"
    else:
        interpretation = "borderline_anchor_public_relation_supervision_recommended"

    degradation = None
    if learned_details:
        before = float(
            learned_details["learned_relation_input"]["development"]["accuracy"]
        )
        after = float(
            learned_details["learned_relation_output"]["development"]["accuracy"]
        )
        degradation = {
            "learned_relation_input_accuracy": before,
            "learned_relation_output_accuracy": after,
            "output_minus_input": after - before,
        }

    csv_path = output_dir / "relation_source_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=_selection_key, reverse=True))
    summary = {
        "schema_version": STAGE_C21_SCHEMA_VERSION,
        "stage": "C2.1-relation-source-audit",
        "zero_training": True,
        "core_frozen": True,
        "confirmation_accessed": False,
        "feature_cache": str(feature_cache.resolve()),
        "feature_cache_sha256": _sha256_file(feature_cache),
        "source_core_sha256": cache["source_core_sha256"],
        "selected_layers_one_based": list(cache["selected_layers_one_based"]),
        "ridge_strength": ridge_strength,
        "selection_protocol": (
            "rank raw frozen-core sources by min(validation, development), then "
            "development and validation; current Stage-C2 test is treated only as "
            "development and sealed confirmation is not read"
        ),
        "selected_residual_anchor": selected_spec,
        "selected_metrics": selected,
        "interpretation": interpretation,
        "learned_relation_degradation": degradation,
        "rows": rows,
        "artifacts": {"csv": str(csv_path.resolve())},
    }
    summary_path = output_dir / "relation_source_audit.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def run_relation_source_audit_from_config(
    config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    audit = values.get("relation_source_audit", values)
    return run_relation_source_audit(
        audit.get(
            "feature_cache", "artifacts/stage_c2/canonicalizer_features.pt"
        ),
        output_dir,
        canonicalizer_checkpoint=audit.get(
            "canonicalizer_checkpoint",
            "artifacts/stage_c2/Q3/canonicalizer.pt",
        ),
        ridge_strength=float(audit.get("ridge_strength", 0.01)),
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _confirmation_public_record(
    *,
    templates_sha256: str,
    rows_sha256: str,
    row_count: int,
    fact_count: int,
    templates_per_relation: int,
) -> dict[str, Any]:
    return {
        "schema_version": STAGE_C21_SCHEMA_VERSION,
        "stage": "C2.1-sealed-confirmation",
        "sealed": True,
        "confirmation_accessed": False,
        "confirmation_access_count": 0,
        "templates_sha256": templates_sha256,
        "rows_sha256": rows_sha256,
        "row_count": row_count,
        "fact_count": fact_count,
        "relation_count": len(CONFIRMATION_RELATION_PHRASES),
        "templates_per_relation": templates_per_relation,
        "lexical_isolation": (
            "relation-bearing phrases are sampled once from confirmation-only "
            "families; the selected templates and seed remain under ignored data/"
        ),
        "selection_policy": (
            "validation selects checkpoints; Stage-C2 test is development-only; "
            "confirmation may be evaluated once after choosing one final variant"
        ),
    }


def _validate_existing_confirmation(destination: Path) -> dict[str, Any]:
    local_seal_path = destination / "private_seal.json"
    templates_path = destination / "confirmation_templates.json"
    rows_path = destination / "confirmation.jsonl"
    if not all(path.exists() for path in (local_seal_path, templates_path, rows_path)):
        raise FileExistsError(
            "confirmation destination is partially populated; preserve it and "
            "repair explicitly instead of silently resealing"
        )
    local = json.loads(local_seal_path.read_text(encoding="utf-8"))
    if _sha256_file(templates_path) != local.get("templates_sha256"):
        raise ValueError("sealed confirmation templates no longer match their hash")
    if _sha256_file(rows_path) != local.get("rows_sha256"):
        raise ValueError("sealed confirmation rows no longer match their hash")
    return _confirmation_public_record(
        templates_sha256=str(local["templates_sha256"]),
        rows_sha256=str(local["rows_sha256"]),
        row_count=int(local["row_count"]),
        fact_count=int(local["fact_count"]),
        templates_per_relation=int(local["templates_per_relation"]),
    )


def seal_confirmation_templates(
    source_train_jsonl: str | Path,
    destination: str | Path,
    *,
    public_record_path: str | Path | None = None,
    templates_per_relation: int = 2,
) -> dict[str, Any]:
    """Create an immutable local confirmation set and publish hashes only."""

    if not 1 <= templates_per_relation <= min(
        len(CONFIRMATION_FRAMES),
        min(len(values) for values in CONFIRMATION_RELATION_PHRASES.values()),
    ):
        raise ValueError("invalid confirmation template count")
    source_train_jsonl = Path(source_train_jsonl)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        public = _validate_existing_confirmation(destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        rows = [
            json.loads(line)
            for line in source_train_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        facts: dict[str, dict[str, Any]] = {}
        for row in rows:
            fact_id = str(row["fact_id"])
            facts.setdefault(fact_id, row)
        expected_relations = set(CONFIRMATION_RELATION_PHRASES)
        observed_relations = {str(row["attribute"]) for row in facts.values()}
        if observed_relations != expected_relations:
            raise ValueError("source facts do not match confirmation relations")

        seed_hex = secrets.token_hex(16)
        rng = random.Random(int(seed_hex, 16))
        selected_templates: dict[str, list[str]] = {}
        for relation, phrases in CONFIRMATION_RELATION_PHRASES.items():
            selected_phrases = rng.sample(list(phrases), templates_per_relation)
            selected_frames = rng.sample(
                list(CONFIRMATION_FRAMES), templates_per_relation
            )
            selected_templates[relation] = [
                frame.replace("{relation_phrase}", phrase)
                for frame, phrase in zip(selected_frames, selected_phrases)
            ]
        templates_payload = {
            "format_version": 1,
            "template_split": "confirmation",
            "templates": selected_templates,
        }
        templates_path = destination / "confirmation_templates.json"
        templates_path.write_bytes(_json_bytes(templates_payload))

        confirmation_rows = []
        for fact_id in sorted(facts):
            source = facts[fact_id]
            relation = str(source["attribute"])
            for index, template in enumerate(selected_templates[relation]):
                confirmation_rows.append(
                    {
                        **source,
                        "entity_exposure": "seen",
                        "template_split": "confirmation",
                        "template_id": f"confirmation-{index}",
                        "prompt": template.format(entity=str(source["entity"])),
                    }
                )
        rows_path = destination / "confirmation.jsonl"
        rows_bytes = b"".join(_json_bytes(row) for row in confirmation_rows)
        rows_path.write_bytes(rows_bytes)
        local_seal = {
            "format_version": 1,
            "selection_seed_hex": seed_hex,
            "source_train_sha256": _sha256_file(source_train_jsonl),
            "templates_sha256": _sha256_file(templates_path),
            "rows_sha256": _sha256_file(rows_path),
            "row_count": len(confirmation_rows),
            "fact_count": len(facts),
            "templates_per_relation": templates_per_relation,
            "confirmation_access_count": 0,
        }
        (destination / "private_seal.json").write_bytes(_json_bytes(local_seal))
        public = _confirmation_public_record(
            templates_sha256=local_seal["templates_sha256"],
            rows_sha256=local_seal["rows_sha256"],
            row_count=local_seal["row_count"],
            fact_count=local_seal["fact_count"],
            templates_per_relation=templates_per_relation,
        )

    if public_record_path is not None:
        public_record_path = Path(public_record_path)
        public_record_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _json_bytes(public)
        if public_record_path.exists() and public_record_path.read_bytes() != encoded:
            raise ValueError("public confirmation seal conflicts with local sealed data")
        public_record_path.write_bytes(encoded)
    return public


def seal_confirmation_from_config(config_path: str | Path) -> dict[str, Any]:
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    config = values.get("confirmation", {})
    return seal_confirmation_templates(
        config.get("source_train_jsonl", "data/stage_b/train.jsonl"),
        config.get("destination", "data/stage_c21_confirmation"),
        public_record_path=config.get(
            "public_record", "artifacts/stage_c21/confirmation_seal.json"
        ),
        templates_per_relation=int(config.get("templates_per_relation", 2)),
    )


def train_c21_variant(
    variant: C21Variant,
    split_features: dict[str, Tensor],
    split_rows: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    *,
    selected_layer_indices: Sequence[int],
    core_hidden_size: int,
    core_sha256: str,
    steps: int,
    relation_pretrain_fraction: float,
    batch_facts: int,
    templates_per_fact: int,
    query_dim: int,
    mlp_hidden_size: int,
    dropout: float,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    evaluation_interval: int,
    evaluation_batch_size: int,
    seed: int,
    device: torch.device,
    ridge_strength: float,
    template_probe_seed: int,
) -> dict[str, Any]:
    if variant.import_q3:
        raise ValueError("the exact R0 baseline must not be retrained")
    if steps <= 0 or evaluation_interval <= 0:
        raise ValueError("training steps and evaluation interval must be positive")
    if not 0 <= relation_pretrain_fraction < 1:
        raise ValueError("relation_pretrain_fraction must be in [0, 1)")
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = split_rows["train"]
    labels = {
        "entities": _label_map(train_rows, "entity"),
        "relations": _label_map(train_rows, "attribute"),
        "templates": _label_map(train_rows, "template_id"),
        "facts": _label_map(train_rows, "fact_id"),
    }
    config = CanonicalizerConfig(
        architecture="factorized",
        num_input_layers=len(selected_layer_indices),
        core_hidden_size=core_hidden_size,
        query_dim=query_dim,
        mlp_hidden_size=mlp_hidden_size,
        dropout=dropout,
        num_entities=len(labels["entities"]),
        num_relations=len(labels["relations"]),
        num_templates=len(labels["templates"]),
        template_adversary_source=(
            "relation" if variant.template_adversary_weight > 0 else "query"
        ),
        normalized_gated_fusion=variant.normalized_gated_fusion,
    )
    system = CanonicalizerSystem(config).to(device)
    sampler = StructuredFactBatchSampler(
        train_rows,
        batch_facts=batch_facts,
        templates_per_fact=templates_per_fact,
        seed=seed + 1,
        aligned_templates=True,
    )
    all_targets = {
        "fact": _targets(train_rows, labels["facts"], "fact_id"),
        "entity": _targets(train_rows, labels["entities"], "entity"),
        "relation": _targets(train_rows, labels["relations"], "attribute"),
        "template": _targets(train_rows, labels["templates"], "template_id"),
    }
    relation_parameter_ids = {
        id(parameter)
        for parameter in system.canonicalizer.relation_encoder.parameters()
    }
    relation_parameters = [
        parameter
        for parameter in system.parameters()
        if id(parameter) in relation_parameter_ids
    ]
    main_parameters = [
        parameter
        for parameter in system.parameters()
        if id(parameter) not in relation_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_parameters, "group_name": "main"},
            {
                "params": relation_parameters,
                "group_name": "relation_encoder",
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    pretrain_steps = (
        max(1, round(steps * relation_pretrain_fraction))
        if variant.relation_first and relation_pretrain_fraction > 0
        else 0
    )
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_state: dict[str, Tensor] | None = None
    best_step = 0
    progress = tqdm(range(steps), desc=f"stage-c21:{variant.name}")
    for step in progress:
        phase = "relation_pretrain" if step < pretrain_steps else "joint"
        schedule_scale = _lr_scale(step, steps, warmup_fraction)
        relation_lr_multiplier = (
            variant.relation_encoder_joint_lr_scale
            if variant.relation_first and phase == "joint"
            else 1.0
        )
        for group in optimizer.param_groups:
            multiplier = (
                relation_lr_multiplier
                if group["group_name"] == "relation_encoder"
                else 1.0
            )
            group["lr"] = learning_rate * schedule_scale * multiplier

        indices = sampler.sample_indices()
        index_tensor = torch.tensor(indices, dtype=torch.long)
        features = split_features["train"][index_tensor].to(device)
        targets = {
            name: values[index_tensor].to(device)
            for name, values in all_targets.items()
        }
        system.train()
        output = system(
            features,
            adversary_strength=(
                1.0 if variant.template_adversary_weight > 0 else 0.0
            ),
        )
        relation_ce = F.cross_entropy(
            output["relation_logits"], targets["relation"]
        )
        relation_supcon = supervised_contrastive_loss(
            output["relation_unit"],
            targets["relation"],
            temperature=temperature,
        )
        template_adversary = F.cross_entropy(
            output["template_logits"], targets["template"]
        )
        if phase == "joint":
            fact_supcon = supervised_contrastive_loss(
                output["query"], targets["fact"], temperature=temperature
            )
            entity_ce = F.cross_entropy(
                output["entity_logits"], targets["entity"]
            )
        else:
            fact_supcon = torch.zeros((), device=device)
            entity_ce = torch.zeros((), device=device)
        loss = (
            fact_supcon
            + variant.relation_ce_weight * relation_ce
            + variant.relation_supcon_weight * relation_supcon
            + variant.entity_ce_weight * entity_ce
            + variant.template_adversary_weight * template_adversary
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            system.parameters(), max_norm=5.0
        )
        optimizer.step()

        should_evaluate = (
            (step + 1) % evaluation_interval == 0 or step + 1 == steps
        )
        if should_evaluate:
            checkpoint_outputs = {
                split: _embed_branch_outputs(
                    system,
                    split_features[split],
                    batch_size=evaluation_batch_size,
                    device=device,
                )
                for split in ("train", "validation")
            }
            validation, _ = _evaluate_c21_split(
                checkpoint_outputs,
                split_rows,
                labels,
                "validation",
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
            )
            score = _selection_score(validation)
            row = {
                "step": step + 1,
                "phase": phase,
                "main_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "relation_encoder_learning_rate": float(
                    optimizer.param_groups[1]["lr"]
                ),
                "loss": float(loss.detach().cpu()),
                "fact_supcon_loss": float(fact_supcon.detach().cpu()),
                "relation_ce_loss": float(relation_ce.detach().cpu()),
                "relation_supcon_loss": float(relation_supcon.detach().cpu()),
                "entity_ce_loss": float(entity_ce.detach().cpu()),
                "template_adversary_loss": float(
                    template_adversary.detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "validation_gate_count": validation["strict_readiness"][
                    "passed_count"
                ],
                "validation_relation_head": validation["auxiliary_heads"][
                    "relation_accuracy"
                ],
                "validation_relation_probe": validation[
                    "branch_information_matrix"
                ]["z_r"]["relation"]["accuracy"],
                "validation_relation_template_margin": validation[
                    "relation_geometry"
                ]["relation_template_margin"],
                "validation_centroid_top1": validation["query_space"][
                    "retrieval"
                ]["centroid_top1_accuracy"],
                "validation_row_1nn": validation["query_space"]["retrieval"][
                    "row_1nn_accuracy"
                ],
                "validation_mrr": validation["query_space"]["retrieval"][
                    "centroid_mrr"
                ],
                "validation_fact_template_margin": validation["query_space"][
                    "geometry"
                ]["fact_over_template_margin"],
                "validation_fact_silhouette": validation["query_space"][
                    "silhouette"
                ]["fact_id"],
                "validation_entity_probe": validation["query_space"][
                    "linear_probes"
                ]["entity_id"]["accuracy"],
                "validation_template_probe": validation[
                    "query_template_probe"
                ]["accuracy"],
            }
            history.append(row)
            if best_score is None or score > best_score:
                best_score = score
                best_state = _clone_state(system)
                best_step = step + 1
            progress.set_postfix(
                gates=f"{validation['strict_readiness']['passed_count']}/10",
                rel=f"{validation['auxiliary_heads']['relation_accuracy']:.3f}",
                centroid=(
                    f"{validation['query_space']['retrieval']['centroid_top1_accuracy']:.3f}"
                ),
            )

    assert best_state is not None and best_score is not None
    system.load_state_dict(best_state, strict=True)
    checkpoint = save_canonicalizer_checkpoint(
        output_dir / "canonicalizer.pt",
        system,
        variant=variant.name,
        selected_core_layers=selected_layer_indices,
        labels=labels,
        source_core_sha256=core_sha256,
    )
    final_outputs = {
        split: _embed_branch_outputs(
            system,
            features,
            batch_size=evaluation_batch_size,
            device=device,
        )
        for split, features in split_features.items()
    }
    torch.save(
        {
            "format_version": 1,
            "variant": variant.name,
            "outputs": final_outputs,
        },
        output_dir / "branch_embeddings.pt",
    )
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "system": system,
        "outputs": final_outputs,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "best_step": best_step,
        "best_validation_score": list(best_score),
        "history": history,
        "canonicalizer_config": asdict(config),
        "labels": labels,
        "pretrain_steps": pretrain_steps,
        "joint_steps": steps - pretrain_steps,
    }


def _load_exact_r0(
    checkpoint: str | Path,
    cache: dict[str, Any],
    split_features: dict[str, Tensor],
    *,
    evaluation_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    system, payload = load_canonicalizer_checkpoint(checkpoint, device=device)
    if payload.get("variant") != "Q3":
        raise ValueError("R0 must import the exact Stage-C2 Q3 checkpoint")
    if payload.get("source_core_sha256") != cache["source_core_sha256"]:
        raise ValueError("R0 checkpoint and feature cache core hashes differ")
    if tuple(payload.get("selected_core_layers", ())) != tuple(
        cache["selected_layer_indices"]
    ):
        raise ValueError("R0 checkpoint and feature cache layer selections differ")
    if system.config.architecture != "factorized":
        raise ValueError("R0 Q3 checkpoint is not factorized")
    outputs = {
        split: _embed_branch_outputs(
            system,
            features,
            batch_size=evaluation_batch_size,
            device=device,
        )
        for split, features in split_features.items()
    }
    best_step = 0
    legacy_evaluation = checkpoint.parent / "evaluation.json"
    if legacy_evaluation.exists():
        best_step = int(
            json.loads(legacy_evaluation.read_text(encoding="utf-8")).get(
                "best_step", 0
            )
        )
    return {
        "system": system,
        "outputs": outputs,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "best_step": best_step,
        "canonicalizer_config": asdict(system.config),
        "labels": {
            name: list(values) for name, values in payload["labels"].items()
        },
        "pretrain_steps": 0,
        "joint_steps": 0,
        "exact_stage_c2_q3_import": True,
    }


def _c21_ablation_row(name: str, detail: dict[str, Any]) -> dict[str, Any]:
    validation = detail["evaluation"]["validation"]
    development = detail["evaluation"]["development"]
    query = development["query_space"]
    retrieval = query["retrieval"]
    fusion = development["fusion_contributions"]
    return {
        "variant": name,
        "exact_q3_baseline": bool(
            detail.get("exact_stage_c2_q3_import", False)
        ),
        "best_step": detail["best_step"],
        "validation_gate_count": validation["strict_readiness"]["passed_count"],
        "development_gate_count": development["strict_readiness"][
            "passed_count"
        ],
        "development_ready_for_confirmation": development["strict_readiness"][
            "passed"
        ],
        "development_relation_head": development["auxiliary_heads"][
            "relation_accuracy"
        ],
        "development_relation_probe": development["branch_information_matrix"][
            "z_r"
        ]["relation"]["accuracy"],
        "development_relation_template_margin": development[
            "relation_geometry"
        ]["relation_template_margin"],
        "development_centroid_top1": retrieval["centroid_top1_accuracy"],
        "development_row_1nn": retrieval["row_1nn_accuracy"],
        "development_mrr": retrieval["centroid_mrr"],
        "development_fact_silhouette": query["silhouette"]["fact_id"],
        "development_fact_template_margin": query["geometry"][
            "fact_over_template_margin"
        ],
        "development_entity_probe": query["linear_probes"]["entity_id"][
            "accuracy"
        ],
        "development_template_probe": development["query_template_probe"][
            "accuracy"
        ],
        "fusion_entity_norm": fusion["mean_l2_norm"]["entity"],
        "fusion_relation_norm": fusion["mean_l2_norm"]["relation"],
        "fusion_interaction_norm": fusion["mean_l2_norm"]["interaction"],
        "fusion_entity_scale": fusion["learned_scales"][0],
        "fusion_relation_scale": fusion["learned_scales"][1],
        "fusion_interaction_scale": fusion["learned_scales"][2],
    }


def run_stage_c21(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run R0-R4 while keeping the sealed confirmation set unread."""

    started = time.perf_counter()
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = values.get("experiment", {})
    training = values.get("training", {})
    evaluation_config = values.get("evaluation", {})
    feature_cache = Path(
        experiment.get(
            "feature_cache", "artifacts/stage_c2/canonicalizer_features.pt"
        )
    )
    r0_checkpoint = Path(
        experiment.get("r0_checkpoint", "artifacts/stage_c2/Q3/canonicalizer.pt")
    )
    public_confirmation_seal = Path(
        experiment.get(
            "public_confirmation_seal",
            "artifacts/stage_c21/confirmation_seal.json",
        )
    )
    variants = parse_c21_variants(values.get("variants"))
    cache = _load_feature_cache(feature_cache)
    split_rows = {
        split: list(cache["metadata"][split])
        for split in ("train", "validation", "test")
    }
    split_features = {
        split: cache["features"][split]["pools"].float()
        for split in split_rows
    }
    if not public_confirmation_seal.exists():
        raise FileNotFoundError("seal confirmation templates before Stage C2.1 training")
    public_seal = json.loads(
        public_confirmation_seal.read_text(encoding="utf-8")
    )
    if bool(public_seal.get("confirmation_accessed", True)):
        raise ValueError("public confirmation seal does not certify zero access")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    seed = int(training.get("seed", 0))
    evaluation_batch_size = int(evaluation_config.get("batch_size", 128))
    ridge_strength = float(evaluation_config.get("ridge_strength", 0.01))
    template_probe_seed = int(
        evaluation_config.get("template_probe_seed", 91)
    )
    selected_layer_indices = tuple(
        int(value) for value in cache["selected_layer_indices"]
    )
    core_hidden_size = int(split_features["train"].size(-1))
    details: dict[str, Any] = {}
    ablation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        if variant.import_q3:
            trained = _load_exact_r0(
                r0_checkpoint,
                cache,
                split_features,
                evaluation_batch_size=evaluation_batch_size,
                device=device,
            )
        else:
            trained = train_c21_variant(
                variant,
                split_features,
                split_rows,
                variant_dir,
                selected_layer_indices=selected_layer_indices,
                core_hidden_size=core_hidden_size,
                core_sha256=str(cache["source_core_sha256"]),
                steps=int(training.get("steps", 1500)),
                relation_pretrain_fraction=float(
                    training.get("relation_pretrain_fraction", 0.20)
                ),
                batch_facts=int(training.get("batch_facts", 8)),
                templates_per_fact=int(
                    training.get("templates_per_fact", 3)
                ),
                query_dim=int(training.get("query_dim", 128)),
                mlp_hidden_size=int(training.get("mlp_hidden_size", 256)),
                dropout=float(training.get("dropout", 0.1)),
                temperature=float(training.get("temperature", 0.07)),
                learning_rate=float(training.get("learning_rate", 1e-3)),
                weight_decay=float(training.get("weight_decay", 0.01)),
                warmup_fraction=float(training.get("warmup_fraction", 0.05)),
                evaluation_interval=int(
                    training.get("evaluation_interval", 50)
                ),
                evaluation_batch_size=evaluation_batch_size,
                seed=seed,
                device=device,
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
            )
        evaluation: dict[str, Any] = {}
        for public_name, source_split in (
            ("validation", "validation"),
            ("development", "test"),
        ):
            split_evaluation, predictions = _evaluate_c21_split(
                trained["outputs"],
                split_rows,
                trained["labels"],
                source_split,
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
            )
            evaluation[public_name] = split_evaluation
            for row in predictions:
                row.update(
                    {
                        "variant": variant.name,
                        "query_split": public_name,
                    }
                )
            prediction_rows.extend(predictions)
        validation_score = list(_selection_score(evaluation["validation"]))
        detail = {
            "variant": asdict(variant),
            "checkpoint": trained["checkpoint"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
            "canonicalizer_config": trained["canonicalizer_config"],
            "best_step": trained["best_step"],
            "pretrain_steps": trained["pretrain_steps"],
            "joint_steps": trained["joint_steps"],
            "selection_split": "validation",
            "selection_score": validation_score,
            "evaluation": evaluation,
            "exact_stage_c2_q3_import": bool(
                trained.get("exact_stage_c2_q3_import", False)
            ),
        }
        details[variant.name] = detail
        ablation_rows.append(_c21_ablation_row(variant.name, detail))
        (variant_dir / "evaluation.json").write_text(
            json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        trained["system"].to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Selection is complete before development diagnostics are consulted.
    selected_variant = max(
        (variant.name for variant in variants),
        key=lambda name: tuple(details[name]["selection_score"]),
    )
    selected_development = details[selected_variant]["evaluation"]["development"]
    development_ready = bool(
        selected_development["strict_readiness"]["passed"]
    )
    r1_development = details["R1"]["evaluation"]["development"]
    r1_relation_supervision_passed = (
        float(r1_development["auxiliary_heads"]["relation_accuracy"]) >= 0.85
        and float(
            r1_development["branch_information_matrix"]["z_r"]["relation"][
                "accuracy"
            ]
        )
        >= 0.85
    )
    if development_ready:
        recommended_next_step = (
            "evaluate_the_selected_variant_once_on_sealed_confirmation"
        )
    elif not r1_relation_supervision_passed:
        recommended_next_step = (
            "add_answer_free_public_relation_paraphrases_without_opening_confirmation"
        )
    else:
        recommended_next_step = (
            "improve_fact_geometry_on_validation_without_opening_confirmation"
        )

    with (output_dir / "stage_c21_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    with (output_dir / "retrieval_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    resolved_config = {
        "source_config": str(config_path.resolve()),
        "values": values,
        "feature_cache": str(feature_cache.resolve()),
        "feature_cache_sha256": _sha256_file(feature_cache),
        "source_core_sha256": cache["source_core_sha256"],
        "selected_layer_indices": list(selected_layer_indices),
        "development_source_split": "test",
        "confirmation_local_data_read": False,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "schema_version": STAGE_C21_SCHEMA_VERSION,
        "stage": "C2.1-relation-preserving-canonicalization",
        "status": (
            "ready_for_one_time_confirmation"
            if development_ready
            else "development_gates_failed"
        ),
        "core_frozen": True,
        "memory_attached": False,
        "private_answers_used_as_training_targets": False,
        "confirmation_accessed": False,
        "confirmation_access_count": 0,
        "c3_eligible": False,
        "c3_eligibility_reason": (
            "sealed confirmation has not been evaluated; development readiness "
            "alone cannot authorize C3"
        ),
        "development_ready_for_confirmation": development_ready,
        "selected_variant": selected_variant,
        "selected_variant_validation_score": details[selected_variant][
            "selection_score"
        ],
        "r1_relation_supervision_passed_on_development": (
            r1_relation_supervision_passed
        ),
        "recommended_next_step": recommended_next_step,
        "selection_protocol": (
            "each checkpoint and the final R0-R4 variant are selected exclusively "
            "on validation; the legacy Stage-C2 test split is relabeled development "
            "and is evaluated only for post-selection diagnosis"
        ),
        "confirmation_protocol": (
            "only the public hash seal was read; local confirmation templates and "
            "rows remain unread and may be consumed once for one selected variant"
        ),
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "feature_cache_sha256": _sha256_file(feature_cache),
        "source_core_sha256": cache["source_core_sha256"],
        "r0_checkpoint_sha256": _sha256_file(r0_checkpoint),
        "public_confirmation_seal": {
            "path": str(public_confirmation_seal.resolve()),
            "sha256": _sha256_file(public_confirmation_seal),
            "templates_sha256": public_seal.get("templates_sha256"),
            "rows_sha256": public_seal.get("rows_sha256"),
        },
        "variants": details,
        "ablation_rows": ablation_rows,
        "environment": collect_environment(device, torch.float32),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "ablation_csv": str(
                (output_dir / "stage_c21_ablation.csv").resolve()
            ),
            "retrieval_predictions": str(
                (output_dir / "retrieval_predictions.csv").resolve()
            ),
            "resolved_config": str((output_dir / "resolved_config.json").resolve()),
        },
    }
    summary_path = output_dir / "stage_c21_summary.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary
