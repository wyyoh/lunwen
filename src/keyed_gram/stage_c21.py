from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from torch import Tensor

from .canonicalizer import POOL_NAMES, load_canonicalizer_checkpoint
from .phase_a import _sha256_file
from .stage_c1 import ridge_probe_accuracy


STAGE_C21_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RelationSource:
    source_id: str
    family: str
    pool_indices: tuple[int, ...]
    layer_offsets: tuple[int, ...]
    layers_one_based: tuple[int, ...]
    input_dim: int
    raw_core_source: bool = True


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
