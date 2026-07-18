from __future__ import annotations

import csv
import itertools
import json
import random
import secrets
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
