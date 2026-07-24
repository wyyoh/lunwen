from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .stage_c23_benchmark import (
    INPUT_VIEWS,
    entity_masked,
    entity_masked_strip_suffix,
    lemma_bigrams,
    lemma_tokens,
    normalize_text,
    phrase_only,
)


SCHEMA_VERSION = 2
RELATIONS = ("registry_id", "city_code", "access_code")
SAMPLE_TYPES = ("known", "ambiguous", "unrelated")
REJECTION_TYPES = ("ambiguous", "unrelated")
PENDING_REVIEW_STATUS = "pending_independent_dual_review"


class PublicSplitV2(str, Enum):
    TRAIN = "public_train_v2"
    CALIBRATION = "public_calibration_v2"
    LOCKED_AUDIT = "public_locked_audit_v2"


ROW_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_version",
        "preprocessing_version",
        "corpus_kind",
        "example_id",
        "row_id",
        "split",
        "kind",
        "sample_type",
        "attribute",
        "relation_id",
        "family_id",
        "phrase_family",
        "family_signature",
        "normalized_token_signature",
        "relation_phrase",
        "phrase",
        "frame_id",
        "template_id",
        "frame",
        "entity_id",
        "entity",
        "fact_id",
        "prompt",
        "input_views",
        "lexical_ood",
        "hard_negative",
        "minimal_pair_id",
        "answer_free",
        "contains_private_answer",
    }
)

REVIEW_FIELDS = (
    "row_id",
    "split",
    "phrase_family",
    "phrase",
    "frame",
    "entity",
    "proposed_label",
    "sample_type",
    "reviewer_1_label",
    "reviewer_2_label",
    "reviewer_1_type",
    "reviewer_2_type",
    "adjudicated_label",
    "adjudicated_type",
    "ambiguous_flag",
    "unrelated_flag",
    "notes",
    "review_status",
)
REVIEW_STATIC_FIELDS = (
    "row_id",
    "split",
    "phrase_family",
    "phrase",
    "frame",
    "entity",
    "proposed_label",
    "sample_type",
    "ambiguous_flag",
    "unrelated_flag",
)

_BENCHMARK_FIELDS = frozenset(
    {
        "version",
        "preprocessing_version",
        "answer_free",
        "contains_private_answers",
        "review_status",
        "public_data_dir",
        "artifact_dir",
        "aggregate_manifest",
        "split_manifest_paths",
        "train_review_csv",
        "calibration_review_csv",
        "locked_review_csv",
        "legacy_c24_review_csv",
        "legacy_c24_locked_rows",
        "generate_legacy_c24_review",
        "historical_sources",
        "lemma_bigram_threshold",
        "splits",
        "definitions",
    }
)
_SPLIT_FIELDS = frozenset(
    {"entities", "frames", "known_families", "rejection_families"}
)
_ANSWER_METADATA_FIELDS = frozenset(
    {"answer_free", "contains_private_answer", "contains_private_answers"}
)
_ANSWER_BEARING_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "answer_index",
        "answer_text",
        "answer_value",
        "candidate_answer",
        "candidate_answers",
        "candidates",
        "completion",
        "completions",
        "ground_truth_value",
        "private_answer",
        "private_answers",
        "response_value",
        "target_value",
    }
)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class FamilySpec:
    phrase: str
    lexical_ood: bool
    hard_negative: bool
    minimal_pair_id: str


@dataclass(frozen=True)
class ValidatedSplitV2:
    name: PublicSplitV2
    entities: dict[str, str]
    frames: dict[str, str]
    known_families: dict[str, dict[str, FamilySpec]]
    rejection_families: dict[str, dict[str, FamilySpec]]


@dataclass(frozen=True)
class ValidatedBenchmarkV2:
    version: str
    preprocessing_version: str
    review_status: str
    lemma_bigram_threshold: float
    definitions: dict[str, list[str]]
    splits: dict[PublicSplitV2, ValidatedSplitV2]
    historical_sources: dict[str, str]


def _surface_text(value: Any) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", str(value)).strip())


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def normalized_token_signature(value: str) -> tuple[str, ...]:
    return tuple(normalize_text(value).split())


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = (strict_json_dumps(value) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def review_static_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """封存 reviewer 可编辑字段之外的完整审核任务内容。"""

    canonical = [
        {field: str(row.get(field, "")) for field in REVIEW_STATIC_FIELDS}
        for row in sorted(rows, key=lambda item: str(item.get("row_id", "")))
    ]
    return sha256_json(canonical)


def reject_sensitive_benchmark_inputs(value: Any, *, path: str = "benchmark") -> None:
    """拒绝 private-answer、confirmation 与 seal 载荷或路径。"""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            if "confirmation" in key or key == "seal" or key.endswith("_seal"):
                raise ValueError(f"prohibited confirmation/seal field at {path}.{raw_key}")
            if (
                (key in _ANSWER_BEARING_FIELDS or "private_answer" in key)
                and key not in _ANSWER_METADATA_FIELDS
            ):
                raise ValueError(f"private-answer field is prohibited at {path}.{raw_key}")
            reject_sensitive_benchmark_inputs(child, path=f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_sensitive_benchmark_inputs(child, path=f"{path}[{index}]")
        return
    if isinstance(value, (str, Path)):
        lowered = str(value).replace("\\", "/").casefold()
        if "confirmation" in lowered or re.search(
            r"(?:^|[/_.-])seal(?:[/_.-]|$)", lowered
        ):
            raise ValueError(f"prohibited confirmation/seal reference at {path}")


def _named_values(raw: Any, *, prefix: str, field_name: str) -> dict[str, str]:
    if isinstance(raw, Mapping):
        values = {_surface_text(key): _surface_text(value) for key, value in raw.items()}
    elif isinstance(raw, list):
        values = {
            f"{prefix}-{index}": _surface_text(value)
            for index, value in enumerate(raw)
        }
    else:
        raise ValueError(f"{field_name} must be a mapping or list")
    if not values or any(not key or not value for key, value in values.items()):
        raise ValueError(f"{field_name} requires non-empty IDs and values")
    normalized = [normalize_text(item) for item in values.values()]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} values must be unique after normalization")
    return values


def _validate_frames(raw: Any, *, prefix: str, field_name: str) -> dict[str, str]:
    frames = _named_values(raw, prefix=prefix, field_name=field_name)
    for frame in frames.values():
        if frame.count("{entity}") != 1 or frame.count("{relation_phrase}") != 1:
            raise ValueError(
                f"every {field_name} frame needs one entity and one relation phrase"
            )
    return frames


def _family_spec(raw: Any, *, split: PublicSplitV2, field_name: str) -> FamilySpec:
    if isinstance(raw, str):
        phrase = _surface_text(raw)
        lexical_ood = split is not PublicSplitV2.TRAIN
        hard_negative = False
        minimal_pair_id = ""
    elif isinstance(raw, Mapping):
        unknown = set(raw) - {
            "phrase",
            "lexical_ood",
            "hard_negative",
            "minimal_pair_id",
        }
        if unknown:
            raise ValueError(f"unknown {field_name} metadata: {sorted(unknown)}")
        phrase = _surface_text(raw.get("phrase", ""))
        lexical_ood = raw.get("lexical_ood", split is not PublicSplitV2.TRAIN)
        hard_negative = raw.get("hard_negative", False)
        minimal_pair_id = _surface_text(raw.get("minimal_pair_id", ""))
        if not isinstance(lexical_ood, bool) or not isinstance(hard_negative, bool):
            raise ValueError(f"{field_name} flags must be boolean")
    else:
        raise ValueError(f"{field_name} must be a phrase or metadata mapping")
    if not phrase:
        raise ValueError(f"{field_name}.phrase cannot be empty")
    if split is PublicSplitV2.TRAIN and lexical_ood:
        raise ValueError("public_train_v2 rows cannot be marked lexical OOD")
    if split is not PublicSplitV2.TRAIN and not lexical_ood:
        raise ValueError("calibration/locked families must be marked lexical OOD")
    return FamilySpec(
        phrase=phrase,
        lexical_ood=lexical_ood,
        hard_negative=hard_negative,
        minimal_pair_id=minimal_pair_id,
    )


def _family_groups(
    raw: Any,
    *,
    groups: Sequence[str],
    split: PublicSplitV2,
    field_name: str,
    required: bool,
) -> dict[str, dict[str, FamilySpec]]:
    if not required and raw is None:
        return {}
    if not isinstance(raw, Mapping) or set(raw) != set(groups):
        raise ValueError(f"{field_name} groups must be exactly {list(groups)}")
    output: dict[str, dict[str, FamilySpec]] = {}
    counts: set[int] = set()
    for group in groups:
        if not isinstance(raw[group], Mapping) or not raw[group]:
            raise ValueError(f"{field_name}.{group} must be a non-empty mapping")
        families = {
            _surface_text(family_id): _family_spec(
                spec,
                split=split,
                field_name=f"{field_name}.{group}.{family_id}",
            )
            for family_id, spec in raw[group].items()
        }
        if len(families) != len(raw[group]):
            raise ValueError(f"{field_name}.{group} family IDs are not unique")
        signatures = [normalize_text(spec.phrase) for spec in families.values()]
        if len(signatures) != len(set(signatures)):
            raise ValueError(f"{field_name}.{group} phrases must be unique")
        output[group] = families
        counts.add(len(families))
    if len(counts) != 1:
        raise ValueError(f"{field_name} must be balanced across groups")
    return output


def _validate_definitions(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping) or set(raw) != set(RELATIONS):
        raise ValueError("benchmark.definitions must contain exactly three relations")
    definitions: dict[str, list[str]] = {}
    for relation in RELATIONS:
        values = raw[relation]
        if not isinstance(values, list) or len(values) < 3:
            raise ValueError(f"definitions.{relation} needs at least three prototypes")
        cleaned = [_surface_text(value) for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError(f"definitions.{relation} is malformed")
        definitions[relation] = cleaned
    return definitions


def _flatten_specs(
    groups: Mapping[str, Mapping[str, FamilySpec]],
) -> dict[str, FamilySpec]:
    return {
        family_id: spec
        for group in groups.values()
        for family_id, spec in group.items()
    }


def _validate_calibration_design(split: ValidatedSplitV2) -> None:
    hard_by_relation = {
        relation: sum(spec.hard_negative for spec in split.known_families[relation].values())
        for relation in RELATIONS
    }
    if min(hard_by_relation.values()) < 2:
        raise ValueError("public_calibration_v2 needs at least two hard negatives per relation")
    pair_labels: dict[str, set[str]] = {}
    for relation, families in split.known_families.items():
        for spec in families.values():
            if spec.minimal_pair_id:
                pair_labels.setdefault(spec.minimal_pair_id, set()).add(relation)
    if not any(labels == set(RELATIONS) for labels in pair_labels.values()):
        raise ValueError(
            "public_calibration_v2 needs a minimal-pair group spanning all relations"
        )
    if not all(
        spec.lexical_ood
        for spec in _flatten_specs(split.known_families).values()
    ):
        raise ValueError("all calibration known families must be lexical OOD")


def validate_benchmark_config_v2(config: Mapping[str, Any]) -> ValidatedBenchmarkV2:
    if not isinstance(config, Mapping):
        raise ValueError("benchmark config must be a mapping")
    reject_sensitive_benchmark_inputs(config)
    unknown = set(config) - _BENCHMARK_FIELDS
    if unknown:
        raise ValueError(f"unknown benchmark fields: {sorted(map(str, unknown))}")
    if config.get("answer_free") is not True:
        raise ValueError("C2.4b benchmark must be answer-free")
    if config.get("contains_private_answers") is not False:
        raise ValueError("C2.4b benchmark cannot contain private answers")
    version = _surface_text(config.get("version", ""))
    preprocessing_version = _surface_text(config.get("preprocessing_version", ""))
    if not version or not preprocessing_version:
        raise ValueError("benchmark version and preprocessing version are required")
    review_status = _surface_text(config.get("review_status", ""))
    if review_status != PENDING_REVIEW_STATUS:
        raise ValueError("v2 benchmark must start pending independent dual review")
    threshold = float(config.get("lemma_bigram_threshold", 0.80))
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("lemma_bigram_threshold must be finite and in (0, 1]")
    raw_sources = config.get("historical_sources")
    required_sources = {"stage_c23_config", "stage_c24_config"}
    allowed_sources = {*required_sources, "stage_c24b_v2_config"}
    if (
        not isinstance(raw_sources, Mapping)
        or not required_sources.issubset(raw_sources)
        or not set(raw_sources).issubset(allowed_sources)
    ):
        raise ValueError(
            "historical_sources must name stage_c23_config/stage_c24_config "
            "and may additionally seal stage_c24b_v2_config"
        )
    sources = {str(key): _surface_text(value) for key, value in raw_sources.items()}
    raw_splits = config.get("splits")
    required_splits = {split.value for split in PublicSplitV2}
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != required_splits:
        raise ValueError(f"benchmark.splits must be exactly {sorted(required_splits)}")

    splits: dict[PublicSplitV2, ValidatedSplitV2] = {}
    for split in PublicSplitV2:
        raw_split = raw_splits[split.value]
        if not isinstance(raw_split, Mapping):
            raise ValueError(f"{split.value} must be a mapping")
        unknown_split = set(raw_split) - _SPLIT_FIELDS
        if unknown_split:
            raise ValueError(f"unknown {split.value} fields: {sorted(unknown_split)}")
        if split is PublicSplitV2.TRAIN and "rejection_families" in raw_split:
            raise ValueError("public_train_v2 must contain known samples only")
        values = ValidatedSplitV2(
            name=split,
            entities=_named_values(
                raw_split.get("entities"),
                prefix=f"{split.value}-entity",
                field_name=f"{split.value}.entities",
            ),
            frames=_validate_frames(
                raw_split.get("frames"),
                prefix=f"{split.value}-frame",
                field_name=f"{split.value}.frames",
            ),
            known_families=_family_groups(
                raw_split.get("known_families"),
                groups=RELATIONS,
                split=split,
                field_name=f"{split.value}.known_families",
                required=True,
            ),
            rejection_families=_family_groups(
                raw_split.get("rejection_families"),
                groups=REJECTION_TYPES,
                split=split,
                field_name=f"{split.value}.rejection_families",
                required=split is not PublicSplitV2.TRAIN,
            ),
        )
        splits[split] = values

    train_count = len(splits[PublicSplitV2.TRAIN].known_families[RELATIONS[0]])
    if not 8 <= train_count <= 12:
        raise ValueError("public_train_v2 requires 8-12 families per relation")
    _validate_calibration_design(splits[PublicSplitV2.CALIBRATION])
    benchmark = ValidatedBenchmarkV2(
        version=version,
        preprocessing_version=preprocessing_version,
        review_status=review_status,
        lemma_bigram_threshold=threshold,
        definitions=_validate_definitions(config.get("definitions")),
        splits=splits,
        historical_sources=sources,
    )
    audit_split_isolation_v2(benchmark)
    return benchmark


def _token_sequence_contained(left: str, right: str, *, minimum_tokens: int = 2) -> bool:
    shorter = normalized_token_signature(left)
    longer = normalized_token_signature(right)
    if len(shorter) < minimum_tokens or len(shorter) >= len(longer):
        return False
    return any(
        longer[index : index + len(shorter)] == shorter
        for index in range(len(longer) - len(shorter) + 1)
    )


def audit_phrase_collections_v2(
    left: Mapping[str, str],
    right: Mapping[str, str],
    *,
    left_name: str,
    right_name: str,
    lemma_bigram_threshold: float,
) -> dict[str, Any]:
    """逐层审计文本碰撞；共享单个词仍然合法。

    五层记录彼此独立，因而同一文本对可以同时触发多个更强/更弱的碰撞
    条件。这样 manifest 不会用一个较早的 ``continue`` 隐藏后续审计层。
    """

    collisions: dict[str, list[dict[str, Any]]] = {
        "exact_text": [],
        "normalized_text": [],
        "substring_containment": [],
        "token_signature": [],
        "lemma_bigram": [],
    }
    for left_id, left_phrase in left.items():
        left_exact = str(left_phrase)
        left_normalized = normalize_text(left_phrase)
        left_tokens = normalized_token_signature(left_phrase)
        left_bigrams = lemma_bigrams(left_phrase)
        for right_id, right_phrase in right.items():
            record = {"left_id": left_id, "right_id": right_id}
            right_exact = str(right_phrase)
            right_normalized = normalize_text(right_phrase)
            right_tokens = normalized_token_signature(right_phrase)
            if left_exact == right_exact:
                collisions["exact_text"].append(record)
            if left_normalized and left_normalized == right_normalized:
                collisions["normalized_text"].append(record)
            if _token_sequence_contained(left_phrase, right_phrase) or _token_sequence_contained(
                right_phrase, left_phrase
            ):
                collisions["substring_containment"].append(record)
            if left_tokens and frozenset(left_tokens) == frozenset(right_tokens):
                collisions["token_signature"].append(record)
            right_bigrams = lemma_bigrams(right_phrase)
            if left_bigrams and right_bigrams:
                overlap = len(left_bigrams & right_bigrams) / min(
                    len(left_bigrams), len(right_bigrams)
                )
                if overlap >= lemma_bigram_threshold:
                    collisions["lemma_bigram"].append(
                        {**record, "overlap_coefficient": overlap}
                    )
    count = sum(len(items) for items in collisions.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "left": left_name,
        "right": right_name,
        "left_phrase_count": len(left),
        "right_phrase_count": len(right),
        "minimum_containment_tokens": 2,
        "lemma_bigram_threshold": lemma_bigram_threshold,
        "collisions": collisions,
        "collision_count": count,
        "passed": count == 0,
    }


def _identity_audit(
    left: Mapping[str, str], right: Mapping[str, str], *, unit: str
) -> dict[str, Any]:
    # 历史收集键带 source/path 前缀以免覆盖；审计 ID 时恢复末尾的原始 ID。
    # v2 ID 本身不含冒号，因此同一逻辑也适用于 split-to-split 审计。
    left_ids: dict[str, list[str]] = {}
    right_ids: dict[str, list[str]] = {}
    for key in left:
        left_ids.setdefault(str(key).rsplit(":", 1)[-1], []).append(str(key))
    for key in right:
        right_ids.setdefault(str(key).rsplit(":", 1)[-1], []).append(str(key))
    id_collisions = [
        {
            "id": value,
            "left_ids": left_ids[value],
            "right_ids": right_ids[value],
        }
        for value in sorted(set(left_ids) & set(right_ids))
    ]
    left_values: dict[str, list[str]] = {}
    right_values: dict[str, list[str]] = {}
    for key, value in left.items():
        left_values.setdefault(normalize_text(value), []).append(key)
    for key, value in right.items():
        right_values.setdefault(normalize_text(value), []).append(key)
    value_collisions = [
        {
            "normalized_value": value,
            "left_ids": left_values[value],
            "right_ids": right_values[value],
        }
        for value in sorted(set(left_values) & set(right_values))
    ]
    count = len(id_collisions) + len(value_collisions)
    return {
        "unit": unit,
        "id_collisions": id_collisions,
        "normalized_value_collisions": value_collisions,
        "collision_count": count,
        "passed": count == 0,
    }


def _split_phrases(split: ValidatedSplitV2) -> dict[str, str]:
    return {
        **{
            family_id: spec.phrase
            for family_id, spec in _flatten_specs(split.known_families).items()
        },
        **{
            family_id: spec.phrase
            for family_id, spec in _flatten_specs(split.rejection_families).items()
        },
    }


def audit_split_isolation_v2(benchmark: ValidatedBenchmarkV2) -> dict[str, Any]:
    pairs = (
        (PublicSplitV2.TRAIN, PublicSplitV2.CALIBRATION),
        (PublicSplitV2.TRAIN, PublicSplitV2.LOCKED_AUDIT),
        (PublicSplitV2.CALIBRATION, PublicSplitV2.LOCKED_AUDIT),
    )
    pair_results: dict[str, Any] = {}
    failures: list[str] = []
    for left_name, right_name in pairs:
        left = benchmark.splits[left_name]
        right = benchmark.splits[right_name]
        left_phrases = _split_phrases(left)
        right_phrases = _split_phrases(right)
        family_ids = sorted(set(left_phrases) & set(right_phrases))
        audit = {
            "phrases": audit_phrase_collections_v2(
                left_phrases,
                right_phrases,
                left_name=left_name.value,
                right_name=right_name.value,
                lemma_bigram_threshold=benchmark.lemma_bigram_threshold,
            ),
            "phrase_family_id": {
                "collisions": family_ids,
                "collision_count": len(family_ids),
                "passed": not family_ids,
            },
            "entity": _identity_audit(left.entities, right.entities, unit="entity"),
            "frame": _identity_audit(left.frames, right.frames, unit="frame"),
        }
        audit["passed"] = all(item["passed"] for item in audit.values())
        key = f"{left_name.value}__vs__{right_name.value}"
        pair_results[key] = audit
        if not audit["passed"]:
            failures.append(key)
    result = {
        "schema_version": SCHEMA_VERSION,
        "normalization": "NFKC+casefold+punctuation/hyphen-to-space",
        "shared_single_tokens_allowed": True,
        "minimum_containment_tokens": 2,
        "pairs": pair_results,
        "passed": not failures,
    }
    if failures:
        raise ValueError("v2 split isolation collision detected: " + ", ".join(failures))
    return result


def _collect_family_phrases(raw: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
    phrases: dict[str, str] = {}

    def visit(value: Any, path: str) -> None:
        if not isinstance(value, Mapping):
            return
        for section in (
            "train_families",
            "validation_families",
            "known_families",
            "rejection_families",
        ):
            groups = value.get(section)
            if isinstance(groups, Mapping):
                for group, families in groups.items():
                    if not isinstance(families, Mapping):
                        continue
                    for family_id, family_value in families.items():
                        if isinstance(family_value, Mapping):
                            phrase_value = family_value.get("phrase")
                        else:
                            phrase_value = family_value
                        if phrase_value is not None:
                            phrases[f"{prefix}:{path}:{section}:{group}:{family_id}"] = str(
                                phrase_value
                            )
        private_views = value.get("private_phrase_views")
        if isinstance(private_views, Mapping):
            for split, relations in private_views.items():
                if not isinstance(relations, Mapping):
                    continue
                for relation, values in relations.items():
                    if isinstance(values, list):
                        for index, phrase in enumerate(values):
                            phrases[
                                f"{prefix}:{path}:private:{split}:{relation}:{index}"
                            ] = str(phrase)
        for key, child in value.items():
            if key not in {
                "train_families",
                "validation_families",
                "known_families",
                "rejection_families",
                "private_phrase_views",
            } and isinstance(child, Mapping):
                visit(child, f"{path}.{key}")

    visit(raw, "root")
    return phrases


def _collect_named_history_values(
    raw: Mapping[str, Any], *, key_name: str, prefix: str
) -> dict[str, str]:
    output: dict[str, str] = {}

    def visit(value: Any, path: str) -> None:
        if not isinstance(value, Mapping):
            return
        matching_names = [
            str(key)
            for key in value
            if str(key) == key_name or str(key).endswith(f"_{key_name}")
        ]
        for matching_name in matching_names:
            candidate = value[matching_name]
            if isinstance(candidate, Mapping):
                for key, item in candidate.items():
                    output[
                        f"{prefix}:{path}:{matching_name}:{key}"
                    ] = str(item)
            elif isinstance(candidate, list):
                for index, item in enumerate(candidate):
                    output[
                        f"{prefix}:{path}:{matching_name}:{index}"
                    ] = str(item)
        for key, child in value.items():
            if isinstance(child, Mapping):
                visit(child, f"{path}.{key}")

    visit(raw, "root")
    return output


def _resolve_source_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # 历史来源只能相对当前 protocol 的 repo root/config 目录解析；不得让
    # 调用者 CWD 中的同名路径 shadow 已冻结来源。
    candidates = [config_path.parent.parent / path, config_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def audit_historical_collisions(
    benchmark: ValidatedBenchmarkV2,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    history_phrases: dict[str, str] = {}
    history_entities: dict[str, str] = {}
    history_frames: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for source_name, raw_path in benchmark.historical_sources.items():
        path = _resolve_source_path(config_path, raw_path)
        if not path.is_file():
            raise ValueError(f"historical source does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"historical source is malformed: {path}")
        history_phrases.update(_collect_family_phrases(raw, prefix=source_name))
        history_entities.update(
            _collect_named_history_values(raw, key_name="entities", prefix=source_name)
        )
        history_frames.update(
            _collect_named_history_values(raw, key_name="frames", prefix=source_name)
        )
        source_hashes[source_name] = sha256_file(path)

    history_label = "+".join(source_hashes)
    split_results: dict[str, Any] = {}
    failures: list[str] = []
    for split_name, split in benchmark.splits.items():
        phrases = _split_phrases(split)
        phrase_audit = audit_phrase_collections_v2(
            history_phrases,
            phrases,
            left_name=f"{history_label} historical families",
            right_name=split_name.value,
            lemma_bigram_threshold=benchmark.lemma_bigram_threshold,
        )
        family_id_collisions = sorted(
            {
                key.rsplit(":", 1)[-1]
                for key in history_phrases
            }
            & set(phrases)
        )
        audit = {
            "phrases": phrase_audit,
            "phrase_family_id": {
                "collisions": family_id_collisions,
                "collision_count": len(family_id_collisions),
                "passed": not family_id_collisions,
            },
            "entity": _identity_audit(history_entities, split.entities, unit="entity"),
            "frame": _identity_audit(history_frames, split.frames, unit="frame"),
        }
        audit["passed"] = all(item["passed"] for item in audit.values())
        split_results[split_name.value] = audit
        if not audit["passed"]:
            failures.append(split_name.value)
    result = {
        "schema_version": SCHEMA_VERSION,
        "historical_sources": source_hashes,
        "historical_phrase_count": len(history_phrases),
        "historical_entity_count": len(history_entities),
        "historical_frame_count": len(history_frames),
        "splits": split_results,
        "passed": not failures,
    }
    if failures:
        raise ValueError(
            "historical C2.3/C2.4 collision detected "
            "(including any additional sealed sources): "
            + ", ".join(failures)
        )
    return result


def _make_row(
    *,
    benchmark: ValidatedBenchmarkV2,
    split: PublicSplitV2,
    sample_type: str,
    relation: str | None,
    family_id: str,
    spec: FamilySpec,
    frame_id: str,
    frame: str,
    entity_id: str,
    entity: str,
) -> dict[str, Any]:
    prompt = frame.format(entity=entity, relation_phrase=spec.phrase)
    example_id = f"{split.value}:{relation or sample_type}:{family_id}:{frame_id}:{entity_id}"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": benchmark.version,
        "preprocessing_version": benchmark.preprocessing_version,
        "corpus_kind": "answer_free_public_selective_relation_router_benchmark",
        "example_id": example_id,
        "row_id": example_id,
        "split": split.value,
        "kind": sample_type,
        "sample_type": sample_type,
        "attribute": relation,
        "relation_id": relation,
        "family_id": family_id,
        "phrase_family": family_id,
        "family_signature": " ".join(lemma_tokens(spec.phrase)),
        "normalized_token_signature": " ".join(normalized_token_signature(spec.phrase)),
        "relation_phrase": spec.phrase,
        "phrase": spec.phrase,
        "frame_id": frame_id,
        "template_id": frame_id,
        "frame": frame,
        "entity_id": entity_id,
        "entity": entity,
        "fact_id": f"public-v2:{entity_id}|{relation or family_id}",
        "prompt": prompt,
        "input_views": {
            "phrase_only": phrase_only(spec.phrase),
            "entity_masked": entity_masked(prompt, entity),
            "entity_masked_strip_suffix": entity_masked_strip_suffix(prompt, entity),
        },
        "lexical_ood": spec.lexical_ood,
        "hard_negative": spec.hard_negative,
        "minimal_pair_id": spec.minimal_pair_id,
        "answer_free": True,
        "contains_private_answer": False,
    }
    if set(row) != ROW_FIELDS:
        raise RuntimeError("internal C2.4b row schema mismatch")
    return row


def _materialize_split(
    benchmark: ValidatedBenchmarkV2, split: ValidatedSplitV2
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = list(split.entities.items())
    frames = list(split.frames.items())
    offset = 0
    for relation in RELATIONS:
        for family_index, (family_id, spec) in enumerate(
            split.known_families[relation].items()
        ):
            for frame_index, (frame_id, frame) in enumerate(frames):
                entity_id, entity = entities[(offset + family_index + frame_index) % len(entities)]
                rows.append(
                    _make_row(
                        benchmark=benchmark,
                        split=split.name,
                        sample_type="known",
                        relation=relation,
                        family_id=family_id,
                        spec=spec,
                        frame_id=frame_id,
                        frame=frame,
                        entity_id=entity_id,
                        entity=entity,
                    )
                )
        offset += len(split.known_families[relation])
    for sample_type in REJECTION_TYPES:
        for family_index, (family_id, spec) in enumerate(
            split.rejection_families.get(sample_type, {}).items()
        ):
            for frame_index, (frame_id, frame) in enumerate(frames):
                entity_id, entity = entities[(offset + family_index + frame_index) % len(entities)]
                rows.append(
                    _make_row(
                        benchmark=benchmark,
                        split=split.name,
                        sample_type=sample_type,
                        relation=None,
                        family_id=family_id,
                        spec=spec,
                        frame_id=frame_id,
                        frame=frame,
                        entity_id=entity_id,
                        entity=entity,
                    )
                )
        offset += len(split.rejection_families.get(sample_type, {}))
    return rows


def _validate_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]], benchmark: ValidatedBenchmarkV2
) -> None:
    if set(rows) != {split.value for split in PublicSplitV2}:
        raise ValueError("v2 rows require exactly train/calibration/locked splits")
    row_ids: set[str] = set()
    for split_name in PublicSplitV2:
        split_rows = rows[split_name.value]
        expected = benchmark.splits[split_name]
        family_count = len(_flatten_specs(expected.known_families)) + len(
            _flatten_specs(expected.rejection_families)
        )
        if len(split_rows) != family_count * len(expected.frames):
            raise ValueError(f"unexpected {split_name.value} row count")
        for row in split_rows:
            reject_sensitive_benchmark_inputs(row)
            if set(row) != ROW_FIELDS or row["split"] != split_name.value:
                raise ValueError("v2 row violates strict schema")
            if row["answer_free"] is not True or row["contains_private_answer"] is not False:
                raise ValueError("v2 row is not answer-free")
            if row["row_id"] in row_ids:
                raise ValueError("v2 row IDs must be globally unique")
            row_ids.add(str(row["row_id"]))
            if set(row["input_views"]) != set(INPUT_VIEWS):
                raise ValueError("v2 row input views are incomplete")
            if row["sample_type"] == "known" and row["relation_id"] not in RELATIONS:
                raise ValueError("known row has invalid relation")
            if row["sample_type"] in REJECTION_TYPES and row["relation_id"] is not None:
                raise ValueError("reject row must not carry a relation")
            if split_name is PublicSplitV2.TRAIN and row["sample_type"] != "known":
                raise ValueError("public_train_v2 contains a reject row")


def build_public_benchmark_v2(
    benchmark_config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    run_historical_audit: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    benchmark = validate_benchmark_config_v2(benchmark_config)
    rows = {
        split.value: _materialize_split(benchmark, benchmark.splits[split])
        for split in PublicSplitV2
    }
    _validate_rows(rows, benchmark)
    split_audit = audit_split_isolation_v2(benchmark)
    historical_audit = None
    if run_historical_audit:
        if config_path is None:
            raise ValueError("config_path is required for the historical collision audit")
        historical_audit = audit_historical_collisions(benchmark, config_path=config_path)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "strict_schema": True,
        "answer_free": True,
        "contains_private_answers": False,
        "public_benchmark_human_reviewed": False,
        "formal_locked_audit_executed": False,
        "split_isolation": split_audit,
        "historical_collision_audit": historical_audit,
        "passed": split_audit["passed"]
        and (historical_audit is None or historical_audit["passed"]),
    }
    return rows, audit


def build_public_split_v2(
    benchmark_config: Mapping[str, Any],
    split: PublicSplitV2 | str,
) -> list[dict[str, Any]]:
    """单独物化一个已验证 split，供 freeze 前后严格分阶段执行。

    该入口不会顺带构造另外两个 split；尤其可用于证明 locked rows 直到
    router 完成 calibration 与冻结之后才首次 materialize。
    """

    benchmark = validate_benchmark_config_v2(benchmark_config)
    try:
        split_id = split if isinstance(split, PublicSplitV2) else PublicSplitV2(str(split))
    except ValueError as exc:
        raise ValueError(f"unknown C2.4b split {split!r}") from exc
    rows = _materialize_split(benchmark, benchmark.splits[split_id])
    expected = benchmark.splits[split_id]
    family_count = len(_flatten_specs(expected.known_families)) + len(
        _flatten_specs(expected.rejection_families)
    )
    if len(rows) != family_count * len(expected.frames):
        raise ValueError(f"unexpected {split_id.value} row count")
    if len({str(row["row_id"]) for row in rows}) != len(rows):
        raise ValueError(f"duplicate row ID inside {split_id.value}")
    for row in rows:
        reject_sensitive_benchmark_inputs(row)
        if set(row) != ROW_FIELDS or row["split"] != split_id.value:
            raise ValueError(f"{split_id.value} row violates strict schema")
        if row["answer_free"] is not True or row["contains_private_answer"] is not False:
            raise ValueError(f"{split_id.value} row is not answer-free")
    return rows


def build_review_rows(
    rows: Sequence[Mapping[str, Any]], *, allowed_splits: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    allowed = set(
        allowed_splits
        if allowed_splits is not None
        else (
            PublicSplitV2.TRAIN.value,
            PublicSplitV2.CALIBRATION.value,
            PublicSplitV2.LOCKED_AUDIT.value,
            "c24_public_locked_audit_legacy_review",
        )
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        split = str(row.get("split", ""))
        if split not in allowed:
            raise ValueError(f"review rows cannot be created for split {split!r}")
        sample_type = str(row.get("sample_type") or row.get("kind") or "")
        if sample_type not in SAMPLE_TYPES:
            raise ValueError("review row has an invalid sample_type")
        proposed_label = row.get("relation_id") or row.get("attribute") or sample_type
        review = {
            "row_id": str(row.get("row_id") or row.get("example_id")),
            "split": split,
            "phrase_family": str(row.get("phrase_family") or row.get("family_id")),
            "phrase": str(row.get("phrase") or row.get("relation_phrase")),
            "frame": str(row.get("prompt") or row.get("frame")),
            "entity": str(row.get("entity", "")),
            "proposed_label": str(proposed_label),
            "sample_type": sample_type,
            "reviewer_1_label": "",
            "reviewer_2_label": "",
            "reviewer_1_type": "",
            "reviewer_2_type": "",
            "adjudicated_label": "",
            "adjudicated_type": "",
            "ambiguous_flag": "true" if sample_type == "ambiguous" else "false",
            "unrelated_flag": "true" if sample_type == "unrelated" else "false",
            "notes": "",
            "review_status": "pending",
        }
        if tuple(review) != REVIEW_FIELDS:
            raise RuntimeError("internal C2.4b review schema mismatch")
        output.append(review)
    if len({row["row_id"] for row in output}) != len(output):
        raise ValueError("review row IDs must be unique")
    return output


def build_legacy_c24_review_rows(
    legacy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    converted = []
    for raw in legacy_rows:
        if raw.get("split") != "public_locked_audit":
            raise ValueError("legacy review input must be the C2.4 public_locked_audit")
        row = dict(raw)
        row["split"] = "c24_public_locked_audit_legacy_review"
        row["row_id"] = f"legacy-c24:{raw.get('example_id')}"
        row["sample_type"] = raw.get("kind")
        row["relation_id"] = raw.get("attribute")
        row["phrase_family"] = raw.get("family_id")
        row["phrase"] = raw.get("relation_phrase")
        converted.append(row)
    return build_review_rows(
        converted, allowed_splits=("c24_public_locked_audit_legacy_review",)
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json_dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(strict_json_dumps(row) + "\n")


def _write_review_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            reject_sensitive_benchmark_inputs(row)
            rows.append(row)
    return rows


def _read_legacy_c24_source(
    path: Path,
    *,
    benchmark: ValidatedBenchmarkV2,
    config_path: Path,
) -> list[dict[str, Any]]:
    if path.suffix.casefold() != ".csv":
        return _read_jsonl(path)
    c24_config_path = _resolve_source_path(
        config_path, benchmark.historical_sources["stage_c24_config"]
    )
    c24_raw = yaml.safe_load(c24_config_path.read_text(encoding="utf-8")) or {}
    raw_entities = (
        c24_raw.get("benchmark", {})
        .get("splits", {})
        .get("public_locked_audit", {})
        .get("entities", [])
    )
    if isinstance(raw_entities, Mapping):
        entity_values = [str(value) for value in raw_entities.values()]
    elif isinstance(raw_entities, list):
        entity_values = [str(value) for value in raw_entities]
    else:
        raise ValueError("C2.4 locked entity source is malformed")
    converted: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for review in csv.DictReader(handle):
            proposed = str(review.get("proposed_label", ""))
            sample_type = proposed if proposed in REJECTION_TYPES else "known"
            prompt = str(review.get("frame", ""))
            matching_entities = [entity for entity in entity_values if entity in prompt]
            if len(matching_entities) != 1:
                raise ValueError("cannot recover one C2.4 entity from a legacy review row")
            converted.append(
                {
                    "example_id": str(review.get("row_id", "")),
                    "split": "public_locked_audit",
                    "kind": sample_type,
                    "attribute": proposed if sample_type == "known" else None,
                    "family_id": str(review.get("phrase_family", "")),
                    "relation_phrase": str(review.get("phrase", "")),
                    "prompt": prompt,
                    "entity": matching_entities[0],
                }
            )
    return converted


def _verify_legacy_c24_review_source(
    path: Path, *, config_path: Path
) -> dict[str, Any]:
    """把旧 120 条审核源绑定到 C2.4 已提交的 benchmark manifest。"""

    manifest_path = path.parent / "public_benchmark_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("C2.4 public benchmark manifest is absent for legacy review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("C2.4 public benchmark manifest is malformed")
    sealed = manifest.get("human_review_file")
    if not isinstance(sealed, Mapping):
        raise ValueError("C2.4 manifest lacks its human-review source seal")
    sealed_path = _resolve_output(config_path, str(sealed.get("path", "")))
    if sealed_path != path.resolve():
        raise ValueError("configured legacy C2.4 review source differs from its seal")
    observed = sha256_file(path)
    expected = str(sealed.get("sha256", ""))
    if observed != expected:
        raise ValueError("legacy C2.4 review source SHA-256 differs from its seal")
    return {
        "source_path": str(path.resolve()),
        "source_sha256": observed,
        "c24_manifest_path": str(manifest_path.resolve()),
        "c24_manifest_sha256": sha256_file(manifest_path),
        "c24_manifest_review_sha256": expected,
        "source_matches_c24_manifest": True,
    }


def _git_provenance(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def _resolve_output(config_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_candidate = config_path.parent.parent / path
    return repo_candidate.resolve()


def prepare_public_benchmark_v2(
    config_path: str | Path,
    *,
    public_data_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """物化 v2 数据与空白审核模板；绝不运行正式 locked audit。"""

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark_config = raw.get("benchmark")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("stage C2.4b config is missing benchmark")
    rows, audit = build_public_benchmark_v2(
        benchmark_config, config_path=config_path, run_historical_audit=True
    )
    model_provenance = raw.get("models", {})
    if not isinstance(model_provenance, Mapping):
        raise ValueError("stage C2.4b model provenance is malformed")
    selection_protocol = {
        "selection_split": PublicSplitV2.CALIBRATION.value,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "locked_audit_usage": "not_executed_by_prepare",
        "router_selection_executed": False,
    }
    data_dir = Path(public_data_dir).resolve() if public_data_dir else _resolve_output(
        config_path, benchmark_config.get("public_data_dir", "data/stage_c24b")
    )
    artifacts = Path(artifact_dir).resolve() if artifact_dir else _resolve_output(
        config_path, benchmark_config.get("artifact_dir", "artifacts/stage_c24b")
    )
    # 在首个生成文件落盘前捕获 provenance；否则 prepare 自身会把一个原本
    # clean 的冻结提交错误记录为 dirty。
    git = _git_provenance(config_path.parent.parent)
    data_paths = {split: data_dir / f"{split}.jsonl" for split in rows}
    for split, split_rows in rows.items():
        _write_jsonl(data_paths[split], split_rows)

    train_review = build_review_rows(rows[PublicSplitV2.TRAIN.value])
    calibration_review = build_review_rows(rows[PublicSplitV2.CALIBRATION.value])
    locked_review = build_review_rows(rows[PublicSplitV2.LOCKED_AUDIT.value])
    train_review_path = artifacts / "public_train_v2_review.csv"
    calibration_review_path = artifacts / "public_calibration_v2_review.csv"
    locked_review_path = artifacts / "public_locked_audit_v2_review.csv"
    legacy_review_path = artifacts / "c24_locked_audit_independent_review.csv"
    _write_review_csv(train_review_path, train_review)
    _write_review_csv(calibration_review_path, calibration_review)
    _write_review_csv(locked_review_path, locked_review)

    generate_legacy_review = benchmark_config.get("generate_legacy_c24_review", True)
    if not isinstance(generate_legacy_review, bool):
        raise ValueError("generate_legacy_c24_review must be boolean")
    legacy_path: Path | None = None
    legacy_review: list[dict[str, Any]] = []
    legacy_source_seal: dict[str, Any] | None = None
    if generate_legacy_review:
        legacy_path = _resolve_output(
            config_path,
            benchmark_config.get(
                "legacy_c24_locked_rows", "data/stage_c24/public_locked_audit.jsonl"
            ),
        )
        legacy_source_seal = _verify_legacy_c24_review_source(
            legacy_path, config_path=config_path
        )
        fixed_sources = raw.get("fixed_sources")
        if not isinstance(fixed_sources, Mapping):
            raise ValueError("stage C2.4b fixed source provenance is malformed")
        expected_manifest_path = _resolve_output(
            config_path,
            str(fixed_sources.get("legacy_c24_benchmark_manifest", "")),
        )
        expected_review_path = _resolve_output(
            config_path,
            str(fixed_sources.get("legacy_c24_review_source", "")),
        )
        if (
            expected_manifest_path
            != Path(legacy_source_seal["c24_manifest_path"])
            or expected_review_path != legacy_path.resolve()
            or str(fixed_sources.get("legacy_c24_benchmark_manifest_sha256", ""))
            != legacy_source_seal["c24_manifest_sha256"]
            or str(fixed_sources.get("legacy_c24_review_source_sha256", ""))
            != legacy_source_seal["source_sha256"]
        ):
            raise ValueError("legacy C2.4 review provenance differs from preregistration")
        legacy_rows = _read_legacy_c24_source(
            legacy_path, benchmark=validate_benchmark_config_v2(benchmark_config), config_path=config_path
        )
        legacy_review = build_legacy_c24_review_rows(legacy_rows)
        _write_review_csv(legacy_review_path, legacy_review)

    review_row_id_sha256 = {
        PublicSplitV2.TRAIN.value: sha256_json(
            sorted(row["row_id"] for row in train_review)
        ),
        PublicSplitV2.CALIBRATION.value: sha256_json(
            sorted(row["row_id"] for row in calibration_review)
        ),
        PublicSplitV2.LOCKED_AUDIT.value: sha256_json(
            sorted(row["row_id"] for row in locked_review)
        ),
    }
    review_static_hashes = {
        PublicSplitV2.TRAIN.value: review_static_sha256(train_review),
        PublicSplitV2.CALIBRATION.value: review_static_sha256(calibration_review),
        PublicSplitV2.LOCKED_AUDIT.value: review_static_sha256(locked_review),
    }

    split_audit_sha = sha256_json(audit["split_isolation"])
    historical_audit_sha = sha256_json(audit["historical_collision_audit"])
    review_paths = {
        PublicSplitV2.TRAIN.value: train_review_path,
        PublicSplitV2.CALIBRATION.value: calibration_review_path,
        PublicSplitV2.LOCKED_AUDIT.value: locked_review_path,
    }
    split_manifests: dict[str, dict[str, Any]] = {}
    for split, split_rows in rows.items():
        data_path = data_paths[split]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "stage": "C2.4b-selective-discrete-relation-router",
            "benchmark_version": benchmark_config["version"],
            "preprocessing_version": benchmark_config["preprocessing_version"],
            "split": split,
            "status": "prepared_pending_human_review",
            "answer_free": True,
            "contains_private_answers": False,
            "public_benchmark_human_reviewed": False,
            "formal_locked_audit_executed": False,
            "row_count": len(split_rows),
            "row_counts_by_sample_type": dict(
                sorted(Counter(row["sample_type"] for row in split_rows).items())
            ),
            "family_count": len({row["family_id"] for row in split_rows}),
            "family_count_by_relation": {
                relation: len(
                    {
                        row["family_id"]
                        for row in split_rows
                        if row["relation_id"] == relation
                    }
                )
                for relation in RELATIONS
            },
            "data_file": {
                "path": str(data_path),
                "sha256": sha256_file(data_path),
                "row_sha256": sha256_json(list(split_rows)),
            },
            "source_config_sha256": sha256_file(config_path),
            "config_sha256": sha256_file(config_path),
            "model_provenance": dict(model_provenance),
            "review_manifest_sha256": None,
            "split_isolation_audit_sha256": split_audit_sha,
            "historical_collision_audit_sha256": historical_audit_sha,
            "selection_protocol": selection_protocol,
            "provenance": {"git": git},
        }
        if split in review_paths:
            review_path = review_paths[split]
            manifest["review_file"] = {
                "path": str(review_path),
                "sha256": sha256_file(review_path),
                "row_count": {
                    PublicSplitV2.TRAIN.value: len(train_review),
                    PublicSplitV2.CALIBRATION.value: len(calibration_review),
                    PublicSplitV2.LOCKED_AUDIT.value: len(locked_review),
                }[split],
                "row_id_sha256": review_row_id_sha256[split],
                "static_review_sha256": review_static_hashes[split],
                "all_rows_pending": True,
            }
        split_manifests[split] = manifest
        _write_json(artifacts / f"{split}_manifest.json", manifest)

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "stage": "C2.4b-public-selective-router-benchmark",
        "benchmark_version": benchmark_config["version"],
        "preprocessing_version": benchmark_config["preprocessing_version"],
        "review_status": PENDING_REVIEW_STATUS,
        "answer_free": True,
        "contains_private_answers": False,
        "public_benchmark_human_reviewed": False,
        "formal_locked_audit_executed": False,
        "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "row_counts_by_sample_type": {
            split: dict(sorted(Counter(row["sample_type"] for row in split_rows).items()))
            for split, split_rows in rows.items()
        },
        "data_sha256": {
            split: sha256_file(data_paths[split]) for split in rows
        },
        "config_sha256": sha256_file(config_path),
        "model_provenance": dict(model_provenance),
        "review_manifest_sha256": None,
        "selection_protocol": selection_protocol,
        "audit": audit,
        "split_manifests": {
            split: {
                "path": str(artifacts / f"{split}_manifest.json"),
                "sha256": sha256_file(artifacts / f"{split}_manifest.json"),
            }
            for split in split_manifests
        },
        "review_files": {
            "public_train_v2": {
                "path": str(train_review_path),
                "sha256": sha256_file(train_review_path),
                "row_count": len(train_review),
                "row_id_sha256": review_row_id_sha256[PublicSplitV2.TRAIN.value],
                "static_review_sha256": review_static_hashes[
                    PublicSplitV2.TRAIN.value
                ],
                "all_rows_pending": True,
            },
            "public_calibration_v2": {
                "path": str(calibration_review_path),
                "sha256": sha256_file(calibration_review_path),
                "row_count": len(calibration_review),
                "row_id_sha256": review_row_id_sha256[
                    PublicSplitV2.CALIBRATION.value
                ],
                "static_review_sha256": review_static_hashes[
                    PublicSplitV2.CALIBRATION.value
                ],
                "all_rows_pending": True,
            },
            "public_locked_audit_v2": {
                "path": str(locked_review_path),
                "sha256": sha256_file(locked_review_path),
                "row_count": len(locked_review),
                "row_id_sha256": review_row_id_sha256[
                    PublicSplitV2.LOCKED_AUDIT.value
                ],
                "static_review_sha256": review_static_hashes[
                    PublicSplitV2.LOCKED_AUDIT.value
                ],
                "all_rows_pending": True,
            },
        },
        "provenance": {
            "git": git,
            "config_sha256": sha256_file(config_path),
            "historical_source_sha256": audit["historical_collision_audit"][
                "historical_sources"
            ],
        },
    }
    if generate_legacy_review:
        assert legacy_path is not None
        assert legacy_source_seal is not None
        aggregate["review_files"]["legacy_c24_locked_audit"] = {
            **legacy_source_seal,
            "path": str(legacy_review_path),
            "sha256": sha256_file(legacy_review_path),
            "row_count": len(legacy_review),
            "row_id_sha256": sha256_json(
                sorted(row["row_id"] for row in legacy_review)
            ),
            "static_review_sha256": review_static_sha256(legacy_review),
            "all_rows_pending": True,
            "changes_historical_c24_files": False,
        }
    aggregate_path = artifacts / "public_benchmark_manifest.json"
    _write_json(aggregate_path, aggregate)
    return aggregate


# CLI 集成层可使用更简短的别名。
build_public_benchmark = build_public_benchmark_v2
prepare_public_benchmark = prepare_public_benchmark_v2
prepare_stage_c24b_benchmark = prepare_public_benchmark_v2
