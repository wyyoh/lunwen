from __future__ import annotations

import csv
import hashlib
import json
import re
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


SCHEMA_VERSION = 1
RELATIONS = ("registry_id", "city_code", "access_code")
REJECTION_KINDS = ("ambiguous", "unrelated")
LOCKED_REVIEW_STATUS = "curated_draft_pending_independent_human_review"


class PublicSplit(str, Enum):
    TRAIN = "public_train"
    CALIBRATION = "public_calibration"
    LOCKED_AUDIT = "public_locked_audit"


ROW_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_version",
        "preprocessing_version",
        "corpus_kind",
        "example_id",
        "split",
        "kind",
        "attribute",
        "family_id",
        "family_signature",
        "normalized_token_signature",
        "relation_phrase",
        "frame_id",
        "template_id",
        "entity_id",
        "entity",
        "fact_id",
        "prompt",
        "input_views",
        "answer_free",
        "contains_private_answer",
    }
)

REVIEW_FIELDS = (
    "row_id",
    "phrase_family",
    "phrase",
    "frame",
    "proposed_label",
    "reviewer_1_label",
    "reviewer_2_label",
    "adjudicated_label",
    "ambiguous_flag",
    "notes",
    "review_status",
)

_BENCHMARK_FIELDS = frozenset(
    {
        "version",
        "preprocessing_version",
        "answer_free",
        "contains_private_answers",
        "review_status",
        "public_data_dir",
        "manifest",
        "review_csv",
        "lemma_bigram_threshold",
        "splits",
        "definitions",
        "private_phrase_views",
    }
)
_SPLIT_FIELDS = frozenset(
    {"entities", "frames", "known_families", "rejection_families"}
)
_ALLOWED_ANSWER_METADATA_FIELDS = frozenset(
    {"answer_free", "contains_private_answers", "contains_private_answer"}
)
_ANSWER_BEARING_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "answer_text",
        "answer_value",
        "candidate_answer",
        "candidate_answers",
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
class ValidatedSplit:
    name: PublicSplit
    entities: dict[str, str]
    frames: dict[str, str]
    known_families: dict[str, dict[str, str]]
    rejection_families: dict[str, dict[str, str]]


@dataclass(frozen=True)
class ValidatedBenchmark:
    version: str
    preprocessing_version: str
    review_status: str
    lemma_bigram_threshold: float
    splits: dict[PublicSplit, ValidatedSplit]
    private_phrase_views: dict[str, dict[str, list[str]]]


def _surface_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return _SPACE.sub(" ", value)


def normalized_token_signature(value: str) -> tuple[str, ...]:
    """Return the ordered NFKC/casefold/punctuation-normalized token sequence."""

    return tuple(normalize_text(value).split())


def _surface_signature(value: str) -> str:
    return _surface_text(value).casefold()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def reject_sensitive_benchmark_inputs(value: Any, *, path: str = "benchmark") -> None:
    """Fail closed on answer payloads and any confirmation/seal reference.

    The three answer-free bookkeeping flags are the only allowed keys containing
    the word ``answer``.  Relation phrases may contain ordinary wording such as
    ``Answer:``; only field names can introduce an answer payload.  Confirmation
    and seal references are rejected in both field names and string values so a
    caller cannot smuggle a prohibited path into this preparation boundary.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            if "confirmation" in key or key == "seal" or key.endswith("_seal"):
                raise ValueError(f"prohibited confirmation/seal field at {path}")
            if (
                key in _ANSWER_BEARING_FIELDS
                and key not in _ALLOWED_ANSWER_METADATA_FIELDS
            ):
                raise ValueError(f"answer-bearing field is prohibited at {path}.{raw_key}")
            reject_sensitive_benchmark_inputs(child, path=f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_sensitive_benchmark_inputs(child, path=f"{path}[{index}]")
        return
    if isinstance(value, (str, Path)):
        lowered = str(value).replace("\\", "/").casefold()
        if "confirmation" in lowered or re.search(r"(?:^|[/_.-])seal(?:[/_.-]|$)", lowered):
            raise ValueError(f"prohibited confirmation/seal reference at {path}")


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
    return hashlib.sha256((strict_json_dumps(value) + "\n").encode("utf-8")).hexdigest()


def _named_values(raw: Any, *, prefix: str, field_name: str) -> dict[str, str]:
    if isinstance(raw, Mapping):
        output = {_surface_text(key): _surface_text(value) for key, value in raw.items()}
    elif isinstance(raw, list):
        output = {
            f"{prefix}-{index}": _surface_text(value)
            for index, value in enumerate(raw)
        }
    else:
        raise ValueError(f"{field_name} must be a mapping or list")
    if not output or any(not key or not value for key, value in output.items()):
        raise ValueError(f"{field_name} must contain non-empty IDs and values")
    if len(output) != len(set(output)):
        raise ValueError(f"{field_name} IDs must be unique")
    signatures = [normalize_text(value) for value in output.values()]
    if len(signatures) != len(set(signatures)):
        raise ValueError(f"{field_name} values must be unique after normalization")
    return output


def _validate_frames(raw: Any, *, prefix: str, field_name: str) -> dict[str, str]:
    frames = _named_values(raw, prefix=prefix, field_name=field_name)
    for frame in frames.values():
        if frame.count("{entity}") != 1 or frame.count("{relation_phrase}") != 1:
            raise ValueError(
                f"every {field_name} frame needs exactly one entity and relation phrase"
            )
    return frames


def _validate_family_groups(
    raw: Any,
    *,
    groups: Sequence[str],
    field_name: str,
    required: bool,
) -> dict[str, dict[str, str]]:
    if not required and raw is None:
        return {}
    if not isinstance(raw, Mapping) or set(raw) != set(groups):
        raise ValueError(f"{field_name} groups must be exactly {list(groups)}")
    output: dict[str, dict[str, str]] = {}
    counts = set()
    for group in groups:
        families = _named_values(
            raw[group], prefix=f"{group}-family", field_name=f"{field_name}.{group}"
        )
        output[group] = families
        counts.add(len(families))
    if len(counts) != 1:
        raise ValueError(f"{field_name} must be balanced across groups")
    return output


def _flatten_families(groups: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    return {
        family_id: phrase
        for families in groups.values()
        for family_id, phrase in families.items()
    }


def _validate_definitions(raw: Any) -> None:
    if raw is None:
        return
    if not isinstance(raw, Mapping) or set(raw) != set(RELATIONS):
        raise ValueError("benchmark.definitions must contain exactly the three relations")
    for relation, definitions in raw.items():
        if not isinstance(definitions, list) or not definitions:
            raise ValueError(f"benchmark.definitions.{relation} must be a non-empty list")
        cleaned = [_surface_text(value) for value in definitions]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError(f"benchmark.definitions.{relation} is malformed")


def _validate_private_phrase_views(raw: Any) -> dict[str, dict[str, list[str]]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark.private_phrase_views must be a mapping")
    output: dict[str, dict[str, list[str]]] = {}
    for split, relation_values in raw.items():
        if not isinstance(relation_values, Mapping) or set(relation_values) != set(RELATIONS):
            raise ValueError("private phrase views must be grouped by all three relations")
        output[str(split)] = {}
        for relation, phrases in relation_values.items():
            if not isinstance(phrases, list) or not phrases:
                raise ValueError("private phrase views must be non-empty string lists")
            cleaned = [_surface_text(value) for value in phrases]
            if any(not value for value in cleaned):
                raise ValueError("private phrase views cannot contain empty text")
            output[str(split)][str(relation)] = cleaned
    return output


def validate_benchmark_config(config: Mapping[str, Any]) -> ValidatedBenchmark:
    if not isinstance(config, Mapping):
        raise ValueError("benchmark config must be a mapping")
    reject_sensitive_benchmark_inputs(config)
    unknown = set(config) - _BENCHMARK_FIELDS
    if unknown:
        raise ValueError(f"unknown benchmark fields: {sorted(map(str, unknown))}")
    if config.get("answer_free") is not True:
        raise ValueError("C2.4 public benchmark must be answer-free")
    if config.get("contains_private_answers") is not False:
        raise ValueError("C2.4 public benchmark cannot contain private answers")
    version = _surface_text(config.get("version", ""))
    preprocessing_version = _surface_text(config.get("preprocessing_version", ""))
    if not version or not preprocessing_version:
        raise ValueError("benchmark version and preprocessing version are required")
    review_status = _surface_text(config.get("review_status", ""))
    if review_status != LOCKED_REVIEW_STATUS:
        raise ValueError(
            "public_locked_audit must remain curated_draft_pending_independent_human_review"
        )
    threshold = float(config.get("lemma_bigram_threshold", 0.80))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("lemma_bigram_threshold must be in (0, 1]")

    raw_splits = config.get("splits")
    required_names = {split.value for split in PublicSplit}
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != required_names:
        raise ValueError(f"benchmark.splits must be exactly {sorted(required_names)}")
    validated_splits: dict[PublicSplit, ValidatedSplit] = {}
    for split in PublicSplit:
        raw = raw_splits[split.value]
        if not isinstance(raw, Mapping):
            raise ValueError(f"benchmark.splits.{split.value} must be a mapping")
        unknown_split = set(raw) - _SPLIT_FIELDS
        if unknown_split:
            raise ValueError(
                f"unknown {split.value} fields: {sorted(map(str, unknown_split))}"
            )
        has_rejection = split is not PublicSplit.TRAIN
        if split is PublicSplit.TRAIN and "rejection_families" in raw:
            raise ValueError("public_train must contain known queries only")
        entities = _named_values(
            raw.get("entities"), prefix=f"{split.value}-entity", field_name=f"{split.value}.entities"
        )
        frames = _validate_frames(
            raw.get("frames"), prefix=f"{split.value}-frame", field_name=f"{split.value}.frames"
        )
        known = _validate_family_groups(
            raw.get("known_families"),
            groups=RELATIONS,
            field_name=f"{split.value}.known_families",
            required=True,
        )
        rejection = _validate_family_groups(
            raw.get("rejection_families"),
            groups=REJECTION_KINDS,
            field_name=f"{split.value}.rejection_families",
            required=has_rejection,
        )
        validated_splits[split] = ValidatedSplit(
            name=split,
            entities=entities,
            frames=frames,
            known_families=known,
            rejection_families=rejection,
        )

    _validate_definitions(config.get("definitions"))
    private_views = _validate_private_phrase_views(config.get("private_phrase_views"))
    benchmark = ValidatedBenchmark(
        version=version,
        preprocessing_version=preprocessing_version,
        review_status=review_status,
        lemma_bigram_threshold=threshold,
        splits=validated_splits,
        private_phrase_views=private_views,
    )
    audit_split_isolation(benchmark)
    return benchmark


def _token_sequence_contained(
    left: str, right: str, *, minimum_tokens: int
) -> bool:
    shorter = normalized_token_signature(left)
    longer = normalized_token_signature(right)
    if len(shorter) < minimum_tokens or len(shorter) >= len(longer):
        return False
    return any(
        longer[index : index + len(shorter)] == shorter
        for index in range(len(longer) - len(shorter) + 1)
    )


def audit_phrase_collections(
    left: Mapping[str, str],
    right: Mapping[str, str],
    *,
    left_name: str,
    right_name: str,
    lemma_bigram_threshold: float,
    minimum_containment_tokens: int = 1,
) -> dict[str, Any]:
    """Audit two phrase collections without silently collapsing collision kinds."""

    collisions: dict[str, list[dict[str, Any]]] = {
        "exact": [],
        "substring_containment": [],
        "normalized_token_signature": [],
        "lemma_bigram": [],
    }
    for left_id, left_phrase in left.items():
        left_surface = _surface_signature(left_phrase)
        left_tokens = normalized_token_signature(left_phrase)
        left_bigrams = lemma_bigrams(left_phrase)
        for right_id, right_phrase in right.items():
            record = {"left_id": left_id, "right_id": right_id}
            right_surface = _surface_signature(right_phrase)
            right_tokens = normalized_token_signature(right_phrase)
            if left_surface == right_surface:
                collisions["exact"].append(record)
                continue
            if left_tokens == right_tokens:
                collisions["normalized_token_signature"].append(record)
                continue
            if _token_sequence_contained(
                left_phrase,
                right_phrase,
                minimum_tokens=minimum_containment_tokens,
            ) or _token_sequence_contained(
                right_phrase,
                left_phrase,
                minimum_tokens=minimum_containment_tokens,
            ):
                collisions["substring_containment"].append(record)
                continue
            left_token_set = frozenset(left_tokens)
            right_token_set = frozenset(right_tokens)
            if left_token_set and left_token_set == right_token_set:
                collisions["normalized_token_signature"].append(
                    {**record, "signature_kind": "unique_token_set"}
                )
                continue
            right_bigrams = lemma_bigrams(right_phrase)
            if left_bigrams and right_bigrams:
                overlap = len(left_bigrams & right_bigrams) / min(
                    len(left_bigrams), len(right_bigrams)
                )
                if overlap >= lemma_bigram_threshold:
                    collisions["lemma_bigram"].append(
                        {**record, "overlap_coefficient": overlap}
                    )
    collision_count = sum(len(items) for items in collisions.values())
    return {
        "left": left_name,
        "right": right_name,
        "left_phrase_count": len(left),
        "right_phrase_count": len(right),
        "definitions": {
            "exact": "NFKC + whitespace collapse + casefold surface equality",
            "substring_containment": (
                "contiguous normalized token sequence with minimum length "
                f"{minimum_containment_tokens}"
            ),
            "normalized_token_signature": (
                "complete ordered sequence equality or complete unique-token-set equality "
                "after NFKC/casefold/punctuation normalization"
            ),
            "lemma_bigram": "overlap coefficient on simple-lemma bigram sets",
        },
        "lemma_bigram_threshold": lemma_bigram_threshold,
        "collisions": collisions,
        "collision_count": collision_count,
        "passed": collision_count == 0,
    }


def _identity_collision_audit(
    left: Mapping[str, str], right: Mapping[str, str], *, unit: str
) -> dict[str, Any]:
    id_overlap = sorted(set(left) & set(right))
    left_values: dict[str, list[str]] = {}
    right_values: dict[str, list[str]] = {}
    for key, value in left.items():
        left_values.setdefault(normalize_text(value), []).append(key)
    for key, value in right.items():
        right_values.setdefault(normalize_text(value), []).append(key)
    value_overlap = sorted(set(left_values) & set(right_values))
    collisions = [
        {
            "normalized_value": value,
            "left_ids": left_values[value],
            "right_ids": right_values[value],
        }
        for value in value_overlap
    ]
    return {
        "unit": unit,
        "id_collisions": id_overlap,
        "normalized_value_collisions": collisions,
        "collision_count": len(id_overlap) + len(collisions),
        "passed": not id_overlap and not collisions,
    }


def audit_split_isolation(benchmark: ValidatedBenchmark) -> dict[str, Any]:
    """Enforce phrase-family, entity, and frame group isolation across splits."""

    pairs = (
        (PublicSplit.TRAIN, PublicSplit.CALIBRATION),
        (PublicSplit.TRAIN, PublicSplit.LOCKED_AUDIT),
        (PublicSplit.CALIBRATION, PublicSplit.LOCKED_AUDIT),
    )
    phrase_refs: dict[PublicSplit, dict[str, str]] = {}
    for split, values in benchmark.splits.items():
        phrase_refs[split] = {
            **_flatten_families(values.known_families),
            **_flatten_families(values.rejection_families),
        }
    for private_split, relation_groups in benchmark.private_phrase_views.items():
        for relation, phrases in relation_groups.items():
            for index, phrase in enumerate(phrases):
                phrase_refs[PublicSplit.TRAIN][
                    f"private-view:{private_split}:{relation}:{index}"
                ] = phrase

    pair_audits: dict[str, Any] = {}
    failed: list[str] = []
    for left_name, right_name in pairs:
        left = benchmark.splits[left_name]
        right = benchmark.splits[right_name]
        key = f"{left_name.value}__vs__{right_name.value}"
        family_id_overlap = sorted(
            set(_flatten_families(left.known_families))
            | set(_flatten_families(left.rejection_families))
        )
        right_family_ids = set(_flatten_families(right.known_families)) | set(
            _flatten_families(right.rejection_families)
        )
        family_id_overlap = sorted(set(family_id_overlap) & right_family_ids)
        audit = {
            "phrases": audit_phrase_collections(
                phrase_refs[left_name],
                phrase_refs[right_name],
                left_name=left_name.value,
                right_name=right_name.value,
                lemma_bigram_threshold=benchmark.lemma_bigram_threshold,
                # Calibration is an allowed family-group split of the previously
                # used public material, where generic one-word overlap is
                # expected.  The locked audit is held to strict substring
                # isolation, including a one-token phrase contained in another.
                minimum_containment_tokens=(
                    1
                    if PublicSplit.LOCKED_AUDIT in (left_name, right_name)
                    else 2
                ),
            ),
            "phrase_family_id": {
                "collisions": family_id_overlap,
                "collision_count": len(family_id_overlap),
                "passed": not family_id_overlap,
            },
            "entity": _identity_collision_audit(left.entities, right.entities, unit="entity"),
            "frame": _identity_collision_audit(left.frames, right.frames, unit="frame"),
        }
        audit["passed"] = all(
            audit[name]["passed"]
            for name in ("phrases", "phrase_family_id", "entity", "frame")
        )
        if not audit["passed"]:
            failed.append(key)
        pair_audits[key] = audit
    result = {
        "schema_version": SCHEMA_VERSION,
        "normalization": "NFKC+casefold+punctuation/hyphen-to-space",
        "normalized_token_collision_definition": (
            "equality of the complete ordered normalized-token sequence or "
            "complete unique-token set; ordinary partial token reuse is not itself a collision"
        ),
        "pairs": pair_audits,
        "passed": not failed,
    }
    if failed:
        raise ValueError("public split isolation collision detected: " + ", ".join(failed))
    return result


def _make_row(
    *,
    benchmark: ValidatedBenchmark,
    split: PublicSplit,
    kind: str,
    relation: str | None,
    family_id: str,
    relation_phrase: str,
    frame_id: str,
    frame: str,
    entity_id: str,
    entity: str,
) -> dict[str, Any]:
    prompt = frame.format(entity=entity, relation_phrase=relation_phrase)
    example_id = f"{split.value}:{relation or kind}:{family_id}:{frame_id}:{entity_id}"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": benchmark.version,
        "preprocessing_version": benchmark.preprocessing_version,
        "corpus_kind": "answer_free_public_relation_contract_benchmark",
        "example_id": example_id,
        "split": split.value,
        "kind": kind,
        "attribute": relation,
        "family_id": family_id,
        "family_signature": " ".join(lemma_tokens(relation_phrase)),
        "normalized_token_signature": " ".join(normalized_token_signature(relation_phrase)),
        "relation_phrase": relation_phrase,
        "frame_id": frame_id,
        "template_id": frame_id,
        "entity_id": entity_id,
        "entity": entity,
        "fact_id": f"public:{entity_id}|{relation or family_id}",
        "prompt": prompt,
        "input_views": {
            "phrase_only": phrase_only(relation_phrase),
            "entity_masked": entity_masked(prompt, entity),
            "entity_masked_strip_suffix": entity_masked_strip_suffix(prompt, entity),
        },
        "answer_free": True,
        "contains_private_answer": False,
    }
    if set(row) != ROW_FIELDS:
        raise RuntimeError("internal C2.4 row schema mismatch")
    return row


def _materialize_split(
    benchmark: ValidatedBenchmark, values: ValidatedSplit
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = list(values.entities.items())
    frames = list(values.frames.items())
    offset = 0
    for relation in RELATIONS:
        for family_index, (family_id, relation_phrase) in enumerate(
            values.known_families[relation].items()
        ):
            for frame_index, (frame_id, frame) in enumerate(frames):
                entity_id, entity = entities[(offset + family_index + frame_index) % len(entities)]
                rows.append(
                    _make_row(
                        benchmark=benchmark,
                        split=values.name,
                        kind="known",
                        relation=relation,
                        family_id=family_id,
                        relation_phrase=relation_phrase,
                        frame_id=frame_id,
                        frame=frame,
                        entity_id=entity_id,
                        entity=entity,
                    )
                )
        offset += len(values.known_families[relation])
    for kind in REJECTION_KINDS:
        for family_index, (family_id, relation_phrase) in enumerate(
            values.rejection_families.get(kind, {}).items()
        ):
            for frame_index, (frame_id, frame) in enumerate(frames):
                entity_id, entity = entities[(offset + family_index + frame_index) % len(entities)]
                rows.append(
                    _make_row(
                        benchmark=benchmark,
                        split=values.name,
                        kind=kind,
                        relation=None,
                        family_id=family_id,
                        relation_phrase=relation_phrase,
                        frame_id=frame_id,
                        frame=frame,
                        entity_id=entity_id,
                        entity=entity,
                    )
                )
        offset += len(values.rejection_families.get(kind, {}))
    return rows


def _validate_materialized_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]], benchmark: ValidatedBenchmark
) -> None:
    if set(rows) != {split.value for split in PublicSplit}:
        raise ValueError("C2.4 rows must contain exactly the three public splits")
    example_ids: set[str] = set()
    for split in PublicSplit:
        split_rows = rows[split.value]
        expected = benchmark.splits[split]
        known_family_count = len(_flatten_families(expected.known_families))
        reject_family_count = len(_flatten_families(expected.rejection_families))
        expected_count = (known_family_count + reject_family_count) * len(expected.frames)
        if len(split_rows) != expected_count:
            raise ValueError(f"unexpected {split.value} row count")
        for row in split_rows:
            if set(row) != ROW_FIELDS or row["split"] != split.value:
                raise ValueError("C2.4 row violates the strict schema")
            if row["answer_free"] is not True or row["contains_private_answer"] is not False:
                raise ValueError("C2.4 rows must be answer-free")
            if row["example_id"] in example_ids:
                raise ValueError("C2.4 example IDs must be globally unique")
            example_ids.add(str(row["example_id"]))
            if set(row["input_views"]) != set(INPUT_VIEWS):
                raise ValueError("C2.4 row has incomplete input views")
            if row["entity"] in row["input_views"]["entity_masked"]:
                raise ValueError("entity masking leaked a public entity")
            if row["input_views"]["entity_masked"].count("<ENTITY>") != 1:
                raise ValueError("entity-masked view must contain one placeholder")
            if split is PublicSplit.TRAIN and row["kind"] != "known":
                raise ValueError("public_train must contain known rows only")
            if row["kind"] == "known":
                if row["attribute"] not in RELATIONS:
                    raise ValueError("known relation row has an invalid label")
            elif row["kind"] not in REJECTION_KINDS or row["attribute"] is not None:
                raise ValueError("reject row has an invalid kind/attribute contract")

        known = [row for row in split_rows if row["kind"] == "known"]
        expected_per_relation = len(expected.known_families[RELATIONS[0]]) * len(
            expected.frames
        )
        if Counter(row["attribute"] for row in known) != Counter(
            {relation: expected_per_relation for relation in RELATIONS}
        ):
            raise ValueError(f"{split.value} known relations are imbalanced")
        if set(Counter(row["family_id"] for row in split_rows).values()) != {
            len(expected.frames)
        }:
            raise ValueError(f"{split.value} phrase families are imbalanced")


def build_public_benchmark(
    benchmark_config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    benchmark = validate_benchmark_config(benchmark_config)
    rows = {
        split.value: _materialize_split(benchmark, benchmark.splits[split])
        for split in PublicSplit
    }
    _validate_materialized_rows(rows, benchmark)
    isolation = audit_split_isolation(benchmark)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "strict_schema": True,
        "balanced_family_frame_entity_design": True,
        "answer_free": True,
        "contains_private_answers": False,
        "public_benchmark_human_reviewed": False,
        "split_isolation": isolation,
    }
    return rows, audit


def build_locked_audit_review_rows(
    locked_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    for row in locked_rows:
        if row.get("split") != PublicSplit.LOCKED_AUDIT.value:
            raise ValueError("review rows may only be created for public_locked_audit")
        proposed_label = row.get("attribute") or row.get("kind")
        review = {
            "row_id": str(row["example_id"]),
            "phrase_family": str(row["family_id"]),
            "phrase": str(row["relation_phrase"]),
            "frame": str(row["prompt"]),
            "proposed_label": str(proposed_label),
            "reviewer_1_label": "",
            "reviewer_2_label": "",
            "adjudicated_label": "",
            "ambiguous_flag": "true" if row["kind"] == "ambiguous" else "false",
            "notes": "",
            "review_status": "pending",
        }
        if tuple(review) != REVIEW_FIELDS:
            raise RuntimeError("internal review schema mismatch")
        review_rows.append(review)
    if len({row["row_id"] for row in review_rows}) != len(review_rows):
        raise ValueError("review row IDs must be unique")
    return review_rows


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


def prepare_public_benchmark(
    config_path: str | Path,
    *,
    public_data_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    review_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark_config = raw.get("benchmark")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("stage C2.4 config is missing benchmark")
    rows, audit = build_public_benchmark(benchmark_config)
    data_dir = Path(
        public_data_dir
        if public_data_dir is not None
        else benchmark_config.get("public_data_dir", "data/stage_c24")
    )
    output_manifest = Path(
        manifest_path
        if manifest_path is not None
        else benchmark_config.get(
            "manifest", "artifacts/stage_c24/public_benchmark_manifest.json"
        )
    )
    output_review = Path(
        review_csv_path
        if review_csv_path is not None
        else benchmark_config.get(
            "review_csv", "artifacts/stage_c24/public_locked_audit_review.csv"
        )
    )
    paths = {split: data_dir / f"{split}.jsonl" for split in rows}
    for split, split_rows in rows.items():
        _write_jsonl(paths[split], split_rows)
    review_rows = build_locked_audit_review_rows(rows[PublicSplit.LOCKED_AUDIT.value])
    _write_review_csv(output_review, review_rows)

    row_counts_by_kind = {
        split: dict(sorted(Counter(row["kind"] for row in split_rows).items()))
        for split, split_rows in rows.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "C2.4-public-discrete-relation-contract-benchmark",
        "benchmark_version": str(benchmark_config["version"]),
        "preprocessing_version": str(benchmark_config["preprocessing_version"]),
        "review_status": LOCKED_REVIEW_STATUS,
        "answer_free": True,
        "contains_private_answers": False,
        "public_benchmark_human_reviewed": False,
        "provisional_locked_audit": True,
        "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "row_counts_by_kind": row_counts_by_kind,
        "family_counts": {
            split: len({row["family_id"] for row in split_rows})
            for split, split_rows in rows.items()
        },
        "input_views": list(INPUT_VIEWS),
        "audit": audit,
        "source_config_sha256": sha256_file(config_path),
        "files": {
            split: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for split, path in paths.items()
        },
        "human_review_file": {
            "path": str(output_review.resolve()),
            "sha256": sha256_file(output_review),
            "row_count": len(review_rows),
            "all_rows_pending": all(row["review_status"] == "pending" for row in review_rows),
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(strict_json_dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


prepare_stage_c24_benchmark = prepare_public_benchmark
