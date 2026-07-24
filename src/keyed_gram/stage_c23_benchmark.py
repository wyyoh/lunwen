from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
RELATIONS = ("registry_id", "city_code", "access_code")
REJECTION_KINDS = ("ambiguous", "unrelated")
INPUT_VIEWS = (
    "phrase_only",
    "entity_masked",
    "entity_masked_strip_suffix",
)

EXPECTED_TRAIN_FAMILIES_PER_RELATION = 6
EXPECTED_VALIDATION_FAMILIES_PER_RELATION = 12
EXPECTED_REJECTION_FAMILIES_PER_KIND = 12
EXPECTED_TRAIN_ENTITY_COUNT = 6
EXPECTED_VALIDATION_ENTITY_COUNT = 6
EXPECTED_TRAIN_FRAME_COUNT = 3
EXPECTED_VALIDATION_FRAME_COUNT = 6
EXPECTED_VALIDATION_ROW_COUNT = 216
EXPECTED_REJECTION_ROW_COUNT = 144
EXPECTED_BENCHMARK_TOTAL = 360

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

_RESPONSE_SUFFIX = re.compile(
    r"\s*(?:answer|response|output|reply|value)\s*:\s*$",
    flags=re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "in",
        "into",
        "is",
        "of",
        "on",
        "the",
        "to",
        "use",
        "when",
        "with",
    }
)
_IRREGULAR_LEMMAS = {
    "assigned": "assign",
    "attached": "attach",
    "cities": "city",
    "denoting": "denote",
    "entries": "entry",
    "filed": "file",
    "filing": "file",
    "indices": "index",
    "issued": "issue",
    "localities": "locality",
    "municipalities": "municipality",
    "printed": "print",
    "records": "record",
    "registries": "registry",
    "representing": "represent",
    "required": "require",
    "services": "service",
    "used": "use",
}


def _surface_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return _SPACE.sub(" ", value)


def normalize_text(value: str) -> str:
    """Normalize case, Unicode, punctuation, and hyphens for leakage checks."""

    value = unicodedata.normalize("NFKC", str(value)).casefold()
    characters = []
    for character in value:
        category = unicodedata.category(character)
        if category.startswith("P") or category == "Pd":
            characters.append(" ")
        else:
            characters.append(character)
    return _SPACE.sub(" ", "".join(characters)).strip()


def phrase_only(relation_phrase: str) -> str:
    return _surface_text(relation_phrase)


def entity_masked(prompt: str, entity: str, *, mask: str = "<ENTITY>") -> str:
    prompt = unicodedata.normalize("NFKC", str(prompt))
    entity = unicodedata.normalize("NFKC", str(entity))
    if not entity or prompt.count(entity) != 1:
        raise ValueError("a benchmark prompt must contain its entity exactly once")
    return _surface_text(prompt.replace(entity, mask))


def strip_response_suffix(text: str) -> str:
    return _surface_text(_RESPONSE_SUFFIX.sub("", _surface_text(text)))


def entity_masked_strip_suffix(
    prompt: str, entity: str, *, mask: str = "<ENTITY>"
) -> str:
    return strip_response_suffix(entity_masked(prompt, entity, mask=mask))


def _simple_lemma(token: str) -> str:
    if token in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        return stem
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def lemma_tokens(value: str) -> tuple[str, ...]:
    tokens = []
    for token in normalize_text(value).split():
        lemma = _simple_lemma(token)
        if lemma and lemma not in _STOP_WORDS:
            tokens.append(lemma)
    return tuple(tokens)


def lemma_bigrams(value: str) -> frozenset[tuple[str, str]]:
    tokens = lemma_tokens(value)
    return frozenset(zip(tokens, tokens[1:]))


def _named_phrases(
    values: Mapping[str, str] | Sequence[str], prefix: str
) -> dict[str, str]:
    if isinstance(values, Mapping):
        return {str(key): str(value) for key, value in values.items()}
    return {f"{prefix}-{index}": str(value) for index, value in enumerate(values)}


def _contains_token_sequence(shorter: str, longer: str) -> bool:
    left = normalize_text(shorter).split()
    right = normalize_text(longer).split()
    if len(left) < 2 or len(left) >= len(right):
        return False
    return any(right[index : index + len(left)] == left for index in range(len(right)))


def audit_lexical_disjointness(
    train_phrases: Mapping[str, str] | Sequence[str],
    validation_phrases: Mapping[str, str] | Sequence[str],
    *,
    lemma_bigram_threshold: float = 0.80,
) -> dict[str, Any]:
    """Reject exact, containment, and lemma-bigram train/validation leakage."""

    if not 0.0 < lemma_bigram_threshold <= 1.0:
        raise ValueError("lemma_bigram_threshold must be in (0, 1]")
    train = _named_phrases(train_phrases, "train")
    validation = _named_phrases(validation_phrases, "validation")
    collisions: dict[str, list[dict[str, Any]]] = {
        "exact": [],
        "containment": [],
        "lemma_bigram": [],
    }
    for train_id, train_phrase in train.items():
        train_normalized = normalize_text(train_phrase)
        train_bigrams = lemma_bigrams(train_phrase)
        for validation_id, validation_phrase in validation.items():
            validation_normalized = normalize_text(validation_phrase)
            record = {
                "train_id": train_id,
                "validation_id": validation_id,
            }
            if train_normalized == validation_normalized:
                collisions["exact"].append(record)
                continue
            if _contains_token_sequence(
                train_phrase, validation_phrase
            ) or _contains_token_sequence(validation_phrase, train_phrase):
                collisions["containment"].append(record)
                continue
            validation_bigrams = lemma_bigrams(validation_phrase)
            if train_bigrams and validation_bigrams:
                overlap = len(train_bigrams & validation_bigrams) / min(
                    len(train_bigrams), len(validation_bigrams)
                )
                if overlap >= lemma_bigram_threshold:
                    collisions["lemma_bigram"].append(
                        {**record, "overlap_coefficient": overlap}
                    )
    collision_count = sum(len(values) for values in collisions.values())
    result = {
        "train_phrase_count": len(train),
        "validation_phrase_count": len(validation),
        "normalization": "NFKC+casefold+punctuation/hyphen-to-space",
        "lemma_bigram_threshold": lemma_bigram_threshold,
        "collision_count": collision_count,
        "collisions": collisions,
        "passed": collision_count == 0,
    }
    if collision_count:
        kinds = [name for name, values in collisions.items() if values]
        raise ValueError(
            "train/validation lexical collision detected: " + ", ".join(kinds)
        )
    return result


def _require_string_list(
    config: Mapping[str, Any], key: str, expected_count: int
) -> list[str]:
    values = config.get(key)
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"benchmark.{key} must contain {expected_count} values")
    cleaned = [_surface_text(value) for value in values]
    if any(not value for value in cleaned) or len(set(cleaned)) != len(cleaned):
        raise ValueError(f"benchmark.{key} must contain unique non-empty strings")
    return cleaned


def _require_frames(
    config: Mapping[str, Any], key: str, expected_count: int
) -> list[str]:
    frames = _require_string_list(config, key, expected_count)
    for frame in frames:
        if frame.count("{entity}") != 1 or frame.count("{relation_phrase}") != 1:
            raise ValueError(
                f"every benchmark.{key} frame needs exactly one entity and phrase"
            )
    return frames


def _require_family_groups(
    raw: Any,
    *,
    expected_groups: Sequence[str],
    expected_count: int,
    field_name: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(raw, Mapping) or set(raw) != set(expected_groups):
        raise ValueError(f"benchmark.{field_name} groups are incomplete")
    output = {}
    for group in expected_groups:
        values = raw[group]
        if not isinstance(values, Mapping) or len(values) != expected_count:
            raise ValueError(
                f"benchmark.{field_name}.{group} must contain {expected_count} families"
            )
        families = {
            _surface_text(family_id): _surface_text(phrase)
            for family_id, phrase in values.items()
        }
        if any(not key or not value for key, value in families.items()):
            raise ValueError("benchmark family IDs and phrases must be non-empty")
        if len(set(map(normalize_text, families.values()))) != len(families):
            raise ValueError(f"benchmark.{field_name}.{group} repeats a phrase")
        output[group] = families
    return output


def _flatten(groups: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    return {
        family_id: phrase
        for families in groups.values()
        for family_id, phrase in families.items()
    }


def validate_benchmark_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("benchmark config must be a mapping")
    if config.get("answer_free") is not True:
        raise ValueError("C2.3 benchmark must be answer-free")
    if config.get("contains_private_answers") is not False:
        raise ValueError("C2.3 benchmark cannot contain private answers")
    version = _surface_text(config.get("version", ""))
    preprocessing_version = _surface_text(config.get("preprocessing_version", ""))
    if not version or not preprocessing_version:
        raise ValueError("benchmark and preprocessing versions are required")

    train_entities = _require_string_list(
        config, "train_entities", EXPECTED_TRAIN_ENTITY_COUNT
    )
    validation_entities = _require_string_list(
        config, "validation_entities", EXPECTED_VALIDATION_ENTITY_COUNT
    )
    if set(map(normalize_text, train_entities)) & set(
        map(normalize_text, validation_entities)
    ):
        raise ValueError("train and validation entities must be disjoint")
    train_frames = _require_frames(
        config, "train_frames", EXPECTED_TRAIN_FRAME_COUNT
    )
    validation_frames = _require_frames(
        config, "validation_frames", EXPECTED_VALIDATION_FRAME_COUNT
    )
    if set(map(normalize_text, train_frames)) & set(
        map(normalize_text, validation_frames)
    ):
        raise ValueError("train and validation frames must be disjoint")

    train_families = _require_family_groups(
        config.get("train_families"),
        expected_groups=RELATIONS,
        expected_count=EXPECTED_TRAIN_FAMILIES_PER_RELATION,
        field_name="train_families",
    )
    validation_families = _require_family_groups(
        config.get("validation_families"),
        expected_groups=RELATIONS,
        expected_count=EXPECTED_VALIDATION_FAMILIES_PER_RELATION,
        field_name="validation_families",
    )
    rejection_families = _require_family_groups(
        config.get("rejection_families"),
        expected_groups=REJECTION_KINDS,
        expected_count=EXPECTED_REJECTION_FAMILIES_PER_KIND,
        field_name="rejection_families",
    )
    all_groups = (train_families, validation_families, rejection_families)
    all_ids = [family_id for groups in all_groups for family_id in _flatten(groups)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("family IDs must be globally unique")
    known_phrases = [
        phrase
        for groups in (train_families, validation_families)
        for phrase in _flatten(groups).values()
    ]
    if len(set(map(normalize_text, known_phrases))) != len(known_phrases):
        raise ValueError("known relation phrases must be globally unique")
    all_phrases = known_phrases + list(_flatten(rejection_families).values())
    if len(set(map(normalize_text, all_phrases))) != len(all_phrases):
        raise ValueError("known and rejection phrases must be exactly disjoint")

    lexical_train = _flatten(train_families)
    private_views = config.get("private_phrase_views", {})
    if private_views:
        if not isinstance(private_views, Mapping):
            raise ValueError("benchmark.private_phrase_views must be a mapping")
        for split, relation_values in private_views.items():
            if not isinstance(relation_values, Mapping):
                raise ValueError("private phrase views must be grouped by relation")
            for relation, phrases in relation_values.items():
                if relation not in RELATIONS or not isinstance(phrases, list):
                    raise ValueError("private phrase views are malformed")
                for index, phrase in enumerate(phrases):
                    lexical_train[
                        f"private-view:{split}:{relation}:{index}"
                    ] = _surface_text(phrase)
    lexical_audit = audit_lexical_disjointness(
        lexical_train, _flatten(validation_families)
    )
    return {
        "version": version,
        "preprocessing_version": preprocessing_version,
        "train_entities": train_entities,
        "validation_entities": validation_entities,
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "train_families": train_families,
        "validation_families": validation_families,
        "rejection_families": rejection_families,
        "lexical_audit": lexical_audit,
    }


def _row(
    *,
    version: str,
    preprocessing_version: str,
    split: str,
    kind: str,
    attribute: str | None,
    family_id: str,
    relation_phrase: str,
    frame_id: str,
    entity_id: str,
    entity: str,
    frame: str,
) -> dict[str, Any]:
    prompt = frame.format(entity=entity, relation_phrase=relation_phrase)
    example_id = f"{split}:{attribute or kind}:{family_id}:{frame_id}:{entity_id}"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": version,
        "preprocessing_version": preprocessing_version,
        "corpus_kind": "answer_free_public_relation_benchmark",
        "example_id": example_id,
        "split": split,
        "kind": kind,
        "attribute": attribute,
        "family_id": family_id,
        "family_signature": " ".join(lemma_tokens(relation_phrase)),
        "relation_phrase": relation_phrase,
        "frame_id": frame_id,
        "template_id": frame_id,
        "entity_id": entity_id,
        "entity": entity,
        "fact_id": f"public:{entity_id}|{attribute or family_id}",
        "prompt": prompt,
        "input_views": {
            "phrase_only": phrase_only(relation_phrase),
            "entity_masked": entity_masked(prompt, entity),
            "entity_masked_strip_suffix": entity_masked_strip_suffix(
                prompt, entity
            ),
        },
        "answer_free": True,
        "contains_private_answer": False,
    }
    if set(row) != ROW_FIELDS:
        raise RuntimeError("internal C2.3 row schema mismatch")
    return row


def _known_rows(
    *,
    split: str,
    version: str,
    preprocessing_version: str,
    entities: Sequence[str],
    frames: Sequence[str],
    families: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for relation_index, relation in enumerate(RELATIONS):
        for family_index, (family_id, relation_phrase) in enumerate(
            families[relation].items()
        ):
            for frame_index, frame in enumerate(frames):
                entity_index = (
                    relation_index + family_index + frame_index
                ) % len(entities)
                rows.append(
                    _row(
                        version=version,
                        preprocessing_version=preprocessing_version,
                        split=split,
                        kind="known",
                        attribute=relation,
                        family_id=family_id,
                        relation_phrase=relation_phrase,
                        frame_id=f"{split}-f{frame_index}",
                        entity_id=f"{split}-e{entity_index}",
                        entity=entities[entity_index],
                        frame=frame,
                    )
                )
    return rows


def _rejection_rows(
    *,
    version: str,
    preprocessing_version: str,
    entities: Sequence[str],
    frames: Sequence[str],
    families: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    family_offset = 0
    for kind in REJECTION_KINDS:
        for family_index, (family_id, relation_phrase) in enumerate(
            families[kind].items()
        ):
            for frame_index, frame in enumerate(frames):
                entity_index = (family_offset + family_index + frame_index) % len(
                    entities
                )
                rows.append(
                    _row(
                        version=version,
                        preprocessing_version=preprocessing_version,
                        split="reject",
                        kind=kind,
                        attribute=None,
                        family_id=family_id,
                        relation_phrase=relation_phrase,
                        frame_id=f"validation-f{frame_index}",
                        entity_id=f"validation-e{entity_index}",
                        entity=entities[entity_index],
                        frame=frame,
                    )
                )
        family_offset += len(families[kind])
    return rows


def _validate_rows(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    expected_counts = {
        "train": (
            len(RELATIONS)
            * EXPECTED_TRAIN_FAMILIES_PER_RELATION
            * EXPECTED_TRAIN_FRAME_COUNT
        ),
        "validation": EXPECTED_VALIDATION_ROW_COUNT,
        "reject": EXPECTED_REJECTION_ROW_COUNT,
    }
    if set(rows) != set(expected_counts):
        raise ValueError("C2.3 rows must have train, validation, and reject splits")
    example_ids = set()
    for split, split_rows in rows.items():
        if len(split_rows) != expected_counts[split]:
            raise ValueError(f"unexpected C2.3 {split} row count")
        for row in split_rows:
            if set(row) != ROW_FIELDS or row["split"] != split:
                raise ValueError("C2.3 row violates the strict schema")
            if row["answer_free"] is not True or row["contains_private_answer"] is not False:
                raise ValueError("C2.3 rows must be answer-free")
            if row["example_id"] in example_ids:
                raise ValueError("C2.3 example IDs must be globally unique")
            example_ids.add(row["example_id"])
            if set(row["input_views"]) != set(INPUT_VIEWS):
                raise ValueError("C2.3 row has incomplete input views")
            if row["input_views"]["phrase_only"] != row["relation_phrase"]:
                raise ValueError("phrase-only view does not match its family")
            if row["entity"] in row["input_views"]["entity_masked"]:
                raise ValueError("entity masking leaked the public entity")
            if row["input_views"]["entity_masked"].count("<ENTITY>") != 1:
                raise ValueError("entity-masked view must contain one placeholder")
            if _RESPONSE_SUFFIX.search(
                row["input_views"]["entity_masked_strip_suffix"]
            ):
                raise ValueError("stripped input view retains a response suffix")
            if split == "reject":
                if row["kind"] not in REJECTION_KINDS or row["attribute"] is not None:
                    raise ValueError("reject labels are malformed")
            elif row["kind"] != "known" or row["attribute"] not in RELATIONS:
                raise ValueError("known relation labels are malformed")

    train = rows["train"]
    validation = rows["validation"]
    reject = rows["reject"]
    if Counter(row["attribute"] for row in train) != Counter(
        {relation: 18 for relation in RELATIONS}
    ):
        raise ValueError("train relations are imbalanced")
    if set(Counter(row["family_id"] for row in train).values()) != {3}:
        raise ValueError("train families are imbalanced")
    if set(Counter(row["frame_id"] for row in train).values()) != {18}:
        raise ValueError("train frames are imbalanced")
    if set(Counter(row["entity_id"] for row in train).values()) != {9}:
        raise ValueError("train entities are imbalanced")
    if set(
        Counter((row["attribute"], row["frame_id"]) for row in train).values()
    ) != {6}:
        raise ValueError("train relation/frame cells are imbalanced")
    if set(
        Counter((row["attribute"], row["entity_id"]) for row in train).values()
    ) != {3}:
        raise ValueError("train relation/entity cells are imbalanced")
    if Counter(row["attribute"] for row in validation) != Counter(
        {relation: 72 for relation in RELATIONS}
    ):
        raise ValueError("validation relations are imbalanced")
    if set(Counter(row["family_id"] for row in validation).values()) != {6}:
        raise ValueError("validation families are imbalanced")
    if set(Counter(row["frame_id"] for row in validation).values()) != {36}:
        raise ValueError("validation frames are imbalanced")
    if set(Counter(row["entity_id"] for row in validation).values()) != {36}:
        raise ValueError("validation entities are imbalanced")
    if set(
        Counter(
            (row["attribute"], row["frame_id"]) for row in validation
        ).values()
    ) != {12}:
        raise ValueError("validation relation/frame cells are imbalanced")
    if set(
        Counter(
            (row["attribute"], row["entity_id"]) for row in validation
        ).values()
    ) != {12}:
        raise ValueError("validation relation/entity cells are imbalanced")
    if Counter(row["kind"] for row in reject) != Counter(
        {kind: 72 for kind in REJECTION_KINDS}
    ):
        raise ValueError("rejection kinds are imbalanced")
    if set(Counter(row["family_id"] for row in reject).values()) != {6}:
        raise ValueError("rejection families are imbalanced")
    if set(Counter(row["frame_id"] for row in reject).values()) != {24}:
        raise ValueError("rejection frames are imbalanced")
    if set(Counter(row["entity_id"] for row in reject).values()) != {24}:
        raise ValueError("rejection entities are imbalanced")
    if set(
        Counter((row["kind"], row["frame_id"]) for row in reject).values()
    ) != {12}:
        raise ValueError("rejection kind/frame cells are imbalanced")
    if set(
        Counter((row["kind"], row["entity_id"]) for row in reject).values()
    ) != {12}:
        raise ValueError("rejection kind/entity cells are imbalanced")


def build_public_lexical_benchmark(
    benchmark_config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    validated = validate_benchmark_config(benchmark_config)
    rows = {
        "train": _known_rows(
            split="train",
            version=validated["version"],
            preprocessing_version=validated["preprocessing_version"],
            entities=validated["train_entities"],
            frames=validated["train_frames"],
            families=validated["train_families"],
        ),
        "validation": _known_rows(
            split="validation",
            version=validated["version"],
            preprocessing_version=validated["preprocessing_version"],
            entities=validated["validation_entities"],
            frames=validated["validation_frames"],
            families=validated["validation_families"],
        ),
        "reject": _rejection_rows(
            version=validated["version"],
            preprocessing_version=validated["preprocessing_version"],
            entities=validated["validation_entities"],
            frames=validated["validation_frames"],
            families=validated["rejection_families"],
        ),
    }
    _validate_rows(rows)
    audit = {
        "lexical_disjointness": validated["lexical_audit"],
        "strict_schema": True,
        "balanced_family_frame_entity_design": True,
        "answer_free": True,
        "contains_private_answers": False,
    }
    return rows, audit


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_public_lexical_benchmark(
    config_path: str | Path,
    *,
    public_data_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark_config = values.get("benchmark")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("stage C2.3 config is missing benchmark")
    rows, audit = build_public_lexical_benchmark(benchmark_config)
    data_dir = Path(
        public_data_dir
        if public_data_dir is not None
        else benchmark_config.get("public_data_dir", "data/stage_c23")
    )
    output_manifest = Path(
        manifest_path
        if manifest_path is not None
        else benchmark_config.get(
            "manifest", "artifacts/stage_c23/public_benchmark_manifest.json"
        )
    )
    paths = {split: data_dir / f"{split}.jsonl" for split in rows}
    for split, split_rows in rows.items():
        _write_jsonl(paths[split], split_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "C2.3-public-lexical-benchmark",
        "benchmark_version": benchmark_config["version"],
        "preprocessing_version": benchmark_config["preprocessing_version"],
        "review_status": benchmark_config.get("review_status"),
        "answer_free": True,
        "contains_private_answers": False,
        "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "benchmark_total": len(rows["validation"]) + len(rows["reject"]),
        "family_counts": {
            "train_known": len({row["family_id"] for row in rows["train"]}),
            "validation_known": len(
                {row["family_id"] for row in rows["validation"]}
            ),
            "reject": len({row["family_id"] for row in rows["reject"]}),
        },
        "input_views": list(INPUT_VIEWS),
        "audit": audit,
        "source_config_sha256": _sha256(config_path),
        "files": {
            split: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for split, path in paths.items()
        },
    }
    if manifest["benchmark_total"] != EXPECTED_BENCHMARK_TOTAL:
        raise RuntimeError("C2.3 benchmark total is not 360")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


prepare_stage_c23_benchmark = prepare_public_lexical_benchmark
