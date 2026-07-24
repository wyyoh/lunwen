from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import yaml

from keyed_gram.stage_c24b_benchmark import (
    PENDING_REVIEW_STATUS,
    REVIEW_FIELDS,
    ROW_FIELDS,
    PublicSplitV2,
    audit_phrase_collections_v2,
    build_legacy_c24_review_rows,
    build_public_benchmark_v2,
    build_public_split_v2,
    build_review_rows,
    prepare_public_benchmark_v2,
    reject_sensitive_benchmark_inputs,
    review_static_sha256,
    sha256_file,
    sha256_json,
    strict_json_dumps,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage_c24b.yaml"
SMOKE_CONFIG_PATH = ROOT / "configs" / "stage_c24b_smoke.yaml"
LEGACY_LOCKED_PATH = ROOT / "artifacts" / "stage_c24" / "public_locked_audit_review.csv"


def _config(path: Path = CONFIG_PATH):
    return copy.deepcopy(yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"])


def _build(path: Path = CONFIG_PATH):
    return build_public_benchmark_v2(_config(path), config_path=path)


def _walk(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def _legacy_rows():
    rows = []
    entities = yaml.safe_load(
        (ROOT / "configs" / "stage_c24.yaml").read_text(encoding="utf-8")
    )["benchmark"]["splits"]["public_locked_audit"]["entities"]
    with LEGACY_LOCKED_PATH.open(encoding="utf-8", newline="") as handle:
        for review in csv.DictReader(handle):
            label = review["proposed_label"]
            kind = label if label in {"ambiguous", "unrelated"} else "known"
            matches = [entity for entity in entities if entity in review["frame"]]
            assert len(matches) == 1
            rows.append(
                {
                    "example_id": review["row_id"],
                    "split": "public_locked_audit",
                    "kind": kind,
                    "attribute": label if kind == "known" else None,
                    "family_id": review["phrase_family"],
                    "relation_phrase": review["phrase"],
                    "prompt": review["frame"],
                    "entity": matches[0],
                }
            )
    return rows


def test_v2_protocol_is_answer_free_balanced_and_contains_required_sample_types():
    rows, audit = _build()

    assert {split: len(values) for split, values in rows.items()} == {
        "public_train_v2": 72,
        "public_calibration_v2": 160,
        "public_locked_audit_v2": 160,
    }
    assert audit["passed"] is True
    assert audit["answer_free"] is True
    assert audit["contains_private_answers"] is False
    assert audit["public_benchmark_human_reviewed"] is False
    assert audit["formal_locked_audit_executed"] is False
    for split, values in rows.items():
        assert all(set(row) == ROW_FIELDS for row in values)
        assert all(row["answer_free"] is True for row in values)
        assert all(row["contains_private_answer"] is False for row in values)
        assert all(row["split"] == split for row in values)
        assert all("private_answer" not in row for row in values)
    assert {row["sample_type"] for row in rows["public_train_v2"]} == {"known"}
    for split in ("public_calibration_v2", "public_locked_audit_v2"):
        assert Counter(row["sample_type"] for row in rows[split]) == Counter(
            {"known": 96, "ambiguous": 32, "unrelated": 32}
        )


def test_train_has_eight_families_per_relation_and_calibration_has_ood_hard_pairs():
    rows, _ = _build()
    train = rows["public_train_v2"]
    calibration = rows["public_calibration_v2"]

    assert {
        relation: len({row["family_id"] for row in train if row["relation_id"] == relation})
        for relation in ("registry_id", "city_code", "access_code")
    } == {"registry_id": 8, "city_code": 8, "access_code": 8}
    assert all(row["lexical_ood"] is False for row in train)
    assert all(row["lexical_ood"] is True for row in calibration)
    for relation in ("registry_id", "city_code", "access_code"):
        hard_families = {
            row["family_id"]
            for row in calibration
            if row["relation_id"] == relation and row["hard_negative"]
        }
        assert len(hard_families) >= 2
    pair_labels = {
        row["relation_id"]
        for row in calibration
        if row["minimal_pair_id"] == "cal_v2_pair_credential"
    }
    assert pair_labels == {"registry_id", "city_code", "access_code"}


def test_v2_split_and_c23_c24_historical_collision_audits_pass():
    _, audit = _build()

    split_audit = audit["split_isolation"]
    assert split_audit["passed"] is True
    assert split_audit["minimum_containment_tokens"] == 2
    assert split_audit["shared_single_tokens_allowed"] is True
    for pair in split_audit["pairs"].values():
        assert pair["passed"] is True
        assert all(item["collision_count"] == 0 for item in pair.values() if isinstance(item, dict))

    historical = audit["historical_collision_audit"]
    assert historical["passed"] is True
    assert set(historical["historical_sources"]) == {
        "stage_c23_config",
        "stage_c24_config",
    }
    assert historical["historical_phrase_count"] >= 100
    # C2.3 使用 train_/validation_ 前缀；这些历史单位不得被收集器漏掉。
    assert historical["historical_entity_count"] >= 25
    assert historical["historical_frame_count"] >= 19
    for split in PublicSplitV2:
        assert historical["splits"][split.value]["passed"] is True


@pytest.mark.parametrize(
    ("left", "right", "collision_kind"),
    [
        ("Alpha Record", "Alpha Record", "exact_text"),
        ("Alpha-Record", "alpha record", "normalized_text"),
        ("bureau enrollment", "new bureau enrollment locator", "substring_containment"),
        ("registry token alpha", "alpha token registry", "token_signature"),
        ("official registries serials", "official registry serial", "lemma_bigram"),
    ],
)
def test_collision_audit_rejects_all_preregistered_collision_levels(
    left, right, collision_kind
):
    audit = audit_phrase_collections_v2(
        {"left": left},
        {"right": right},
        left_name="left",
        right_name="right",
        lemma_bigram_threshold=0.8,
    )
    assert audit["passed"] is False
    assert audit["collision_count"] >= 1
    assert audit["collisions"][collision_kind]


def test_collision_audit_allows_a_shared_single_word():
    audit = audit_phrase_collections_v2(
        {"left": "registry constellation glyph"},
        {"right": "registry lunar pointer"},
        left_name="left",
        right_name="right",
        lemma_bigram_threshold=0.8,
    )
    assert audit["passed"] is True
    assert audit["collision_count"] == 0


@pytest.mark.parametrize("unit", ["known_families", "entities", "frames"])
def test_family_entity_and_frame_collisions_fail_closed(unit):
    config = _config()
    train = config["splits"]["public_train_v2"]
    locked = config["splits"]["public_locked_audit_v2"]
    if unit == "known_families":
        family_id, phrase = next(iter(train[unit]["registry_id"].items()))
        locked[unit]["registry_id"].pop(next(iter(locked[unit]["registry_id"])))
        locked[unit]["registry_id"][family_id] = {
            "phrase": phrase,
            "lexical_ood": True,
        }
    else:
        locked[unit][0] = train[unit][0]
    with pytest.raises(ValueError, match="split isolation collision"):
        build_public_benchmark_v2(config, config_path=CONFIG_PATH)


def test_historical_c24_phrase_reuse_fails_closed():
    config = _config()
    first = next(iter(config["splits"]["public_locked_audit_v2"]["known_families"]["registry_id"]))
    config["splits"]["public_locked_audit_v2"]["known_families"]["registry_id"][first][
        "phrase"
    ] = "archival accession locator"
    with pytest.raises(ValueError, match="historical C2.3/C2.4 collision"):
        build_public_benchmark_v2(config, config_path=CONFIG_PATH)


@pytest.mark.parametrize(
    ("unit", "historical_value"),
    [
        ("entities", "amber public profile"),
        (
            "frames",
            "Public relation request for {entity}: {relation_phrase}. Answer:",
        ),
    ],
)
def test_historical_c23_entity_and_frame_reuse_fails_closed(unit, historical_value):
    config = _config()
    config["splits"]["public_locked_audit_v2"][unit][0] = historical_value
    with pytest.raises(ValueError, match="historical C2.3/C2.4 collision"):
        build_public_benchmark_v2(config, config_path=CONFIG_PATH)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update({"private_answer": "must-not-load"}),
        lambda c: c.update({"answers": ["must-not-load"]}),
        lambda c: c.update({"confirmation_path": "forbidden.jsonl"}),
        lambda c: c.update({"unexpected_path": "artifacts/old-seal/data.json"}),
    ],
)
def test_private_answer_and_confirmation_or_seal_inputs_are_rejected(mutation):
    config = _config()
    mutation(config)
    with pytest.raises(ValueError, match="private-answer|confirmation/seal"):
        reject_sensitive_benchmark_inputs(config)
    with pytest.raises(ValueError, match="private-answer|confirmation/seal"):
        build_public_benchmark_v2(config, config_path=CONFIG_PATH)


def test_review_templates_are_blank_pending_and_have_full_dual_review_schema():
    rows, _ = _build()
    for split, expected_count in (
        ("public_train_v2", 72),
        ("public_calibration_v2", 160),
        ("public_locked_audit_v2", 160),
    ):
        review = build_review_rows(rows[split])
        assert len(review) == expected_count
        assert all(tuple(row) == REVIEW_FIELDS for row in review)
        assert {row["review_status"] for row in review} == {"pending"}
        for field in (
            "reviewer_1_label",
            "reviewer_2_label",
            "reviewer_1_type",
            "reviewer_2_type",
            "adjudicated_label",
            "adjudicated_type",
            "notes",
        ):
            assert {row[field] for row in review} == {""}
        expected_types = (
            {"known"}
            if split == "public_train_v2"
            else {"known", "ambiguous", "unrelated"}
        )
        assert {row["sample_type"] for row in review} == expected_types


def test_legacy_c24_dual_review_template_has_120_pending_rows_without_mutation():
    before = sha256_file(LEGACY_LOCKED_PATH)
    review = build_legacy_c24_review_rows(_legacy_rows())

    assert len(review) == 120
    assert {row["split"] for row in review} == {
        "c24_public_locked_audit_legacy_review"
    }
    assert {row["review_status"] for row in review} == {"pending"}
    assert {row["reviewer_1_label"] for row in review} == {""}
    assert {row["reviewer_2_label"] for row in review} == {""}
    assert {row["adjudicated_label"] for row in review} == {""}
    assert sha256_file(LEGACY_LOCKED_PATH) == before


def test_prepare_writes_jsonl_manifests_hashes_and_four_blank_review_files(tmp_path):
    legacy_before = sha256_file(LEGACY_LOCKED_PATH)
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    manifest = prepare_public_benchmark_v2(
        CONFIG_PATH, public_data_dir=data_dir, artifact_dir=artifact_dir
    )

    assert manifest["review_status"] == PENDING_REVIEW_STATUS
    assert manifest["public_benchmark_human_reviewed"] is False
    assert manifest["formal_locked_audit_executed"] is False
    assert manifest["row_counts"] == {
        "public_train_v2": 72,
        "public_calibration_v2": 160,
        "public_locked_audit_v2": 160,
    }
    assert manifest["audit"]["historical_collision_audit"]["passed"] is True
    assert sha256_file(LEGACY_LOCKED_PATH) == legacy_before
    assert manifest["review_files"]["legacy_c24_locked_audit"]["row_count"] == 120
    legacy = manifest["review_files"]["legacy_c24_locked_audit"]
    assert legacy["source_matches_c24_manifest"] is True
    assert legacy["source_sha256"] == legacy["c24_manifest_review_sha256"]
    assert manifest["review_files"]["public_train_v2"]["row_count"] == 72
    assert (
        manifest["review_files"]["legacy_c24_locked_audit"][
            "changes_historical_c24_files"
        ]
        is False
    )

    for split, expected_count in manifest["row_counts"].items():
        path = data_dir / f"{split}.jsonl"
        split_manifest_path = artifact_dir / f"{split}_manifest.json"
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        assert split_manifest["data_file"]["sha256"] == sha256_file(path)
        assert split_manifest["row_count"] == expected_count
        assert split_manifest["selection_protocol"]["development_used_for_selection"] is False
        assert split_manifest["selection_protocol"]["locked_audit_used_for_selection"] is False
        assert split_manifest["formal_locked_audit_executed"] is False

    for details in manifest["review_files"].values():
        review_path = Path(details["path"])
        with review_path.open(encoding="utf-8", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        assert tuple(review_rows[0]) == REVIEW_FIELDS
        assert len(review_rows) == details["row_count"]
        assert details["row_id_sha256"] == sha256_json(
            sorted(row["row_id"] for row in review_rows)
        )
        assert details["static_review_sha256"] == review_static_sha256(
            review_rows
        )
        assert {row["review_status"] for row in review_rows} == {"pending"}
        assert {row["reviewer_1_label"] for row in review_rows} == {""}
        assert {row["reviewer_2_label"] for row in review_rows} == {""}

    persisted = json.loads(
        (artifact_dir / "public_benchmark_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest
    assert all(not isinstance(value, float) or math.isfinite(value) for value in _walk(persisted))


def test_smoke_protocol_uses_only_synthetic_rows_and_skips_legacy_locked_read(tmp_path):
    rows, audit = _build(SMOKE_CONFIG_PATH)
    assert audit["passed"] is True
    assert {split: len(values) for split, values in rows.items()} == {
        "public_train_v2": 24,
        "public_calibration_v2": 10,
        "public_locked_audit_v2": 10,
    }
    assert all(
        row["relation_phrase"].startswith(("synthetic ", "mock ", "fixture "))
        for values in rows.values()
        for row in values
    )
    manifest = prepare_public_benchmark_v2(
        SMOKE_CONFIG_PATH,
        public_data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
    )
    assert "legacy_c24_locked_audit" not in manifest["review_files"]
    assert not (tmp_path / "artifacts" / "c24_locked_audit_independent_review.csv").exists()


def test_single_split_builder_does_not_materialize_locked_as_a_side_effect(monkeypatch):
    import keyed_gram.stage_c24b_benchmark as module

    observed = []
    original = module._materialize_split

    def spy(benchmark, split):
        observed.append(split.name.value)
        return original(benchmark, split)

    monkeypatch.setattr(module, "_materialize_split", spy)
    rows = build_public_split_v2(_config(SMOKE_CONFIG_PATH), PublicSplitV2.TRAIN)
    assert len(rows) == 24
    assert observed == [PublicSplitV2.TRAIN.value]


def test_strict_json_rejects_nan_and_infinity():
    with pytest.raises(ValueError):
        strict_json_dumps({"bad": float("nan")})
    with pytest.raises(ValueError):
        strict_json_dumps({"bad": float("inf")})
