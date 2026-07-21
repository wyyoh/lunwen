from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import yaml

from keyed_gram.stage_c24_benchmark import (
    LOCKED_REVIEW_STATUS,
    REVIEW_FIELDS,
    ROW_FIELDS,
    PublicSplit,
    audit_phrase_collections,
    build_locked_audit_review_rows,
    build_public_benchmark,
    normalized_token_signature,
    prepare_public_benchmark,
    reject_sensitive_benchmark_inputs,
    sha256_file,
    strict_json_dumps,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage_c24.yaml"


def _config():
    return copy.deepcopy(
        yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["benchmark"]
    )


def _walk(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def test_three_part_public_benchmark_is_strict_answer_free_and_group_isolated():
    rows, audit = build_public_benchmark(_config())

    assert {name: len(values) for name, values in rows.items()} == {
        "public_train": 54,
        "public_calibration": 90,
        "public_locked_audit": 120,
    }
    assert all(set(row) == ROW_FIELDS for values in rows.values() for row in values)
    assert all(
        row["answer_free"] is True and row["contains_private_answer"] is False
        for values in rows.values()
        for row in values
    )
    assert {row["kind"] for row in rows["public_train"]} == {"known"}
    for split in ("public_calibration", "public_locked_audit"):
        assert Counter(row["kind"] for row in rows[split]) == Counter(
            {"known": 18 * (3 if split == "public_calibration" else 4),
             "ambiguous": 6 * (3 if split == "public_calibration" else 4),
             "unrelated": 6 * (3 if split == "public_calibration" else 4)}
        )

    assert audit["answer_free"] is True
    assert audit["public_benchmark_human_reviewed"] is False
    assert audit["split_isolation"]["passed"] is True
    assert (
        "ordinary partial token reuse"
        in audit["split_isolation"]["normalized_token_collision_definition"]
    )
    for pair in audit["split_isolation"]["pairs"].values():
        assert pair["passed"] is True
        assert pair["phrase_family_id"]["collision_count"] == 0
        assert pair["entity"]["collision_count"] == 0
        assert pair["frame"]["collision_count"] == 0
        assert pair["phrases"]["collision_count"] == 0

    family_sets = [
        {row["family_id"] for row in rows[split.value]}
        for split in PublicSplit
    ]
    entity_sets = [
        {row["entity_id"] for row in rows[split.value]}
        for split in PublicSplit
    ]
    frame_sets = [
        {row["frame_id"] for row in rows[split.value]}
        for split in PublicSplit
    ]
    for collections in (family_sets, entity_sets, frame_sets):
        assert collections[0].isdisjoint(collections[1])
        assert collections[0].isdisjoint(collections[2])
        assert collections[1].isdisjoint(collections[2])


@pytest.mark.parametrize(
    ("left", "right", "kind"),
    [
        ("Registry Number", "registry number", "exact"),
        ("official registry serial", "new official registry serial value", "substring_containment"),
        ("registry-number", "registry number", "normalized_token_signature"),
        ("official registries serials", "official registry serial", "lemma_bigram"),
    ],
)
def test_phrase_collision_audit_covers_all_preregistered_levels(left, right, kind):
    audit = audit_phrase_collections(
        {"left": left},
        {"right": right},
        left_name="left",
        right_name="right",
        lemma_bigram_threshold=0.8,
    )
    assert audit["passed"] is False
    assert audit["collision_count"] == 1
    assert audit["collisions"][kind]

    assert normalized_token_signature(" Registry-number! ") == (
        "registry",
        "number",
    )


def test_config_fails_closed_on_family_entity_and_frame_reuse():
    for unit in ("known_families", "entities", "frames"):
        config = _config()
        train = config["splits"]["public_train"]
        locked = config["splits"]["public_locked_audit"]
        if unit == "known_families":
            family_id, phrase = next(iter(train[unit]["registry_id"].items()))
            locked[unit]["registry_id"].pop(
                next(iter(locked[unit]["registry_id"]))
            )
            locked[unit]["registry_id"][family_id] = phrase
        else:
            locked[unit][0] = train[unit][0]
        with pytest.raises(ValueError, match="split isolation collision"):
            build_public_benchmark(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update({"private_answer": "do-not-load"}),
        lambda c: c.update({"answers": ["do-not-load"]}),
        lambda c: c.update({"confirmation_path": "elsewhere.json"}),
        lambda c: c.update({"unexpected_path": "artifacts/old-seal/data.json"}),
    ],
)
def test_answer_bearing_and_confirmation_or_seal_inputs_are_rejected(mutation):
    config = _config()
    mutation(config)
    with pytest.raises(ValueError, match="answer-bearing|confirmation/seal"):
        reject_sensitive_benchmark_inputs(config)
    with pytest.raises(ValueError, match="answer-bearing|confirmation/seal"):
        build_public_benchmark(config)


def test_strict_config_rejects_unknown_fields_and_non_pending_review_status():
    config = _config()
    config["unexpected"] = "value"
    with pytest.raises(ValueError, match="unknown benchmark fields"):
        build_public_benchmark(config)

    config = _config()
    config["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="curated_draft_pending"):
        build_public_benchmark(config)

    config = _config()
    config["splits"]["public_train"]["rejection_families"] = {
        "ambiguous": {"x": "unclear thing"},
        "unrelated": {"y": "weather forecast"},
    }
    with pytest.raises(ValueError, match="known queries only"):
        build_public_benchmark(config)


def test_locked_review_template_is_entirely_pending_and_never_fakes_labels():
    rows, _ = build_public_benchmark(_config())
    review = build_locked_audit_review_rows(rows["public_locked_audit"])

    assert len(review) == 120
    assert all(tuple(row) == REVIEW_FIELDS for row in review)
    assert all(row["review_status"] == "pending" for row in review)
    assert all(row["reviewer_1_label"] == "" for row in review)
    assert all(row["reviewer_2_label"] == "" for row in review)
    assert all(row["adjudicated_label"] == "" for row in review)
    assert all(row["notes"] == "" for row in review)
    assert {row["ambiguous_flag"] for row in review} == {"true", "false"}
    with pytest.raises(ValueError, match="only be created"):
        build_locked_audit_review_rows(rows["public_calibration"][:1])


def test_prepare_writes_finite_jsonl_manifest_hashes_and_pending_review_csv(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    review_path = tmp_path / "review.csv"
    manifest = prepare_public_benchmark(
        CONFIG_PATH,
        public_data_dir=data_dir,
        manifest_path=manifest_path,
        review_csv_path=review_path,
    )

    assert manifest["review_status"] == LOCKED_REVIEW_STATUS
    assert manifest["public_benchmark_human_reviewed"] is False
    assert manifest["provisional_locked_audit"] is True
    assert manifest["row_counts"] == {
        "public_train": 54,
        "public_calibration": 90,
        "public_locked_audit": 120,
    }
    assert manifest["human_review_file"]["row_count"] == 120
    assert manifest["human_review_file"]["all_rows_pending"] is True
    assert manifest["human_review_file"]["sha256"] == sha256_file(review_path)

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == manifest
    assert all(not isinstance(value, float) or math.isfinite(value) for value in _walk(persisted))
    for split, details in manifest["files"].items():
        path = data_dir / f"{split}.jsonl"
        assert details["sha256"] == sha256_file(path)
        parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(parsed) == manifest["row_counts"][split]

    with review_path.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert tuple(review_rows[0]) == REVIEW_FIELDS
    assert {row["review_status"] for row in review_rows} == {"pending"}
    assert {row["reviewer_1_label"] for row in review_rows} == {""}
    assert {row["reviewer_2_label"] for row in review_rows} == {""}
    assert {row["adjudicated_label"] for row in review_rows} == {""}

    with pytest.raises(ValueError):
        strict_json_dumps({"bad": float("nan")})
