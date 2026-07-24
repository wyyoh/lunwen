from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from keyed_gram.stage_c23_benchmark import (
    EXPECTED_BENCHMARK_TOTAL,
    ROW_FIELDS,
    audit_lexical_disjointness,
    build_public_lexical_benchmark,
    entity_masked,
    entity_masked_strip_suffix,
    lemma_bigrams,
    normalize_text,
    phrase_only,
    prepare_public_lexical_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage_c23.yaml"


def _benchmark_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["benchmark"]


def test_public_benchmark_has_strict_answer_free_balanced_schema():
    rows, audit = build_public_lexical_benchmark(_benchmark_config())

    assert {name: len(values) for name, values in rows.items()} == {
        "train": 54,
        "validation": 216,
        "reject": 144,
    }
    assert len(rows["validation"]) + len(rows["reject"]) == EXPECTED_BENCHMARK_TOTAL
    assert audit["lexical_disjointness"]["passed"] is True
    assert all(set(row) == ROW_FIELDS for values in rows.values() for row in values)
    assert all(
        row["answer_free"] is True
        and row["contains_private_answer"] is False
        and "answer" not in row
        and "candidates" not in row
        for values in rows.values()
        for row in values
    )

    validation = rows["validation"]
    assert set(Counter(row["family_id"] for row in validation).values()) == {6}
    assert set(Counter(row["frame_id"] for row in validation).values()) == {36}
    assert set(Counter(row["entity_id"] for row in validation).values()) == {36}
    assert Counter(row["attribute"] for row in validation) == Counter(
        {"registry_id": 72, "city_code": 72, "access_code": 72}
    )
    assert set(
        Counter(
            (row["attribute"], row["frame_id"]) for row in validation
        ).values()
    ) == {12}
    assert set(
        Counter(
            (row["attribute"], row["entity_id"]) for row in validation
        ).values()
    ) == {12}

    reject = rows["reject"]
    assert set(Counter(row["family_id"] for row in reject).values()) == {6}
    assert set(Counter(row["frame_id"] for row in reject).values()) == {24}
    assert set(Counter(row["entity_id"] for row in reject).values()) == {24}
    assert Counter(row["kind"] for row in reject) == Counter(
        {"ambiguous": 72, "unrelated": 72}
    )
    assert set(
        Counter((row["kind"], row["frame_id"]) for row in reject).values()
    ) == {12}
    assert all(row["attribute"] is None for row in reject)


def test_public_benchmark_input_views_and_unicode_normalization():
    prompt = "For North‑Lake, retrieve the access‑code. Answer:"
    assert phrase_only("  access‑code  ") == "access‐code"
    assert entity_masked(prompt, "North‑Lake") == (
        "For <ENTITY>, retrieve the access‐code. Answer:"
    )
    assert entity_masked_strip_suffix(prompt, "North‑Lake") == (
        "For <ENTITY>, retrieve the access‐code."
    )
    assert normalize_text("  Access‑Code! ") == "access code"
    with pytest.raises(ValueError, match="exactly once"):
        entity_masked("No subject appears here.", "missing")


@pytest.mark.parametrize(
    ("train", "validation", "collision_kind"),
    [
        (["Registry-ID"], ["registry id"], "exact"),
        (["official registry serial"], ["new official registry serial value"], "containment"),
        (
            ["official registries serials"],
            ["official registry serial"],
            "lemma_bigram",
        ),
    ],
)
def test_train_validation_lexical_collision_checks(
    train, validation, collision_kind
):
    with pytest.raises(ValueError, match=collision_kind):
        audit_lexical_disjointness(train, validation)

    clean = audit_lexical_disjointness(
        ["official registry serial"], ["password for a protected account"]
    )
    assert clean["collision_count"] == 0
    assert clean["passed"] is True
    assert ("official", "registry") in lemma_bigrams("Official registries")


def test_config_rejects_family_leakage_and_imbalance():
    config = _benchmark_config()
    first_validation_id = next(iter(config["validation_families"]["registry_id"]))
    config["validation_families"]["registry_id"][first_validation_id] = (
        "new registry number value"
    )
    with pytest.raises(ValueError, match="lexical collision"):
        build_public_lexical_benchmark(config)

    config = copy.deepcopy(_benchmark_config())
    config["validation_families"]["city_code"].pop(
        next(iter(config["validation_families"]["city_code"]))
    )
    with pytest.raises(ValueError, match="12 families"):
        build_public_lexical_benchmark(config)


def test_prepare_public_benchmark_writes_jsonl_and_lightweight_manifest(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_public_lexical_benchmark(
        CONFIG_PATH,
        public_data_dir=data_dir,
        manifest_path=manifest_path,
    )

    assert manifest["row_counts"] == {
        "train": 54,
        "validation": 216,
        "reject": 144,
    }
    assert manifest["benchmark_total"] == 360
    assert manifest["answer_free"] is True
    assert manifest["audit"]["lexical_disjointness"]["collision_count"] == 0
    assert manifest_path.exists()
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["benchmark_version"] == manifest["benchmark_version"]
    for split, count in manifest["row_counts"].items():
        path = data_dir / f"{split}.jsonl"
        assert path.exists()
        parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(parsed) == count
        assert all(row["answer_free"] is True for row in parsed)
