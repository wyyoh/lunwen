from __future__ import annotations

import copy

import pytest

from keyed_gram.stage_c22 import (
    DEFAULT_PUBLIC_VARIANTS,
    build_public_relation_rows,
    parse_public_variants,
)


def _corpus_config():
    return {
        "entities": {
            "train": ["a", "b", "c", "d"],
            "validation": ["e", "f", "g", "h"],
        },
        "frames": {
            "train": [
                "Field {relation_phrase} for {entity}:",
                "Public {entity} asks for {relation_phrase}:",
            ],
            "validation": [
                "Read {entity}'s {relation_phrase}:",
                "Which {relation_phrase} maps to {entity}?",
            ],
        },
        "phrases": {
            "train": {
                "registry_id": ["member number", "filing key"],
                "city_code": ["town number", "district key"],
                "access_code": ["security key", "login number"],
            },
            "validation": {
                "registry_id": ["catalog marker", "ledger identifier"],
                "city_code": ["regional marker", "civic identifier"],
                "access_code": ["protective marker", "safety identifier"],
            },
        },
        "forbidden_phrases": {
            "registry_id": ["registry number"],
            "city_code": ["city code"],
            "access_code": ["access code"],
        },
    }


def test_public_relation_rows_are_answer_free_and_balanced():
    rows = build_public_relation_rows(_corpus_config(), "train")
    assert len(rows) == 4 * 3 * 2 * 2
    assert {row["attribute"] for row in rows} == {
        "registry_id",
        "city_code",
        "access_code",
    }
    assert all(row["contains_private_answer"] is False for row in rows)
    assert all("answer" not in row and "candidates" not in row for row in rows)
    assert len({row["template_id"] for row in rows}) == 4


def test_public_relation_rows_reject_reserved_or_reused_phrases():
    config = _corpus_config()
    config["phrases"]["train"]["city_code"][0] = "city code"
    with pytest.raises(ValueError, match="reserved"):
        build_public_relation_rows(config, "train")

    config = copy.deepcopy(_corpus_config())
    config["phrases"]["validation"]["registry_id"][0] = "member number"
    with pytest.raises(ValueError, match="reused"):
        build_public_relation_rows(config, "validation")


def test_public_variants_keep_exact_base_and_replay_ablation():
    variants = parse_public_variants(None)
    assert variants == DEFAULT_PUBLIC_VARIANTS
    assert variants[0].import_base is True
    assert variants[1].private_replay_weight == 0.0
    assert variants[2].private_replay_weight == 1.0
