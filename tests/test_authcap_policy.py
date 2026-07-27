from __future__ import annotations

import json
from pathlib import Path

import pytest

from keyed_gram.authcap_benchmark import generate_authzroutebench
from keyed_gram.authcap_policy import (
    PolicySchemaError,
    canonical_sha256,
    load_policy_set,
    parse_policy_set,
    strict_load_document,
)


def sample_policy() -> dict:
    return generate_authzroutebench(splits=("train",))["train"][0]["policy_set"]


def test_canonical_hash_is_deterministic() -> None:
    value = sample_policy()
    assert canonical_sha256(value) == canonical_sha256(
        json.loads(json.dumps(value))
    )


def test_unknown_field_is_rejected() -> None:
    value = sample_policy()
    value["policies"][0]["unexpected"] = True
    with pytest.raises(PolicySchemaError):
        parse_policy_set(value)


def test_duplicate_policy_id_is_rejected() -> None:
    value = sample_policy()
    value["policies"].append(dict(value["policies"][0]))
    with pytest.raises(PolicySchemaError, match="duplicate policy ID"):
        parse_policy_set(value)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(PolicySchemaError, match="duplicate JSON key"):
        strict_load_document(path)


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(PolicySchemaError):
        strict_load_document(path)


def test_nan_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(PolicySchemaError):
        strict_load_document(path)


def test_strict_json_policy_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(sample_policy()), encoding="utf-8")
    loaded = load_policy_set(path)
    assert loaded.policy_epoch == 2
