from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from keyed_gram.cli import build_parser
from keyed_gram.stage_d2 import run_d2_smoke
from keyed_gram.stage_d2_protocol import (
    D2ProtocolError,
    load_config,
    runtime_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage_d2.yaml"
SMOKE_CONFIG = ROOT / "configs" / "stage_d2_smoke.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_inherits_passed_d1_without_unlocking_private_or_c3() -> None:
    config = load_config(CONFIG)
    governance = config["governance"]
    protocol = config["protocol"]
    assert governance["trusted_capability_contract_status"] == "passed"
    assert governance["ready_for_stage_d2_public_synthetic_values"] is True
    assert governance["semantic_router_in_tcb"] is False
    for key in (
        "private_value_memory_allowed",
        "private_answers_allowed",
        "lm_answer_injection_allowed",
        "natural_language_router_allowed",
        "confirmation_allowed",
        "key_attack_allowed",
        "original_c3_allowed",
        "c3_eligible",
    ):
        assert protocol[key] is False


def test_config_freezes_principal_and_key_separation_boundaries() -> None:
    config = load_config(CONFIG)
    trust = config["trust_boundary"]
    memory = config["data_memory"]
    assert trust["identity_mode"] == "trusted_in_process_mock"
    assert trust["caller_supplied_principal_allowed"] is False
    assert trust["principal_seal_required"] is True
    assert trust["process_isolation_implemented"] is False
    assert trust["distributed_replay_store_implemented"] is False
    assert memory["algorithm"] == "AES-256-GCM"
    assert memory["key_bytes"] == 32
    assert memory["nonce_bytes"] == 12
    assert memory["tag_bytes"] == 16
    assert memory["independent_from_capability_key"] is True


def test_aad_field_order_is_frozen() -> None:
    assert load_config(CONFIG)["data_memory"]["aad_fields"] == [
        "schema_version",
        "algorithm",
        "record_id",
        "entity_id",
        "relation_id",
        "data_key_id",
        "record_version",
    ]


def test_d1_artifacts_and_sources_remain_byte_identical() -> None:
    config = load_config(CONFIG)
    for name in ("d1_summary", "d1_artifact_manifest"):
        declaration = config["governance"][name]
        assert _sha256(ROOT / declaration["path"]) == declaration["sha256"]
    d1_manifest = json.loads(
        (ROOT / "artifacts/stage_d1/source_sha256_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    audited_runtime_extensions = {".gitattributes", "src/keyed_gram/cli.py"}
    for declaration in d1_manifest["files"]:
        if declaration["path"] in audited_runtime_extensions:
            continue
        assert _sha256(ROOT / declaration["path"]) == declaration["sha256"]


def test_runtime_source_manifest_includes_d1_boundary_and_d2_sources() -> None:
    manifest = runtime_source_manifest(CONFIG)
    paths = {entry["path"] for entry in manifest["files"]}
    assert {
        "artifacts/stage_d1/stage_d1_summary.json",
        "src/keyed_gram/stage_c24_contract.py",
        "src/keyed_gram/stage_d1_gateway.py",
        "src/keyed_gram/stage_d2_contract.py",
        "src/keyed_gram/stage_d2_crypto.py",
        "src/keyed_gram/stage_d2_memory.py",
    } <= paths
    assert manifest["cryptography_version"] == "49.0.0"


def test_cli_exposes_d2_audit_smoke_and_finalize() -> None:
    parser = build_parser()
    for command, config in (
        ("stage-d2-audit", CONFIG),
        ("stage-d2-smoke", SMOKE_CONFIG),
        ("stage-d2-finalize", CONFIG),
    ):
        parsed = parser.parse_args([command, "--config", str(config)])
        assert parsed.command == command


def test_smoke_releases_once_and_replay_releases_nothing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke"
    result = run_d2_smoke(SMOKE_CONFIG, output_dir=output)
    persisted = json.loads(
        (output / "smoke_summary.json").read_text(encoding="utf-8")
    )
    assert result == persisted
    assert result["authorized_release_count"] == 1
    assert result["replay_reason"] == "replay"
    assert result["replay_plaintext_release_count"] == 0
    assert result["plaintext_persisted"] is False
    assert result["private_value_used"] is False
    assert result["c3_eligible"] is False


def test_config_rejects_wrong_cryptography_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    values = CONFIG.read_text(encoding="utf-8").replace(
        "implementation_version: 49.0.0",
        "implementation_version: 0.0.0",
    )
    path = tmp_path / "stage_d2.yaml"
    path.write_text(values, encoding="utf-8")
    monkeypatch.setattr(
        "keyed_gram.stage_d2_protocol.repo_root",
        lambda _: ROOT,
    )
    with pytest.raises(D2ProtocolError, match="冻结声明"):
        load_config(path)


def test_pyproject_pins_cryptography() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"cryptography==49.0.0"' in text


def test_output_schema_declares_no_plaintext_or_secret_files() -> None:
    config = load_config(CONFIG)
    assert set(config["outputs"]) == {"artifact_dir", "runtime_dir"}
    serialized = json.dumps(config["outputs"]).lower()
    for forbidden in (
        "plaintext",
        "private",
        "token",
        "key",
        "credential",
    ):
        assert forbidden not in serialized
