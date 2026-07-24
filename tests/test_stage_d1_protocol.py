from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from keyed_gram.cli import build_parser
from keyed_gram.stage_d1 import (
    _run_attack_matrix,
    _summary,
    run_d1_smoke,
)
from keyed_gram.stage_d1_protocol import (
    D1ProtocolError,
    load_config,
    runtime_source_manifest,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage_d1.yaml"
SMOKE_CONFIG = ROOT / "configs" / "stage_d1_smoke.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_freezes_router_stop_and_private_boundaries() -> None:
    config = load_config(CONFIG)
    governance = config["governance"]
    protocol = config["protocol"]
    assert (
        governance["selective_router_research_status"]
        == "stopped_external_validation_failed"
    )
    assert governance["additional_router_complexity_allowed"] is False
    assert governance["semantic_router_in_tcb"] is False
    assert governance["suggested_relation_authorizes_memory"] is False
    trust = config["trust_boundary"]
    assert "natural_language_router" in trust["untrusted_components"]
    assert "relation_proposal" in trust["untrusted_components"]
    assert "trusted_capability_gateway" in trust["trusted_components"]
    assert "typed_keyed_memory" in trust["trusted_components"]
    assert trust["subject_identity_provider_implemented"] is False
    assert trust["process_isolation_implemented"] is False
    assert trust["verifier_can_mint_with_symmetric_key"] is True
    for key in (
        "private_value_memory_allowed",
        "private_answers_allowed",
        "lm_answer_injection_allowed",
        "key_attack_allowed",
        "confirmation_allowed",
        "original_c3_allowed",
        "c3_eligible",
    ):
        assert protocol[key] is False


def test_config_uses_standard_hmac_and_explicit_single_use_scope() -> None:
    capability = load_config(CONFIG)["capability"]
    assert capability["algorithm"] == "HMAC-SHA-256"
    assert capability["standard_reference"] == "RFC 2104"
    assert capability["key_bytes"] >= 32
    assert capability["single_use"] is True
    assert capability["entity_scope_mode"] == "explicit_exact_ids_only"
    assert capability["wildcard_scope_allowed"] is False


def test_attack_matrix_covers_all_preregistered_scenarios() -> None:
    config = load_config(CONFIG)
    rows = _run_attack_matrix(config)
    observed = {row["scenario_id"] for row in rows}
    expected = set(config["required_positive_scenarios"]) | set(
        config["required_negative_scenarios"]
    )
    assert observed == expected
    assert len(rows) == 25
    assert all(row["passed"] for row in rows)


def test_attack_matrix_has_zero_unauthorized_memory_access() -> None:
    config = load_config(CONFIG)
    rows = _run_attack_matrix(config)
    source = runtime_source_manifest(CONFIG)
    summary = _summary(
        config,
        rows,
        git={"commit": "test", "branch": "test", "tracked_dirty": False},
        source_manifest=source,
    )
    metrics = summary["metrics"]
    assert metrics["positive_scenario_count"] == 5
    assert metrics["negative_scenario_count"] == 20
    assert metrics["passed_scenario_count"] == 25
    assert metrics["unauthorized_memory_access_count"] == 0
    assert metrics["cross_relation_access_count"] == 0
    assert metrics["cross_entity_access_count"] == 0
    assert metrics["rejected_request_memory_access_count"] == 0
    assert summary["trusted_capability_contract_status"] == "passed"
    assert summary["ready_for_stage_d2_public_synthetic_values"] is True
    assert summary["c3_eligible"] is False


def test_summary_preserves_finite_and_limited_claims() -> None:
    config = load_config(CONFIG)
    summary = _summary(
        config,
        _run_attack_matrix(config),
        git={"commit": "test", "branch": "test", "tracked_dirty": False},
        source_manifest=runtime_source_manifest(CONFIG),
    )
    encoded = json.dumps(summary, allow_nan=False)
    assert "NaN" not in encoded
    assert summary["private_value_memory_trained"] is False
    assert summary["private_answers_loaded"] is False
    assert summary["lm_answer_injection_executed"] is False
    assert summary["key_attack_executed"] is False
    assert summary["confirmation_created_or_read"] is False
    assert summary["token_material_persisted"] is False
    assert summary["hmac_key_material_persisted"] is False
    assert any("不提供机密性" in item for item in summary["limitations"])
    assert any("不是部署级安全认证" in item for item in summary["limitations"])


def test_strict_json_writer_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(D1ProtocolError, match="NaN/Infinity"):
        write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_runtime_source_manifest_binds_frozen_c24_contract() -> None:
    manifest = runtime_source_manifest(CONFIG)
    declarations = {entry["path"]: entry for entry in manifest["files"]}
    contract = "src/keyed_gram/stage_c24_contract.py"
    assert declarations[contract]["sha256"] == (
        "c8a3b5fd33621349659e504b1bd31502688896cf63207ae59f644849a0d96692"
    )
    assert _sha256(ROOT / contract) == declarations[contract]["sha256"]


def test_smoke_is_ephemeral_and_never_claims_c3(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    result = run_d1_smoke(SMOKE_CONFIG, output_dir=output)
    persisted = json.loads(
        (output / "smoke_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == result
    assert result["accepted_memory_access_count"] == 1
    assert result["rejected_request_memory_access_count"] == 0
    assert result["private_value_memory_used"] is False
    assert result["key_material_persisted"] is False
    assert result["c3_eligible"] is False


def test_cli_exposes_d1_commands() -> None:
    parser = build_parser()
    for command, config in (
        ("stage-d1-audit", CONFIG),
        ("stage-d1-smoke", SMOKE_CONFIG),
        ("stage-d1-finalize", CONFIG),
    ):
        parsed = parser.parse_args(
            [command, "--config", str(config)]
        )
        assert parsed.command == command


def test_required_negative_matrix_contains_router_and_scope_attacks() -> None:
    required = set(load_config(CONFIG)["required_negative_scenarios"])
    assert {
        "unauthorized_router_proposal",
        "wrong_subject",
        "wrong_entity",
        "wrong_relation",
        "cross_entity_use",
        "cross_relation_use",
        "expired_token",
        "revoked_token",
        "replay_second_use",
        "algorithm_confusion",
    } <= required


def test_no_dataset_or_secret_material_is_declared_as_output() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["private_answers_allowed"] is False
    assert config["protocol"]["token_material_persisted"] is False
    assert config["protocol"]["hmac_key_material_persisted"] is False
    outputs = config["outputs"]
    assert set(outputs) == {"artifact_dir", "runtime_dir"}
    serialized_outputs = json.dumps(outputs, ensure_ascii=False).lower()
    assert "private_answer" not in serialized_outputs
    assert "capability_token" not in serialized_outputs
    assert "hmac_key" not in serialized_outputs


def test_attack_matrix_csv_schema_contains_no_token_or_key(tmp_path: Path) -> None:
    rows = _run_attack_matrix(load_config(CONFIG))
    path = tmp_path / "matrix.csv"
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    header = next(csv.reader(path.open(encoding="utf-8")))
    assert "capability_token" not in header
    assert "hmac_key" not in header
    assert "private_answer" not in header
