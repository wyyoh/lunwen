from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import keyed_gram.stage_d21 as stage_d21
import keyed_gram.stage_d21_protocol as protocol
from keyed_gram.cli import build_parser
from keyed_gram.stage_d21 import (
    _run_attack_matrix,
    _summary,
    _write_csv,
    run_d21_smoke,
)
from keyed_gram.stage_d21_generator import FrozenMockCopyProbe


CONFIG = Path("configs/stage_d21.yaml")
SMOKE_CONFIG = Path("configs/stage_d21_smoke.yaml")


def _zero_scan(values):
    return {
        key: 0
        for key in values["required_zero_metrics"]
        if key.startswith("plaintext_")
    }


def _context_contract():
    return {
        "fresh_context_per_request": True,
        "conversation_history_size": 0,
        "persistent_kv_cache_size": 0,
        "use_cache": False,
        "tool_count": 0,
        "last_tool_call_count": 0,
        "last_context_destroyed": True,
        "last_context_contained_token_or_key": False,
    }


def _model_manifest():
    return {
        "manifest_payload_sha256": "1" * 64,
        "model_id": "mock-development-only",
        "revision": "mock",
        "parameter_count": 102714,
        "initial_parameter_sha256": (
            "0b25b10b57bb0b668893455cae1d2c460aae779960385f02ce78dea37b146d3d"
        ),
        "final_parameter_sha256": (
            "0b25b10b57bb0b668893455cae1d2c460aae779960385f02ce78dea37b146d3d"
        ),
        "all_parameters_frozen": True,
        "model_checkpoint_committed_to_repository": False,
    }


def test_formal_and_smoke_configs_validate() -> None:
    assert protocol.load_config(CONFIG)["stage"].startswith("D2.1")
    assert protocol.load_config(SMOKE_CONFIG, smoke=True)[
        "stage"
    ].endswith("-smoke")


def test_config_rejects_private_value_gate(tmp_path: Path) -> None:
    values = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    values["protocol"]["private_value_memory_allowed"] = True
    target = tmp_path / "stage_d21.yaml"
    target.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(protocol.D21ProtocolError):
        protocol.load_config(target)


def test_source_manifest_has_no_model_checkpoint() -> None:
    manifest = protocol.runtime_source_manifest(CONFIG)
    paths = {entry["path"] for entry in manifest["files"]}
    assert "src/keyed_gram/stage_d21.py" in paths
    assert not any(path.endswith((".bin", ".safetensors")) for path in paths)


def test_prepare_manifest_does_not_persist_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(protocol, "model_snapshot", lambda *_args, **_kwargs: tmp_path)
    result = protocol.prepare_model(CONFIG)
    assert result["model_checkpoint_committed_to_repository"] is False
    assert result["cache_path"] == str(tmp_path)
    payload = {
        key: value for key, value in result.items() if key != "cache_path"
    }
    assert "cache_path" not in payload


def test_phase_state_refuses_second_start(tmp_path: Path) -> None:
    protocol.mark_phase_started(tmp_path, "audit")
    with pytest.raises(protocol.D21ProtocolError):
        protocol.mark_phase_started(tmp_path, "audit")


def test_attack_matrix_and_summary_pass_with_mock() -> None:
    values = protocol.load_config(CONFIG)
    rows, concurrency, _, _, _ = _run_attack_matrix(
        values, g1=FrozenMockCopyProbe()
    )
    summary = _summary(
        values,
        rows,
        concurrency,
        git={"commit": "test", "branch": "test", "tracked_dirty": True},
        source_manifest={"manifest_payload_sha256": "0" * 64},
        model_manifest=_model_manifest(),
        context_contract=_context_contract(),
        scan_metrics=_zero_scan(values),
        parameter_hash_changed=False,
    )
    assert len(rows) == 40
    assert summary["metrics"]["passed_scenario_count"] == 40
    assert summary["capability_gated_ephemeral_generation_status"] == "passed"
    assert summary["c3_eligible"] is False


def test_summary_rejects_missing_preregistered_scenario() -> None:
    values = protocol.load_config(CONFIG)
    rows, concurrency, _, _, _ = _run_attack_matrix(
        values, g1=FrozenMockCopyProbe()
    )
    with pytest.raises(protocol.D21ProtocolError):
        _summary(
            values,
            rows[:-1],
            concurrency,
            git={"commit": "test"},
            source_manifest={"manifest_payload_sha256": "0" * 64},
            model_manifest=_model_manifest(),
            context_contract=_context_contract(),
            scan_metrics=_zero_scan(values),
            parameter_hash_changed=False,
        )


def test_summary_fails_when_parameter_hash_changes() -> None:
    values = protocol.load_config(CONFIG)
    rows, concurrency, _, _, _ = _run_attack_matrix(
        values, g1=FrozenMockCopyProbe()
    )
    summary = _summary(
        values,
        rows,
        concurrency,
        git={"commit": "test"},
        source_manifest={"manifest_payload_sha256": "0" * 64},
        model_manifest=_model_manifest(),
        context_contract=_context_contract(),
        scan_metrics=_zero_scan(values),
        parameter_hash_changed=True,
    )
    assert summary["capability_gated_ephemeral_generation_status"] == "failed"


def test_csv_rejects_plaintext_schema(tmp_path: Path) -> None:
    with pytest.raises(protocol.D21ProtocolError):
        _write_csv(
            tmp_path / "bad.csv",
            [{"scenario_id": "bad", "output_text": "not-allowed"}],
        )


def test_smoke_artifact_contains_no_runtime_canary(tmp_path: Path) -> None:
    result = run_d21_smoke(SMOKE_CONFIG, output_dir=tmp_path)
    assert result["g0_exact_delivery"] is True
    assert result["g1_mock_exact_delivery"] is True
    assert "SYN-D21-" not in (tmp_path / "smoke_summary.json").read_text(
        encoding="utf-8"
    )


def test_audit_refuses_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    runtime = tmp_path / "runtime"
    artifact.mkdir()
    monkeypatch.setattr(stage_d21, "load_config", lambda *_args: {})
    monkeypatch.setattr(
        stage_d21,
        "output_paths",
        lambda *_args: (artifact, runtime),
    )
    with pytest.raises(protocol.D21ProtocolError):
        stage_d21.run_d21_audit("unused.yaml")


@pytest.mark.parametrize(
    "command",
    [
        "stage-d21-prepare",
        "stage-d21-audit",
        "stage-d21-smoke",
        "stage-d21-finalize",
    ],
)
def test_cli_registers_d21_commands(command: str) -> None:
    parser = build_parser()
    arguments = [command, "--config", "config.yaml"]
    assert parser.parse_args(arguments).command == command
