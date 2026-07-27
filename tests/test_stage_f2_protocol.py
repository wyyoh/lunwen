from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import keyed_gram.stage_f2_protocol as protocol
from keyed_gram.stage_f2_protocol import (
    F2ProtocolError,
    artifact_manifest,
    claim_phase,
    complete_phase,
    formal_preflight,
    initialize_protocol,
    load_config,
    read_json,
    safe_relative,
    validate_final_protocol,
    verify_f1_frozen,
    write_json,
)


def test_formal_config_fixes_variants_and_permanent_boundaries():
    value = load_config("configs/stage_f2.yaml")
    assert value["variants"] == ["B0", "B1", "B2", "B3", "A0", "A1"]
    assert value["protocol"]["final_variant"] == "A1"
    assert value["protocol"]["private_data_allowed"] is False
    assert value["final_status"]["c3_eligible"] is False


def test_safe_relative_rejects_absolute_and_parent_paths():
    with pytest.raises(F2ProtocolError):
        safe_relative("/tmp/result", "test")
    with pytest.raises(F2ProtocolError):
        safe_relative("../result", "test")


def test_strict_json_rejects_duplicate_key_and_nan(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    nan = tmp_path / "nan.json"
    nan.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(F2ProtocolError):
        read_json(duplicate)
    with pytest.raises(F2ProtocolError):
        read_json(nan)


def test_config_rejects_private_source_reference(tmp_path):
    value = yaml.safe_load(Path("configs/stage_f2.yaml").read_text())
    value["f1_frozen"]["summary"]["path"] = "private_answer.json"
    path = tmp_path / "stage_f2.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(F2ProtocolError):
        load_config(path)


def test_phase_order_development_and_locked_exactly_once(tmp_path):
    runtime = tmp_path / "runtime"
    initialize_protocol(
        runtime,
        git={"commit": "a" * 40, "branch": "agent/stage-f2-authcap-compiler"},
    )
    previous = "initialized"
    for phase in ("prepare", "calibration", "freeze", "development", "locked_test"):
        claim_phase(runtime, phase=phase, expected_previous=previous)
        complete_phase(runtime, phase=phase)
        previous = phase
    state = validate_final_protocol(runtime)
    assert state["development_score_count"] == 1
    assert state["locked_test_score_count"] == 1
    with pytest.raises(F2ProtocolError):
        claim_phase(runtime, phase="locked_test", expected_previous="development")


def test_locked_before_freeze_is_rejected(tmp_path):
    runtime = tmp_path / "runtime"
    initialize_protocol(
        runtime,
        git={"commit": "a" * 40, "branch": "agent/stage-f2-authcap-compiler"},
    )
    with pytest.raises(F2ProtocolError):
        claim_phase(runtime, phase="locked_test", expected_previous="development")


def test_dirty_worktree_rejects_formal_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(
        protocol,
        "git_state",
        lambda _: {
            "commit": "a" * 40,
            "branch": "agent/stage-f2-authcap-compiler",
            "tracked_dirty": True,
        },
    )
    values = load_config("configs/stage_f2.yaml")
    with pytest.raises(F2ProtocolError):
        formal_preflight("configs/stage_f2.yaml", values)


def test_f1_frozen_hash_and_base_tree_validate():
    values = load_config("configs/stage_f2.yaml")
    result = verify_f1_frozen("configs/stage_f2.yaml", values)
    assert result["status"] == "passed"
    assert len(result["split_sha256"]) == 4


def test_f1_manifest_mismatch_is_rejected():
    values = load_config("configs/stage_f2.yaml")
    values["f1_frozen"]["summary"]["sha256"] = "0" * 64
    with pytest.raises(F2ProtocolError):
        verify_f1_frozen("configs/stage_f2.yaml", values)


def test_artifact_inventory_must_be_exact(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    write_json(artifacts / "a.json", {"schema_version": 1})
    report = tmp_path / "report.md"
    report.write_text("report", encoding="utf-8")
    manifest = artifact_manifest(
        artifacts,
        report_path=report,
        expected_files=("a.json",),
    )
    assert manifest["private_key_present"] is False
    write_json(artifacts / "extra.json", {})
    with pytest.raises(F2ProtocolError):
        artifact_manifest(
            artifacts,
            report_path=report,
            expected_files=("a.json",),
        )


def test_write_json_rejects_non_finite(tmp_path):
    with pytest.raises(F2ProtocolError):
        write_json(tmp_path / "bad.json", {"value": float("inf")})
