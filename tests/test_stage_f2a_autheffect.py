from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from keyed_gram.stage_f2a_autheffect import run_smoke
from keyed_gram.stage_f2a_autheffect_protocol import (
    EXPECTED_ARTIFACT_FILES,
    F2AProtocolError,
    assert_no_sensitive_material,
    formal_preflight,
    load_config,
    read_json,
    runtime_source_manifest,
    safe_relative,
    validate_data_inventory,
    validate_strict_serialization,
    verify_frozen_f1,
)


def test_config_freezes_all_hard_boundaries() -> None:
    values = load_config("configs/stage_f2a.yaml")
    assert values["protocol"]["f1_locked_test_scoring_allowed"] is False
    assert values["protocol"]["private_data_allowed"] is False
    assert values["research_boundary"]["llm_judge_used"] is False
    assert values["protocol"]["formal_benchmark_seed_source"] == ("code_freeze_commit")
    assert values["protocol"]["preflight_fixture_is_formal_benchmark"] is False
    assert values["final_status"]["d_series_status"] == "completed"
    assert values["final_status"]["semantic_router_authorization_research"] == "stopped"
    assert values["final_status"]["c3_eligible"] is False


def test_f1_frozen_manifests_and_status_match() -> None:
    values = load_config("configs/stage_f2a.yaml")
    result = verify_frozen_f1("configs/stage_f2a.yaml", values)
    assert result["status"] == "passed"
    assert result["f1_frozen_assets_modified"] is False
    assert result["f1_locked_test_case_content_read"] is False


def test_runtime_source_manifest_is_complete() -> None:
    manifest = runtime_source_manifest("configs/stage_f2a.yaml")
    assert len(manifest["files"]) >= 20
    assert all(entry["sha256"] for entry in manifest["files"])


def test_artifact_inventory_is_preregistered_exactly() -> None:
    assert "stage_f2a_summary.json" in EXPECTED_ARTIFACT_FILES
    assert "split_manifests/locked_test.json" in EXPECTED_ARTIFACT_FILES
    assert "artifact_sha256_manifest.json" not in EXPECTED_ARTIFACT_FILES


def test_data_inventory_rejects_extra_file(tmp_path: Path) -> None:
    for split in ("train", "calibration", "development", "locked_test"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(F2AProtocolError, match="inventory"):
        validate_data_inventory(tmp_path, {})


def test_unsafe_relative_path_is_rejected() -> None:
    with pytest.raises(F2AProtocolError):
        safe_relative("../escape", "test")


def test_duplicate_json_key_and_nan_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(F2AProtocolError):
        read_json(duplicate)
    nan = tmp_path / "nan.json"
    nan.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(F2AProtocolError):
        read_json(nan)


def test_strict_jsonl_rejects_duplicate_key(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    data = tmp_path / "data"
    artifacts.mkdir()
    data.mkdir()
    (data / "train.jsonl").write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(F2AProtocolError, match="duplicate"):
        validate_strict_serialization(artifacts, data)


def test_enabling_private_data_is_rejected(tmp_path: Path) -> None:
    values = yaml.safe_load(Path("configs/stage_f2a.yaml").read_text())
    values["protocol"]["private_data_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(F2AProtocolError):
        load_config(path)


def test_dirty_worktree_rejects_formal_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import keyed_gram.stage_f2a_autheffect_protocol as protocol

    values = load_config("configs/stage_f2a.yaml")
    monkeypatch.setattr(
        protocol,
        "git_state",
        lambda _: {
            "commit": "0" * 40,
            "branch": "agent/stage-f2a-autheffect-feasibility",
            "tracked_dirty": True,
        },
    )
    monkeypatch.setattr(
        protocol,
        "output_paths",
        lambda *_: (
            tmp_path / "data",
            tmp_path / "artifacts",
            tmp_path / "runtime",
            tmp_path / "report",
        ),
    )
    with pytest.raises(F2AProtocolError, match="非干净"):
        formal_preflight("configs/stage_f2a.yaml", values)


def test_sensitive_material_scan_fails_closed(tmp_path: Path) -> None:
    clean = tmp_path / "clean.json"
    clean.write_text('{"private_data_used":false}', encoding="utf-8")
    assert (
        assert_no_sensitive_material((clean,))["sensitive_material_occurrence_count"]
        == 0
    )
    secret = tmp_path / "secret.txt"
    secret.write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
    with pytest.raises(F2AProtocolError):
        assert_no_sensitive_material((secret,))


def test_smoke_does_not_score_locked_test() -> None:
    result = run_smoke()
    assert result["status"] == "passed"
    assert result["evaluated_splits"] == ["train", "calibration"]
    assert result["locked_test_scored"] is False
    assert result["real_tool_execution"] is False
