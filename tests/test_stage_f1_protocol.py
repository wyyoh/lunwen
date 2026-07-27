from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from keyed_gram.stage_f1 import run_stage_f1_smoke
from keyed_gram.stage_f1_protocol import (
    F1ProtocolError,
    formal_preflight,
    load_baseline_split,
    load_config,
    read_json,
    validate_artifact_inventory,
    verify_frozen_upstream,
)


def test_formal_config_preserves_all_hard_gates() -> None:
    values = load_config("configs/stage_f1.yaml")
    assert values["protocol"]["locked_test_scoring_allowed"] is False
    assert values["protocol"]["private_data_allowed"] is False
    assert values["final_status"]["c3_eligible"] is False


def test_upstream_frozen_hashes_match() -> None:
    values = load_config("configs/stage_f1.yaml")
    manifest = verify_frozen_upstream("configs/stage_f1.yaml", values)
    assert manifest["status"] == "passed"
    assert manifest["frozen_upstream_code_modified"] is False


def test_private_data_gate_cannot_be_enabled(tmp_path: Path) -> None:
    values = yaml.safe_load(Path("configs/stage_f1.yaml").read_text())
    values["protocol"]["private_data_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(F1ProtocolError):
        load_config(path)


def test_private_source_reference_is_rejected(tmp_path: Path) -> None:
    values = yaml.safe_load(Path("configs/stage_f1.yaml").read_text())
    values["frozen_upstream"][0]["path"] = "artifacts/private_answer.json"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(F1ProtocolError, match="private/secret"):
        load_config(path)


def test_unsafe_output_path_is_rejected(tmp_path: Path) -> None:
    values = yaml.safe_load(Path("configs/stage_f1.yaml").read_text())
    values["outputs"]["data_dir"] = "../escape"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(F1ProtocolError):
        load_config(path)


def test_locked_test_baseline_loader_is_closed(tmp_path: Path) -> None:
    with pytest.raises(F1ProtocolError, match="禁止打开 locked_test"):
        load_baseline_split(tmp_path, "locked_test")


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(F1ProtocolError):
        read_json(path)


def test_dirty_worktree_rejects_formal_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import keyed_gram.stage_f1_protocol as protocol

    values = load_config("configs/stage_f1.yaml")
    monkeypatch.setattr(
        protocol,
        "git_state",
        lambda _: {
            "commit": "0" * 40,
            "branch": "agent/stage-f1-authzroutebench",
            "tracked_dirty": True,
        },
    )
    monkeypatch.setattr(
        protocol,
        "output_paths",
        lambda *_: (
            tmp_path / "data",
            tmp_path / "artifact",
            tmp_path / "runtime",
            tmp_path / "report.md",
        ),
    )
    with pytest.raises(F1ProtocolError, match="非干净"):
        formal_preflight("configs/stage_f1.yaml", values)


def test_existing_output_rejects_repeat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import keyed_gram.stage_f1_protocol as protocol

    values = load_config("configs/stage_f1.yaml")
    existing = tmp_path / "artifact"
    existing.mkdir()
    monkeypatch.setattr(
        protocol,
        "git_state",
        lambda _: {
            "commit": "0" * 40,
            "branch": "agent/stage-f1-authzroutebench",
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        protocol,
        "output_paths",
        lambda *_: (
            tmp_path / "data",
            existing,
            tmp_path / "runtime",
            tmp_path / "report.md",
        ),
    )
    with pytest.raises(F1ProtocolError, match="拒绝重复构建"):
        formal_preflight("configs/stage_f1.yaml", values)


def test_smoke_never_generates_locked_test(tmp_path: Path) -> None:
    result = run_stage_f1_smoke(
        "configs/stage_f1_smoke.yaml",
        output_dir=tmp_path,
    )
    assert result["status"] == "passed"
    assert result["locked_test_generated"] is False
    assert result["locked_test_scored"] is False


def test_artifact_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    report = tmp_path / "report.md"
    report.write_text("report", encoding="utf-8")
    (artifact / "result.json").write_text("{}\n", encoding="utf-8")
    (artifact / "artifact_sha256_manifest.json").write_text(
        '{"schema_version":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(F1ProtocolError, match="inventory/hash mismatch"):
        validate_artifact_inventory(artifact, report)
