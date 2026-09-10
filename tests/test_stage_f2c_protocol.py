from __future__ import annotations

import json
from pathlib import Path

import pytest

from keyed_gram.authsynth_symbolic_shared import GrammarLimits
from keyed_gram.authsynth_symbolic_verifier.benchmark import generate_authsymbolbench
from keyed_gram.stage_f2c_protocol import (
    EXPECTED_ARTIFACT_FILES,
    F2CProtocolError,
    artifact_manifest,
    formal_preflight,
    load_config,
    mark_started,
    read_json,
    record_phase,
    safe_relative,
    validate_artifact_inventory,
    write_json,
)


def test_locked_materialization_requires_real_analyzer_freeze_sha() -> None:
    with pytest.raises(ValueError, match="freeze"):
        generate_authsymbolbench(
            "test",
            analyzer_freeze_commit="not-frozen",
            grammar=GrammarLimits(),
            replay_budget=16,
            solver_timeout_ms=3000,
        )


def test_phase_order_and_repeat_are_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    mark_started(runtime, {"commit": "a" * 40, "branch": "test"})
    with pytest.raises(F2CProtocolError, match="前序"):
        record_phase(runtime, "development")
    record_phase(runtime, "materialize")
    with pytest.raises(F2CProtocolError, match="重复"):
        record_phase(runtime, "materialize")


def test_strict_json_rejects_duplicate_key_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    with pytest.raises(F2CProtocolError, match="duplicate"):
        read_json(duplicate)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(F2CProtocolError, match="非法常量"):
        read_json(nonfinite)


def test_safe_relative_rejects_escape_and_absolute_paths() -> None:
    with pytest.raises(F2CProtocolError):
        safe_relative("../escape", "test")
    with pytest.raises(F2CProtocolError):
        safe_relative("/absolute", "test")


@pytest.mark.parametrize(
    "path",
    [
        "C:/escape",
        "C:relative",
        "\\\\server\\share",
        "\\escape",
        "x:stream",
        "",
        "a/../b",
        "./file",
        "a//b",
    ],
)
def test_safe_relative_is_independent_of_host_os(path: str) -> None:
    with pytest.raises(F2CProtocolError):
        safe_relative(path, "test")


def test_config_preserves_research_boundaries() -> None:
    values = load_config("configs/stage_f2c.yaml")
    assert values["research_boundary"]["private_data_used"] is False
    assert values["protocol"]["f2b_locked_test_rerun_allowed"] is False


def test_dirty_worktree_is_rejected_before_formal_outputs(monkeypatch) -> None:
    import keyed_gram.stage_f2c_protocol as protocol

    values = load_config("configs/stage_f2c.yaml")
    monkeypatch.setattr(
        protocol,
        "git_state",
        lambda _path: {
            "commit": "a" * 40,
            "branch": values["protocol"]["required_branch"],
            "tracked_dirty": True,
        },
    )
    with pytest.raises(F2CProtocolError, match="非干净"):
        formal_preflight("configs/stage_f2c.yaml", values)


def test_exact_artifact_inventory_and_hash_validation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    for relative in EXPECTED_ARTIFACT_FILES:
        path = artifact / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "{}\n" if path.suffix == ".json" else "x\n1\n", encoding="utf-8"
        )
    report = tmp_path / "report.md"
    report.write_text("# report\n", encoding="utf-8")
    write_json(
        artifact / "artifact_sha256_manifest.json", artifact_manifest(artifact, report)
    )
    validate_artifact_inventory(artifact, report)
    extra = artifact / "extra.json"
    extra.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(F2CProtocolError, match="inventory"):
        validate_artifact_inventory(artifact, report)
