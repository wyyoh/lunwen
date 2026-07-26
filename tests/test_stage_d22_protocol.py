from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from keyed_gram.cli import build_parser
from keyed_gram.stage_d22 import run_d22_smoke
from keyed_gram.stage_d22_protocol import (
    git_state,
    load_config,
    runtime_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_formal_protocol_freezes_35_unique_scenarios() -> None:
    values = load_config(ROOT / "configs/stage_d22.yaml")
    names = [
        name
        for field in (
            "required_positive_scenarios",
            "required_authorization_negative_scenarios",
            "required_service_boundary_scenarios",
            "required_ipc_minimization_scenarios",
            "required_crash_scenarios",
            "required_multi_instance_scenarios",
        )
        for name in values[field]
    ]
    assert len(names) == len(set(names)) == 35


def test_canary_schema_uses_four_unambiguous_counts() -> None:
    canary = load_config(ROOT / "configs/stage_d22.yaml")["canary"]
    assert {
        canary["persisted_output_digest_count_field"],
        canary["runtime_canary_count_field"],
        canary["runtime_canary_variant_count_field"],
        canary["scanned_location_count_field"],
    } == {
        "persisted_output_digest_count",
        "runtime_canary_count",
        "runtime_canary_variant_count",
        "scanned_location_count",
    }


def test_source_manifest_preserves_upstream_and_d22_sources() -> None:
    manifest = runtime_source_manifest(ROOT / "configs/stage_d22.yaml")
    paths = {item["path"] for item in manifest["files"]}
    assert "src/keyed_gram/stage_d1_gateway.py" in paths
    assert "src/keyed_gram/stage_d2_memory.py" in paths
    assert "src/keyed_gram/stage_d22_services.py" in paths
    assert len(manifest["manifest_payload_sha256"]) == 64


def test_git_state_supports_repository_git_version() -> None:
    state = git_state(ROOT / "configs/stage_d22.yaml")
    expected_branch = subprocess.check_output(
        ("git", "symbolic-ref", "--short", "HEAD"),
        cwd=ROOT,
        text=True,
    ).strip()
    assert state["branch"] == expected_branch
    assert len(state["commit"]) == 40


def test_smoke_delivers_once_and_rejects_replay() -> None:
    result = run_d22_smoke(ROOT / "configs/stage_d22_smoke.yaml")
    assert result["passed"] is True
    assert result["authorized_exact_delivery"] is True
    assert result["replay_reason"] == "replay"
    assert result["plaintext_recorded"] is False


@pytest.mark.parametrize(
    "command",
    ("stage-d22-audit", "stage-d22-smoke", "stage-d22-finalize"),
)
def test_cli_registers_d22_commands(command: str) -> None:
    args = build_parser().parse_args(
        [command, "--config", "configs/stage_d22.yaml"]
    )
    assert args.command == command
