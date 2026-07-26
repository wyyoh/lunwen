"""Stage D2.3 配置冻结、一次性运行与 artifact 完整性协议。"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .stage_d1_protocol import (
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)

D23_SCHEMA_VERSION = 1
D23_STAGE = "D2.3-fault-observability-lifecycle-audit"


class D23ProtocolError(RuntimeError):
    """D2.3 协议、配置或 artifact 完整性错误。"""


def repo_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise D23ProtocolError("无法解析 D2.3 仓库根目录")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(config_path) / path


def required_scenarios(values: Mapping[str, Any]) -> tuple[str, ...]:
    keys = (
        "required_positive_scenarios",
        "required_observability_scenarios",
        "required_fault_scenarios",
        "required_endpoint_ipc_scenarios",
        "required_state_recovery_scenarios",
    )
    scenarios = tuple(
        str(item)
        for key in keys
        for item in values.get(key, ())
    )
    if len(scenarios) != 51 or len(set(scenarios)) != 51:
        raise D23ProtocolError("D2.3 必须冻结 51 个互异场景")
    return scenarios


def _require_false(mapping: Mapping[str, Any], names: Sequence[str]) -> None:
    if any(mapping.get(name) is not False for name in names):
        raise D23ProtocolError("D2.3 private/router/training/C3 门禁必须固定 false")


def load_config(
    config_path: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    try:
        values = yaml.safe_load(
            Path(config_path).read_text(encoding="utf-8")
        ) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise D23ProtocolError("无法读取 D2.3 配置") from exc
    expected = f"{D23_STAGE}-smoke" if smoke else D23_STAGE
    if (
        not isinstance(values, Mapping)
        or values.get("schema_version") != D23_SCHEMA_VERSION
        or values.get("stage") != expected
    ):
        raise D23ProtocolError("D2.3 配置 schema/stage 不匹配")
    protocol = values.get("protocol")
    if not isinstance(protocol, Mapping):
        raise D23ProtocolError("D2.3 缺少 protocol")
    _require_false(
        protocol,
        (
            "physical_disk_exhaustion_allowed",
            "raw_ipc_capture_persistence_allowed",
            "remote_model_allowed",
            "network_socket_allowed",
            "plaintext_in_source_allowed",
            "plaintext_in_config_allowed",
            "plaintext_in_artifacts_allowed",
            "command_line_secret_allowed",
            "environment_secret_allowed",
            "private_value_memory_allowed",
            "private_answers_allowed",
            "natural_language_router_authorization_allowed",
            "confirmation_allowed",
            "fine_tuning_allowed",
            "key_attack_allowed",
            "original_c3_allowed",
            "c3_eligible",
        ),
    )
    if smoke:
        return dict(values)
    if (
        protocol.get("process_start_method") != "spawn"
        or protocol.get("ipc_family") != "AF_UNIX"
        or protocol.get("byte_level_capture")
        != "memory_only_digest_projection"
        or protocol.get("persistent_state")
        != "sqlite_begin_immediate_full_sync_delete_journal"
        or protocol.get("rollback_detection")
        != "separate_monotonic_anchor_and_hmac_metadata"
        or protocol.get("deterministic_fault_injection_allowed") is not True
    ):
        raise D23ProtocolError("D2.3 进程、IPC、故障或状态协议漂移")
    required_scenarios(values)
    governance = values.get("governance")
    epochs = values.get("epochs")
    observability = values.get("observability")
    ipc = values.get("ipc")
    canary = values.get("canary")
    if not all(
        isinstance(item, Mapping)
        for item in (governance, epochs, observability, ipc, canary)
    ):
        raise D23ProtocolError("D2.3 冻结声明不完整")
    if (
        governance.get("isolated_service_generation_status") != "passed"
        or governance.get(
            "service_boundary_validated_with_public_synthetic_values"
        )
        is not True
        or governance.get("ready_for_d2_3_fault_and_observability_probe")
        is not True
        or governance.get("semantic_router_in_tcb") is not False
        or governance.get("original_c3_allowed") is not False
        or governance.get("c3_eligible") is not False
        or epochs.get("rollback_fail_closed") is not True
        or observability.get(
            "sensitive_values_allowed_in_metric_labels"
        )
        is not False
        or observability.get(
            "sensitive_values_allowed_in_trace_attributes"
        )
        is not False
        or ipc.get("socket_peer_uid_required") is not True
        or ipc.get("symlink_replacement_allowed") is not False
        or canary.get("entropy_bits") != 128
    ):
        raise D23ProtocolError("D2.3 governance/epoch/observability 边界漂移")
    for label in ("d22_summary", "d22_artifact_manifest", "d22_report"):
        declaration = governance.get(label)
        if not isinstance(declaration, Mapping):
            raise D23ProtocolError(f"D2.3 缺少 {label} 引用")
        path = resolve_path(config_path, declaration["path"])
        if sha256_file(path) != declaration["sha256"]:
            raise D23ProtocolError(f"{label} SHA-256 漂移")
    summary = read_json(
        resolve_path(config_path, governance["d22_summary"]["path"]),
        label="D2.2 summary",
    )
    if (
        summary.get("isolated_service_generation_status") != "passed"
        or summary.get(
            "service_boundary_validated_with_public_synthetic_values"
        )
        is not True
        or summary.get("ready_for_d2_3_fault_and_observability_probe")
        is not True
        or summary.get("private_value_memory_ready") is not False
        or summary.get("c3_eligible") is not False
    ):
        raise D23ProtocolError("D2.2 readiness 边界漂移")
    return dict(values)


def output_paths(config_path: str | Path) -> tuple[Path, Path]:
    values = load_config(config_path)
    return (
        resolve_path(config_path, values["outputs"]["artifact_dir"]),
        resolve_path(config_path, values["outputs"]["runtime_dir"]),
    )


def git_state(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=root, text=True
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("symbolic-ref", "--short", "HEAD"),
        "tracked_dirty": bool(
            run("status", "--porcelain", "--untracked-files=no")
        ),
    }


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    paths = (
        ".gitattributes",
        "pyproject.toml",
        "configs/stage_d23.yaml",
        "artifacts/stage_d22/stage_d22_summary.json",
        "artifacts/stage_d22/artifact_sha256_manifest.json",
        "PHASE_D22_REPORT.md",
        "src/keyed_gram/__init__.py",
        "src/keyed_gram/cli.py",
        "src/keyed_gram/stage_c24_contract.py",
        "src/keyed_gram/stage_d1_contract.py",
        "src/keyed_gram/stage_d1_gateway.py",
        "src/keyed_gram/stage_d1_policy.py",
        "src/keyed_gram/stage_d1_token.py",
        "src/keyed_gram/stage_d2_contract.py",
        "src/keyed_gram/stage_d2_crypto.py",
        "src/keyed_gram/stage_d2_identity.py",
        "src/keyed_gram/stage_d2_memory.py",
        "src/keyed_gram/stage_d21_leakage.py",
        "src/keyed_gram/stage_d22.py",
        "src/keyed_gram/stage_d22_contract.py",
        "src/keyed_gram/stage_d22_ipc.py",
        "src/keyed_gram/stage_d22_metrics.py",
        "src/keyed_gram/stage_d22_protocol.py",
        "src/keyed_gram/stage_d22_runtime.py",
        "src/keyed_gram/stage_d22_services.py",
        "src/keyed_gram/stage_d22_state.py",
        "src/keyed_gram/stage_d22_ticket.py",
        "src/keyed_gram/stage_d23.py",
        "src/keyed_gram/stage_d23_contract.py",
        "src/keyed_gram/stage_d23_ipc.py",
        "src/keyed_gram/stage_d23_metrics.py",
        "src/keyed_gram/stage_d23_observability.py",
        "src/keyed_gram/stage_d23_protocol.py",
        "src/keyed_gram/stage_d23_state.py",
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise D23ProtocolError(f"D2.3 runtime source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {"schema_version": 1, "files": files}
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def phase_state_path(runtime_dir: Path) -> Path:
    return runtime_dir / "protocol_state.json"


def read_phase_state(runtime_dir: Path) -> dict[str, Any]:
    path = phase_state_path(runtime_dir)
    if not path.is_file():
        return {"schema_version": 1, "phases": {}}
    return read_json(path, label="D2.3 protocol state")


def mark_phase_started(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.setdefault("phases", {})
    if phase in phases:
        raise D23ProtocolError(f"D2.3 {phase} 已启动过，拒绝重跑")
    phases[phase] = {"status": "started"}
    write_json(phase_state_path(runtime_dir), state)


def mark_phase_completed(
    runtime_dir: Path,
    phase: str,
    outputs: Sequence[Path],
) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.get("phases", {})
    if phases.get(phase, {}).get("status") != "started":
        raise D23ProtocolError(f"D2.3 {phase} 未处于 started")
    phases[phase] = {
        "status": "completed",
        "outputs": [
            {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    write_json(phase_state_path(runtime_dir), state)


def require_phase_completed(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    if state.get("phases", {}).get(phase, {}).get("status") != "completed":
        raise D23ProtocolError(f"D2.3 {phase} 尚未完成")


def artifact_manifest(
    artifact_dir: Path,
    *,
    exclude: Sequence[str] = ("artifact_sha256_manifest.json",),
) -> dict[str, Any]:
    excluded = set(exclude)
    files = [
        {
            "path": path.relative_to(artifact_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(artifact_dir).as_posix() not in excluded
    ]
    payload: dict[str, Any] = {"schema_version": 1, "files": files}
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload
