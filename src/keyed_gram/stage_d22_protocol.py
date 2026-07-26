"""Stage D2.2 配置冻结、一次性运行状态与 artifact 完整性工具。"""

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

D22_SCHEMA_VERSION = 1
D22_STAGE = "D2.2-isolated-service-generation-probe"


class D22ProtocolError(RuntimeError):
    """D2.2 协议、配置或 artifact 完整性错误。"""


def repo_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise D22ProtocolError("无法解析 D2.2 仓库根目录")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(config_path) / path


def _require_false(mapping: Mapping[str, Any], names: Sequence[str]) -> None:
    if any(mapping.get(name) is not False for name in names):
        raise D22ProtocolError("D2.2 private/router/training/C3 门禁必须固定 false")


def _required_scenarios(values: Mapping[str, Any]) -> tuple[str, ...]:
    keys = (
        "required_positive_scenarios",
        "required_authorization_negative_scenarios",
        "required_service_boundary_scenarios",
        "required_ipc_minimization_scenarios",
        "required_crash_scenarios",
        "required_multi_instance_scenarios",
    )
    scenarios = tuple(
        str(item)
        for key in keys
        for item in values.get(key, ())
    )
    if len(scenarios) != 35 or len(set(scenarios)) != 35:
        raise D22ProtocolError("D2.2 必须冻结 35 个互异场景")
    return scenarios


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
        raise D22ProtocolError("无法读取 D2.2 配置") from exc
    expected = f"{D22_STAGE}-smoke" if smoke else D22_STAGE
    if (
        not isinstance(values, Mapping)
        or values.get("schema_version") != D22_SCHEMA_VERSION
        or values.get("stage") != expected
    ):
        raise D22ProtocolError("D2.2 配置 schema/stage 不匹配")
    protocol = values.get("protocol")
    roles = values.get("service_roles")
    if not isinstance(protocol, Mapping) or not isinstance(roles, Mapping):
        raise D22ProtocolError("D2.2 配置缺少 protocol/service_roles")
    _require_false(
        protocol,
        (
            "remote_model_allowed",
            "network_socket_allowed",
            "plaintext_in_source_allowed",
            "plaintext_in_config_allowed",
            "plaintext_in_artifacts_allowed",
            "raw_ipc_capture_persistence_allowed",
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
    if (
        protocol.get("process_start_method") != "spawn"
        or protocol.get("ipc_family") != "AF_UNIX"
        or protocol.get("ipc_serialization") != "strict_json_send_bytes"
        or protocol.get("replay_store")
        != "sqlite_begin_immediate_full_sync_delete_journal"
    ):
        raise D22ProtocolError("D2.2 进程、IPC 或持久化协议漂移")
    capability = values.get("capability", {})
    ticket = values.get("release_ticket", {})
    memory = values.get("data_memory", {})
    if (
        capability.get("algorithm") != "HMAC-SHA-256"
        or int(capability.get("key_bytes", 0)) < 32
        or capability.get("single_use") is not True
        or ticket.get("algorithm") != "HMAC-SHA-256"
        or int(ticket.get("key_bytes", 0)) < 32
        or memory.get("algorithm") != "AES-256-GCM"
        or int(memory.get("key_bytes", 0)) != 32
        or int(memory.get("nonce_bytes", 0)) != 12
    ):
        raise D22ProtocolError("D2.2 capability/ticket/AEAD 算法冻结声明漂移")
    gateway = roles.get("gateway", {})
    memory_role = roles.get("memory", {})
    generator = roles.get("generator", {})
    if (
        gateway.get("holds_capability_hmac_key") is not True
        or gateway.get("holds_release_ticket_key") is not True
        or gateway.get("holds_data_encryption_key") is not False
        or memory_role.get("holds_capability_hmac_key") is not False
        or memory_role.get("holds_release_ticket_key") is not True
        or memory_role.get("holds_data_encryption_key") is not True
        or memory_role.get("accepts_external_capability") is not False
        or memory_role.get("accepts_natural_language") is not False
        or generator.get("holds_capability_hmac_key") is not False
        or generator.get("holds_release_ticket_key") is not False
        or generator.get("holds_data_encryption_key") is not False
        or generator.get("accepts_capability_token") is not False
        or generator.get("connects_to_memory") is not False
        or generator.get("tools") != []
    ):
        raise D22ProtocolError("D2.2 服务角色或 key 分离声明漂移")
    if smoke:
        return dict(values)

    _required_scenarios(values)
    governance = values.get("governance")
    canary = values.get("canary")
    if (
        not isinstance(governance, Mapping)
        or not isinstance(canary, Mapping)
        or governance.get("capability_gated_ephemeral_generation_status")
        != "passed"
        or governance.get("public_synthetic_generation_boundary_validated")
        is not True
        or governance.get("ready_for_d2_2_isolated_service_probe") is not True
        or governance.get("semantic_router_in_tcb") is not False
        or governance.get("original_c3_allowed") is not False
        or governance.get("c3_eligible") is not False
        or canary.get("entropy_bits") != 128
        or canary.get("persisted_output_digest_count_field")
        != "persisted_output_digest_count"
        or canary.get("runtime_canary_count_field") != "runtime_canary_count"
        or canary.get("runtime_canary_variant_count_field")
        != "runtime_canary_variant_count"
        or canary.get("scanned_location_count_field")
        != "scanned_location_count"
        or set(canary.get("scan_variants", ()))
        != {
            "raw",
            "lowercase",
            "uppercase",
            "hex",
            "standard_base64",
            "urlsafe_base64",
            "character_spaced",
            "hyphen_stripped",
        }
    ):
        raise D22ProtocolError("D2.2 governance/canary 冻结声明不完整")
    for label in ("d21_summary", "d21_artifact_manifest", "d21_report"):
        declaration = governance.get(label)
        if not isinstance(declaration, Mapping):
            raise D22ProtocolError(f"D2.2 缺少 {label} 引用")
        path = resolve_path(config_path, declaration["path"])
        if sha256_file(path) != declaration["sha256"]:
            raise D22ProtocolError(f"{label} SHA-256 漂移")
    summary = read_json(
        resolve_path(config_path, governance["d21_summary"]["path"]),
        label="D2.1 summary",
    )
    if (
        summary.get("capability_gated_ephemeral_generation_status")
        != "passed"
        or summary.get("public_synthetic_generation_boundary_validated")
        is not True
        or summary.get("ready_for_d2_2_isolated_service_probe") is not True
        or summary.get("private_value_memory_ready") is not False
        or summary.get("c3_eligible") is not False
    ):
        raise D22ProtocolError("D2.1 readiness 边界漂移")
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
        "configs/stage_d22.yaml",
        "artifacts/stage_d21/stage_d21_summary.json",
        "artifacts/stage_d21/artifact_sha256_manifest.json",
        "PHASE_D21_REPORT.md",
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
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise D22ProtocolError(f"D2.2 runtime source 缺失：{relative}")
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
    return read_json(path, label="D2.2 protocol state")


def mark_phase_started(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.setdefault("phases", {})
    if phase in phases:
        raise D22ProtocolError(f"D2.2 {phase} 已启动过，拒绝重跑")
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
        raise D22ProtocolError(f"D2.2 {phase} 未处于 started")
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
        raise D22ProtocolError(f"D2.2 {phase} 尚未完成")


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
