"""Stage D2 配置、冻结源、一次性运行与严格 artifact 工具。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cryptography
import yaml

from .stage_d1_protocol import (
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)


D2_SCHEMA_VERSION = 1
D2_STAGE = "D2-capability-gated-authenticated-keyed-memory"


class D2ProtocolError(RuntimeError):
    """D2 协议、冻结状态或 artifact 完整性错误。"""


def repo_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise D2ProtocolError("无法解析 D2 仓库根目录")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(config_path) / path


def _require_false(mapping: Mapping[str, Any], names: Sequence[str]) -> None:
    if any(mapping.get(name) is not False for name in names):
        raise D2ProtocolError("D2 private/router/LM/C3 门禁必须固定 false")


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
        raise D2ProtocolError("无法读取 D2 配置") from exc
    expected = f"{D2_STAGE}-smoke" if smoke else D2_STAGE
    if (
        not isinstance(values, Mapping)
        or values.get("schema_version") != D2_SCHEMA_VERSION
        or values.get("stage") != expected
    ):
        raise D2ProtocolError("D2 配置 schema/stage 不匹配")
    protocol = values.get("protocol")
    capability = values.get("capability")
    data_memory = values.get("data_memory")
    if not all(
        isinstance(item, Mapping)
        for item in (protocol, capability, data_memory)
    ):
        raise D2ProtocolError("D2 配置缺少 protocol/capability/data_memory")
    _require_false(
        protocol,
        (
            "private_value_memory_allowed",
            "private_answers_allowed",
            "lm_answer_injection_allowed",
            "natural_language_router_allowed",
            "confirmation_allowed",
            "key_attack_allowed",
            "original_c3_allowed",
            "c3_eligible",
        ),
    )
    if (
        capability.get("algorithm") != "HMAC-SHA-256"
        or capability.get("single_use") is not True
        or data_memory.get("algorithm") != "AES-256-GCM"
        or int(data_memory.get("key_bytes", 0)) != 32
        or int(data_memory.get("nonce_bytes", 0)) != 12
        or int(data_memory.get("tag_bytes", 0)) != 16
        or data_memory.get("independent_from_capability_key") is not True
        or data_memory.get("implementation") != "cryptography"
        or data_memory.get("implementation_version")
        != cryptography.__version__
    ):
        raise D2ProtocolError("D2 capability/data-key/AEAD 冻结声明不匹配")
    if not smoke:
        governance = values.get("governance")
        trust = values.get("trust_boundary")
        if (
            not isinstance(governance, Mapping)
            or not isinstance(trust, Mapping)
            or governance.get("trusted_capability_contract_status")
            != "passed"
            or governance.get("ready_for_stage_d2_public_synthetic_values")
            is not True
            or governance.get("selective_router_research_status")
            != "stopped_external_validation_failed"
            or governance.get("semantic_router_in_tcb") is not False
            or governance.get("suggested_relation_authorizes_memory") is not False
            or trust.get("identity_mode") != "trusted_in_process_mock"
            or trust.get("caller_supplied_principal_allowed") is not False
            or trust.get("principal_seal_required") is not True
            or trust.get("process_isolation_implemented") is not False
            or trust.get("distributed_replay_store_implemented") is not False
            or trust.get("capability_authority_verifier_separated") is not False
        ):
            raise D2ProtocolError("D2 governance/trust boundary 声明不完整")
        for label in ("d1_summary", "d1_artifact_manifest"):
            declaration = governance.get(label)
            if not isinstance(declaration, Mapping):
                raise D2ProtocolError(f"D2 缺少 {label} 引用")
            path = resolve_path(config_path, declaration["path"])
            if sha256_file(path) != declaration["sha256"]:
                raise D2ProtocolError(f"{label} SHA-256 漂移")
        summary = read_json(
            resolve_path(config_path, governance["d1_summary"]["path"]),
            label="D1 summary",
        )
        if (
            summary.get("trusted_capability_contract_status") != "passed"
            or summary.get("ready_for_stage_d2_public_synthetic_values")
            is not True
            or summary.get("private_value_memory_ready") is not False
            or summary.get("c3_eligible") is not False
        ):
            raise D2ProtocolError("D1 readiness 边界漂移")
        expected_aad = [
            "schema_version",
            "algorithm",
            "record_id",
            "entity_id",
            "relation_id",
            "data_key_id",
            "record_version",
        ]
        if data_memory.get("aad_fields") != expected_aad:
            raise D2ProtocolError("D2 AAD 字段或顺序漂移")
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
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(
            run("status", "--porcelain", "--untracked-files=no")
        ),
    }


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    paths = (
        ".gitattributes",
        "pyproject.toml",
        "configs/stage_d2.yaml",
        "artifacts/stage_d1/stage_d1_summary.json",
        "artifacts/stage_d1/artifact_sha256_manifest.json",
        "src/keyed_gram/cli.py",
        "src/keyed_gram/stage_c24_contract.py",
        "src/keyed_gram/stage_d1_contract.py",
        "src/keyed_gram/stage_d1_gateway.py",
        "src/keyed_gram/stage_d1_policy.py",
        "src/keyed_gram/stage_d1_token.py",
        "src/keyed_gram/stage_d2.py",
        "src/keyed_gram/stage_d2_contract.py",
        "src/keyed_gram/stage_d2_crypto.py",
        "src/keyed_gram/stage_d2_identity.py",
        "src/keyed_gram/stage_d2_memory.py",
        "src/keyed_gram/stage_d2_protocol.py",
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise D2ProtocolError(f"D2 runtime source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "cryptography_version": cryptography.__version__,
        "files": files,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def phase_state_path(runtime_dir: Path) -> Path:
    return runtime_dir / "protocol_state.json"


def read_phase_state(runtime_dir: Path) -> dict[str, Any]:
    path = phase_state_path(runtime_dir)
    if not path.is_file():
        return {"schema_version": 1, "phases": {}}
    return read_json(path, label="D2 protocol state")


def mark_phase_started(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.setdefault("phases", {})
    if phase in phases:
        raise D2ProtocolError(f"D2 {phase} 已启动过，拒绝重跑")
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
        raise D2ProtocolError(f"D2 {phase} 未处于 started")
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
        raise D2ProtocolError(f"D2 {phase} 尚未完成")
