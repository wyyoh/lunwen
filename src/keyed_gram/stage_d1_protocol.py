"""Stage D1 配置、一次性运行状态与 artifact 完整性工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
STAGE_NAME = "D1-trusted-capability-contract"


class D1ProtocolError(RuntimeError):
    """D1 协议或完整性门禁失败。"""


def repo_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise D1ProtocolError("无法解析仓库根目录")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(config_path) / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise D1ProtocolError(f"JSON 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    _assert_finite(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"非法 JSON 常量 {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise D1ProtocolError(f"{label} 不是严格 JSON") from exc
    if not isinstance(value, Mapping):
        raise D1ProtocolError(f"{label} 必须是 JSON object")
    _assert_finite(value)
    return dict(value)


def load_config(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    try:
        values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise D1ProtocolError("无法读取 D1 配置") from exc
    expected = f"{STAGE_NAME}-smoke" if smoke else STAGE_NAME
    if (
        not isinstance(values, Mapping)
        or values.get("schema_version") != SCHEMA_VERSION
        or values.get("stage") != expected
    ):
        raise D1ProtocolError("D1 配置 schema/stage 不匹配")
    protocol = values.get("protocol")
    capability = values.get("capability")
    if not isinstance(protocol, Mapping) or not isinstance(capability, Mapping):
        raise D1ProtocolError("D1 配置缺少 protocol/capability")
    required_false = (
        "private_value_memory_allowed",
        "private_answers_allowed",
        "lm_answer_injection_allowed",
        "key_attack_allowed",
        "confirmation_allowed",
        "original_c3_allowed",
        "c3_eligible",
    )
    if any(protocol.get(key) is not False for key in required_false):
        raise D1ProtocolError("D1 private/LM/key/C3 门禁必须固定 false")
    if capability.get("algorithm") != "HMAC-SHA-256":
        raise D1ProtocolError("D1 只允许预注册 HMAC-SHA-256")
    if int(capability.get("key_bytes", 0)) < 32:
        raise D1ProtocolError("D1 HMAC key 必须至少 32 bytes")
    if capability.get("single_use") is not True:
        raise D1ProtocolError("D1 必须启用 single-use replay protection")
    if not smoke:
        governance = values.get("governance")
        if (
            not isinstance(governance, Mapping)
            or governance.get("selective_router_research_status")
            != "stopped_external_validation_failed"
            or governance.get("additional_router_complexity_allowed") is not False
            or governance.get("semantic_router_in_tcb") is not False
            or governance.get("suggested_relation_authorizes_memory") is not False
        ):
            raise D1ProtocolError("D1 router-outside-TCB 治理声明不完整")
        reference = governance.get("research_status")
        if not isinstance(reference, Mapping):
            raise D1ProtocolError("D1 缺少 C2 stop record 引用")
        path = resolve_path(config_path, reference["path"])
        if sha256_file(path) != reference["sha256"]:
            raise D1ProtocolError("C2 stop record SHA-256 漂移")
        status = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        router = status.get("router_research", {})
        if (
            router.get("additional_router_complexity_allowed") is not False
            or router.get("selective_router_research_status")
            != "stopped_external_validation_failed"
        ):
            raise D1ProtocolError("C2 stop record 状态漂移")
        trust = values.get("trust_boundary")
        if (
            not isinstance(trust, Mapping)
            or "natural_language_router"
            not in trust.get("untrusted_components", ())
            or "relation_proposal"
            not in trust.get("untrusted_components", ())
            or "trusted_capability_gateway"
            not in trust.get("trusted_components", ())
            or "typed_keyed_memory"
            not in trust.get("trusted_components", ())
            or trust.get("subject_identity_provider_implemented") is not False
            or trust.get("process_isolation_implemented") is not False
            or trust.get("verifier_can_mint_with_symmetric_key") is not True
        ):
            raise D1ProtocolError("D1 trust boundary 或原型限制声明不完整")
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
        "configs/research_status.yaml",
        "configs/stage_d1.yaml",
        "src/keyed_gram/cli.py",
        "src/keyed_gram/stage_c24_contract.py",
        "src/keyed_gram/stage_d1.py",
        "src/keyed_gram/stage_d1_contract.py",
        "src/keyed_gram/stage_d1_gateway.py",
        "src/keyed_gram/stage_d1_policy.py",
        "src/keyed_gram/stage_d1_protocol.py",
        "src/keyed_gram/stage_d1_token.py",
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise D1ProtocolError(f"D1 runtime source 缺失：{relative}")
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
    return read_json(path, label="D1 protocol state")


def mark_phase_started(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.setdefault("phases", {})
    if phase in phases:
        raise D1ProtocolError(f"D1 {phase} 已启动过，拒绝重跑")
    phases[phase] = {"status": "started"}
    write_json(phase_state_path(runtime_dir), state)


def mark_phase_completed(
    runtime_dir: Path, phase: str, outputs: Sequence[Path]
) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.get("phases", {})
    if phases.get(phase, {}).get("status") != "started":
        raise D1ProtocolError(f"D1 {phase} 未处于 started")
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
        raise D1ProtocolError(f"D1 {phase} 尚未完成")
