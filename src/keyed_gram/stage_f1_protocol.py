"""Stage F1 配置冻结、一次性运行与完整性协议。"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .authcap_policy import canonical_sha256

F1_STAGE = "F1-authzroutebench"
F1_SCHEMA_VERSION = 1


class F1ProtocolError(RuntimeError):
    """F1 协议、路径、冻结状态或 artifact 不满足要求。"""


def repo_root(config_path: str | Path) -> Path:
    target = Path(config_path).resolve()
    for parent in (target.parent, *target.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise F1ProtocolError("无法解析仓库根目录")


def _safe_relative(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise F1ProtocolError(f"{label} 必须是安全相对路径")
    return path


def resolve_path(
    config_path: str | Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    return repo_root(config_path) / _safe_relative(value, label)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise F1ProtocolError(f"JSON 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def write_json(path: str | Path, value: Any) -> None:
    _finite(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise F1ProtocolError(f"duplicate JSON key：{key}")
        value[key] = child
    return value


def read_json(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                F1ProtocolError(f"{label} 非法常量：{item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise F1ProtocolError(f"无法读取 {label}") from exc
    if not isinstance(value, dict):
        raise F1ProtocolError(f"{label} 顶层必须是 object")
    _finite(value)
    return value


def load_config(
    config_path: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    try:
        value = yaml.safe_load(
            Path(config_path).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise F1ProtocolError("无法读取 F1 配置") from exc
    if not isinstance(value, dict):
        raise F1ProtocolError("F1 配置顶层必须是 object")
    expected = f"{F1_STAGE}-smoke" if smoke else F1_STAGE
    if value.get("schema_version") != 1 or value.get("stage") != expected:
        raise F1ProtocolError("F1 schema/stage 不匹配")
    protocol = value.get("protocol")
    final = value.get("final_status")
    if not isinstance(protocol, Mapping) or not isinstance(final, Mapping):
        raise F1ProtocolError("F1 protocol/final_status 缺失")
    false_protocol = (
        "locked_test_scoring_allowed",
        "learned_baseline_allowed",
        "final_compiler_allowed",
        "capability_issuance_allowed",
        "memory_access_allowed",
        "tool_execution_allowed",
        "private_data_allowed",
        "private_answer_allowed",
        "private_value_allowed",
        "confirmation_allowed",
        "key_attack_allowed",
        "lm_answer_injection_allowed",
        "natural_language_router_authorization_allowed",
    )
    if any(protocol.get(name) is not False for name in false_protocol):
        raise F1ProtocolError("F1 禁止项必须全部固定 false")
    false_final = (
        "authority_non_amplification_implemented",
        "formal_model_completed",
        "real_agent_integration_completed",
        "private_value_memory_ready",
        "original_c3_allowed",
        "c3_eligible",
    )
    if any(final.get(name) is not False for name in false_final):
        raise F1ProtocolError("F1 最终门禁必须全部固定 false")
    if not smoke:
        benchmark = value.get("benchmark")
        if not isinstance(benchmark, Mapping):
            raise F1ProtocolError("formal benchmark 配置缺失")
        if benchmark.get("policy_ground_truth") != "deterministic_oracle":
            raise F1ProtocolError("ground truth 必须是 deterministic oracle")
        if benchmark.get("independent_human_validation") is not False:
            raise F1ProtocolError("不得伪造独立人审")
        for name in ("data_dir", "artifact_dir", "runtime_dir", "report_path"):
            _safe_relative(value["outputs"][name], f"outputs.{name}")
        frozen = value.get("frozen_upstream")
        if not isinstance(frozen, list) or len(frozen) < 10:
            raise F1ProtocolError("frozen_upstream 清单不完整")
        forbidden_fragments = (
            "private_answer",
            "private_value",
            "feature_cache",
            "confirmation",
            "artifacts/keys",
        )
        for item in frozen:
            relative = _safe_relative(
                item.get("path", ""), "frozen_upstream.path"
            )
            lowered = relative.as_posix().casefold()
            if any(fragment in lowered for fragment in forbidden_fragments):
                raise F1ProtocolError("frozen_upstream 禁止引用 private/secret source")
    return value


def git_state(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments),
            cwd=root,
            text=True,
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("symbolic-ref", "--short", "HEAD"),
        "tracked_dirty": bool(
            run("status", "--porcelain", "--untracked-files=no")
        ),
    }


def verify_frozen_upstream(
    config_path: str | Path,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for item in values["frozen_upstream"]:
        relative = _safe_relative(item["path"], "frozen_upstream.path")
        target = root / relative
        observed = sha256_file(target) if target.is_file() else None
        if observed != item["sha256"]:
            raise F1ProtocolError(f"冻结上游 hash mismatch：{relative}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": observed,
                "size_bytes": target.stat().st_size,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "files": files,
        "frozen_upstream_code_modified": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


_SOURCE_PATHS = (
    ".gitattributes",
    "AUTHCAP_RESEARCH_PLAN.md",
    "PHASE_F0_REPORT.md",
    "configs/stage_f0.yaml",
    "configs/stage_f1.yaml",
    "configs/stage_f1_smoke.yaml",
    "src/keyed_gram/authcap_types.py",
    "src/keyed_gram/authcap_authority.py",
    "src/keyed_gram/authcap_policy.py",
    "src/keyed_gram/authcap_oracle.py",
    "src/keyed_gram/authcap_benchmark.py",
    "src/keyed_gram/authcap_collision.py",
    "src/keyed_gram/stage_f1.py",
    "src/keyed_gram/stage_f1_protocol.py",
    "src/keyed_gram/stage_f1_metrics.py",
    "src/keyed_gram/cli.py",
    "tests/test_authcap_types.py",
    "tests/test_authcap_authority.py",
    "tests/test_authcap_policy.py",
    "tests/test_authcap_oracle.py",
    "tests/test_authcap_benchmark.py",
    "tests/test_authcap_collision.py",
    "tests/test_stage_f1_protocol.py",
)


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for relative in _SOURCE_PATHS:
        target = root / relative
        if not target.is_file():
            raise F1ProtocolError(f"F1 runtime source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    payload = {"schema_version": 1, "files": files}
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def output_paths(
    config_path: str | Path,
    values: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path]:
    outputs = values["outputs"]
    return (
        resolve_path(config_path, outputs["data_dir"], label="data_dir"),
        resolve_path(config_path, outputs["artifact_dir"], label="artifact_dir"),
        resolve_path(config_path, outputs["runtime_dir"], label="runtime_dir"),
        resolve_path(config_path, outputs["report_path"], label="report_path"),
    )


def formal_preflight(
    config_path: str | Path,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    state = git_state(config_path)
    if state["tracked_dirty"]:
        raise F1ProtocolError("tracked worktree 非干净，拒绝 formal build")
    if state["branch"] != values["protocol"]["required_branch"]:
        raise F1ProtocolError("formal build 分支不匹配")
    data_dir, artifact_dir, runtime_dir, report_path = output_paths(
        config_path, values
    )
    for path in (data_dir, artifact_dir, runtime_dir, report_path):
        if path.exists():
            raise F1ProtocolError(f"formal output 已存在，拒绝重复构建：{path}")
    upstream = verify_frozen_upstream(config_path, values)
    return {"git": state, "upstream": upstream}


def mark_started(runtime_dir: Path, git: Mapping[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        runtime_dir / "protocol_state.json",
        {
            "schema_version": 1,
            "formal_build": {
                "status": "started",
                "code_freeze_commit": git["commit"],
                "branch": git["branch"],
            },
        },
    )


def mark_completed(
    runtime_dir: Path,
    outputs: Sequence[Path],
) -> None:
    state = read_json(
        runtime_dir / "protocol_state.json",
        label="F1 protocol state",
    )
    phase = state.get("formal_build")
    if not isinstance(phase, dict) or phase.get("status") != "started":
        raise F1ProtocolError("F1 formal build 未处于 started")
    phase["status"] = "completed"
    phase["outputs"] = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in outputs
    ]
    write_json(runtime_dir / "protocol_state.json", state)


def mark_failed(runtime_dir: Path, reason: str) -> None:
    state_path = runtime_dir / "protocol_state.json"
    if not state_path.is_file():
        return
    state = read_json(state_path, label="F1 protocol state")
    state["formal_build"]["status"] = "failed"
    state["formal_build"]["failure_reason"] = reason
    write_json(state_path, state)


def artifact_inventory(
    artifact_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(artifact_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    ]
    payload = {
        "schema_version": 1,
        "files": files,
        "external_files": [
            {
                "path": report_path.name,
                "size_bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            }
        ],
        "private_data_present": False,
        "model_or_optimizer_present": False,
        "capability_or_key_material_present": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_artifact_inventory(
    artifact_dir: Path,
    report_path: Path,
) -> None:
    path = artifact_dir / "artifact_sha256_manifest.json"
    observed = read_json(path, label="F1 artifact manifest")
    expected = artifact_inventory(artifact_dir, report_path)
    if observed != expected:
        raise F1ProtocolError("F1 artifact inventory/hash mismatch")


def load_baseline_split(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """为后续 baseline 预留；F1 永远拒绝打开 locked_test。"""

    if split == "locked_test":
        raise F1ProtocolError("F1 baseline loader 禁止打开 locked_test")
    if split not in {"train", "calibration", "development"}:
        raise F1ProtocolError("未知 baseline split")
    path = Path(data_dir) / f"{split}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line, object_pairs_hook=_pairs)
        if not isinstance(value, dict):
            raise F1ProtocolError("benchmark row 必须是 object")
        rows.append(value)
    return rows
