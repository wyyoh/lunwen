"""Stage C2.4c 的冻结、数据门禁与一次性状态机。

本模块只管理公开、answer-free 的 v2.1 benchmark。它不会解析 private
answer、confirmation、key 或模型输出，也不会把 exploratory 结果提升为正式验证。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .stage_c24b_metrics import assert_finite_json
from .stage_c24b_v21_seal import (
    atomic_write_text,
    load_exact_artifact_manifest,
    strict_json_loads,
)


SCHEMA_VERSION = 1
STAGE_NAME = "C2.4c-exploratory-selective-router-evaluation"
BENCHMARK_VERSION = "c24b-public-selective-router-v2.1"
EVALUATION_MODE = "exploratory_non_independent"
STATUS_VALUES = frozenset({"not_evaluated", "passed", "failed"})
PHASES = ("calibration", "development", "locked_scoring")

_FORBIDDEN_CONFIG_TERMS = (
    "confirmation_path",
    "confirmation_pool",
    "private_answer",
    "private_value",
    "permutation_key",
)


class C24CProtocolError(RuntimeError):
    """C2.4c fail-closed 协议错误。"""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_root(config_path: str | Path) -> Path:
    source = Path(config_path).resolve()
    for parent in (source.parent, *source.parents):
        if (parent / ".git").exists() and (parent / "pyproject.toml").is_file():
            return parent
    raise C24CProtocolError("无法定位 C2.4c 仓库根目录")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    root = _repo_root(config_path)
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _reject_forbidden_config(value: Any, *, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if any(term in normalized for term in _FORBIDDEN_CONFIG_TERMS):
                if child is not False:
                    raise C24CProtocolError(
                        f"C2.4c 配置包含禁止字段：{location}.{key}"
                    )
            _reject_forbidden_config(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_config(child, location=f"{location}[{index}]")


def load_config(config_path: str | Path) -> dict[str, Any]:
    source = Path(config_path)
    values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(values, Mapping):
        raise C24CProtocolError("C2.4c 配置必须是 mapping")
    values = dict(values)
    if (
        values.get("schema_version") != SCHEMA_VERSION
        or values.get("stage") != STAGE_NAME
    ):
        raise C24CProtocolError("C2.4c 配置 schema/stage 不匹配")
    protocol = values.get("protocol")
    if not isinstance(protocol, Mapping):
        raise C24CProtocolError("C2.4c 缺少 protocol")
    required_false = (
        "formal_calibration_allowed",
        "independent_external_validation",
        "public_benchmark_human_reviewed",
        "create_confirmation",
        "train_private_memory",
        "load_private_answers",
        "execute_answer_injection",
        "execute_key_attack",
    )
    if any(protocol.get(key) is not False for key in required_false):
        raise C24CProtocolError("C2.4c 正式/私有/confirmation 门禁不是固定 false")
    if (
        protocol.get("evaluation_mode") != EVALUATION_MODE
        or protocol.get("benchmark_version") != BENCHMARK_VERSION
        or protocol.get("data_revision_status")
        != "frozen_for_exploratory_model_scoring"
        or protocol.get("exploratory_calibration_allowed") is not True
        or protocol.get("exploratory_locked_scoring_allowed") is not True
        or protocol.get("public_benchmark_ai_reviewed") is not True
    ):
        raise C24CProtocolError("C2.4c exploratory 状态声明不完整")
    selection = values.get("selection")
    if not isinstance(selection, Mapping) or any(
        selection.get(key) is not False
        for key in ("development_used_for_selection", "locked_audit_used_for_selection")
    ):
        raise C24CProtocolError("development/locked 不得参与 C2.4c 选择")
    _reject_forbidden_config(values)
    return values


def git_state(config_path: str | Path) -> dict[str, Any]:
    root = _repo_root(config_path)

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


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    assert_finite_json(payload)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(Path(path), text + "\n")


def load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    value = strict_json_loads(Path(path).read_text(encoding="utf-8"), label=label)
    if not isinstance(value, Mapping):
        raise C24CProtocolError(f"{label} 必须是 JSON object")
    return dict(value)


def _verify_declared_file(
    config_path: str | Path, declaration: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    path = resolve_path(config_path, str(declaration["path"]))
    if not path.is_file():
        raise C24CProtocolError(f"冻结文件缺失：{label}")
    observed = sha256_file(path)
    if observed != declaration.get("sha256"):
        raise C24CProtocolError(f"冻结文件 SHA-256 漂移：{label}")
    result = {
        "path": path.relative_to(_repo_root(config_path)).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": observed,
    }
    if "rows" in declaration:
        result["rows"] = int(declaration["rows"])
    return result


def verify_frozen_benchmark(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    frozen = values.get("frozen_benchmark")
    if not isinstance(frozen, Mapping):
        raise C24CProtocolError("缺少 frozen_benchmark")
    files: dict[str, Any] = {}
    for name in (
        "config",
        "ai_review_manifest",
        "final_artifact_manifest",
        "benchmark_manifest",
        "typed_memory_contract",
        "selective_router",
    ):
        declaration = frozen.get(name)
        if not isinstance(declaration, Mapping):
            raise C24CProtocolError(f"冻结声明缺失：{name}")
        files[name] = _verify_declared_file(
            config_path, declaration, label=name
        )
    raw_data = frozen.get("data")
    if not isinstance(raw_data, Mapping) or set(raw_data) != {
        "public_train_v2_1",
        "public_calibration_v2_1",
        "public_locked_audit_v2_1",
    }:
        raise C24CProtocolError("v2.1 三个 split 冻结声明不完整")
    data = {
        name: _verify_declared_file(config_path, declaration, label=name)
        for name, declaration in raw_data.items()
        if isinstance(declaration, Mapping)
    }
    if set(data) != set(raw_data):
        raise C24CProtocolError("v2.1 data 声明 malformed")

    artifact_manifest_path = resolve_path(
        config_path, str(frozen["final_artifact_manifest"]["path"])
    )
    artifact_dir = artifact_manifest_path.parent
    final_artifact = load_exact_artifact_manifest(
        artifact_dir,
        artifact_manifest_path,
        label="v2.1 final artifact manifest",
    )
    review = load_json(
        resolve_path(config_path, str(frozen["ai_review_manifest"]["path"])),
        label="v2.1 AI review manifest",
    )
    expected_status = "completed_with_declared_boundaries_exploratory_non_independent"
    if (
        review.get("status") != expected_status
        or review.get("public_benchmark_human_reviewed") is not False
        or review.get("public_benchmark_ai_reviewed") is not True
        or review.get("independent_external_validation") is not False
        or final_artifact.get("formal_locked_audit_executed") is not False
        or final_artifact.get("public_benchmark_human_reviewed") is not False
        or final_artifact.get("public_benchmark_ai_reviewed") is not True
    ):
        raise C24CProtocolError("v2.1 AI 审核门禁发生漂移")
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "data_revision_status": "frozen_for_exploratory_model_scoring",
        "files": files,
        "data": data,
        "ai_review_status": expected_status,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": True,
        "independent_external_validation": False,
    }


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = _repo_root(config_path)
    relative_paths = (
        "configs/stage_c24c.yaml",
        "src/keyed_gram/cli.py",
        "src/keyed_gram/stage_c23_benchmark.py",
        "src/keyed_gram/stage_c23_semantic.py",
        "src/keyed_gram/stage_c24_contract.py",
        "src/keyed_gram/stage_c24b_router.py",
        "src/keyed_gram/stage_c24b_calibration.py",
        "src/keyed_gram/stage_c24b_metrics.py",
        "src/keyed_gram/stage_c24c.py",
        "src/keyed_gram/stage_c24c_models.py",
        "src/keyed_gram/stage_c24c_protocol.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise C24CProtocolError(f"C2.4c runtime source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"schema_version": SCHEMA_VERSION, "files": files}
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def output_paths(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    values = load_config(config_path)
    outputs = values["outputs"]
    artifact = resolve_path(
        config_path, output_dir if output_dir is not None else outputs["artifact_dir"]
    )
    runtime = resolve_path(config_path, outputs["runtime_dir"])
    return artifact, runtime


def base_status() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "evaluation_mode": EVALUATION_MODE,
        "benchmark_version": BENCHMARK_VERSION,
        "data_revision_status": "frozen_for_exploratory_model_scoring",
        "exploratory_calibration_allowed": True,
        "exploratory_locked_scoring_allowed": True,
        "formal_calibration_allowed": False,
        "independent_external_validation": False,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": True,
        "closed_set_selective_router_status": "not_evaluated",
        "open_set_abstention_status": "not_evaluated",
        "closed_set_selective_router_ready": False,
        "open_set_abstention_ready": False,
        "discrete_memory_contract_preserved": True,
        "ready_to_create_new_confirmation_pool": False,
        "new_confirmation_pool_created_after_freeze": False,
        "c3_eligible": False,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "answer_injection_executed": False,
        "key_attack_executed": False,
        "confirmation_created_or_read": False,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
    }


def seal_benchmark(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir = output_paths(config_path, output_dir=output_dir)
    freeze_path = artifact_dir / "benchmark_freeze_manifest.json"
    if artifact_dir.exists() or runtime_dir.exists():
        raise C24CProtocolError("C2.4c 输出目录已存在，禁止覆盖或重做 seal")
    git = git_state(config_path)
    if git["tracked_dirty"]:
        raise C24CProtocolError("C2.4c seal 要求 tracked worktree 干净")
    benchmark = verify_frozen_benchmark(config_path)
    development = _verify_declared_file(
        config_path, values["development"]["source_config"], label="development_config"
    )
    sources = runtime_source_manifest(config_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "benchmark_and_code_frozen_before_exploratory_calibration",
        "evaluation_mode": EVALUATION_MODE,
        "benchmark": benchmark,
        "development_source": development,
        "runtime_source_manifest": sources,
        "git": git,
        "config_sha256": sha256_file(config_path),
        "definitions_sha256": load_json(
            resolve_path(config_path, values["frozen_benchmark"]["benchmark_manifest"]["path"]),
            label="v2.1 benchmark manifest",
        )["definitions_sha256"],
        "selection_split": "public_calibration_v2_1",
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "formal_calibration_allowed": False,
        "independent_external_validation": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    write_json(freeze_path, payload)
    status = {
        **base_status(),
        "status": "frozen_waiting_for_exploratory_calibration",
        "git": git,
        "benchmark_freeze_manifest_sha256": sha256_file(freeze_path),
        "calibration_executed": False,
        "development_executed_once": False,
        "exploratory_locked_scoring_executed_once": False,
    }
    write_json(artifact_dir / "protocol_status.json", status)
    return status


def verify_benchmark_seal(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    path = artifact_dir / "benchmark_freeze_manifest.json"
    payload = load_json(path, label="C2.4c benchmark freeze")
    digest = payload.pop("manifest_payload_sha256", None)
    if digest != canonical_sha256(payload):
        raise C24CProtocolError("C2.4c benchmark freeze payload hash 不匹配")
    payload["manifest_payload_sha256"] = digest
    if payload.get("git", {}).get("commit") != git_state(config_path)["commit"]:
        raise C24CProtocolError("C2.4c git HEAD 已偏离代码冻结提交")
    if payload.get("runtime_source_manifest") != runtime_source_manifest(config_path):
        raise C24CProtocolError("C2.4c runtime source 在冻结后发生变化")
    if payload.get("benchmark") != verify_frozen_benchmark(config_path):
        raise C24CProtocolError("v2.1 benchmark/AI 审核在冻结后发生变化")
    return payload


def phase_paths(
    config_path: str | Path,
    phase: str,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    if phase not in PHASES:
        raise ValueError(f"未知 C2.4c phase：{phase}")
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    return (
        artifact_dir / f"{phase}_started.json",
        artifact_dir / f"{phase}_completed.json",
    )


def assert_phase_can_start(
    config_path: str | Path,
    phase: str,
    *,
    output_dir: str | Path | None = None,
) -> None:
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    if (artifact_dir / "protocol_incident.json").exists():
        raise C24CProtocolError("C2.4c 已存在 protocol incident，禁止继续")
    started, completed = phase_paths(config_path, phase, output_dir=output_dir)
    if started.exists() or completed.exists():
        raise C24CProtocolError(f"{phase} 已开始或完成，禁止重跑")
    required = {
        "development": "calibration_completed.json",
        "locked_scoring": "development_completed.json",
    }.get(phase)
    if required and not (artifact_dir / required).is_file():
        raise C24CProtocolError(f"{phase} 的前置阶段尚未完成")


def mark_phase_started(
    config_path: str | Path,
    phase: str,
    details: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    assert_phase_can_start(config_path, phase, output_dir=output_dir)
    started, _ = phase_paths(config_path, phase, output_dir=output_dir)
    write_json(
        started,
        {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "phase": phase,
            "status": f"{phase}_started_once",
            "git_commit": git_state(config_path)["commit"],
            **dict(details),
        },
    )
    return started


def mark_phase_completed(
    config_path: str | Path,
    phase: str,
    details: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    started, completed = phase_paths(config_path, phase, output_dir=output_dir)
    if not started.is_file() or completed.exists():
        raise C24CProtocolError(f"{phase} completion marker 状态非法")
    write_json(
        completed,
        {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "phase": phase,
            "status": f"{phase}_completed_once",
            "git_commit": git_state(config_path)["commit"],
            **dict(details),
        },
    )
    return completed


def write_protocol_incident_once(
    config_path: str | Path,
    *,
    phase: str,
    violation: str,
    output_dir: str | Path | None = None,
) -> None:
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    path = artifact_dir / "protocol_incident.json"
    if path.exists():
        return
    write_json(
        path,
        {
            **base_status(),
            "status": "protocol_incident_locked_v2_1_invalidated",
            "phase": phase,
            "violation": str(violation),
            "current_locked_audit_v2_1_valid": False,
            "future_independent_audit_requires_new_namespace": True,
            "git": git_state(config_path),
        },
    )


def read_jsonl_split(
    config_path: str | Path,
    split: str,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    values = load_config(config_path)
    declaration = values["frozen_benchmark"]["data"].get(split)
    if not isinstance(declaration, Mapping):
        raise C24CProtocolError(f"未知 v2.1 split：{split}")
    path = resolve_path(config_path, declaration["path"])
    if sha256_file(path) != expected_sha256 or expected_sha256 != declaration["sha256"]:
        raise C24CProtocolError(f"{split} bytes 在读取前发生变化")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = strict_json_loads(line, label=f"{split}:{line_number}")
            if not isinstance(value, Mapping):
                raise C24CProtocolError(f"{split}:{line_number} 不是 object")
            row = dict(value)
            if (
                row.get("split") != split
                or row.get("benchmark_version") != BENCHMARK_VERSION
                or row.get("answer_free") is not True
                or row.get("contains_private_answer") is not False
                or "answer" in row
            ):
                raise C24CProtocolError(f"{split}:{line_number} schema/安全字段非法")
            rows.append(row)
    if len(rows) != expected_rows or expected_rows != int(declaration["rows"]):
        raise C24CProtocolError(f"{split} row count 不匹配")
    if sha256_file(path) != expected_sha256:
        raise C24CProtocolError(f"{split} bytes 在读取后发生变化")
    return rows


def read_declared_split(
    config_path: str | Path, split: str
) -> list[dict[str, Any]]:
    declaration = load_config(config_path)["frozen_benchmark"]["data"][split]
    return read_jsonl_split(
        config_path,
        split,
        expected_sha256=str(declaration["sha256"]),
        expected_rows=int(declaration["rows"]),
    )


def exact_file_inventory(
    directory: str | Path, *, exclude: Sequence[str] = ()
) -> list[dict[str, Any]]:
    root = Path(directory)
    excluded = set(exclude)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


__all__ = [
    "BENCHMARK_VERSION",
    "C24CProtocolError",
    "EVALUATION_MODE",
    "SCHEMA_VERSION",
    "STAGE_NAME",
    "STATUS_VALUES",
    "assert_phase_can_start",
    "base_status",
    "canonical_sha256",
    "exact_file_inventory",
    "git_state",
    "load_config",
    "load_json",
    "mark_phase_completed",
    "mark_phase_started",
    "output_paths",
    "read_declared_split",
    "resolve_path",
    "runtime_source_manifest",
    "seal_benchmark",
    "sha256_file",
    "verify_benchmark_seal",
    "verify_frozen_benchmark",
    "write_json",
    "write_protocol_incident_once",
]
