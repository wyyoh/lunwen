"""F2B 的冻结、一次性 split 打开和严格 artifact 协议。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from keyed_gram.authsynth_shared import canonical_digest

STAGE = "F2B-blind-cgar"
SPLITS = ("train", "calibration", "development", "locked_test")
EXPECTED_ARTIFACT_FILES = frozenset(
    {
        "aggregate_metrics.csv",
        "analyzer_freeze_manifest.json",
        "benchmark_manifest.json",
        "calibration_results.csv",
        "collision_audit.json",
        "development_results.csv",
        "information_boundary_audit.json",
        "locked_test_results.csv",
        "per_category_metrics.csv",
        "protocol_status.json",
        "refinement_metrics.json",
        "refinement_trace_digests.csv",
        "resolved_config.json",
        "sensitive_material_scan.json",
        "source_sha256_manifest.json",
        "split_manifests/calibration.json",
        "split_manifests/development.json",
        "split_manifests/locked_test.json",
        "split_manifests/train.json",
        "stage_f2b_summary.json",
        "upstream_frozen_sha256.json",
        "variant_registry.json",
    }
)


class F2BProtocolError(RuntimeError):
    """F2B 的路径、冻结、运行次数或 manifest 违反协议。"""


def repo_root(config_path: str | Path) -> Path:
    target = Path(config_path).resolve()
    for parent in (target.parent, *target.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise F2BProtocolError("无法解析仓库根目录")


def safe_relative(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise F2BProtocolError(f"{label} 必须是安全相对路径")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise F2BProtocolError(f"JSON 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def write_json(path: str | Path, value: Any) -> None:
    _finite(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            _finite(row)
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2BProtocolError(f"duplicate JSON key：{key}")
        result[key] = value
    return result


def read_json(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                F2BProtocolError(f"{label} 非法常量：{item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise F2BProtocolError(f"无法读取 {label}") from exc
    if not isinstance(value, dict):
        raise F2BProtocolError(f"{label} 顶层必须为 object")
    _finite(value)
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise F2BProtocolError("无法读取 F2B config") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise F2BProtocolError("F2B config schema 非法")
    if value.get("stage") != STAGE:
        raise F2BProtocolError("F2B stage 不匹配")
    boundary = value.get("research_boundary", {})
    forbidden_true = (
        "analyzer_concrete_implementation_access",
        "analyzer_gold_contract_access",
        "analyzer_gold_policy_access",
        "analyzer_mutation_label_access",
        "llm_judge_used",
        "real_tool_execution_used",
        "real_agent_used",
        "private_data_used",
        "arbitrary_program_completeness_claimed",
        "external_baseline_superiority_claimed",
    )
    if any(boundary.get(key) is not False for key in forbidden_true):
        raise F2BProtocolError("F2B research boundary 未固定 false")
    if value["methods"].get("selected_method") != "AuthSynth-CGAR":
        raise F2BProtocolError("F2B selected method 未预注册")
    if value["benchmark"].get("expected_case_count") != 270:
        raise F2BProtocolError("F2B case count 未预注册为 270")
    for key in ("data_dir", "artifact_dir", "runtime_dir", "report_path"):
        safe_relative(value["outputs"][key], f"outputs.{key}")
    return value


def git_state(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)

    def run(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=root, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("symbolic-ref", "--short", "HEAD"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def verify_frozen_files(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for group in ("frozen_upstream", "frozen_analyzer"):
        for item in values[group]:
            relative = safe_relative(item["path"], f"{group}.path")
            target = root / relative
            observed = sha256_file(target) if target.is_file() else None
            if observed != item["sha256"]:
                raise F2BProtocolError(f"冻结 hash mismatch：{relative}")
            files.append({"path": relative.as_posix(), "sha256": observed})
    analyzer_commit = values["protocol"]["analyzer_freeze_commit"]
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", analyzer_commit, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise F2BProtocolError("HEAD 不是 analyzer freeze commit 的后代")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "analyzer_freeze_commit": analyzer_commit,
        "files": files,
        "f1_frozen_assets_modified": False,
        "f2a_frozen_assets_modified": False,
        "f1_locked_test_scored": False,
        "f2a_locked_test_rerun": False,
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def output_paths(
    config_path: str | Path, values: Mapping[str, Any]
) -> tuple[Path, Path, Path, Path]:
    root = repo_root(config_path)
    return tuple(
        root / safe_relative(values["outputs"][key], f"outputs.{key}")
        for key in ("data_dir", "artifact_dir", "runtime_dir", "report_path")
    )  # type: ignore[return-value]


def formal_preflight(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    state = git_state(config_path)
    if state["tracked_dirty"]:
        raise F2BProtocolError("tracked worktree 非干净，拒绝 formal build")
    if state["branch"] != values["protocol"]["required_branch"]:
        raise F2BProtocolError("formal build 分支不匹配")
    for target in output_paths(config_path, values):
        if target.exists():
            raise F2BProtocolError(f"formal output 已存在，拒绝重复：{target}")
    return {"git": state, "frozen": verify_frozen_files(config_path, values)}


def mark_started(runtime_dir: Path, git: Mapping[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        runtime_dir / "protocol_state.json",
        {
            "schema_version": 1,
            "formal_build": {"status": "started", **dict(git)},
            "phases": {
                phase: 0
                for phase in ("prepare", "calibration", "development", "locked_test")
            },
        },
    )


def record_phase(runtime_dir: Path, phase: str) -> None:
    path = runtime_dir / "protocol_state.json"
    state = read_json(path, label="F2B protocol state")
    order = ("prepare", "calibration", "development", "locked_test")
    if phase not in order or state["phases"].get(phase) != 0:
        raise F2BProtocolError(f"非法或重复 phase：{phase}")
    index = order.index(phase)
    if any(state["phases"].get(item) != 1 for item in order[:index]):
        raise F2BProtocolError(f"{phase} 前序 phase 未完成")
    state["phases"][phase] = 1
    write_json(path, state)


def mark_completed(runtime_dir: Path, summary_sha256: str) -> None:
    path = runtime_dir / "protocol_state.json"
    state = read_json(path, label="F2B protocol state")
    if any(value != 1 for value in state["phases"].values()):
        raise F2BProtocolError("F2B phases 未全部且仅运行一次")
    state["formal_build"]["status"] = "completed"
    state["formal_build"]["summary_sha256"] = summary_sha256
    write_json(path, state)


def artifact_manifest(artifact_dir: Path, report_path: Path) -> dict[str, Any]:
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
        "report": {
            "path": report_path.name,
            "size_bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def validate_artifact_inventory(artifact_dir: Path, report_path: Path) -> None:
    manifest = read_json(
        artifact_dir / "artifact_sha256_manifest.json", label="F2B artifact manifest"
    )
    actual = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    }
    expected = {item["path"] for item in manifest["files"]}
    if actual != EXPECTED_ARTIFACT_FILES or actual != expected:
        raise F2BProtocolError("F2B artifact inventory mismatch")
    for item in manifest["files"]:
        target = artifact_dir / safe_relative(item["path"], "artifact path")
        if sha256_file(target) != item["sha256"]:
            raise F2BProtocolError("F2B artifact hash mismatch")
    if sha256_file(report_path) != manifest["report"]["sha256"]:
        raise F2BProtocolError("F2B report hash mismatch")


def validate_strict_serialization(artifact_dir: Path, data_dir: Path) -> None:
    for path in (*artifact_dir.rglob("*.json"),):
        read_json(path, label=path.name)
    for path in data_dir.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(
                line,
                object_pairs_hook=_pairs,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    F2BProtocolError(item)
                ),
            )
