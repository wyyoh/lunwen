"""AuthSynth F2A 的冻结、一次性评分和 artifact 完整性协议。"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .tool_effects.types import canonical_digest

STAGE = "F2A-autheffect-feasibility"
SCHEMA_VERSION = 1

EXPECTED_ARTIFACT_FILES = frozenset(
    {
        "analyzer_freeze_manifest.json",
        "benchmark_manifest.json",
        "calibration_sanity.csv",
        "case_results.csv",
        "cgar_refinement_summary.json",
        "collision_audit.json",
        "development_results.csv",
        "exact_shield_summary.json",
        "feasibility_gate.json",
        "locked_test_results.csv",
        "method_metrics.csv",
        "per_category_metrics.csv",
        "protocol_status.json",
        "resolved_config.json",
        "sensitive_material_scan.json",
        "source_sha256_manifest.json",
        "split_manifests/calibration.json",
        "split_manifests/development.json",
        "split_manifests/locked_test.json",
        "split_manifests/train.json",
        "stage_f2a_summary.json",
        "upstream_f1_sha256.json",
        "variant_registry.json",
    }
)


class F2AProtocolError(RuntimeError):
    """F2A 路径、冻结、运行次数或 manifest 违反协议。"""


def repo_root(config_path: str | Path) -> Path:
    target = Path(config_path).resolve()
    for parent in (target.parent, *target.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise F2AProtocolError("无法解析仓库根目录")


def safe_relative(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise F2AProtocolError(f"{label} 必须是安全相对路径")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise F2AProtocolError(f"JSON 含 NaN/Infinity：{path}")
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
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2AProtocolError(f"duplicate JSON key：{key}")
        result[key] = value
    return result


def read_json(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                F2AProtocolError(f"{label} 非法常量：{item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise F2AProtocolError(f"无法读取 {label}") from exc
    if not isinstance(value, dict):
        raise F2AProtocolError(f"{label} 顶层必须为 object")
    _finite(value)
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise F2AProtocolError("无法读取 F2A config") from exc
    if not isinstance(value, dict):
        raise F2AProtocolError("F2A config 顶层必须为 object")
    if value.get("schema_version") != 1 or value.get("stage") != STAGE:
        raise F2AProtocolError("F2A schema/stage 不匹配")
    boundary = value.get("research_boundary")
    protocol = value.get("protocol")
    final = value.get("final_status")
    if not all(isinstance(item, Mapping) for item in (boundary, protocol, final)):
        raise F2AProtocolError("F2A boundary/protocol/final 缺失")
    required_false = (
        "arbitrary_program_completeness_claimed",
        "ccf_b_novelty_claimed",
        "llm_judge_used",
        "real_tool_execution_used",
        "real_agent_used",
        "private_data_used",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise F2AProtocolError("研究边界禁止项必须固定 false")
    protocol_false = (
        "formal_build_repeat_allowed",
        "f1_locked_test_scoring_allowed",
        "benchmark_regeneration_after_audit_allowed",
        "threshold_change_after_development_allowed",
        "threshold_change_after_locked_allowed",
        "private_data_allowed",
        "private_answer_allowed",
        "private_value_allowed",
        "real_credential_allowed",
        "lm_answer_injection_allowed",
        "key_attack_allowed",
    )
    if any(protocol.get(key) is not False for key in protocol_false):
        raise F2AProtocolError("F2A protocol 禁止项必须固定 false")
    if (
        protocol.get("formal_benchmark_seed_source") != "code_freeze_commit"
        or protocol.get("preflight_fixture_is_formal_benchmark") is not False
    ):
        raise F2AProtocolError("正式 benchmark 必须绑定 code-freeze commit")
    final_false = (
        "ready_for_stage_f2b",
        "additional_d_series_stages_allowed",
        "private_data_experiment_allowed",
        "f1_frozen_assets_modified",
        "f1_locked_test_scored",
        "real_tool_completeness_validated",
        "formal_model_completed",
        "private_value_memory_ready",
        "original_c3_allowed",
        "c3_eligible",
    )
    if any(final.get(key) is not False for key in final_false):
        raise F2AProtocolError("正式构建前 final gate 必须固定 false")
    if (
        final.get("d_series_status") != "completed"
        or final.get("semantic_router_authorization_research") != "stopped"
    ):
        raise F2AProtocolError("上游冻结研究状态不匹配")
    if value["benchmark"].get("ground_truth") != (
        "finite_transition_system_and_forbidden_state_predicate"
    ):
        raise F2AProtocolError("ground truth 必须来自有限 transition system")
    if value["benchmark"].get("expected_case_count") != 150:
        raise F2AProtocolError("feasibility case 数必须预注册为 150")
    if value["methods"].get("selected_method") != "AuthSynth-CGAR":
        raise F2AProtocolError("正式方法必须预注册为 AuthSynth-CGAR")
    for key in ("data_dir", "artifact_dir", "runtime_dir", "report_path"):
        safe_relative(value["outputs"][key], f"outputs.{key}")
    frozen = value.get("frozen_f1")
    if not isinstance(frozen, list) or len(frozen) < 10:
        raise F2AProtocolError("frozen_f1 清单不完整")
    forbidden_fragments = (
        "private_answer",
        "private_value",
        "feature_cache",
        "confirmation",
        "artifacts/keys",
    )
    for item in frozen:
        relative = safe_relative(item.get("path", ""), "frozen_f1.path")
        if any(part in relative.as_posix().casefold() for part in forbidden_fragments):
            raise F2AProtocolError("frozen_f1 禁止引用 private/secret source")
    return value


def git_state(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)

    def run(*arguments: str) -> str:
        return subprocess.check_output(("git", *arguments), cwd=root, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("symbolic-ref", "--short", "HEAD"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def verify_frozen_f1(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    root = repo_root(config_path)
    required_commit = str(values["protocol"]["required_f1_commit"])
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", required_commit, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise F2AProtocolError("当前 HEAD 不是冻结 F1 commit 的后代")
    files = []
    for item in values["frozen_f1"]:
        relative = safe_relative(item["path"], "frozen_f1.path")
        target = root / relative
        observed = sha256_file(target) if target.is_file() else None
        if observed != item["sha256"]:
            raise F2AProtocolError(f"F1 frozen hash mismatch：{relative}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": observed,
                "size_bytes": target.stat().st_size,
            }
        )
    source_manifest = read_json(
        root / "artifacts/stage_f1/source_sha256_manifest.json",
        label="F1 source manifest",
    )
    source_failures = []
    for entry in source_manifest["files"]:
        target = root / safe_relative(entry["path"], "F1 source path")
        if not target.is_file() or sha256_file(target) != entry["sha256"]:
            source_failures.append(entry["path"])
    artifact_root = root / "artifacts/stage_f1"
    artifact_manifest = read_json(
        artifact_root / "artifact_sha256_manifest.json",
        label="F1 artifact manifest",
    )
    artifact_failures = []
    expected_inventory = set()
    for entry in artifact_manifest["files"]:
        relative = safe_relative(entry["path"], "F1 artifact path")
        expected_inventory.add(relative.as_posix())
        target = artifact_root / relative
        if not target.is_file() or sha256_file(target) != entry["sha256"]:
            artifact_failures.append(relative.as_posix())
    actual_inventory = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    }
    if source_failures or artifact_failures or actual_inventory != expected_inventory:
        raise F2AProtocolError("F1 source/artifact manifest 校验失败")
    summary = read_json(
        root / "artifacts/stage_f1/stage_f1_summary.json",
        label="F1 summary",
    )
    required_status = {
        "authzroutebench_status": "passed_preregistered_construction",
        "policy_oracle_status": "passed",
        "policy_ground_truth_is_deterministic": True,
        "semantic_proposal_authorizes_execution": False,
        "locked_test_generated": True,
        "locked_test_scored": False,
        "ready_for_stage_f2": True,
    }
    if any(summary.get(key) != expected for key, expected in required_status.items()):
        raise F2AProtocolError("F1 正式状态不满足冻结前提")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "required_f1_commit": required_commit,
        "files": files,
        "source_manifest_entry_count": len(source_manifest["files"]),
        "artifact_manifest_entry_count": len(artifact_manifest["files"]),
        "f1_frozen_assets_modified": False,
        "f1_locked_test_case_content_read": False,
        "f1_locked_test_scored": False,
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


_SOURCE_PATHS = (
    "AUTHSYNTH_RESEARCH_PLAN.md",
    "configs/stage_f2a.yaml",
    "docs/f2a/COUNTEREXAMPLES.md",
    "docs/f2a/EXPERIMENT_PROTOCOL.md",
    "docs/f2a/FORMAL_MODEL.md",
    "docs/f2a/RELATED_WORK_GAP.md",
    "src/keyed_gram/tool_effects/__init__.py",
    "src/keyed_gram/tool_effects/benchmark.py",
    "src/keyed_gram/tool_effects/contracts.py",
    "src/keyed_gram/tool_effects/ir.py",
    "src/keyed_gram/tool_effects/methods.py",
    "src/keyed_gram/tool_effects/metrics.py",
    "src/keyed_gram/tool_effects/shield.py",
    "src/keyed_gram/tool_effects/types.py",
    "src/keyed_gram/stage_f2a_autheffect.py",
    "src/keyed_gram/stage_f2a_autheffect_protocol.py",
    "tests/test_stage_f2a_autheffect.py",
    "tests/test_tool_effect_benchmark.py",
    "tests/test_tool_effect_contracts.py",
    "tests/test_tool_effect_ir.py",
    "tests/test_tool_effect_methods.py",
    "tests/test_tool_effect_shield.py",
)


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for relative in _SOURCE_PATHS:
        target = root / relative
        if not target.is_file():
            raise F2AProtocolError(f"F2A runtime source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    payload = {"schema_version": 1, "files": files}
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def output_paths(
    config_path: str | Path, values: Mapping[str, Any]
) -> tuple[Path, Path, Path, Path]:
    root = repo_root(config_path)
    outputs = values["outputs"]
    return tuple(
        root / safe_relative(outputs[key], f"outputs.{key}")
        for key in ("data_dir", "artifact_dir", "runtime_dir", "report_path")
    )  # type: ignore[return-value]


def formal_preflight(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    state = git_state(config_path)
    if state["tracked_dirty"]:
        raise F2AProtocolError("tracked worktree 非干净，拒绝 formal build")
    if state["branch"] != values["protocol"]["required_branch"]:
        raise F2AProtocolError("formal build 分支不匹配")
    data_dir, artifact_dir, runtime_dir, report_path = output_paths(config_path, values)
    for target in (data_dir, artifact_dir, runtime_dir, report_path):
        if target.exists():
            raise F2AProtocolError(f"formal output 已存在，拒绝重复：{target}")
    return {
        "git": state,
        "frozen_f1": verify_frozen_f1(config_path, values),
    }


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
            "phases": {
                "prepare": 0,
                "calibration": 0,
                "development": 0,
                "locked_test": 0,
            },
        },
    )


def record_phase(runtime_dir: Path, phase: str) -> None:
    state_path = runtime_dir / "protocol_state.json"
    state = read_json(state_path, label="F2A protocol state")
    phases = state.get("phases")
    if phase not in {"prepare", "calibration", "development", "locked_test"}:
        raise F2AProtocolError("未知 protocol phase")
    if not isinstance(phases, dict):
        raise F2AProtocolError("protocol phase state 缺失")
    if phases.get(phase) != 0:
        raise F2AProtocolError(f"{phase} 已运行，拒绝重复")
    order = ("prepare", "calibration", "development", "locked_test")
    index = order.index(phase)
    if any(phases.get(previous) != 1 for previous in order[:index]):
        raise F2AProtocolError(f"{phase} 前序 phase 未完成")
    phases[phase] = 1
    write_json(state_path, state)


def mark_completed(runtime_dir: Path, summary_digest: str) -> None:
    state_path = runtime_dir / "protocol_state.json"
    state = read_json(state_path, label="F2A protocol state")
    if any(value != 1 for value in state["phases"].values()):
        raise F2AProtocolError("F2A phase 未全部且仅运行一次")
    state["formal_build"]["status"] = "completed"
    state["formal_build"]["summary_sha256"] = summary_digest
    write_json(state_path, state)


def mark_failed(runtime_dir: Path, reason: str) -> None:
    state_path = runtime_dir / "protocol_state.json"
    if not state_path.is_file():
        return
    state = read_json(state_path, label="F2A protocol state")
    state["formal_build"]["status"] = "failed"
    state["formal_build"]["failure_reason"] = reason
    write_json(state_path, state)


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
        artifact_dir / "artifact_sha256_manifest.json",
        label="F2A artifact manifest",
    )
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    }
    if actual != EXPECTED_ARTIFACT_FILES:
        raise F2AProtocolError("artifact inventory 不符合预注册 schema")
    if actual != expected:
        raise F2AProtocolError("artifact inventory mismatch")
    for entry in manifest["files"]:
        target = artifact_dir / safe_relative(entry["path"], "artifact path")
        if sha256_file(target) != entry["sha256"]:
            raise F2AProtocolError("artifact hash mismatch")
    report = manifest["report"]
    if (
        report_path.name != report["path"]
        or sha256_file(report_path) != report["sha256"]
    ):
        raise F2AProtocolError("report hash mismatch")


def validate_data_inventory(
    data_dir: Path,
    split_manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        f"{split}.jsonl"
        for split in ("train", "calibration", "development", "locked_test")
    }
    actual = {
        path.relative_to(data_dir).as_posix()
        for path in data_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected or set(split_manifests) != {
        "train",
        "calibration",
        "development",
        "locked_test",
    }:
        raise F2AProtocolError("benchmark data inventory mismatch")
    for split, manifest in split_manifests.items():
        target = data_dir / f"{split}.jsonl"
        if (
            manifest.get("data_sha256") != sha256_file(target)
            or manifest.get("data_size_bytes") != target.stat().st_size
        ):
            raise F2AProtocolError("benchmark split manifest mismatch")


def validate_strict_serialization(artifact_dir: Path, data_dir: Path) -> dict[str, int]:
    json_count = 0
    jsonl_row_count = 0
    for path in sorted(artifact_dir.rglob("*.json")):
        read_json(path, label=f"artifact JSON {path.name}")
        json_count += 1
    for path in sorted(data_dir.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_pairs,
                    parse_constant=lambda item: (_ for _ in ()).throw(
                        F2AProtocolError(f"JSONL 非法常量：{item}")
                    ),
                )
            except json.JSONDecodeError as exc:
                raise F2AProtocolError(
                    f"JSONL 解析失败：{path.name}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise F2AProtocolError("JSONL row 顶层必须为 object")
            _finite(value)
            jsonl_row_count += 1
    return {
        "strict_json_file_count": json_count,
        "strict_jsonl_row_count": jsonl_row_count,
    }


def assert_no_sensitive_material(paths: Sequence[Path]) -> dict[str, Any]:
    forbidden = (
        b"-----BEGIN PRIVATE KEY-----",
        b"-----BEGIN OPENSSH PRIVATE KEY-----",
        b'"capability_token":"',
        b'"private_answer":"',
        b'"private_value":"',
        b'"real_credential":"',
    )
    hits = []
    for path in paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        if any(marker in data for marker in forbidden):
            hits.append(path.as_posix())
    if hits:
        raise F2AProtocolError("artifact 含禁止的 private/secret material")
    return {
        "scanned_file_count": len(paths),
        "sensitive_material_occurrence_count": 0,
    }
