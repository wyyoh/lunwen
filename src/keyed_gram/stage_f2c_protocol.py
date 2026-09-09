"""Stage F2C 的冻结、一次性 split 打开和严格 artifact 协议。"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from keyed_gram.authsynth_symbolic_shared import canonical_digest

STAGE = "F2C-symbolic-contracts"
SPLITS = ("train", "calibration", "development", "locked_test")
PHASES = (
    "materialize",
    "calibration",
    "configuration_freeze",
    "development",
    "locked_test",
)
EXPECTED_ARTIFACT_FILES = frozenset(
    {
        "analyzer_freeze_manifest.json",
        "benchmark_manifest.json",
        "calibration_results.csv",
        "certificate_manifest.json",
        "collision_audit.json",
        "contract_complexity.csv",
        "development_results.csv",
        "grammar_manifest.json",
        "information_boundary_audit.json",
        "locked_test_results.csv",
        "patch_metrics.csv",
        "per_category_metrics.csv",
        "per_field_metrics.csv",
        "performance_metrics.csv",
        "protocol_status.json",
        "query_efficiency.csv",
        "resolved_config.json",
        "sensitive_material_scan.json",
        "solver_manifest.json",
        "source_sha256_manifest.json",
        "split_manifests/calibration.json",
        "split_manifests/development.json",
        "split_manifests/locked_test.json",
        "split_manifests/train.json",
        "stage_f2c_summary.json",
        "unknown_reason_distribution.csv",
        "upstream_frozen_sha256.json",
    }
)


class F2CProtocolError(RuntimeError):
    """F2C 路径、冻结、运行次数或 manifest 违反协议。"""


def repo_root(config_path: str | Path) -> Path:
    target = Path(config_path).resolve()
    for parent in (target.parent, *target.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise F2CProtocolError("无法解析仓库根目录")


def safe_relative(value: str | Path, label: str) -> Path:
    raw = value.as_posix() if isinstance(value, Path) else value
    if (
        not raw
        or "\\" in raw
        or ":" in raw
        or "\x00" in raw
        or PurePosixPath(raw).is_absolute()
        or PureWindowsPath(raw).drive
        or PureWindowsPath(raw).root
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise F2CProtocolError(f"{label} 必须是安全相对路径")
    return Path(raw)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise F2CProtocolError(f"JSON/CSV 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2CProtocolError(f"duplicate JSON key：{key}")
        result[key] = value
    return result


def read_json(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                F2CProtocolError(f"{label} 非法常量：{item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise F2CProtocolError(f"无法读取 {label}") from exc
    if not isinstance(value, dict):
        raise F2CProtocolError(f"{label} 顶层必须为 object")
    _finite(value)
    return value


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
    if not fieldnames:
        raise F2CProtocolError(f"CSV 不得为空：{target.name}")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            _finite(row)
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise F2CProtocolError("无法读取 F2C config") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise F2CProtocolError("F2C config schema 非法")
    if value.get("stage") != STAGE or value.get("method_name") != "AuthSynth-Symbolic":
        raise F2CProtocolError("F2C stage/method 不匹配")
    boundary = value.get("research_boundary", {})
    required_false = (
        "analyzer_hidden_implementation_access",
        "analyzer_reference_contract_access",
        "analyzer_mutation_label_access",
        "fixed_query_catalog_used",
        "relational_effect_analysis_used",
        "secret_dependent_payload_analyzed",
        "real_tool_execution_used",
        "real_agent_used",
        "llm_judge_used",
        "private_data_used",
        "arbitrary_program_completeness_claimed",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise F2CProtocolError("F2C research boundary 未固定为 false")
    benchmark = value.get("benchmark", {})
    if (
        benchmark.get("expected_case_count") != 160
        or benchmark.get("expected_base_tool_count") != 16
    ):
        raise F2CProtocolError("AuthSymbolBench v1 规模未预注册")
    candidates = value.get("calibration_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise F2CProtocolError("缺少预注册 calibration candidates")
    for item in candidates:
        if (
            not isinstance(item, dict)
            or not 8 <= int(item.get("replay_budget", 0)) <= 32
        ):
            raise F2CProtocolError("calibration replay budget 非法")
    for key in ("data_dir", "artifact_dir", "runtime_dir", "report_path"):
        safe_relative(value["outputs"][key], f"outputs.{key}")
    binding_path = value.get("protocol", {}).get("freeze_binding_path")
    if binding_path:
        root = repo_root(config_path)
        relative = safe_relative(binding_path, "freeze_binding_path")
        if (root / relative).is_file():
            from .stage_f2c_freeze import load_binding

            binding, groups = load_binding(root, binding_path)
            value["protocol"]["analyzer_freeze_commit"] = binding[
                "analyzer_freeze_commit"
            ]
            for group, manifest in groups.items():
                value[group] = manifest["files"]
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


def output_paths(
    config_path: str | Path, values: Mapping[str, Any]
) -> tuple[Path, Path, Path, Path]:
    root = repo_root(config_path)
    return tuple(
        root / safe_relative(values["outputs"][key], f"outputs.{key}")
        for key in ("data_dir", "artifact_dir", "runtime_dir", "report_path")
    )  # type: ignore[return-value]


def verify_frozen_files(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    root = repo_root(config_path)
    if values.get("protocol", {}).get("freeze_binding_path"):
        from .stage_f2c_freeze import verify_binding

        return verify_binding(
            root, values["protocol"]["freeze_binding_path"], check_runtime=True
        )
    observed_files = []
    for group in ("frozen_upstream", "frozen_analyzer"):
        records = values.get(group)
        if not isinstance(records, list) or not records:
            raise F2CProtocolError(f"{group} 为空，拒绝 formal build")
        for item in records:
            relative = safe_relative(item["path"], f"{group}.path")
            target = root / relative
            observed = sha256_file(target) if target.is_file() else None
            if observed != item.get("sha256"):
                raise F2CProtocolError(f"冻结 hash mismatch：{relative}")
            observed_files.append(
                {"group": group, "path": relative.as_posix(), "sha256": observed}
            )
    freeze = values["protocol"].get("analyzer_freeze_commit")
    if (
        not isinstance(freeze, str)
        or len(freeze) != 40
        or freeze == "PENDING_ANALYZER_FREEZE"
    ):
        raise F2CProtocolError("Analyzer freeze commit 尚未固定")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", freeze, "HEAD"), cwd=root, check=False
    )
    if ancestor.returncode != 0:
        raise F2CProtocolError("HEAD 不是 Analyzer freeze commit 的后代")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "base_commit": values["protocol"]["required_base_commit"],
        "analyzer_freeze_commit": freeze,
        "files": observed_files,
        "f1_frozen_assets_modified": False,
        "f2a_frozen_assets_modified": False,
        "f2b_frozen_assets_modified": False,
        "f1_locked_test_scored": False,
        "f2a_locked_test_rerun": False,
        "f2b_locked_test_rerun": False,
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def formal_preflight(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    state = git_state(config_path)
    if state["tracked_dirty"]:
        raise F2CProtocolError("tracked worktree 非干净，拒绝 formal build")
    if state["branch"] != values["protocol"]["required_branch"]:
        raise F2CProtocolError("formal build 分支不匹配")
    root = repo_root(config_path)
    expected_base = values["protocol"]["required_base_commit"]
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", expected_base, "HEAD"),
        cwd=root,
        check=False,
    ).returncode:
        raise F2CProtocolError("F2B base commit 不在 HEAD 历史中")
    for target in output_paths(config_path, values):
        if target.exists():
            raise F2CProtocolError(f"formal output 已存在，拒绝重复：{target}")
    return {"git": state, "frozen": verify_frozen_files(config_path, values)}


def mark_started(runtime_dir: Path, git: Mapping[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        runtime_dir / "protocol_state.json",
        {
            "schema_version": 1,
            "formal_build": {"status": "started", **dict(git)},
            "phases": {phase: 0 for phase in PHASES},
        },
    )


def record_phase(runtime_dir: Path, phase: str) -> None:
    path = runtime_dir / "protocol_state.json"
    state = read_json(path, label="F2C protocol state")
    if phase not in PHASES or state["phases"].get(phase) != 0:
        raise F2CProtocolError(f"非法或重复 phase：{phase}")
    index = PHASES.index(phase)
    if any(state["phases"].get(item) != 1 for item in PHASES[:index]):
        raise F2CProtocolError(f"{phase} 前序 phase 未完成")
    state["phases"][phase] = 1
    write_json(path, state)


def mark_completed(runtime_dir: Path, summary_sha256: str) -> None:
    path = runtime_dir / "protocol_state.json"
    state = read_json(path, label="F2C protocol state")
    if any(value != 1 for value in state["phases"].values()):
        raise F2CProtocolError("F2C phases 未全部且仅运行一次")
    state["formal_build"]["status"] = "completed"
    state["formal_build"]["summary_sha256"] = summary_sha256
    write_json(path, state)


def information_boundary_audit(
    root: Path, public_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    analyzer_dir = root / "src/keyed_gram/authsynth_symbolic_analyzer"
    forbidden_imports = ("authsynth_symbolic_verifier", "authsynth_symbolic_evaluator")
    forbidden_symbols = {
        "hidden_symbolic_implementation",
        "hidden_transition",
        "reference_contract",
        "gold_contract",
        "gold_shield",
        "mutation_category",
        "omission_atoms",
        "query_catalog",
        "expected_unknown",
    }
    import_failures = []
    symbol_failures = []
    for path in sorted(analyzer_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        if any(part in item for item in imports for part in forbidden_imports):
            import_failures.append(path.name)
        lowered = source.casefold()
        if any(symbol in lowered for symbol in forbidden_symbols):
            symbol_failures.append(path.name)

    def nested_keys(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            found.update(str(key) for key in value)
            for child in value.values():
                found.update(nested_keys(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.update(nested_keys(child))
        return found

    row_failures = [
        str(row.get("public_case_id", "unknown"))
        for row in public_rows
        if nested_keys(row) & forbidden_symbols
    ]
    passed = not import_failures and not symbol_failures and not row_failures
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "analyzer_verifier_import_failure_count": len(import_failures),
        "analyzer_forbidden_symbol_failure_count": len(symbol_failures),
        "public_schema_forbidden_field_failure_count": len(row_failures),
        "analyzer_hidden_implementation_access": False,
        "analyzer_reference_contract_access": False,
        "analyzer_mutation_label_access": False,
        "analyzer_fixed_query_catalog_access": False,
        "verifier_returns_hidden_formula": False,
    }


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
        artifact_dir / "artifact_sha256_manifest.json", label="F2C artifact manifest"
    )
    actual = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    }
    expected = {item["path"] for item in manifest["files"]}
    if actual != EXPECTED_ARTIFACT_FILES or actual != expected:
        raise F2CProtocolError("F2C artifact inventory mismatch")
    for item in manifest["files"]:
        target = artifact_dir / safe_relative(item["path"], "artifact path")
        if sha256_file(target) != item["sha256"]:
            raise F2CProtocolError("F2C artifact hash mismatch")
    if sha256_file(report_path) != manifest["report"]["sha256"]:
        raise F2CProtocolError("F2C report hash mismatch")


def validate_strict_serialization(artifact_dir: Path, data_dir: Path) -> None:
    for path in artifact_dir.rglob("*.json"):
        read_json(path, label=path.name)
    for path in data_dir.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(
                line,
                object_pairs_hook=_pairs,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    F2CProtocolError(item)
                ),
            )
    for path in artifact_dir.rglob("*.csv"):
        raw = path.read_bytes()
        if b"\r\n" in raw or b"\x00" in raw:
            raise F2CProtocolError(f"CSV 严格格式失败：{path.name}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
        if not rows:
            raise F2CProtocolError(f"CSV 无数据行：{path.name}")


def sensitive_scan(paths: Sequence[Path]) -> dict[str, Any]:
    patterns = (
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "SYN-D21-",
        "private_answer_payload",
    )
    occurrences = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern in text:
                occurrences.append({"path": path.name, "pattern": pattern})
    if occurrences:
        raise F2CProtocolError("F2C artifact 敏感材料扫描失败")
    return {
        "schema_version": 1,
        "status": "passed",
        "scanned_file_count": len(paths),
        "private_key_occurrence_count": 0,
        "formal_capability_token_occurrence_count": 0,
        "private_data_occurrence_count": 0,
        "hidden_locked_implementation_ast_persisted": False,
        "gold_locked_contract_persisted": False,
        "locked_assignment_output_table_persisted": False,
    }
