"""Stage F2 冻结、顺序一次性审计与 artifact 完整性协议。"""

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

from .authcap_policy import canonical_sha256

F2_STAGE = "F2-authcap-compiler"
F2_SCHEMA_VERSION = 1
F2_VARIANTS = ("B0", "B1", "B2", "B3", "A0", "A1")
_SHARED_F1_MUTABLE = frozenset({".gitattributes", "src/keyed_gram/cli.py"})


class F2ProtocolError(RuntimeError):
    """F2 路径、冻结状态、阶段顺序或 artifact 不满足协议。"""


def repo_root(config_path: str | Path) -> Path:
    target = Path(config_path).resolve()
    for parent in (target.parent, *target.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise F2ProtocolError("无法解析仓库根目录")


def safe_relative(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise F2ProtocolError(f"{label} 必须是安全相对路径")
    return path


def resolve_path(
    config_path: str | Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    return repo_root(config_path) / safe_relative(value, label)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise F2ProtocolError(f"JSON 含 NaN/Infinity：{path}")
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
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (list(rows[0]) if rows else []))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            _finite(row)
            writer.writerow(dict(row))


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise F2ProtocolError(f"duplicate JSON key：{key}")
        value[key] = child
    return value


def read_json(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                F2ProtocolError(f"{label} 非法常量：{item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise F2ProtocolError(f"无法读取 {label}") from exc
    if not isinstance(value, dict):
        raise F2ProtocolError(f"{label} 顶层必须是 object")
    _finite(value)
    return value


def load_config(
    config_path: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise F2ProtocolError("无法读取 F2 配置") from exc
    expected = f"{F2_STAGE}-smoke" if smoke else F2_STAGE
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != F2_SCHEMA_VERSION
        or value.get("stage") != expected
    ):
        raise F2ProtocolError("F2 config schema/stage 不匹配")
    protocol = value.get("protocol")
    final = value.get("final_status")
    if not isinstance(protocol, Mapping) or not isinstance(final, Mapping):
        raise F2ProtocolError("F2 protocol/final_status 缺失")
    prohibited = (
        "private_data_allowed",
        "memory_access_allowed",
        "tool_execution_allowed",
    )
    if any(protocol.get(name) is not False for name in prohibited):
        raise F2ProtocolError("F2 禁止项必须固定 false")
    if any(
        final.get(name) is not False
        for name in (
            "formal_model_completed",
            "real_agent_integration_completed",
            "private_value_memory_ready",
            "original_c3_allowed",
            "c3_eligible",
        )
    ):
        raise F2ProtocolError("F2 永久门禁必须固定 false")
    if smoke:
        if protocol.get("locked_test_allowed") is not False:
            raise F2ProtocolError("smoke 不得读取 locked_test")
        return value
    if tuple(value.get("variants", ())) != F2_VARIANTS:
        raise F2ProtocolError("F2 variants 或顺序被修改")
    if protocol.get("final_variant") != "A1":
        raise F2ProtocolError("最终 variant 必须预注册为 A1")
    if protocol.get("semantic_proposal_direct_authorization_allowed") is not False:
        raise F2ProtocolError("proposal 不得直接授权")
    if value["calibration"].get("safety_invariant_tuning_allowed") is not False:
        raise F2ProtocolError("calibration 不得放宽 safety invariant")
    for name in ("artifact_dir", "runtime_dir", "report_path"):
        safe_relative(value["outputs"][name], f"outputs.{name}")
    for item in value["f1_frozen"].values():
        if isinstance(item, Mapping) and "path" in item:
            path = safe_relative(item["path"], "f1_frozen.path")
            if any(
                fragment in path.as_posix().casefold()
                for fragment in (
                    "private_answer",
                    "private_value",
                    "feature_cache",
                    "confirmation",
                    "artifacts/keys",
                )
            ):
                raise F2ProtocolError("F2 禁止引用 private/confirmation source")
    for item in value["f1_frozen"]["splits"].values():
        safe_relative(item["path"], "f1_frozen.splits.path")
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


def _git_blob(root: Path, revision: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ("git", "show", f"{revision}:{path}"),
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        raise F2ProtocolError(f"F1 base blob 缺失：{path}") from exc


def verify_f1_frozen(
    config_path: str | Path,
    values: Mapping[str, Any],
    *,
    include_split_bytes: tuple[str, ...] = (
        "train",
        "calibration",
        "development",
        "locked_test",
    ),
) -> dict[str, Any]:
    """验证 F1 正式产物、base tree 与指定 split 的字节哈希。

    ``prepare`` 只哈希 development/locked，不解析 case 内容。
    """

    root = repo_root(config_path)
    frozen = values["f1_frozen"]
    files = []
    for name, item in frozen.items():
        if name == "splits":
            continue
        target = root / safe_relative(item["path"], f"f1_frozen.{name}")
        observed = sha256_file(target) if target.is_file() else None
        if observed != item["sha256"]:
            raise F2ProtocolError(f"F1 frozen hash mismatch：{name}")
        files.append({"path": item["path"], "sha256": observed})
    expected_base = values["protocol"]["expected_base_commit"]
    ancestor = subprocess.call(
        ("git", "merge-base", "--is-ancestor", expected_base, "HEAD"),
        cwd=root,
    )
    if ancestor != 0:
        raise F2ProtocolError("F1 最终提交不是当前 HEAD 祖先")
    source_manifest = read_json(
        root / frozen["source_manifest"]["path"],
        label="F1 source manifest",
    )
    for item in source_manifest["files"]:
        blob = _git_blob(root, expected_base, item["path"])
        if (
            hashlib.sha256(blob).hexdigest() != item["sha256"]
            or len(blob) != item["size_bytes"]
        ):
            raise F2ProtocolError(f"F1 base source manifest mismatch：{item['path']}")
        if item["path"] not in _SHARED_F1_MUTABLE:
            current = root / item["path"]
            if not current.is_file() or sha256_file(current) != item["sha256"]:
                raise F2ProtocolError(f"F1 frozen source 被修改：{item['path']}")
    artifact_manifest = read_json(
        root / frozen["artifact_manifest"]["path"],
        label="F1 artifact manifest",
    )
    artifact_root = root / "artifacts/stage_f1"
    for item in artifact_manifest["files"]:
        target = artifact_root / safe_relative(item["path"], "F1 artifact path")
        if (
            not target.is_file()
            or sha256_file(target) != item["sha256"]
            or target.stat().st_size != item["size_bytes"]
        ):
            raise F2ProtocolError(f"F1 artifact inventory mismatch：{item['path']}")
    split_hashes = {}
    for split in include_split_bytes:
        item = frozen["splits"][split]
        target = root / safe_relative(item["path"], f"F1 split {split}")
        observed = sha256_file(target) if target.is_file() else None
        if observed != item["sha256"]:
            raise F2ProtocolError(f"F1 split hash mismatch：{split}")
        split_hashes[split] = observed
    summary = read_json(
        root / frozen["summary"]["path"],
        label="F1 summary",
    )
    required = {
        "authzroutebench_status": "passed_preregistered_construction",
        "policy_oracle_status": "passed",
        "policy_ground_truth_is_deterministic": True,
        "semantic_proposal_authorizes_execution": False,
        "locked_test_generated": True,
        "locked_test_scored": False,
        "ready_for_stage_f2": True,
    }
    if any(summary.get(key) != child for key, child in required.items()):
        raise F2ProtocolError("F1 readiness 状态不满足 F2 前置条件")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "expected_base_commit": expected_base,
        "frozen_files": files,
        "split_sha256": split_hashes,
        "f1_source_files_checked_at_base": len(source_manifest["files"]),
        "f1_artifact_files_checked": len(artifact_manifest["files"]),
        "frozen_upstream_code_modified": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


_SOURCE_PATHS = (
    ".gitattributes",
    "configs/stage_f2.yaml",
    "configs/stage_f2_smoke.yaml",
    "src/keyed_gram/authcap_atoms.py",
    "src/keyed_gram/authcap_compiler.py",
    "src/keyed_gram/authcap_witness.py",
    "src/keyed_gram/authcap_capability.py",
    "src/keyed_gram/authcap_workflow.py",
    "src/keyed_gram/authcap_verifier.py",
    "src/keyed_gram/stage_f2.py",
    "src/keyed_gram/stage_f2_metrics.py",
    "src/keyed_gram/stage_f2_protocol.py",
    "src/keyed_gram/cli.py",
    "tests/test_authcap_atoms.py",
    "tests/test_authcap_compiler.py",
    "tests/test_authcap_witness.py",
    "tests/test_authcap_capability.py",
    "tests/test_authcap_workflow.py",
    "tests/test_authcap_verifier.py",
    "tests/test_stage_f2_metrics.py",
    "tests/test_stage_f2_protocol.py",
    "tests/f2_helpers.py",
)


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for relative in _SOURCE_PATHS:
        target = root / relative
        if not target.is_file():
            raise F2ProtocolError(f"F2 source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
    payload = {"schema_version": 1, "files": files}
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def output_paths(
    config_path: str | Path,
    values: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    return (
        resolve_path(
            config_path,
            values["outputs"]["artifact_dir"],
            label="artifact_dir",
        ),
        resolve_path(
            config_path,
            values["outputs"]["runtime_dir"],
            label="runtime_dir",
        ),
        resolve_path(
            config_path,
            values["outputs"]["report_path"],
            label="report_path",
        ),
    )


def formal_preflight(
    config_path: str | Path,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    state = git_state(config_path)
    if state["tracked_dirty"]:
        raise F2ProtocolError("tracked worktree 非干净，拒绝 formal prepare")
    if state["branch"] != values["protocol"]["required_branch"]:
        raise F2ProtocolError("F2 formal branch 不匹配")
    artifact_dir, runtime_dir, report_path = output_paths(config_path, values)
    for path in (artifact_dir, runtime_dir, report_path):
        if path.exists():
            raise F2ProtocolError(f"F2 formal output 已存在：{path}")
    upstream = verify_f1_frozen(config_path, values)
    return {"git": state, "upstream": upstream}


def initialize_protocol(
    runtime_dir: Path,
    *,
    git: Mapping[str, Any],
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        runtime_dir / "protocol_state.json",
        {
            "schema_version": 1,
            "stage": F2_STAGE,
            "code_freeze_commit": git["commit"],
            "branch": git["branch"],
            "current_phase": "initialized",
            "phases": {},
            "development_score_count": 0,
            "locked_test_score_count": 0,
            "locked_test_content_read_before_freeze": False,
        },
    )


def claim_phase(
    runtime_dir: Path,
    *,
    phase: str,
    expected_previous: str,
) -> dict[str, Any]:
    state_path = runtime_dir / "protocol_state.json"
    state = read_json(state_path, label="F2 protocol state")
    if state.get("current_phase") != expected_previous or phase in state["phases"]:
        raise F2ProtocolError(f"F2 phase 重复或乱序：{phase}")
    state["phases"][phase] = {"status": "started"}
    if phase == "development":
        state["development_score_count"] += 1
    if phase == "locked_test":
        state["locked_test_score_count"] += 1
    state["current_phase"] = f"{phase}:started"
    write_json(state_path, state)
    return state


def complete_phase(runtime_dir: Path, *, phase: str) -> None:
    state_path = runtime_dir / "protocol_state.json"
    state = read_json(state_path, label="F2 protocol state")
    if state.get("current_phase") != f"{phase}:started":
        raise F2ProtocolError(f"F2 phase 未处于 started：{phase}")
    state["phases"][phase]["status"] = "completed"
    state["current_phase"] = phase
    write_json(state_path, state)


def validate_final_protocol(runtime_dir: Path) -> dict[str, Any]:
    state = read_json(
        runtime_dir / "protocol_state.json",
        label="F2 protocol state",
    )
    expected = (
        "prepare",
        "calibration",
        "freeze",
        "development",
        "locked_test",
    )
    if (
        state.get("current_phase") != "locked_test"
        or set(state.get("phases", ())) != set(expected)
        or any(state["phases"][item]["status"] != "completed" for item in expected)
        or state.get("development_score_count") != 1
        or state.get("locked_test_score_count") != 1
        or state.get("locked_test_content_read_before_freeze") is not False
    ):
        raise F2ProtocolError("F2 一次性阶段状态不满足最终要求")
    return state


def artifact_manifest(
    artifact_dir: Path,
    *,
    report_path: Path,
    expected_files: Sequence[str],
) -> dict[str, Any]:
    observed = sorted(
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    )
    if observed != sorted(expected_files):
        raise F2ProtocolError(
            f"F2 artifact inventory mismatch：observed={observed}"
        )
    files = [
        {
            "path": relative,
            "sha256": sha256_file(artifact_dir / relative),
            "size_bytes": (artifact_dir / relative).stat().st_size,
        }
        for relative in observed
    ]
    payload = {
        "schema_version": 1,
        "files": files,
        "external_files": [
            {
                "path": report_path.name,
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            }
        ],
        "private_key_present": False,
        "capability_token_present": False,
        "private_data_present": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def scan_for_forbidden_material(paths: Sequence[Path]) -> dict[str, Any]:
    forbidden_names = (".pem", ".key", "private_key", "capability_token")
    token_prefixes = (b"-----BEGIN PRIVATE KEY-----", b"eyJhbGciOiJFZDI1NTE5")
    occurrences = []
    scanned = 0
    for root in paths:
        candidates = [root] if root.is_file() else list(root.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            scanned += 1
            lowered = path.name.casefold()
            if any(item in lowered for item in forbidden_names):
                occurrences.append(path.as_posix())
                continue
            data = path.read_bytes()
            if any(prefix in data for prefix in token_prefixes):
                occurrences.append(path.as_posix())
    return {
        "scanned_file_count": scanned,
        "forbidden_material_occurrence_count": len(occurrences),
        "occurrences": occurrences,
    }
