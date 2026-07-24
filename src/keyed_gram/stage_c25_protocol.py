"""Stage C2.5 的外部数据冻结、test-once 与有限状态协议。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
STAGE_NAME = "C2.5-external-oos-generalization-audit"
PRIVATE_TERMS = (
    "private_answer",
    "private_value",
    "confirmation",
    "permutation",
    "optimizer_state",
    "key_material",
)


class C25ProtocolError(RuntimeError):
    """C2.5 协议门禁失败。"""


def repo_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise C25ProtocolError("无法解析仓库根目录")


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
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise C25ProtocolError(f"JSON 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    _finite(payload)
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
                ValueError(f"非法常量 {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise C25ProtocolError(f"{label} 不是严格 JSON") from exc
    if not isinstance(value, Mapping):
        raise C25ProtocolError(f"{label} 必须是 JSON object")
    _finite(value)
    return dict(value)


def load_config(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise C25ProtocolError("无法读取 C2.5 配置") from exc
    expected_stage = f"{STAGE_NAME}-smoke" if smoke else STAGE_NAME
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("stage") != expected_stage
    ):
        raise C25ProtocolError("C2.5 配置 schema/stage 不匹配")
    protocol = value.get("protocol")
    if not isinstance(protocol, Mapping):
        raise C25ProtocolError("C2.5 缺少 protocol")
    for key in (
        "create_confirmation",
        "train_private_memory",
        "load_private_answers",
        "execute_answer_injection",
        "execute_key_attack",
        "task_specific_router_ready",
        "c3_eligible",
    ):
        if protocol.get(key) is not False:
            raise C25ProtocolError(f"C2.5 门禁 {key} 必须固定为 false")
    if not smoke:
        if (
            protocol.get("evaluation_mode")
            != "external_public_benchmark_test_once"
            or protocol.get("test_predictions_once") is not True
            or protocol.get("synthetic_oos_allowed") is not False
            or protocol.get("label_edits_allowed") is not False
            or protocol.get("development_used_for_selection") is not False
            or protocol.get("test_used_for_selection") is not False
        ):
            raise C25ProtocolError("C2.5 外部/test-once 声明不完整")
        seeds = protocol.get("seeds")
        if (
            not isinstance(seeds, list)
            or len(seeds) < 10
            or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) for seed in seeds)
        ):
            raise C25ProtocolError("C2.5 至少需要 10 个唯一整数 seed")
    return dict(value)


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


def download_verified(
    target: Path, *, url: str, sha256: str, opener: Any = None
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_name(f".{target.name}.download")
        try:
            if opener is None:
                urllib.request.urlretrieve(url, temporary)
            else:
                opener(url, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    observed = sha256_file(target)
    if observed != sha256:
        raise C25ProtocolError(f"外部源 SHA-256 漂移：{target.name}")
    return {
        "path": target.as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": observed,
        "url": url,
    }


def runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    paths = (
        "configs/stage_c25.yaml",
        "src/keyed_gram/cli.py",
        "src/keyed_gram/stage_c23_semantic.py",
        "src/keyed_gram/stage_c25.py",
        "src/keyed_gram/stage_c25_data.py",
        "src/keyed_gram/stage_c25_metrics.py",
        "src/keyed_gram/stage_c25_protocol.py",
        "src/keyed_gram/stage_c25_router.py",
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise C25ProtocolError(f"C2.5 runtime source 缺失：{relative}")
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


def verify_runtime_sources(
    config_path: str | Path, frozen: Mapping[str, Any]
) -> None:
    current = runtime_source_manifest(config_path)
    if current != frozen:
        raise C25ProtocolError("C2.5 代码或配置在 freeze 后发生漂移")


def initial_status() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "external_oos_validation_status": "not_evaluated",
        "external_pairwise_evidence_validated": False,
        "external_selective_abstention_validated": False,
        "exploratory_c3_allowed": False,
        "task_specific_router_ready": False,
        "c3_eligible": False,
        "test_predictions_executed": False,
        "test_used_for_selection": False,
        "synthetic_oos_used": False,
        "labels_modified": False,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "answer_injection_executed": False,
        "key_attack_executed": False,
        "confirmation_created_or_read": False,
    }


def phase_state_path(runtime_dir: Path) -> Path:
    return runtime_dir / "protocol_state.json"


def read_phase_state(runtime_dir: Path) -> dict[str, Any]:
    path = phase_state_path(runtime_dir)
    if not path.is_file():
        return {"schema_version": 1, "phases": {}}
    return read_json(path, label="C2.5 protocol state")


def mark_phase_started(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.setdefault("phases", {})
    if phase in phases:
        raise C25ProtocolError(f"C2.5 {phase} 已启动过，拒绝重跑")
    phases[phase] = {"status": "started"}
    write_json(phase_state_path(runtime_dir), state)


def mark_phase_completed(
    runtime_dir: Path, phase: str, *, outputs: Sequence[Path]
) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.get("phases", {})
    if phases.get(phase, {}).get("status") != "started":
        raise C25ProtocolError(f"C2.5 {phase} 未处于 started")
    phases[phase] = {
        "status": "completed",
        "outputs": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    write_json(phase_state_path(runtime_dir), state)


def require_phase_completed(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    if state.get("phases", {}).get(phase, {}).get("status") != "completed":
        raise C25ProtocolError(f"C2.5 {phase} 尚未完成")
