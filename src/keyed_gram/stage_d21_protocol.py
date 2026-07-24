"""Stage D2.1 模型冻结、配置、一次性运行与 artifact 工具。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers
import yaml
from huggingface_hub import snapshot_download

from .stage_d1_protocol import (
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)


D21_SCHEMA_VERSION = 1
D21_STAGE = "D2.1-capability-gated-ephemeral-generation-probe"


class D21ProtocolError(RuntimeError):
    """D2.1 协议、模型冻结或 artifact 完整性错误。"""


def repo_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise D21ProtocolError("无法解析 D2.1 仓库根目录")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(config_path) / path


def _require_false(mapping: Mapping[str, Any], names: Sequence[str]) -> None:
    if any(mapping.get(name) is not False for name in names):
        raise D21ProtocolError("D2.1 private/router/training/C3 门禁必须固定 false")


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
        raise D21ProtocolError("无法读取 D2.1 配置") from exc
    expected = f"{D21_STAGE}-smoke" if smoke else D21_STAGE
    if (
        not isinstance(values, Mapping)
        or values.get("schema_version") != D21_SCHEMA_VERSION
        or values.get("stage") != expected
    ):
        raise D21ProtocolError("D2.1 配置 schema/stage 不匹配")
    protocol = values.get("protocol")
    generators = values.get("generators")
    if not isinstance(protocol, Mapping) or not isinstance(generators, Mapping):
        raise D21ProtocolError("D2.1 配置缺少 protocol/generators")
    _require_false(
        protocol,
        (
            "plaintext_in_artifacts_allowed",
            "conversation_history_allowed",
            "cross_request_kv_cache_allowed",
            "generator_memory_tools_allowed",
            "generator_capability_tools_allowed",
            "private_value_memory_allowed",
            "private_answers_allowed",
            "confirmation_allowed",
            "fine_tuning_allowed",
            "key_attack_allowed",
            "original_c3_allowed",
            "c3_eligible",
        ),
    )
    if (
        generators.get("G0", {}).get("tools") != []
        or generators.get("G1", {}).get("tools") != []
    ):
        raise D21ProtocolError("D2.1 generator tools 必须为空")
    if not smoke:
        governance = values.get("governance")
        canary = values.get("canary")
        g1 = generators.get("G1")
        if (
            not isinstance(governance, Mapping)
            or not isinstance(canary, Mapping)
            or not isinstance(g1, Mapping)
            or governance.get("capability_gated_keyed_memory_status")
            != "passed"
            or governance.get("ready_for_d2_1_public_generation_probe")
            is not True
            or governance.get("semantic_router_in_tcb") is not False
            or governance.get("original_c3_allowed") is not False
            or governance.get("c3_eligible") is not False
            or protocol.get("plaintext_in_source_allowed") is not False
            or protocol.get("plaintext_in_config_allowed") is not False
            or protocol.get("prompt_trace_persistence_allowed") is not False
            or protocol.get("generator_authorization_decision_allowed") is not False
            or protocol.get("natural_language_router_authorization_allowed")
            is not False
            or canary.get("entropy_bits") != 128
            or canary.get("unique_per_record_scenario_run") is not True
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
            or g1.get("frozen") is not True
            or g1.get("training_allowed") is not False
            or g1.get("fresh_context_per_request") is not True
            or g1.get("use_kv_cache") is not False
            or g1.get("conversation_history") is not False
            or g1.get("constrained_decoding")
            != "exact_authorized_value_tokens_only"
            or g1.get("general_prompt_following_evaluated") is not False
            or g1.get("general_lm_confidentiality_evaluated") is not False
            or g1.get("transformers_version") != transformers.__version__
            or g1.get("torch_version") != torch.__version__
        ):
            raise D21ProtocolError("D2.1 governance/canary/G1 冻结声明不完整")
        for label in ("d2_summary", "d2_artifact_manifest"):
            declaration = governance.get(label)
            if not isinstance(declaration, Mapping):
                raise D21ProtocolError(f"D2.1 缺少 {label} 引用")
            path = resolve_path(config_path, declaration["path"])
            if sha256_file(path) != declaration["sha256"]:
                raise D21ProtocolError(f"{label} SHA-256 漂移")
        summary = read_json(
            resolve_path(config_path, governance["d2_summary"]["path"]),
            label="D2 summary",
        )
        if (
            summary.get("capability_gated_keyed_memory_status") != "passed"
            or summary.get("ready_for_d2_1_public_generation_probe") is not True
            or summary.get("private_value_memory_ready") is not False
            or summary.get("c3_eligible") is not False
        ):
            raise D21ProtocolError("D2 readiness 边界漂移")
    return dict(values)


def output_paths(config_path: str | Path) -> tuple[Path, Path]:
    values = load_config(config_path)
    return (
        resolve_path(config_path, values["outputs"]["artifact_dir"]),
        resolve_path(config_path, values["outputs"]["runtime_dir"]),
    )


def model_snapshot(
    config_path: str | Path,
    *,
    download: bool,
) -> Path:
    values = load_config(config_path)
    g1 = values["generators"]["G1"]
    patterns = [entry["path"] for entry in g1["files"]]
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=g1["model_id"],
                revision=g1["revision"],
                allow_patterns=patterns,
                local_files_only=not download,
            )
        )
    except Exception as exc:
        raise D21ProtocolError(
            "D2.1 frozen model snapshot 不可用；先运行 prepare"
        ) from exc
    for declaration in g1["files"]:
        path = snapshot / declaration["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(declaration["size_bytes"])
            or sha256_file(path) != declaration["sha256"]
        ):
            raise D21ProtocolError(
                f"frozen model file SHA/size 漂移：{declaration['path']}"
            )
    return snapshot


def prepare_model(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    snapshot = model_snapshot(config_path, download=True)
    g1 = values["generators"]["G1"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_id": g1["model_id"],
        "revision": g1["revision"],
        "files": g1["files"],
        "model_checkpoint_committed_to_repository": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return {
        **payload,
        "cache_path": str(snapshot),
    }


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
        "configs/stage_d21.yaml",
        "artifacts/stage_d2/stage_d2_summary.json",
        "artifacts/stage_d2/artifact_sha256_manifest.json",
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
        "src/keyed_gram/stage_d21.py",
        "src/keyed_gram/stage_d21_contract.py",
        "src/keyed_gram/stage_d21_generator.py",
        "src/keyed_gram/stage_d21_leakage.py",
        "src/keyed_gram/stage_d21_protocol.py",
        "src/keyed_gram/stage_d21_service.py",
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise D21ProtocolError(f"D2.1 runtime source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
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
    return read_json(path, label="D2.1 protocol state")


def mark_phase_started(runtime_dir: Path, phase: str) -> None:
    state = read_phase_state(runtime_dir)
    phases = state.setdefault("phases", {})
    if phase in phases:
        raise D21ProtocolError(f"D2.1 {phase} 已启动过，拒绝重跑")
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
        raise D21ProtocolError(f"D2.1 {phase} 未处于 started")
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
        raise D21ProtocolError(f"D2.1 {phase} 尚未完成")
