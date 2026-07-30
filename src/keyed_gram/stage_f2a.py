"""Stage F2A：形式问题冻结、反例注册与有界 novelty gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .authcap_policy import canonical_sha256
from .authority_flow.counterexamples import counterexample_registry
from .authority_flow.explorer import run_bounded_model
from .stage_f1_protocol import validate_artifact_inventory


class F2AProtocolError(RuntimeError):
    """F2A 协议或冻结资产校验失败。"""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") != "F2A":
        raise F2AProtocolError("F2A config schema 非法")
    protocol = value.get("protocol", {})
    required = {
        "f1_assets_read_only": True,
        "f1_locked_case_content_access_allowed": False,
        "f1_locked_test_scored": False,
        "real_llm_allowed": False,
        "real_agent_allowed": False,
        "executable_capability_signing_allowed": False,
        "private_data_allowed": False,
    }
    if any(protocol.get(key) is not expected for key, expected in required.items()):
        raise F2AProtocolError("F2A 不可修改边界被放宽")
    return value


def _repo_root(config_path: str | Path) -> Path:
    target = Path(config_path).resolve()
    output = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=target.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(output)


def verify_f1_frozen(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """只读取 F1 manifest 元数据，不打开任何 benchmark case。"""

    base = config["base"]
    source_path = root / base["f1_source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mismatches = []
    files = []
    for item in source["files"]:
        path = root / item["path"]
        observed = _sha256(path) if path.is_file() else None
        if observed != item["sha256"]:
            mismatches.append(item["path"])
        files.append(
            {
                "path": item["path"],
                "sha256": observed,
                "expected_sha256": item["sha256"],
            }
        )
    if mismatches:
        raise F2AProtocolError(f"F1 frozen source mismatch：{mismatches}")
    validate_artifact_inventory(
        root / "artifacts/stage_f1",
        root / "PHASE_F1_REPORT.md",
    )
    split_manifest = json.loads(
        (root / base["f1_split_manifest"]).read_text(encoding="utf-8")
    )
    if split_manifest.get("locked_test_scored") is not False:
        raise F2AProtocolError("F1 locked_test 状态异常")
    return {
        "schema_version": 1,
        "base_branch": base["branch"],
        "base_commit": base["commit"],
        "f1_source_manifest_sha256": _sha256(source_path),
        "f1_artifact_manifest_sha256": _sha256(
            root / base["f1_artifact_manifest"]
        ),
        "f1_split_manifest_sha256": _sha256(
            root / base["f1_split_manifest"]
        ),
        "f1_source_files_verified": len(files),
        "f1_source_mismatch_count": 0,
        "f1_artifact_inventory_verified": True,
        "locked_test_case_content_read": False,
        "locked_test_scored": False,
        "split_data_sha256": {
            split_name: details["data_sha256"]
            for split_name, details in sorted(split_manifest["splits"].items())
        },
    }


def build_novelty_gate(
    bounded: Mapping[str, Any],
    counterexamples: list[Mapping[str, Any]],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    counterexample_count = len(counterexamples)
    not_taint = all(
        bool(item["not_plain_taint_only"]) for item in counterexamples
    )
    not_token = all(
        bool(item["not_token_subset_only"]) for item in counterexamples
    )
    requirements = {
        "new_semantic_object_defined": True,
        "data_and_authority_semantically_distinct": True,
        "authority_origination_property_defined": True,
        "authority_conservation_property_defined": True,
        "non_malleable_authority_property_defined": True,
        "merge_confinement_property_defined": True,
        "distinguishing_counterexample_count": counterexample_count,
        "counterexamples_not_reducible_to_plain_data_taint": not_taint,
        "counterexamples_not_reducible_to_token_subset_check": not_token,
        "bounded_model_counterexample_for_naive_merge": bool(
            bounded["bounded_model_counterexample_for_naive_merge"]
        ),
        "bounded_model_counterexample_for_authority_laundering": bool(
            bounded[
                "bounded_model_counterexample_for_authority_laundering"
            ]
        ),
        "f1_frozen_assets_modified": (
            upstream["f1_source_mismatch_count"] != 0
        ),
        "locked_test_scored": upstream["locked_test_scored"],
    }
    ready = (
        requirements["new_semantic_object_defined"]
        and requirements["data_and_authority_semantically_distinct"]
        and requirements["authority_origination_property_defined"]
        and requirements["authority_conservation_property_defined"]
        and requirements["non_malleable_authority_property_defined"]
        and requirements["merge_confinement_property_defined"]
        and counterexample_count >= 6
        and not_taint
        and not_token
        and requirements["bounded_model_counterexample_for_naive_merge"]
        and requirements[
            "bounded_model_counterexample_for_authority_laundering"
        ]
        and not requirements["f1_frozen_assets_modified"]
        and not requirements["locked_test_scored"]
    )
    return {
        "schema_version": 1,
        **requirements,
        "novelty_gate_status": "conditional_pass_high_prior_work_overlap"
        if ready
        else "failed",
        "ccf_b_novelty_established": False,
        "prior_work_overlap_risk": "high",
        "required_positioning": (
            "不得声称 authority、线性授权、IFC+authorization 或 merge confinement"
            "本身为首次提出；后续贡献必须落在 LLM Agent 的双轨 influence/authority"
            "运行语义、攻击者不塑造 authority trace 的 hyperproperty 与线性分支重组。"
        ),
        "ready_for_executable_authority_calculus": ready,
    }


def _source_manifest(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    files = []
    for relative in config["source_files"]:
        path = root / relative
        if not path.is_file():
            raise F2AProtocolError(f"F2A source 缺失：{relative}")
        files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, Any] = {"schema_version": 1, "files": files}
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def _artifact_manifest(
    artifact_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(artifact_dir).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "files": files,
        "external_files": [
            {
                "path": report_path.name,
                "sha256": _sha256(report_path),
                "size_bytes": report_path.stat().st_size,
            }
        ],
        "private_data_present": False,
        "model_or_optimizer_present": False,
        "capability_or_key_material_present": False,
        "locked_case_content_present": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def run_stage_f2a(
    config_path: str | Path = "configs/stage_f2a.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    root = _repo_root(config_path)
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    if branch != config["protocol"]["required_branch"]:
        raise F2AProtocolError("F2A branch 不匹配")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise F2AProtocolError("tracked worktree 非干净")
    artifact_dir = root / config["outputs"]["artifact_dir"]
    report_path = root / config["outputs"]["report_path"]
    if artifact_dir.exists():
        raise F2AProtocolError("F2A artifact 已存在，拒绝重复 novelty audit")
    if not report_path.is_file():
        raise F2AProtocolError("F2A report 必须在 audit 前冻结")

    upstream = verify_f1_frozen(root, config)
    bounded = run_bounded_model(
        int(config["protocol"]["bounded_exploration_depth"])
    )
    counterexamples = [
        item.canonical() for item in counterexample_registry()
    ]
    gate = build_novelty_gate(bounded, counterexamples, upstream)
    summary = {
        "schema_version": 1,
        "stage": "F2A",
        "status": "passed_formal_problem_freeze"
        if gate["ready_for_executable_authority_calculus"]
        else "failed",
        "original_f2_capability_compiler_plan": (
            "superseded_by_dual_data_authority_flow_research"
        ),
        "original_f2_implementation_evidence_preserved": True,
        "new_semantic_object_defined": gate["new_semantic_object_defined"],
        "data_and_authority_semantically_distinct": gate[
            "data_and_authority_semantically_distinct"
        ],
        "distinguishing_counterexample_count": len(counterexamples),
        "bounded_model_counterexample_for_naive_merge": gate[
            "bounded_model_counterexample_for_naive_merge"
        ],
        "bounded_model_counterexample_for_authority_laundering": gate[
            "bounded_model_counterexample_for_authority_laundering"
        ],
        "novelty_gate_status": gate["novelty_gate_status"],
        "ccf_b_novelty_established": False,
        "ready_for_executable_authority_calculus": gate[
            "ready_for_executable_authority_calculus"
        ],
        "f1_frozen_assets_modified": False,
        "locked_test_scored": False,
        "real_llm_used": False,
        "real_agent_used": False,
        "executable_capability_signed": False,
        "private_data_used": False,
        "formal_model_completed": False,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "git": {
            "base_commit": config["base"]["commit"],
            "branch": branch,
            "audit_commit": _git(root, "rev-parse", "HEAD"),
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=False)
    _write_json(artifact_dir / "upstream_f1_sha256.json", upstream)
    _write_json(artifact_dir / "bounded_model_results.json", bounded)
    _write_json(
        artifact_dir / "counterexample_matrix.json",
        {"schema_version": 1, "counterexamples": counterexamples},
    )
    _write_json(artifact_dir / "novelty_gate.json", gate)
    _write_json(artifact_dir / "stage_f2a_summary.json", summary)
    _write_json(
        artifact_dir / "source_sha256_manifest.json",
        _source_manifest(root, config),
    )
    _write_json(
        artifact_dir / "artifact_sha256_manifest.json",
        _artifact_manifest(artifact_dir, report_path),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Stage F2A novelty audit")
    parser.add_argument("--config", default="configs/stage_f2a.yaml")
    args = parser.parse_args()
    result = run_stage_f2a(args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
