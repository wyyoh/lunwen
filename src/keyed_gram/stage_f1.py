"""Stage F1 AuthZRouteBench 的一次性正式构建。"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .authcap_benchmark import (
    SPLITS,
    generate_authzroutebench,
    public_case_schema,
)
from .authcap_collision import audit_split_collisions
from .authcap_policy import canonical_sha256
from .stage_f1_metrics import benchmark_quality_metrics
from .stage_f1_protocol import (
    F1_STAGE,
    F1ProtocolError,
    artifact_inventory,
    formal_preflight,
    load_config,
    mark_completed,
    mark_failed,
    mark_started,
    output_paths,
    runtime_source_manifest,
    sha256_file,
    validate_artifact_inventory,
    write_json,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _split_manifest(
    split: str,
    rows: list[dict[str, Any]],
    data_path: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "benchmark_version": "authzroutebench-v1",
        "split": split,
        "case_count": len(rows),
        "family_count": len({row["family_id"] for row in rows}),
        "domain_count": len({row["domain"] for row in rows}),
        "attack_category_count": len(
            {row["attack_category"] for row in rows}
        ),
        "allow_case_count": sum(
            row["oracle_decision"] == "allow" for row in rows
        ),
        "deny_case_count": sum(
            row["oracle_decision"] == "deny" for row in rows
        ),
        "data_path": data_path.relative_to(data_path.parents[2]).as_posix(),
        "data_size_bytes": data_path.stat().st_size,
        "data_sha256": sha256_file(data_path),
        "learned_baseline_scored": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def _write_distribution_csv(
    path: Path,
    cases: list[dict[str, Any]],
) -> None:
    rows: list[tuple[str, str, str, int]] = []
    dimensions = {
        "domain_split": lambda case: (case["domain"], case["split"]),
        "attack_split": lambda case: (
            case["attack_category"],
            case["split"],
        ),
        "decision_split": lambda case: (
            case["oracle_decision"],
            case["split"],
        ),
    }
    for dimension, extractor in dimensions.items():
        counts = Counter(extractor(case) for case in cases)
        rows.extend(
            (dimension, left, right, count)
            for (left, right), count in sorted(counts.items())
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("dimension", "value", "split", "count"))
        writer.writerows(rows)


def _gate_status(
    metrics: Mapping[str, Any],
    collision: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "domain_count": metrics["domain_count"]
        >= gates["minimum_domain_count"],
        "attack_category_count": metrics["attack_category_count"]
        >= gates["minimum_attack_category_count"],
        "family_count": metrics["family_count"]
        >= gates["minimum_family_count"],
        "total_case_count": metrics["case_count"]
        >= gates["minimum_total_case_count"],
        "multi_step_workflow_count": metrics["multi_step_workflow_count"]
        >= gates["minimum_multi_step_workflow_count"],
        "explicit_deny_case_count": metrics["explicit_deny_case_count"]
        >= gates["minimum_explicit_deny_case_count"],
        "cross_tenant_case_count": metrics["cross_tenant_case_count"]
        >= gates["minimum_cross_tenant_case_count"],
        "stale_policy_case_count": metrics["stale_policy_case_count"]
        >= gates["minimum_stale_policy_case_count"],
        "cross_split_collision_count": collision[
            "cross_split_collision_count"
        ]
        <= gates["maximum_cross_split_collision_count"],
        "oracle_determinism_failure_count": metrics[
            "oracle_determinism_failure_count"
        ]
        <= gates["maximum_oracle_determinism_failure_count"],
        "oracle_schema_failure_count": metrics["oracle_schema_failure_count"]
        <= gates["maximum_oracle_schema_failure_count"],
        "authority_subset_failure_count": metrics[
            "authority_subset_failure_count"
        ]
        <= gates["maximum_authority_subset_failure_count"],
        "policy_hash_mismatch_count": metrics["policy_hash_mismatch_count"]
        <= gates["maximum_policy_hash_mismatch_count"],
    }
    return all(checks.values()), checks


def _report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    status = summary["authzroutebench_status"]
    ready = str(summary["ready_for_stage_f2"]).lower()
    return f"""# Stage F1：AuthZRouteBench Construction and Policy Oracle

## 结论

```text
authzroutebench_status = {status}
policy_oracle_status = {summary["policy_oracle_status"]}
policy_ground_truth_is_deterministic = {str(summary["policy_ground_truth_is_deterministic"]).lower()}
semantic_proposal_authorizes_execution = false
locked_test_generated = true
locked_test_scored = false
ready_for_stage_f2 = {ready}
```

F1 只构造 policy-grounded benchmark 并审计 oracle。没有训练或评价最终
AuthCap compiler，没有运行 learned baseline，也没有对 locked test 做模型评分。

## 1. Ground truth 与文本来源

全部 allow/deny、maximum authority、policy hash、decision ID 和 witness 均由
纯确定性的结构化 policy oracle 生成。同一输入重复计算得到逐字节相同的
canonical output。

```text
benchmark_text_generation = AI_assisted
policy_ground_truth = deterministic_oracle
independent_human_validation = false
private_data_used = false
```

AI 只参与自然语言表面模板设计，不参与标签、authority 或 adjudication。本阶段
没有独立人工审核，也不伪称有人审。

## 2. Benchmark 规模

| 指标 | 数量 |
|---|---:|
| cases | {metrics["case_count"]} |
| families | {metrics["family_count"]} |
| domains | {metrics["domain_count"]} |
| attack categories | {metrics["attack_category_count"]} |
| policies | {metrics["policy_count"]} |
| principals | {metrics["principal_count"]} |
| tenants | {metrics["tenant_count"]} |
| resources | {metrics["resource_count"]} |
| multi-step workflows | {metrics["multi_step_workflow_count"]} |
| allow / deny | {metrics["allow_case_count"]} / {metrics["deny_case_count"]} |
| restricted authority | {metrics["restricted_authority_case_count"]} |
| explicit deny | {metrics["explicit_deny_case_count"]} |
| cross-tenant | {metrics["cross_tenant_case_count"]} |
| stale-policy | {metrics["stale_policy_case_count"]} |

四个 split 各 360 cases、36 families；attack family、policy template、alias
family、tenant、workflow template 和 purpose template 跨 split 隔离。

## 3. Policy language

Policy 使用严格 JSON/YAML schema，表达显式 subject/tenant/resource/type/
action/relation/purpose、principal attributes、有效期、delegation depth、
priority、specificity、deny、policy epoch 和有限 authority constraints。
默认 deny；先比较 priority，再比较 specificity；winning tier 内显式 deny
优先。不存在隐式 allow，unknown field、duplicate ID/key、wildcard 和
NaN/Infinity 均被拒绝。

它不表达自然语言 policy、分布式一致性、概率条件、任意代码条件、真实 IAM/KMS
或 executable capability。

## 4. Authority 与多步标注

Authority subset 覆盖 subject、tenant、resource、action、relation、purpose
以及约束保留。候选删除约束或增加任一集合元素都会被判为放大。

`multi_step_privilege_amplification` case 保存两个独立 allow step 与一个
combined deny；所有 step 和 combined decision 都由同一 oracle 复算。F1
实现的是 subset/union invariant checker，不是最终 compiler 的
authority-non-amplification 实现或形式证明。

## 5. 数据质量

```text
cross_split_collision_count = {summary["cross_split_collision_count"]}
oracle_determinism_failure_count = {metrics["oracle_determinism_failure_count"]}
oracle_schema_failure_count = {metrics["oracle_schema_failure_count"]}
authority_subset_failure_count = {metrics["authority_subset_failure_count"]}
policy_hash_mismatch_count = {metrics["policy_hash_mismatch_count"]}
```

locked test 只被生成器、schema/oracle 一致性和跨 split collision audit 读取，
未被任何 baseline、模型选择或效果评价逻辑读取，`locked_test_scored=false`。

## 6. 尚未完成

```text
authority_non_amplification_implemented = false
formal_model_completed = false
real_agent_integration_completed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

当前结果尚不足以投稿 ACSAC/ESORICS；仍需要 F2 compiler、F3 formal model/
trace refinement 与 F4 real-agent external evaluation。readiness 只表示允许
开始 F2，不代表 CCF B 创新性、部署安全或 private-data readiness。
"""


def run_stage_f1_build(
    config_path: str | Path,
) -> dict[str, Any]:
    values = load_config(config_path)
    preflight = formal_preflight(config_path, values)
    data_dir, artifact_dir, runtime_dir, report_path = output_paths(
        config_path, values
    )
    mark_started(runtime_dir, preflight["git"])
    try:
        data_dir.mkdir(parents=True, exist_ok=False)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        splits = generate_authzroutebench(
            cases_per_family=values["benchmark"]["cases_per_family"]
        )
        collision = audit_split_collisions(splits)
        metrics = benchmark_quality_metrics(splits)
        passed, checks = _gate_status(
            metrics,
            collision,
            values["readiness_gates"],
        )
        if not passed:
            raise F1ProtocolError(
                "F1 readiness gate failed："
                + ",".join(name for name, ok in checks.items() if not ok)
            )
        source_manifest = runtime_source_manifest(config_path)
        write_json(
            artifact_dir / "source_sha256_manifest.json",
            source_manifest,
        )
        write_json(
            artifact_dir / "upstream_frozen_sha256.json",
            preflight["upstream"],
        )
        write_json(
            artifact_dir / "benchmark_case_schema.json",
            public_case_schema(),
        )
        split_manifests = {}
        for split in SPLITS:
            path = data_dir / f"{split}.jsonl"
            _write_jsonl(path, splits[split])
            manifest = _split_manifest(split, splits[split], path)
            split_manifests[split] = manifest
            write_json(
                artifact_dir / "split_manifests" / f"{split}.json",
                manifest,
            )
        benchmark_manifest = {
            "schema_version": 1,
            "benchmark_version": values["benchmark_version"],
            "splits": split_manifests,
            "benchmark_text_generation": "AI_assisted",
            "policy_ground_truth": "deterministic_oracle",
            "independent_human_validation": False,
            "locked_test_generated": True,
            "locked_test_scored": False,
        }
        benchmark_manifest["manifest_payload_sha256"] = canonical_sha256(
            benchmark_manifest
        )
        write_json(
            artifact_dir / "benchmark_split_manifest.json",
            benchmark_manifest,
        )
        all_cases = [row for split in SPLITS for row in splits[split]]
        policy_manifest = {
            "schema_version": 1,
            "policy_count": metrics["policy_count"],
            "case_policy_hashes": [
                {
                    "case_id": case["case_id"],
                    "policy_hash": case["oracle_grant"]["policy_hash"],
                }
                for case in all_cases
            ],
        }
        policy_manifest["manifest_payload_sha256"] = canonical_sha256(
            policy_manifest
        )
        write_json(artifact_dir / "policy_manifest.json", policy_manifest)
        oracle_manifest = {
            "schema_version": 1,
            "oracle": "deterministic_structured_policy_oracle",
            "learned_component_used": False,
            "locked_test_scored": False,
            "decisions": [
                {
                    "case_id": case["case_id"],
                    "decision": case["oracle_decision"],
                    "decision_id": case["oracle_grant"]["decision_id"],
                    "policy_hash": case["oracle_grant"]["policy_hash"],
                    "witness_sha256": canonical_sha256(
                        case["authorization_witness"]
                    ),
                }
                for case in all_cases
            ],
        }
        oracle_manifest["manifest_payload_sha256"] = canonical_sha256(
            oracle_manifest
        )
        write_json(artifact_dir / "oracle_manifest.json", oracle_manifest)
        write_json(artifact_dir / "collision_audit.json", collision)
        write_json(artifact_dir / "benchmark_quality.json", metrics)
        _write_distribution_csv(
            artifact_dir / "benchmark_distribution.csv",
            all_cases,
        )
        resolved = {
            **values,
            "config_sha256": sha256_file(config_path),
            "code_freeze_commit": preflight["git"]["commit"],
        }
        write_json(artifact_dir / "resolved_config.json", resolved)
        summary = {
            "schema_version": 1,
            "stage": F1_STAGE,
            "authzroutebench_status": "passed_preregistered_construction",
            "policy_oracle_status": "passed",
            "policy_ground_truth_is_deterministic": True,
            "semantic_proposal_authorizes_execution": False,
            "locked_test_generated": True,
            "locked_test_scored": False,
            "ready_for_stage_f2": True,
            "authority_subset_checker_implemented": True,
            "authority_non_amplification_implemented": False,
            "formal_model_completed": False,
            "real_agent_integration_completed": False,
            "private_data_used": False,
            "frozen_upstream_code_modified": False,
            "private_value_memory_ready": False,
            "original_c3_allowed": False,
            "c3_eligible": False,
            "cross_split_collision_count": collision[
                "cross_split_collision_count"
            ],
            "metrics": metrics,
            "readiness_checks": checks,
            "code_freeze_commit": preflight["git"]["commit"],
            "source_manifest_payload_sha256": source_manifest[
                "manifest_payload_sha256"
            ],
            "benchmark_manifest_payload_sha256": benchmark_manifest[
                "manifest_payload_sha256"
            ],
        }
        write_json(artifact_dir / "stage_f1_summary.json", summary)
        protocol_status = {
            "schema_version": 1,
            "formal_build_status": "completed",
            "formal_build_executed_once": True,
            "locked_test_generated": True,
            "locked_test_scored": False,
            "learned_baseline_executed": False,
            "compiler_trained_or_evaluated": False,
            "private_data_used": False,
            "tracked_worktree_clean_at_start": True,
            "code_freeze_commit": preflight["git"]["commit"],
        }
        write_json(artifact_dir / "protocol_status.json", protocol_status)
        report_path.write_text(_report(summary), encoding="utf-8")
        manifest = artifact_inventory(artifact_dir, report_path)
        write_json(
            artifact_dir / "artifact_sha256_manifest.json",
            manifest,
        )
        validate_artifact_inventory(artifact_dir, report_path)
        outputs = tuple(
            sorted(path for path in artifact_dir.rglob("*") if path.is_file())
        ) + (report_path,)
        mark_completed(runtime_dir, outputs)
        return {
            "status": summary["authzroutebench_status"],
            "ready_for_stage_f2": True,
            "case_count": metrics["case_count"],
            "family_count": metrics["family_count"],
            "locked_test_scored": False,
            "summary": str(artifact_dir / "stage_f1_summary.json"),
            "report": str(report_path),
        }
    except Exception as exc:
        mark_failed(runtime_dir, f"{type(exc).__name__}:{exc}")
        raise


def run_stage_f1_smoke(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    load_config(config_path, smoke=True)
    splits = generate_authzroutebench(splits=("train",))
    rows = splits["train"][:12]
    metrics = benchmark_quality_metrics({"train": rows})
    result = {
        "schema_version": 1,
        "status": "passed",
        "case_count": len(rows),
        "oracle_determinism_failure_count": metrics[
            "oracle_determinism_failure_count"
        ],
        "locked_test_generated": False,
        "locked_test_scored": False,
        "private_data_used": False,
        "c3_eligible": False,
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "stage_f1_smoke_summary.json", result)
    return result
