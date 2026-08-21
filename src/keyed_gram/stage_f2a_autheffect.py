"""AuthSynth F2A：有限 ToolEffectIR 与一次性 feasibility audit。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .stage_f2a_autheffect_protocol import (
    F2AProtocolError,
    artifact_manifest,
    assert_no_sensitive_material,
    formal_preflight,
    load_config,
    mark_completed,
    mark_failed,
    mark_started,
    output_paths,
    record_phase,
    runtime_source_manifest,
    sha256_file,
    validate_artifact_inventory,
    validate_data_inventory,
    validate_strict_serialization,
    write_json,
)
from .tool_effects.benchmark import (
    MUTATIONS,
    SPLITS,
    EffectCase,
    generate_feasibility_cases,
    split_cases,
    split_collision_audit,
)
from .tool_effects.methods import (
    METHODS,
    CaseEvaluation,
    Method,
    evaluate_cases,
)
from .tool_effects.metrics import (
    aggregate_all,
    distinguishing_categories,
    readiness_gate,
)
from .tool_effects.types import canonical_digest


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise F2AProtocolError(f"CSV 不允许空 rows：{path.name}")
    fieldnames = tuple(rows[0].keys())
    if any(tuple(row.keys()) != fieldnames for row in rows):
        raise F2AProtocolError(f"CSV schema 不一致：{path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _split_manifest(
    split: str,
    rows: Sequence[EffectCase],
    data_path: Path,
    root: Path,
    benchmark_seed_digest: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "benchmark_version": "autheffectbench-feasibility-v1",
        "split": split,
        "case_count": len(rows),
        "base_tool_family_count": len({row.base_tool_family for row in rows}),
        "mutation_category_count": len({row.mutation_category for row in rows}),
        "forbidden_trace_count": sum(
            row.attack_trace.expected_forbidden for row in rows
        ),
        "clean_control_count": sum(
            row.mutation_category == "clean_control" for row in rows
        ),
        "data_path": data_path.relative_to(root).as_posix(),
        "data_size_bytes": data_path.stat().st_size,
        "data_sha256": sha256_file(data_path),
        "ground_truth": "finite_transition_system",
        "generation_seed_source": "code_freeze_commit",
        "generation_seed_digest": benchmark_seed_digest,
        "llm_judge_used": False,
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def _metrics_for(rows: Sequence[CaseEvaluation], method: Method) -> dict[str, Any]:
    return next(item for item in aggregate_all(rows) if item["method"] == method.value)


def _category_metrics(
    rows: Sequence[CaseEvaluation],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[CaseEvaluation]] = defaultdict(list)
    for row in rows:
        grouped[(row.split, row.mutation_category, row.method.value)].append(row)
    output = []
    for (split, category, method_name), values in sorted(grouped.items()):
        metrics = aggregate_all(values)[0]
        output.append(
            {
                "split": split,
                "mutation_category": category,
                "method": method_name,
                "case_count": len(values),
                "contract_effect_recall": metrics["contract_effect_recall"],
                "forbidden_trace_acceptance_rate": metrics[
                    "forbidden_trace_acceptance_rate"
                ],
                "safe_utility": metrics["safe_utility"],
                "maximal_permissiveness_ratio": metrics["maximal_permissiveness_ratio"],
            }
        )
    return output


def summarize_evaluations(
    rows: Sequence[CaseEvaluation],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise F2AProtocolError("feasibility summary 不允许空 evaluation 集")
    by_split = {
        split: tuple(row for row in rows if row.split == split) for split in SPLITS
    }
    aggregate = list(aggregate_all(rows))
    split_aggregate = {
        split: list(aggregate_all(split_rows))
        for split, split_rows in by_split.items()
        if split_rows
    }
    cgar_all = _metrics_for(rows, Method.AUTHSYNTH_CGAR)
    overall_distinguishing = distinguishing_categories(rows)
    gates = {
        "overall": readiness_gate(
            cgar_all,
            distinguishing_count=len(overall_distinguishing),
            thresholds=thresholds,
        )
    }
    split_distinguishing: dict[str, tuple[str, ...]] = {}
    for split, split_rows in by_split.items():
        if not split_rows:
            continue
        categories = distinguishing_categories(split_rows)
        split_distinguishing[split] = categories
        gates[split] = readiness_gate(
            _metrics_for(split_rows, Method.AUTHSYNTH_CGAR),
            distinguishing_count=len(categories),
            thresholds=thresholds,
        )
    exact_shield_safety = cgar_all["forbidden_trace_acceptance_rate"] == 0.0
    exact_maximal = (
        cgar_all["maximal_permissiveness_ratio"] == 1.0
        and cgar_all["exact_shield_match_rate"] == 1.0
    )
    passed = (
        all(all(checks.values()) for checks in gates.values())
        and exact_shield_safety
        and exact_maximal
    )
    return {
        "rows": rows,
        "aggregate": aggregate,
        "split_aggregate": split_aggregate,
        "category_metrics": _category_metrics(rows),
        "gates": gates,
        "passed": passed,
        "exact_shield_safety": exact_shield_safety,
        "exact_maximal_permissiveness": exact_maximal,
        "distinguishing_categories": {
            "overall": list(overall_distinguishing),
            **{split: list(split_distinguishing.get(split, ())) for split in SPLITS},
        },
    }


def evaluate_feasibility(
    cases: Sequence[EffectCase],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """恰好执行一次给定 case，再对不可变结果做汇总。"""

    if not cases:
        raise F2AProtocolError("feasibility evaluation 不允许空 case 集")
    return summarize_evaluations(evaluate_cases(cases, METHODS), thresholds)


def _report(summary: Mapping[str, Any]) -> str:
    metrics = {item["method"]: item for item in summary["method_metrics"]}
    rows = []
    for method in METHODS:
        item = metrics[method.value]
        rows.append(
            "| {method} | {cer:.3f} | {ftar:.3f} | {podr:.3f} | "
            "{cep:.3f} | {utility:.3f} | {mpr:.3f} | {drift:.3f} |".format(
                method=method.value,
                cer=item["contract_effect_recall"],
                ftar=item["forbidden_trace_acceptance_rate"],
                podr=item["policy_omission_discovery_recall"],
                cep=item["counterexample_precision"],
                utility=item["safe_utility"],
                mpr=item["maximal_permissiveness_ratio"],
                drift=item["drift_detection_recall"],
            )
        )
    status = summary["autheffectbench_feasibility_status"]
    ready = str(summary["ready_for_stage_f2b"]).lower()
    return """# Stage F2A：AuthSynth Effect-Completeness Feasibility

## 结论

```text
autheffectbench_feasibility_status = {status}
effect_complete_definition_frozen = true
exact_shield_safety = {shield}
exact_maximal_permissiveness = {maximal}
ready_for_stage_f2b = {ready}

f1_frozen_assets_modified = false
f1_locked_test_scored = false
auth_effect_locked_test_scored_once = true
additional_d_series_stages_allowed = false
d_series_status = completed
semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
real_tool_completeness_validated = false
ccf_b_novelty_established = false
private_data_used = false
```

本阶段将上一版 F2A 双流演算保留在独立分支，但把论文主问题改为工具实现、
声明契约与授权策略之间的效果完备性。F1 的 policy oracle、schema 与正式产物
只读绑定；未读取或评分 F1 locked case。

## 1. 精确定义与范围

在有限 ToolEffectIR 输入域内，工具契约 `C_T` 效果完备，当且仅当排空 bounded
异步队列后的全部具体效果均属于 `γ(C_T)`。安全 Shield 由有限安全博弈的最大
不动点精确合成。该结论不外推到任意 Python/JavaScript/MCP 实现；证据不足的
真实工具后续必须返回 `UNKNOWN`。

## 2. Benchmark

```text
case_count = {case_count}
base_tool_count = {tool_count}
mutation_category_count = {mutation_count}
clean_control_count = {clean_count}
cross_split_collision_count = {collision_count}
ground_truth = finite_transition_system_and_forbidden_state_predicate
llm_judge_used = false
```

150 个 case 由 8 个 synthetic base tools、6 类 mutation、每类每工具 3 个实例
和 6 个 clean controls 构成。base tool、mutation family、workflow、policy、
version lineage 与 hidden-effect combination 均跨 split 隔离。

## 3. 方法结果

| 方法 | CER | FTAR | PODR | CEP | Safe Utility | MPR | Drift Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
{method_rows}

`Solver-Policy`、`ToolGate-Declared` 和 `ToolGuardian-Style` 是本仓库内的机制级
模拟，不是作者官方实现。它们用于隔离“给定 policy/contract 正确执行”与
“policy/contract 本身完整”之间的差异。`Gold-Contract` 只是有限模型上界。

## 4. 区分性结果

以下 mutation 中均出现了：给定契约的 monitor 判为可执行、具体 transition
进入 forbidden state、而 AuthSynth-CGAR 经 replay/refinement 后阻断：

```text
{distinguishing}
```

CGAR 将可重现偏差分类为 contract/parameter-role omission、policy omission、
composition omission 或 implementation drift；抽象伪反例只修正 abstraction，
不生成通用 deny rule。

## 5. 有限结论

本阶段只证明：在冻结的有限 synthetic transition systems 上，可以精确检查
effect completeness，并合成安全且最大许可的 Shield。尚未实现源码分析、真实
sandbox、任意异步系统、MCP、真实 Agent、自适应攻击或任意程序完备性证明。
当前证据只允许进入 F2B CGAR 工程化，尚不足以声称达到 CCF B 创新或投稿质量。
""".format(
        status=status,
        shield=str(summary["exact_shield_safety"]).lower(),
        maximal=str(summary["exact_maximal_permissiveness"]).lower(),
        ready=ready,
        case_count=summary["benchmark"]["case_count"],
        tool_count=summary["benchmark"]["base_tool_count"],
        mutation_count=summary["benchmark"]["mutation_category_count"],
        clean_count=summary["benchmark"]["clean_control_count"],
        collision_count=summary["cross_split_collision_count"],
        method_rows="\n".join(rows),
        distinguishing=", ".join(summary["distinguishing_counterexample_categories"]),
    )


def _variant_registry() -> dict[str, Any]:
    descriptions = {
        Method.DENY_ALL: "拒绝全部 workflow 的安全但无效用下界",
        Method.SOLVER_POLICY: "基于已声明 effect/policy 的 SMT-style 机制模拟",
        Method.TOOLGATE: "基于给定 Hoare contract 的机制模拟",
        Method.TOOLGUARDIAN: "多证据 effect characterization + 声明策略模拟",
        Method.AUTHSYNTH_NO_CEGAR: "一次 evidence fusion 后直接合成 Shield",
        Method.AUTHSYNTH_CGAR: "replay、分类 refinement 与 Shield 重合成",
        Method.GOLD_CONTRACT: "使用具体 transition system 的理想上界",
    }
    return {
        "schema_version": 1,
        "selected_method": Method.AUTHSYNTH_CGAR.value,
        "variants": [
            {
                "method": method.value,
                "description": descriptions[method],
                "official_external_implementation_used": False,
                "real_tool_execution": False,
            }
            for method in METHODS
        ],
    }


def run_formal_build(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    preflight = formal_preflight(config_path, values)
    data_dir, artifact_dir, runtime_dir, report_path = output_paths(config_path, values)
    root = report_path.parent
    mark_started(runtime_dir, preflight["git"])
    try:
        data_dir.mkdir(parents=True, exist_ok=False)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "split_manifests").mkdir()
        # 正式 benchmark 的精确 case bytes 由 code-freeze commit 绑定；
        # 开发期 fixture seed 不能生成正式 locked_test。
        benchmark_seed = preflight["git"]["commit"]
        benchmark_seed_digest = canonical_digest(
            {"benchmark_seed": benchmark_seed, "schema_version": 1}
        )
        cases = generate_feasibility_cases(benchmark_seed)
        splits = split_cases(cases)
        collision = split_collision_audit(splits)
        if collision["cross_split_collision_count"] != 0:
            raise F2AProtocolError("AuthEffectBench split collision 非 0")
        split_manifests = {}
        for split in SPLITS:
            path = data_dir / f"{split}.jsonl"
            _write_jsonl(path, [case.to_public_dict() for case in splits[split]])
            split_manifests[split] = _split_manifest(
                split,
                splits[split],
                path,
                root,
                benchmark_seed_digest,
            )
            write_json(
                artifact_dir / "split_manifests" / f"{split}.json",
                split_manifests[split],
            )
        validate_data_inventory(data_dir, split_manifests)
        record_phase(runtime_dir, "prepare")

        source_manifest = runtime_source_manifest(config_path)
        freeze_manifest = {
            "schema_version": 1,
            "code_freeze_commit": preflight["git"]["commit"],
            "source_manifest_payload_sha256": source_manifest[
                "manifest_payload_sha256"
            ],
            "selected_method": Method.AUTHSYNTH_CGAR.value,
            "thresholds": dict(values["readiness_gates"]),
            "benchmark_seed_source": "code_freeze_commit",
            "benchmark_seed_digest": benchmark_seed_digest,
            "locked_test_opened": False,
            "configuration_changed_after_freeze": False,
        }
        freeze_manifest["manifest_payload_sha256"] = canonical_digest(freeze_manifest)

        # train/calibration 只做确定性 variant sanity；不选择安全阈值。
        train_calibration = (*splits["train"], *splits["calibration"])
        sanity = evaluate_feasibility(train_calibration, values["readiness_gates"])
        record_phase(runtime_dir, "calibration")
        development_bundle = evaluate_feasibility(
            splits["development"], values["readiness_gates"]
        )
        record_phase(runtime_dir, "development")
        locked_bundle = evaluate_feasibility(
            splits["locked_test"], values["readiness_gates"]
        )
        record_phase(runtime_dir, "locked_test")

        # 总表只汇总三阶段已经产生的不可变 case results；不得为汇总而
        # 第二次执行 development 或 locked_test。
        complete_bundle = summarize_evaluations(
            (
                *sanity["rows"],
                *development_bundle["rows"],
                *locked_bundle["rows"],
            ),
            values["readiness_gates"],
        )
        passed = (
            complete_bundle["passed"]
            and development_bundle["passed"]
            and locked_bundle["passed"]
        )
        cgar_metrics = next(
            item
            for item in complete_bundle["aggregate"]
            if item["method"] == Method.AUTHSYNTH_CGAR.value
        )
        benchmark_metrics = {
            "case_count": len(cases),
            "base_tool_count": len({case.base_tool_family for case in cases}),
            "domain_count": len({case.domain for case in cases}),
            "mutation_category_count": len(MUTATIONS),
            "clean_control_count": sum(
                case.mutation_category == "clean_control" for case in cases
            ),
            "forbidden_trace_count": sum(
                case.attack_trace.expected_forbidden for case in cases
            ),
            "split_case_counts": {split: len(rows) for split, rows in splits.items()},
            "generation_seed_source": "code_freeze_commit",
            "generation_seed_digest": benchmark_seed_digest,
            "omission_type_counts": dict(
                sorted(Counter(case.omission_type.value for case in cases).items())
            ),
        }
        status = "passed" if passed else "failed"
        summary = {
            "schema_version": 1,
            "stage": "F2A-autheffect-feasibility",
            "autheffectbench_feasibility_status": status,
            "effect_complete_definition_frozen": True,
            "exact_shield_safety": complete_bundle["exact_shield_safety"],
            "exact_maximal_permissiveness": complete_bundle[
                "exact_maximal_permissiveness"
            ],
            "ready_for_stage_f2b": passed,
            "additional_d_series_stages_allowed": False,
            "d_series_status": "completed",
            "semantic_router_authorization_research": "stopped",
            "private_data_experiment_allowed": False,
            "benchmark": benchmark_metrics,
            "method_metrics": complete_bundle["aggregate"],
            "development_method_metrics": development_bundle["aggregate"],
            "locked_test_method_metrics": locked_bundle["aggregate"],
            "readiness_checks": {
                "overall": complete_bundle["gates"],
                "development": development_bundle["gates"],
                "locked_test": locked_bundle["gates"],
            },
            "distinguishing_counterexample_categories": complete_bundle[
                "distinguishing_categories"
            ]["overall"],
            "distinguishing_counterexample_type_count": len(
                complete_bundle["distinguishing_categories"]["overall"]
            ),
            "cross_split_collision_count": collision["cross_split_collision_count"],
            "cgar_metrics": cgar_metrics,
            "f1_frozen_assets_modified": False,
            "f1_locked_test_case_content_read": False,
            "f1_locked_test_scored": False,
            "auth_effect_locked_test_scored": True,
            "auth_effect_locked_test_scored_once": True,
            "real_tool_completeness_validated": False,
            "arbitrary_program_completeness_claimed": False,
            "ccf_b_novelty_established": False,
            "llm_judge_used": False,
            "real_llm_used": False,
            "real_agent_used": False,
            "private_data_used": False,
            "strict_json_serialization_validated": True,
            "exact_artifact_inventory_validated": True,
            "formal_model_completed": False,
            "private_value_memory_ready": False,
            "original_c3_allowed": False,
            "c3_eligible": False,
            "git": {
                "branch": preflight["git"]["branch"],
                "code_freeze_commit": preflight["git"]["commit"],
            },
        }

        write_json(artifact_dir / "resolved_config.json", values)
        write_json(artifact_dir / "upstream_f1_sha256.json", preflight["frozen_f1"])
        write_json(artifact_dir / "source_sha256_manifest.json", source_manifest)
        write_json(artifact_dir / "analyzer_freeze_manifest.json", freeze_manifest)
        write_json(artifact_dir / "variant_registry.json", _variant_registry())
        write_json(
            artifact_dir / "benchmark_manifest.json",
            {
                "schema_version": 1,
                "benchmark": benchmark_metrics,
                "splits": split_manifests,
                "f1_locked_test_case_content_read": False,
            },
        )
        write_json(artifact_dir / "collision_audit.json", collision)
        write_json(
            artifact_dir / "feasibility_gate.json",
            {
                "schema_version": 1,
                "status": status,
                "checks": summary["readiness_checks"],
                "exact_shield_safety": summary["exact_shield_safety"],
                "exact_maximal_permissiveness": summary["exact_maximal_permissiveness"],
            },
        )
        write_json(
            artifact_dir / "cgar_refinement_summary.json",
            {
                "schema_version": 1,
                "counterexample_precision": cgar_metrics["counterexample_precision"],
                "counterexample_count": cgar_metrics["counterexample_count"],
                "validated_counterexample_count": cgar_metrics[
                    "validated_counterexample_count"
                ],
                "policy_omission_discovery_recall": cgar_metrics[
                    "policy_omission_discovery_recall"
                ],
                "drift_detection_recall": cgar_metrics["drift_detection_recall"],
                "refinement_outputs": [
                    "effect_contract_patch",
                    "parameter_role_patch",
                    "policy_restriction_patch",
                    "composition_restriction",
                    "version_invalidation_rule",
                    "runtime_shield_guard",
                ],
            },
        )
        write_json(
            artifact_dir / "exact_shield_summary.json",
            {
                "schema_version": 1,
                "finite_model_only": True,
                "shield_safety": summary["exact_shield_safety"],
                "maximal_permissiveness": summary["exact_maximal_permissiveness"],
                "maximal_permissiveness_ratio": cgar_metrics[
                    "maximal_permissiveness_ratio"
                ],
                "unknown_for_arbitrary_programs": True,
            },
        )
        write_json(artifact_dir / "stage_f2a_summary.json", summary)
        write_json(
            artifact_dir / "protocol_status.json",
            {
                "schema_version": 1,
                "formal_build": "completed",
                "code_freeze_commit": preflight["git"]["commit"],
                "prepare_runs": 1,
                "calibration_runs": 1,
                "development_runs": 1,
                "locked_test_runs": 1,
                "configuration_changed_after_development": False,
                "configuration_changed_after_locked": False,
                "f1_locked_test_scored": False,
                "strict_json_serialization_validated": True,
                "exact_artifact_inventory_validated": True,
            },
        )

        _write_csv(
            artifact_dir / "method_metrics.csv",
            complete_bundle["aggregate"],
        )
        _write_csv(
            artifact_dir / "per_category_metrics.csv",
            complete_bundle["category_metrics"],
        )
        _write_csv(
            artifact_dir / "case_results.csv",
            [row.to_dict() for row in complete_bundle["rows"]],
        )
        _write_csv(
            artifact_dir / "calibration_sanity.csv",
            sanity["aggregate"],
        )
        _write_csv(
            artifact_dir / "development_results.csv",
            development_bundle["aggregate"],
        )
        _write_csv(
            artifact_dir / "locked_test_results.csv",
            locked_bundle["aggregate"],
        )

        report_path.write_text(_report(summary), encoding="utf-8")
        artifact_files = [path for path in artifact_dir.rglob("*") if path.is_file()]
        scan = assert_no_sensitive_material((*artifact_files, report_path))
        write_json(artifact_dir / "sensitive_material_scan.json", scan)
        manifest = artifact_manifest(artifact_dir, report_path)
        write_json(artifact_dir / "artifact_sha256_manifest.json", manifest)
        validate_artifact_inventory(artifact_dir, report_path)
        validate_strict_serialization(artifact_dir, data_dir)
        mark_completed(
            runtime_dir, sha256_file(artifact_dir / "stage_f2a_summary.json")
        )
        return summary
    except Exception as exc:
        mark_failed(runtime_dir, str(exc))
        raise


def run_smoke() -> dict[str, Any]:
    splits = split_cases(generate_feasibility_cases())
    # Smoke 只使用 development/locked 之外的 synthetic rows。
    cases = (*splits["train"][:4], *splits["calibration"][:4])
    rows = evaluate_cases(cases)
    cgar = _metrics_for(rows, Method.AUTHSYNTH_CGAR)
    passed = (
        cgar["forbidden_trace_acceptance_rate"] == 0.0
        and cgar["safe_utility"] == 1.0
        and cgar["maximal_permissiveness_ratio"] == 1.0
    )
    return {
        "status": "passed" if passed else "failed",
        "case_count": len(cases),
        "evaluated_splits": ["train", "calibration"],
        "locked_test_scored": False,
        "real_tool_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_f2a.yaml")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run_smoke() if arguments.smoke else run_formal_build(arguments.config)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
