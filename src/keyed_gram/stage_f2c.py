"""Stage F2C symbolic unary contract 的正式一次性构建入口。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import z3

from keyed_gram.authsynth_symbolic_shared import (
    GrammarLimits,
    canonical_digest,
    canonical_json,
)
from keyed_gram.authsynth_symbolic_verifier.benchmark import (
    SPLITS,
    AuthSymbolBenchVault,
    collision_audit,
    generate_authsymbolbench,
    generate_symbolic_smoke_cases,
    iter_evaluator_manifest,
)
from keyed_gram.authsynth_symbolic_verifier.evaluator import (
    SymbolicEvaluationRow,
    SymbolicMethod,
    aggregate_metrics,
    evaluate_cases,
)

from .stage_f2c_metrics import (
    contract_complexity,
    patch_metrics,
    per_category_metrics,
    per_field_metrics,
    performance_metrics,
    query_efficiency,
    readiness_gate,
    selected_metrics,
    unknown_reason_distribution,
)
from .stage_f2c_protocol import (
    F2CProtocolError,
    artifact_manifest,
    formal_preflight,
    information_boundary_audit,
    load_config,
    mark_completed,
    mark_started,
    output_paths,
    record_phase,
    repo_root,
    sensitive_scan,
    sha256_file,
    validate_artifact_inventory,
    validate_strict_serialization,
    write_csv,
    write_json,
)

_SOURCE_PATHS = (
    "AUTHSYNTH_F2C_RESEARCH_PLAN.md",
    "configs/stage_f2c.yaml",
    "docs/f2c/F2B_AUDIT_AND_REINTERPRETATION.md",
    "docs/f2c/SYMBOLIC_CONTRACT_LANGUAGE.md",
    "docs/f2c/CEGIS_ALGORITHM.md",
    "docs/f2c/COMPLETENESS_CERTIFICATE.md",
    "docs/f2c/EXPERIMENT_PROTOCOL.md",
    "docs/f2c/THREATS_TO_VALIDITY.md",
    "requirements-stage-f2c.txt",
    "src/keyed_gram/authsynth_symbolic_shared/__init__.py",
    "src/keyed_gram/authsynth_symbolic_shared/schema.py",
    "src/keyed_gram/authsynth_symbolic_shared/formulas.py",
    "src/keyed_gram/authsynth_symbolic_shared/effects.py",
    "src/keyed_gram/authsynth_symbolic_shared/contracts.py",
    "src/keyed_gram/authsynth_symbolic_shared/certificates.py",
    "src/keyed_gram/authsynth_symbolic_shared/interfaces.py",
    "src/keyed_gram/authsynth_symbolic_shared/patches.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/__init__.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/models.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/diagnostics.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/learner.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/query_generator.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/shield.py",
    "src/keyed_gram/authsynth_symbolic_analyzer/cegis.py",
    "src/keyed_gram/authsynth_symbolic_verifier/__init__.py",
    "src/keyed_gram/authsynth_symbolic_verifier/hidden_ir.py",
    "src/keyed_gram/authsynth_symbolic_verifier/omission_oracle.py",
    "src/keyed_gram/authsynth_symbolic_verifier/solver.py",
    "src/keyed_gram/authsynth_symbolic_verifier/equivalence.py",
    "src/keyed_gram/authsynth_symbolic_verifier/replay_service.py",
    "src/keyed_gram/authsynth_symbolic_verifier/benchmark.py",
    "src/keyed_gram/authsynth_symbolic_verifier/evaluator.py",
    "src/keyed_gram/stage_f2c.py",
    "src/keyed_gram/stage_f2c_metrics.py",
    "src/keyed_gram/stage_f2c_protocol.py",
    "tests/f2c_symbolic_helpers.py",
    "tests/test_symbolic_types.py",
    "tests/test_symbolic_formulas.py",
    "tests/test_symbolic_contracts.py",
    "tests/test_symbolic_learner.py",
    "tests/test_symbolic_cegis.py",
    "tests/test_symbolic_verifier.py",
    "tests/test_symbolic_collisions.py",
    "tests/test_symbolic_safety_regressions.py",
    "tests/test_symbolic_patch_atoms.py",
    "scripts/audit_f2c_prefreeze.py",
    "scripts/validate_f2c_draft.py",
    "tests/test_symbolic_query_generation.py",
    "tests/test_symbolic_information_boundary.py",
    "tests/test_symbolic_shield.py",
    "tests/test_stage_f2c_metrics.py",
    "tests/test_stage_f2c_protocol.py",
)


def _source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for relative in _SOURCE_PATHS:
        target = root / relative
        if not target.is_file():
            raise F2CProtocolError(f"F2C runtime source 缺失：{relative}")
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _split_manifest(
    split: str, rows: Sequence[Mapping[str, Any]], path: Path
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "split": split,
        "case_count": len(rows),
        "data_sha256": sha256_file(path),
        "data_size_bytes": path.stat().st_size,
        "case_id_digest": canonical_digest(
            sorted(row["public_case_id"] for row in rows)
        ),
        "analyzer_input_only": True,
        "hidden_implementation_ast_persisted": False,
        "gold_contract_persisted": False,
        "full_assignment_output_table_persisted": False,
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def _candidate_settings(item: Mapping[str, Any]) -> tuple[GrammarLimits, int, int]:
    return (
        GrammarLimits(
            max_disjuncts=int(item["max_disjuncts"]),
            max_literals_per_conjunction=int(item["max_literals_per_conjunction"]),
            max_total_literals=int(item["max_total_literals"]),
            allow_ite_terms=bool(item["allow_ite_terms"]),
        ),
        int(item["replay_budget"]),
        int(item["solver_timeout_ms"]),
    )


def _configured_cases(
    cases: Sequence[Any],
    grammar: GrammarLimits,
    replay_budget: int,
    solver_timeout_ms: int,
) -> tuple[Any, ...]:
    return tuple(
        replace(
            case,
            analyzer_input=replace(
                case.analyzer_input,
                grammar_limits=grammar,
                replay_budget=replay_budget,
                solver_timeout_ms=solver_timeout_ms,
                max_iterations=replay_budget + 4,
            ),
        )
        for case in cases
    )


def _calibrate(
    cases: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    passing = []
    for item in candidates:
        grammar, replay_budget, timeout = _candidate_settings(item)
        bundle = evaluate_cases(
            _configured_cases(cases, grammar, replay_budget, timeout),
            replay_budget=replay_budget,
            solver_timeout_ms=timeout,
            methods=(SymbolicMethod.AUTHSYNTH,),
        )
        metrics = selected_metrics(bundle["metrics"])
        gate = readiness_gate(metrics, thresholds)
        record = {
            "candidate_id": item["candidate_id"],
            **metrics,
            "all_safety_and_utility_gates_passed": all(gate.values()),
        }
        rows.append(record)
        if all(gate.values()):
            passing.append(item)
    pool = passing or list(candidates)
    selected = min(
        pool,
        key=lambda item: (
            int(item["replay_budget"]),
            int(item["max_total_literals"]),
            int(item["solver_timeout_ms"]),
            str(item["candidate_id"]),
        ),
    )
    for row in rows:
        row["selected"] = row["candidate_id"] == selected["candidate_id"]
    return dict(selected), rows


def _public_rows(
    vault: AuthSymbolBenchVault, split: str, configured: Sequence[Any]
) -> tuple[dict, ...]:
    selected = [case for case in configured if case.split == split]
    return tuple(case.analyzer_input.to_public_dict() for case in selected)


def _benchmark_summary(
    cases: Sequence[Any], collision: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "case_count": len(cases),
        "base_tool_count": len({case.tool_family for case in cases}),
        "domain_count": len({case.domain for case in cases}),
        "mutation_category_count": len({case.mutation_category for case in cases}),
        "mutation_category_counts": dict(
            sorted(Counter(case.mutation_category for case in cases).items())
        ),
        "split_case_counts": collision["split_case_counts"],
        "domain_assignment_count_per_case": min(
            case.analyzer_input.schema.cardinality for case in cases
        ),
        "total_bounded_domain_assignments": sum(
            case.analyzer_input.schema.cardinality for case in cases
        ),
        "expected_unknown_case_count": sum(case.expected_unknown for case in cases),
        "clean_control_case_count": sum(case.clean_control for case in cases),
    }


def _certificate_manifest(rows: Sequence[SymbolicEvaluationRow]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.method == SymbolicMethod.AUTHSYNTH.value
        and row.certificate_id is not None
    ]
    payload = {
        "schema_version": 1,
        "certificate_count": len(selected),
        "certificates": [
            {
                "case_id_digest": canonical_digest(row.case_id),
                "split": row.split,
                "certificate_id": row.certificate_id,
                "contract_digest": row.contract_digest,
            }
            for row in sorted(selected, key=lambda item: item.case_id)
        ],
        "certificate_contains_secret": False,
        "old_certificate_reuse_after_drift_count": sum(
            row.old_certificate_reused_after_drift for row in selected
        ),
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def _gate_text(gate: Mapping[str, bool]) -> str:
    return "passed" if all(gate.values()) else "failed"


def _report(summary: Mapping[str, Any]) -> str:
    metric_rows = []
    for item in summary["method_metrics"]:
        metric_rows.append(
            "| {method} | {fvcr:.3f} | {uer:.3f} | {uep:.3f} | {gp:.3f} | "
            "{gr:.3f} | {fb:.3f} | {sb:.3f} | {cor:.3f} | {ftar:.3f} | "
            "{utility:.3f} | {mpr:.3f} | {rqr:.3f} | {conv:.3f} | {unknown:.3f} |".format(
                method=item["method"],
                fvcr=item["false_verified_complete_rate"],
                uer=item["unseen_effect_recall"],
                uep=item["unseen_effect_precision"],
                gp=item["guard_precision"],
                gr=item["guard_recall"],
                fb=item["field_binding_accuracy"],
                sb=item["state_update_binding_accuracy"],
                cor=item["contract_overapproximation_ratio"],
                ftar=item["forbidden_trace_acceptance_rate"],
                utility=item["safe_utility"],
                mpr=item["maximal_permissiveness_ratio"],
                rqr=item["replay_query_reduction"],
                conv=item["convergence_rate"],
                unknown=item["unknown_rate"],
            )
        )
    benchmark = summary["benchmark"]
    return f"""# Stage F2C：Symbolic Unary Effect-Contract Synthesis

## 核心状态

```text
symbolic_contract_synthesis_status = {summary["symbolic_contract_synthesis_status"]}
query_generating_cegis_status = {summary["query_generating_cegis_status"]}
bounded_domain_contract_completeness_validated = {str(summary["bounded_domain_contract_completeness_validated"]).lower()}
symbolic_unseen_generalization_validated = {str(summary["symbolic_unseen_generalization_validated"]).lower()}
false_verified_complete_count = {summary["false_verified_complete_count"]}
ready_for_stage_f2d_relational_contracts = {str(summary["ready_for_stage_f2d_relational_contracts"]).lower()}
```

## F2B 的重新解释

F2B 保持字节冻结并继续记为 `passed_blind_finite_query_characterization`。其
`VERIFIED_COMPLETE` 只针对冻结 finite query catalog；refined contract 是
query-specific trace map，query 由 benchmark 枚举，旧 `refinement_precision=1.0`
是指标定义性结果，PODR 是 case-level。F2B 没有结构化 state-diff 语义、没有
inductive loop invariant，也没有完成 drift 后新版本再认证。它没有验证 general
symbolic synthesis、query-generating CEGIS 或 relational completeness。

## F2C 方法

候选契约由受限 DNF guard 与显式 constant/input/state/ITE effect terms 构成，不含
query ID→trace 映射。Analyzer 只看 public schema、declared contract、可信 safety
spec、examples 和窄 verifier/replay 接口；它不能导入 evaluator，也看不到 hidden
AST、gold guard/contract/Shield、mutation label 或完整 assignment table。Verifier
用 SMT 在 candidate 与 hidden ToolSymbolicIR 的差异公式上主动生成新 assignment；
sandbox 只回传该 assignment 的有序事件、结构化 state diff、quiescence、coverage
与版本摘要。每轮 candidate AST 和 Shield digest 必须真实变化，直到有界等价或
fail-closed `UNKNOWN`。

## Benchmark

```text
case_count = {benchmark["case_count"]}
base_tool_count = {benchmark["base_tool_count"]}
domain_count = {benchmark["domain_count"]}
bounded_assignments_per_case = {benchmark["domain_assignment_count_per_case"]}
total_bounded_domain_assignments = {benchmark["total_bounded_domain_assignments"]}
expected_unknown_case_count = {benchmark["expected_unknown_case_count"]}
cross_split_collision_count = {summary["cross_split_collision_count"]}
locked_test_scored_once = true
llm_judge_used = false
private_data_used = false
```

Analyzer freeze 后才物化独立 AuthSymbolBench v1。四个 split 按 base tool、guard、
field-binding、version lineage 与 AST shape 隔离；正式 artifacts 不含 hidden locked
AST、gold locked contract/path predicates 或完整 assignment/output table。

## 方法结果

| 方法 | FVCR | UER | Unseen P | Guard P | Guard R | Field | State | COR | FTAR | Safe Utility | MPR | RQR | Conv. | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

`DenyAll` 是零效用控制；两种 Replay Table 不做符号泛化；Passive learner 不使用
verifier counterexample；`CEGIS-No-Minimization` 保留反例循环但不优化契约；U0
Gold 只作为上界。所有名称均为仓库内机制定义，不冒充外部系统官方实现。

## 完备性与 UNKNOWN

`CONTRACT_VERIFIED_COMPLETE` 只表示：在本阶段公开 schema 定义的 {benchmark["domain_assignment_count_per_case"]}
个有界 assignment 上，Z3 未找到 under/over approximation、field-binding 或
state-update 差异；版本匹配，bounded async 已排空，instrumentation 完整，且 grammar
可表达候选。证据缺口、solver timeout、grammar 不足、unsupported region、未知循环、
无法排空的 delayed effect 都返回 `CONTRACT_UNKNOWN`。一般循环未处理；固定展开两次
不构成证明。旧 certificate 在 version digest 变化后先失效，新版本必须独立分析并
获得不同 certificate ID。

## 诚实边界

F2C 只验证单次执行、单运行性质。它不分析 secret-dependent payload、双运行
noninterference 或 HyperLTL（留给 F2D），不使用真实 MCP、真实 Agent、private data
或 LLM judge，也不证明任意 Python/JavaScript 工具。结构化 safety specification
来自可信控制面；执行 trace 只能发现“发生了什么”，不能自行决定组织禁止什么。
当前结果不能声称真实工具效果完备，也不足以单独支撑 CCF B 投稿；仍需 F2D、F3、F4。

## 协议状态

Analyzer source、solver/grammar/budget 候选先冻结；locked 随后独立物化。Calibration
只选择预注册的非安全性复杂度与预算参数；development 和 locked 各运行一次，locked
反例未反馈给 Analyzer。F1/F2A/F2B locked 均未重跑，冻结资产未修改。
"""


def run_formal_build(
    config_path: str | Path = "configs/stage_f2c.yaml",
) -> dict[str, Any]:
    values = load_config(config_path)
    preflight = formal_preflight(config_path, values)
    data_dir, artifact_dir, runtime_dir, report_path = output_paths(config_path, values)
    root = repo_root(config_path)
    mark_started(runtime_dir, preflight["git"])
    data_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "split_manifests").mkdir()

    defaults = values["materialization_defaults"]
    default_grammar, default_budget, default_timeout = _candidate_settings(defaults)
    vault = generate_authsymbolbench(
        canonical_digest(
            {
                "base": values["protocol"]["required_base_commit"],
                "freeze": values["protocol"]["analyzer_freeze_commit"],
                "benchmark": values["benchmark_version"],
            }
        ),
        analyzer_freeze_commit=values["protocol"]["analyzer_freeze_commit"],
        grammar=default_grammar,
        replay_budget=default_budget,
        solver_timeout_ms=default_timeout,
    )
    collision = collision_audit(vault.cases)
    if collision["cross_split_collision_count"]:
        raise F2CProtocolError("AuthSymbolBench cross-split collision 非 0")
    record_phase(runtime_dir, "materialize")

    calibration_cases = vault.open_for_evaluation("calibration")
    selected_candidate, calibration_rows = _calibrate(
        calibration_cases,
        values["calibration_candidates"],
        values["readiness_gates"],
    )
    record_phase(runtime_dir, "calibration")
    selected_grammar, selected_budget, selected_timeout = _candidate_settings(
        selected_candidate
    )
    configured_cases = _configured_cases(
        vault.cases, selected_grammar, selected_budget, selected_timeout
    )
    record_phase(runtime_dir, "configuration_freeze")

    public_rows = []
    split_manifests = {}
    for split in SPLITS:
        rows = _public_rows(vault, split, configured_cases)
        public_rows.extend(rows)
        split_path = data_dir / f"{split}.jsonl"
        _write_jsonl(split_path, rows)
        manifest = _split_manifest(split, rows, split_path)
        split_manifests[split] = manifest
        write_json(artifact_dir / "split_manifests" / f"{split}.json", manifest)
    boundary = information_boundary_audit(root, public_rows)
    if boundary["status"] != "passed":
        raise F2CProtocolError("F2C information boundary audit 失败")

    by_split = {
        split: tuple(case for case in configured_cases if case.split == split)
        for split in SPLITS
    }
    vault.open_for_evaluation("train")
    train_bundle = evaluate_cases(
        by_split["train"],
        replay_budget=selected_budget,
        solver_timeout_ms=selected_timeout,
    )
    calibration_bundle = evaluate_cases(
        by_split["calibration"],
        replay_budget=selected_budget,
        solver_timeout_ms=selected_timeout,
    )
    vault.open_for_evaluation("development")
    development_bundle = evaluate_cases(
        by_split["development"],
        replay_budget=selected_budget,
        solver_timeout_ms=selected_timeout,
    )
    record_phase(runtime_dir, "development")
    vault.open_for_evaluation("locked_test")
    locked_bundle = evaluate_cases(
        by_split["locked_test"],
        replay_budget=selected_budget,
        solver_timeout_ms=selected_timeout,
    )
    record_phase(runtime_dir, "locked_test")

    all_rows = (
        *train_bundle["rows"],
        *calibration_bundle["rows"],
        *development_bundle["rows"],
        *locked_bundle["rows"],
    )
    all_metrics = aggregate_metrics(all_rows)
    development_gate = readiness_gate(
        selected_metrics(development_bundle["metrics"]), values["readiness_gates"]
    )
    locked_gate = readiness_gate(
        selected_metrics(locked_bundle["metrics"]), values["readiness_gates"]
    )
    passed = (
        all(development_gate.values())
        and all(locked_gate.values())
        and collision["cross_split_collision_count"] == 0
        and boundary["status"] == "passed"
    )
    selected = selected_metrics(all_metrics)
    benchmark = _benchmark_summary(configured_cases, collision)
    summary = {
        "schema_version": 1,
        "stage": "F2C-symbolic-contracts",
        "symbolic_contract_synthesis_status": "passed" if passed else "failed",
        "query_generating_cegis_status": "passed" if passed else "failed",
        "bounded_domain_contract_completeness_validated": passed,
        "symbolic_unseen_generalization_validated": passed,
        "false_verified_complete_count": selected["false_verified_complete_count"],
        "ready_for_stage_f2d_relational_contracts": passed,
        "benchmark": benchmark,
        "selected_calibration_candidate": selected_candidate,
        "method_metrics": all_metrics,
        "development_method_metrics": development_bundle["metrics"],
        "locked_test_method_metrics": locked_bundle["metrics"],
        "readiness_checks": {
            "development": development_gate,
            "development_status": _gate_text(development_gate),
            "locked_test": locked_gate,
            "locked_test_status": _gate_text(locked_gate),
        },
        "cross_split_collision_count": collision["cross_split_collision_count"],
        "f2b_status": "passed_blind_finite_query_characterization",
        "f2b_general_symbolic_contract_synthesis_validated": False,
        "f2b_relational_effect_completeness_validated": False,
        "f2b_query_generating_cegis_validated": False,
        "f1_locked_test_scored": False,
        "f2a_locked_test_rerun": False,
        "f2b_locked_test_rerun": False,
        "f2b_frozen_assets_modified": False,
        "locked_test_scored_once": True,
        "relational_effect_completeness_validated": False,
        "real_tool_completeness_validated": False,
        "formal_model_completed": False,
        "real_agent_integration_completed": False,
        "private_data_used": False,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "git": {
            "branch": preflight["git"]["branch"],
            "code_freeze_commit": preflight["git"]["commit"],
            "analyzer_freeze_commit": values["protocol"]["analyzer_freeze_commit"],
            "base_commit": values["protocol"]["required_base_commit"],
        },
    }

    write_json(artifact_dir / "resolved_config.json", values)
    write_json(artifact_dir / "upstream_frozen_sha256.json", preflight["frozen"])
    write_json(
        artifact_dir / "source_sha256_manifest.json", _source_manifest(config_path)
    )
    write_json(
        artifact_dir / "analyzer_freeze_manifest.json",
        {
            "schema_version": 1,
            "analyzer_freeze_commit": values["protocol"]["analyzer_freeze_commit"],
            "code_freeze_commit": preflight["git"]["commit"],
            "analyzer_source_hashes": values["frozen_analyzer"],
            "analyzer_modified_after_freeze": False,
            "locked_materialized_before_analyzer_freeze": False,
        },
    )
    write_json(
        artifact_dir / "solver_manifest.json",
        {
            "schema_version": 1,
            "solver": "Z3",
            "version": z3.get_version_string(),
            "standard_solver_component_not_algorithmic_innovation": True,
            "timeout_ms": selected_timeout,
        },
    )
    write_json(
        artifact_dir / "grammar_manifest.json",
        {
            "schema_version": 1,
            "selected_candidate": selected_candidate,
            "grammar": selected_grammar.to_dict(),
            "selected_before_development": True,
            "changed_after_development": False,
            "changed_after_locked": False,
        },
    )
    write_json(artifact_dir / "information_boundary_audit.json", boundary)
    write_json(artifact_dir / "collision_audit.json", collision)
    write_json(
        artifact_dir / "benchmark_manifest.json",
        {
            "schema_version": 1,
            "benchmark": benchmark,
            "splits": split_manifests,
            "evaluator_manifest_digest": canonical_digest(
                list(iter_evaluator_manifest(configured_cases))
            ),
            "analyzer_freeze_commit": values["protocol"]["analyzer_freeze_commit"],
            "locked_materialized_after_analyzer_freeze": True,
            "hidden_locked_implementation_ast_persisted": False,
            "gold_locked_contract_persisted": False,
        },
    )
    write_json(
        artifact_dir / "certificate_manifest.json", _certificate_manifest(all_rows)
    )
    write_json(artifact_dir / "stage_f2c_summary.json", summary)
    write_json(
        artifact_dir / "protocol_status.json",
        {
            "schema_version": 1,
            "formal_build": "completed",
            "analyzer_freeze_commit": values["protocol"]["analyzer_freeze_commit"],
            "code_freeze_commit": preflight["git"]["commit"],
            "materialization_runs": 1,
            "calibration_runs": 1,
            "development_runs": 1,
            "locked_test_runs": 1,
            "locked_test_scored_once": True,
            "f1_locked_test_scored": False,
            "f2a_locked_test_rerun": False,
            "f2b_locked_test_rerun": False,
            "configuration_changed_after_development": False,
            "configuration_changed_after_locked": False,
            "locked_counterexample_fed_back_to_analyzer": False,
        },
    )
    write_csv(artifact_dir / "calibration_results.csv", calibration_rows)
    write_csv(artifact_dir / "development_results.csv", development_bundle["metrics"])
    write_csv(artifact_dir / "locked_test_results.csv", locked_bundle["metrics"])
    write_csv(artifact_dir / "per_category_metrics.csv", per_category_metrics(all_rows))
    write_csv(artifact_dir / "per_field_metrics.csv", per_field_metrics(all_rows))
    write_csv(artifact_dir / "patch_metrics.csv", patch_metrics(all_rows))
    write_csv(artifact_dir / "query_efficiency.csv", query_efficiency(all_rows))
    write_csv(artifact_dir / "contract_complexity.csv", contract_complexity(all_rows))
    write_csv(
        artifact_dir / "unknown_reason_distribution.csv",
        unknown_reason_distribution(all_rows),
    )
    write_csv(artifact_dir / "performance_metrics.csv", performance_metrics(all_rows))
    report_path.write_text(_report(summary), encoding="utf-8")
    scan_paths = [
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "sensitive_material_scan.json"
    ] + [report_path]
    write_json(
        artifact_dir / "sensitive_material_scan.json", sensitive_scan(scan_paths)
    )
    write_json(
        artifact_dir / "artifact_sha256_manifest.json",
        artifact_manifest(artifact_dir, report_path),
    )
    validate_artifact_inventory(artifact_dir, report_path)
    validate_strict_serialization(artifact_dir, data_dir)
    mark_completed(runtime_dir, sha256_file(artifact_dir / "stage_f2c_summary.json"))
    return summary


def run_smoke() -> dict[str, Any]:
    grammar = GrammarLimits(4, 5, 16, True)
    cases = generate_symbolic_smoke_cases(
        "f2c-smoke-nonformal",
        grammar=grammar,
        replay_budget=16,
        solver_timeout_ms=3000,
    )[:10]
    bundle = evaluate_cases(
        cases,
        replay_budget=16,
        solver_timeout_ms=3000,
        methods=(SymbolicMethod.AUTHSYNTH,),
    )
    metrics = selected_metrics(bundle["metrics"])
    return {
        "status": "passed"
        if metrics["false_verified_complete_count"] == 0
        and metrics["forbidden_trace_acceptance_rate"] == 0.0
        else "failed",
        "case_count": len(cases),
        "evaluated_splits": ["train"],
        "development_scored": False,
        "locked_test_scored": False,
        "query_generating_cegis": True,
        "private_data_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_f2c.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_smoke() if args.smoke else run_formal_build(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
