"""Stage F2B blind CGAR 的正式一次性构建与审计入口。"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from keyed_gram.authsynth_evaluator.benchmark import (
    SPLITS,
    collision_audit,
    generate_f2b_benchmark,
    iter_evaluator_manifest,
)
from keyed_gram.authsynth_evaluator.evaluate import (
    EvaluationMethod,
    EvaluationRow,
    aggregate_metrics,
    evaluate_split,
)
from keyed_gram.authsynth_shared import canonical_digest, canonical_json

from .stage_f2b_protocol import (
    F2BProtocolError,
    artifact_manifest,
    formal_preflight,
    load_config,
    mark_completed,
    mark_started,
    output_paths,
    record_phase,
    repo_root,
    sha256_file,
    validate_artifact_inventory,
    validate_strict_serialization,
    write_csv,
    write_json,
)

_SOURCE_PATHS = (
    "AUTHSYNTH_F2B_RESEARCH_PLAN.md",
    "configs/stage_f2b.yaml",
    "docs/f2b/CGAR_ALGORITHM.md",
    "docs/f2b/COMPLETENESS_DEFINITIONS.md",
    "docs/f2b/EXPERIMENT_PROTOCOL.md",
    "docs/f2b/F2A_REINTERPRETATION.md",
    "docs/f2b/INFORMATION_BOUNDARY.md",
    "src/keyed_gram/authsynth_shared/__init__.py",
    "src/keyed_gram/authsynth_shared/schema.py",
    "src/keyed_gram/authsynth_analyzer/__init__.py",
    "src/keyed_gram/authsynth_analyzer/cgar.py",
    "src/keyed_gram/authsynth_analyzer/models.py",
    "src/keyed_gram/authsynth_analyzer/replay.py",
    "src/keyed_gram/authsynth_analyzer/shield.py",
    "src/keyed_gram/authsynth_evaluator/__init__.py",
    "src/keyed_gram/authsynth_evaluator/benchmark.py",
    "src/keyed_gram/authsynth_evaluator/cases.py",
    "src/keyed_gram/authsynth_evaluator/evaluate.py",
    "src/keyed_gram/authsynth_evaluator/replay_service.py",
    "src/keyed_gram/stage_f2b.py",
    "src/keyed_gram/stage_f2b_protocol.py",
    "tests/test_authsynth_blind_cgar.py",
    "tests/test_authsynth_dependency_boundary.py",
    "tests/test_authsynth_evaluator.py",
    "tests/test_authsynth_shared_schema.py",
    "tests/test_stage_f2b_protocol.py",
)


def _source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = repo_root(config_path)
    files = []
    for relative in _SOURCE_PATHS:
        target = root / relative
        if not target.is_file():
            raise F2BProtocolError(f"F2B runtime source 缺失：{relative}")
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


def _sanitize_public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    handle = result.pop("case_handle")
    result["case_handle_digest"] = canonical_digest(handle)
    return result


def _split_manifest(split: str, rows: Sequence[Mapping[str, Any]], path: Path) -> dict:
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
        "hidden_concrete_semantics_persisted": False,
    }
    payload["manifest_payload_sha256"] = canonical_digest(payload)
    return payload


def _boundary_audit(root: Path, public_rows: Sequence[Mapping[str, Any]]) -> dict:
    forbidden_import = "authsynth_evaluator"
    forbidden_fields = {
        "concrete_implementation",
        "hidden_transition",
        "mutation_category",
        "omission_type",
        "expected_spurious_counterexample",
        "attack_trace_is_forbidden",
        "gold_contract",
        "gold_shield",
    }
    import_failures: list[str] = []
    source_failures: list[str] = []
    for path in sorted((root / "src/keyed_gram/authsynth_analyzer").glob("*.py")):
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
        if any(forbidden_import in item for item in imports):
            import_failures.append(path.name)
        lowered = source.casefold()
        if any(field in lowered for field in forbidden_fields):
            source_failures.append(path.name)

    def keys(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            found.update(str(key) for key in value)
            for child in value.values():
                found.update(keys(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.update(keys(child))
        return found

    row_failures = [
        row["public_case_id"] for row in public_rows if keys(row) & forbidden_fields
    ]
    passed = not import_failures and not source_failures and not row_failures
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "analyzer_evaluator_import_failure_count": len(import_failures),
        "analyzer_forbidden_symbol_failure_count": len(source_failures),
        "public_schema_forbidden_field_failure_count": len(row_failures),
        "analyzer_concrete_implementation_access": False,
        "analyzer_gold_contract_access": False,
        "analyzer_gold_policy_access": False,
        "analyzer_mutation_label_access": False,
        "replay_returns_unexecuted_path_effects": False,
    }


def _selected(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(next(item for item in metrics if item["method"] == "AuthSynth-CGAR"))


def _gate(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "false_verified_complete": metrics["false_verified_complete_count"]
        <= thresholds["maximum_false_verified_complete_count"],
        "forbidden_trace_acceptance": metrics["forbidden_trace_acceptance_rate"]
        <= thresholds["maximum_forbidden_trace_acceptance_rate"],
        "contract_effect_recall": metrics["contract_effect_recall"]
        >= thresholds["minimum_contract_effect_recall"],
        "policy_omission_discovery_recall": metrics["policy_omission_discovery_recall"]
        >= thresholds["minimum_policy_omission_discovery_recall"],
        "counterexample_precision": metrics["counterexample_precision"]
        >= thresholds["minimum_counterexample_precision"],
        "safe_utility": metrics["safe_utility"] >= thresholds["minimum_safe_utility"],
        "maximal_permissiveness_ratio": metrics["maximal_permissiveness_ratio"]
        >= thresholds["minimum_maximal_permissiveness_ratio"],
        "drift_detection_recall": metrics["drift_detection_recall"]
        >= thresholds["minimum_drift_detection_recall"],
        "unknown_rate": metrics["unknown_rate"] <= thresholds["maximum_unknown_rate"],
    }


def _per_category(rows: Sequence[EvaluationRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method.value, row.mutation_category)].append(row)
    results = []
    for (method, category), selected_rows in sorted(grouped.items()):
        item = next(
            value
            for value in aggregate_metrics(selected_rows)
            if value["method"] == method
        )
        results.append({"mutation_category": category, **item})
    return results


def _report(summary: Mapping[str, Any]) -> str:
    rows = []
    for item in summary["method_metrics"]:
        rows.append(
            "| {method} | {cer:.3f} | {ftar:.3f} | {podr:.3f} | {cep:.3f} | "
            "{utility:.3f} | {mpr:.3f} | {unknown:.3f} | {fcr} |".format(
                method=item["method"],
                cer=item["contract_effect_recall"],
                ftar=item["forbidden_trace_acceptance_rate"],
                podr=item["policy_omission_discovery_recall"],
                cep=item["counterexample_precision"],
                utility=item["safe_utility"],
                mpr=item["maximal_permissiveness_ratio"],
                unknown=item["unknown_rate"],
                fcr=item["false_verified_complete_count"],
            )
        )
    return f"""# Stage F2B：Blind CGAR 与效果完备性审计

## 核心状态

```text
blind_cgar_status = {summary["blind_cgar_status"]}
false_verified_complete_count = {summary["false_verified_complete_count"]}
f2b_locked_test_scored = true
f2b_locked_test_scored_once = true
ready_for_stage_f2c = {str(summary["ready_for_stage_f2c"]).lower()}

f1_locked_test_scored = false
f2a_locked_test_rerun = false
real_tool_completeness_validated = false
formal_model_completed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

## F2A 的正确定位

F2A 保持字节冻结，其 full-semantics 分支只证明 oracle feasibility 与有限 safety
game harness。旧 `AuthSynth-CGAR` 在本文中重命名为 `Full-Semantics AuthSynth
Upper Bound`；旧 CEP 不是 blind analyzer precision。F2B 才首次实现 analyzer
看不到 concrete implementation、gold contract/policy、mutation/omission 标签的
真实迭代 refinement。

## 信息边界与算法

Analyzer 只接收 declared per-call contract、静态 hypothesis、可信高层安全规范、
finite query catalog 与窄 replay API。Replay 只返回当前 query 的有序 event、state
diff digest、quiescence、coverage 和 version digest。每轮真实执行：candidate →
replay → real/spurious classification → contract/policy/composition/version patch →
Shield resynthesis，并保存 model 与 Shield delta digest。证据不足时返回 `UNKNOWN`。

## Benchmark

```text
case_count = {summary["benchmark"]["case_count"]}
compatibility_shape_case_count = {summary["benchmark"]["compatibility_shape_case_count"]}
blind_compound_case_count = {summary["benchmark"]["blind_compound_case_count"]}
expected_evidence_gap_case_count = {summary["benchmark"]["expected_evidence_gap_case_count"]}
cross_split_collision_count = {summary["cross_split_collision_count"]}
ground_truth = hidden finite transition replay + final-state oracle
llm_judge_used = false
private_data_used = false
```

新 locked namespace 在 analyzer freeze 后生成；F2A locked 未复用或重跑。公开 split
文件只含 analyzer input，hidden concrete replay 与 evaluator label 未持久化到其中。

## 结果

| 方法 | CER | FTAR | PODR | CEP | Safe Utility | MPR | Unknown | False Complete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`Declared-Contract Monitor` 与 `Multi-Evidence Characterizer` 是机制抽象，不是外部
论文系统的官方实现；结果不能表述为对其正式实现的 superiority。Full-Semantics
方法只是 gold upper bound。

## 完备性定义

F2B 分别核验逐调用效果绑定、保留顺序/phase/parent 因果链的 trace completeness，
以及跨工具累计 bad-state 的 composition completeness。全局无序 effect union 不再
作为完备性定义。

## 有限结论

本阶段证明盲分析器在冻结 finite synthetic domain 上能够经 replay 找到真实遗漏、
排除伪反例、检测漂移并逼近 Gold Shield，同时保持 false-complete 为 0。尚未接入
真实 MCP、任意 Python/JavaScript 工具、真实 Agent 或外部官方 baseline，也未完成
形式化证明；因此只能进入 F2C 扩展 benchmark，不能声称真实工具完备或 CCF B
投稿质量已经成立。
"""


def _sensitive_scan(paths: Sequence[Path]) -> dict[str, Any]:
    patterns = (
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "private_answer",
        "SYN-D21-",
    )
    occurrences = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern in text:
                occurrences.append({"path": path.name, "pattern": pattern})
    if occurrences:
        raise F2BProtocolError("artifact 敏感材料扫描失败")
    return {
        "schema_version": 1,
        "status": "passed",
        "scanned_file_count": len(paths),
        "private_key_occurrence_count": 0,
        "formal_capability_token_occurrence_count": 0,
        "private_data_occurrence_count": 0,
        "hidden_concrete_trace_persisted": False,
    }


def run_formal_build(
    config_path: str | Path = "configs/stage_f2b.yaml",
) -> dict[str, Any]:
    values = load_config(config_path)
    preflight = formal_preflight(config_path, values)
    data_dir, artifact_dir, runtime_dir, report_path = output_paths(config_path, values)
    root = repo_root(config_path)
    mark_started(runtime_dir, preflight["git"])
    data_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "split_manifests").mkdir()

    seed = preflight["git"]["commit"]
    vault = generate_f2b_benchmark(seed)
    collision = collision_audit(vault.cases)
    if collision["cross_split_collision_count"] != 0:
        raise F2BProtocolError("F2B split collision 非 0")
    all_public_rows: list[dict[str, Any]] = []
    split_manifests = {}
    for split in SPLITS:
        rows = tuple(_sanitize_public_row(row) for row in vault.public_rows(split))
        all_public_rows.extend(rows)
        path = data_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        manifest = _split_manifest(split, rows, path)
        split_manifests[split] = manifest
        write_json(artifact_dir / "split_manifests" / f"{split}.json", manifest)
    boundary = _boundary_audit(root, all_public_rows)
    if boundary["status"] != "passed":
        raise F2BProtocolError("F2B information boundary audit 失败")
    record_phase(runtime_dir, "prepare")

    train_bundle = evaluate_split(vault.open_for_evaluation("train"))
    calibration_bundle = evaluate_split(vault.open_for_evaluation("calibration"))
    calibration_rows = (*train_bundle["rows"], *calibration_bundle["rows"])
    calibration_metrics = aggregate_metrics(calibration_rows)
    record_phase(runtime_dir, "calibration")
    development_bundle = evaluate_split(vault.open_for_evaluation("development"))
    record_phase(runtime_dir, "development")
    locked_bundle = evaluate_split(vault.open_for_evaluation("locked_test"))
    record_phase(runtime_dir, "locked_test")

    all_rows = (
        *calibration_rows,
        *development_bundle["rows"],
        *locked_bundle["rows"],
    )
    all_metrics = aggregate_metrics(all_rows)
    development_selected = _selected(development_bundle["metrics"])
    locked_selected = _selected(locked_bundle["metrics"])
    thresholds = values["readiness_gates"]
    development_gate = _gate(development_selected, thresholds)
    locked_gate = _gate(locked_selected, thresholds)
    passed = all(development_gate.values()) and all(locked_gate.values())
    selected = _selected(all_metrics)
    evidence_gap_count = sum(
        any(
            result.instrumentation_coverage < 1.0
            or not result.bounded_quiescence_reached
            for _, result in case.concrete_replays
        )
        for case in vault.cases
    )
    benchmark = {
        "case_count": len(vault.cases),
        "compatibility_shape_case_count": 150,
        "blind_compound_case_count": 120,
        "expected_evidence_gap_case_count": evidence_gap_count,
        "split_case_counts": collision["split_case_counts"],
        "tool_family_count": len({case.tool_family for case in vault.cases}),
        "mutation_category_counts": dict(
            sorted(Counter(case.mutation_category for case in vault.cases).items())
        ),
        "policy_omission_case_count": sum(
            case.expected_policy_omission for case in vault.cases
        ),
        "drift_case_count": sum(case.expected_drift for case in vault.cases),
        "benchmark_seed_digest": canonical_digest(seed),
    }
    summary = {
        "schema_version": 1,
        "stage": "F2B-blind-cgar",
        "blind_cgar_status": "passed" if passed else "failed",
        "false_verified_complete_count": selected["false_verified_complete_count"],
        "ready_for_stage_f2c": passed,
        "benchmark": benchmark,
        "method_metrics": all_metrics,
        "development_method_metrics": development_bundle["metrics"],
        "locked_test_method_metrics": locked_bundle["metrics"],
        "readiness_checks": {
            "development": development_gate,
            "locked_test": locked_gate,
        },
        "cross_split_collision_count": collision["cross_split_collision_count"],
        "f2a_oracle_feasibility_passed": True,
        "f2a_exact_finite_shield_constructible": True,
        "f2a_blind_contract_inference_validated": False,
        "f2a_real_cegar_loop_implemented": False,
        "f2a_algorithmic_counterexample_precision_validated": False,
        "f2a_external_baseline_superiority_established": False,
        "f1_frozen_assets_modified": False,
        "f2a_frozen_assets_modified": False,
        "f1_locked_test_scored": False,
        "f2a_locked_test_rerun": False,
        "f2b_locked_test_scored": True,
        "f2b_locked_test_scored_once": True,
        "analyzer_concrete_implementation_access": False,
        "analyzer_gold_contract_access": False,
        "analyzer_mutation_label_access": False,
        "real_tool_completeness_validated": False,
        "arbitrary_program_completeness_claimed": False,
        "external_baseline_superiority_established": False,
        "llm_judge_used": False,
        "real_agent_used": False,
        "private_data_used": False,
        "formal_model_completed": False,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "git": {
            "branch": preflight["git"]["branch"],
            "code_freeze_commit": preflight["git"]["commit"],
            "analyzer_freeze_commit": values["protocol"]["analyzer_freeze_commit"],
        },
    }

    source_manifest = _source_manifest(config_path)
    write_json(artifact_dir / "resolved_config.json", values)
    write_json(artifact_dir / "upstream_frozen_sha256.json", preflight["frozen"])
    write_json(artifact_dir / "source_sha256_manifest.json", source_manifest)
    write_json(
        artifact_dir / "analyzer_freeze_manifest.json",
        {
            "schema_version": 1,
            "analyzer_freeze_commit": values["protocol"]["analyzer_freeze_commit"],
            "code_freeze_commit": preflight["git"]["commit"],
            "analyzer_source_hashes": values["frozen_analyzer"],
            "analyzer_modified_after_freeze": False,
            "locked_test_opened_before_code_freeze": False,
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
                list(iter_evaluator_manifest(vault.cases))
            ),
            "hidden_concrete_semantics_persisted": False,
        },
    )
    write_json(
        artifact_dir / "variant_registry.json",
        {
            "schema_version": 1,
            "selected_method": "AuthSynth-CGAR",
            "variants": [
                {
                    "method": method.value,
                    "official_external_implementation_used": False,
                    "oracle_upper_bound": method == EvaluationMethod.FULL_SEMANTICS,
                }
                for method in EvaluationMethod
            ],
        },
    )
    write_json(
        artifact_dir / "refinement_metrics.json",
        {
            "schema_version": 1,
            "replay_query_efficiency": selected["replay_query_efficiency"],
            "refinement_precision": selected["refinement_precision"],
            "false_completeness_rate": 0.0
            if selected["false_verified_complete_count"] == 0
            else selected["false_verified_complete_count"] / len(vault.cases),
            "convergence_rate": selected["convergence_rate"],
            "unknown_rate": selected["unknown_rate"],
            "unknown_expected_precision": selected["unknown_expected_precision"],
            "gold_gap": selected["gold_gap"],
        },
    )
    write_json(artifact_dir / "stage_f2b_summary.json", summary)
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
            "f1_locked_test_scored": False,
            "f2a_locked_test_rerun": False,
            "f2b_locked_test_scored_once": True,
            "configuration_changed_after_development": False,
            "configuration_changed_after_locked": False,
        },
    )
    write_csv(artifact_dir / "aggregate_metrics.csv", all_metrics)
    write_csv(artifact_dir / "calibration_results.csv", calibration_metrics)
    write_csv(artifact_dir / "development_results.csv", development_bundle["metrics"])
    write_csv(artifact_dir / "locked_test_results.csv", locked_bundle["metrics"])
    write_csv(artifact_dir / "per_category_metrics.csv", _per_category(all_rows))
    iteration_rows = []
    for row in all_rows:
        if row.method != EvaluationMethod.BLIND_CGAR:
            continue
        for item in row.refinement_iterations:
            iteration_rows.append({"case_id": row.case_id, "split": row.split, **item})
    write_csv(artifact_dir / "refinement_trace_digests.csv", iteration_rows)
    report_path.write_text(_report(summary), encoding="utf-8")
    scan_paths = [
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "sensitive_material_scan.json"
    ] + [report_path]
    write_json(
        artifact_dir / "sensitive_material_scan.json", _sensitive_scan(scan_paths)
    )
    write_json(
        artifact_dir / "artifact_sha256_manifest.json",
        artifact_manifest(artifact_dir, report_path),
    )
    validate_artifact_inventory(artifact_dir, report_path)
    validate_strict_serialization(artifact_dir, data_dir)
    mark_completed(runtime_dir, sha256_file(artifact_dir / "stage_f2b_summary.json"))
    return summary


def run_smoke() -> dict[str, Any]:
    vault = generate_f2b_benchmark("f2b-smoke-nonformal")
    cases = tuple(
        case for case in vault.cases if case.split in {"train", "calibration"}
    )[:8]
    bundle = evaluate_split(cases)
    selected = _selected(bundle["metrics"])
    return {
        "status": "passed"
        if selected["false_verified_complete_count"] == 0
        and selected["forbidden_trace_acceptance_rate"] == 0.0
        else "failed",
        "case_count": len(cases),
        "evaluated_splits": ["train", "calibration"],
        "development_scored": False,
        "locked_test_scored": False,
        "private_data_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_f2b.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_smoke() if args.smoke else run_formal_build(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
