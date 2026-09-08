"""Stage F2C 指标、gate 与分布表。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from keyed_gram.authsynth_symbolic_verifier.evaluator import (
    SymbolicEvaluationRow,
    SymbolicMethod,
    aggregate_metrics,
)


def selected_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(
        next(
            item for item in metrics if item["method"] == SymbolicMethod.AUTHSYNTH.value
        )
    )


def readiness_gate(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "false_verified_complete": metrics["false_verified_complete_count"]
        <= thresholds["maximum_false_verified_complete_count"],
        "forbidden_trace_acceptance": metrics["forbidden_trace_acceptance_rate"]
        <= thresholds["maximum_forbidden_trace_acceptance_rate"],
        "unseen_effect_recall": metrics["unseen_effect_recall"]
        >= thresholds["minimum_unseen_effect_recall"],
        "unseen_effect_precision": metrics["unseen_effect_precision"]
        >= thresholds["minimum_unseen_effect_precision"],
        "guard_precision": metrics["guard_precision"]
        >= thresholds["minimum_guard_precision"],
        "guard_recall": metrics["guard_recall"] >= thresholds["minimum_guard_recall"],
        "field_binding_accuracy": metrics["field_binding_accuracy"]
        >= thresholds["minimum_field_binding_accuracy"],
        "state_update_binding_accuracy": metrics["state_update_binding_accuracy"]
        >= thresholds["minimum_state_update_binding_accuracy"],
        "contract_overapproximation_ratio": metrics["contract_overapproximation_ratio"]
        <= thresholds["maximum_contract_overapproximation_ratio"],
        "safe_utility": metrics["safe_utility"] >= thresholds["minimum_safe_utility"],
        "maximal_permissiveness_ratio": metrics["maximal_permissiveness_ratio"]
        >= thresholds["minimum_maximal_permissiveness_ratio"],
        "replay_query_reduction": metrics["replay_query_reduction"]
        >= thresholds["minimum_replay_query_reduction"],
        "convergence_rate": metrics["convergence_rate"]
        >= thresholds["minimum_convergence_rate"],
        "unknown_rate": metrics["unknown_rate"] <= thresholds["maximum_unknown_rate"],
        "drift_detection_recall": metrics["drift_detection_recall"]
        >= thresholds["minimum_drift_detection_recall"],
        "old_certificate_reuse": metrics["old_certificate_reuse_after_drift_count"]
        <= thresholds["maximum_old_certificate_reuse_after_drift_count"],
    }


def _ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def per_category_metrics(rows: Sequence[SymbolicEvaluationRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SymbolicEvaluationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method, row.mutation_category)].append(row)
    result = []
    for (method, category), selected in sorted(grouped.items()):
        metric = next(
            item for item in aggregate_metrics(selected) if item["method"] == method
        )
        result.append({"mutation_category": category, **metric})
    return result


def per_field_metrics(rows: Sequence[SymbolicEvaluationRow]) -> list[dict[str, Any]]:
    fields = ("tenant", "resource", "destination", "amount", "phase", "state")
    result = []
    for method in sorted({item.method for item in rows}):
        selected = [item for item in rows if item.method == method]
        for field in fields:
            correct = sum(
                getattr(item, f"{field}_binding_correct") for item in selected
            )
            total = sum(getattr(item, f"{field}_binding_total") for item in selected)
            result.append(
                {
                    "method": method,
                    "field": field,
                    "correct_count": correct,
                    "total_count": total,
                    "accuracy": _ratio(correct, total),
                }
            )
    return result


def patch_metrics(rows: Sequence[SymbolicEvaluationRow]) -> list[dict[str, Any]]:
    result = []
    for method in sorted({item.method for item in rows}):
        selected = [item for item in rows if item.method == method]
        predicted = Counter(
            item for row in selected for item in set(row.predicted_patch_types)
        )
        expected = Counter(
            item for row in selected for item in set(row.expected_patch_types)
        )
        for patch_type in sorted(set(predicted) | set(expected)):
            true_positive = sum(
                patch_type in row.predicted_patch_types
                and patch_type in row.expected_patch_types
                for row in selected
            )
            result.append(
                {
                    "method": method,
                    "patch_type": patch_type,
                    "predicted_count": predicted[patch_type],
                    "expected_count": expected[patch_type],
                    "true_positive_count": true_positive,
                    "precision": _ratio(true_positive, predicted[patch_type]),
                    "recall": _ratio(true_positive, expected[patch_type]),
                }
            )
    return result


def query_efficiency(rows: Sequence[SymbolicEvaluationRow]) -> list[dict[str, Any]]:
    return [
        {
            "method": row.method,
            "case_id": row.case_id,
            "split": row.split,
            "domain_assignment_count": row.domain_assignment_count,
            "replay_query_count": row.replay_query_count,
            "solver_call_count": row.solver_call_count,
            "replay_query_reduction": 1.0
            - row.replay_query_count / row.domain_assignment_count,
        }
        for row in rows
    ]


def contract_complexity(rows: Sequence[SymbolicEvaluationRow]) -> list[dict[str, Any]]:
    return [
        {
            "method": row.method,
            "case_id": row.case_id,
            "split": row.split,
            "clause_count": row.contract_clause_count,
            "literal_count": row.contract_literal_count,
            "ast_node_count": row.contract_ast_node_count,
            "serialized_bytes": row.contract_size_bytes,
            "certificate_bytes": row.certificate_size_bytes,
        }
        for row in rows
    ]


def unknown_reason_distribution(
    rows: Sequence[SymbolicEvaluationRow],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        if not row.analysis_unknown:
            continue
        for reason in row.reason_codes or ("unspecified_unknown",):
            counts[(row.method, row.split, reason)] += 1
    return [
        {"method": method, "split": split, "reason_code": reason, "count": count}
        for (method, split, reason), count in sorted(counts.items())
    ]


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile_value)))
    return ordered[index]


def performance_metrics(rows: Sequence[SymbolicEvaluationRow]) -> list[dict[str, Any]]:
    result = []
    for method in sorted({item.method for item in rows}):
        selected = [item for item in rows if item.method == method]
        analysis = [item.analysis_time_ms for item in selected]
        synthesis = [item.synthesis_time_ms for item in selected]
        verification = [item.verification_time_ms for item in selected]
        result.append(
            {
                "method": method,
                "analysis_p50_ms": percentile(analysis, 0.50),
                "analysis_p95_ms": percentile(analysis, 0.95),
                "synthesis_p50_ms": percentile(synthesis, 0.50),
                "synthesis_p95_ms": percentile(synthesis, 0.95),
                "verification_p50_ms": percentile(verification, 0.50),
                "verification_p95_ms": percentile(verification, 0.95),
                "solver_call_count": sum(item.solver_call_count for item in selected),
                "replay_query_count": sum(item.replay_query_count for item in selected),
            }
        )
    return result
