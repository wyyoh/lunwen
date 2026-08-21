"""AuthEffectBench feasibility 指标与 readiness gate。"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .methods import CaseEvaluation, Method
from .types import OmissionType


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def aggregate_method(rows: Sequence[CaseEvaluation]) -> dict[str, Any]:
    if not rows:
        raise ValueError("method metrics 不允许空 rows")
    forbidden = sum(row.attack_concrete_forbidden for row in rows)
    accepted_forbidden = sum(row.forbidden_trace_accepted for row in rows)
    benign = sum(row.benign_concrete_safe for row in rows)
    benign_allowed = sum(
        row.benign_concrete_safe and row.benign_allowed for row in rows
    )
    actual_effects = sum(row.contract_effect_actual_count for row in rows)
    covered_effects = sum(row.contract_effect_covered_count for row in rows)
    declared_effects = sum(row.contract_effect_declared_count for row in rows)
    false_effects = sum(row.contract_effect_false_positive_count for row in rows)
    policy_omissions = sum(
        row.omission_type
        in {
            OmissionType.POLICY_OMISSION.value,
            OmissionType.COMPOSITION_OMISSION.value,
        }
        for row in rows
    )
    policy_discovered = sum(row.policy_omission_discovered for row in rows)
    cex_generated = sum(row.counterexamples_generated for row in rows)
    cex_validated = sum(row.counterexamples_validated for row in rows)
    drift_cases = sum(
        row.omission_type == OmissionType.IMPLEMENTATION_DRIFT.value for row in rows
    )
    drift_detected = sum(row.drift_detected for row in rows)
    safe_actions = sum(row.gold_safe_action_count for row in rows)
    safe_actions_allowed = sum(row.allowed_gold_safe_action_count for row in rows)
    latencies = sorted(row.analysis_time_ms for row in rows)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    certificate_sizes = [
        row.certificate_size_bytes for row in rows if row.certificate_size_bytes > 0
    ]
    return {
        "method": rows[0].method.value,
        "case_count": len(rows),
        "contract_effect_recall": _ratio(covered_effects, actual_effects),
        "contract_overapproximation_ratio": _ratio(false_effects, declared_effects),
        "forbidden_trace_acceptance_rate": _ratio(accepted_forbidden, forbidden),
        "forbidden_trace_accepted_count": accepted_forbidden,
        "policy_omission_discovery_recall": _ratio(policy_discovered, policy_omissions),
        "counterexample_precision": (
            1.0 if cex_generated == 0 else _ratio(cex_validated, cex_generated)
        ),
        "counterexample_count": cex_generated,
        "validated_counterexample_count": cex_validated,
        "safe_utility": _ratio(benign_allowed, benign),
        "maximal_permissiveness_ratio": _ratio(safe_actions_allowed, safe_actions),
        "drift_detection_recall": _ratio(drift_detected, drift_cases),
        "unknown_rate": 0.0,
        "analysis_time_mean_ms": statistics.fmean(latencies),
        "analysis_time_p95_ms": latencies[p95_index],
        "cegar_iterations_mean": statistics.fmean(row.cegar_iterations for row in rows),
        "certificate_size_mean_bytes": (
            statistics.fmean(certificate_sizes) if certificate_sizes else 0.0
        ),
        "exact_shield_match_rate": _ratio(
            sum(row.exact_shield_match for row in rows), len(rows)
        ),
    }


def aggregate_all(
    rows: Iterable[CaseEvaluation],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[Method, list[CaseEvaluation]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
    return tuple(
        aggregate_method(grouped[method]) for method in Method if method in grouped
    )


def distinguishing_categories(
    rows: Sequence[CaseEvaluation],
) -> tuple[str, ...]:
    baseline_bad = {
        row.mutation_category
        for row in rows
        if row.method in {Method.SOLVER_POLICY, Method.TOOLGATE}
        and row.forbidden_trace_accepted
    }
    cgar_bad = {
        row.mutation_category
        for row in rows
        if row.method == Method.AUTHSYNTH_CGAR and row.forbidden_trace_accepted
    }
    return tuple(sorted((baseline_bad - cgar_bad) - {"clean_control"}))


def readiness_gate(
    metrics: Mapping[str, Any],
    *,
    distinguishing_count: int,
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    checks = {
        "contract_effect_recall": metrics["contract_effect_recall"]
        >= thresholds["minimum_contract_effect_recall"],
        "forbidden_trace_acceptance_rate": metrics["forbidden_trace_acceptance_rate"]
        <= thresholds["maximum_forbidden_trace_acceptance_rate"],
        "policy_omission_discovery_recall": metrics["policy_omission_discovery_recall"]
        >= thresholds["minimum_policy_omission_discovery_recall"],
        "counterexample_precision": metrics["counterexample_precision"]
        >= thresholds["minimum_counterexample_precision"],
        "safe_utility": metrics["safe_utility"] >= thresholds["minimum_safe_utility"],
        "maximal_permissiveness_ratio": metrics["maximal_permissiveness_ratio"]
        >= thresholds["minimum_maximal_permissiveness_ratio"],
        "drift_detection_recall": metrics["drift_detection_recall"]
        >= thresholds["minimum_drift_detection_recall"],
        "distinguishing_counterexample_types": distinguishing_count
        >= thresholds["minimum_distinguishing_counterexample_types"],
    }
    return checks
