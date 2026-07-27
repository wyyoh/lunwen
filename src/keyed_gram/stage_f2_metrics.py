"""Stage F2 compiler evaluation 的指标聚合。"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

SECURITY_COUNT_FIELDS = (
    "authority_amplification_count",
    "unauthorized_action_count",
    "cross_subject_capability_count",
    "cross_tenant_capability_count",
    "cross_resource_capability_count",
    "cross_action_capability_count",
    "cross_relation_capability_count",
    "cross_purpose_capability_count",
    "explicit_deny_bypass_count",
    "stale_policy_acceptance_count",
    "invalid_witness_acceptance_count",
    "tampered_capability_acceptance_count",
    "forged_signature_acceptance_count",
    "compositional_authority_amplification_count",
    "multi_step_forbidden_effect_acceptance_count",
    "workflow_state_replay_acceptance_count",
    "workflow_state_fork_acceptance_count",
    "semantic_proposal_direct_execution_count",
    "silent_scope_narrowing_count",
    "partial_execution_without_confirmation_count",
    "compiler_determinism_failure_count",
    "capability_verification_failure_on_valid_count",
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def aggregate_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合一个 split/variant，所有率的分母均显式记录。"""

    case_count = len(rows)
    allow_rows = [row for row in rows if row["oracle_decision"] == "allow"]
    deny_rows = [row for row in rows if row["oracle_decision"] == "deny"]
    predicted_denies = [row for row in rows if row["compiler_outcome"] == "deny"]
    exact = sum(row["exact_execution"] for row in allow_rows)
    recoverable = sum(row["recoverable_safe"] for row in allow_rows)
    true_denies = sum(
        row["oracle_decision"] == "deny" for row in predicted_denies
    )
    counts = {
        field: sum(int(row.get(field, 0)) for row in rows)
        for field in SECURITY_COUNT_FIELDS
    }
    capability_sizes = [
        float(row["capability_size_bytes"])
        for row in rows
        if row["capability_size_bytes"] > 0
    ]
    witness_sizes = [
        float(row["witness_size_bytes"])
        for row in rows
        if row["witness_size_bytes"] > 0
    ]
    workflow_sizes = [
        float(row["workflow_state_size_bytes"])
        for row in rows
        if row["workflow_state_size_bytes"] > 0
    ]
    compile_latencies = [float(row["compile_latency_ms"]) for row in rows]
    verify_latencies = [
        float(row["verify_latency_ms"])
        for row in rows
        if row["verify_latency_ms"] > 0
    ]
    result: dict[str, Any] = {
        "case_count": case_count,
        "oracle_allow_case_count": len(allow_rows),
        "oracle_deny_case_count": len(deny_rows),
        **counts,
        "authority_amplification_rate": (
            counts["authority_amplification_count"] / case_count
            if case_count
            else 0.0
        ),
        "unauthorized_action_rate": (
            counts["unauthorized_action_count"] / case_count
            if case_count
            else 0.0
        ),
        "clean_allow_exact_execution_rate": (
            exact / len(allow_rows) if allow_rows else 0.0
        ),
        "safe_utility": exact / len(allow_rows) if allow_rows else 0.0,
        "recoverable_safe_utility": (
            recoverable / len(allow_rows) if allow_rows else 0.0
        ),
        "authorized_authority_preservation_rate": (
            sum(row["authority_preserved"] for row in allow_rows) / len(allow_rows)
            if allow_rows
            else 0.0
        ),
        "unnecessary_rejection_rate": (
            sum(row["unnecessary_rejection"] for row in allow_rows) / len(allow_rows)
            if allow_rows
            else 0.0
        ),
        "confirmation_required_rate": (
            sum(row["confirmation_required"] for row in rows) / case_count
            if case_count
            else 0.0
        ),
        "correct_confirmation_rate": (
            sum(row["correct_confirmation"] for row in rows)
            / max(1, sum(row["confirmation_required"] for row in rows))
        ),
        "deny_precision": (
            true_denies / len(predicted_denies) if predicted_denies else 0.0
        ),
        "deny_recall": (
            sum(row["compiler_outcome"] == "deny" for row in deny_rows)
            / len(deny_rows)
            if deny_rows
            else 0.0
        ),
        "compile_latency_p50_ms": _percentile(compile_latencies, 0.50),
        "compile_latency_p95_ms": _percentile(compile_latencies, 0.95),
        "verify_latency_p50_ms": _percentile(verify_latencies, 0.50),
        "verify_latency_p95_ms": _percentile(verify_latencies, 0.95),
        "capability_size_mean_bytes": (
            mean(capability_sizes) if capability_sizes else 0.0
        ),
        "capability_size_p95_bytes": _percentile(capability_sizes, 0.95),
        "witness_size_mean_bytes": mean(witness_sizes) if witness_sizes else 0.0,
        "workflow_state_size_mean_bytes": (
            mean(workflow_sizes) if workflow_sizes else 0.0
        ),
    }
    return result


def aggregate_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["variant"])].append(row)
    return [
        {"split": split, "variant": variant, **aggregate_variant_rows(values)}
        for (split, variant), values in sorted(grouped.items())
    ]


def per_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["split"], row["variant"], row["attack_category"])
        ].append(row)
    return [
        {
            "split": split,
            "variant": variant,
            "attack_category": category,
            **aggregate_variant_rows(values),
        }
        for (split, variant, category), values in sorted(grouped.items())
    ]


def security_violation_matrix(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["split"], row["variant"])
        for field in SECURITY_COUNT_FIELDS:
            grouped[key][field] += int(row.get(field, 0))
    return [
        {
            "split": split,
            "variant": variant,
            **{field: counts[field] for field in SECURITY_COUNT_FIELDS},
        }
        for (split, variant), counts in sorted(grouped.items())
    ]
