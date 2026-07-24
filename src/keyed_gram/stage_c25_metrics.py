"""Stage C2.5 外部 OOS、选择性风险与跨 seed 汇总指标。"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .stage_c25_data import IntentExample
from .stage_c25_router import IntentDecision


def _rate(numerator: int | float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def binary_auroc(scores: Sequence[float], targets: Sequence[bool]) -> float:
    if len(scores) != len(targets) or not scores:
        return 0.0
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return 0.0
    ordered = sorted(
        range(len(scores)), key=lambda index: (scores[index], index)
    )
    ranks = [0.0] * len(scores)
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[offset]]:
            end += 1
        average_rank = (offset + 1 + end) / 2.0
        for index in ordered[offset:end]:
            ranks[index] = average_rank
        offset = end
    positive_rank_sum = sum(
        rank for rank, target in zip(ranks, targets) if target
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def binary_aupr(scores: Sequence[float], targets: Sequence[bool]) -> float:
    if len(scores) != len(targets) or not scores or not any(targets):
        return 0.0
    ordered = sorted(
        range(len(scores)), key=lambda index: (-scores[index], index)
    )
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(ordered, start=1):
        if targets[index]:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / sum(targets)


def risk_coverage_curve(
    rows: Sequence[IntentExample],
    decisions: Sequence[IntentDecision],
    confidence: Sequence[float],
    *,
    points: int = 20,
) -> tuple[list[dict[str, float]], float]:
    ordered = sorted(
        range(len(rows)), key=lambda index: (-confidence[index], index)
    )
    errors = [
        rows[index].sample_type != "known"
        or decisions[index].predicted_intent != rows[index].label
        for index in ordered
    ]
    cumulative, risks = 0, []
    all_curve = []
    for rank, error in enumerate(errors, start=1):
        cumulative += int(error)
        risk = cumulative / rank
        risks.append(risk)
        all_curve.append(
            {"coverage": rank / len(rows), "risk": risk}
        )
    selected = []
    for point in range(1, points + 1):
        index = max(0, math.ceil(len(rows) * point / points) - 1)
        selected.append(all_curve[index])
    return selected, sum(risks) / len(risks)


def evaluate_external(
    rows: Sequence[IntentExample],
    scores: Tensor,
    intents: Sequence[str],
    decisions: Sequence[IntentDecision],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    matrix = scores.detach().cpu().float()
    if (
        matrix.ndim != 2
        or len(matrix) != len(rows)
        or matrix.shape[1] != len(intents)
        or len(decisions) != len(rows)
    ):
        raise ValueError("C2.5 metrics 输入不对齐")
    known = [
        index for index, row in enumerate(rows) if row.sample_type == "known"
    ]
    unknown = [
        index for index, row in enumerate(rows) if row.sample_type == "unknown"
    ]
    accepted_known = [
        index for index in known if decisions[index].status == "accept"
    ]
    safe = [
        index
        for index in accepted_known
        if decisions[index].predicted_intent == rows[index].label
    ]
    wrong = [
        index
        for index in accepted_known
        if decisions[index].predicted_intent != rows[index].label
    ]
    false_access = [
        index for index in unknown if decisions[index].status == "accept"
    ]
    rejected = [
        index for index, decision in enumerate(decisions)
        if decision.status == "reject"
    ]
    confidence = matrix.max(dim=1).values.tolist()
    reject_score = [-float(value) for value in confidence]
    unknown_targets = [row.sample_type == "unknown" for row in rows]
    curve, aurc = risk_coverage_curve(rows, decisions, confidence)
    families: dict[str, list[int]] = defaultdict(list)
    for index in known:
        assert rows[index].label is not None
        families[rows[index].label].append(index)
    per_intent = {
        label: _rate(
            sum(
                decisions[index].status == "accept"
                and decisions[index].predicted_intent == rows[index].label
                for index in indices
            ),
            len(indices),
        )
        for label, indices in families.items()
    }
    true_rejects = sum(index in unknown for index in rejected)
    false_rejects = sum(index in known for index in rejected)
    reject_precision = _rate(true_rejects, true_rejects + false_rejects)
    reject_recall = _rate(true_rejects, len(unknown))
    oos_f1 = (
        2 * reject_precision * reject_recall / (reject_precision + reject_recall)
        if reject_precision + reject_recall
        else 0.0
    )
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({rows[index].unknown_kind for index in unknown}):
        indices = [
            index for index in unknown if rows[index].unknown_kind == kind
        ]
        accepted = sum(decisions[index].status == "accept" for index in indices)
        by_kind[kind] = {
            "row_count": float(len(indices)),
            "false_memory_access_rate": _rate(accepted, len(indices)),
            "rejection_rate": _rate(len(indices) - accepted, len(indices)),
        }
    metrics: dict[str, Any] = {
        "row_count": len(rows),
        "known_row_count": len(known),
        "unknown_row_count": len(unknown),
        "known_accuracy": _rate(len(safe), len(known)),
        "known_coverage": _rate(len(accepted_known), len(known)),
        "accepted_route_accuracy": _rate(len(safe), len(accepted_known)),
        "safe_coverage": _rate(len(safe), len(known)),
        "wrong_bucket_access_rate": _rate(len(wrong), len(known)),
        "false_memory_access_rate": _rate(len(false_access), len(unknown)),
        "oos_recall": reject_recall,
        "oos_precision": reject_precision,
        "oos_f1": oos_f1,
        "oos_auroc": binary_auroc(reject_score, unknown_targets),
        "oos_aupr": binary_aupr(reject_score, unknown_targets),
        "abstention_rate": _rate(len(rejected), len(rows)),
        "accepted_relation_precision": _rate(
            len(safe), sum(decision.status == "accept" for decision in decisions)
        ),
        "worst_intent_accuracy": min(per_intent.values()) if per_intent else 0.0,
        "intent_macro_accuracy": mean(per_intent.values()) if per_intent else 0.0,
        "average_candidate_set_size": mean(
            len(decision.candidate_set) for decision in decisions
        ),
        "empty_candidate_set_rate": _rate(
            sum(not decision.candidate_set for decision in decisions), len(rows)
        ),
        "multi_candidate_set_rate": _rate(
            sum(len(decision.candidate_set) > 1 for decision in decisions),
            len(rows),
        ),
        "unknown_reject_reason_rate": _rate(
            sum(decision.reason == "unknown" for decision in decisions), len(rows)
        ),
        "ambiguous_reject_reason_rate": _rate(
            sum(decision.reason == "ambiguous" for decision in decisions), len(rows)
        ),
        "aurc": aurc,
        "unknown_kind_metrics": by_kind,
        "cross_intent_candidate_access_count": 0,
        "memory_access_count": sum(
            decision.status == "accept" for decision in decisions
        ),
    }
    return metrics, curve


def aggregate_seed_metrics(
    rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["benchmark"]), str(row["variant"]))].append(row)
    output = []
    for (benchmark, variant), values in sorted(groups.items()):
        for metric in metric_names:
            samples = [float(row[metric]) for row in values]
            sample_mean = mean(samples)
            standard_error = (
                stdev(samples) / math.sqrt(len(samples))
                if len(samples) > 1
                else 0.0
            )
            output.append(
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "metric": metric,
                    "seed_count": len(samples),
                    "mean": sample_mean,
                    "ci95_low": sample_mean - 1.96 * standard_error,
                    "ci95_high": sample_mean + 1.96 * standard_error,
                    "minimum": min(samples),
                    "maximum": max(samples),
                }
            )
    return output

