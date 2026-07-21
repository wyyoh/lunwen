"""Pure metric and calibration utilities for the Stage C2.4 audit.

This module performs no I/O.  Reject-guard selection accepts public-calibration
arrays only, uses known-row ECE alone for temperature selection, and freezes a
threshold at the requested known coverage before comparing preregistered scores.
"""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .stage_c23_semantic import (
    binary_auroc,
    binary_average_precision,
    expected_calibration_error,
)
from .stage_c24_contract import RELATION_ORDER, RelationId


_SCORE_NAMES = ("G1_max_softmax", "G2_top_logit_margin", "G3_definition_margin")
_REJECT_GROUPS = frozenset({"known", "ambiguous", "unrelated"})
_RESEARCH_GATES = {
    "known_detection_auroc": (">=", 0.95),
    "tnr_at_95_known_coverage": (">=", 0.80),
    "ambiguous_false_accept_rate": ("<=", 0.20),
    "unrelated_false_accept_rate": ("<=", 0.10),
    "worst_reject_family_false_accept_rate": ("<=", 0.30),
    "known_ece": ("<=", 0.15),
    "known_coverage": (">=", 0.95),
}


def _relation_order(
    values: Sequence[RelationId | str],
) -> tuple[RelationId, ...]:
    try:
        order = tuple(
            value if isinstance(value, RelationId) else RelationId(str(value))
            for value in values
        )
    except ValueError as exc:
        raise ValueError("relation class order contains an unknown relation") from exc
    if len(order) != len(RELATION_ORDER) or len(set(order)) != len(order):
        raise ValueError("relation class order must contain three unique relations")
    if set(order) != set(RELATION_ORDER):
        raise ValueError("relation class order does not match the C2.4 closed set")
    return order


def _relation_value(
    value: RelationId | str | int,
    order: tuple[RelationId, ...],
) -> RelationId:
    if isinstance(value, RelationId):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(order):
            return order[value]
        raise ValueError("relation class index is outside the class order")
    try:
        return RelationId(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown relation label {value!r}") from exc


def _pairwise_cross_group_agreement(
    predictions: Sequence[RelationId],
    outer_groups: Sequence[str],
    inner_groups: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(outer_groups):
        grouped[str(group)].append(index)
    per_group: dict[str, float] = {}
    total_equal = 0
    total_pairs = 0
    for group, indices in sorted(grouped.items()):
        equal = 0
        pairs = 0
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if str(inner_groups[left]) == str(inner_groups[right]):
                    continue
                pairs += 1
                equal += predictions[left] is predictions[right]
        # A group with no cross-unit comparison has no observed disagreement.
        per_group[group] = equal / pairs if pairs else 1.0
        total_equal += equal
        total_pairs += pairs
    return {
        "macro_agreement": sum(per_group.values()) / len(per_group),
        "pair_weighted_agreement": total_equal / total_pairs if total_pairs else 1.0,
        "num_cross_unit_pairs": total_pairs,
        "per_group_agreement": per_group,
    }


def closed_route_metrics(
    predictions: Sequence[RelationId | str | int],
    targets: Sequence[RelationId | str | int],
    families: Sequence[str],
    frames: Sequence[str],
    *,
    relation_order: Sequence[RelationId | str] = RELATION_ORDER,
) -> dict[str, Any]:
    """Compute closed-set relation routing and cross-template consistency."""

    if not predictions or not (
        len(predictions) == len(targets) == len(families) == len(frames)
    ):
        raise ValueError("closed route metric inputs must be non-empty and aligned")
    order = _relation_order(relation_order)
    predicted = [_relation_value(value, order) for value in predictions]
    target = [_relation_value(value, order) for value in targets]
    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        family_indices[str(family)].append(index)
    per_family = {
        family: sum(predicted[index] is target[index] for index in indices)
        / len(indices)
        for family, indices in sorted(family_indices.items())
    }
    micro = sum(left is right for left, right in zip(predicted, target)) / len(target)
    macro = sum(per_family.values()) / len(per_family)
    lookup = {relation: index for index, relation in enumerate(order)}
    confusion = [[0 for _ in order] for _ in order]
    for prediction, truth in zip(predicted, target):
        confusion[lookup[truth]][lookup[prediction]] += 1

    family_frame = _pairwise_cross_group_agreement(
        predicted,
        [str(value) for value in families],
        [str(value) for value in frames],
    )
    relation_family = _pairwise_cross_group_agreement(
        predicted,
        [value.value for value in target],
        [str(value) for value in families],
    )
    return {
        "relation_micro_accuracy": micro,
        "relation_family_macro_accuracy": macro,
        "worst_family_accuracy": min(per_family.values()),
        "num_families": len(per_family),
        "per_family_accuracy": per_family,
        "confusion_matrix": {
            "class_order": [value.value for value in order],
            "rows": "target",
            "columns": "prediction",
            "matrix": confusion,
        },
        "same_family_cross_frame_route_agreement": family_frame,
        "same_relation_cross_phrase_family_route_agreement": relation_family,
    }


def _canonical_logits(
    logits: Tensor,
    *,
    class_order: Sequence[RelationId | str],
    relation_order: Sequence[RelationId | str] = RELATION_ORDER,
) -> tuple[Tensor, tuple[RelationId, ...]]:
    source_order = _relation_order(class_order)
    target_order = _relation_order(relation_order)
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("reject score logits must be a matrix")
    if logits.shape[1] != len(source_order):
        raise ValueError("reject score logit width does not match class order")
    indices = [source_order.index(relation) for relation in target_order]
    return logits.detach().cpu().float()[:, indices], target_order


def _temperature(value: float) -> float:
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return temperature


def _score_components(logits: Tensor, *, temperature: float) -> dict[str, Tensor]:
    scaled = logits / _temperature(temperature)
    finite = torch.isfinite(scaled).all(dim=1)
    safe = torch.where(finite[:, None], scaled, torch.zeros_like(scaled))
    probabilities = safe.softmax(dim=1)
    confidence, indices = probabilities.max(dim=1)
    top_two = safe.topk(2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    nan = torch.full_like(confidence, float("nan"))
    return {
        "confidence": torch.where(finite, confidence, nan),
        "margin": torch.where(finite, margin, nan),
        "predicted_indices": torch.where(
            finite, indices, torch.full_like(indices, -1)
        ),
        "finite": finite,
    }


def g1_max_softmax_probability(
    head_logits: Tensor,
    *,
    class_order: Sequence[RelationId | str],
    temperature: float = 1.0,
    relation_order: Sequence[RelationId | str] = RELATION_ORDER,
) -> Tensor:
    canonical, _ = _canonical_logits(
        head_logits, class_order=class_order, relation_order=relation_order
    )
    return _score_components(canonical, temperature=temperature)["confidence"]


def g2_top_logit_margin(
    head_logits: Tensor,
    *,
    class_order: Sequence[RelationId | str],
    temperature: float = 1.0,
    relation_order: Sequence[RelationId | str] = RELATION_ORDER,
) -> Tensor:
    canonical, _ = _canonical_logits(
        head_logits, class_order=class_order, relation_order=relation_order
    )
    return _score_components(canonical, temperature=temperature)["margin"]


def g3_definition_similarity_margin(
    definition_logits: Tensor,
    *,
    class_order: Sequence[RelationId | str],
    temperature: float = 1.0,
    relation_order: Sequence[RelationId | str] = RELATION_ORDER,
) -> Tensor:
    canonical, _ = _canonical_logits(
        definition_logits, class_order=class_order, relation_order=relation_order
    )
    return _score_components(canonical, temperature=temperature)["margin"]


def extract_reject_score_candidates(
    head_logits: Tensor,
    definition_logits: Tensor,
    *,
    head_class_order: Sequence[RelationId | str],
    definition_class_order: Sequence[RelationId | str],
    temperature: float = 1.0,
    relation_order: Sequence[RelationId | str] = RELATION_ORDER,
) -> dict[str, Any]:
    """Extract G1/G2/G3 and the discrete relation-head predictions."""

    head, order = _canonical_logits(
        head_logits,
        class_order=head_class_order,
        relation_order=relation_order,
    )
    definitions, definition_order = _canonical_logits(
        definition_logits,
        class_order=definition_class_order,
        relation_order=relation_order,
    )
    if order != definition_order or len(head) != len(definitions):
        raise ValueError("head and definition score inputs must be row-aligned")
    # Ranking scores stay at temperature 1.0.  Temperature scaling is used only
    # for calibrated confidence/ECE, so it cannot improve a candidate's AUROC.
    raw_head_values = _score_components(head, temperature=1.0)
    raw_definition_values = _score_components(definitions, temperature=1.0)
    calibrated_head_values = _score_components(head, temperature=temperature)
    calibrated_definition_values = _score_components(
        definitions, temperature=temperature
    )
    predicted_relations: list[RelationId | None] = []
    for index in raw_head_values["predicted_indices"].tolist():
        predicted_relations.append(None if index < 0 else order[index])
    return {
        "relation_order": order,
        "predicted_relations": tuple(predicted_relations),
        "scores": {
            "G1_max_softmax": raw_head_values["confidence"],
            "G2_top_logit_margin": raw_head_values["margin"],
            "G3_definition_margin": raw_definition_values["margin"],
        },
        "calibration_confidence": {
            "G1_max_softmax": calibrated_head_values["confidence"],
            "G2_top_logit_margin": calibrated_head_values["confidence"],
            "G3_definition_margin": calibrated_definition_values["confidence"],
        },
    }


def _optional_relation(
    value: RelationId | str | None,
) -> RelationId | None:
    if value is None:
        return None
    if isinstance(value, RelationId):
        return value
    try:
        return RelationId(str(value))
    except ValueError:
        return None


def _safe_detection_scores(scores: Tensor, valid_route: Tensor) -> tuple[Tensor, Tensor]:
    values = scores.detach().cpu().float().flatten()
    finite = torch.isfinite(values) & valid_route
    finite_values = values[finite]
    dtype_minimum = float(torch.finfo(values.dtype).min)
    if len(finite_values):
        minimum = float(finite_values.min())
        scale = max(1.0, abs(minimum))
        floor = max(minimum - scale, dtype_minimum)
    else:
        floor = -1.0
    return torch.where(finite, values, torch.full_like(values, floor)), finite


def _known_coverage_threshold(
    safe_scores: Tensor,
    known: Tensor,
    valid: Tensor,
    target: float,
) -> tuple[float, Tensor, bool]:
    if not 0 < target <= 1:
        raise ValueError("known coverage target must be in (0, 1]")
    known_count = int(known.sum())
    required = max(1, math.ceil(target * known_count))
    eligible = safe_scores[known & valid].sort(descending=True).values
    attainable = len(eligible) >= required
    if not len(eligible):
        threshold = float(torch.finfo(safe_scores.dtype).max)
    elif attainable:
        threshold = float(eligible[required - 1])
    else:
        threshold = float(eligible[-1])
    accepted = valid & safe_scores.ge(threshold)
    return threshold, accepted, attainable


def _coverage_risk_curve(
    safe_known_scores: Tensor,
    known_correct: Tensor,
) -> dict[str, Any]:
    order = torch.argsort(safe_known_scores, descending=True, stable=True)
    scores = safe_known_scores[order]
    correct = known_correct[order].float()
    count = len(correct)
    cumulative_accuracy = correct.cumsum(0) / torch.arange(1, count + 1)
    risk = 1.0 - cumulative_accuracy
    return {
        "aurc": float(risk.mean()),
        "curve": [
            {
                "coverage": (index + 1) / count,
                "risk": float(risk[index]),
                "accuracy": float(cumulative_accuracy[index]),
                "threshold": float(scores[index]),
            }
            for index in range(count)
        ],
    }


def evaluate_reject_guard(
    scores: Tensor,
    predicted_relations: Sequence[RelationId | str | None],
    target_relations: Sequence[RelationId | str | None],
    reject_groups: Sequence[str],
    reject_families: Sequence[str],
    *,
    threshold: float,
    calibration_confidence: Tensor,
    known_coverage_target: float = 0.95,
    num_ece_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate a frozen reject guard, rejecting every invalid input by default."""

    values = scores.detach().cpu().float().flatten()
    confidence = calibration_confidence.detach().cpu().float().flatten()
    size = len(values)
    if not size or not (
        size
        == len(confidence)
        == len(predicted_relations)
        == len(target_relations)
        == len(reject_groups)
        == len(reject_families)
    ):
        raise ValueError("reject metric inputs must be non-empty and aligned")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("reject threshold must be finite")
    groups = [str(value) for value in reject_groups]
    invalid_groups = sorted(set(groups).difference(_REJECT_GROUPS))
    if invalid_groups:
        raise ValueError(f"unknown reject groups {invalid_groups}")
    targets = [_optional_relation(value) for value in target_relations]
    predictions = [_optional_relation(value) for value in predicted_relations]
    known = torch.tensor([target is not None for target in targets], dtype=torch.bool)
    if not bool(known.any()) or not bool((~known).any()):
        raise ValueError("reject metrics require known and open-set examples")
    for index, (is_known, group) in enumerate(zip(known.tolist(), groups)):
        expected = "known" if is_known else group
        if (is_known and group != "known") or (not is_known and group == "known"):
            raise ValueError(f"reject group/target mismatch at row {index}: {expected}")

    valid_prediction = torch.tensor(
        [prediction is not None for prediction in predictions], dtype=torch.bool
    )
    valid_confidence = torch.isfinite(confidence) & confidence.ge(0) & confidence.le(1)
    safe_scores, finite_route = _safe_detection_scores(
        values, valid_prediction & valid_confidence
    )
    accepted = finite_route & safe_scores.ge(threshold)
    invalid = ~finite_route
    correct = torch.tensor(
        [
            prediction is not None and target is not None and prediction is target
            for prediction, target in zip(predictions, targets)
        ],
        dtype=torch.bool,
    )
    known_correct = correct[known]
    known_confidence = torch.where(
        valid_confidence[known], confidence[known], torch.zeros_like(confidence[known])
    )
    curve = _coverage_risk_curve(safe_scores[known], known_correct & finite_route[known])

    # The supplied threshold is frozen on public_calibration.  Evaluation never
    # derives a new threshold from development or locked-audit known rows.
    tnr = int((~accepted[~known]).sum()) / int((~known).sum())
    unknown = ~known
    unknown_count = int(unknown.sum())

    def false_accept(mask: Tensor) -> float:
        return int(accepted[mask].sum()) / int(mask.sum()) if bool(mask.any()) else 0.0

    ambiguous = torch.tensor([value == "ambiguous" for value in groups])
    unrelated = torch.tensor([value == "unrelated" for value in groups])
    per_family: dict[str, float] = {}
    per_family_counts: dict[str, int] = {}
    for family in sorted({str(reject_families[i]) for i in range(size) if unknown[i]}):
        mask = torch.tensor(
            [unknown[i] and str(reject_families[i]) == family for i in range(size)]
        )
        per_family[family] = false_accept(mask)
        per_family_counts[family] = int(mask.sum())

    accepted_known = accepted & known
    achieved_known_coverage = int(accepted[known].sum()) / int(known.sum())
    known_accepted_accuracy = (
        int(correct[accepted_known].sum()) / int(accepted_known.sum())
        if bool(accepted_known.any())
        else 0.0
    )
    invalid_count = int(invalid.sum())
    invalid_rejected = int((invalid & ~accepted).sum())
    by_open_set_kind: dict[str, Any] = {}
    for name, open_mask in (
        ("ambiguous", ambiguous),
        ("unrelated", unrelated),
        ("ambiguous_unrelated_overall", unknown),
    ):
        comparison = known | open_mask
        comparison_scores = safe_scores[comparison]
        comparison_known = known[comparison]
        by_open_set_kind[name] = {
            "num_known": int(known.sum()),
            "num_open_set": int(open_mask.sum()),
            "known_detection_auroc": binary_auroc(
                comparison_scores, comparison_known
            ),
            "known_detection_aupr": binary_average_precision(
                comparison_scores, comparison_known
            ),
            "true_negative_rate_at_frozen_calibration_threshold": float(
                (~accepted[open_mask]).float().mean()
            ),
            "false_accept_rate_at_frozen_calibration_threshold": false_accept(
                open_mask
            ),
        }
    by_open_set_kind["known"] = {
        "num_rows": int(known.sum()),
        "coverage_at_frozen_calibration_threshold": int(accepted[known].sum())
        / int(known.sum()),
        "relation_accuracy": int(known_correct.sum()) / len(known_correct),
        "accepted_relation_accuracy": known_accepted_accuracy,
    }
    return {
        "known_detection_auroc": binary_auroc(safe_scores, known),
        "known_detection_aupr": binary_average_precision(safe_scores, known),
        "known_ece": expected_calibration_error(
            known_confidence, known_correct, num_bins=num_ece_bins
        ),
        "coverage_risk_curve": curve["curve"],
        "coverage_aurc": curve["aurc"],
        "tnr_at_95_known_coverage": tnr,
        "tnr_at_target_known_coverage": tnr,
        "tnr_at_frozen_calibration_threshold": tnr,
        "tnr_at_95_definition": (
            "threshold selected for 95% known coverage on public_calibration; "
            "never recomputed on the evaluation split"
        ),
        "known_coverage_target": float(known_coverage_target),
        "target_coverage_attainable": achieved_known_coverage
        >= float(known_coverage_target),
        "false_accept_rate": false_accept(unknown),
        "ambiguous_false_accept_rate": false_accept(ambiguous),
        "unrelated_false_accept_rate": false_accept(unrelated),
        "worst_reject_family_false_accept_rate": max(per_family.values(), default=0.0),
        "per_reject_family_false_accept_rate": per_family,
        "per_reject_family_count": per_family_counts,
        "known_coverage": achieved_known_coverage,
        "known_accepted_accuracy": known_accepted_accuracy,
        "known_relation_accuracy": int(known_correct.sum()) / len(known_correct),
        "reject_threshold": threshold,
        "threshold_source": "public_calibration_frozen",
        "num_known": int(known.sum()),
        "num_ambiguous": int(ambiguous.sum()),
        "num_unrelated": int(unrelated.sum()),
        "num_open_set": unknown_count,
        "num_invalid_scores_or_routes": invalid_count,
        "invalid_score_rejection_rate": (
            invalid_rejected / invalid_count if invalid_count else 1.0
        ),
        "fail_closed_rate": invalid_rejected / invalid_count if invalid_count else 1.0,
        "nan_or_invalid_score_behavior": "reject",
        "by_open_set_kind": by_open_set_kind,
    }


def _known_targets(
    values: Sequence[RelationId | str | None],
) -> tuple[Tensor, list[RelationId | None]]:
    targets = [_optional_relation(value) for value in values]
    return torch.tensor([value is not None for value in targets]), targets


def _temperature_ece(
    confidence: Tensor,
    predicted: Sequence[RelationId | None],
    targets: Sequence[RelationId | None],
    known: Tensor,
    *,
    num_bins: int,
) -> float:
    values = confidence.detach().cpu().float().flatten()[known]
    valid = torch.isfinite(values) & values.ge(0) & values.le(1)
    values = torch.where(valid, values, torch.zeros_like(values))
    correctness = torch.tensor(
        [
            prediction is not None and target is not None and prediction is target
            for prediction, target, selected in zip(predicted, targets, known.tolist())
            if selected
        ],
        dtype=torch.bool,
    )
    return expected_calibration_error(values, correctness, num_bins=num_bins)


def _research_gate_count(metrics: Mapping[str, Any]) -> tuple[int, dict[str, bool]]:
    outcomes = {}
    for name, (direction, threshold) in _RESEARCH_GATES.items():
        value = float(metrics[name])
        outcomes[name] = value >= threshold if direction == ">=" else value <= threshold
    return sum(outcomes.values()), outcomes


def select_calibrated_reject_guard(
    calibration_head_logits: Tensor,
    calibration_definition_logits: Tensor,
    calibration_target_relations: Sequence[RelationId | str | None],
    calibration_reject_groups: Sequence[str],
    calibration_reject_families: Sequence[str],
    *,
    head_class_order: Sequence[RelationId | str],
    definition_class_order: Sequence[RelationId | str],
    temperature_grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    known_coverage_target: float = 0.95,
    num_ece_bins: int = 10,
) -> dict[str, Any]:
    """Freeze score, temperature, and threshold using calibration rows only.

    The function has deliberately no development or locked-audit argument.  For
    every score, temperature minimizes known-row ECE only.  Threshold is then the
    score quantile needed for target known coverage.  Finally, candidates are
    ranked by a preregistered deterministic lexicographic objective.
    """

    forbidden = ("development", "locked", "audit")
    if any(
        token in name.lower()
        for name in inspect.signature(select_calibrated_reject_guard).parameters
        for token in forbidden
    ):
        raise RuntimeError("selection API accidentally exposed a post-calibration input")
    temperatures = sorted({_temperature(value) for value in temperature_grid})
    if not temperatures:
        raise ValueError("temperature grid cannot be empty")
    known, targets = _known_targets(calibration_target_relations)
    if not bool(known.any()) or not bool((~known).any()):
        raise ValueError("calibration requires known and open-set rows")
    if not (
        len(calibration_head_logits)
        == len(calibration_definition_logits)
        == len(targets)
        == len(calibration_reject_groups)
        == len(calibration_reject_families)
    ):
        raise ValueError("calibration inputs must be row-aligned")

    temperature_sweeps: dict[str, list[dict[str, float]]] = {
        name: [] for name in _SCORE_NAMES
    }
    selected_temperature: dict[str, float] = {}
    for score_name in _SCORE_NAMES:
        ranked = []
        for temperature in temperatures:
            extracted = extract_reject_score_candidates(
                calibration_head_logits,
                calibration_definition_logits,
                head_class_order=head_class_order,
                definition_class_order=definition_class_order,
                temperature=temperature,
            )
            ece = _temperature_ece(
                extracted["calibration_confidence"][score_name],
                extracted["predicted_relations"],
                targets,
                known,
                num_bins=num_ece_bins,
            )
            temperature_sweeps[score_name].append(
                {"temperature": temperature, "known_ece": ece}
            )
            # No open-set score participates in temperature selection.
            ranked.append((ece, abs(math.log(temperature)), temperature))
        selected_temperature[score_name] = min(ranked)[2]

    candidate_results: dict[str, dict[str, Any]] = {}
    ranked_candidates = []
    for preference, score_name in enumerate(_SCORE_NAMES):
        temperature = selected_temperature[score_name]
        extracted = extract_reject_score_candidates(
            calibration_head_logits,
            calibration_definition_logits,
            head_class_order=head_class_order,
            definition_class_order=definition_class_order,
            temperature=temperature,
        )
        scores = extracted["scores"][score_name]
        valid_prediction = torch.tensor(
            [value is not None for value in extracted["predicted_relations"]]
        )
        valid_confidence = torch.isfinite(
            extracted["calibration_confidence"][score_name]
        )
        safe, valid = _safe_detection_scores(
            scores, valid_prediction & valid_confidence
        )
        threshold, _, attainable = _known_coverage_threshold(
            safe, known, valid, known_coverage_target
        )
        metrics = evaluate_reject_guard(
            scores,
            extracted["predicted_relations"],
            calibration_target_relations,
            calibration_reject_groups,
            calibration_reject_families,
            threshold=threshold,
            calibration_confidence=extracted["calibration_confidence"][score_name],
            known_coverage_target=known_coverage_target,
            num_ece_bins=num_ece_bins,
        )
        gate_count, gates = _research_gate_count(metrics)
        # Preregistered descending objective: attain coverage, pass more research
        # gates, reduce worst/overall false accepts, improve ranking, improve ECE,
        # then use fixed G1/G2/G3 order for exact ties.
        objective = (
            int(attainable and metrics["known_coverage"] >= known_coverage_target),
            gate_count,
            -float(metrics["worst_reject_family_false_accept_rate"]),
            -float(metrics["false_accept_rate"]),
            float(metrics["known_detection_auroc"]),
            float(metrics["known_detection_aupr"]),
            -float(metrics["known_ece"]),
            -preference,
        )
        candidate_results[score_name] = {
            "score_name": score_name,
            "temperature": temperature,
            "threshold": threshold,
            "target_coverage_attainable": attainable,
            "research_gate_count": gate_count,
            "research_gates": gates,
            "selection_objective": list(objective),
            "calibration_metrics": metrics,
            "temperature_sweep": temperature_sweeps[score_name],
        }
        ranked_candidates.append((objective, score_name))
    selected_name = max(ranked_candidates)[1]
    selected = candidate_results[selected_name]
    return {
        "selected_score": selected_name,
        "selected_temperature": selected["temperature"],
        "selected_threshold": selected["threshold"],
        "selected_candidate": selected,
        "candidates": candidate_results,
        "selection_objective_order": [
            "known_coverage_target_met",
            "research_gate_count",
            "lower_worst_reject_family_false_accept_rate",
            "lower_overall_false_accept_rate",
            "higher_known_detection_auroc",
            "higher_known_detection_aupr",
            "lower_known_ece",
            "fixed_G1_G2_G3_tie_break",
        ],
        "selection_provenance": {
            "selection_split": "public_calibration",
            "score_selection_data": "public_calibration_only",
            "temperature_selection_data": "public_calibration_known_rows_only",
            "temperature_selection_objective": "minimum_known_ece_only",
            "threshold_selection_data": "public_calibration_known_scores_only",
            "threshold_known_coverage_target": float(known_coverage_target),
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
        },
    }
