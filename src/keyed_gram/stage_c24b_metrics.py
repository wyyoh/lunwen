"""Stage C2.4b 选择性离散路由的纯指标函数。

本模块不执行 I/O，也不接收原始问题、private answer 或 confirmation 数据。
所有比率在分母为空时固定返回 ``0.0``，同时通过 ``denominators`` 和
``empty_denominator_policy`` 明确披露该情况。所有公开返回值均经过严格的
JSON 有限数检查。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .stage_c24_contract import RelationId


SAMPLE_TYPES = frozenset({"known", "ambiguous", "unrelated"})
REJECT_REASONS = frozenset({"unknown", "ambiguous", "invalid"})
EMPTY_DENOMINATOR_POLICY = "return_0.0_and_report_denominator"

_FORBIDDEN_JSON_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_value",
        "candidate_answer",
        "candidate_answers",
        "private_answer",
        "private_answers",
        "private_value",
        "private_values",
        "confirmation_data",
        "confirmation_path",
        "confirmation_rows",
    }
)
_ALLOWED_BOOLEAN_CONFIRMATION_STATUS = "ready_to_create_new_confirmation_pool"
_FIXED_FALSE_PRIVATE_STATUS = frozenset(
    {
        "contains_private_answer",
        "contains_private_answers",
        "create_private_answer",
        "load_private_answers",
        "private_answers_loaded",
        "private_value_memory_trained",
        "train_private_memory",
    }
)


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def assert_finite_json(value: Any, *, location: str = "root") -> None:
    """验证值可由 ``json.dumps(..., allow_nan=False)`` 安全序列化。

    同时拒绝 answer-bearing 字段和 confirmation 数据字段。协议产物可以保存
    “未执行”审计位：private/confirmation 字段只有在白名单状态或值严格为
    ``False`` 时才允许；这不会为对应数据留下序列化通道。
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            private_status = key in _FIXED_FALSE_PRIVATE_STATUS and child is False
            answer_free_metadata = "answer_free" in key
            if key in _FORBIDDEN_JSON_KEYS and not private_status:
                raise ValueError(f"answer-bearing field is forbidden at {location}.{raw_key}")
            if "private_answer" in key and not (
                private_status or answer_free_metadata
            ):
                raise ValueError(f"answer-bearing field is forbidden at {location}.{raw_key}")
            if "confirmation" in key:
                fixed_false = child is False
                readiness_flag = (
                    key == _ALLOWED_BOOLEAN_CONFIRMATION_STATUS
                    and isinstance(child, bool)
                )
                if not (fixed_false or readiness_flag):
                    raise ValueError(
                        f"confirmation data/status is forbidden at {location}.{raw_key}"
                    )
            assert_finite_json(child, location=f"{location}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_json(child, location=f"{location}[{index}]")
        return
    if isinstance(value, Tensor):
        raise TypeError(f"tensor is not JSON output at {location}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {location}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise TypeError(f"unsupported JSON value at {location}: {type(value).__name__}")


def _finish(value: dict[str, Any]) -> dict[str, Any]:
    value.setdefault("empty_denominator_policy", EMPTY_DENOMINATOR_POLICY)
    assert_finite_json(value)
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _rate(numerator: int, denominator: int) -> float:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("invalid metric count")
    return numerator / denominator if denominator else 0.0


def _relation(value: RelationId | str | None, *, allow_none: bool) -> RelationId | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("known target relation cannot be empty")
    if isinstance(value, RelationId):
        return value
    try:
        return RelationId(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown relation {value!r}") from exc


def _sample_types(values: Sequence[str]) -> list[str]:
    output = [str(value) for value in values]
    invalid = sorted(set(output).difference(SAMPLE_TYPES))
    if invalid:
        raise ValueError(f"unknown sample types {invalid}")
    return output


def _families(values: Sequence[str]) -> list[str]:
    output = []
    for value in values:
        family = str(value).strip()
        normalized = _normalized_key(family)
        if not family:
            raise ValueError("family IDs cannot be empty")
        if "private_answer" in normalized or "confirmation" in normalized:
            raise ValueError("private-answer/confirmation family input is forbidden")
        output.append(family)
    return output


def _accepted(values: Sequence[bool]) -> list[bool]:
    if any(not isinstance(value, bool) for value in values):
        raise ValueError("route acceptance values must be booleans")
    return list(values)


def _aligned(size: int, **values: Sequence[object]) -> None:
    if size <= 0:
        raise ValueError("metric inputs must be non-empty")
    mismatched = [name for name, sequence in values.items() if len(sequence) != size]
    if mismatched:
        raise ValueError(f"metric inputs are not aligned: {mismatched}")


def _agreement(
    labels: Sequence[str], outer_groups: Sequence[str], inner_groups: Sequence[str]
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
                equal += labels[left] == labels[right]
        per_group[group] = _rate(equal, pairs) if pairs else 1.0
        total_equal += equal
        total_pairs += pairs
    return {
        "macro_agreement": sum(per_group.values()) / len(per_group),
        "pair_weighted_agreement": _rate(total_equal, total_pairs)
        if total_pairs
        else 1.0,
        "num_cross_unit_pairs": total_pairs,
        "per_group_agreement": per_group,
    }


def known_routing_metrics(
    predicted_relations: Sequence[RelationId | str | None],
    target_relations: Sequence[RelationId | str],
    accepted: Sequence[bool],
    families: Sequence[str],
    frames: Sequence[str],
    *,
    hard_negative_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """计算 known query 的闭集、选择性和专项混淆指标。

    被拒绝的 known query 在 micro/family accuracy 中记为错误；这样该指标
    与访问控制语义一致，而 ``accepted_route_accuracy`` 单独度量接受后的精度。
    """

    size = len(target_relations)
    _aligned(
        size,
        predicted_relations=predicted_relations,
        accepted=accepted,
        families=families,
        frames=frames,
    )
    is_accepted = _accepted(accepted)
    target = [_relation(value, allow_none=False) for value in target_relations]
    predicted = [_relation(value, allow_none=True) for value in predicted_relations]
    family_values = _families(families)
    frame_values = [str(value) for value in frames]
    if any(not value for value in frame_values):
        raise ValueError("frame IDs cannot be empty")
    for index, (route_accepted, prediction) in enumerate(zip(is_accepted, predicted)):
        if route_accepted and prediction is None:
            raise ValueError(f"accepted row {index} has no discrete relation")
        if not route_accepted and prediction is not None:
            raise ValueError(f"rejected row {index} exposes a routed relation")

    correct = [
        route_accepted and prediction is truth
        for route_accepted, prediction, truth in zip(is_accepted, predicted, target)
    ]
    family_indices: dict[str, list[int]] = defaultdict(list)
    relation_indices: dict[RelationId, list[int]] = defaultdict(list)
    for index, (family, truth) in enumerate(zip(family_values, target)):
        assert truth is not None
        family_indices[family].append(index)
        relation_indices[truth].append(index)
    per_family = {
        family: _rate(sum(correct[index] for index in indices), len(indices))
        for family, indices in sorted(family_indices.items())
    }
    per_relation = {
        relation.value: _rate(
            sum(correct[index] for index in relation_indices[relation]),
            len(relation_indices[relation]),
        )
        for relation in RelationId
    }
    order = tuple(RelationId)
    relation_lookup = {relation: index for index, relation in enumerate(order)}
    confusion = [[0 for _ in range(len(order) + 1)] for _ in order]
    for prediction, truth in zip(predicted, target):
        assert truth is not None
        column = len(order) if prediction is None else relation_lookup[prediction]
        confusion[relation_lookup[truth]][column] += 1

    accepted_count = sum(is_accepted)
    correct_count = sum(correct)

    def directed_error(source: RelationId, destination: RelationId) -> float:
        indices = relation_indices[source]
        errors = sum(predicted[index] is destination for index in indices)
        return _rate(errors, len(indices))

    city_indices = relation_indices[RelationId.CITY_CODE]
    city_other = sum(
        predicted[index] is not None
        and predicted[index] is not RelationId.CITY_CODE
        for index in city_indices
    )
    route_labels = [
        prediction.value if prediction is not None else "reject"
        for prediction in predicted
    ]
    hard_mask = (
        [False] * size if hard_negative_mask is None else _accepted(hard_negative_mask)
    )
    if len(hard_mask) != size:
        raise ValueError("hard-negative mask is not aligned")
    hard_count = sum(hard_mask)
    hard_correct = sum(value and correct[index] for index, value in enumerate(hard_mask))
    hard_rejected = sum(
        value and not is_accepted[index] for index, value in enumerate(hard_mask)
    )
    return _finish(
        {
            "relation_micro_accuracy": _rate(correct_count, size),
            "relation_family_macro_accuracy": sum(per_family.values()) / len(per_family),
            "worst_family_accuracy": min(per_family.values()),
            "per_family_accuracy": per_family,
            "per_relation_accuracy": per_relation,
            "confusion_matrix": {
                "class_order": [relation.value for relation in order] + ["reject"],
                "rows": "target",
                "columns": "accepted_prediction_or_reject",
                "matrix": confusion,
            },
            "access_to_registry_error_rate": directed_error(
                RelationId.ACCESS_CODE, RelationId.REGISTRY_ID
            ),
            "registry_to_access_error_rate": directed_error(
                RelationId.REGISTRY_ID, RelationId.ACCESS_CODE
            ),
            "city_to_other_error_rate": _rate(city_other, len(city_indices)),
            "accepted_route_accuracy": _rate(correct_count, accepted_count),
            "accepted_relation_precision": _rate(correct_count, accepted_count),
            "known_coverage": _rate(accepted_count, size),
            "abstention_rate": _rate(size - accepted_count, size),
            "hard_negative_accuracy": _rate(hard_correct, hard_count),
            "hard_negative_rejection_rate": _rate(hard_rejected, hard_count),
            "same_family_cross_frame_route_agreement": _agreement(
                route_labels, family_values, frame_values
            ),
            "same_relation_cross_family_route_agreement": _agreement(
                route_labels,
                [truth.value for truth in target if truth is not None],
                family_values,
            ),
            "counts": {
                "known": size,
                "accepted_known": accepted_count,
                "correct_accepted_known": correct_count,
                "families": len(per_family),
                "hard_negative": hard_count,
                "per_relation": {
                    relation.value: len(relation_indices[relation])
                    for relation in RelationId
                },
            },
            "denominators": {
                "relation_micro_accuracy": size,
                "accepted_route_accuracy": accepted_count,
                "hard_negative_accuracy": hard_count,
                "access_to_registry_error_rate": len(
                    relation_indices[RelationId.ACCESS_CODE]
                ),
                "registry_to_access_error_rate": len(
                    relation_indices[RelationId.REGISTRY_ID]
                ),
                "city_to_other_error_rate": len(city_indices),
            },
        }
    )


def set_valued_routing_metrics(
    candidate_sets: Sequence[Sequence[RelationId | str]],
    target_relations: Sequence[RelationId | str | None],
    sample_types: Sequence[str],
) -> dict[str, Any]:
    """度量 0/1/多候选集合及 known true-relation inclusion。"""

    size = len(candidate_sets)
    _aligned(size, target_relations=target_relations, sample_types=sample_types)
    kinds = _sample_types(sample_types)
    targets = [_relation(value, allow_none=True) for value in target_relations]
    canonical_sets: list[frozenset[RelationId]] = []
    for index, raw_values in enumerate(candidate_sets):
        try:
            values = frozenset(
                _relation(value, allow_none=False) for value in raw_values
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate set at row {index}") from exc
        canonical_sets.append(frozenset(value for value in values if value is not None))
    for index, (kind, target) in enumerate(zip(kinds, targets)):
        if (kind == "known") != (target is not None):
            raise ValueError(f"sample type/target mismatch at row {index}")

    sizes = [len(values) for values in canonical_sets]
    empty = [value == 0 for value in sizes]
    singleton = [value == 1 for value in sizes]
    multi = [value > 1 for value in sizes]
    known_indices = [index for index, kind in enumerate(kinds) if kind == "known"]
    ambiguous_indices = [
        index for index, kind in enumerate(kinds) if kind == "ambiguous"
    ]
    unrelated_indices = [
        index for index, kind in enumerate(kinds) if kind == "unrelated"
    ]
    included = [
        targets[index] in canonical_sets[index] for index in known_indices
    ]
    singleton_correct = [
        singleton[index] and targets[index] in canonical_sets[index]
        for index in known_indices
    ]
    return _finish(
        {
            "average_candidate_set_size": sum(sizes) / size,
            "singleton_rate": _rate(sum(singleton), size),
            "empty_set_rate": _rate(sum(empty), size),
            "multi_label_set_rate": _rate(sum(multi), size),
            "true_relation_inclusion_rate": _rate(sum(included), len(known_indices)),
            "singleton_correctness": _rate(
                sum(singleton_correct),
                sum(singleton[index] for index in known_indices),
            ),
            "ambiguous_set_rate": _rate(
                sum(multi[index] for index in ambiguous_indices),
                len(ambiguous_indices),
            ),
            "unknown_set_rate": _rate(
                sum(empty[index] for index in unrelated_indices),
                len(unrelated_indices),
            ),
            "set_coverage": _rate(sum(included), len(known_indices)),
            "set_inefficiency": sum(sizes[index] for index in known_indices)
            / len(known_indices)
            if known_indices
            else 0.0,
            "known_singleton_rate": _rate(
                sum(singleton[index] for index in known_indices), len(known_indices)
            ),
            "known_empty_set_rate": _rate(
                sum(empty[index] for index in known_indices), len(known_indices)
            ),
            "known_multi_relation_set_rate": _rate(
                sum(multi[index] for index in known_indices), len(known_indices)
            ),
            "counts": {
                "rows": size,
                "known": len(known_indices),
                "ambiguous": len(ambiguous_indices),
                "unrelated": len(unrelated_indices),
                "singleton": sum(singleton),
                "empty": sum(empty),
                "multi": sum(multi),
            },
            "denominators": {
                "known": len(known_indices),
                "known_singletons": sum(singleton[index] for index in known_indices),
                "ambiguous": len(ambiguous_indices),
                "unrelated": len(unrelated_indices),
            },
        }
    )


def _binary_auroc(scores: Sequence[float], positive: Sequence[bool]) -> tuple[float, bool]:
    positives = [score for score, label in zip(scores, positive) if label]
    negatives = [score for score, label in zip(scores, positive) if not label]
    if not positives or not negatives:
        return 0.0, False
    value = sum(
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positives
        for right in negatives
    ) / (len(positives) * len(negatives))
    return value, True


def _binary_aupr(scores: Sequence[float], positive: Sequence[bool]) -> tuple[float, bool]:
    positive_count = sum(positive)
    if positive_count == 0 or positive_count == len(positive):
        return 0.0, False
    thresholds = sorted(set(scores), reverse=True)
    previous_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        selected = [score >= threshold for score in scores]
        true_positive = sum(flag and label for flag, label in zip(selected, positive))
        false_positive = sum(flag and not label for flag, label in zip(selected, positive))
        recall = true_positive / positive_count
        precision = true_positive / max(true_positive + false_positive, 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area, True


def risk_coverage_curve(
    accept_scores: Sequence[float], safe_accept: Sequence[bool]
) -> dict[str, Any]:
    """按 accept score 递减计算安全访问的 risk–coverage 与 AURC。"""

    size = len(accept_scores)
    _aligned(size, safe_accept=safe_accept)
    scores = [_finite(value, label="accept score") for value in accept_scores]
    if any(not isinstance(value, bool) for value in safe_accept):
        raise ValueError("safe_accept values must be booleans")
    order = sorted(range(size), key=lambda index: (-scores[index], index))
    correct = 0
    curve = []
    for rank, index in enumerate(order, start=1):
        correct += bool(safe_accept[index])
        accuracy = correct / rank
        curve.append(
            {
                "coverage": rank / size,
                "risk": 1.0 - accuracy,
                "accuracy": accuracy,
                "threshold": scores[index],
            }
        )
    return _finish(
        {
            "aurc": sum(point["risk"] for point in curve) / size,
            "curve": curve,
            "counts": {"rows": size},
        }
    )


def reject_quality_metrics(
    accepted: Sequence[bool],
    reject_reasons: Sequence[str | None],
    predicted_relations: Sequence[RelationId | str | None],
    target_relations: Sequence[RelationId | str | None],
    sample_types: Sequence[str],
    families: Sequence[str],
    reject_scores: Sequence[float],
) -> dict[str, Any]:
    """分别评估 ambiguous、unrelated 和 combined reject。

    ``reject_scores`` 越大表示越应拒绝；risk–coverage 使用其相反数作为
    accept score。实际 FAR/错误分解来自已冻结的 route 决策。
    """

    size = len(sample_types)
    _aligned(
        size,
        accepted=accepted,
        reject_reasons=reject_reasons,
        predicted_relations=predicted_relations,
        target_relations=target_relations,
        families=families,
        reject_scores=reject_scores,
    )
    is_accepted = _accepted(accepted)
    kinds = _sample_types(sample_types)
    family_values = _families(families)
    scores = [_finite(value, label="reject score") for value in reject_scores]
    targets = [_relation(value, allow_none=True) for value in target_relations]
    predictions = [_relation(value, allow_none=True) for value in predicted_relations]
    reasons: list[str | None] = []
    for index, (route_accepted, raw_reason, prediction) in enumerate(
        zip(is_accepted, reject_reasons, predictions)
    ):
        reason = None if raw_reason is None else str(raw_reason)
        if route_accepted:
            if reason is not None or prediction is None:
                raise ValueError(f"accepted route schema mismatch at row {index}")
        else:
            if reason not in REJECT_REASONS or prediction is not None:
                raise ValueError(f"rejected route schema mismatch at row {index}")
        reasons.append(reason)
    for index, (kind, target) in enumerate(zip(kinds, targets)):
        if (kind == "known") != (target is not None):
            raise ValueError(f"sample type/target mismatch at row {index}")

    masks = {
        name: [kind == name for kind in kinds]
        for name in ("known", "ambiguous", "unrelated")
    }
    masks["combined_reject"] = [kind != "known" for kind in kinds]

    def group_metrics(name: str) -> dict[str, Any]:
        open_mask = masks[name]
        comparison = [known or opened for known, opened in zip(masks["known"], open_mask)]
        selected_scores = [score for score, keep in zip(scores, comparison) if keep]
        positive = [opened for opened, keep in zip(open_mask, comparison) if keep]
        auroc, auroc_defined = _binary_auroc(selected_scores, positive)
        aupr, aupr_defined = _binary_aupr(selected_scores, positive)
        indices = [index for index, value in enumerate(open_mask) if value]
        accepts = sum(is_accepted[index] for index in indices)
        return {
            "count": len(indices),
            "rejection_rate": _rate(len(indices) - accepts, len(indices)),
            "false_accept_rate": _rate(accepts, len(indices)),
            "accepted_error_rate": 1.0 if accepts else 0.0,
            "accepted_count": accepts,
            "auroc": auroc,
            "auroc_defined": auroc_defined,
            "aupr": aupr,
            "aupr_defined": aupr_defined,
        }

    grouped = {
        "ambiguous": group_metrics("ambiguous"),
        "unrelated": group_metrics("unrelated"),
        "combined": group_metrics("combined_reject"),
    }
    known_indices = [index for index, value in enumerate(masks["known"]) if value]
    known_accepted = [index for index in known_indices if is_accepted[index]]
    known_accepted_errors = sum(
        predictions[index] is not targets[index] for index in known_accepted
    )
    grouped["known"] = {
        "count": len(known_indices),
        "coverage": _rate(len(known_accepted), len(known_indices)),
        "accepted_error_rate": _rate(known_accepted_errors, len(known_accepted)),
        "accepted_count": len(known_accepted),
    }
    open_indices = [index for index, kind in enumerate(kinds) if kind != "known"]
    per_family_far: dict[str, float] = {}
    per_family_count: dict[str, int] = {}
    for family in sorted({family_values[index] for index in open_indices}):
        indices = [index for index in open_indices if family_values[index] == family]
        per_family_far[family] = _rate(
            sum(is_accepted[index] for index in indices), len(indices)
        )
        per_family_count[family] = len(indices)

    safe = [
        kind == "known" and prediction is target
        for kind, prediction, target in zip(kinds, predictions, targets)
    ]
    risk = risk_coverage_curve([-score for score in scores], safe)
    ambiguous_indices = [index for index, kind in enumerate(kinds) if kind == "ambiguous"]
    unrelated_indices = [index for index, kind in enumerate(kinds) if kind == "unrelated"]
    invalid_indices = [index for index, reason in enumerate(reasons) if reason == "invalid"]
    return _finish(
        {
            "by_group": grouped,
            "ambiguous_rejection_rate": grouped["ambiguous"]["rejection_rate"],
            "unrelated_rejection_rate": grouped["unrelated"]["rejection_rate"],
            "combined_rejection_rate": grouped["combined"]["rejection_rate"],
            "ambiguous_false_accept_rate": grouped["ambiguous"]["false_accept_rate"],
            "unrelated_false_accept_rate": grouped["unrelated"]["false_accept_rate"],
            "combined_false_accept_rate": grouped["combined"]["false_accept_rate"],
            "worst_reject_family_false_accept_rate": max(
                per_family_far.values(), default=0.0
            ),
            "per_reject_family_false_accept_rate": per_family_far,
            "per_reject_family_count": per_family_count,
            "unknown_to_accept_rate": grouped["unrelated"]["false_accept_rate"],
            "ambiguous_to_accept_rate": grouped["ambiguous"]["false_accept_rate"],
            "ambiguous_to_unknown_rate": _rate(
                sum(reasons[index] == "unknown" for index in ambiguous_indices),
                len(ambiguous_indices),
            ),
            "unrelated_to_ambiguous_rate": _rate(
                sum(reasons[index] == "ambiguous" for index in unrelated_indices),
                len(unrelated_indices),
            ),
            "fail_closed_rate": _rate(
                sum(not is_accepted[index] for index in invalid_indices),
                len(invalid_indices),
            )
            if invalid_indices
            else 1.0,
            "risk_coverage": risk,
            "coverage_aurc": risk["aurc"],
            "counts": {
                "rows": size,
                "known": len(known_indices),
                "ambiguous": len(ambiguous_indices),
                "unrelated": len(unrelated_indices),
                "invalid": len(invalid_indices),
            },
            "denominators": {
                "accepted_known": len(known_accepted),
                "ambiguous": len(ambiguous_indices),
                "unrelated": len(unrelated_indices),
                "combined_reject": len(open_indices),
                "invalid": len(invalid_indices),
            },
        }
    )


def access_control_metrics(
    accepted: Sequence[bool],
    predicted_relations: Sequence[RelationId | str | None],
    target_relations: Sequence[RelationId | str | None],
    sample_types: Sequence[str],
) -> dict[str, Any]:
    """计算 False Memory Access、Wrong Bucket Access 与 Safe Coverage。"""

    size = len(sample_types)
    _aligned(
        size,
        accepted=accepted,
        predicted_relations=predicted_relations,
        target_relations=target_relations,
    )
    is_accepted = _accepted(accepted)
    kinds = _sample_types(sample_types)
    predictions = [_relation(value, allow_none=True) for value in predicted_relations]
    targets = [_relation(value, allow_none=True) for value in target_relations]
    for index, (kind, route_accepted, prediction, target) in enumerate(
        zip(kinds, is_accepted, predictions, targets)
    ):
        if (kind == "known") != (target is not None):
            raise ValueError(f"sample type/target mismatch at row {index}")
        if route_accepted != (prediction is not None):
            raise ValueError(f"route status/prediction mismatch at row {index}")

    known = [index for index, kind in enumerate(kinds) if kind == "known"]
    ambiguous = [index for index, kind in enumerate(kinds) if kind == "ambiguous"]
    unrelated = [index for index, kind in enumerate(kinds) if kind == "unrelated"]
    opened = ambiguous + unrelated
    false_accesses = sum(is_accepted[index] for index in opened)
    accepted_total = sum(is_accepted)
    wrong_bucket = sum(
        is_accepted[index] and predictions[index] is not targets[index]
        for index in known
    )
    safe = sum(
        is_accepted[index] and predictions[index] is targets[index]
        for index in known
    )
    return _finish(
        {
            "false_memory_access_rate": _rate(false_accesses, len(opened)),
            "ambiguous_false_memory_access_rate": _rate(
                sum(is_accepted[index] for index in ambiguous), len(ambiguous)
            ),
            "unrelated_false_memory_access_rate": _rate(
                sum(is_accepted[index] for index in unrelated), len(unrelated)
            ),
            "wrong_bucket_access_rate": _rate(wrong_bucket, len(known)),
            "safe_coverage": _rate(safe, len(known)),
            # AcceptedRoute 始终代表一个 singleton bucket；因此该精度把
            # wrong-known 与 open-set false accepts 都纳入分母。
            "singleton_acceptance_precision": _rate(safe, accepted_total),
            "counts": {
                "known": len(known),
                "ambiguous": len(ambiguous),
                "unrelated": len(unrelated),
                "false_memory_access": false_accesses,
                "wrong_bucket_access": wrong_bucket,
                "safe_known_access": safe,
                "accepted_singleton": accepted_total,
            },
            "denominators": {
                "false_memory_access_rate": len(opened),
                "wrong_bucket_access_rate": len(known),
                "safe_coverage": len(known),
                "singleton_acceptance_precision": accepted_total,
            },
        }
    )


def fact_retrieval_metrics(
    accepted: Sequence[bool],
    sample_types: Sequence[str],
    fact_top1_correct: Sequence[bool | None],
    row_1nn_correct: Sequence[bool | None],
    reciprocal_ranks: Sequence[float | None],
    centroid_margins: Sequence[float | None],
    relation_bucket_candidate_counts: Sequence[int | None],
    cross_relation_candidate_counts: Sequence[int | None],
    *,
    oracle_fact_top1: float | None = None,
) -> dict[str, Any]:
    """聚合被接受 known query 的 answer-free fact/slot retrieval 指标。"""

    size = len(sample_types)
    _aligned(
        size,
        accepted=accepted,
        fact_top1_correct=fact_top1_correct,
        row_1nn_correct=row_1nn_correct,
        reciprocal_ranks=reciprocal_ranks,
        centroid_margins=centroid_margins,
        relation_bucket_candidate_counts=relation_bucket_candidate_counts,
        cross_relation_candidate_counts=cross_relation_candidate_counts,
    )
    is_accepted = _accepted(accepted)
    kinds = _sample_types(sample_types)
    known = [index for index, kind in enumerate(kinds) if kind == "known"]
    accepted_known = [index for index in known if is_accepted[index]]
    for index in range(size):
        values = (
            fact_top1_correct[index],
            row_1nn_correct[index],
            reciprocal_ranks[index],
            centroid_margins[index],
            relation_bucket_candidate_counts[index],
            cross_relation_candidate_counts[index],
        )
        needs_retrieval = index in accepted_known
        if needs_retrieval and any(value is None for value in values):
            raise ValueError(f"accepted known row {index} lacks retrieval metrics")
        if not needs_retrieval and any(value is not None for value in values):
            raise ValueError(f"non-retrieved row {index} exposes retrieval metrics")

    top1 = []
    row_hits = []
    ranks = []
    margins = []
    candidates = []
    cross_candidates = []
    for index in accepted_known:
        top_value = fact_top1_correct[index]
        row_value = row_1nn_correct[index]
        if not isinstance(top_value, bool) or not isinstance(row_value, bool):
            raise ValueError("retrieval correctness values must be booleans")
        top1.append(top_value)
        row_hits.append(row_value)
        rank = _finite(reciprocal_ranks[index], label="reciprocal rank")
        if not 0.0 <= rank <= 1.0:
            raise ValueError("reciprocal rank must lie in [0, 1]")
        ranks.append(rank)
        margins.append(_finite(centroid_margins[index], label="centroid margin"))
        candidate = relation_bucket_candidate_counts[index]
        cross = cross_relation_candidate_counts[index]
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
            raise ValueError("relation bucket candidate count must be positive")
        if isinstance(cross, bool) or not isinstance(cross, int) or cross < 0:
            raise ValueError("cross-relation candidate count must be non-negative")
        candidates.append(candidate)
        cross_candidates.append(cross)

    accepted_count = len(accepted_known)
    correct_count = sum(top1)
    all_query_top1 = _rate(correct_count, len(known))
    oracle_gap = None
    if oracle_fact_top1 is not None:
        oracle = _finite(oracle_fact_top1, label="oracle fact Top-1")
        if not 0.0 <= oracle <= 1.0:
            raise ValueError("oracle fact Top-1 must lie in [0, 1]")
        oracle_gap = oracle - all_query_top1
    result: dict[str, Any] = {
        "accepted_fact_top1": _rate(correct_count, accepted_count),
        "all_query_fact_top1": all_query_top1,
        "row_1nn": _rate(sum(row_hits), accepted_count),
        "mrr": sum(ranks) / accepted_count if accepted_count else 0.0,
        "mean_centroid_margin": sum(margins) / accepted_count
        if accepted_count
        else 0.0,
        "oracle_gap": oracle_gap,
        "relation_bucket_candidate_count": {
            "mean": sum(candidates) / accepted_count if accepted_count else 0.0,
            "min": min(candidates, default=0),
            "max": max(candidates, default=0),
        },
        "cross_relation_candidate_count": sum(cross_candidates),
        "cross_relation_candidate_count_max_per_query": max(
            cross_candidates, default=0
        ),
        "counts": {
            "known": len(known),
            "accepted_known": accepted_count,
            "correct_fact_top1": correct_count,
        },
        "denominators": {
            "accepted_fact_top1": accepted_count,
            "all_query_fact_top1": len(known),
            "row_1nn_mrr_margin": accepted_count,
        },
    }
    return _finish(result)


def compute_readiness(
    *,
    closed_set_selective_router_ready: bool,
    open_set_abstention_ready: bool,
    discrete_memory_contract_preserved: bool,
    public_benchmark_human_reviewed: bool,
) -> dict[str, bool]:
    """计算 C2.4b readiness；本阶段 confirmation 固定为 false。"""

    values = (
        closed_set_selective_router_ready,
        open_set_abstention_ready,
        discrete_memory_contract_preserved,
        public_benchmark_human_reviewed,
    )
    if any(not isinstance(value, bool) for value in values):
        raise ValueError("readiness inputs must be booleans")
    ready = all(values)
    result = {
        "closed_set_selective_router_ready": closed_set_selective_router_ready,
        "open_set_abstention_ready": open_set_abstention_ready,
        "discrete_memory_contract_preserved": discrete_memory_contract_preserved,
        "public_benchmark_human_reviewed": public_benchmark_human_reviewed,
        "new_confirmation_pool_created_after_freeze": False,
        "ready_to_create_new_confirmation_pool": ready,
        "c3_eligible": False,
    }
    assert_finite_json(result)
    return result


__all__ = [
    "EMPTY_DENOMINATOR_POLICY",
    "SAMPLE_TYPES",
    "access_control_metrics",
    "assert_finite_json",
    "compute_readiness",
    "fact_retrieval_metrics",
    "known_routing_metrics",
    "reject_quality_metrics",
    "risk_coverage_curve",
    "set_valued_routing_metrics",
]
