"""Stage C2.4c：冻结 v2.1 上的非独立探索性 R0–R4 评估。"""

from __future__ import annotations

import csv
import inspect
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .stage_c24_contract import (
    AcceptedRoute,
    RelationId,
    assert_discrete_memory_contract,
)
from .stage_c24b_calibration import (
    ClassConditionalConformalCalibrator,
    fit_class_conditional_conformal,
)
from .stage_c24b_metrics import (
    access_control_metrics,
    assert_finite_json,
    fact_retrieval_metrics,
    known_routing_metrics,
    reject_quality_metrics,
    set_valued_routing_metrics,
)
from .stage_c24b_router import (
    RejectedRoute,
    RouteResult,
    candidate_set_from_scores,
    execute_selective_route,
    route_from_candidate_set,
)
from .stage_c24c_models import (
    PublicModelContext,
    build_c23_public_rows,
    build_public_model_context,
    score_public_rows,
    verify_selected_head,
)
from .stage_c24c_protocol import (
    C24CProtocolError,
    EVALUATION_MODE,
    base_status,
    canonical_sha256,
    exact_file_inventory,
    git_state,
    load_config,
    load_json,
    mark_phase_completed,
    mark_phase_started,
    output_paths,
    read_declared_split,
    resolve_path,
    seal_benchmark,
    sha256_file,
    verify_benchmark_seal,
    write_json,
    write_protocol_incident_once,
)


VARIANTS = ("R0", "R1", "R2", "R3", "R4")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        raise C24CProtocolError(f"拒绝写入空 CSV：{path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
        handle.flush()
    temporary.replace(path)


def _relation_target(row: Mapping[str, Any]) -> RelationId | None:
    if str(row["sample_type"]) != "known":
        return None
    try:
        return RelationId(str(row["relation_id"]))
    except ValueError as exc:
        raise C24CProtocolError("known row 包含未知 RelationId") from exc


def _argmax_decisions(
    scores: Sequence[Mapping[RelationId, float]],
) -> tuple[list[RouteResult], list[frozenset[RelationId]]]:
    routes: list[RouteResult] = []
    sets: list[frozenset[RelationId]] = []
    for score in scores:
        if set(score) != set(RelationId) or any(
            not math.isfinite(float(value)) for value in score.values()
        ):
            routes.append(RejectedRoute("invalid"))
            sets.append(frozenset())
            continue
        relation = max(RelationId, key=lambda item: (float(score[item]), item.value))
        routes.append(AcceptedRoute(relation))
        sets.append(frozenset({relation}))
    return routes, sets


def _threshold_decisions(
    scores: Sequence[Mapping[RelationId, float]],
    thresholds: Mapping[RelationId | str, float],
    *,
    minimum_margin: float,
) -> tuple[list[RouteResult], list[frozenset[RelationId]]]:
    routes: list[RouteResult] = []
    sets: list[frozenset[RelationId]] = []
    canonical = {
        relation: float(
            thresholds[relation]
            if relation in thresholds
            else thresholds[relation.value]
        )
        for relation in RelationId
    }
    for score in scores:
        try:
            candidates = candidate_set_from_scores(score, canonical)
            if len(candidates) == 1:
                only = next(iter(candidates))
                margin = float(score[only]) - max(
                    float(value)
                    for relation, value in score.items()
                    if relation is not only
                )
            else:
                ordered = sorted((float(value) for value in score.values()), reverse=True)
                margin = ordered[0] - ordered[1]
            route = route_from_candidate_set(
                candidates,
                top1_top2_margin=margin,
                minimum_margin=float(minimum_margin),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            candidates = frozenset()
            route = RejectedRoute("invalid")
        routes.append(route)
        sets.append(candidates)
    return routes, sets


def _conformal_decisions(
    scores: Sequence[Mapping[RelationId, float]],
    calibrator: ClassConditionalConformalCalibrator,
) -> tuple[list[RouteResult], list[frozenset[RelationId]]]:
    routes, sets = [], []
    for score in scores:
        try:
            candidates = calibrator.candidate_set(score)
            route = calibrator.route(score)
        except (TypeError, ValueError, OverflowError):
            candidates = frozenset()
            route = RejectedRoute("invalid")
        routes.append(route)
        sets.append(candidates)
    return routes, sets


def _decision_columns(
    routes: Sequence[RouteResult],
) -> tuple[list[bool], list[RelationId | None], list[str | None]]:
    accepted, predicted, reasons = [], [], []
    for route in routes:
        if isinstance(route, AcceptedRoute):
            accepted.append(True)
            predicted.append(route.relation_id)
            reasons.append(None)
        elif isinstance(route, RejectedRoute):
            accepted.append(False)
            predicted.append(None)
            reasons.append(route.reason)
        else:
            raise C24CProtocolError("router 返回未知 route 类型")
    return accepted, predicted, reasons


def _reject_scores(
    scores: Sequence[Mapping[RelationId, float]],
    candidate_sets: Sequence[frozenset[RelationId]],
) -> list[float]:
    output = []
    for score, candidates in zip(scores, candidate_sets):
        ordered = sorted((float(value) for value in score.values()), reverse=True)
        unknown_signal = -ordered[0]
        ambiguity_signal = ordered[1] - ordered[0] + (
            1.0 if len(candidates) > 1 else 0.0
        )
        output.append(max(unknown_signal, ambiguity_signal))
    return output


def _compact_metrics(
    rows: Sequence[Mapping[str, Any]],
    routes: Sequence[RouteResult],
    candidate_sets: Sequence[frozenset[RelationId]],
) -> dict[str, float]:
    accepted, predicted, _ = _decision_columns(routes)
    targets = [_relation_target(row) for row in rows]
    known = [index for index, value in enumerate(targets) if value is not None]
    ambiguous = [
        index for index, row in enumerate(rows) if row["sample_type"] == "ambiguous"
    ]
    unrelated = [
        index for index, row in enumerate(rows) if row["sample_type"] == "unrelated"
    ]
    opened = ambiguous + unrelated
    correct = [
        accepted[index] and predicted[index] is targets[index] for index in known
    ]
    accepted_known = sum(accepted[index] for index in known)
    safe = sum(correct)
    wrong = sum(
        accepted[index] and predicted[index] is not targets[index] for index in known
    )
    accepted_total = sum(accepted)
    families: dict[str, list[int]] = defaultdict(list)
    for index in known:
        families[str(rows[index]["phrase_family"])].append(index)
    per_family = [
        sum(
            accepted[index] and predicted[index] is targets[index]
            for index in indices
        )
        / len(indices)
        for indices in families.values()
    ]
    hard = [index for index in known if bool(rows[index].get("hard_negative", False))]

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "known_coverage": rate(accepted_known, len(known)),
        "accepted_route_accuracy": rate(safe, accepted_known),
        "accepted_relation_precision": rate(safe, accepted_known),
        "singleton_acceptance_precision": rate(safe, accepted_total),
        "safe_coverage": rate(safe, len(known)),
        "wrong_bucket_access_rate": rate(wrong, len(known)),
        "false_memory_access_rate": rate(
            sum(accepted[index] for index in opened), len(opened)
        ),
        "ambiguous_false_memory_access_rate": rate(
            sum(accepted[index] for index in ambiguous), len(ambiguous)
        ),
        "unrelated_false_memory_access_rate": rate(
            sum(accepted[index] for index in unrelated), len(unrelated)
        ),
        "ambiguous_rejection_rate": rate(
            sum(not accepted[index] for index in ambiguous), len(ambiguous)
        ),
        "unrelated_rejection_rate": rate(
            sum(not accepted[index] for index in unrelated), len(unrelated)
        ),
        "family_macro_accuracy": sum(per_family) / len(per_family),
        "worst_family_accuracy": min(per_family),
        "hard_negative_accuracy": rate(
            sum(
                accepted[index] and predicted[index] is targets[index]
                for index in hard
            ),
            len(hard),
        ),
        "true_relation_inclusion_rate": rate(
            sum(targets[index] in candidate_sets[index] for index in known),
            len(known),
        ),
        "average_candidate_set_size": sum(map(len, candidate_sets))
        / len(candidate_sets),
        "known_multi_relation_set_rate": rate(
            sum(len(candidate_sets[index]) > 1 for index in known), len(known)
        ),
        "known_empty_set_rate": rate(
            sum(len(candidate_sets[index]) == 0 for index in known), len(known)
        ),
        "cross_relation_candidate_count": 0.0,
    }


def _selection_objective(metrics: Mapping[str, float]) -> tuple[float, ...]:
    gates = (
        metrics["false_memory_access_rate"] <= 0.20,
        metrics["wrong_bucket_access_rate"] <= 0.10,
        metrics["accepted_route_accuracy"] >= 0.90,
        metrics["known_coverage"] >= 0.75,
        metrics["safe_coverage"] >= 0.70,
        metrics["ambiguous_rejection_rate"] >= 0.50,
        metrics["unrelated_rejection_rate"] >= 0.90,
    )
    return (
        float(all(gates)),
        float(sum(gates)),
        metrics["safe_coverage"],
        metrics["accepted_route_accuracy"],
        metrics["singleton_acceptance_precision"],
        metrics["ambiguous_rejection_rate"],
        metrics["unrelated_rejection_rate"],
        -metrics["false_memory_access_rate"],
        -metrics["wrong_bucket_access_rate"],
        metrics["known_coverage"],
        metrics["family_macro_accuracy"],
        metrics["worst_family_accuracy"],
        metrics["hard_negative_accuracy"],
        -metrics["average_candidate_set_size"],
    )


def _pick_best(
    candidates: Mapping[str, tuple[Mapping[str, float], Any]],
    *,
    preference: Sequence[str] = (),
) -> tuple[str, Mapping[str, float], Any]:
    rank = {name: len(preference) - index for index, name in enumerate(preference)}
    name = max(
        candidates,
        key=lambda key: (
            _selection_objective(candidates[key][0]),
            float(rank.get(key, 0)),
            key,
        ),
    )
    metrics, payload = candidates[name]
    return name, metrics, payload


def _select_calibration(
    values: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, list[dict[RelationId, float]]],
    context: PublicModelContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_rows: list[dict[str, Any]] = []
    base: dict[str, tuple[dict[str, float], Any]] = {}
    for name, candidate_scores in scores.items():
        if not (name == "R0" or name.startswith(("R1:", "R2:"))):
            continue
        routes, sets = _argmax_decisions(candidate_scores)
        metrics = _compact_metrics(rows, routes, sets)
        base[name] = (metrics, {"routes": routes, "sets": sets})
        selection_rows.append(
            {"stage": "base", "candidate": name, **metrics}
        )

    r1_candidates = {key: value for key, value in base.items() if key.startswith("R1:")}
    r2_candidates = {key: value for key, value in base.items() if key.startswith("R2:")}
    selected_r1, _, _ = _pick_best(r1_candidates)
    selected_r2, _, _ = _pick_best(r2_candidates)

    r3_candidates: dict[str, tuple[dict[str, float], Any]] = {}
    source_keys = {"R1": selected_r1, "R2": selected_r2}
    grid = [float(value) for value in values["selection"]["threshold_grid"]]
    margins = [float(value) for value in values["selection"]["minimum_margin_grid"]]
    for source in values["selection"]["r3_evidence_sources"]:
        source_key = source_keys[str(source)]
        source_scores = scores[source_key]
        for threshold_values in itertools.product(grid, repeat=len(RelationId)):
            thresholds = dict(zip(RelationId, threshold_values))
            for margin in margins:
                routes, sets = _threshold_decisions(
                    source_scores, thresholds, minimum_margin=margin
                )
                metrics = _compact_metrics(rows, routes, sets)
                key = (
                    f"{source}:"
                    + ",".join(f"{value:g}" for value in threshold_values)
                    + f":m={margin:g}"
                )
                payload = {
                    "evidence_source": str(source),
                    "source_score_key": source_key,
                    "thresholds": {
                        relation.value: float(thresholds[relation])
                        for relation in RelationId
                    },
                    "minimum_margin": margin,
                }
                r3_candidates[key] = (metrics, payload)
    selected_r3_key, selected_r3_metrics, selected_r3 = _pick_best(r3_candidates)
    selected_r3_routes, selected_r3_sets = _threshold_decisions(
        scores[str(selected_r3["source_score_key"])],
        selected_r3["thresholds"],
        minimum_margin=float(selected_r3["minimum_margin"]),
    )
    for key, (metrics, payload) in r3_candidates.items():
        selection_rows.append(
            {
                "stage": "R3",
                "candidate": key,
                "selected": key == selected_r3_key,
                "evidence_source": payload["evidence_source"],
                "minimum_margin": payload["minimum_margin"],
                "threshold_registry_id": payload["thresholds"]["registry_id"],
                "threshold_city_code": payload["thresholds"]["city_code"],
                "threshold_access_code": payload["thresholds"]["access_code"],
                **metrics,
            }
        )

    targets = [_relation_target(row) for row in rows]
    r4_candidates: dict[str, tuple[dict[str, float], Any]] = {}
    for source in values["selection"]["r4_evidence_sources"]:
        source_key = source_keys[str(source)]
        source_scores = scores[source_key]
        for alpha in values["selection"]["conformal_alpha_grid"]:
            calibrator = fit_class_conditional_conformal(
                source_scores, targets, alpha=float(alpha)
            )
            routes, sets = _conformal_decisions(source_scores, calibrator)
            metrics = _compact_metrics(rows, routes, sets)
            key = f"{source}:alpha={float(alpha):g}"
            r4_candidates[key] = (
                metrics,
                {
                    "evidence_source": str(source),
                    "source_score_key": source_key,
                    "calibrator": calibrator.to_dict(),
                },
            )
    selected_r4_key, selected_r4_metrics, selected_r4 = _pick_best(r4_candidates)
    selected_r4_calibrator = ClassConditionalConformalCalibrator.from_dict(
        selected_r4["calibrator"]
    )
    selected_r4_routes, selected_r4_sets = _conformal_decisions(
        scores[str(selected_r4["source_score_key"])],
        selected_r4_calibrator,
    )
    for key, (metrics, payload) in r4_candidates.items():
        selection_rows.append(
            {
                "stage": "R4",
                "candidate": key,
                "selected": key == selected_r4_key,
                "evidence_source": payload["evidence_source"],
                "alpha": payload["calibrator"]["alpha"],
                **metrics,
            }
        )

    final_candidates = {
        "R0": base["R0"],
        "R1": base[selected_r1],
        "R2": base[selected_r2],
        "R3": (
            selected_r3_metrics,
            {"routes": selected_r3_routes, "sets": selected_r3_sets},
        ),
        "R4": (
            selected_r4_metrics,
            {"routes": selected_r4_routes, "sets": selected_r4_sets},
        ),
    }
    preference = [str(value) for value in values["selection"]["final_router_preference"]]
    selected_router, selected_metrics, _ = _pick_best(
        final_candidates, preference=preference
    )
    for name, (metrics, _) in final_candidates.items():
        selection_rows.append(
            {
                "stage": "final_router",
                "candidate": name,
                "selected": name == selected_router,
                **metrics,
            }
        )

    selected_r2_key = selected_r2.removeprefix("R2:")
    parameters = {
        "schema_version": 1,
        "selected_router": selected_router,
        "selected_r1_score_key": selected_r1,
        "selected_r1_aggregation": selected_r1.removeprefix("R1:"),
        "selected_r2_score_key": selected_r2,
        "selected_r2_candidate": selected_r2_key,
        "selected_r2_head_state_sha256": context.r2_head_state_sha256[
            selected_r2_key
        ],
        "selected_r3_candidate": selected_r3_key,
        "selected_r3": {
            key: value
            for key, value in selected_r3.items()
        },
        "selected_r4_candidate": selected_r4_key,
        "selected_r4": {
            key: value
            for key, value in selected_r4.items()
        },
        "r0_head_file_sha256": context.r0_head_file_sha256,
        "r0_head_state_sha256": context.r0_head_state_sha256,
        "final_calibration_metrics": {
            name: metrics for name, (metrics, _) in final_candidates.items()
        },
        "selected_calibration_metrics": selected_metrics,
        "selection_objective": [
            "all_exploratory_effective_calibration_gates",
            "gate_count",
            "safe_coverage",
            "accepted_route_accuracy",
            "singleton_acceptance_precision",
            "ambiguous_rejection_rate",
            "unrelated_rejection_rate",
            "lower_false_memory_access_rate",
            "lower_wrong_bucket_access_rate",
            "known_coverage",
            "family_macro_accuracy",
            "worst_family_accuracy",
            "hard_negative_accuracy",
            "lower_average_candidate_set_size",
            "preregistered_router_preference",
        ],
        "r4_guarantee_limit": (
            "alpha/evidence source are selected on the same calibration split; "
            "no nominal split-conformal coverage guarantee is claimed, especially "
            "under lexical or semantic distribution shift"
        ),
    }
    return parameters, selection_rows


def _decisions_from_parameters(
    scores: Mapping[str, list[dict[RelationId, float]]],
    parameters: Mapping[str, Any],
) -> dict[
    str,
    tuple[
        list[dict[RelationId, float]],
        list[RouteResult],
        list[frozenset[RelationId]],
    ],
]:
    output = {}
    base_keys = {
        "R0": "R0",
        "R1": str(parameters["selected_r1_score_key"]),
        "R2": str(parameters["selected_r2_score_key"]),
    }
    for variant, key in base_keys.items():
        routes, sets = _argmax_decisions(scores[key])
        output[variant] = (scores[key], routes, sets)

    r3 = parameters["selected_r3"]
    r3_scores = scores[str(r3["source_score_key"])]
    r3_routes, r3_sets = _threshold_decisions(
        r3_scores,
        r3["thresholds"],
        minimum_margin=float(r3["minimum_margin"]),
    )
    output["R3"] = (r3_scores, r3_routes, r3_sets)

    r4 = parameters["selected_r4"]
    r4_scores = scores[str(r4["source_score_key"])]
    calibrator = ClassConditionalConformalCalibrator.from_dict(r4["calibrator"])
    r4_routes, r4_sets = _conformal_decisions(r4_scores, calibrator)
    output["R4"] = (r4_scores, r4_routes, r4_sets)
    return output


def _answer_free_contract_probe(
    rows: Sequence[Mapping[str, Any]], routes: Sequence[RouteResult]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """用公开实体一热原型实际穿过 typed API；不训练或模拟 private value。"""

    entities = sorted({str(row["entity"]) for row in rows})
    lookup = {entity: index for index, entity in enumerate(entities)}
    prototypes = torch.eye(len(entities), dtype=torch.float32)
    buckets = {relation: prototypes for relation in RelationId}
    accepted, _, _ = _decision_columns(routes)
    fact_hits: list[bool | None] = []
    row_hits: list[bool | None] = []
    reciprocal: list[float | None] = []
    margins: list[float | None] = []
    candidate_counts: list[int | None] = []
    cross_counts: list[int | None] = []
    records = []
    for row, route in zip(rows, routes):
        embedding = prototypes[lookup[str(row["entity"])]]
        execution = execute_selective_route(embedding, route, buckets)
        known = str(row["sample_type"]) == "known"
        if isinstance(route, AcceptedRoute) and known:
            assert execution.retrieval is not None
            correct = (
                route.relation_id is RelationId(str(row["relation_id"]))
                and execution.retrieval.candidate_index
                == lookup[str(row["entity"])]
            )
            fact_hits.append(correct)
            row_hits.append(correct)
            reciprocal.append(1.0 if correct else 0.0)
            margins.append(1.0)
            candidate_counts.append(execution.retrieval.candidate_count)
            cross_counts.append(
                execution.retrieval.cross_relation_candidate_count
            )
        else:
            fact_hits.append(None)
            row_hits.append(None)
            reciprocal.append(None)
            margins.append(None)
            candidate_counts.append(None)
            cross_counts.append(None)
        records.append(
            {
                "row_id": row["row_id"],
                "sample_type": row["sample_type"],
                "route_status": route.status,
                "memory_access_count": execution.memory_access_count,
                "relation_id": (
                    route.relation_id.value
                    if isinstance(route, AcceptedRoute)
                    else ""
                ),
                "cross_relation_candidate_count": (
                    execution.retrieval.cross_relation_candidate_count
                    if execution.retrieval is not None
                    else 0
                ),
            }
        )
    metrics = fact_retrieval_metrics(
        accepted,
        [str(row["sample_type"]) for row in rows],
        fact_hits,
        row_hits,
        reciprocal,
        margins,
        candidate_counts,
        cross_counts,
        oracle_fact_top1=1.0,
    )
    metrics.update(
        {
            "scope": "deterministic_public_answer_free_typed_bucket_contract_probe",
            "independent_fact_retrieval_claimed": False,
            "private_value_memory_used": False,
            "fact_metrics_are_route_equivalent": True,
        }
    )
    return metrics, records


def _minimal_pair_metrics(
    rows: Sequence[Mapping[str, Any]], routes: Sequence[RouteResult]
) -> dict[str, Any]:
    accepted, predicted, _ = _decision_columns(routes)
    indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("minimal_pair_id", "")).strip()
        and row["sample_type"] == "known"
    ]
    correct = sum(
        accepted[index]
        and predicted[index] is RelationId(str(rows[index]["relation_id"]))
        for index in indices
    )
    return {
        "hard_minimal_pair_accuracy": correct / len(indices) if indices else 0.0,
        "row_count": len(indices),
        "pair_ids": sorted({str(rows[index]["minimal_pair_id"]) for index in indices}),
    }


def _evaluate_variant(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[RelationId, float]],
    routes: Sequence[RouteResult],
    candidate_sets: Sequence[frozenset[RelationId]],
    *,
    variant: str,
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, predicted, reasons = _decision_columns(routes)
    targets = [_relation_target(row) for row in rows]
    known_indices = [index for index, target in enumerate(targets) if target is not None]
    known = known_routing_metrics(
        [predicted[index] for index in known_indices],
        [targets[index] for index in known_indices],  # type: ignore[list-item]
        [accepted[index] for index in known_indices],
        [str(rows[index]["phrase_family"]) for index in known_indices],
        [str(rows[index]["frame_id"]) for index in known_indices],
        hard_negative_mask=[
            bool(rows[index].get("hard_negative", False)) for index in known_indices
        ],
    )
    sets = set_valued_routing_metrics(
        candidate_sets,
        targets,
        [str(row["sample_type"]) for row in rows],
    )
    rejects = reject_quality_metrics(
        accepted,
        reasons,
        predicted,
        targets,
        [str(row["sample_type"]) for row in rows],
        [str(row["phrase_family"]) for row in rows],
        _reject_scores(scores, candidate_sets),
    )
    access = access_control_metrics(
        accepted,
        predicted,
        targets,
        [str(row["sample_type"]) for row in rows],
    )
    fact, retrieval_rows = _answer_free_contract_probe(rows, routes)
    evaluation = {
        "schema_version": 1,
        "stage": "C2.4c-exploratory-selective-router-evaluation",
        "evaluation_mode": EVALUATION_MODE,
        "variant": variant,
        "split": split_name,
        "known_routing": known,
        "set_valued_routing": sets,
        "reject_quality": rejects,
        "access_control": access,
        "fact_retrieval": fact,
        "minimal_pairs": _minimal_pair_metrics(rows, routes),
        "memory_contract": {
            "typed_relation_id_only": True,
            "single_bucket_per_accepted_route": True,
            "rejected_route_memory_access_count": 0,
            "cross_relation_candidate_count": fact[
                "cross_relation_candidate_count"
            ],
            "continuous_relation_reaches_memory": False,
            "confidence_reaches_memory": False,
            "semantic_embedding_reaches_memory": False,
            "raw_text_reaches_memory": False,
            "phrase_family_reaches_memory": False,
        },
        "formal_result": False,
        "independent_external_validation": False,
    }
    assert_finite_json(evaluation)
    prediction_rows, score_rows = [], []
    for row, score, route, candidates in zip(rows, scores, routes, candidate_sets):
        prediction_rows.append(
            {
                "variant": variant,
                "split": split_name,
                "row_id": row["row_id"],
                "phrase_family": row["phrase_family"],
                "sample_type": row["sample_type"],
                "target_relation": row.get("relation_id") or "",
                "route_status": route.status,
                "predicted_relation": (
                    route.relation_id.value
                    if isinstance(route, AcceptedRoute)
                    else ""
                ),
                "reject_reason": (
                    route.reason if isinstance(route, RejectedRoute) else ""
                ),
                "candidate_set": "|".join(
                    sorted(relation.value for relation in candidates)
                ),
            }
        )
        score_rows.append(
            {
                "variant": variant,
                "split": split_name,
                "row_id": row["row_id"],
                "registry_id_score": float(score[RelationId.REGISTRY_ID]),
                "city_code_score": float(score[RelationId.CITY_CODE]),
                "access_code_score": float(score[RelationId.ACCESS_CODE]),
                "candidate_set": "|".join(
                    sorted(relation.value for relation in candidates)
                ),
            }
        )
    return evaluation, prediction_rows, score_rows, retrieval_rows


def _evaluate_all(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, list[dict[RelationId, float]]],
    parameters: Mapping[str, Any],
    *,
    split_name: str,
    destination: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = _decisions_from_parameters(scores, parameters)
    evaluations, predictions, score_rows, retrieval_rows = {}, [], [], []
    for variant in VARIANTS:
        variant_scores, routes, sets = decisions[variant]
        evaluation, variant_predictions, variant_scores_rows, variant_retrieval = (
            _evaluate_variant(
                rows,
                variant_scores,
                routes,
                sets,
                variant=variant,
                split_name=split_name,
            )
        )
        evaluations[variant] = evaluation
        predictions.extend(variant_predictions)
        score_rows.extend(variant_scores_rows)
        for row in variant_retrieval:
            retrieval_rows.append({"variant": variant, "split": split_name, **row})
        write_json(destination / variant / "evaluation.json", evaluation)
    return evaluations, predictions, score_rows, retrieval_rows


def _load_router_freeze(
    config_path: str | Path,
    context: PublicModelContext,
    *,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    path = artifact_dir / "router_freeze_manifest.json"
    payload = load_json(path, label="C2.4c router freeze")
    digest = payload.pop("manifest_payload_sha256", None)
    if digest != canonical_sha256(payload):
        raise C24CProtocolError("router freeze payload SHA-256 不匹配")
    payload["manifest_payload_sha256"] = digest
    benchmark = verify_benchmark_seal(config_path, output_dir=output_dir)
    if payload.get("benchmark_freeze_manifest_sha256") != sha256_file(
        artifact_dir / "benchmark_freeze_manifest.json"
    ):
        raise C24CProtocolError("router freeze 未绑定当前 benchmark seal")
    if payload.get("git_commit") != benchmark["git"]["commit"]:
        raise C24CProtocolError("router freeze git commit 漂移")
    if payload.get("model_provenance") != context.model_provenance:
        raise C24CProtocolError("router freeze encoder manifests 漂移")
    verify_selected_head(context, payload["router_parameters"])
    return payload


def calibrate_exploratory(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    benchmark = verify_benchmark_seal(config_path, output_dir=output_dir)
    train_rows = read_declared_split(config_path, "public_train_v2_1")
    context = build_public_model_context(config_path, train_rows, device=device)
    assert_discrete_memory_contract()
    mark_phase_started(
        config_path,
        "calibration",
        {
            "selection_split": "public_calibration_v2_1",
            "train_rows": len(train_rows),
            "model_provenance": context.model_provenance,
            "r0_head_file_sha256": context.r0_head_file_sha256,
        },
        output_dir=output_dir,
    )
    try:
        rows = read_declared_split(config_path, "public_calibration_v2_1")
        scores = score_public_rows(config_path, context, rows)
        parameters, selection_rows = _select_calibration(
            values, rows, scores, context
        )
        freeze = {
            "schema_version": 1,
            "stage": "C2.4c-exploratory-selective-router-evaluation",
            "status": "router_frozen_after_public_calibration_v2_1",
            "evaluation_mode": EVALUATION_MODE,
            "git_commit": benchmark["git"]["commit"],
            "benchmark_freeze_manifest_sha256": sha256_file(
                artifact_dir / "benchmark_freeze_manifest.json"
            ),
            "data_sha256": {
                name: declaration["sha256"]
                for name, declaration in values["frozen_benchmark"]["data"].items()
            },
            "model_provenance": context.model_provenance,
            "router_parameters": parameters,
            "selection_protocol": {
                "train_split": "public_train_v2_1",
                "selection_split": "public_calibration_v2_1",
                "development_used_for_selection": False,
                "locked_audit_used_for_selection": False,
                "parameters_frozen_before_development": True,
                "formal_calibration_allowed": False,
                "independent_external_validation": False,
            },
            "private_value_memory_trained": False,
            "private_answers_loaded": False,
            "confirmation_created_or_read": False,
        }
        freeze["manifest_payload_sha256"] = canonical_sha256(freeze)
        write_json(artifact_dir / "router_freeze_manifest.json", freeze)
        _write_csv(artifact_dir / "router_selection.csv", selection_rows)

        decisions = _decisions_from_parameters(scores, parameters)
        calibration_evaluations = {}
        for variant in VARIANTS:
            variant_scores, routes, sets = decisions[variant]
            evaluation, _, _, _ = _evaluate_variant(
                rows,
                variant_scores,
                routes,
                sets,
                variant=variant,
                split_name="public_calibration_v2_1",
            )
            calibration_evaluations[variant] = evaluation
        write_json(
            artifact_dir / "calibration_results.json",
            {
                "schema_version": 1,
                "stage": "C2.4c-exploratory-selective-router-evaluation",
                "evaluation_mode": EVALUATION_MODE,
                "selected_router": parameters["selected_router"],
                "router_parameters": parameters,
                "evaluations": calibration_evaluations,
                "selection_split": "public_calibration_v2_1",
                "development_used_for_selection": False,
                "locked_audit_used_for_selection": False,
                "formal_calibration_allowed": False,
                "independent_external_validation": False,
            },
        )
        completed = mark_phase_completed(
            config_path,
            "calibration",
            {
                "calibration_rows": len(rows),
                "selected_router": parameters["selected_router"],
                "router_freeze_manifest_sha256": sha256_file(
                    artifact_dir / "router_freeze_manifest.json"
                ),
                "development_opened": False,
                "locked_audit_opened": False,
            },
            output_dir=output_dir,
        )
    except Exception as exc:
        write_protocol_incident_once(
            config_path,
            phase="calibration",
            violation=str(exc),
            output_dir=output_dir,
        )
        raise
    status = {
        **base_status(),
        "status": "calibration_completed_router_frozen_waiting_for_development",
        "git": git_state(config_path),
        "benchmark_freeze_manifest_sha256": sha256_file(
            artifact_dir / "benchmark_freeze_manifest.json"
        ),
        "router_freeze_manifest_sha256": sha256_file(
            artifact_dir / "router_freeze_manifest.json"
        ),
        "calibration_executed": True,
        "development_executed_once": False,
        "exploratory_locked_scoring_executed_once": False,
    }
    write_json(artifact_dir / "protocol_status.json", status)
    return {
        "selected_router": parameters["selected_router"],
        "router_freeze_manifest_sha256": status["router_freeze_manifest_sha256"],
        "calibration_completed_marker_sha256": sha256_file(completed),
    }


def _adapt_c23_development_rows(
    config_path: str | Path,
) -> list[dict[str, Any]]:
    public = build_c23_public_rows(config_path)
    rows = []
    for source in [*public["validation"], *public["reject"]]:
        sample_type = str(source["kind"])
        relation = source["attribute"] if sample_type == "known" else None
        rows.append(
            {
                **dict(source),
                "row_id": f"c23-development:{source['example_id']}",
                "sample_type": sample_type,
                "relation_id": relation,
                "phrase_family": source["family_id"],
                "frame": source["frame_id"],
                "hard_negative": False,
                "minimal_pair_id": "",
            }
        )
    return rows


def run_development_once(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    verify_benchmark_seal(config_path, output_dir=output_dir)
    train_rows = read_declared_split(config_path, "public_train_v2_1")
    context = build_public_model_context(config_path, train_rows, device=device)
    frozen = _load_router_freeze(
        config_path, context, output_dir=output_dir
    )
    mark_phase_started(
        config_path,
        "development",
        {
            "scope": "historical_c23_public_validation_and_reject_diagnostic_once",
            "used_for_selection": False,
            "router_freeze_manifest_sha256": sha256_file(
                artifact_dir / "router_freeze_manifest.json"
            ),
        },
        output_dir=output_dir,
    )
    try:
        rows = _adapt_c23_development_rows(config_path)
        scores = score_public_rows(config_path, context, rows)
        destination = artifact_dir / "development"
        evaluations, predictions, score_rows, retrieval = _evaluate_all(
            rows,
            scores,
            frozen["router_parameters"],
            split_name="historical_c23_public_development_diagnostic",
            destination=destination,
        )
        _write_csv(destination / "route_predictions.csv", predictions)
        _write_csv(destination / "route_candidate_scores.csv", score_rows)
        write_json(
            destination / "retrieval_predictions.json",
            {
                "schema_version": 1,
                "scope": "deterministic_public_answer_free_typed_bucket_contract_probe",
                "rows": retrieval,
            },
        )
        write_json(
            destination / "development_summary.json",
            {
                "schema_version": 1,
                "stage": "C2.4c-exploratory-selective-router-evaluation",
                "evaluation_mode": EVALUATION_MODE,
                "scope": "historical_c23_public_validation_and_reject_diagnostic_once",
                "row_count": len(rows),
                "source_rows_sha256": canonical_sha256(rows),
                "evaluations": evaluations,
                "selected_router": frozen["router_parameters"]["selected_router"],
                "used_for_selection": False,
                "used_for_status": False,
                "locked_audit_opened": False,
            },
        )
        completed = mark_phase_completed(
            config_path,
            "development",
            {
                "development_rows": len(rows),
                "development_rows_sha256": canonical_sha256(rows),
                "used_for_selection": False,
                "used_for_status": False,
                "locked_audit_opened": False,
            },
            output_dir=output_dir,
        )
    except Exception as exc:
        write_protocol_incident_once(
            config_path,
            phase="development",
            violation=str(exc),
            output_dir=output_dir,
        )
        raise
    status = {
        **base_status(),
        "status": "development_completed_once_waiting_for_locked_scoring",
        "git": git_state(config_path),
        "benchmark_freeze_manifest_sha256": sha256_file(
            artifact_dir / "benchmark_freeze_manifest.json"
        ),
        "router_freeze_manifest_sha256": sha256_file(
            artifact_dir / "router_freeze_manifest.json"
        ),
        "calibration_executed": True,
        "development_executed_once": True,
        "exploratory_locked_scoring_executed_once": False,
    }
    write_json(artifact_dir / "protocol_status.json", status)
    return {
        "development_rows": len(rows),
        "selected_router": frozen["router_parameters"]["selected_router"],
        "development_completed_marker_sha256": sha256_file(completed),
    }


def _exploratory_status(
    values: Mapping[str, Any],
    selected: Mapping[str, Any],
    r0: Mapping[str, Any],
) -> dict[str, Any]:
    known = selected["known_routing"]
    access = selected["access_control"]
    reject = selected["reject_quality"]
    fact = selected["fact_retrieval"]
    r0_ambiguous = r0["access_control"]["ambiguous_false_memory_access_rate"]
    selected_ambiguous = access["ambiguous_false_memory_access_rate"]
    reduction = (
        (r0_ambiguous - selected_ambiguous) / r0_ambiguous
        if r0_ambiguous > 0
        else 0.0
    )
    effective = values["exploratory_gates"]["effective"]
    strong = values["exploratory_gates"]["strong"]
    closed_gates = {
        "wrong_bucket_access_rate": access["wrong_bucket_access_rate"]
        <= float(effective["wrong_bucket_access_rate"]),
        "accepted_route_accuracy": known["accepted_route_accuracy"]
        >= float(effective["accepted_route_accuracy"]),
        "known_coverage": known["known_coverage"]
        >= float(effective["known_coverage"]),
        "safe_coverage": access["safe_coverage"]
        >= float(effective["safe_coverage"]),
        "cross_relation_candidate_count": fact["cross_relation_candidate_count"]
        == int(effective["cross_relation_candidate_count"]),
    }
    open_gates = {
        "ambiguous_fmar_relative_reduction_from_r0": reduction
        >= float(effective["ambiguous_fmar_relative_reduction_from_r0"]),
        "false_memory_access_rate": access["false_memory_access_rate"]
        <= float(effective["false_memory_access_rate"]),
    }
    strong_gates = {
        "ambiguous_rejection_rate": reject["ambiguous_rejection_rate"]
        >= float(strong["ambiguous_rejection_rate"]),
        "unrelated_rejection_rate": reject["unrelated_rejection_rate"]
        >= float(strong["unrelated_rejection_rate"]),
        "false_memory_access_rate": access["false_memory_access_rate"]
        <= float(strong["false_memory_access_rate"]),
        "accepted_route_accuracy": known["accepted_route_accuracy"]
        >= float(strong["accepted_route_accuracy"]),
        "known_coverage": known["known_coverage"]
        >= float(strong["known_coverage"]),
        "safe_coverage": access["safe_coverage"]
        >= float(strong["safe_coverage"]),
        "worst_family_accuracy": known["worst_family_accuracy"]
        >= float(strong["worst_family_accuracy"]),
    }
    return {
        "closed_set_selective_router_status": (
            "passed" if all(closed_gates.values()) else "failed"
        ),
        "open_set_abstention_status": (
            "passed" if all(open_gates.values()) else "failed"
        ),
        "exploratory_method_effective": all(
            [*closed_gates.values(), *open_gates.values()]
        ),
        "exploratory_method_strong_effect": all(strong_gates.values()),
        "closed_exploratory_gates": closed_gates,
        "open_exploratory_gates": open_gates,
        "strong_exploratory_gates": strong_gates,
        "ambiguous_fmar_relative_reduction_from_r0": reduction,
    }


def _ablation_rows(evaluations: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        evaluation = evaluations[variant]
        known = evaluation["known_routing"]
        reject = evaluation["reject_quality"]
        access = evaluation["access_control"]
        fact = evaluation["fact_retrieval"]
        rows.append(
            {
                "variant": variant,
                "known_coverage": known["known_coverage"],
                "accepted_route_accuracy": known["accepted_route_accuracy"],
                "family_macro_accuracy": known["relation_family_macro_accuracy"],
                "worst_family_accuracy": known["worst_family_accuracy"],
                "access_to_registry_error_rate": known[
                    "access_to_registry_error_rate"
                ],
                "registry_to_access_error_rate": known[
                    "registry_to_access_error_rate"
                ],
                "city_to_other_error_rate": known["city_to_other_error_rate"],
                "hard_minimal_pair_accuracy": evaluation["minimal_pairs"][
                    "hard_minimal_pair_accuracy"
                ],
                "ambiguous_rejection_rate": reject["ambiguous_rejection_rate"],
                "unrelated_rejection_rate": reject["unrelated_rejection_rate"],
                "false_memory_access_rate": access["false_memory_access_rate"],
                "ambiguous_false_memory_access_rate": access[
                    "ambiguous_false_memory_access_rate"
                ],
                "unrelated_false_memory_access_rate": access[
                    "unrelated_false_memory_access_rate"
                ],
                "wrong_bucket_access_rate": access["wrong_bucket_access_rate"],
                "safe_coverage": access["safe_coverage"],
                "accepted_fact_top1_contract_probe": fact["accepted_fact_top1"],
                "all_query_fact_top1_contract_probe": fact["all_query_fact_top1"],
                "cross_relation_candidate_count": fact[
                    "cross_relation_candidate_count"
                ],
            }
        )
    return rows


def _error_analysis_rows(
    predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in predictions
        if (
            row["sample_type"] != "known"
            and row["route_status"] == "accept"
        )
        or (
            row["sample_type"] == "known"
            and row["predicted_relation"] != row["target_relation"]
        )
    ]


def _render_report(
    config_path: str | Path,
    summary: Mapping[str, Any],
    evaluations: Mapping[str, Any],
) -> str:
    selected_router = str(summary["selected_router"])
    selected = evaluations[selected_router]
    status = summary["exploratory_status"]
    table = [
        "| Router | Known coverage | Accepted route acc. | Worst family | Ambiguous reject | Unrelated reject | FMAR | Wrong bucket | Safe coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = evaluations[variant]
        known, reject, access = (
            item["known_routing"],
            item["reject_quality"],
            item["access_control"],
        )
        table.append(
            "| {variant} | {coverage:.2%} | {accepted:.2%} | {worst:.2%} | "
            "{ambiguous:.2%} | {unrelated:.2%} | {fmar:.2%} | {wrong:.2%} | "
            "{safe:.2%} |".format(
                variant=variant,
                coverage=known["known_coverage"],
                accepted=known["accepted_route_accuracy"],
                worst=known["worst_family_accuracy"],
                ambiguous=reject["ambiguous_rejection_rate"],
                unrelated=reject["unrelated_rejection_rate"],
                fmar=access["false_memory_access_rate"],
                wrong=access["wrong_bucket_access_rate"],
                safe=access["safe_coverage"],
            )
        )
    parameters = summary["router_parameters"]
    return "\n".join(
        [
            "# Stage C2.4c：Exploratory Router Evaluation",
            "",
            "## 结论",
            "",
            "本阶段在已冻结的 C2.4b v2.1 合成压力测试上完成了一次",
            "`calibration → development → locked scoring`。评估是",
            "`exploratory_non_independent`：同一单模型 AI 参与过数据修订并看过",
            "locked 文本，因此结果不是独立验证、真人审核 benchmark 或部署安全证据。",
            "",
            f"最终 router 为 **{selected_router}**。闭集探索状态为",
            f"`{status['closed_set_selective_router_status']}`，开放集探索状态为",
            f"`{status['open_set_abstention_status']}`；readiness 与 C3 门禁仍为 false。",
            "",
            "## Locked R0–R4",
            "",
            *table,
            "",
            "## 冻结参数",
            "",
            f"- R1 aggregation：`{parameters['selected_r1_aggregation']}`",
            f"- R2 encoder/ridge：`{parameters['selected_r2_candidate']}`",
            f"- R3：`{parameters['selected_r3_candidate']}`",
            f"- R4：`{parameters['selected_r4_candidate']}`",
            f"- router freeze SHA-256：`{summary['router_freeze_manifest_sha256']}`",
            "",
            "R0 使用同一 BGE revision、C2.3 public train、view、class order 与",
            "ridge 配方的确定性功能重建。原始 checkpoint 按仓库策略未提交，",
            "远端 refs/releases/actions 亦不存在；重建在 216 条历史 public",
            " validation 上复现了相同准确率和唯一错误，但 `.pt` SHA 不同，",
            "因此不声称新输入 logits 与历史 checkpoint 逐位相同。",
            "",
            "R4 的 evidence source 与 alpha 在同一 calibration split 上选择，",
            "所以不声称标准 split-conformal 的名义有限样本覆盖保证；在 lexical/",
            "semantic shift 下尤其没有严格保证。",
            "",
            "## 访问控制解释",
            "",
            f"- R0 ambiguous FMAR："
            f"`{evaluations['R0']['access_control']['ambiguous_false_memory_access_rate']:.2%}`",
            f"- {selected_router} ambiguous FMAR："
            f"`{selected['access_control']['ambiguous_false_memory_access_rate']:.2%}`",
            f"- 相对下降："
            f"`{status['ambiguous_fmar_relative_reduction_from_r0']:.2%}`",
            f"- {selected_router} safe coverage："
            f"`{selected['access_control']['safe_coverage']:.2%}`",
            "",
            "fact retrieval 数字来自确定性的公开 answer-free typed-bucket contract",
            " probe，与路由正确性等价；它不是独立的 learned entity/fact retrieval",
            " 结果，也没有加载 private value memory。",
            "",
            "## 有限表述",
            "",
            "选择性离散路由闭集正式门槛未评估。",
            "开放集弃权正式门槛未评估。",
            "离散 memory contract 保持。",
            "`ready_to_create_new_confirmation_pool = false`。",
            "`c3_eligible = false`。",
            "本阶段没有训练 private memory，没有加载 private answer，没有执行答案注入或密钥攻击。",
            "",
            "v2.1 是受控 lexical-OOD/ambiguity 合成压力测试，37 个 boundary family",
            "反映合成构造限制而非 37 个错标；结论不得外推为自然语言真实分布表现。",
            "",
        ]
    )


def run_locked_once(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, _ = output_paths(config_path, output_dir=output_dir)
    benchmark = verify_benchmark_seal(config_path, output_dir=output_dir)
    train_rows = read_declared_split(config_path, "public_train_v2_1")
    context = build_public_model_context(config_path, train_rows, device=device)
    frozen = _load_router_freeze(config_path, context, output_dir=output_dir)
    development_marker = load_json(
        artifact_dir / "development_completed.json",
        label="C2.4c development completion",
    )
    if development_marker.get("used_for_selection") is not False:
        raise C24CProtocolError("development completion 标记参与了选择")
    assert_discrete_memory_contract()
    mark_phase_started(
        config_path,
        "locked_scoring",
        {
            "split": "public_locked_audit_v2_1",
            "evaluation_mode": EVALUATION_MODE,
            "router_freeze_manifest_sha256": sha256_file(
                artifact_dir / "router_freeze_manifest.json"
            ),
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
        },
        output_dir=output_dir,
    )
    try:
        locked_rows = read_declared_split(
            config_path, "public_locked_audit_v2_1"
        )
        scores = score_public_rows(config_path, context, locked_rows)
        evaluations, predictions, score_rows, retrieval = _evaluate_all(
            locked_rows,
            scores,
            frozen["router_parameters"],
            split_name="public_locked_audit_v2_1",
            destination=artifact_dir,
        )
        selected_router = str(frozen["router_parameters"]["selected_router"])
        status = _exploratory_status(
            values, evaluations[selected_router], evaluations["R0"]
        )
        _write_csv(artifact_dir / "stage_c24c_ablation.csv", _ablation_rows(evaluations))
        _write_csv(artifact_dir / "route_predictions.csv", predictions)
        _write_csv(artifact_dir / "route_candidate_scores.csv", score_rows)
        _write_csv(artifact_dir / "error_analysis.csv", _error_analysis_rows(predictions))
        write_json(
            artifact_dir / "retrieval_predictions.json",
            {
                "schema_version": 1,
                "scope": "deterministic_public_answer_free_typed_bucket_contract_probe",
                "independent_fact_retrieval_claimed": False,
                "private_value_memory_used": False,
                "rows": retrieval,
            },
        )
        completed = mark_phase_completed(
            config_path,
            "locked_scoring",
            {
                "locked_rows": len(locked_rows),
                "selected_router": selected_router,
                "locked_audit_used_for_selection": False,
                "benchmark_bytes_unchanged_after_scoring": (
                    verify_benchmark_seal(config_path, output_dir=output_dir)
                    == benchmark
                ),
            },
            output_dir=output_dir,
        )
    except Exception as exc:
        write_protocol_incident_once(
            config_path,
            phase="locked_scoring",
            violation=str(exc),
            output_dir=output_dir,
        )
        raise

    summary = {
        **base_status(),
        "status": "exploratory_locked_scoring_completed_once",
        "evaluation_mode": EVALUATION_MODE,
        "selected_router": selected_router,
        "router_parameters": frozen["router_parameters"],
        "router_freeze_manifest_sha256": sha256_file(
            artifact_dir / "router_freeze_manifest.json"
        ),
        "benchmark_freeze_manifest_sha256": sha256_file(
            artifact_dir / "benchmark_freeze_manifest.json"
        ),
        "locked_scoring_completed_marker_sha256": sha256_file(completed),
        "exploratory_status": status,
        "closed_set_selective_router_status": status[
            "closed_set_selective_router_status"
        ],
        "open_set_abstention_status": status["open_set_abstention_status"],
        "closed_set_selective_router_ready": False,
        "open_set_abstention_ready": False,
        "ready_to_create_new_confirmation_pool": False,
        "c3_eligible": False,
        "calibration_executed": True,
        "development_executed_once": True,
        "exploratory_locked_scoring_executed_once": True,
        "formal_locked_audit_executed": False,
        "selected_evaluation": evaluations[selected_router],
        "evaluations": evaluations,
        "git": git_state(config_path),
        "model_provenance": context.model_provenance,
        "data_sha256": {
            name: declaration["sha256"]
            for name, declaration in values["frozen_benchmark"]["data"].items()
        },
        "source_hashes": {
            "typed_memory_contract": values["frozen_benchmark"][
                "typed_memory_contract"
            ]["sha256"],
            "selective_router": values["frozen_benchmark"]["selective_router"][
                "sha256"
            ],
            "v21_config": values["frozen_benchmark"]["config"]["sha256"],
        },
        "selection_protocol": frozen["selection_protocol"],
        "locked_data_labels_or_results_used_for_selection": False,
        "development_metrics_used_for_selection": False,
        "benchmark_modified_after_freeze": False,
        "independent_fact_retrieval_claimed": False,
        "formal_research_thresholds_assessed": False,
        "paper_claim_limit": (
            "在单模型 AI 审核的合成压力测试中，选择性离散路由显示出"
            "降低错误 memory access 的趋势；不构成独立验证或部署安全证据。"
        ),
    }
    write_json(artifact_dir / "stage_c24c_summary.json", summary)
    write_json(
        artifact_dir / "protocol_status.json",
        {
            **base_status(),
            "status": summary["status"],
            "closed_set_selective_router_status": summary[
                "closed_set_selective_router_status"
            ],
            "open_set_abstention_status": summary["open_set_abstention_status"],
            "calibration_executed": True,
            "development_executed_once": True,
            "exploratory_locked_scoring_executed_once": True,
            "formal_locked_audit_executed": False,
            "selected_router": selected_router,
            "router_freeze_manifest_sha256": summary[
                "router_freeze_manifest_sha256"
            ],
            "git": summary["git"],
        },
    )
    report_path = _repo_report_path(config_path)
    report_path.write_text(
        _render_report(config_path, summary, evaluations), encoding="utf-8"
    )
    inventory = exact_file_inventory(
        artifact_dir, exclude=("artifact_sha256_manifest.json",)
    )
    artifact_manifest = {
        "schema_version": 1,
        "stage": "C2.4c-exploratory-selective-router-evaluation",
        "evaluation_mode": EVALUATION_MODE,
        "git": summary["git"],
        "model_provenance": context.model_provenance,
        "data_sha256": summary["data_sha256"],
        "router_freeze_manifest_sha256": summary[
            "router_freeze_manifest_sha256"
        ],
        "file_count": len(inventory),
        "files": inventory,
    }
    artifact_manifest["manifest_payload_sha256"] = canonical_sha256(
        artifact_manifest
    )
    write_json(artifact_dir / "artifact_sha256_manifest.json", artifact_manifest)
    return {
        "selected_router": selected_router,
        "closed_set_selective_router_status": summary[
            "closed_set_selective_router_status"
        ],
        "open_set_abstention_status": summary["open_set_abstention_status"],
        "ready_to_create_new_confirmation_pool": False,
        "c3_eligible": False,
        "artifact_sha256_manifest": sha256_file(
            artifact_dir / "artifact_sha256_manifest.json"
        ),
    }


def _repo_report_path(config_path: str | Path) -> Path:
    config = Path(config_path).resolve()
    root = config.parent.parent
    if not (root / "pyproject.toml").is_file():
        raise C24CProtocolError("无法定位 C2.4c report 根目录")
    return root / "PHASE_C24C_REPORT.md"


def run_smoke(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """纯 synthetic evidence smoke；绝不读取 v2.1 locked。"""

    path = Path(config_path)
    values = json.loads(json.dumps(__import__("yaml").safe_load(path.read_text())))
    if (
        values.get("mode") != "smoke"
        or values.get("touches_frozen_v21_locked_audit") is not False
    ):
        raise C24CProtocolError("C2.4c smoke 配置非法")
    destination = (
        Path(output_dir)
        if output_dir is not None
        else resolve_path(config_path, values["output_dir"])
    )
    if destination.exists():
        raise C24CProtocolError("C2.4c smoke 输出已存在")
    rows, scores = [], []
    relations = list(RelationId)
    for index in range(int(values["rows_per_type"])):
        relation = relations[index % len(relations)]
        rows.append(
            {
                "row_id": f"smoke-known-{index}",
                "sample_type": "known",
                "relation_id": relation.value,
                "phrase_family": f"smoke-known-family-{index}",
                "frame_id": f"smoke-frame-{index % 2}",
                "entity": f"smoke-entity-{index % 3}",
                "hard_negative": False,
                "minimal_pair_id": "",
            }
        )
        scores.append(
            {
                item: (0.9 if item is relation else 0.1) for item in RelationId
            }
        )
    for sample_type, base in (("ambiguous", 0.8), ("unrelated", 0.05)):
        for index in range(int(values["rows_per_type"])):
            rows.append(
                {
                    "row_id": f"smoke-{sample_type}-{index}",
                    "sample_type": sample_type,
                    "relation_id": None,
                    "phrase_family": f"smoke-{sample_type}-family-{index}",
                    "frame_id": f"smoke-frame-{index % 2}",
                    "entity": f"smoke-entity-{index % 3}",
                    "hard_negative": False,
                    "minimal_pair_id": "",
                }
            )
            scores.append(
                {
                    relation: (
                        base if sample_type == "unrelated" else base - 0.01 * offset
                    )
                    for offset, relation in enumerate(RelationId)
                }
            )
    routes, sets = _threshold_decisions(
        scores,
        {relation: 0.5 for relation in RelationId},
        minimum_margin=0.0,
    )
    evaluation, predictions, score_rows, retrieval = _evaluate_variant(
        rows,
        scores,
        routes,
        sets,
        variant="R3",
        split_name="synthetic_smoke",
    )
    destination.mkdir(parents=True)
    write_json(destination / "evaluation.json", evaluation)
    _write_csv(destination / "route_predictions.csv", predictions)
    _write_csv(destination / "route_candidate_scores.csv", score_rows)
    write_json(
        destination / "retrieval_predictions.json",
        {"schema_version": 1, "rows": retrieval},
    )
    write_json(
        destination / "protocol_status.json",
        {
            **base_status(),
            "status": "synthetic_smoke_completed",
            "touches_frozen_v21_locked_audit": False,
        },
    )
    return {
        "status": "synthetic_smoke_completed",
        "row_count": len(rows),
        "touches_frozen_v21_locked_audit": False,
    }


def assert_selection_api_is_calibration_only() -> None:
    parameters = inspect.signature(_select_calibration).parameters
    forbidden = ("development", "locked", "audit")
    if any(token in name.casefold() for name in parameters for token in forbidden):
        raise RuntimeError("C2.4c selection API 暴露 post-calibration 输入")


assert_selection_api_is_calibration_only()


__all__ = [
    "C24CProtocolError",
    "calibrate_exploratory",
    "run_development_once",
    "run_locked_once",
    "run_smoke",
    "seal_benchmark",
]
