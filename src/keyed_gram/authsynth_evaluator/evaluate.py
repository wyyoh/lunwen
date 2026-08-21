"""独立 evaluator：比较盲分析结果与 hidden concrete semantics。"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from keyed_gram.authsynth_analyzer import BlindCGARAnalyzer
from keyed_gram.authsynth_analyzer.models import AnalysisOutcome
from keyed_gram.authsynth_analyzer.shield import synthesize_blind_shield
from keyed_gram.authsynth_shared import (
    AnalysisStatus,
    EventRecord,
    HighLevelSafetySpec,
    ShieldDecision,
    canonical_digest,
)

from .cases import HiddenEvaluatorCase
from .replay_service import InMemorySandboxReplay


class EvaluationMethod(str, Enum):
    DENY_ALL = "DenyAll"
    DECLARED_MONITOR = "Declared-Contract Monitor"
    MULTI_EVIDENCE = "Multi-Evidence Characterizer"
    NO_CGAR = "AuthSynth-NoCEGAR"
    BLIND_CGAR = "AuthSynth-CGAR"
    FULL_SEMANTICS = "Full-Semantics AuthSynth Upper Bound"


METHODS = tuple(EvaluationMethod)


@dataclass(frozen=True)
class EvaluationRow:
    method: EvaluationMethod
    case_id: str
    split: str
    mutation_category: str
    omission_types: str
    benign_allowed: bool
    attack_allowed: bool
    attack_forbidden: bool
    forbidden_trace_accepted: bool
    effect_covered_count: int
    actual_effect_count: int
    policy_omission_discovered: bool
    policy_omission_expected: bool
    counterexamples_generated: int
    counterexamples_validated: int
    spurious_counterexamples: int
    correct_refinement_count: int
    drift_detected: bool
    false_verified_complete: bool
    analysis_unknown: bool
    unknown_expected: bool
    converged: bool
    allowed_gold_safe_action_count: int
    gold_safe_action_count: int
    analysis_time_ms: float
    replay_query_count: int
    shield_digest: str
    certificate_digest: str
    refinement_iterations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "case_id": self.case_id,
            "split": self.split,
            "mutation_category": self.mutation_category,
            "omission_types": self.omission_types,
            "benign_allowed": self.benign_allowed,
            "attack_allowed": self.attack_allowed,
            "attack_forbidden": self.attack_forbidden,
            "forbidden_trace_accepted": self.forbidden_trace_accepted,
            "effect_covered_count": self.effect_covered_count,
            "actual_effect_count": self.actual_effect_count,
            "policy_omission_discovered": self.policy_omission_discovered,
            "policy_omission_expected": self.policy_omission_expected,
            "counterexamples_generated": self.counterexamples_generated,
            "counterexamples_validated": self.counterexamples_validated,
            "spurious_counterexamples": self.spurious_counterexamples,
            "correct_refinement_count": self.correct_refinement_count,
            "drift_detected": self.drift_detected,
            "false_verified_complete": self.false_verified_complete,
            "analysis_unknown": self.analysis_unknown,
            "unknown_expected": self.unknown_expected,
            "converged": self.converged,
            "allowed_gold_safe_action_count": self.allowed_gold_safe_action_count,
            "gold_safe_action_count": self.gold_safe_action_count,
            "analysis_time_ms": self.analysis_time_ms,
            "replay_query_count": self.replay_query_count,
            "shield_digest": self.shield_digest,
            "certificate_digest": self.certificate_digest,
        }


def _declared_spec(case: HiddenEvaluatorCase) -> HighLevelSafetySpec:
    source = case.analyzer_input.trusted_safety_spec
    if source is None:
        return HighLevelSafetySpec("declared-empty", "tenant-a")
    codes = case.analyzer_input.declared_policy_codes
    return HighLevelSafetySpec(
        spec_id=f"declared-{source.spec_id}",
        protected_tenant_id=source.protected_tenant_id,
        forbidden_external_destinations=(
            source.forbidden_external_destinations
            if "forbidden_external_destination" in codes
            else frozenset()
        ),
        forbidden_effect_kinds=(
            source.forbidden_effect_kinds
            if "forbidden_effect_kind" in codes
            else frozenset()
        ),
        forbidden_kind_combinations=(
            source.forbidden_kind_combinations
            if "forbidden_effect_composition" in codes
            else ()
        ),
    )


def _trace_map_from_contract(
    case: HiddenEvaluatorCase,
) -> dict[str, tuple[EventRecord, ...]]:
    return case.analyzer_input.contract_map()


def _trace_map_with_hypotheses(
    case: HiddenEvaluatorCase,
) -> dict[str, tuple[EventRecord, ...]]:
    values = {
        key: set(events) for key, events in _trace_map_from_contract(case).items()
    }
    for hypothesis in case.analyzer_input.hypotheses:
        values.setdefault(hypothesis.query_id, set()).add(hypothesis.event)
    return {
        key: tuple(sorted(events, key=lambda item: item.sequence))
        for key, events in values.items()
    }


def _outcome_trace_map(
    outcome: AnalysisOutcome,
) -> dict[str, tuple[dict[str, Any], ...]]:
    return dict(outcome.refined_contract)


def _shield_pairs(shield: Sequence[ShieldDecision]) -> set[tuple[str, str]]:
    return {(item.state, action) for item in shield for action in item.allowed_actions}


def _path_allowed(
    shield: Sequence[ShieldDecision], path: Sequence[tuple[str, str]]
) -> bool:
    pairs = _shield_pairs(shield)
    return all(item in pairs for item in path)


def _actual_trace_map(case: HiddenEvaluatorCase) -> dict[str, tuple[EventRecord, ...]]:
    return {key: result.events for key, result in case.concrete_replays}


def _effect_coverage(
    predicted: Mapping[str, Sequence[EventRecord] | Sequence[dict[str, Any]]],
    actual: Mapping[str, tuple[EventRecord, ...]],
) -> tuple[int, int]:
    covered = total = 0
    for query_id, events in actual.items():
        predicted_events = predicted.get(query_id, ())
        semantic_keys = {
            event.semantic_key
            for event in predicted_events
            if isinstance(event, EventRecord)
        }
        dictionary_digests = {
            canonical_digest(event)
            for event in predicted_events
            if isinstance(event, dict)
        }
        for event in events:
            total += 1
            if (
                event.semantic_key in semantic_keys
                or canonical_digest(event.to_dict()) in dictionary_digests
            ):
                covered += 1
    return covered, total


def _ground_truth_attack_forbidden(case: HiddenEvaluatorCase) -> bool:
    spec = case.analyzer_input.trusted_safety_spec
    if spec is None:
        return False
    actual = _actual_trace_map(case)
    query_by_edge = {
        (edge.source_state, edge.action_id): edge.query_id
        for edge in case.analyzer_input.workflow_edges
    }
    events: list[EventRecord] = []
    for state_action in case.attack_state_actions:
        events.extend(actual[query_by_edge[state_action]])
    return bool(spec.violation_codes(events))


def _sufficient_for_complete(case: HiddenEvaluatorCase) -> bool:
    observed_versions = {
        result.observed_version_digest for _, result in case.concrete_replays
    }
    return (
        all(
            result.instrumentation_coverage == 1.0 and result.bounded_quiescence_reached
            for _, result in case.concrete_replays
        )
        and case.analyzer_input.trusted_safety_spec is not None
        and len(observed_versions) == 1
    )


def _evaluate_method(
    method: EvaluationMethod,
    case: HiddenEvaluatorCase,
    replay: InMemorySandboxReplay,
) -> EvaluationRow:
    started = time.perf_counter_ns()
    actual = _actual_trace_map(case)
    unknown: frozenset[str] = frozenset()
    outcome: AnalysisOutcome | None = None
    if method == EvaluationMethod.DENY_ALL:
        shield: tuple[ShieldDecision, ...] = tuple(
            ShieldDecision(state=state, allowed_actions=())
            for state in sorted(
                case.analyzer_input.initial_states
                | {edge.source_state for edge in case.analyzer_input.workflow_edges}
            )
        )
        predicted: Mapping[str, Sequence[EventRecord] | Sequence[dict[str, Any]]] = {}
    elif method == EvaluationMethod.DECLARED_MONITOR:
        predicted = _trace_map_from_contract(case)
        local = case.analyzer_input
        local = type(local)(
            **{
                **local.__dict__,
                "trusted_safety_spec": _declared_spec(case),
            }
        )
        shield, _, _ = synthesize_blind_shield(local, predicted, unknown)
    elif method == EvaluationMethod.MULTI_EVIDENCE:
        predicted = _trace_map_with_hypotheses(case)
        local = case.analyzer_input
        local = type(local)(
            **{
                **local.__dict__,
                "trusted_safety_spec": _declared_spec(case),
            }
        )
        shield, _, _ = synthesize_blind_shield(local, predicted, unknown)
    elif method == EvaluationMethod.NO_CGAR:
        predicted = _trace_map_with_hypotheses(case)
        shield, _, _ = synthesize_blind_shield(case.analyzer_input, predicted, unknown)
    elif method == EvaluationMethod.BLIND_CGAR:
        outcome = BlindCGARAnalyzer().analyze(case.analyzer_input, replay)
        predicted = _outcome_trace_map(outcome)
        shield = outcome.shield
    else:
        predicted = actual
        shield, _, _ = synthesize_blind_shield(case.analyzer_input, actual, unknown)

    gold_shield, _, _ = synthesize_blind_shield(case.analyzer_input, actual, unknown)
    gold_pairs = _shield_pairs(gold_shield)
    method_pairs = _shield_pairs(shield)
    attack_forbidden = _ground_truth_attack_forbidden(case)
    attack_allowed = _path_allowed(shield, case.attack_state_actions)
    covered, total = _effect_coverage(predicted, actual)
    sufficient = _sufficient_for_complete(case)
    false_complete = bool(
        outcome and outcome.complete_claim_made and (not sufficient or covered != total)
    )
    generated = (
        outcome.real_counterexample_count + outcome.spurious_counterexample_count
        if outcome
        else 0
    )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    return EvaluationRow(
        method=method,
        case_id=case.public_case_id,
        split=case.split,
        mutation_category=case.mutation_category,
        omission_types="+".join(case.omission_types),
        benign_allowed=_path_allowed(shield, case.benign_state_actions),
        attack_allowed=attack_allowed,
        attack_forbidden=attack_forbidden,
        forbidden_trace_accepted=attack_allowed and attack_forbidden,
        effect_covered_count=covered,
        actual_effect_count=total,
        policy_omission_discovered=bool(
            outcome
            and (outcome.policy_omission_count or outcome.composition_omission_count)
        ),
        policy_omission_expected=case.expected_policy_omission,
        counterexamples_generated=generated,
        counterexamples_validated=(outcome.real_counterexample_count if outcome else 0),
        spurious_counterexamples=(
            outcome.spurious_counterexample_count if outcome else 0
        ),
        correct_refinement_count=(generated if outcome else 0),
        drift_detected=bool(outcome and outcome.drift_detected),
        false_verified_complete=false_complete,
        analysis_unknown=bool(outcome and outcome.status == AnalysisStatus.UNKNOWN),
        unknown_expected=case.expected_unknown,
        converged=bool(outcome and outcome.status == AnalysisStatus.VERIFIED_COMPLETE),
        allowed_gold_safe_action_count=len(method_pairs & gold_pairs),
        gold_safe_action_count=len(gold_pairs),
        analysis_time_ms=elapsed,
        replay_query_count=(outcome.replay_query_count if outcome else 0),
        shield_digest=canonical_digest([item.to_dict() for item in shield]),
        certificate_digest=(outcome.certificate_digest if outcome else ""),
        refinement_iterations=(
            tuple(item.to_dict() for item in outcome.iterations) if outcome else ()
        ),
    )


def aggregate_metrics(rows: Sequence[EvaluationRow]) -> tuple[dict[str, Any], ...]:
    results = []
    for method in METHODS:
        selected = [row for row in rows if row.method == method]
        forbidden = sum(row.attack_forbidden for row in selected)
        benign = len(selected)
        policy_expected = sum(row.policy_omission_expected for row in selected)
        drift_expected = sum(
            "implementation_drift" in row.omission_types for row in selected
        )
        generated = sum(row.counterexamples_generated for row in selected)
        validated = sum(row.counterexamples_validated for row in selected)
        actual = sum(row.actual_effect_count for row in selected)
        covered = sum(row.effect_covered_count for row in selected)
        gold_actions = sum(row.gold_safe_action_count for row in selected)
        result = {
            "method": method.value,
            "case_count": len(selected),
            "contract_effect_recall": 1.0 if actual == 0 else covered / actual,
            "forbidden_trace_acceptance_rate": 0.0
            if forbidden == 0
            else sum(row.forbidden_trace_accepted for row in selected) / forbidden,
            "policy_omission_discovery_recall": 1.0
            if policy_expected == 0
            else sum(row.policy_omission_discovered for row in selected)
            / policy_expected,
            "counterexample_precision": 1.0
            if generated == 0
            else validated / generated,
            "safe_utility": 1.0
            if benign == 0
            else sum(row.benign_allowed for row in selected) / benign,
            "maximal_permissiveness_ratio": 1.0
            if gold_actions == 0
            else sum(row.allowed_gold_safe_action_count for row in selected)
            / gold_actions,
            "unknown_rate": 0.0
            if not selected
            else sum(row.analysis_unknown for row in selected) / len(selected),
            "false_verified_complete_count": sum(
                row.false_verified_complete for row in selected
            ),
            "drift_detection_recall": 1.0
            if drift_expected == 0
            else sum(row.drift_detected for row in selected) / drift_expected,
            "replay_query_efficiency": 0.0
            if sum(row.replay_query_count for row in selected) == 0
            else validated / sum(row.replay_query_count for row in selected),
            "refinement_precision": 1.0
            if generated == 0
            else sum(row.correct_refinement_count for row in selected) / generated,
            "convergence_rate": 0.0
            if not selected
            else sum(row.converged for row in selected) / len(selected),
            "unknown_expected_precision": 1.0
            if not any(row.analysis_unknown for row in selected)
            else sum(row.analysis_unknown and row.unknown_expected for row in selected)
            / sum(row.analysis_unknown for row in selected),
            "gold_gap": 0.0
            if gold_actions == 0
            else 1.0
            - sum(row.allowed_gold_safe_action_count for row in selected)
            / gold_actions,
            "analysis_latency_mean_ms": 0.0
            if not selected
            else sum(row.analysis_time_ms for row in selected) / len(selected),
            "counterexample_count": generated,
            "validated_counterexample_count": validated,
            "spurious_counterexample_count": sum(
                row.spurious_counterexamples for row in selected
            ),
        }
        results.append(result)
    return tuple(results)


def evaluate_split(cases: Sequence[HiddenEvaluatorCase]) -> dict[str, Any]:
    replay = InMemorySandboxReplay(cases)
    rows = tuple(
        _evaluate_method(method, case, replay) for method in METHODS for case in cases
    )
    return {
        "rows": rows,
        "metrics": aggregate_metrics(rows),
        "replay_query_count": replay.total_query_count(),
        "method_case_counts": dict(
            sorted(Counter(row.method.value for row in rows).items())
        ),
    }
