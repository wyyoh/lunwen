"""不访问 concrete implementation 或 gold 标签的真实迭代 CGAR。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from keyed_gram.authsynth_shared import (
    AnalysisStatus,
    AnalyzerCaseInput,
    EventRecord,
    FindingKind,
    QuerySpec,
    canonical_digest,
)

from .models import AnalysisOutcome, RefinementIteration
from .replay import ReplayClient
from .shield import synthesize_blind_shield


@dataclass
class _AbstractModel:
    traces: dict[str, tuple[EventRecord, ...]]
    unknown_queries: set[str]

    def digest(self) -> str:
        return canonical_digest(
            {
                "traces": {
                    key: [event.to_dict() for event in value]
                    for key, value in sorted(self.traces.items())
                },
                "unknown_queries": sorted(self.unknown_queries),
            }
        )


def _event_keys(events: tuple[EventRecord, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(event.semantic_key for event in events)


def _candidate_digest(
    query: QuerySpec,
    predicted: tuple[EventRecord, ...],
    source: str,
) -> str:
    return canonical_digest(
        {
            "query": query.to_dict(),
            "predicted_trace": [event.to_dict() for event in predicted],
            "candidate_source": source,
        }
    )


class BlindCGARAnalyzer:
    """以有限 replay 查询逐步修正 contract、policy guard 与 Shield。"""

    def analyze(self, case: AnalyzerCaseInput, replay: ReplayClient) -> AnalysisOutcome:
        contract = case.contract_map()
        hypotheses: dict[str, list[EventRecord]] = defaultdict(list)
        for item in case.hypotheses:
            hypotheses[item.query_id].append(item.event)
        initial: dict[str, tuple[EventRecord, ...]] = {}
        for query in case.query_catalog:
            events = {
                *contract.get(query.query_id, ()),
                *hypotheses.get(query.query_id, ()),
            }
            initial[query.query_id] = tuple(
                sorted(events, key=lambda item: item.sequence)
            )
        model = _AbstractModel(
            traces=initial,
            unknown_queries={query.query_id for query in case.query_catalog},
        )
        observed: dict[str, tuple[EventRecord, ...]] = {}
        iterations: list[RefinementIteration] = []
        real_count = spurious_count = policy_count = composition_count = 0
        drift = False
        observed_versions: set[str] = set()
        reason_codes: set[str] = set()

        def priority(query: QuerySpec) -> tuple[int, str]:
            predicted = model.traces.get(query.query_id, ())
            risky = bool(
                case.trusted_safety_spec
                and case.trusted_safety_spec.violation_codes(predicted)
            )
            return (0 if risky else 1, query.query_id)

        ordered_queries = sorted(case.query_catalog, key=priority)
        for query in ordered_queries[: case.replay_budget]:
            model_before = model.digest()
            _, shield_before, _ = synthesize_blind_shield(
                case, observed, frozenset(model.unknown_queries)
            )
            predicted = model.traces.get(query.query_id, ())
            candidate = _candidate_digest(
                query,
                predicted,
                "abstract_model_check"
                if hypotheses.get(query.query_id)
                else "coverage_guided_replay",
            )
            result = replay.replay(case.case_handle, query)
            if result.query_id != query.query_id:
                raise RuntimeError("replay result/query 不匹配")
            actual = result.events
            observed_versions.add(result.observed_version_digest)
            observed[query.query_id] = actual
            model.traces[query.query_id] = actual
            model.unknown_queries.discard(query.query_id)

            predicted_keys = set(_event_keys(predicted))
            actual_keys = set(_event_keys(actual))
            missing = actual_keys - predicted_keys
            impossible = predicted_keys - actual_keys
            finding = FindingKind.SPURIOUS_ABSTRACTION
            patch: dict[str, Any] = {
                "operation": "confirm_trace",
                "query_id": query.query_id,
            }
            if result.observed_version_digest != case.expected_version_digest:
                finding = FindingKind.IMPLEMENTATION_DRIFT
                drift = True
                reason_codes.add("implementation_version_drift")
                patch = {
                    "operation": "invalidate_and_rebind_version_after_full_coverage",
                    "query_id": query.query_id,
                    "observed_version_digest": result.observed_version_digest,
                }
                real_count += 1
            elif not result.bounded_quiescence_reached:
                finding = FindingKind.QUIESCENCE_UNKNOWN
                reason_codes.add("bounded_quiescence_not_reached")
                model.unknown_queries.add(query.query_id)
                patch = {
                    "operation": "retain_unknown_async_edge",
                    "query_id": query.query_id,
                }
            elif result.instrumentation_coverage < 1.0:
                finding = FindingKind.COVERAGE_INCOMPLETE
                reason_codes.add("instrumentation_coverage_incomplete")
                model.unknown_queries.add(query.query_id)
                patch = {
                    "operation": "retain_unknown_coverage_edge",
                    "query_id": query.query_id,
                }
            elif missing:
                finding = FindingKind.REAL_CONTRACT_OMISSION
                real_count += 1
                patch = {
                    "operation": "add_observed_call_trace",
                    "query_id": query.query_id,
                    "added_event_digests": sorted(
                        canonical_digest(item.to_dict())
                        for item in actual
                        if item.semantic_key in missing
                    ),
                }
            elif impossible:
                finding = FindingKind.SPURIOUS_ABSTRACTION
                spurious_count += 1
                patch = {
                    "operation": "remove_spurious_hypothesis",
                    "query_id": query.query_id,
                    "removed_event_count": len(impossible),
                }

            if case.trusted_safety_spec is None and actual:
                finding = FindingKind.SECURITY_CLASSIFICATION_UNKNOWN
                reason_codes.add("trusted_safety_spec_absent")
                model.unknown_queries.add(query.query_id)
            elif case.trusted_safety_spec is not None:
                violations = case.trusted_safety_spec.violation_codes(actual)
                undeclared = set(violations) - set(case.declared_policy_codes)
                if undeclared:
                    policy_count += 1
                    finding = FindingKind.REAL_POLICY_OMISSION
                    patch = {
                        "operation": "add_runtime_policy_guard",
                        "query_id": query.query_id,
                        "reason_codes": sorted(undeclared),
                    }
                    real_count += int(not missing)

            # 组合 violation 只能在累计路径上发现；逐调用均安全时单独标注。
            if case.trusted_safety_spec is not None:
                shield_now, _, _ = synthesize_blind_shield(
                    case, observed, frozenset(model.unknown_queries)
                )
                individually_safe = not case.trusted_safety_spec.violation_codes(actual)
                denied_states = {
                    item.state
                    for item in shield_now
                    if not item.allowed_actions
                    and item.state not in case.terminal_states
                }
                if individually_safe and denied_states and not model.unknown_queries:
                    composition_count += 1
                    finding = FindingKind.REAL_COMPOSITION_OMISSION
                    patch = {
                        "operation": "add_composition_guard",
                        "denied_state_digests": sorted(
                            canonical_digest(item) for item in denied_states
                        ),
                    }

            model_after = model.digest()
            _, shield_after, _ = synthesize_blind_shield(
                case, observed, frozenset(model.unknown_queries)
            )
            iterations.append(
                RefinementIteration(
                    iteration=len(iterations) + 1,
                    query_id=query.query_id,
                    abstract_counterexample_digest=candidate,
                    replay_result_digest=canonical_digest(result.to_dict()),
                    finding_kind=finding,
                    refinement_patch_digest=canonical_digest(patch),
                    model_before_digest=model_before,
                    model_after_digest=model_after,
                    shield_before_digest=shield_before,
                    shield_after_digest=shield_after,
                    remaining_unknown_query_count=len(model.unknown_queries),
                )
            )

        if model.unknown_queries:
            reason_codes.add("finite_query_domain_not_fully_observed")
        version_consistent = len(observed_versions) == 1
        if not version_consistent:
            reason_codes.add("observed_version_domain_inconsistent")
        trace_complete = not model.unknown_queries and all(
            _event_keys(model.traces[query_id]) == _event_keys(events)
            for query_id, events in observed.items()
        )
        call_complete = trace_complete and len(observed) == len(case.query_catalog)
        shield, shield_digest, initial_safe = synthesize_blind_shield(
            case, observed, frozenset(model.unknown_queries)
        )
        composition_complete = (
            call_complete
            and case.trusted_safety_spec is not None
            and version_consistent
        )
        verified = call_complete and trace_complete and composition_complete
        status = (
            AnalysisStatus.VERIFIED_COMPLETE if verified else AnalysisStatus.UNKNOWN
        )
        if not initial_safe:
            reason_codes.add("no_safe_initial_winning_state")
        refined = tuple(
            (
                query_id,
                tuple(event.to_dict() for event in events),
            )
            for query_id, events in sorted(model.traces.items())
        )
        certificate = canonical_digest(
            {
                "public_case_id": case.public_case_id,
                "contract": refined,
                "shield_digest": shield_digest,
                "status": status.value,
                "effective_version_digest": (
                    next(iter(observed_versions))
                    if version_consistent
                    else "inconsistent"
                ),
            }
        )
        return AnalysisOutcome(
            public_case_id=case.public_case_id,
            status=status,
            refined_contract=refined,
            shield=shield,
            iterations=tuple(iterations),
            replay_query_count=len(iterations),
            real_counterexample_count=real_count,
            spurious_counterexample_count=spurious_count,
            policy_omission_count=policy_count,
            composition_omission_count=composition_count,
            drift_detected=drift,
            call_complete=call_complete,
            trace_complete=trace_complete,
            composition_complete=composition_complete,
            complete_claim_made=verified,
            reason_codes=tuple(sorted(reason_codes)),
            certificate_digest=certificate,
        )
