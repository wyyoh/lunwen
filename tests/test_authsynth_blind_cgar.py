from __future__ import annotations

from dataclasses import replace

from keyed_gram.authsynth_analyzer import BlindCGARAnalyzer
from keyed_gram.authsynth_shared import (
    AnalysisStatus,
    AnalyzerCaseInput,
    EventRecord,
    EvidenceHypothesis,
    EvidenceOrigin,
    HighLevelSafetySpec,
    QuerySpec,
    ReplayResult,
    WorkflowEdge,
    canonical_digest,
)


class FakeReplay:
    def __init__(self, results: dict[str, ReplayResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def replay(self, case_handle: str, query: QuerySpec) -> ReplayResult:
        assert case_handle == "opaque-case"
        self.queries.append(query.query_id)
        return self.results[query.query_id]


def event(
    event_id: str,
    call_id: str,
    kind: str,
    *,
    sequence: int = 0,
    destination: str | None = None,
) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        call_id=call_id,
        sequence=sequence,
        phase="immediate",
        kind=kind,
        tenant_id="tenant-a",
        resource_id=f"resource-{call_id}",
        destination_id=destination,
    )


def result(
    query_id: str,
    events: tuple[EventRecord, ...],
    *,
    coverage: float = 1.0,
    quiescent: bool = True,
    version: str = "version-digest-1",
) -> ReplayResult:
    return ReplayResult(
        query_id=query_id,
        events=events,
        state_diff_digest=canonical_digest([item.to_dict() for item in events]),
        exit_status="ok",
        bounded_quiescence_reached=quiescent,
        instrumentation_coverage=coverage,
        observed_version_digest=version,
    )


def base_case(
    *, budget: int = 2, include_hypothesis: bool = False
) -> AnalyzerCaseInput:
    q1 = QuerySpec("q1", "tool-a", "call-1", "v1", "env-1", ())
    q2 = QuerySpec("q2", "tool-b", "call-2", "v1", "env-2", ())
    declared_read = event("declared-read", "call-1", "read")
    declared_write = event("declared-write", "call-2", "write")
    hypotheses = ()
    if include_hypothesis:
        hypotheses = (
            EvidenceHypothesis(
                "hyp-1",
                "q1",
                event("hyp-send", "call-1", "send", destination="external"),
                EvidenceOrigin.STATIC,
            ),
        )
    return AnalyzerCaseInput(
        case_handle="opaque-case",
        public_case_id="public-case",
        declared_contract=(("q1", (declared_read,)), ("q2", (declared_write,))),
        declared_policy_codes=frozenset(),
        trusted_safety_spec=HighLevelSafetySpec(
            "spec-1",
            "tenant-a",
            forbidden_external_destinations=frozenset({"external"}),
            forbidden_kind_combinations=(("read", "send"),),
        ),
        query_catalog=(q1, q2),
        workflow_edges=(
            WorkflowEdge("root", "read", "q1", "after-read"),
            WorkflowEdge("after-read", "write", "q2", "done"),
        ),
        initial_states=frozenset({"root"}),
        terminal_states=frozenset({"done"}),
        hypotheses=hypotheses,
        replay_budget=budget,
        expected_version_digest="version-digest-1",
    )


def test_real_loop_refines_contract_and_records_shield_delta() -> None:
    case = base_case()
    read = event("observed-read", "call-1", "read")
    hidden = event("hidden-send", "call-2", "send", destination="external")
    replay = FakeReplay({"q1": result("q1", (read,)), "q2": result("q2", (hidden,))})
    outcome = BlindCGARAnalyzer().analyze(case, replay)
    assert outcome.status == AnalysisStatus.VERIFIED_COMPLETE
    assert outcome.real_counterexample_count >= 1
    assert outcome.policy_omission_count == 1
    assert outcome.replay_query_count == 2
    assert (
        outcome.iterations[1].model_before_digest
        != outcome.iterations[1].model_after_digest
    )
    assert outcome.complete_claim_made is True


def test_spurious_candidate_is_identified_by_replay_not_label() -> None:
    case = base_case(include_hypothesis=True)
    read = event("observed-read", "call-1", "read")
    write = event("observed-write", "call-2", "write")
    replay = FakeReplay({"q1": result("q1", (read,)), "q2": result("q2", (write,))})
    outcome = BlindCGARAnalyzer().analyze(case, replay)
    assert replay.queries[0] == "q1"
    assert outcome.spurious_counterexample_count == 1
    assert outcome.status == AnalysisStatus.VERIFIED_COMPLETE


def test_incomplete_coverage_returns_unknown_and_never_claims_complete() -> None:
    case = base_case(budget=1)
    read = event("observed-read", "call-1", "read")
    replay = FakeReplay({"q1": result("q1", (read,))})
    outcome = BlindCGARAnalyzer().analyze(case, replay)
    assert outcome.status == AnalysisStatus.UNKNOWN
    assert outcome.complete_claim_made is False
    assert "finite_query_domain_not_fully_observed" in outcome.reason_codes


def test_instrumentation_or_quiescence_gap_returns_unknown() -> None:
    case = base_case()
    read = event("observed-read", "call-1", "read")
    write = event("observed-write", "call-2", "write")
    replay = FakeReplay(
        {
            "q1": result("q1", (read,), coverage=0.75),
            "q2": result("q2", (write,), quiescent=False),
        }
    )
    outcome = BlindCGARAnalyzer().analyze(case, replay)
    assert outcome.status == AnalysisStatus.UNKNOWN
    assert set(outcome.reason_codes) >= {
        "instrumentation_coverage_incomplete",
        "bounded_quiescence_not_reached",
    }


def test_version_drift_invalidates_old_certificate_then_rebinds_consistent_version() -> (
    None
):
    case = base_case()
    read = event("observed-read", "call-1", "read")
    write = event("observed-write", "call-2", "write")
    replay = FakeReplay(
        {
            "q1": result("q1", (read,), version="version-digest-2"),
            "q2": result("q2", (write,), version="version-digest-2"),
        }
    )
    outcome = BlindCGARAnalyzer().analyze(case, replay)
    assert outcome.drift_detected is True
    assert outcome.status == AnalysisStatus.VERIFIED_COMPLETE
    assert outcome.complete_claim_made is True
    assert "implementation_version_drift" in outcome.reason_codes


def test_absent_high_level_safety_spec_cannot_be_called_policy_discovery() -> None:
    case = replace(base_case(), trusted_safety_spec=None)
    read = event("observed-read", "call-1", "read")
    write = event("observed-write", "call-2", "write")
    replay = FakeReplay({"q1": result("q1", (read,)), "q2": result("q2", (write,))})
    outcome = BlindCGARAnalyzer().analyze(case, replay)
    assert outcome.status == AnalysisStatus.UNKNOWN
    assert outcome.policy_omission_count == 0
    assert "trusted_safety_spec_absent" in outcome.reason_codes
