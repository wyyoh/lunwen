from __future__ import annotations

import pytest

from keyed_gram.authsynth_shared import (
    AnalyzerCaseInput,
    EventRecord,
    HighLevelSafetySpec,
    QuerySpec,
    WorkflowEdge,
    canonical_digest,
)
from keyed_gram.authsynth_shared.schema import SchemaError


def event(event_id: str = "e1", *, sequence: int = 0) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        call_id="call-1",
        sequence=sequence,
        phase="immediate",
        kind="read",
        tenant_id="tenant-a",
        resource_id="doc-1",
        argument_roles=(("resource", "doc-1"),),
    )


def test_event_digest_is_deterministic_and_order_sensitive() -> None:
    first = event("e1", sequence=0)
    second = event("e2", sequence=1)
    assert canonical_digest([first.to_dict(), second.to_dict()]) == canonical_digest(
        [first.to_dict(), second.to_dict()]
    )
    assert canonical_digest([first.to_dict(), second.to_dict()]) != canonical_digest(
        [second.to_dict(), first.to_dict()]
    )


def test_schema_rejects_wildcard_and_duplicate_query() -> None:
    with pytest.raises(SchemaError, match="wildcard"):
        QuerySpec("q1", "tool-*", "call-1", "v1", "env-1", ())
    query = QuerySpec("q1", "tool-1", "call-1", "v1", "env-1", ())
    with pytest.raises(SchemaError, match="重复 query_id"):
        AnalyzerCaseInput(
            case_handle="case-handle",
            public_case_id="case-public",
            declared_contract=(),
            declared_policy_codes=frozenset(),
            trusted_safety_spec=HighLevelSafetySpec("spec-1", "tenant-a"),
            query_catalog=(query, query),
            workflow_edges=(),
            initial_states=frozenset({"root"}),
            terminal_states=frozenset({"done"}),
            hypotheses=(),
            replay_budget=2,
            expected_version_digest="version-1",
        )


def test_high_level_spec_detects_cross_tenant_and_composition() -> None:
    spec = HighLevelSafetySpec(
        "spec-1",
        "tenant-a",
        forbidden_kind_combinations=(("read", "send"),),
    )
    read = event()
    send = EventRecord(
        event_id="e2",
        call_id="call-2",
        sequence=1,
        phase="immediate",
        kind="send",
        tenant_id="tenant-b",
        resource_id="mail-1",
    )
    assert spec.violation_codes((read, send)) == (
        "cross_tenant_effect",
        "forbidden_effect_composition",
    )


def test_workflow_must_reference_catalog_query() -> None:
    query = QuerySpec("q1", "tool-1", "call-1", "v1", "env-1", ())
    with pytest.raises(SchemaError, match="未知 query"):
        AnalyzerCaseInput(
            case_handle="case-handle",
            public_case_id="case-public",
            declared_contract=(),
            declared_policy_codes=frozenset(),
            trusted_safety_spec=HighLevelSafetySpec("spec-1", "tenant-a"),
            query_catalog=(query,),
            workflow_edges=(WorkflowEdge("root", "go", "q2", "done"),),
            initial_states=frozenset({"root"}),
            terminal_states=frozenset({"done"}),
            hypotheses=(),
            replay_budget=1,
            expected_version_digest="version-1",
        )
