from __future__ import annotations

from keyed_gram.authority_flow.counterexamples import counterexample_registry
from keyed_gram.authority_flow.explorer import (
    find_authority_laundering,
    find_naive_merge_amplification,
    run_bounded_model,
)


def test_bounded_model_finds_laundering_only_in_unsafe_semantics() -> None:
    result = find_authority_laundering(bound=4)
    assert result.unsafe_counterexample_found is True
    assert result.safe_counterexample_found is False
    assert result.unsafe_trace == (
        "llm_summary",
        "trusted_tool_echo_marks_trusted",
        "derive_authority_from_integrity",
    )


def test_bounded_model_finds_naive_merge_amplification() -> None:
    result = find_naive_merge_amplification(bound=4)
    assert result.unsafe_counterexample_found is True
    assert result.safe_counterexample_found is False
    assert result.unsafe_trace[0] == "naive_branch_copy"


def test_bounded_model_summary_has_both_required_counterexamples() -> None:
    result = run_bounded_model()
    assert result["bounded_model_counterexample_for_naive_merge"] is True
    assert (
        result["bounded_model_counterexample_for_authority_laundering"] is True
    )


def test_six_counterexamples_are_distinguishing() -> None:
    cases = counterexample_registry()
    assert len(cases) >= 6
    assert len({item.case_id for item in cases}) == len(cases)
    assert all(item.not_plain_taint_only for item in cases)
    assert all(item.not_token_subset_only for item in cases)
    assert {
        "authority_origination",
        "authority_conservation",
        "non_malleable_authority",
        "merge_confinement",
    }.issubset({item.violated_property for item in cases})
