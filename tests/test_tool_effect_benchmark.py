from __future__ import annotations

from collections import Counter

from keyed_gram.tool_effects.benchmark import (
    MUTATIONS,
    generate_feasibility_cases,
    split_cases,
    split_collision_audit,
)
from keyed_gram.tool_effects.ir import execute_trace
from keyed_gram.tool_effects.types import OmissionType


def test_benchmark_has_exact_preregistered_size() -> None:
    cases = generate_feasibility_cases()
    assert len(cases) == 150
    assert len({case.case_id for case in cases}) == 150
    assert len({case.base_tool_family for case in cases}) == 8
    counts = Counter(case.mutation_category for case in cases)
    assert all(counts[category] == 24 for category in MUTATIONS)
    assert counts["clean_control"] == 6


def test_all_split_isolation_dimensions_have_zero_collision() -> None:
    audit = split_collision_audit(split_cases(generate_feasibility_cases()))
    assert audit["cross_split_collision_count"] == 0


def test_formal_seed_changes_exact_cases_but_not_preregistered_shape() -> None:
    first = generate_feasibility_cases("freeze-commit-a")
    second = generate_feasibility_cases("freeze-commit-b")
    assert {case.case_id for case in first}.isdisjoint(
        {case.case_id for case in second}
    )
    assert Counter(case.mutation_category for case in first) == Counter(
        case.mutation_category for case in second
    )


def test_mutation_ground_truth_comes_from_concrete_transition() -> None:
    # 单元测试不得在 analyzer freeze 前运行 development/locked oracle。
    cases = (
        case
        for case in generate_feasibility_cases()
        if case.split in {"train", "calibration"}
    )
    for case in cases:
        effects = execute_trace(
            case.implementation_map(), case.attack_trace, drain_async=True
        )
        forbidden = any(rule.violated(effects) for rule in case.safety_rules)
        assert forbidden == case.attack_trace.expected_forbidden


def test_contract_and_policy_omissions_are_semantically_separate() -> None:
    cases = generate_feasibility_cases()
    parameter_role = [
        case for case in cases if case.mutation_category == "parameter_role_omission"
    ]
    state_policy = [
        case
        for case in cases
        if case.mutation_category == "state_dependent_effect"
        and case.omission_type == OmissionType.POLICY_OMISSION
    ]
    assert all(
        case.omission_type == OmissionType.CONTRACT_OMISSION for case in parameter_role
    )
    assert len(state_policy) == 8


def test_public_rows_exclude_effect_values_and_llm_judgment() -> None:
    row = generate_feasibility_cases()[0].to_public_dict()
    assert row["llm_judge_used"] is False
    assert row["public_or_synthetic_only"] is True
    assert "natural_language_request" not in row
    assert "effects" not in row
    assert "source_code" not in row
