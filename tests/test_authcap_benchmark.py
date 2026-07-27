from __future__ import annotations

import pytest

from keyed_gram.authcap_benchmark import (
    ATTACK_CATEGORIES,
    DOMAINS,
    SPLITS,
    generate_authzroutebench,
    validate_benchmark_case,
)
from keyed_gram.stage_f1_metrics import benchmark_quality_metrics


def test_formal_benchmark_has_preregistered_scale() -> None:
    splits = generate_authzroutebench()
    metrics = benchmark_quality_metrics(splits)
    assert set(splits) == set(SPLITS)
    assert all(len(rows) == 360 for rows in splits.values())
    assert metrics["case_count"] == 1440
    assert metrics["family_count"] == 144
    assert metrics["domain_count"] == len(DOMAINS) == 3
    assert metrics["attack_category_count"] == len(ATTACK_CATEGORIES) == 12


def test_each_split_has_all_domains_and_categories() -> None:
    splits = generate_authzroutebench()
    for rows in splits.values():
        assert {row["domain"] for row in rows} == set(DOMAINS)
        assert {row["attack_category"] for row in rows} == set(
            ATTACK_CATEGORIES
        )


def test_ai_surface_is_separate_from_oracle_ground_truth() -> None:
    case = generate_authzroutebench(splits=("train",))["train"][0]
    assert (
        case["metadata"]["surface_text_generation"]
        == "AI_assisted_deterministic_templates"
    )
    assert case["metadata"]["ground_truth_source"] == (
        "deterministic_policy_oracle"
    )
    assert "natural_language_request" not in case["policy_set"]


def test_required_attack_counts_exceed_gates() -> None:
    metrics = benchmark_quality_metrics(generate_authzroutebench())
    assert metrics["multi_step_workflow_count"] >= 100
    assert metrics["explicit_deny_case_count"] >= 150
    assert metrics["cross_tenant_case_count"] >= 100
    assert metrics["stale_policy_case_count"] >= 100


def test_locked_split_is_generated_but_never_scored() -> None:
    locked = generate_authzroutebench()["locked_test"]
    assert locked
    assert all(row["split"] == "locked_test" for row in locked)
    assert all(
        row["metadata"]["ground_truth_source"]
        == "deterministic_policy_oracle"
        for row in locked
    )


def test_unknown_case_field_is_rejected() -> None:
    case = generate_authzroutebench(splits=("train",))["train"][0]
    case["unknown"] = True
    with pytest.raises(ValueError, match="schema"):
        validate_benchmark_case(case)


def test_proposal_cannot_smuggle_principal_or_capability() -> None:
    case = generate_authzroutebench(splits=("train",))["train"][0]
    case["semantic_proposal_candidates"][0]["principal"] = "admin"
    with pytest.raises(ValueError, match="authority"):
        validate_benchmark_case(case)
