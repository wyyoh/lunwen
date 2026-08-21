from __future__ import annotations

from keyed_gram.tool_effects.benchmark import generate_feasibility_cases
from keyed_gram.tool_effects.methods import Method, evaluate_cases
from keyed_gram.tool_effects.metrics import (
    aggregate_all,
    distinguishing_categories,
)


def _metrics():
    # 开发期测试仅使用 train/calibration，development 与 locked 由冻结后的
    # 一次性正式流程分别评分。
    cases = tuple(
        case
        for case in generate_feasibility_cases()
        if case.split in {"train", "calibration"}
    )
    rows = evaluate_cases(cases)
    return rows, {item["method"]: item for item in aggregate_all(rows)}


def test_cgar_meets_all_feasibility_thresholds() -> None:
    _, metrics = _metrics()
    cgar = metrics[Method.AUTHSYNTH_CGAR.value]
    assert cgar["contract_effect_recall"] >= 0.90
    assert cgar["forbidden_trace_acceptance_rate"] == 0.0
    assert cgar["policy_omission_discovery_recall"] >= 0.80
    assert cgar["counterexample_precision"] >= 0.75
    assert cgar["safe_utility"] >= 0.85
    assert cgar["maximal_permissiveness_ratio"] >= 0.90
    assert cgar["drift_detection_recall"] == 1.0


def test_deny_all_is_safe_but_has_zero_utility() -> None:
    _, metrics = _metrics()
    deny = metrics[Method.DENY_ALL.value]
    assert deny["forbidden_trace_acceptance_rate"] == 0.0
    assert deny["safe_utility"] == 0.0
    assert deny["maximal_permissiveness_ratio"] == 0.0


def test_given_contract_monitors_accept_real_forbidden_traces() -> None:
    _, metrics = _metrics()
    assert metrics[Method.SOLVER_POLICY.value]["forbidden_trace_acceptance_rate"] > 0.0
    assert metrics[Method.TOOLGATE.value]["forbidden_trace_acceptance_rate"] > 0.0


def test_cegar_improves_over_one_shot_abstraction() -> None:
    _, metrics = _metrics()
    one_shot = metrics[Method.AUTHSYNTH_NO_CEGAR.value]
    cgar = metrics[Method.AUTHSYNTH_CGAR.value]
    assert cgar["contract_effect_recall"] > one_shot["contract_effect_recall"]
    assert (
        cgar["forbidden_trace_acceptance_rate"]
        < one_shot["forbidden_trace_acceptance_rate"]
    )
    assert cgar["safe_utility"] > one_shot["safe_utility"]


def test_six_mutation_categories_are_distinguishing() -> None:
    rows, _ = _metrics()
    assert len(distinguishing_categories(rows)) == 6


def test_cgar_matches_gold_safe_action_set() -> None:
    rows, metrics = _metrics()
    assert metrics[Method.AUTHSYNTH_CGAR.value]["exact_shield_match_rate"] == 1.0
    assert all(
        row.exact_shield_match for row in rows if row.method == Method.AUTHSYNTH_CGAR
    )
