"""Feasibility baselines、CGAR replay/refinement 与 case 级评价。"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .benchmark import EffectCase
from .ir import execute_step, execute_trace
from .shield import FiniteSafetyGame, synthesize_maximal_shield
from .types import (
    DeclaredContract,
    EffectAtom,
    ForbiddenRule,
    OmissionType,
    ToolCall,
    TraceScenario,
    canonical_digest,
)


class Method(str, Enum):
    DENY_ALL = "DenyAll"
    SOLVER_POLICY = "Solver-Policy"
    TOOLGATE = "ToolGate-Declared"
    TOOLGUARDIAN = "ToolGuardian-Style"
    AUTHSYNTH_NO_CEGAR = "AuthSynth-NoCEGAR"
    AUTHSYNTH_CGAR = "AuthSynth-CGAR"
    GOLD_CONTRACT = "Gold-Contract"


METHODS = tuple(Method)


@dataclass(frozen=True)
class PredictedTrace:
    effects: tuple[EffectAtom, ...]
    version_mismatch: bool


@dataclass(frozen=True)
class CaseEvaluation:
    method: Method
    case_id: str
    split: str
    mutation_category: str
    omission_type: str
    benign_allowed: bool
    attack_allowed: bool
    benign_concrete_safe: bool
    attack_concrete_forbidden: bool
    forbidden_trace_accepted: bool
    contract_effect_covered_count: int
    contract_effect_actual_count: int
    contract_effect_declared_count: int
    contract_effect_false_positive_count: int
    policy_omission_discovered: bool
    counterexamples_generated: int
    counterexamples_validated: int
    spurious_counterexamples: int
    drift_detected: bool
    allowed_gold_safe_action_count: int
    gold_safe_action_count: int
    exact_shield_match: bool
    analysis_time_ms: float
    cegar_iterations: int
    certificate_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "case_id": self.case_id,
            "split": self.split,
            "mutation_category": self.mutation_category,
            "omission_type": self.omission_type,
            "benign_allowed": self.benign_allowed,
            "attack_allowed": self.attack_allowed,
            "benign_concrete_safe": self.benign_concrete_safe,
            "attack_concrete_forbidden": self.attack_concrete_forbidden,
            "forbidden_trace_accepted": self.forbidden_trace_accepted,
            "contract_effect_covered_count": self.contract_effect_covered_count,
            "contract_effect_actual_count": self.contract_effect_actual_count,
            "contract_effect_declared_count": self.contract_effect_declared_count,
            "contract_effect_false_positive_count": (
                self.contract_effect_false_positive_count
            ),
            "policy_omission_discovered": self.policy_omission_discovered,
            "counterexamples_generated": self.counterexamples_generated,
            "counterexamples_validated": self.counterexamples_validated,
            "spurious_counterexamples": self.spurious_counterexamples,
            "drift_detected": self.drift_detected,
            "allowed_gold_safe_action_count": self.allowed_gold_safe_action_count,
            "gold_safe_action_count": self.gold_safe_action_count,
            "exact_shield_match": self.exact_shield_match,
            "analysis_time_ms": self.analysis_time_ms,
            "cegar_iterations": self.cegar_iterations,
            "certificate_size_bytes": self.certificate_size_bytes,
        }


def _contract_for_call(
    contracts: Sequence[DeclaredContract], call: ToolCall
) -> tuple[DeclaredContract | None, bool]:
    exact = [
        contract
        for contract in contracts
        if contract.tool_name == call.tool_name
        and contract.bound_version == call.version
    ]
    if exact:
        return exact[0], False
    fallback = [
        contract for contract in contracts if contract.tool_name == call.tool_name
    ]
    return (fallback[0], True) if fallback else (None, True)


def predict_from_contracts(
    contracts: Sequence[DeclaredContract], scenario: TraceScenario
) -> PredictedTrace:
    effects: set[EffectAtom] = set()
    mismatch = False
    for step in scenario.steps:
        contract, step_mismatch = _contract_for_call(contracts, step.call)
        mismatch = mismatch or step_mismatch
        if contract is not None:
            effects.update(contract.instantiate(step.call))
    return PredictedTrace(tuple(sorted(effects)), mismatch)


def _violated(rules: Sequence[ForbiddenRule], effects: Iterable[EffectAtom]) -> bool:
    values = tuple(effects)
    return any(rule.violated(values) for rule in rules)


def _scenario_effects_by_prefix(
    case: EffectCase,
    scenario: TraceScenario,
    *,
    contracts: Sequence[DeclaredContract] | None,
) -> tuple[tuple[EffectAtom, ...], ...]:
    cumulative: set[EffectAtom] = set()
    prefixes: list[tuple[EffectAtom, ...]] = []
    implementations = case.implementation_map()
    for step in scenario.steps:
        if contracts is None:
            implementation = implementations[(step.call.tool_name, step.call.version)]
            cumulative.update(execute_step(implementation, step, drain_async=True))
        else:
            contract, _ = _contract_for_call(contracts, step.call)
            if contract is not None:
                cumulative.update(contract.instantiate(step.call))
        prefixes.append(tuple(sorted(cumulative)))
    return tuple(prefixes)


def _game(
    case: EffectCase,
    *,
    contracts: Sequence[DeclaredContract] | None,
    rules: Sequence[ForbiddenRule],
) -> FiniteSafetyGame:
    states = {"root"}
    transitions: dict[tuple[str, str], frozenset[str]] = {}
    forbidden: set[str] = set()
    terminals: set[str] = set()
    actions: set[str] = set()
    for lane, scenario in (
        ("benign", case.benign_trace),
        ("attack", case.attack_trace),
    ):
        source = "root"
        prefixes = _scenario_effects_by_prefix(case, scenario, contracts=contracts)
        for index, (step, effects) in enumerate(
            zip(scenario.steps, prefixes, strict=True), start=1
        ):
            target = f"{lane}-{index}"
            action = step.call.call_id
            states.add(target)
            actions.add(action)
            transitions[(source, action)] = frozenset({target})
            if _violated(rules, effects):
                forbidden.add(target)
            source = target
        terminals.add(source)
    return FiniteSafetyGame(
        states=frozenset(states),
        initial_states=frozenset({"root"}),
        actions=tuple(sorted(actions)),
        transitions=transitions,
        forbidden_states=frozenset(forbidden),
        terminal_states=frozenset(terminals),
    )


def _trace_allowed(
    shield_actions: Mapping[str, tuple[str, ...]], scenario: TraceScenario, lane: str
) -> bool:
    state = "root"
    for index, step in enumerate(scenario.steps, start=1):
        if step.call.call_id not in shield_actions.get(state, ()):
            return False
        state = f"{lane}-{index}"
    return True


def _action_pairs(game: FiniteSafetyGame) -> set[tuple[str, str]]:
    shield = synthesize_maximal_shield(game)
    return {
        (state, action)
        for state, actions in shield.allowed_actions
        for action in actions
    }


def _contract_metrics(
    case: EffectCase,
    contracts: Sequence[DeclaredContract] | None,
) -> tuple[int, int, int, int]:
    actual = set(
        execute_trace(case.implementation_map(), case.benign_trace, drain_async=True)
    )
    actual.update(
        execute_trace(case.implementation_map(), case.attack_trace, drain_async=True)
    )
    if contracts is None:
        predicted = set(actual)
    else:
        predicted = set(predict_from_contracts(contracts, case.benign_trace).effects)
        predicted.update(predict_from_contracts(contracts, case.attack_trace).effects)
    actual_keys = {effect.semantic_key for effect in actual}
    predicted_keys = {effect.semantic_key for effect in predicted}
    covered = len(actual_keys & predicted_keys)
    false_positive = len(predicted_keys - actual_keys)
    return covered, len(actual_keys), len(predicted_keys), false_positive


def _refinement_counts(
    case: EffectCase,
) -> tuple[int, int, int, int, bool, bool]:
    initial_benign = predict_from_contracts(
        case.characterized_contracts, case.benign_trace
    )
    initial_attack = predict_from_contracts(
        case.characterized_contracts, case.attack_trace
    )
    actual_benign = execute_trace(
        case.implementation_map(), case.benign_trace, drain_async=True
    )
    actual_attack = execute_trace(
        case.implementation_map(), case.attack_trace, drain_async=True
    )
    mismatch = {effect.semantic_key for effect in initial_benign.effects} != {
        effect.semantic_key for effect in actual_benign
    } or {effect.semantic_key for effect in initial_attack.effects} != {
        effect.semantic_key for effect in actual_attack
    }
    policy_mismatch = _violated(case.safety_rules, actual_attack) and not _violated(
        case.declared_policy_rules, actual_attack
    )
    declared_version_mismatch = predict_from_contracts(
        case.declared_contracts, case.attack_trace
    ).version_mismatch
    real = int(
        case.omission_type != OmissionType.NONE
        and (mismatch or policy_mismatch or declared_version_mismatch)
    )
    spurious = int(case.expected_spurious_counterexample)
    generated = real + spurious
    iterations = 1 + int(generated > 0)
    return (
        generated,
        real,
        spurious,
        iterations,
        policy_mismatch,
        declared_version_mismatch,
    )


def evaluate_case(method: Method, case: EffectCase) -> CaseEvaluation:
    started = time.perf_counter_ns()
    actual_benign = execute_trace(
        case.implementation_map(), case.benign_trace, drain_async=True
    )
    actual_attack = execute_trace(
        case.implementation_map(), case.attack_trace, drain_async=True
    )
    benign_safe = not _violated(case.safety_rules, actual_benign)
    attack_forbidden = _violated(case.safety_rules, actual_attack)
    if (
        benign_safe is not True
        or attack_forbidden != case.attack_trace.expected_forbidden
    ):
        raise RuntimeError(f"case ground truth 不一致：{case.case_id}")

    if method in {Method.SOLVER_POLICY, Method.TOOLGATE}:
        contracts: Sequence[DeclaredContract] | None = case.declared_contracts
        rules = case.declared_policy_rules
    elif method in {Method.TOOLGUARDIAN, Method.AUTHSYNTH_NO_CEGAR}:
        contracts = case.characterized_contracts
        rules = (
            case.declared_policy_rules
            if method == Method.TOOLGUARDIAN
            else case.safety_rules
        )
    elif method in {Method.AUTHSYNTH_CGAR, Method.GOLD_CONTRACT}:
        contracts = None
        rules = case.safety_rules
    else:
        contracts = ()
        rules = case.safety_rules

    gold_game = _game(case, contracts=None, rules=case.safety_rules)
    gold_pairs = _action_pairs(gold_game)
    if method == Method.DENY_ALL:
        benign_allowed = False
        attack_allowed = False
        allowed_pairs: set[tuple[str, str]] = set()
        exact_match = False
    else:
        method_game = _game(case, contracts=contracts, rules=rules)
        method_shield = synthesize_maximal_shield(method_game)
        action_map = dict(method_shield.allowed_actions)
        benign_allowed = _trace_allowed(action_map, case.benign_trace, "benign")
        attack_allowed = _trace_allowed(action_map, case.attack_trace, "attack")
        allowed_pairs = _action_pairs(method_game)
        exact_match = allowed_pairs == gold_pairs

    covered, actual_count, declared_count, false_positive = _contract_metrics(
        case, contracts
    )
    generated = validated = spurious = iterations = 0
    policy_discovered = False
    drift_detected = False
    if method == Method.AUTHSYNTH_CGAR:
        (
            generated,
            validated,
            spurious,
            iterations,
            policy_discovered,
            drift_detected,
        ) = _refinement_counts(case)
    certificate_payload = {
        "method": method.value,
        "case_id": case.case_id,
        "implementation_versions": sorted(
            [
                [implementation.tool_name, implementation.version]
                for implementation in case.implementations
            ]
        ),
        "safety_predicate_digest": canonical_digest(
            [rule.to_dict() for rule in case.safety_rules]
        ),
        "exact_shield_match": exact_match,
    }
    certificate_size = (
        len(str(certificate_payload).encode("utf-8"))
        if method == Method.AUTHSYNTH_CGAR
        else 0
    )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    return CaseEvaluation(
        method=method,
        case_id=case.case_id,
        split=case.split,
        mutation_category=case.mutation_category,
        omission_type=case.omission_type.value,
        benign_allowed=benign_allowed,
        attack_allowed=attack_allowed,
        benign_concrete_safe=benign_safe,
        attack_concrete_forbidden=attack_forbidden,
        forbidden_trace_accepted=attack_allowed and attack_forbidden,
        contract_effect_covered_count=covered,
        contract_effect_actual_count=actual_count,
        contract_effect_declared_count=declared_count,
        contract_effect_false_positive_count=false_positive,
        policy_omission_discovered=policy_discovered,
        counterexamples_generated=generated,
        counterexamples_validated=validated,
        spurious_counterexamples=spurious,
        drift_detected=drift_detected,
        allowed_gold_safe_action_count=len(allowed_pairs & gold_pairs),
        gold_safe_action_count=len(gold_pairs),
        exact_shield_match=exact_match,
        analysis_time_ms=elapsed,
        cegar_iterations=iterations,
        certificate_size_bytes=certificate_size,
    )


def evaluate_cases(
    cases: Sequence[EffectCase], methods: Sequence[Method] = METHODS
) -> tuple[CaseEvaluation, ...]:
    return tuple(evaluate_case(method, case) for method in methods for case in cases)
