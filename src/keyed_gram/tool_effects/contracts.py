"""Tool contract 的精确效果完备性与过近似检查。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ir import execute_trace
from .types import (
    DeclaredContract,
    EffectAtom,
    ToolImplementation,
    TraceScenario,
)


@dataclass(frozen=True)
class ContractCheck:
    effect_complete: bool
    covered_effect_count: int
    concrete_effect_count: int
    uncovered_effects: tuple[EffectAtom, ...]
    impossible_declared_effects: tuple[EffectAtom, ...]
    contract_effect_recall: float
    contract_overapproximation_ratio: float
    version_binding_matches: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_complete": self.effect_complete,
            "covered_effect_count": self.covered_effect_count,
            "concrete_effect_count": self.concrete_effect_count,
            "uncovered_effect_count": len(self.uncovered_effects),
            "impossible_declared_effect_count": len(self.impossible_declared_effects),
            "contract_effect_recall": self.contract_effect_recall,
            "contract_overapproximation_ratio": (self.contract_overapproximation_ratio),
            "version_binding_matches": self.version_binding_matches,
        }


def check_effect_completeness(
    contract: DeclaredContract,
    implementations: Mapping[tuple[str, str], ToolImplementation],
    scenarios: Sequence[TraceScenario],
) -> ContractCheck:
    """在给定有限输入域上精确检查 Traces(T) 是否包含于 γ(C)。"""

    concrete_by_call: list[tuple[EffectAtom, Any]] = []
    declared: set[EffectAtom] = set()
    version_matches = True
    for scenario in scenarios:
        observed = execute_trace(implementations, scenario, drain_async=True)
        for step in scenario.steps:
            if step.call.tool_name != contract.tool_name:
                continue
            version_matches = version_matches and (
                step.call.version == contract.bound_version
            )
            declared.update(contract.instantiate(step.call))
            for effect in observed:
                if effect.source_tool == contract.tool_name:
                    concrete_by_call.append((effect, step.call))
    concrete = {effect for effect, _ in concrete_by_call}
    uncovered = {
        effect for effect, call in concrete_by_call if not contract.covers(effect, call)
    }
    concrete_keys = {effect.semantic_key for effect in concrete}
    impossible = {
        effect for effect in declared if effect.semantic_key not in concrete_keys
    }
    covered_count = len(concrete) - len(uncovered)
    recall = 1.0 if not concrete else covered_count / len(concrete)
    over = 0.0 if not declared else len(impossible) / len(declared)
    return ContractCheck(
        effect_complete=not uncovered and version_matches,
        covered_effect_count=covered_count,
        concrete_effect_count=len(concrete),
        uncovered_effects=tuple(sorted(uncovered)),
        impossible_declared_effects=tuple(sorted(impossible)),
        contract_effect_recall=recall,
        contract_overapproximation_ratio=over,
        version_binding_matches=version_matches,
    )
