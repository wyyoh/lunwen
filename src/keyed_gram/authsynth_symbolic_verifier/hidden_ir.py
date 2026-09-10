"""仅 Verifier/Evaluator world 可见的 ToolSymbolicIR concrete semantics。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    ConcreteAssignment,
    ConcreteEffect,
    PatchAtom,
    SymbolicAnalyzerInput,
    SymbolicEffectContract,
    SymbolicEffectTemplate,
    SymbolicStateUpdate,
    canonical_digest,
)

from .bounded_loop import BoundedLoop


@dataclass(frozen=True)
class GuardedTransition:
    transition_id: str
    guard: BooleanFormula
    immediate_effects: tuple[SymbolicEffectTemplate, ...] = field(default_factory=tuple)
    delayed_effects: tuple[SymbolicEffectTemplate, ...] = field(default_factory=tuple)
    state_updates: tuple[SymbolicStateUpdate, ...] = field(default_factory=tuple)

    def all_effects(self) -> tuple[SymbolicEffectTemplate, ...]:
        return (*self.immediate_effects, *self.delayed_effects)

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "guard": self.guard.to_dict(),
            "immediate_effects": [item.to_dict() for item in self.immediate_effects],
            "delayed_effects": [item.to_dict() for item in self.delayed_effects],
            "state_updates": [item.to_dict() for item in self.state_updates],
        }


@dataclass(frozen=True)
class HiddenSymbolicImplementation:
    tool_id: str
    schema: BoundedSchema
    version_digest: str
    transitions: tuple[GuardedTransition, ...]
    bounded_loop_max: int | None = 0
    delayed_depth: int = 0
    instrumentation_coverage: float = 1.0
    proof_obstacle: str | None = None
    loop: BoundedLoop | None = None
    after_loop: tuple[GuardedTransition, ...] = ()

    def __post_init__(self) -> None:
        if self.after_loop and self.loop is None:
            raise ValueError("after_loop 必须附属于显式 loop，不能静默忽略")
        if self.bounded_loop_max is not None and self.bounded_loop_max < 0:
            raise ValueError("bounded loop 上限不能为负")
        if self.delayed_depth < 0:
            raise ValueError("delayed depth 不能为负")
        if not 0.0 <= self.instrumentation_coverage <= 1.0:
            raise ValueError("instrumentation coverage 超界")

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "tool_id": self.tool_id,
                "schema": self.schema.to_dict(),
                "version_digest": self.version_digest,
                "transitions": [item.to_digest_dict() for item in self.transitions],
                "bounded_loop_max": self.bounded_loop_max,
                "delayed_depth": self.delayed_depth,
                "instrumentation_coverage": self.instrumentation_coverage,
                "proof_obstacle": self.proof_obstacle,
                "loop": None if self.loop is None else self.loop.to_dict(),
                "after_loop": [t.to_digest_dict() for t in self.after_loop],
            }
        )

    def execute(
        self, assignment: ConcreteAssignment
    ) -> tuple[tuple[ConcreteEffect, ...], dict[str, Any]]:
        self.schema.validate_assignment(assignment)
        if self.loop is not None:
            result = self.execute_detailed(assignment)
            return result.events, {
                change.field: change.after for change in result.state_diff.changes
            }
        events: set[ConcreteEffect] = set()
        updates: dict[str, Any] = {}
        for transition in self.transitions:
            if not transition.guard.evaluate(assignment):
                continue
            for template in transition.all_effects():
                events.add(template.instantiate(assignment))
            for update in transition.state_updates:
                value = update.instantiate(assignment)
                if update.field in updates and updates[update.field] != value:
                    raise ValueError("hidden transitions 对同一字段产生冲突 update")
                updates[update.field] = value
        return tuple(sorted(events, key=lambda item: item.semantic_key)), updates

    def execute_detailed(self, assignment: ConcreteAssignment):
        from .bounded_loop import execute_program

        return execute_program(self, assignment)


@dataclass(frozen=True)
class HiddenSymbolicCase:
    analyzer_input: SymbolicAnalyzerInput
    split: str
    domain: str
    tool_family: str
    mutation_category: str
    mutation_family: str
    guard_template_family: str
    field_binding_family: str
    version_lineage: str
    ast_shape_family: str
    implementation: HiddenSymbolicImplementation
    reference_contract: SymbolicEffectContract
    expected_unknown: bool
    clean_control: bool

    @cached_property
    def omission_atoms(self) -> frozenset[PatchAtom]:
        from .omission_oracle import expected_omission_atoms

        return expected_omission_atoms(self)

    @property
    def public_case_id(self) -> str:
        return self.analyzer_input.public_case_id

    def evaluator_manifest_row(self) -> dict[str, Any]:
        """不持久化 implementation AST/reference contract。"""

        return {
            "public_case_id": self.public_case_id,
            "split": self.split,
            "domain": self.domain,
            "tool_family_digest": canonical_digest(self.tool_family),
            "mutation_family_digest": canonical_digest(self.mutation_family),
            "guard_template_family_digest": canonical_digest(
                self.guard_template_family
            ),
            "field_binding_family_digest": canonical_digest(self.field_binding_family),
            "version_lineage_digest": canonical_digest(self.version_lineage),
            "ast_shape_family_digest": canonical_digest(self.ast_shape_family),
            "mutation_category": self.mutation_category,
            "omission_atom_digests": sorted(
                item.digest for item in self.omission_atoms
            ),
            "implementation_digest": self.implementation.digest,
            "reference_contract_digest": self.reference_contract.digest,
            "expected_unknown": self.expected_unknown,
            "clean_control": self.clean_control,
        }
