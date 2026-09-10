"""guarded symbolic effect contract 与可信高层 safety specification。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .effects import ConcreteEffect, SymbolicEffectTemplate, SymbolicStateUpdate
from .formulas import BooleanFormula, GrammarLimits
from .schema import (
    BoundedSchema,
    ConcreteAssignment,
    StructuredStateDiff,
    SymbolicSchemaError,
    canonical_digest,
    canonical_json,
    restricted_token,
)


@dataclass(frozen=True)
class SymbolicEffectClause:
    guard: BooleanFormula
    effect: SymbolicEffectTemplate

    def validate(self, schema: BoundedSchema, limits: GrammarLimits) -> None:
        self.guard.validate(schema, limits)
        self.effect.validate(schema, allow_ite=limits.allow_ite_terms)

    def to_dict(self) -> dict[str, Any]:
        return {"guard": self.guard.to_dict(), "effect": self.effect.to_dict()}


@dataclass(frozen=True)
class SymbolicStateUpdateClause:
    guard: BooleanFormula
    update: SymbolicStateUpdate

    def validate(self, schema: BoundedSchema, limits: GrammarLimits) -> None:
        self.guard.validate(schema, limits)
        self.update.validate(schema, allow_ite=limits.allow_ite_terms)

    def to_dict(self) -> dict[str, Any]:
        return {"guard": self.guard.to_dict(), "update": self.update.to_dict()}


@dataclass(frozen=True)
class SymbolicEffectContract:
    tool_id: str
    implementation_version_digest: str
    input_schema_digest: str
    state_schema_digest: str
    clauses: tuple[SymbolicEffectClause, ...] = field(default_factory=tuple)
    state_updates: tuple[SymbolicStateUpdateClause, ...] = field(default_factory=tuple)
    unsupported_regions: tuple[BooleanFormula, ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self) -> None:
        restricted_token(self.tool_id, "contract tool_id")
        if self.schema_version != 1:
            raise SymbolicSchemaError("symbolic contract schema_version 非法")
        for label, value in (
            ("implementation_version_digest", self.implementation_version_digest),
            ("input_schema_digest", self.input_schema_digest),
            ("state_schema_digest", self.state_schema_digest),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise SymbolicSchemaError(f"{label} 不是 SHA-256")
        object.__setattr__(self, "clauses", self._canonical(self.clauses))
        object.__setattr__(self, "state_updates", self._canonical(self.state_updates))
        object.__setattr__(
            self, "unsupported_regions", self._canonical(self.unsupported_regions)
        )

    @staticmethod
    def _canonical(values: Sequence[Any]) -> tuple[Any, ...]:
        indexed = {canonical_json(item.to_dict()): item for item in values}
        return tuple(indexed[key] for key in sorted(indexed))

    def validate(self, schema: BoundedSchema, limits: GrammarLimits) -> None:
        if self.input_schema_digest != canonical_digest(
            {"fields": [item.to_dict() for item in schema.input_fields]}
        ):
            raise SymbolicSchemaError("contract input schema digest mismatch")
        if self.state_schema_digest != canonical_digest(
            {"fields": [item.to_dict() for item in schema.state_fields]}
        ):
            raise SymbolicSchemaError("contract state schema digest mismatch")
        for clause in self.clauses:
            clause.validate(schema, limits)
        for clause in self.state_updates:
            clause.validate(schema, limits)
        for region in self.unsupported_regions:
            region.validate(schema, limits)

    def with_version(self, digest: str) -> SymbolicEffectContract:
        return type(self)(
            tool_id=self.tool_id,
            implementation_version_digest=digest,
            input_schema_digest=self.input_schema_digest,
            state_schema_digest=self.state_schema_digest,
            clauses=self.clauses,
            state_updates=self.state_updates,
            unsupported_regions=self.unsupported_regions,
            schema_version=self.schema_version,
        )

    def predict(
        self, assignment: ConcreteAssignment
    ) -> tuple[tuple[ConcreteEffect, ...], dict[str, Any]]:
        effects = tuple(
            sorted(
                {
                    clause.effect.instantiate(assignment)
                    for clause in self.clauses
                    if clause.guard.evaluate(assignment)
                },
                key=lambda item: item.semantic_key,
            )
        )
        updates: dict[str, Any] = {}
        for clause in self.state_updates:
            if clause.guard.evaluate(assignment):
                value = clause.update.instantiate(assignment)
                if (
                    clause.update.field in updates
                    and updates[clause.update.field] != value
                ):
                    raise SymbolicSchemaError("overlap state update 产生冲突值")
                updates[clause.update.field] = value
        return effects, updates

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    @property
    def clause_count(self) -> int:
        return len(self.clauses) + len(self.state_updates)

    @property
    def literal_count(self) -> int:
        return sum(
            item.guard.literal_count for item in (*self.clauses, *self.state_updates)
        )

    @property
    def ast_node_count(self) -> int:
        return (
            self.literal_count
            + sum(item.effect.ast_nodes for item in self.clauses)
            + sum(item.update.value.ast_nodes for item in self.state_updates)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_id": self.tool_id,
            "implementation_version_digest": self.implementation_version_digest,
            "input_schema_digest": self.input_schema_digest,
            "state_schema_digest": self.state_schema_digest,
            "clauses": [item.to_dict() for item in self.clauses],
            "state_updates": [item.to_dict() for item in self.state_updates],
            "unsupported_regions": [
                item.to_dict() for item in self.unsupported_regions
            ],
        }


@dataclass(frozen=True)
class TrustedSafetySpecification:
    """组织提供的 bad-effect predicate；不是从工具执行中推断。"""

    spec_id: str
    forbidden_kinds: frozenset[str] = field(default_factory=frozenset)
    forbidden_destinations: frozenset[str] = field(default_factory=frozenset)
    maximum_amount: int | None = None

    def __post_init__(self) -> None:
        restricted_token(self.spec_id, "safety spec id")
        object.__setattr__(
            self,
            "forbidden_kinds",
            frozenset(
                restricted_token(item, "forbidden kind")
                for item in self.forbidden_kinds
            ),
        )
        object.__setattr__(
            self,
            "forbidden_destinations",
            frozenset(
                restricted_token(item, "forbidden destination")
                for item in self.forbidden_destinations
            ),
        )
        if self.maximum_amount is not None and (
            not isinstance(self.maximum_amount, int)
            or isinstance(self.maximum_amount, bool)
        ):
            raise SymbolicSchemaError("maximum_amount 必须是 int")

    def violation_codes(
        self,
        effects: Sequence[ConcreteEffect],
        state_diff: StructuredStateDiff | None = None,
    ) -> tuple[str, ...]:
        codes: set[str] = set()
        for event in effects:
            if event.kind in self.forbidden_kinds:
                codes.add("forbidden_effect_kind")
            if event.destination in self.forbidden_destinations:
                codes.add("forbidden_destination")
            if (
                self.maximum_amount is not None
                and isinstance(event.amount, int)
                and not isinstance(event.amount, bool)
                and event.amount > self.maximum_amount
            ):
                codes.add("amount_limit_exceeded")
        if state_diff is not None and any(
            item.object_type == "forbidden" for item in state_diff.changes
        ):
            codes.add("forbidden_state_update")
        return tuple(sorted(codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "forbidden_kinds": sorted(self.forbidden_kinds),
            "forbidden_destinations": sorted(self.forbidden_destinations),
            "maximum_amount": self.maximum_amount,
        }
