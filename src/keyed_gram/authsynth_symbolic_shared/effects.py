"""符号 effect template、字段绑定和具体事件。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .formulas import Literal, VariableRef
from .schema import (
    BoundedSchema,
    ConcreteAssignment,
    Scalar,
    SymbolicSchemaError,
    restricted_token,
)


class TermKind(str, Enum):
    CONSTANT = "constant"
    VARIABLE = "variable"
    ITE = "ite"
    NONE = "none"


@dataclass(frozen=True)
class SymbolicTerm:
    kind: TermKind
    constant: Scalar | None = None
    variable: VariableRef | None = None
    condition: Literal | None = None
    then_term: SymbolicTerm | None = None
    else_term: SymbolicTerm | None = None

    def __post_init__(self) -> None:
        populated = {
            "constant": self.constant is not None,
            "variable": self.variable is not None,
            "condition": self.condition is not None,
            "then_term": self.then_term is not None,
            "else_term": self.else_term is not None,
        }
        if self.kind == TermKind.NONE:
            if any(populated.values()):
                raise SymbolicSchemaError("none term 不能携带值")
        elif self.kind == TermKind.CONSTANT:
            if self.constant is None or any(
                populated[key]
                for key in ("variable", "condition", "then_term", "else_term")
            ):
                raise SymbolicSchemaError("constant term schema 非法")
            if not isinstance(self.constant, (bool, int, str)):
                raise SymbolicSchemaError("constant term 不是 scalar")
            if isinstance(self.constant, str):
                restricted_token(self.constant, "term constant")
        elif self.kind == TermKind.VARIABLE:
            if self.variable is None or any(
                populated[key]
                for key in ("constant", "condition", "then_term", "else_term")
            ):
                raise SymbolicSchemaError("variable term schema 非法")
        elif self.kind == TermKind.ITE:
            if (
                self.condition is None
                or self.then_term is None
                or self.else_term is None
                or self.constant is not None
                or self.variable is not None
            ):
                raise SymbolicSchemaError("ite term schema 非法")
            if (
                self.then_term.kind == TermKind.ITE
                or self.else_term.kind == TermKind.ITE
            ):
                raise SymbolicSchemaError("F2C v1 不允许嵌套 ite")
        else:
            raise SymbolicSchemaError("未知 term kind")

    @classmethod
    def none(cls) -> SymbolicTerm:
        return cls(TermKind.NONE)

    @classmethod
    def const(cls, value: Scalar) -> SymbolicTerm:
        return cls(TermKind.CONSTANT, constant=value)

    @classmethod
    def var(cls, source: str, name: str) -> SymbolicTerm:
        return cls(TermKind.VARIABLE, variable=VariableRef(source, name))

    @classmethod
    def ite(
        cls,
        condition: Literal,
        then_term: SymbolicTerm,
        else_term: SymbolicTerm,
    ) -> SymbolicTerm:
        return cls(
            TermKind.ITE,
            condition=condition,
            then_term=then_term,
            else_term=else_term,
        )

    def validate(self, schema: BoundedSchema, *, allow_ite: bool = True) -> None:
        if self.kind == TermKind.VARIABLE:
            assert self.variable is not None
            schema.field(self.variable.source, self.variable.name)
        elif self.kind == TermKind.ITE:
            if not allow_ite:
                raise SymbolicSchemaError("grammar 禁止 ite term")
            assert self.condition and self.then_term and self.else_term
            self.condition.validate(schema)
            self.then_term.validate(schema, allow_ite=False)
            self.else_term.validate(schema, allow_ite=False)

    def evaluate(self, assignment: ConcreteAssignment) -> Scalar | None:
        if self.kind == TermKind.NONE:
            return None
        if self.kind == TermKind.CONSTANT:
            return self.constant
        if self.kind == TermKind.VARIABLE:
            assert self.variable is not None
            return assignment.value(self.variable.source, self.variable.name)
        assert self.condition and self.then_term and self.else_term
        branch = (
            self.then_term if self.condition.evaluate(assignment) else self.else_term
        )
        return branch.evaluate(assignment)

    @property
    def ast_nodes(self) -> int:
        if self.kind != TermKind.ITE:
            return 1
        assert self.then_term and self.else_term
        return 2 + self.then_term.ast_nodes + self.else_term.ast_nodes

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value}
        if self.kind == TermKind.CONSTANT:
            result["constant"] = self.constant
        elif self.kind == TermKind.VARIABLE:
            assert self.variable is not None
            result["variable"] = self.variable.to_dict()
        elif self.kind == TermKind.ITE:
            assert self.condition and self.then_term and self.else_term
            result.update(
                {
                    "condition": self.condition.to_dict(),
                    "then": self.then_term.to_dict(),
                    "else": self.else_term.to_dict(),
                }
            )
        return result


@dataclass(frozen=True, order=True)
class ConcreteEffect:
    slot: str
    sequence: int
    phase: str
    kind: str
    tenant: Scalar
    resource: Scalar
    destination: Scalar | None = None
    amount: Scalar | None = None
    parent_slot: str | None = None

    def __post_init__(self) -> None:
        restricted_token(self.slot, "effect slot")
        restricted_token(self.phase, "effect phase")
        restricted_token(self.kind, "effect kind")
        if self.phase not in {"immediate", "delayed"}:
            raise SymbolicSchemaError("effect phase 非法")
        if self.sequence < 0:
            raise SymbolicSchemaError("effect sequence 必须非负")
        if self.parent_slot is not None:
            restricted_token(self.parent_slot, "parent slot")

    @property
    def signature(self) -> tuple[Any, ...]:
        return (self.slot, self.sequence, self.phase, self.kind, self.parent_slot)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            *self.signature,
            self.tenant,
            self.resource,
            self.destination,
            self.amount,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "sequence": self.sequence,
            "phase": self.phase,
            "kind": self.kind,
            "tenant": self.tenant,
            "resource": self.resource,
            "destination": self.destination,
            "amount": self.amount,
            "parent_slot": self.parent_slot,
        }


@dataclass(frozen=True)
class SymbolicEffectTemplate:
    slot: str
    sequence: int
    phase: str
    kind: str
    tenant: SymbolicTerm
    resource: SymbolicTerm
    destination: SymbolicTerm = SymbolicTerm(TermKind.NONE)
    amount: SymbolicTerm = SymbolicTerm(TermKind.NONE)
    parent_slot: str | None = None

    def __post_init__(self) -> None:
        restricted_token(self.slot, "effect slot")
        restricted_token(self.phase, "effect phase")
        restricted_token(self.kind, "effect kind")
        if self.phase not in {"immediate", "delayed"}:
            raise SymbolicSchemaError("effect template phase 非法")
        if self.sequence < 0:
            raise SymbolicSchemaError("effect template sequence 必须非负")
        if self.parent_slot is not None:
            restricted_token(self.parent_slot, "effect parent slot")

    @property
    def signature(self) -> tuple[Any, ...]:
        return (self.slot, self.sequence, self.phase, self.kind, self.parent_slot)

    def validate(self, schema: BoundedSchema, *, allow_ite: bool) -> None:
        for term in (self.tenant, self.resource, self.destination, self.amount):
            term.validate(schema, allow_ite=allow_ite)
        if self.tenant.kind == TermKind.NONE or self.resource.kind == TermKind.NONE:
            raise SymbolicSchemaError("effect tenant/resource 不能为 none")

    def instantiate(self, assignment: ConcreteAssignment) -> ConcreteEffect:
        tenant = self.tenant.evaluate(assignment)
        resource = self.resource.evaluate(assignment)
        if tenant is None or resource is None:
            raise SymbolicSchemaError("effect 必填字段实例化为 none")
        return ConcreteEffect(
            slot=self.slot,
            sequence=self.sequence,
            phase=self.phase,
            kind=self.kind,
            tenant=tenant,
            resource=resource,
            destination=self.destination.evaluate(assignment),
            amount=self.amount.evaluate(assignment),
            parent_slot=self.parent_slot,
        )

    @property
    def ast_nodes(self) -> int:
        return 1 + sum(
            term.ast_nodes
            for term in (self.tenant, self.resource, self.destination, self.amount)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "sequence": self.sequence,
            "phase": self.phase,
            "kind": self.kind,
            "tenant": self.tenant.to_dict(),
            "resource": self.resource.to_dict(),
            "destination": self.destination.to_dict(),
            "amount": self.amount.to_dict(),
            "parent_slot": self.parent_slot,
        }


@dataclass(frozen=True)
class SymbolicStateUpdate:
    field: str
    value: SymbolicTerm

    def __post_init__(self) -> None:
        restricted_token(self.field, "state update field")

    def validate(self, schema: BoundedSchema, *, allow_ite: bool) -> None:
        schema.field("state", self.field)
        self.value.validate(schema, allow_ite=allow_ite)

    def instantiate(self, assignment: ConcreteAssignment) -> Scalar:
        value = self.value.evaluate(assignment)
        if value is None:
            raise SymbolicSchemaError("state update 不能实例化为 none")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "value": self.value.to_dict()}
