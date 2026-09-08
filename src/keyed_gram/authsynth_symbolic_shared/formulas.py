"""F2C 受限 DNF guard 语言。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .schema import (
    BoundedSchema,
    ConcreteAssignment,
    Scalar,
    SymbolicSchemaError,
    canonical_json,
    restricted_token,
)


class LiteralOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LE = "le"
    GE = "ge"


@dataclass(frozen=True, order=True)
class VariableRef:
    source: str
    name: str

    def __post_init__(self) -> None:
        if self.source not in {"input", "state"}:
            raise SymbolicSchemaError("variable source 必须是 input/state")
        restricted_token(self.name, "variable name")

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "name": self.name}


@dataclass(frozen=True, order=True)
class Literal:
    variable: VariableRef
    operator: LiteralOperator
    value: Scalar

    def __post_init__(self) -> None:
        if not isinstance(self.operator, LiteralOperator):
            raise SymbolicSchemaError("literal operator 非法")
        if not isinstance(self.value, (bool, int, str)):
            raise SymbolicSchemaError("literal value 不是 scalar")
        if self.operator in {LiteralOperator.LE, LiteralOperator.GE} and (
            not isinstance(self.value, int) or isinstance(self.value, bool)
        ):
            raise SymbolicSchemaError("有序比较只支持 int")

    def validate(self, schema: BoundedSchema) -> None:
        field = schema.field(self.variable.source, self.variable.name)
        if not field.accepts(self.value):
            raise SymbolicSchemaError("literal constant 不属于字段域")
        if (
            self.operator in {LiteralOperator.LE, LiteralOperator.GE}
            and field.kind != "int"
        ):
            raise SymbolicSchemaError("非 int 字段不能使用 <=/>=")

    def evaluate(self, assignment: ConcreteAssignment) -> bool:
        current = assignment.value(self.variable.source, self.variable.name)
        if self.operator == LiteralOperator.EQ:
            return current == self.value
        if self.operator == LiteralOperator.NE:
            return current != self.value
        if self.operator == LiteralOperator.LE:
            return int(current) <= int(self.value)
        return int(current) >= int(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable.to_dict(),
            "operator": self.operator.value,
            "value": self.value,
        }


@dataclass(frozen=True, order=True)
class Conjunction:
    literals: tuple[Literal, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(set(self.literals)))
        object.__setattr__(self, "literals", ordered)

    def validate(self, schema: BoundedSchema, max_literals: int) -> None:
        if len(self.literals) > max_literals:
            raise SymbolicSchemaError("conjunction 超过 literal 上限")
        for literal in self.literals:
            literal.validate(schema)

    def evaluate(self, assignment: ConcreteAssignment) -> bool:
        return all(item.evaluate(assignment) for item in self.literals)

    def to_dict(self) -> dict[str, Any]:
        return {"literals": [item.to_dict() for item in self.literals]}


@dataclass(frozen=True)
class BooleanFormula:
    """DNF；空 disjuncts 表示 false，空 conjunction 表示 true。"""

    disjuncts: tuple[Conjunction, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        unique = {canonical_json(item.to_dict()): item for item in self.disjuncts}
        ordered = tuple(unique[key] for key in sorted(unique))
        object.__setattr__(self, "disjuncts", ordered)

    @classmethod
    def true(cls) -> BooleanFormula:
        return cls((Conjunction(),))

    @classmethod
    def false(cls) -> BooleanFormula:
        return cls(())

    def validate(self, schema: BoundedSchema, limits: GrammarLimits) -> None:
        if len(self.disjuncts) > limits.max_disjuncts:
            raise SymbolicSchemaError("guard 超过 disjunct 上限")
        total = sum(len(item.literals) for item in self.disjuncts)
        if total > limits.max_total_literals:
            raise SymbolicSchemaError("guard 超过 total literal 上限")
        for item in self.disjuncts:
            item.validate(schema, limits.max_literals_per_conjunction)

    def evaluate(self, assignment: ConcreteAssignment) -> bool:
        return any(item.evaluate(assignment) for item in self.disjuncts)

    @property
    def literal_count(self) -> int:
        return sum(len(item.literals) for item in self.disjuncts)

    def to_dict(self) -> dict[str, Any]:
        return {"dnf": [item.to_dict() for item in self.disjuncts]}


@dataclass(frozen=True)
class GrammarLimits:
    max_disjuncts: int = 4
    max_literals_per_conjunction: int = 5
    max_total_literals: int = 16
    allow_ite_terms: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_disjuncts <= 8:
            raise SymbolicSchemaError("max_disjuncts 超界")
        if not 1 <= self.max_literals_per_conjunction <= 8:
            raise SymbolicSchemaError("max_literals_per_conjunction 超界")
        if not 1 <= self.max_total_literals <= 32:
            raise SymbolicSchemaError("max_total_literals 超界")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_disjuncts": self.max_disjuncts,
            "max_literals_per_conjunction": self.max_literals_per_conjunction,
            "max_total_literals": self.max_total_literals,
            "allow_ite_terms": self.allow_ite_terms,
        }


def assignment_mapping(
    assignment: ConcreteAssignment,
) -> Mapping[tuple[str, str], Scalar]:
    return {
        **{("input", key): value for key, value in assignment.inputs},
        **{("state", key): value for key, value in assignment.state},
    }
