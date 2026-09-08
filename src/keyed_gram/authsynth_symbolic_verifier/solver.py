"""ToolSymbolicIR 到 Z3 的严格有界编码。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import z3

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    ConcreteAssignment,
    Literal,
    LiteralOperator,
    SymbolicTerm,
    TermKind,
)


@dataclass
class Z3Domain:
    schema: BoundedSchema

    def __post_init__(self) -> None:
        self.variables: dict[tuple[str, str], z3.ExprRef] = {}
        constraints = []
        for field in self.schema.fields:
            key = (field.source, field.name)
            name = f"{field.source}__{field.name}"
            if field.kind == "bool":
                variable = z3.Bool(name)
            elif field.kind == "enum":
                variable = z3.String(name)
            else:
                variable = z3.Int(name)
            self.variables[key] = variable
            if field.kind == "enum":
                constraints.append(
                    z3.Or(
                        *[
                            variable == z3.StringVal(value)
                            for value in field.enum_values
                        ]
                    )
                )
            elif field.kind == "int":
                constraints.append(variable >= field.minimum)
                constraints.append(variable <= field.maximum)
        self.domain_constraints = tuple(constraints)

    def scalar(self, source: str, name: str, value: Any) -> z3.ExprRef:
        field = self.schema.field(source, name)
        if not field.accepts(value):
            raise ValueError("SMT scalar 类型或域不匹配")
        if field.kind == "enum":
            return z3.StringVal(str(value))
        if field.kind == "bool":
            return z3.BoolVal(bool(value))
        return z3.IntVal(int(value))

    def literal(self, literal: Literal) -> z3.BoolRef:
        ref = literal.variable
        variable = self.variables[(ref.source, ref.name)]
        value = self.scalar(ref.source, ref.name, literal.value)
        if literal.operator == LiteralOperator.EQ:
            return variable == value
        if literal.operator == LiteralOperator.NE:
            return variable != value
        if literal.operator == LiteralOperator.LE:
            return variable <= value
        return variable >= value

    def formula(self, formula: BooleanFormula) -> z3.BoolRef:
        if not formula.disjuncts:
            return z3.BoolVal(False)
        return z3.Or(
            *[
                z3.And(*[self.literal(item) for item in conjunction.literals])
                for conjunction in formula.disjuncts
            ]
        )

    def term(self, term: SymbolicTerm) -> z3.ExprRef | None:
        if term.kind == TermKind.NONE:
            return None
        if term.kind == TermKind.VARIABLE:
            assert term.variable is not None
            return self.variables[(term.variable.source, term.variable.name)]
        if term.kind == TermKind.CONSTANT:
            # 无字段上下文的 enum 常量用不可碰撞 String 表示；变量比较由 coerce 处理。
            if isinstance(term.constant, bool):
                return z3.BoolVal(term.constant)
            if isinstance(term.constant, int):
                return z3.IntVal(term.constant)
            return z3.StringVal(str(term.constant))
        assert term.condition and term.then_term and term.else_term
        then_value = self.term(term.then_term)
        else_value = self.term(term.else_term)
        if then_value is None or else_value is None:
            raise ValueError("nullable ITE 语义尚不支持")
        then_value, else_value = self._coerce_pair(
            then_value, else_value, term.then_term, term.else_term
        )
        return z3.If(self.literal(term.condition), then_value, else_value)

    def _coerce_pair(
        self,
        left: z3.ExprRef,
        right: z3.ExprRef,
        left_term: SymbolicTerm,
        right_term: SymbolicTerm,
    ) -> tuple[z3.ExprRef, z3.ExprRef]:
        if left.sort() == right.sort():
            return left, right
        raise ValueError("不可隐式强制转换 symbolic term sorts")

    def equal_terms(self, left: SymbolicTerm, right: SymbolicTerm) -> z3.BoolRef:
        if left.kind == TermKind.NONE or right.kind == TermKind.NONE:
            return z3.BoolVal(left.kind == right.kind)
        left_expr = self.term(left)
        right_expr = self.term(right)
        assert left_expr is not None and right_expr is not None
        if left_expr.sort() != right_expr.sort():
            return z3.BoolVal(False)
        left_expr, right_expr = self._coerce_pair(left_expr, right_expr, left, right)
        return left_expr == right_expr

    def assignment_from_model(self, model: z3.ModelRef) -> ConcreteAssignment:
        inputs = []
        state = []
        for field in self.schema.fields:
            variable = self.variables[(field.source, field.name)]
            value = model.eval(variable, model_completion=True)
            if field.kind == "bool":
                concrete: Any = z3.is_true(value)
            elif field.kind == "enum":
                concrete = value.as_string()
            else:
                concrete = value.as_long()
            (inputs if field.source == "input" else state).append(
                (field.name, concrete)
            )
        assignment = ConcreteAssignment(tuple(inputs), tuple(state))
        self.schema.validate_assignment(assignment)
        return assignment


def deterministic_model(
    domain: Z3Domain,
    predicate: z3.BoolRef,
    *,
    timeout_ms: int,
) -> tuple[str, ConcreteAssignment | None]:
    """逐字段固定最小值；不枚举笛卡尔积。"""

    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*domain.domain_constraints, predicate)
    status = solver.check()
    if status == z3.unknown:
        return "unknown", None
    if status == z3.unsat:
        return "unsat", None
    for field in domain.schema.fields:
        variable = domain.variables[(field.source, field.name)]
        for value in field.values:
            solver.push()
            solver.add(variable == domain.scalar(field.source, field.name, value))
            candidate = solver.check()
            solver.pop()
            if candidate == z3.unknown:
                return "unknown", None
            if candidate == z3.sat:
                solver.add(variable == domain.scalar(field.source, field.name, value))
                break
    final = solver.check()
    if final == z3.unknown:
        return "unknown", None
    if final != z3.sat:
        return "unsat", None
    return "sat", domain.assignment_from_model(solver.model())
