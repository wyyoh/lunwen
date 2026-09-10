from __future__ import annotations

import pytest

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    ConcreteAssignment,
    Conjunction,
    DomainField,
    GrammarLimits,
    Literal,
    LiteralOperator,
    SymbolicSchemaError,
    VariableRef,
)


def _schema() -> BoundedSchema:
    return BoundedSchema(
        (
            DomainField("mode", "input", "enum", enum_values=("admin", "member")),
            DomainField("amount", "input", "int", minimum=0, maximum=3),
        )
    )


def test_guard_canonicalization_and_evaluation() -> None:
    literal = Literal(VariableRef("input", "mode"), LiteralOperator.EQ, "admin")
    guard = BooleanFormula((Conjunction((literal, literal)), Conjunction((literal,))))
    assert len(guard.disjuncts) == 1
    guard.validate(_schema(), GrammarLimits())
    assert guard.evaluate(ConcreteAssignment((("amount", 0), ("mode", "admin")), ()))
    assert not guard.evaluate(
        ConcreteAssignment((("amount", 0), ("mode", "member")), ())
    )


def test_dnf_limits_are_enforced() -> None:
    formula = BooleanFormula(
        Conjunction(
            (Literal(VariableRef("input", "amount"), LiteralOperator.GE, value),)
        )
        for value in (0, 1, 2)
    )
    with pytest.raises(SymbolicSchemaError, match="disjunct"):
        formula.validate(_schema(), GrammarLimits(max_disjuncts=2))


def test_ordered_comparison_is_int_only() -> None:
    literal = Literal(VariableRef("input", "mode"), LiteralOperator.GE, 1)
    with pytest.raises(SymbolicSchemaError):
        literal.validate(_schema())


def test_true_and_false_have_explicit_dnf_semantics() -> None:
    assignment = ConcreteAssignment((("amount", 0), ("mode", "admin")), ())
    assert BooleanFormula.true().evaluate(assignment)
    assert not BooleanFormula.false().evaluate(assignment)
