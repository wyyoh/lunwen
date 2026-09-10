from __future__ import annotations

from dataclasses import replace

import pytest
from f2c_symbolic_helpers import GRAMMAR, case_for

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    SymbolicEffectClause,
    SymbolicEffectContract,
    SymbolicSchemaError,
    SymbolicStateUpdate,
    SymbolicStateUpdateClause,
    SymbolicTerm,
)
from keyed_gram.authsynth_symbolic_verifier.benchmark import enumerate_assignments


def test_contract_canonicalizes_duplicate_clauses_and_digest() -> None:
    case = case_for("exact_declared_control")
    contract = case.reference_contract
    duplicate = replace(contract, clauses=(*contract.clauses, contract.clauses[0]))
    assert duplicate.clauses == contract.clauses
    assert duplicate.digest == contract.digest
    duplicate.validate(case.analyzer_input.schema, GRAMMAR)


def test_contract_effect_fields_are_symbolic_not_case_lookup() -> None:
    case = case_for("parameter_role_alias")
    assignment = next(enumerate_assignments(case.analyzer_input.schema))
    events, _ = case.reference_contract.predict(assignment)
    assert events[0].tenant == assignment.value("input", "tenant")
    assert "query_id" not in case.reference_contract.to_dict()


def test_unreachable_clause_is_safe_and_does_not_fire() -> None:
    case = case_for("exact_declared_control")
    clause = SymbolicEffectClause(
        BooleanFormula.false(), case.reference_contract.clauses[0].effect
    )
    contract = replace(
        case.reference_contract, clauses=(*case.reference_contract.clauses, clause)
    )
    assignment = next(enumerate_assignments(case.analyzer_input.schema))
    assert contract.predict(assignment) == case.reference_contract.predict(assignment)


def test_overlapping_conflicting_state_updates_fail_closed() -> None:
    case = case_for("exact_declared_control")
    updates = (
        SymbolicStateUpdateClause(
            BooleanFormula.true(), SymbolicStateUpdate("counter", SymbolicTerm.const(0))
        ),
        SymbolicStateUpdateClause(
            BooleanFormula.true(), SymbolicStateUpdate("counter", SymbolicTerm.const(1))
        ),
    )
    contract = SymbolicEffectContract(
        tool_id=case.reference_contract.tool_id,
        implementation_version_digest=case.reference_contract.implementation_version_digest,
        input_schema_digest=case.reference_contract.input_schema_digest,
        state_schema_digest=case.reference_contract.state_schema_digest,
        clauses=case.reference_contract.clauses,
        state_updates=updates,
    )
    assignment = next(enumerate_assignments(case.analyzer_input.schema))
    with pytest.raises(SymbolicSchemaError, match="冲突"):
        contract.predict(assignment)


def test_contract_is_bound_to_implementation_version() -> None:
    case = case_for("exact_declared_control")
    assert (
        case.reference_contract.implementation_version_digest
        == case.implementation.version_digest
    )
    changed = case.reference_contract.with_version("0" * 64)
    assert changed.digest != case.reference_contract.digest
