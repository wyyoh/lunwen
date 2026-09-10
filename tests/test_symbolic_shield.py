from __future__ import annotations

from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_analyzer.shield import (
    shield_digest,
    symbolic_shield_allows,
)
from keyed_gram.authsynth_symbolic_shared import ContractStatus
from keyed_gram.authsynth_symbolic_verifier.benchmark import enumerate_assignments


def test_unknown_regions_fail_closed_and_are_counted_as_utility_loss() -> None:
    case = case_for("exact_declared_control")
    assignment = next(enumerate_assignments(case.analyzer_input.schema))
    assert not symbolic_shield_allows(
        case.reference_contract,
        case.analyzer_input.trusted_safety_spec,
        ContractStatus.UNKNOWN,
        assignment,
    )


def test_verified_safe_assignment_is_allowed_and_forbidden_one_is_blocked() -> None:
    case = case_for("hidden_guarded_effect")
    assignments = tuple(enumerate_assignments(case.analyzer_input.schema))
    safe = next(
        item
        for item in assignments
        if item.value("input", "mode") == "member"
        and item.value("input", "destination") == "internal"
    )
    unsafe = next(
        item
        for item in assignments
        if item.value("input", "mode") == "admin"
        and item.value("input", "destination") == "external"
    )
    assert symbolic_shield_allows(
        case.reference_contract,
        case.analyzer_input.trusted_safety_spec,
        ContractStatus.VERIFIED_COMPLETE,
        safe,
    )
    assert not symbolic_shield_allows(
        case.reference_contract,
        case.analyzer_input.trusted_safety_spec,
        ContractStatus.VERIFIED_COMPLETE,
        unsafe,
    )


def test_shield_digest_binds_contract_spec_and_status() -> None:
    case = case_for("hidden_guarded_effect")
    complete = shield_digest(
        case.reference_contract,
        case.analyzer_input.trusted_safety_spec,
        ContractStatus.VERIFIED_COMPLETE,
    )
    unknown = shield_digest(
        case.reference_contract,
        case.analyzer_input.trusted_safety_spec,
        ContractStatus.UNKNOWN,
    )
    assert complete != unknown
