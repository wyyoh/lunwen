from __future__ import annotations

from dataclasses import replace

from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_shared import VerificationKind
from keyed_gram.authsynth_symbolic_verifier.equivalence import BoundedSymbolicVerifier


def test_verifier_finds_under_approximation_with_concrete_assignment() -> None:
    case = case_for("hidden_guarded_effect")
    result = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        case.analyzer_input.case_handle, case.analyzer_input.declared_contract
    )
    assert result.kind == VerificationKind.MISSING_EFFECT
    assert result.assignment is not None
    assert result.counterexample_digest is not None


def test_verifier_finds_field_binding_mismatch() -> None:
    case = case_for("parameter_role_alias")
    result = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        case.analyzer_input.case_handle, case.analyzer_input.declared_contract
    )
    assert result.kind == VerificationKind.FIELD_BINDING_MISMATCH


def test_verifier_finds_spurious_effect() -> None:
    case = case_for("conservative_control")
    result = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        case.analyzer_input.case_handle, case.analyzer_input.declared_contract
    )
    assert result.kind == VerificationKind.SPURIOUS_EFFECT


def test_exact_contract_is_equivalent_on_bounded_domain() -> None:
    case = case_for("exact_declared_control")
    exact = replace(
        case, implementation=replace(case.implementation, proof_obstacle=None)
    )
    result = BoundedSymbolicVerifier((exact,), timeout_ms=3000).check(
        exact.analyzer_input.case_handle, exact.reference_contract
    )
    assert result.kind == VerificationKind.EQUIVALENT


def test_unbounded_loop_and_async_horizon_return_unknown() -> None:
    base = case_for("hidden_guarded_effect")
    loop = replace(
        base, implementation=replace(base.implementation, bounded_loop_max=None)
    )
    delayed = replace(
        base, implementation=replace(base.implementation, delayed_depth=3)
    )
    loop_result = BoundedSymbolicVerifier((loop,), timeout_ms=3000).check(
        loop.analyzer_input.case_handle, loop.reference_contract
    )
    delayed_result = BoundedSymbolicVerifier((delayed,), timeout_ms=3000).check(
        delayed.analyzer_input.case_handle, delayed.reference_contract
    )
    assert loop_result.kind == VerificationKind.UNKNOWN
    assert loop_result.reason_code == "unbounded_loop_not_supported"
    assert delayed_result.kind == VerificationKind.UNKNOWN
    assert delayed_result.reason_code == "bounded_quiescence_horizon_exceeded"


def test_bounded_loop_metadata_alone_cannot_certify_unimplemented_semantics() -> None:
    base = case_for("hidden_guarded_effect")
    bounded = replace(
        base, implementation=replace(base.implementation, bounded_loop_max=4)
    )
    result = BoundedSymbolicVerifier((bounded,), timeout_ms=3000).check(
        bounded.analyzer_input.case_handle, bounded.reference_contract
    )
    assert result.kind == VerificationKind.UNKNOWN
    assert result.reason_code == "bounded_loop_semantics_not_implemented"
