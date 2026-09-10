"""预冻结安全回归；仅使用人工与 train fixture，不执行正式 split。"""

from dataclasses import replace

import pytest
import z3
from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_analyzer import AuthSynthSymbolicAnalyzer
from keyed_gram.authsynth_symbolic_analyzer.learner import (
    SynthesisTimeout,
    infer_term,
    learn_guard,
)
from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    ConcreteAssignment,
    ContractStatus,
    DomainField,
    GrammarLimits,
    SymbolicStateUpdate,
    SymbolicStateUpdateClause,
    SymbolicTerm,
    VerificationKind,
    VerificationResult,
    canonical_digest,
)
from keyed_gram.authsynth_symbolic_verifier.equivalence import BoundedSymbolicVerifier
from keyed_gram.authsynth_symbolic_verifier.hidden_ir import GuardedTransition
from keyed_gram.authsynth_symbolic_verifier.replay_service import SymbolicSandboxReplay
from keyed_gram.authsynth_symbolic_verifier.solver import Z3Domain


def test_conflicting_updates_cannot_be_hidden_by_last_write_encoding():
    base = case_for("exact_declared_control")
    update = SymbolicStateUpdate("counter", SymbolicTerm.const(1))
    transition = replace(base.implementation.transitions[0], state_updates=(update,))
    case = replace(
        base,
        implementation=replace(
            base.implementation, proof_obstacle=None, transitions=(transition,)
        ),
    )
    candidate = replace(
        base.reference_contract,
        state_updates=(
            SymbolicStateUpdateClause(
                BooleanFormula.true(),
                SymbolicStateUpdate("counter", SymbolicTerm.const(0)),
            ),
            SymbolicStateUpdateClause(BooleanFormula.true(), update),
        ),
    )
    result = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        case.analyzer_input.case_handle, candidate
    )
    assert result.kind == VerificationKind.UNKNOWN
    assert result.reason_code == "candidate_state_semantics_invalid"


def test_out_of_domain_implementation_state_cannot_be_certified():
    base = case_for("hidden_guarded_effect")
    transition = GuardedTransition(
        "invalid",
        BooleanFormula.true(),
        state_updates=(SymbolicStateUpdate("counter", SymbolicTerm.const(99)),),
    )
    case = replace(
        base, implementation=replace(base.implementation, transitions=(transition,))
    )
    result = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        case.analyzer_input.case_handle, base.reference_contract
    )
    assert result.kind == VerificationKind.UNKNOWN
    assert result.reason_code == "implementation_state_semantics_invalid"


def test_tool_identity_is_bound_in_addition_to_version():
    case = case_for("hidden_guarded_effect")
    candidate = replace(case.reference_contract, tool_id="different-tool")
    result = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        case.analyzer_input.case_handle, candidate
    )
    assert result.kind == VerificationKind.UNKNOWN
    assert result.reason_code == "candidate_tool_identity_mismatch"


def test_bool_and_int_are_distinct_in_smt_and_learner():
    schema = BoundedSchema((DomainField("x", "input", "bool"),))
    domain = Z3Domain(schema)
    assert z3.is_false(
        z3.simplify(
            domain.equal_terms(SymbolicTerm.var("input", "x"), SymbolicTerm.const(1))
        )
    )
    term = infer_term(
        schema, (ConcreteAssignment((("x", True),), ()),), (1,), GrammarLimits()
    )
    assert term == SymbolicTerm.const(1)


def test_equivalence_without_observed_version_does_not_issue_certificate():
    case = case_for("hidden_guarded_effect")

    class UnboundVerifier:
        solver_name = "test"
        solver_version = "1"

        def check(self, _handle, _candidate):
            return VerificationResult(VerificationKind.EQUIVALENT)

    result = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input, UnboundVerifier(), SymbolicSandboxReplay((case,))
    )
    assert result.contract_status == ContractStatus.UNKNOWN
    assert result.certificate is None
    assert "equivalence_version_unbound_or_mismatch" in result.reason_codes


def test_failed_replay_is_counted_but_not_learned():
    case = case_for("hidden_guarded_effect")
    service = SymbolicSandboxReplay((case,))

    class FailedReplay:
        def replay(self, handle, assignment):
            return replace(service.replay(handle, assignment), exit_status="failed")

    result = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input,
        BoundedSymbolicVerifier((case,), timeout_ms=3000),
        FailedReplay(),
    )
    assert result.contract_status == ContractStatus.UNKNOWN
    assert result.certificate is None
    assert result.replay_query_count == 1
    assert not result.evidence
    assert "replay_failed" in result.reason_codes


def test_drift_cannot_refund_replay_budget(monkeypatch):
    case = case_for("hidden_guarded_effect")
    public = replace(case.analyzer_input, replay_budget=1)
    service = SymbolicSandboxReplay((case,))
    first = BoundedSymbolicVerifier((case,), timeout_ms=3000).check(
        public.case_handle, public.declared_contract
    )
    assignment = first.assignment
    assert assignment is not None
    second = replace(first, counterexample_digest=canonical_digest("second"))

    class DriftVerifier:
        solver_name = "test"
        solver_version = "1"

        def __init__(self):
            self.results = iter(
                (
                    first,
                    VerificationResult(
                        VerificationKind.VERSION_MISMATCH,
                        observed_version_digest="f" * 64,
                    ),
                    second,
                )
            )

        def check(self, _handle, _candidate):
            return next(self.results)

    analyzer = AuthSynthSymbolicAnalyzer()
    monkeypatch.setattr(
        analyzer._learner, "fit", lambda **_kwargs: public.declared_contract
    )
    result = analyzer.analyze(public, DriftVerifier(), service)
    assert result.contract_status == ContractStatus.UNKNOWN
    assert result.replay_query_count == service.query_count(public.case_handle) == 1
    assert "replay_budget_exhausted" in result.reason_codes
    assert result.certificate is None


def _guard_examples():
    schema = BoundedSchema(
        (DomainField("x", "input", "bool"), DomainField("y", "input", "bool"))
    )
    values = [
        ConcreteAssignment((("x", x), ("y", y)), ())
        for x in (False, True)
        for y in (False, True)
    ]
    return schema, values[1:], values[:1]


def test_optimize_receives_timeout_and_returns_deterministic_dnf(monkeypatch):
    from keyed_gram.authsynth_symbolic_analyzer import learner

    settings = []
    real = z3.Optimize

    class ObservedOptimize:
        def __init__(self):
            self.instance = real()

        def set(self, **kwargs):
            settings.append(kwargs)
            return self.instance.set(**kwargs)

        def __getattr__(self, name):
            return getattr(self.instance, name)

    monkeypatch.setattr(learner.z3, "Optimize", ObservedOptimize)
    schema, positives, negatives = _guard_examples()
    first = learn_guard(
        schema, positives, negatives, GrammarLimits(), minimize=True, timeout_ms=1000
    )
    second = learn_guard(
        schema,
        positives[::-1],
        negatives,
        GrammarLimits(),
        minimize=True,
        timeout_ms=1000,
    )
    assert first is not None and first == second
    assert any(0 < item.get("timeout", 0) <= 1000 for item in settings)


def test_expired_synthesis_budget_does_not_start_unbounded_search():
    schema, positives, negatives = _guard_examples()
    with pytest.raises(SynthesisTimeout):
        learn_guard(
            schema, positives, negatives, GrammarLimits(), minimize=True, deadline=0.0
        )


def test_unknown_optimizer_is_not_a_successful_candidate(monkeypatch):
    from keyed_gram.authsynth_symbolic_analyzer import learner

    real = z3.Optimize

    class UnknownOptimize:
        def __init__(self):
            self.instance = real()

        def check(self):
            return z3.unknown

        def __getattr__(self, name):
            return getattr(self.instance, name)

    monkeypatch.setattr(learner.z3, "Optimize", UnknownOptimize)
    schema, positives, negatives = _guard_examples()
    with pytest.raises(SynthesisTimeout):
        learn_guard(schema, positives, negatives, GrammarLimits(), minimize=True)


@pytest.mark.parametrize("method_name", ["DECLARED", "PASSIVE"])
def test_unverified_baselines_do_not_make_completeness_claims(method_name):
    from keyed_gram.authsynth_symbolic_verifier.evaluator import (
        SymbolicMethod,
        _run_method,
    )

    execution = _run_method(
        SymbolicMethod[method_name],
        case_for("hidden_guarded_effect"),
        replay_budget=8,
        solver_timeout_ms=3000,
    )
    assert execution.contract_status == ContractStatus.UNKNOWN
    assert not execution.complete_claim_made


def test_guard_metrics_include_inactive_slots_as_true_negatives():
    from keyed_gram.authsynth_symbolic_verifier.evaluator import (
        SymbolicMethod,
        evaluate_cases,
    )

    bundle = evaluate_cases(
        (case_for("hidden_guarded_effect"),),
        methods=(SymbolicMethod.GOLD,),
        replay_budget=8,
        solver_timeout_ms=3000,
    )
    row = bundle["rows"][0]
    assert row.guard_true_negative_count > 0
    assert bundle["metrics"][0]["guard_accuracy"] == 1.0
