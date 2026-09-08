from __future__ import annotations

from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_analyzer import SymbolicContractLearner
from keyed_gram.authsynth_symbolic_analyzer.learner import learn_guard
from keyed_gram.authsynth_symbolic_shared import (
    AnalyzerEvidence,
    BoundedSchema,
    ConcreteAssignment,
    DomainField,
    GrammarLimits,
)
from keyed_gram.authsynth_symbolic_verifier.benchmark import enumerate_assignments
from keyed_gram.authsynth_symbolic_verifier.replay_service import SymbolicSandboxReplay


def test_examples_only_learner_produces_guarded_symbolic_contract() -> None:
    case = case_for("hidden_guarded_effect")
    replay = SymbolicSandboxReplay((case,))
    assignments = tuple(enumerate_assignments(case.analyzer_input.schema))[:24]
    evidence = []
    for assignment in assignments:
        result = replay.replay(case.analyzer_input.case_handle, assignment)
        evidence.append(
            AnalyzerEvidence(
                assignment,
                result.events,
                result.state_diff,
                result.observed_version_digest,
            )
        )
    learned = SymbolicContractLearner().fit(
        tool_id=case.analyzer_input.tool_id,
        schema=case.analyzer_input.schema,
        version_digest=case.implementation.version_digest,
        limits=case.analyzer_input.grammar_limits,
        evidence=evidence,
        input_schema_digest=case.reference_contract.input_schema_digest,
        state_schema_digest=case.reference_contract.state_schema_digest,
    )
    assert learned is not None
    assert learned.clauses
    assert all(
        "query" not in str(item.to_dict()).casefold() for item in learned.clauses
    )


def test_grammar_insufficiency_returns_none_instead_of_lookup_table() -> None:
    schema = BoundedSchema(
        (
            DomainField("x", "input", "bool"),
            DomainField("y", "input", "bool"),
        )
    )
    values = [
        ConcreteAssignment((("x", x), ("y", y)), ())
        for x in (False, True)
        for y in (False, True)
    ]
    positives = [
        item for item in values if item.value("input", "x") != item.value("input", "y")
    ]
    negatives = [item for item in values if item not in positives]
    result = learn_guard(
        schema,
        positives,
        negatives,
        GrammarLimits(
            max_disjuncts=1, max_literals_per_conjunction=2, max_total_literals=2
        ),
        minimize=True,
    )
    assert result is None


def test_field_binding_infers_input_variable_not_concrete_value() -> None:
    case = case_for("parameter_role_alias")
    from keyed_gram.authsynth_symbolic_analyzer.cegis import AuthSynthSymbolicAnalyzer
    from keyed_gram.authsynth_symbolic_verifier.equivalence import (
        BoundedSymbolicVerifier,
    )

    outcome = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input,
        BoundedSymbolicVerifier((case,), timeout_ms=3000),
        SymbolicSandboxReplay((case,)),
    )
    send = next(
        item.effect
        for item in outcome.candidate_contract.clauses
        if item.effect.kind == "send"
    )
    assert send.destination.variable is not None
    assert send.destination.variable.name == "destination"
