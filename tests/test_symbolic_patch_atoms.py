"""原子定位与独立真值的回归，不物化或评分 development/locked。"""

from dataclasses import replace
from functools import lru_cache

import pytest
from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_analyzer import AuthSynthSymbolicAnalyzer
from keyed_gram.authsynth_symbolic_analyzer.diagnostics import diagnose_replay
from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    PatchAtom,
    PatchRecord,
    StateChange,
    StructuredStateDiff,
    SymbolicEffectClause,
    SymbolicTerm,
)
from keyed_gram.authsynth_symbolic_verifier.benchmark import enumerate_assignments
from keyed_gram.authsynth_symbolic_verifier.equivalence import BoundedSymbolicVerifier
from keyed_gram.authsynth_symbolic_verifier.evaluator import (
    SymbolicMethod,
    aggregate_metrics,
    evaluate_cases,
)
from keyed_gram.authsynth_symbolic_verifier.replay_service import SymbolicSandboxReplay


def _sample(category="parameter_role_alias"):
    case = case_for(category)
    replay = SymbolicSandboxReplay((case,))
    assignment = next(
        a
        for a in enumerate_assignments(case.analyzer_input.schema)
        if a.value("input", "mode") == "admin"
        and a.value("input", "destination") == "external"
    )
    return case, replay.replay(case.analyzer_input.case_handle, assignment)


def test_two_same_type_omissions_are_two_addressable_atoms():
    case = case_for("multi_branch_effect")
    effects = [a for a in case.omission_atoms if a.patch_type == "effect_patch"]
    assert len(effects) == 2
    assert len({a.target_id for a in effects}) == 2
    assert len({a.digest for a in effects}) == 2


def test_ground_truth_does_not_depend_on_mutation_category_or_analyzer(monkeypatch):
    from keyed_gram.authsynth_symbolic_analyzer import diagnostics

    case = case_for("multi_branch_effect")
    expected = case.omission_atoms
    monkeypatch.setattr(diagnostics, "diagnose_replay", lambda *_: ())
    relabeled = replace(case, mutation_category="arbitrary-category")
    assert relabeled.omission_atoms == expected


def test_field_binding_atom_identifies_exact_field_not_only_type():
    case, replay = _sample()
    atoms = diagnose_replay(case.analyzer_input.declared_contract, replay)
    assert len(atoms) == 1
    assert atoms[0].patch_type == "field_binding_patch"
    assert atoms[0].component == "destination"
    assert atoms[0] in case.omission_atoms


def test_bool_int_field_difference_is_not_erased():
    case, replay = _sample()
    event = replay.events[0]
    template = case.reference_contract.clauses[0].effect
    template = replace(
        template,
        slot=event.slot,
        sequence=event.sequence,
        phase=event.phase,
        kind=event.kind,
        parent_slot=event.parent_slot,
        amount=SymbolicTerm.const(True),
    )
    contract = replace(
        case.reference_contract,
        clauses=(SymbolicEffectClause(BooleanFormula.true(), template),),
        state_updates=(),
    )
    predicted, _ = contract.predict(replay.assignment)
    actual = replace(predicted[0], amount=1)
    replay = replace(
        replay,
        immediate_events=(actual,),
        delayed_events=(),
        state_diff=StructuredStateDiff(()),
    )
    atoms = diagnose_replay(contract, replay)
    assert {a.component for a in atoms} == {"amount"}


def test_state_atoms_identify_different_fields():
    case, replay = _sample()
    contract = replace(case.reference_contract, clauses=(), state_updates=())
    replay = replace(
        replay,
        immediate_events=(),
        delayed_events=(),
        state_diff=StructuredStateDiff(
            (
                StateChange(
                    "record",
                    "state",
                    "counter",
                    replay.assignment.value("state", "counter"),
                    3,
                ),
                StateChange(
                    "record",
                    "state",
                    "flag",
                    replay.assignment.value("state", "flag"),
                    True,
                ),
            )
        ),
    )
    atoms = diagnose_replay(contract, replay)
    assert {a.target_id for a in atoms} == {"counter", "flag"}


def test_empty_control_has_no_omission_atoms_even_when_instrumentation_is_unknown():
    assert not case_for("exact_declared_control").omission_atoms


def test_joint_binding_swap_is_detected_when_each_projection_is_unchanged():
    case, replay = _sample()
    template = case.reference_contract.clauses[0].effect
    template = replace(
        template, tenant=SymbolicTerm.const("a"), resource=SymbolicTerm.const("x")
    )
    second = replace(
        template, tenant=SymbolicTerm.const("b"), resource=SymbolicTerm.const("y")
    )
    contract = replace(
        case.reference_contract,
        clauses=tuple(
            SymbolicEffectClause(BooleanFormula.true(), t) for t in (template, second)
        ),
        state_updates=(),
    )
    events, _ = contract.predict(replay.assignment)
    actual = (replace(events[0], resource="y"), replace(events[1], resource="x"))
    replay = replace(
        replay,
        immediate_events=actual,
        delayed_events=(),
        state_diff=StructuredStateDiff(()),
    )
    atoms = diagnose_replay(contract, replay)
    assert len(atoms) == 1 and atoms[0].component == "joint_binding"


@lru_cache(maxsize=1)
def _row():
    return evaluate_cases(
        (case_for("multi_branch_effect"),),
        replay_budget=16,
        solver_timeout_ms=3000,
        methods=(SymbolicMethod.AUTHSYNTH,),
    )["rows"][0]


def test_type_recall_one_does_not_imply_atom_recall_one():
    row = replace(
        _row(),
        predicted_patch_types=("effect_patch",),
        expected_patch_types=("effect_patch",),
        predicted_patch_atom_digests=("first",),
        expected_patch_atom_digests=("first", "second"),
        discovered_omission_atom_digests=("first",),
        expected_omission_atom_digests=("first", "second"),
    )
    metrics = aggregate_metrics((row,))[0]
    assert metrics["patch_type_recall"] == 1
    assert metrics["patch_atom_recall"] == 0.5
    assert metrics["omission_atom_discovery_recall"] == 0.5
    assert metrics["exact_patch_set_rate"] == 0


def test_right_patch_type_wrong_location_gets_zero_atom_precision():
    row = replace(
        _row(),
        predicted_patch_types=("field_binding_patch",),
        expected_patch_types=("field_binding_patch",),
        predicted_patch_atom_digests=("wrong-resource",),
        expected_patch_atom_digests=("right-resource",),
    )
    metrics = aggregate_metrics((row,))[0]
    assert metrics["patch_type_precision"] == 1
    assert metrics["patch_atom_precision"] == 0
    assert metrics["exact_patch_set_rate"] == 0


def test_duplicate_evidence_does_not_count_one_atom_twice():
    row = replace(
        _row(),
        expected_omission_atom_digests=("one", "two"),
        discovered_omission_atom_digests=("one", "one", "one"),
    )
    metrics = aggregate_metrics((row,))[0]
    assert metrics["discovered_omission_atom_count"] == 1
    assert metrics["omission_atom_discovery_recall"] == 0.5


def test_actual_cegis_discovers_atoms_without_reading_reference_contract():
    case = case_for("multi_branch_effect")
    outcome = AuthSynthSymbolicAnalyzer().analyze(
        case.analyzer_input,
        BoundedSymbolicVerifier((case,), timeout_ms=3000),
        SymbolicSandboxReplay((case,)),
    )
    assert set(outcome.discovered_atoms) == case.omission_atoms
    assert all(record.atom is not None for record in outcome.patch_records)


@pytest.mark.parametrize(
    "args",
    [
        ("unknown", "effect", "slot", "presence"),
        ("effect_patch", "state", "slot", "presence"),
        ("effect_patch", "effect", "*", "presence"),
        ("field_binding_patch", "effect", "slot", "nonexistent-field"),
    ],
)
def test_patch_atom_schema_rejects_invalid_or_free_text_scope(args):
    with pytest.raises(ValueError):
        PatchAtom(*args)


def test_record_atom_type_mismatch_is_rejected():
    atom = PatchAtom("state_update_patch", "state", "counter", "final_value")
    with pytest.raises(ValueError):
        PatchRecord("effect_patch", "evidence-digest", atom)


def test_unlocated_record_is_counted_as_false_positive_not_discarded(monkeypatch):
    from keyed_gram.authsynth_symbolic_verifier import evaluator
    from keyed_gram.authsynth_symbolic_verifier.evaluator import (
        _evaluate_case_method,
        _run_method,
    )

    case = case_for("multi_branch_effect")
    execution = _run_method(
        SymbolicMethod.AUTHSYNTH, case, replay_budget=16, solver_timeout_ms=3000
    )
    assert execution.outcome is not None
    outcome = replace(
        execution.outcome,
        patch_records=(PatchRecord("effect_patch", "unlocated-evidence"),),
    )
    monkeypatch.setattr(
        evaluator,
        "_run_method",
        lambda *_args, **_kwargs: replace(execution, outcome=outcome),
    )
    row = _evaluate_case_method(
        case, SymbolicMethod.AUTHSYNTH, replay_budget=16, solver_timeout_ms=3000
    )
    metrics = aggregate_metrics((row,))[0]
    assert metrics["unlocated_patch_record_count"] == 1
    assert metrics["predicted_patch_atom_count"] == 1
    assert metrics["patch_atom_precision"] == 0
    assert metrics["omission_atom_discovery_recall"] == 1


def test_atom_counts_are_per_case_not_globally_deduplicated():
    row = replace(
        _row(),
        expected_omission_atom_digests=("same-location",),
        discovered_omission_atom_digests=("same-location",),
    )
    second = replace(row, case_id="another-case", discovered_omission_atom_digests=())
    metrics = aggregate_metrics((row, second))[0]
    assert metrics["expected_omission_atom_count"] == 2
    assert metrics["discovered_omission_atom_count"] == 1
    assert metrics["omission_atom_discovery_recall"] == 0.5
