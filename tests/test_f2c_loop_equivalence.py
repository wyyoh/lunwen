"""普通单元测试的小域穷举，不涉及 benchmark development/locked。"""

from dataclasses import replace

import pytest
from f2c_symbolic_helpers import case_for

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula as F,
)
from keyed_gram.authsynth_symbolic_shared import (
    BoundedSchema,
    ConcreteAssignment,
    Conjunction,
    DomainField,
    Literal,
    SymbolicEffectClause,
    SymbolicEffectContract,
    SymbolicEffectTemplate,
    SymbolicStateUpdate,
    SymbolicStateUpdateClause,
    VariableRef,
    VerificationKind,
    canonical_digest,
)
from keyed_gram.authsynth_symbolic_shared import (
    LiteralOperator as Op,
)
from keyed_gram.authsynth_symbolic_shared import (
    SymbolicTerm as T,
)
from keyed_gram.authsynth_symbolic_verifier.benchmark import enumerate_assignments
from keyed_gram.authsynth_symbolic_verifier.bounded_loop import (
    BoundedLoop,
    LoopUpdate,
    UnsupportedLoop,
    execute_program,
)
from keyed_gram.authsynth_symbolic_verifier.equivalence import BoundedSymbolicVerifier
from keyed_gram.authsynth_symbolic_verifier.hidden_ir import (
    GuardedTransition,
    HiddenSymbolicImplementation,
)
from keyed_gram.authsynth_symbolic_verifier.symbolic_loop import symbolic_execute


def guard(name, value, op=Op.EQ):
    return F((Conjunction((Literal(VariableRef("state", name), op, value),)),))


def program(bound=4, *, decrement=False, delayed=True, toggle_exit=False):
    schema = BoundedSchema(
        (
            DomainField("enabled", "input", "bool"),
            DomainField("count", "state", "int", minimum=0, maximum=4),
            DomainField("flag", "state", "bool"),
            DomainField("status", "state", "enum", ("pending", "done")),
        )
    )
    event = SymbolicEffectTemplate(
        "iteration",
        0,
        "immediate",
        "read",
        T.const("tenant"),
        T.const("resource"),
        amount=T.var("state", "count"),
    )
    acknowledgement = replace(
        event,
        slot="receipt",
        sequence=1,
        phase="delayed",
        kind="write",
        parent_slot=event.slot,
    )
    updates = (
        LoopUpdate("count", T.var("state", "count"), -1 if decrement else 1),
        LoopUpdate("status", T.const("done")),
        LoopUpdate(
            "flag",
            T.ite(
                Literal(VariableRef("input", "enabled"), Op.EQ, True),
                T.const(False),
                T.const(True),
            ),
        ),
    )
    loop_guard = guard("count", 1, Op.GE) if decrement else guard("count", 2, Op.LE)
    if toggle_exit:
        loop_guard = guard("flag", True)
        updates = (LoopUpdate("flag", T.const(False)),)
    body = GuardedTransition(
        "body", F.true(), (event,), (acknowledgement,) if delayed else (), updates
    )
    after = GuardedTransition(
        "after",
        guard("count", 3, Op.GE),
        (replace(event, slot="final", amount=T.var("state", "count")),),
    )
    return HiddenSymbolicImplementation(
        "unit-loop",
        schema,
        canonical_digest("loop-version"),
        (),
        bounded_loop_max=bound,
        delayed_depth=int(delayed),
        loop=BoundedLoop(loop_guard, (body,), bound),
        after_loop=(after,),
    )


def assignment(count=0, flag=True):
    return ConcreteAssignment(
        (("enabled", True),), (("count", count), ("flag", flag), ("status", "pending"))
    )


@pytest.mark.parametrize(
    "initial,bound,expected,reason",
    [
        (3, 4, 0, "GUARD_FALSE"),
        (2, 4, 1, "GUARD_FALSE"),
        (0, 4, 3, "GUARD_FALSE"),
        (0, 3, 3, "BOUND_REACHED"),
        (0, 0, 0, "BOUND_REACHED"),
    ],
)
def test_loop_zero_one_multiple_and_exact_bound(initial, bound, expected, reason):
    result = execute_program(program(bound), assignment(initial))
    assert (result.iteration_count, result.termination_reason) == (expected, reason)
    assert dict(result.final_state)["count"] == initial + expected


def test_state_recurrence_order_fields_and_delayed_parent():
    result = execute_program(program(), assignment())
    iterations = [
        e for e in result.events if e.kind == "read" and e.slot.startswith("loop-")
    ]
    assert [e.amount for e in iterations] == [0, 1, 2]
    assert [e.sequence for e in result.events] == sorted(
        {e.sequence for e in result.events}
    )
    for e in result.events:
        if e.phase == "delayed":
            assert e.parent_slot == e.slot.replace("receipt", "iteration")
    assert result.events[-1].slot == "post-final"
    assert result.events[-1].amount == 3


def test_guard_false_never_reactivates_and_bool_enum_updates():
    p = program(toggle_exit=True)
    a = execute_program(p, assignment(flag=True))
    b = execute_program(p, assignment(flag=False))
    assert a.iteration_count == 1 and b.iteration_count == 0
    assert dict(a.final_state)["flag"] is False
    assert (
        dict(execute_program(program(), assignment()).final_state)["status"] == "done"
    )


@pytest.mark.parametrize("bound", [None, 9])
def test_unsupported_loop_is_unknown_not_unrolled_twice(bound):
    p = program(bound)
    with pytest.raises(UnsupportedLoop):
        execute_program(p, assignment())
    with pytest.raises(UnsupportedLoop):
        symbolic_execute(p)


def test_integer_overflow_rejected_without_wrapping():
    p = program(1)
    p = replace(p, loop=replace(p.loop, guard=F.true()))
    with pytest.raises(UnsupportedLoop):
        execute_program(p, assignment(4))
    with pytest.raises(UnsupportedLoop):
        symbolic_execute(p).concrete_at(assignment(4))


def test_cutoff_is_not_a_termination_proof():
    p = program(1)
    p = replace(p, loop=replace(p.loop, require_guard_false_at_bound=True))
    with pytest.raises(UnsupportedLoop):
        execute_program(p, assignment(0))
    with pytest.raises(UnsupportedLoop):
        symbolic_execute(p).concrete_at(assignment(0))


def differential_audit():
    count, failures = 0, []
    programs = [
        program(n, decrement=d, delayed=a)
        for n in (0, 1, 3, 4, 8)
        for d in (False, True)
        for a in (False, True)
    ]
    programs += [program(n, toggle_exit=True) for n in (0, 1, 4, 8)]
    for index, p in enumerate(programs):
        symbolic = symbolic_execute(p)
        for a in enumerate_assignments(p.schema):
            count += 1
            concrete, evaluated = execute_program(p, a), symbolic.concrete_at(a)
            if concrete != evaluated:
                failures.append({"program_index": index, "assignment_digest": a.digest})
    return {
        "program_count": len(programs),
        "exhaustive_assignment_count": count,
        "concrete_symbolic_loop_equivalence_failure_count": len(failures),
        "failures": failures,
        "formal_development_started": False,
        "formal_locked_started": False,
    }


def test_concrete_symbolic_equivalence_exhaustive():
    result = differential_audit()
    assert result["exhaustive_assignment_count"] == 960
    assert result["concrete_symbolic_loop_equivalence_failure_count"] == 0, result


def loop_case():
    p = replace(program(2, delayed=False, toggle_exit=True), after_loop=())
    g = p.loop.guard
    event = replace(p.loop.body[0].immediate_effects[0], slot="loop-0-iteration")
    contract = SymbolicEffectContract(
        p.tool_id,
        p.version_digest,
        canonical_digest({"fields": [f.to_dict() for f in p.schema.input_fields]}),
        canonical_digest({"fields": [f.to_dict() for f in p.schema.state_fields]}),
        (SymbolicEffectClause(g, event),),
        (SymbolicStateUpdateClause(g, SymbolicStateUpdate("flag", T.const(False))),),
    )
    base = case_for("exact_declared_control")
    inp = replace(
        base.analyzer_input,
        tool_id=p.tool_id,
        schema=p.schema,
        declared_contract=contract,
        static_hypotheses=(),
        expected_version_digest=p.version_digest,
    )
    return replace(
        base, analyzer_input=inp, implementation=p, reference_contract=contract
    )


def test_real_bounded_loop_can_be_certified_and_omission_is_detected():
    c = loop_case()
    verifier = BoundedSymbolicVerifier((c,), timeout_ms=3000)
    assert (
        verifier.check(c.analyzer_input.case_handle, c.reference_contract).kind
        == VerificationKind.EQUIVALENT
    )
    assert (
        verifier.check(
            c.analyzer_input.case_handle, replace(c.reference_contract, clauses=())
        ).kind
        == VerificationKind.MISSING_EFFECT
    )


def test_unknown_bound_verifier_and_replay_do_not_claim_success():
    from keyed_gram.authsynth_symbolic_verifier.replay_service import (
        SymbolicSandboxReplay,
    )

    c = loop_case()
    c = replace(
        c,
        implementation=replace(
            c.implementation,
            bounded_loop_max=9,
            loop=replace(c.implementation.loop, max_iterations=9),
        ),
    )
    assert (
        BoundedSymbolicVerifier((c,), timeout_ms=3000)
        .check(c.analyzer_input.case_handle, c.reference_contract)
        .kind
        == VerificationKind.UNKNOWN
    )
    result = SymbolicSandboxReplay((c,)).replay(
        c.analyzer_input.case_handle, assignment()
    )
    assert result.exit_status == "unknown" and not result.bounded_quiescence_reached


def test_replay_keeps_global_sequence_not_lexical_slot_order():
    from keyed_gram.authsynth_symbolic_verifier.replay_service import (
        SymbolicSandboxReplay,
    )

    c = loop_case()
    before = replace(c.implementation.loop.body[0], state_updates=())
    p = replace(c.implementation, transitions=(before,))
    c = replace(c, implementation=p)
    result = SymbolicSandboxReplay((c,)).replay(
        c.analyzer_input.case_handle, assignment()
    )
    assert [e.sequence for e in result.events] == [0, 1]
    assert result.events[0].slot.startswith("pre-")
    assert result.iteration_count == 1 and result.termination_reason == "GUARD_FALSE"


def test_config_freezes_exact_loop_limit_and_integer_semantics():
    from pathlib import Path

    import yaml

    from keyed_gram.authsynth_symbolic_verifier.bounded_loop import MAX_LOOP_ITERATIONS

    config = yaml.safe_load(Path("configs/stage_f2c.yaml").read_text())
    assert config["loop_semantics"]["max_loop_iterations"] == MAX_LOOP_ITERATIONS == 8
    assert (
        config["loop_semantics"]["integer_semantics"]
        == "bounded_mathematical_int_reject_overflow"
    )
