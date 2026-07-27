from __future__ import annotations

from dataclasses import replace
from threading import Barrier, Thread

from f2_helpers import compile_parts, load_train_case

from keyed_gram.authcap_compiler import (
    CompiledAllow,
    CompiledDeny,
    trusted_request_envelope,
)
from keyed_gram.authcap_types import Action, ProposalSource, SemanticProposal
from keyed_gram.authcap_workflow import (
    WorkflowAuthCapCompiler,
    WorkflowAuthorityState,
    WorkflowStateStore,
)


def _workflow_fixture():
    case = load_train_case(lambda item: item["workflow"] is not None)
    (
        context,
        policy_set,
        _,
        _,
        resources,
        _,
        _,
        compiler,
        _,
    ) = compile_parts(case)
    steps = []
    for item in case["workflow"]["steps"]:
        step_context = replace(
            context,
            requested_action=Action(item["requested_action"]),
        )
        proposal = SemanticProposal(
            f"proposal-step-{item['step']}",
            (context.requested_resource.resource_id,),
            (item["requested_action"],),
            context.purpose,
            {},
            ProposalSource.RULE,
        )
        envelope = trusted_request_envelope(
            step_context,
            request_id=f"request-step-{item['step']}",
            request_nonce=f"nonce-step-{item['step']}",
        )
        steps.append((step_context, proposal, envelope))
    state = WorkflowAuthorityState.create(
        workflow_id=case["workflow"]["workflow_id"],
        subject_id=context.principal.subject_id,
        policy_epoch=context.policy_epoch,
    )
    return case, context, policy_set, resources, compiler, steps, state


def _run(
    workflow_compiler,
    state,
    number,
    step,
    policy_set,
    resources,
    combined=None,
    expected_workflow_id=None,
):
    context, proposal, envelope = step
    return workflow_compiler.compile_next(
        state=state,
        step_id=f"step-{number}",
        step_number=number,
        expected_workflow_id=expected_workflow_id or state.workflow_id,
        proposal=proposal,
        trusted_context=context,
        trusted_request=envelope,
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at="2026-07-01T12:00:00Z",
        combined_context=combined,
    )


def test_two_allowed_steps_with_allowed_composition():
    _, _, policy, resources, compiler, steps, state0 = _workflow_fixture()
    store = WorkflowStateStore()
    store.initialize(state0)
    workflow = WorkflowAuthCapCompiler(compiler, store)
    first, state1 = _run(workflow, state0, 1, steps[0], policy, resources)
    assert isinstance(first, CompiledAllow)
    second, state2 = _run(
        workflow,
        state1,
        2,
        steps[1],
        policy,
        resources,
        combined=steps[1][0],
    )
    assert isinstance(second, CompiledAllow)
    assert state2.completed_steps == ("step-1", "step-2")


def test_two_allowed_steps_with_combined_deny_are_blocked():
    _, combined, policy, resources, compiler, steps, state0 = _workflow_fixture()
    store = WorkflowStateStore()
    store.initialize(state0)
    workflow = WorkflowAuthCapCompiler(compiler, store)
    _, state1 = _run(workflow, state0, 1, steps[0], policy, resources)
    second, state2 = _run(
        workflow,
        state1,
        2,
        steps[1],
        policy,
        resources,
        combined=combined,
    )
    assert isinstance(second, CompiledDeny)
    assert second.decision_reason_code == "combined_policy_deny"
    assert state2 is None


def test_step_replay_and_state_fork_are_rejected():
    _, _, policy, resources, compiler, steps, state0 = _workflow_fixture()
    store = WorkflowStateStore()
    store.initialize(state0)
    workflow = WorkflowAuthCapCompiler(compiler, store)
    first, _ = _run(workflow, state0, 1, steps[0], policy, resources)
    replay, _ = _run(workflow, state0, 1, steps[0], policy, resources)
    fork, _ = _run(workflow, replace(state0), 1, steps[0], policy, resources)
    assert isinstance(first, CompiledAllow)
    assert isinstance(replay, CompiledDeny)
    assert isinstance(fork, CompiledDeny)


def test_old_epoch_out_of_order_cross_workflow_and_digest_mutation_rejected():
    _, _, policy, resources, compiler, steps, state0 = _workflow_fixture()
    candidates = (
        WorkflowAuthorityState.create(
            workflow_id=state0.workflow_id,
            subject_id=state0.subject_id,
            policy_epoch=state0.policy_epoch - 1,
        ),
        state0,
        WorkflowAuthorityState.create(
            workflow_id="different-workflow",
            subject_id=state0.subject_id,
            policy_epoch=state0.policy_epoch,
        ),
        replace(state0, state_digest="0" * 64),
    )
    numbers = (1, 2, 1, 1)
    for candidate, number in zip(candidates, numbers, strict=True):
        store = WorkflowStateStore()
        store.initialize(candidate) if candidate.state_digest != "0" * 64 else None
        workflow = WorkflowAuthCapCompiler(compiler, store)
        result, _ = _run(
            workflow,
            candidate,
            number,
            steps[0],
            policy,
            resources,
            expected_workflow_id=state0.workflow_id,
        )
        assert isinstance(result, CompiledDeny)


def test_concurrent_branch_only_one_capability_is_issued():
    _, _, policy, resources, compiler, steps, state0 = _workflow_fixture()
    store = WorkflowStateStore()
    store.initialize(state0)
    workflow = WorkflowAuthCapCompiler(compiler, store)
    barrier = Barrier(2)
    results = []

    def worker():
        barrier.wait()
        results.append(_run(workflow, state0, 1, steps[0], policy, resources)[0])

    threads = [Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(isinstance(item, CompiledAllow) for item in results) == 1
    assert sum(isinstance(item, CompiledDeny) for item in results) == 1
