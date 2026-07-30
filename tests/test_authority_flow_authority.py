from __future__ import annotations

import pytest

from keyed_gram.authority_flow.authority import (
    AuthorityAtom,
    AuthorityBudget,
    AuthorityFlowError,
    AuthorityOrigin,
    AuthorityResource,
    BranchState,
    ForbiddenEffectCombination,
    InfluenceGuard,
    LinearAuthorityContext,
    consume,
    delegate,
    issue_authenticated,
    merge_branches,
    split,
)
from keyed_gram.authority_flow.labels import (
    Confidentiality,
    DataLabel,
    InfluenceOrigin,
    Integrity,
)

TRUSTED_CONTROL = DataLabel(
    Confidentiality.INTERNAL,
    Integrity.TRUSTED,
    (InfluenceOrigin("trusted_ui", "request-1", False),),
)


def atom(**overrides: object) -> AuthorityAtom:
    values = {
        "issuer": "policy-authority",
        "subject": "agent-a",
        "tenant": "tenant-a",
        "resource": "email/account-7",
        "action": "send",
        "purpose": "customer-support",
        "epoch": 42,
        "influence_guard": InfluenceGuard(
            minimum_integrity=Integrity.TRUSTED,
            allowed_source_kinds=("trusted_ui",),
        ),
    }
    values.update(overrides)
    return AuthorityAtom(**values)


def origin(authenticated: bool = True) -> AuthorityOrigin:
    return AuthorityOrigin(
        grant_id="grant-1",
        issuer="policy-authority",
        authenticated=authenticated,
    )


def budget(
    uses: int = 1,
    *,
    depth: int = 0,
    resources: int = 1,
    ttl: int = 8,
) -> AuthorityBudget:
    return AuthorityBudget(
        uses=uses,
        amount_microunits=uses * 100,
        data_bytes=uses * 1024,
        resource_count=resources,
        delegation_depth=depth,
        ttl_steps=ttl,
    )


def test_only_authenticated_origin_can_issue() -> None:
    with pytest.raises(AuthorityFlowError, match="authenticated"):
        issue_authenticated(atom(), budget(), origin(False))


def test_wildcard_scope_is_rejected() -> None:
    with pytest.raises(AuthorityFlowError, match="wildcard"):
        atom(resource="*")


def test_budget_partial_order_detects_expansion() -> None:
    maximum = budget(2, depth=2, resources=2)
    assert budget(1, depth=1, resources=1).attenuates(maximum)
    assert not budget(3, depth=1, resources=1).attenuates(maximum)
    assert not budget(1, depth=3, resources=1).attenuates(maximum)


def test_split_conserves_additive_budget_and_uses_unique_children() -> None:
    context = issue_authenticated(
        atom(),
        budget(2, depth=2, resources=2),
        origin(),
    )
    parent = context.resources[0]
    child_budget = budget(1, depth=1, resources=1)
    updated, children = split(
        context,
        parent.resource_id,
        (child_budget, child_budget),
    )
    assert len(updated.resources) == 2
    assert len({item.resource_id for item in children}) == 2
    assert all(parent.resource_id in item.lineage for item in children)


def test_split_rejects_budget_duplication() -> None:
    context = issue_authenticated(atom(), budget(1, resources=1), origin())
    parent = context.resources[0]
    with pytest.raises(AuthorityFlowError, match="总和"):
        split(context, parent.resource_id, (budget(), budget()))


def test_delegate_consumes_parent_and_reduces_depth() -> None:
    context = issue_authenticated(atom(), budget(1, depth=2), origin())
    parent = context.resources[0]
    updated, child = delegate(
        context,
        parent.resource_id,
        "agent-child",
        budget(1, depth=1),
    )
    assert child.atom.subject == "agent-child"
    assert parent.resource_id not in {item.resource_id for item in updated.resources}
    assert child.origin == parent.origin


def test_delegate_fork_is_impossible_after_parent_consumption() -> None:
    context = issue_authenticated(atom(), budget(1, depth=2), origin())
    parent = context.resources[0]
    updated, _ = delegate(
        context,
        parent.resource_id,
        "agent-child-a",
        budget(1, depth=1),
    )
    with pytest.raises(AuthorityFlowError, match="不存在或已消费"):
        delegate(
            updated,
            parent.resource_id,
            "agent-child-b",
            budget(1, depth=1),
        )


def test_consume_requires_exact_scope_and_removes_exhausted_resource() -> None:
    context = issue_authenticated(atom(), budget(), origin())
    resource = context.resources[0]
    updated, effect = consume(
        context,
        resource.resource_id,
        effect_id="effect-1",
        subject="agent-a",
        tenant="tenant-a",
        resource="email/account-7",
        action="send",
        purpose="customer-support",
        epoch=42,
        cost=budget(),
        control_label=TRUSTED_CONTROL,
    )
    assert not updated.resources
    assert effect.origin_grant_id == "grant-1"


def test_cross_scope_consume_is_rejected() -> None:
    context = issue_authenticated(atom(), budget(), origin())
    resource = context.resources[0]
    with pytest.raises(AuthorityFlowError, match="作用域"):
        consume(
            context,
            resource.resource_id,
            effect_id="effect-x",
            subject="agent-a",
            tenant="tenant-b",
            resource="email/account-7",
            action="send",
            purpose="customer-support",
            epoch=42,
            cost=budget(),
            control_label=TRUSTED_CONTROL,
        )


def test_high_integrity_data_is_not_authority_without_eligible_origin() -> None:
    context = issue_authenticated(atom(), budget(), origin())
    resource = context.resources[0]
    echoed_attacker_data = DataLabel(
        Confidentiality.PUBLIC,
        Integrity.TRUSTED,
        (InfluenceOrigin("untrusted_web", "echo-1", True),),
    )
    with pytest.raises(AuthorityFlowError, match="influence"):
        consume(
            context,
            resource.resource_id,
            effect_id="effect-echo",
            subject="agent-a",
            tenant="tenant-a",
            resource="email/account-7",
            action="send",
            purpose="customer-support",
            epoch=42,
            cost=budget(),
            control_label=echoed_attacker_data,
        )


def test_linear_context_rejects_duplicate_resource() -> None:
    context = issue_authenticated(atom(), budget(), origin())
    resource = context.resources[0]
    with pytest.raises(AuthorityFlowError, match="重复"):
        LinearAuthorityContext((resource, resource))


def test_naive_branch_copy_is_rejected_at_merge() -> None:
    parent = issue_authenticated(atom(), budget(), origin())
    left = BranchState("left", parent.digest, parent)
    right = BranchState("right", parent.digest, parent)
    with pytest.raises(AuthorityFlowError, match="重复持有"):
        merge_branches(parent, (left, right))


def test_partitioned_branch_merge_is_allowed() -> None:
    parent = issue_authenticated(
        atom(),
        budget(2, depth=1, resources=2),
        origin(),
    )
    split_context, children = split(
        parent,
        parent.resources[0].resource_id,
        (
            budget(1, resources=1),
            budget(1, resources=1),
        ),
    )
    left = LinearAuthorityContext((children[0],))
    right = LinearAuthorityContext((children[1],))
    merged, effects = merge_branches(
        parent,
        (
            BranchState("left", parent.digest, left),
            BranchState("right", parent.digest, right),
        ),
    )
    assert merged == split_context
    assert effects == ()


def test_combined_deny_blocks_individually_scoped_effects() -> None:
    read_parent = issue_authenticated(
        atom(resource="customer/7", action="read"),
        budget(),
        origin(),
    )
    send_origin = AuthorityOrigin("grant-2", "policy-authority", True)
    send_parent = issue_authenticated(atom(), budget(), send_origin)
    parent = LinearAuthorityContext(
        read_parent.resources + send_parent.resources
    )
    read_ctx, read_effect = consume(
        LinearAuthorityContext(read_parent.resources),
        read_parent.resources[0].resource_id,
        effect_id="read-1",
        subject="agent-a",
        tenant="tenant-a",
        resource="customer/7",
        action="read",
        purpose="customer-support",
        epoch=42,
        cost=budget(),
        control_label=TRUSTED_CONTROL,
    )
    send_ctx, send_effect = consume(
        LinearAuthorityContext(send_parent.resources),
        send_parent.resources[0].resource_id,
        effect_id="send-1",
        subject="agent-a",
        tenant="tenant-a",
        resource="email/account-7",
        action="send",
        purpose="customer-support",
        epoch=42,
        cost=budget(),
        control_label=TRUSTED_CONTROL,
    )
    deny = ForbiddenEffectCombination(
        "deny-exfiltration",
        frozenset({read_effect.signature, send_effect.signature}),
    )
    with pytest.raises(AuthorityFlowError, match="combined deny"):
        merge_branches(
            parent,
            (
                BranchState("left", parent.digest, read_ctx, (read_effect,)),
                BranchState("right", parent.digest, send_ctx, (send_effect,)),
            ),
            combined_denies=(deny,),
        )


def test_resource_constructor_requires_lineage_binding() -> None:
    with pytest.raises(AuthorityFlowError, match="lineage"):
        AuthorityResource(
            resource_id="auth-x",
            atom=atom(),
            origin=origin(),
            budget=budget(),
            lineage=("auth-other",),
        )
