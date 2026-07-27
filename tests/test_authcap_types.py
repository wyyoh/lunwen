from __future__ import annotations

import inspect

import pytest

from keyed_gram.authcap_types import (
    Action,
    AuthCapTypeError,
    AuthenticatedPrincipal,
    AuthoritySet,
    EnvironmentContext,
    PolicyRequestContext,
    ProposalSource,
    ResourceDescriptor,
    SemanticProposal,
)


def test_authority_canonicalization_is_order_independent() -> None:
    left = AuthoritySet(
        subject_ids=frozenset({"s2", "s1"}),
        tenant_ids=frozenset({"t1"}),
        resource_ids=frozenset({"r1"}),
        actions=frozenset({"read"}),
        relation_ids=frozenset({"field"}),
        purposes=frozenset({"research"}),
    )
    right = AuthoritySet(
        subject_ids=frozenset({"s1", "s2"}),
        tenant_ids=frozenset({"t1"}),
        resource_ids=frozenset({"r1"}),
        actions=frozenset({"read"}),
        relation_ids=frozenset({"field"}),
        purposes=frozenset({"research"}),
    )
    assert left == right
    assert left.canonical() == right.canonical()


def test_empty_authority_is_explicit() -> None:
    assert AuthoritySet.empty().is_empty


@pytest.mark.parametrize("value", ["*", "**", "all", "tenant-*"])
def test_wildcard_is_rejected(value: str) -> None:
    with pytest.raises(AuthCapTypeError):
        AuthoritySet(tenant_ids=frozenset({value}))


def test_proposal_has_no_principal_or_policy_field() -> None:
    signature = inspect.signature(SemanticProposal)
    assert "principal" not in signature.parameters
    assert "policy" not in signature.parameters
    with pytest.raises(TypeError):
        SemanticProposal(
            proposal_id="proposal-1",
            proposed_resource_ids=("r1",),
            proposed_actions=("read",),
            proposed_purpose=None,
            proposed_constraints={},
            source_type=ProposalSource.LLM,
            principal="admin",  # type: ignore[call-arg]
        )


def test_cross_tenant_context_is_not_silently_rewritten() -> None:
    context = PolicyRequestContext(
        principal=AuthenticatedPrincipal("s1", "t1", {"department": "d1"}),
        requested_resource=ResourceDescriptor("t2", "r1", "document", "field"),
        requested_action=Action.READ,
        purpose="research",
        environment=EnvironmentContext(
            "2026-01-01T00:00:00Z", {"zone": "z1"}
        ),
        policy_epoch=1,
    )
    assert context.principal.tenant_id == "t1"
    assert context.requested_resource.tenant_id == "t2"
