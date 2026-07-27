from __future__ import annotations

import pytest

from keyed_gram.authcap_authority import (
    AuthorityInvariantError,
    compositional_non_amplification,
    is_authority_subset,
    require_authority_subset,
    require_fresh_witness,
    semantic_proposal_authorizes_execution,
    union_authorities,
)
from keyed_gram.authcap_types import (
    AuthoritySet,
    AuthorizationWitness,
    Constraint,
    ProposalSource,
    SemanticProposal,
)


def authority(**overrides: object) -> AuthoritySet:
    values = {
        "subject_ids": frozenset({"s1"}),
        "tenant_ids": frozenset({"t1"}),
        "resource_ids": frozenset({"r1"}),
        "actions": frozenset({"read"}),
        "relation_ids": frozenset({"field"}),
        "purposes": frozenset({"research"}),
        "constraints": (
            Constraint("policy_epoch", "equals", "1"),
        ),
    }
    values.update(overrides)
    return AuthoritySet(**values)


def test_subset_and_equality() -> None:
    maximum = authority(resource_ids=frozenset({"r1", "r2"}))
    assert is_authority_subset(authority(), maximum)
    assert is_authority_subset(maximum, maximum)


def test_empty_is_subset_of_everything() -> None:
    assert is_authority_subset(AuthoritySet.empty(), authority())


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_ids": frozenset({"t2"})},
        {"actions": frozenset({"write"})},
        {"purposes": frozenset({"marketing"})},
        {"relation_ids": frozenset({"secret"})},
    ],
)
def test_scope_escalation_fails(overrides: dict[str, object]) -> None:
    assert not is_authority_subset(authority(**overrides), authority())


def test_dropping_constraint_is_escalation() -> None:
    assert not is_authority_subset(
        authority(constraints=()),
        authority(),
    )


def test_compositional_union_escalation() -> None:
    step1 = authority(actions=frozenset({"read"}))
    step2 = authority(actions=frozenset({"write"}))
    closure = authority(actions=frozenset({"read"}))
    assert union_authorities((step1, step2)).actions == {"read", "write"}
    assert not compositional_non_amplification((step1, step2), closure)
    with pytest.raises(AuthorityInvariantError):
        require_authority_subset(union_authorities((step1, step2)), closure)


def test_proposal_never_authorizes_execution() -> None:
    proposal = SemanticProposal(
        "p1", ("r1",), ("read",), None, {}, ProposalSource.ROUTER
    )
    assert semantic_proposal_authorizes_execution(proposal) is False


def test_old_policy_witness_is_rejected() -> None:
    witness = AuthorizationWitness(
        decision_id="a" * 64,
        policy_hash="b" * 64,
        principal_attributes_digest="c" * 64,
        resource_action_digest="d" * 64,
        purpose_digest=None,
        environment_digest="e" * 64,
        policy_epoch=1,
        matched_policy_ids=("p1",),
    )
    with pytest.raises(AuthorityInvariantError, match="已过期"):
        require_fresh_witness(
            witness,
            current_policy_epoch=2,
            current_policy_hash="b" * 64,
        )
