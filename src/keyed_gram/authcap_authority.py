"""AuthoritySet 的包含关系与组合不放大检查。"""

from __future__ import annotations

from collections.abc import Iterable

from .authcap_types import (
    AuthoritySet,
    AuthorizationWitness,
    SemanticProposal,
)


class AuthorityInvariantError(ValueError):
    """authority 违反显式作用域不变量。"""


def is_authority_subset(
    candidate: AuthoritySet,
    maximum: AuthoritySet,
) -> bool:
    """判断候选 authority 是否不超过 policy grant。

    constraint 被解释为限制条件，因此候选必须保留 grant 的全部限制；增加限制
    会缩小权限，删除限制会扩大权限。
    """

    if candidate.is_empty:
        return True
    if maximum.is_empty:
        return False
    dimensions = (
        "subject_ids",
        "tenant_ids",
        "resource_ids",
        "actions",
        "relation_ids",
        "purposes",
    )
    if any(
        not getattr(candidate, name).issubset(getattr(maximum, name))
        for name in dimensions
    ):
        return False
    return set(maximum.constraints).issubset(candidate.constraints)


def require_authority_subset(
    candidate: AuthoritySet,
    maximum: AuthoritySet,
) -> None:
    if not is_authority_subset(candidate, maximum):
        raise AuthorityInvariantError("candidate authority 超过 policy grant")


def union_authorities(authorities: Iterable[AuthoritySet]) -> AuthoritySet:
    values = tuple(authorities)
    if not values:
        return AuthoritySet.empty()
    common_constraints = set(values[0].constraints)
    for item in values[1:]:
        common_constraints.intersection_update(item.constraints)
    return AuthoritySet(
        subject_ids=frozenset().union(*(item.subject_ids for item in values)),
        tenant_ids=frozenset().union(*(item.tenant_ids for item in values)),
        resource_ids=frozenset().union(*(item.resource_ids for item in values)),
        actions=frozenset().union(*(item.actions for item in values)),
        relation_ids=frozenset().union(*(item.relation_ids for item in values)),
        purposes=frozenset().union(*(item.purposes for item in values)),
        constraints=tuple(sorted(common_constraints)),
    )


def compositional_non_amplification(
    step_authorities: Iterable[AuthoritySet],
    allowed_closure: AuthoritySet,
) -> bool:
    return is_authority_subset(
        union_authorities(step_authorities),
        allowed_closure,
    )


def semantic_proposal_authorizes_execution(
    proposal: SemanticProposal,
) -> bool:
    """P1：proposal 永远不是 authorization ground truth。"""

    if not isinstance(proposal, SemanticProposal):
        raise TypeError("只接受 SemanticProposal")
    return False


def require_fresh_witness(
    witness: AuthorizationWitness,
    *,
    current_policy_epoch: int,
    current_policy_hash: str,
) -> None:
    """P5：旧 epoch 或旧 policy hash 的 witness 不能自动复用。"""

    if (
        witness.policy_epoch != current_policy_epoch
        or witness.policy_hash != current_policy_hash
    ):
        raise AuthorityInvariantError("authorization witness 已过期")
