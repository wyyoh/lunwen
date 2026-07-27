"""纯确定性的 AuthCap policy oracle 与 authorization witness。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .authcap_policy import PolicyRule, PolicySet, canonical_sha256
from .authcap_types import (
    AuthoritySet,
    AuthorizationWitness,
    Constraint,
    PolicyGrant,
    PolicyRequestContext,
)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("时间必须包含 UTC 时区")
    return parsed.astimezone(timezone.utc)


def _matches(rule: PolicyRule, context: PolicyRequestContext) -> bool:
    principal = context.principal
    resource = context.requested_resource
    purpose = context.purpose if context.purpose is not None else "purpose:none"
    now = _utc(context.environment.timestamp)
    return all(
        (
            context.policy_epoch >= 1,
            principal.subject_id in rule.subject_ids,
            principal.tenant_id in rule.tenant_ids,
            principal.attributes.get("department") in rule.departments,
            resource.tenant_id in rule.tenant_ids,
            resource.resource_id in rule.resource_ids,
            resource.resource_type in rule.resource_types,
            context.requested_action.value in rule.actions,
            resource.relation_id in rule.relation_ids,
            purpose in rule.purposes,
            all(principal.attributes.get(key) == value for key, value in rule.required_attributes),
            _utc(rule.valid_from) <= now < _utc(rule.valid_until),
            (not rule.delegation_required or bool(principal.delegation_chain)),
            len(principal.delegation_chain) <= rule.max_delegation_depth,
        )
    )


def _request_authority(
    context: PolicyRequestContext,
    rule: PolicyRule,
) -> AuthoritySet:
    purpose = context.purpose if context.purpose is not None else "purpose:none"
    authority = AuthoritySet(
        subject_ids=frozenset({context.principal.subject_id}),
        tenant_ids=frozenset({context.requested_resource.tenant_id}),
        resource_ids=frozenset({context.requested_resource.resource_id}),
        actions=frozenset({context.requested_action.value}),
        relation_ids=frozenset({context.requested_resource.relation_id}),
        purposes=frozenset({purpose}),
        constraints=tuple(
            sorted(
                set(rule.authority.constraints)
                | {
                    Constraint(
                        key="policy_epoch",
                        operator="equals",
                        value=str(context.policy_epoch),
                    )
                }
            )
        ),
    )
    dimensions = (
        "subject_ids",
        "tenant_ids",
        "resource_ids",
        "actions",
        "relation_ids",
        "purposes",
    )
    if any(
        not getattr(authority, name).issubset(getattr(rule.authority, name))
        for name in dimensions
    ):
        raise ValueError("policy authority 与 selector 不一致")
    return authority


def evaluate_policy(
    context: PolicyRequestContext,
    policy_set: PolicySet,
) -> PolicyGrant:
    """返回逐字节可复现的 allow/deny 和最大 authority。

    解析规则：epoch 不同立即 deny；其后按 priority、specificity 取最高层级；
    同层存在 deny 时 deny 优先。更高 priority 是 policy language 中唯一允许的
    override 机制。
    """

    policy_hash = policy_set.policy_hash
    if context.policy_epoch != policy_set.policy_epoch:
        matched: tuple[str, ...] = ()
        decision = "deny"
        reason = "policy_epoch_mismatch"
        maximum = AuthoritySet.empty()
    else:
        matches = [rule for rule in policy_set.policies if _matches(rule, context)]
        if not matches:
            matched = ()
            decision = "deny"
            reason = "default_deny"
            maximum = AuthoritySet.empty()
        else:
            highest_priority = max(rule.priority for rule in matches)
            priority_rules = [
                rule for rule in matches if rule.priority == highest_priority
            ]
            highest_specificity = max(rule.specificity for rule in priority_rules)
            winners = tuple(
                sorted(
                    (
                        rule
                        for rule in priority_rules
                        if rule.specificity == highest_specificity
                    ),
                    key=lambda rule: rule.policy_id,
                )
            )
            matched = tuple(rule.policy_id for rule in winners)
            denies = [rule for rule in winners if rule.effect == "deny"]
            if denies:
                decision = "deny"
                reason = "explicit_deny"
                maximum = AuthoritySet.empty()
            else:
                decision = "allow"
                reason = "explicit_allow"
                authorities = [_request_authority(context, rule) for rule in winners]
                maximum = authorities[0]
                if any(item != maximum for item in authorities[1:]):
                    raise ValueError("同层 allow 产生不一致的 request authority")
    decision_material: dict[str, Any] = {
        "context": context.canonical(),
        "policy_hash": policy_hash,
        "decision": decision,
        "maximum_authority": maximum.canonical(),
        "matched_policy_ids": list(matched),
        "decision_reason_code": reason,
    }
    return PolicyGrant(
        decision=decision,
        maximum_authority=maximum,
        matched_policy_ids=matched,
        policy_hash=policy_hash,
        decision_id=canonical_sha256(decision_material),
        decision_reason_code=reason,
    )


def build_authorization_witness(
    context: PolicyRequestContext,
    grant: PolicyGrant,
) -> AuthorizationWitness:
    return AuthorizationWitness(
        decision_id=grant.decision_id,
        policy_hash=grant.policy_hash,
        principal_attributes_digest=canonical_sha256(
            context.principal.canonical()
        ),
        resource_action_digest=canonical_sha256(
            {
                "resource": context.requested_resource.canonical(),
                "action": context.requested_action.value,
            }
        ),
        purpose_digest=(
            None
            if context.purpose is None
            else canonical_sha256({"purpose": context.purpose})
        ),
        environment_digest=canonical_sha256(context.environment.canonical()),
        policy_epoch=context.policy_epoch,
        matched_policy_ids=grant.matched_policy_ids,
    )
