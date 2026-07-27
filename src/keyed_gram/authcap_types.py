"""AuthCap 的类型边界。

这些类型只表达建议、可信请求和确定性策略结果；本模块不包含 capability
签发、memory 访问或工具执行能力。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}$")
_WILDCARDS = frozenset({"*", "**", "all", "any"})


class AuthCapTypeError(ValueError):
    """AuthCap 类型不满足显式、有限作用域要求。"""


class ProposalSource(str, Enum):
    LLM = "llm"
    ROUTER = "router"
    RULE = "rule"
    HUMAN = "human"


class Action(str, Enum):
    SEARCH = "search"
    RETRIEVE = "retrieve"
    SUMMARIZE = "summarize"
    QUOTE = "quote"
    EXPORT = "export"
    READ = "read"
    LIST = "list"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SEND = "send"
    EXECUTE = "execute"
    READ_MEMORY = "read_memory"
    APPEND_MEMORY = "append_memory"
    UPDATE_MEMORY = "update_memory"
    DELETE_MEMORY = "delete_memory"
    DERIVE_SUMMARY = "derive_summary"


def _safe_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise AuthCapTypeError(f"{label} 不是安全标识符")
    if value.casefold() in _WILDCARDS or "*" in value:
        raise AuthCapTypeError(f"{label} 禁止 wildcard")
    return value


def _finite_scope(values: frozenset[str], label: str) -> frozenset[str]:
    if not isinstance(values, frozenset):
        values = frozenset(values)
    return frozenset(_safe_identifier(item, label) for item in values)


@dataclass(frozen=True, order=True)
class Constraint:
    key: str
    operator: Literal["equals", "before", "max_int"]
    value: str

    def __post_init__(self) -> None:
        _safe_identifier(self.key, "constraint.key")
        if not isinstance(self.value, str) or not self.value:
            raise AuthCapTypeError("constraint.value 不能为空")

    def canonical(self) -> dict[str, str]:
        return {"key": self.key, "operator": self.operator, "value": self.value}


@dataclass(frozen=True)
class AuthoritySet:
    subject_ids: frozenset[str] = field(default_factory=frozenset)
    tenant_ids: frozenset[str] = field(default_factory=frozenset)
    resource_ids: frozenset[str] = field(default_factory=frozenset)
    actions: frozenset[str] = field(default_factory=frozenset)
    relation_ids: frozenset[str] = field(default_factory=frozenset)
    purposes: frozenset[str] = field(default_factory=frozenset)
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "subject_ids",
            "tenant_ids",
            "resource_ids",
            "actions",
            "relation_ids",
            "purposes",
        ):
            object.__setattr__(self, name, _finite_scope(getattr(self, name), name))
        ordered = tuple(sorted(set(self.constraints)))
        if len(ordered) != len(self.constraints):
            raise AuthCapTypeError("AuthoritySet constraints 不得重复")
        object.__setattr__(self, "constraints", ordered)

    @classmethod
    def empty(cls) -> AuthoritySet:
        return cls()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.subject_ids,
                self.tenant_ids,
                self.resource_ids,
                self.actions,
                self.relation_ids,
                self.purposes,
                self.constraints,
            )
        )

    def canonical(self) -> dict[str, object]:
        return {
            "subject_ids": sorted(self.subject_ids),
            "tenant_ids": sorted(self.tenant_ids),
            "resource_ids": sorted(self.resource_ids),
            "actions": sorted(self.actions),
            "relation_ids": sorted(self.relation_ids),
            "purposes": sorted(self.purposes),
            "constraints": [item.canonical() for item in self.constraints],
        }


@dataclass(frozen=True)
class SemanticProposal:
    proposal_id: str
    proposed_resource_ids: tuple[str, ...]
    proposed_actions: tuple[str, ...]
    proposed_purpose: str | None
    proposed_constraints: Mapping[str, str]
    source_type: ProposalSource

    def __post_init__(self) -> None:
        _safe_identifier(self.proposal_id, "proposal_id")
        resources = tuple(
            sorted(
                {
                    _safe_identifier(item, "proposed_resource_ids")
                    for item in self.proposed_resource_ids
                }
            )
        )
        actions = tuple(
            sorted(
                {
                    _safe_identifier(item, "proposed_actions")
                    for item in self.proposed_actions
                }
            )
        )
        constraints = {
            _safe_identifier(str(key), "proposed_constraints.key"): str(value)
            for key, value in self.proposed_constraints.items()
        }
        object.__setattr__(self, "proposed_resource_ids", resources)
        object.__setattr__(self, "proposed_actions", actions)
        object.__setattr__(
            self,
            "proposed_constraints",
            MappingProxyType(dict(sorted(constraints.items()))),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposed_resource_ids": list(self.proposed_resource_ids),
            "proposed_actions": list(self.proposed_actions),
            "proposed_purpose": self.proposed_purpose,
            "proposed_constraints": dict(self.proposed_constraints),
            "source_type": self.source_type.value,
        }


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject_id: str
    tenant_id: str
    attributes: Mapping[str, str]
    delegation_chain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _safe_identifier(self.subject_id, "subject_id")
        _safe_identifier(self.tenant_id, "tenant_id")
        attributes = {
            _safe_identifier(str(key), "principal.attributes.key"): str(value)
            for key, value in self.attributes.items()
        }
        object.__setattr__(
            self, "attributes", MappingProxyType(dict(sorted(attributes.items())))
        )
        object.__setattr__(
            self,
            "delegation_chain",
            tuple(
                _safe_identifier(item, "delegation_chain")
                for item in self.delegation_chain
            ),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "tenant_id": self.tenant_id,
            "attributes": dict(self.attributes),
            "delegation_chain": list(self.delegation_chain),
        }


@dataclass(frozen=True)
class ResourceDescriptor:
    tenant_id: str
    resource_id: str
    resource_type: str
    relation_id: str

    def __post_init__(self) -> None:
        for name in ("tenant_id", "resource_id", "resource_type", "relation_id"):
            _safe_identifier(getattr(self, name), name)

    def canonical(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "relation_id": self.relation_id,
        }


@dataclass(frozen=True)
class EnvironmentContext:
    timestamp: str
    attributes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, str) or not self.timestamp.endswith("Z"):
            raise AuthCapTypeError("timestamp 必须是 UTC Z 时间")
        attributes = {
            _safe_identifier(str(key), "environment.attributes.key"): str(value)
            for key, value in self.attributes.items()
        }
        object.__setattr__(
            self, "attributes", MappingProxyType(dict(sorted(attributes.items())))
        )

    def canonical(self) -> dict[str, object]:
        return {"timestamp": self.timestamp, "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class PolicyRequestContext:
    principal: AuthenticatedPrincipal
    requested_resource: ResourceDescriptor
    requested_action: Action
    purpose: str | None
    environment: EnvironmentContext
    policy_epoch: int

    def __post_init__(self) -> None:
        if self.policy_epoch < 1:
            raise AuthCapTypeError("policy_epoch 必须为正整数")
        # 跨租户请求可进入 oracle，但不会被隐式修正或授权。
        if self.purpose is not None:
            _safe_identifier(self.purpose, "purpose")

    def canonical(self) -> dict[str, object]:
        return {
            "principal": self.principal.canonical(),
            "requested_resource": self.requested_resource.canonical(),
            "requested_action": self.requested_action.value,
            "purpose": self.purpose,
            "environment": self.environment.canonical(),
            "policy_epoch": self.policy_epoch,
        }


@dataclass(frozen=True)
class PolicyGrant:
    decision: Literal["allow", "deny"]
    maximum_authority: AuthoritySet
    matched_policy_ids: tuple[str, ...]
    policy_hash: str
    decision_id: str
    decision_reason_code: str

    def __post_init__(self) -> None:
        if self.decision == "deny" and not self.maximum_authority.is_empty:
            raise AuthCapTypeError("deny grant 的 maximum_authority 必须为空")
        if len(self.policy_hash) != 64 or len(self.decision_id) != 64:
            raise AuthCapTypeError("policy_hash/decision_id 必须是 SHA-256")

    def canonical(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "maximum_authority": self.maximum_authority.canonical(),
            "matched_policy_ids": list(self.matched_policy_ids),
            "policy_hash": self.policy_hash,
            "decision_id": self.decision_id,
            "decision_reason_code": self.decision_reason_code,
        }


@dataclass(frozen=True)
class AuthorizationWitness:
    decision_id: str
    policy_hash: str
    principal_attributes_digest: str
    resource_action_digest: str
    purpose_digest: str | None
    environment_digest: str
    policy_epoch: int
    matched_policy_ids: tuple[str, ...]

    def canonical(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "policy_hash": self.policy_hash,
            "principal_attributes_digest": self.principal_attributes_digest,
            "resource_action_digest": self.resource_action_digest,
            "purpose_digest": self.purpose_digest,
            "environment_digest": self.environment_digest,
            "policy_epoch": self.policy_epoch,
            "matched_policy_ids": list(self.matched_policy_ids),
        }
