"""来源绑定、线性且带预算的 authority resource algebra。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .labels import DataLabel, Integrity

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_WILDCARDS = frozenset({"*", "**", "all", "any"})


class AuthorityFlowError(ValueError):
    """authority 操作违反来源、作用域、线性或预算约束。"""


def _identifier(value: str, label: str) -> str:
    if (
        isinstance(value, str)
        and (value.casefold() in _WILDCARDS or "*" in value)
    ):
        raise AuthorityFlowError(f"{label} 禁止 wildcard")
    if not isinstance(value, str) or not _SAFE.fullmatch(value):
        raise AuthorityFlowError(f"{label} 不是安全有限标识符")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, order=True)
class InfluenceGuard:
    """哪些数据影响有资格控制一次 authority 消费。"""

    minimum_integrity: Integrity
    allowed_source_kinds: tuple[str, ...]
    attacker_controlled_allowed: bool = False

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                {
                    _identifier(item, "allowed_source_kind")
                    for item in self.allowed_source_kinds
                }
            )
        )
        if not ordered:
            raise AuthorityFlowError("InfluenceGuard 至少允许一个来源类别")
        object.__setattr__(self, "allowed_source_kinds", ordered)

    def permits(self, label: DataLabel) -> bool:
        if label.integrity < self.minimum_integrity:
            return False
        if any(
            item.source_kind not in self.allowed_source_kinds
            for item in label.influences
        ):
            return False
        return self.attacker_controlled_allowed or not any(
            item.attacker_controlled for item in label.influences
        )

    def canonical(self) -> dict[str, object]:
        return {
            "allowed_source_kinds": list(self.allowed_source_kinds),
            "attacker_controlled_allowed": self.attacker_controlled_allowed,
            "minimum_integrity": self.minimum_integrity.name.lower(),
        }


@dataclass(frozen=True, order=True)
class AuthorityAtom:
    issuer: str
    subject: str
    tenant: str
    resource: str
    action: str
    purpose: str
    epoch: int
    influence_guard: InfluenceGuard

    def __post_init__(self) -> None:
        for field_name in (
            "issuer",
            "subject",
            "tenant",
            "resource",
            "action",
            "purpose",
        ):
            _identifier(getattr(self, field_name), field_name)
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise AuthorityFlowError("epoch 必须是非负整数")

    def canonical(self) -> dict[str, object]:
        return {
            "action": self.action,
            "epoch": self.epoch,
            "issuer": self.issuer,
            "influence_guard": self.influence_guard.canonical(),
            "purpose": self.purpose,
            "resource": self.resource,
            "subject": self.subject,
            "tenant": self.tenant,
        }

    def same_effect_scope(self, effect: AuthorityEffect) -> bool:
        return (
            self.subject == effect.subject
            and self.tenant == effect.tenant
            and self.resource == effect.resource
            and self.action == effect.action
            and self.purpose == effect.purpose
            and self.epoch == effect.epoch
        )


@dataclass(frozen=True, order=True)
class AuthorityBudget:
    """预算维度均为自然数；候选逐维不大于父预算才是衰减。"""

    uses: int
    amount_microunits: int = 0
    data_bytes: int = 0
    resource_count: int = 1
    delegation_depth: int = 0
    ttl_steps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "uses",
            "amount_microunits",
            "data_bytes",
            "resource_count",
            "delegation_depth",
            "ttl_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise AuthorityFlowError(f"budget.{name} 必须是非负整数")

    @property
    def exhausted(self) -> bool:
        return self.uses == 0 or self.resource_count == 0 or self.ttl_steps == 0

    def canonical(self) -> dict[str, int]:
        return {
            "amount_microunits": self.amount_microunits,
            "data_bytes": self.data_bytes,
            "delegation_depth": self.delegation_depth,
            "resource_count": self.resource_count,
            "ttl_steps": self.ttl_steps,
            "uses": self.uses,
        }

    def attenuates(self, maximum: AuthorityBudget) -> bool:
        return all(
            getattr(self, name) <= getattr(maximum, name)
            for name in (
                "uses",
                "amount_microunits",
                "data_bytes",
                "resource_count",
                "delegation_depth",
                "ttl_steps",
            )
        )

    def consume(self, cost: AuthorityBudget) -> AuthorityBudget:
        if not cost.attenuates(self):
            raise AuthorityFlowError("effect cost 超过 authority budget")
        if cost.uses <= 0:
            raise AuthorityFlowError("每个 effect 至少消费一次 use")
        return AuthorityBudget(
            uses=self.uses - cost.uses,
            amount_microunits=self.amount_microunits
            - cost.amount_microunits,
            data_bytes=self.data_bytes - cost.data_bytes,
            resource_count=self.resource_count - cost.resource_count,
            delegation_depth=self.delegation_depth,
            ttl_steps=self.ttl_steps,
        )

    @classmethod
    def aggregate(cls, values: Iterable[AuthorityBudget]) -> AuthorityBudget:
        items = tuple(values)
        if not items:
            return cls(uses=0, resource_count=0, ttl_steps=0)
        return cls(
            uses=sum(item.uses for item in items),
            amount_microunits=sum(item.amount_microunits for item in items),
            data_bytes=sum(item.data_bytes for item in items),
            resource_count=sum(item.resource_count for item in items),
            delegation_depth=max(item.delegation_depth for item in items),
            ttl_steps=max(item.ttl_steps for item in items),
        )


@dataclass(frozen=True, order=True)
class AuthorityOrigin:
    grant_id: str
    issuer: str
    authenticated: bool
    origin_kind: str = "policy_grant"

    def __post_init__(self) -> None:
        _identifier(self.grant_id, "grant_id")
        _identifier(self.issuer, "origin.issuer")
        _identifier(self.origin_kind, "origin_kind")

    def canonical(self) -> dict[str, object]:
        return {
            "authenticated": self.authenticated,
            "grant_id": self.grant_id,
            "issuer": self.issuer,
            "origin_kind": self.origin_kind,
        }


@dataclass(frozen=True, order=True)
class AuthorityResource:
    resource_id: str
    atom: AuthorityAtom
    origin: AuthorityOrigin
    budget: AuthorityBudget
    lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "authority_resource_id")
        if not self.lineage or self.lineage[-1] != self.resource_id:
            raise AuthorityFlowError("lineage 必须以当前 resource_id 结尾")
        for item in self.lineage:
            _identifier(item, "lineage")

    def canonical(self) -> dict[str, object]:
        return {
            "atom": self.atom.canonical(),
            "budget": self.budget.canonical(),
            "lineage": list(self.lineage),
            "origin": self.origin.canonical(),
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True, order=True)
class AuthorityEffect:
    effect_id: str
    subject: str
    tenant: str
    resource: str
    action: str
    purpose: str
    epoch: int
    cost: AuthorityBudget
    authority_resource_id: str
    origin_grant_id: str

    def __post_init__(self) -> None:
        for name in (
            "effect_id",
            "subject",
            "tenant",
            "resource",
            "action",
            "purpose",
            "authority_resource_id",
            "origin_grant_id",
        ):
            _identifier(getattr(self, name), name)
        if self.cost.uses <= 0:
            raise AuthorityFlowError("effect 必须消费 authority")

    @property
    def signature(self) -> tuple[str, str, str, str, str]:
        return (
            self.subject,
            self.tenant,
            self.resource,
            self.action,
            self.purpose,
        )

    def canonical(self) -> dict[str, object]:
        return {
            "action": self.action,
            "authority_resource_id": self.authority_resource_id,
            "cost": self.cost.canonical(),
            "effect_id": self.effect_id,
            "epoch": self.epoch,
            "origin_grant_id": self.origin_grant_id,
            "purpose": self.purpose,
            "resource": self.resource,
            "subject": self.subject,
            "tenant": self.tenant,
        }


@dataclass(frozen=True)
class LinearAuthorityContext:
    resources: tuple[AuthorityResource, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.resources, key=lambda item: item.resource_id))
        ids = [item.resource_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise AuthorityFlowError("线性上下文不能重复持有同一 authority resource")
        if any(not item.origin.authenticated for item in ordered):
            raise AuthorityFlowError("未认证来源不能进入线性 authority context")
        object.__setattr__(self, "resources", ordered)

    @property
    def digest(self) -> str:
        return _digest(self.canonical())

    def canonical(self) -> dict[str, object]:
        return {"resources": [item.canonical() for item in self.resources]}

    def get(self, resource_id: str) -> AuthorityResource:
        for item in self.resources:
            if item.resource_id == resource_id:
                return item
        raise AuthorityFlowError("authority resource 不存在或已消费")

    def replace(
        self,
        resource_id: str,
        replacements: Iterable[AuthorityResource],
    ) -> LinearAuthorityContext:
        self.get(resource_id)
        return LinearAuthorityContext(
            tuple(item for item in self.resources if item.resource_id != resource_id)
            + tuple(replacements)
        )


def issue_authenticated(
    atom: AuthorityAtom,
    budget: AuthorityBudget,
    origin: AuthorityOrigin,
) -> LinearAuthorityContext:
    if not origin.authenticated:
        raise AuthorityFlowError("只有 authenticated grant 可以创生 authority")
    resource_id = "auth-" + _digest(
        {
            "atom": atom.canonical(),
            "budget": budget.canonical(),
            "origin": origin.canonical(),
        }
    )[:24]
    return LinearAuthorityContext(
        (
            AuthorityResource(
                resource_id=resource_id,
                atom=atom,
                origin=origin,
                budget=budget,
                lineage=(resource_id,),
            ),
        )
    )


def _child_id(
    parent: AuthorityResource,
    operation: str,
    index: int,
    budget: AuthorityBudget,
    subject: str,
) -> str:
    return "auth-" + _digest(
        {
            "budget": budget.canonical(),
            "index": index,
            "operation": operation,
            "parent": parent.resource_id,
            "subject": subject,
        }
    )[:24]


def split(
    context: LinearAuthorityContext,
    resource_id: str,
    allocations: tuple[AuthorityBudget, ...],
) -> tuple[LinearAuthorityContext, tuple[AuthorityResource, ...]]:
    """消费父 resource 并创建预算总和不超过父资源的互异子资源。"""

    parent = context.get(resource_id)
    if len(allocations) < 2:
        raise AuthorityFlowError("split 至少需要两个子预算")
    aggregate = AuthorityBudget.aggregate(allocations)
    if not aggregate.attenuates(parent.budget):
        raise AuthorityFlowError("split 子预算总和超过父预算")
    children = tuple(
        AuthorityResource(
            resource_id=(
                child_id := _child_id(
                    parent,
                    "split",
                    index,
                    allocation,
                    parent.atom.subject,
                )
            ),
            atom=parent.atom,
            origin=parent.origin,
            budget=allocation,
            lineage=parent.lineage + (child_id,),
        )
        for index, allocation in enumerate(allocations)
    )
    return context.replace(resource_id, children), children


def delegate(
    context: LinearAuthorityContext,
    resource_id: str,
    child_subject: str,
    child_budget: AuthorityBudget,
) -> tuple[LinearAuthorityContext, AuthorityResource]:
    """线性地转移并衰减 authority；父 resource 不再留在原上下文。"""

    _identifier(child_subject, "child_subject")
    parent = context.get(resource_id)
    if parent.budget.delegation_depth <= 0:
        raise AuthorityFlowError("delegation depth 已耗尽")
    if not child_budget.attenuates(parent.budget):
        raise AuthorityFlowError("delegation 放大预算")
    if child_budget.delegation_depth > parent.budget.delegation_depth - 1:
        raise AuthorityFlowError("delegation depth 未衰减")
    atom = AuthorityAtom(
        issuer=parent.atom.issuer,
        subject=child_subject,
        tenant=parent.atom.tenant,
        resource=parent.atom.resource,
        action=parent.atom.action,
        purpose=parent.atom.purpose,
        epoch=parent.atom.epoch,
        influence_guard=parent.atom.influence_guard,
    )
    child_id = _child_id(parent, "delegate", 0, child_budget, child_subject)
    child = AuthorityResource(
        resource_id=child_id,
        atom=atom,
        origin=parent.origin,
        budget=child_budget,
        lineage=parent.lineage + (child_id,),
    )
    return context.replace(resource_id, (child,)), child


def consume(
    context: LinearAuthorityContext,
    resource_id: str,
    *,
    effect_id: str,
    subject: str,
    tenant: str,
    resource: str,
    action: str,
    purpose: str,
    epoch: int,
    cost: AuthorityBudget,
    control_label: DataLabel,
) -> tuple[LinearAuthorityContext, AuthorityEffect]:
    authority = context.get(resource_id)
    effect = AuthorityEffect(
        effect_id=effect_id,
        subject=subject,
        tenant=tenant,
        resource=resource,
        action=action,
        purpose=purpose,
        epoch=epoch,
        cost=cost,
        authority_resource_id=resource_id,
        origin_grant_id=authority.origin.grant_id,
    )
    if not authority.atom.same_effect_scope(effect):
        raise AuthorityFlowError("effect 与 authority atom 作用域不匹配")
    if not authority.atom.influence_guard.permits(control_label):
        raise AuthorityFlowError("数据 influence 无资格控制该 authority effect")
    residual = authority.budget.consume(cost)
    replacements: tuple[AuthorityResource, ...]
    if residual.exhausted:
        replacements = ()
    else:
        replacements = (
            AuthorityResource(
                resource_id=authority.resource_id,
                atom=authority.atom,
                origin=authority.origin,
                budget=residual,
                lineage=authority.lineage,
            ),
        )
    return context.replace(resource_id, replacements), effect


@dataclass(frozen=True)
class BranchState:
    branch_id: str
    parent_context_digest: str
    authority: LinearAuthorityContext
    effects: tuple[AuthorityEffect, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.branch_id, "branch_id")
        if len(self.parent_context_digest) != 64:
            raise AuthorityFlowError("parent_context_digest 非法")
        object.__setattr__(self, "effects", tuple(sorted(self.effects)))


@dataclass(frozen=True)
class ForbiddenEffectCombination:
    combination_id: str
    required_signatures: frozenset[tuple[str, str, str, str, str]]

    def __post_init__(self) -> None:
        _identifier(self.combination_id, "combination_id")
        if len(self.required_signatures) < 2:
            raise AuthorityFlowError("combined deny 至少包含两个 effect")


def _budget_by_origin(
    resources: Iterable[AuthorityResource],
    effects: Iterable[AuthorityEffect],
) -> dict[str, AuthorityBudget]:
    grouped: dict[str, list[AuthorityBudget]] = {}
    for item in resources:
        grouped.setdefault(item.origin.grant_id, []).append(item.budget)
    for item in effects:
        grouped.setdefault(item.origin_grant_id, []).append(item.cost)
    return {
        origin: AuthorityBudget.aggregate(budgets)
        for origin, budgets in grouped.items()
    }


def merge_branches(
    parent: LinearAuthorityContext,
    branches: tuple[BranchState, ...],
    *,
    combined_denies: tuple[ForbiddenEffectCombination, ...] = (),
) -> tuple[LinearAuthorityContext, tuple[AuthorityEffect, ...]]:
    """合并数据/effect trace，但拒绝 authority 复制、超预算和组合 deny。

    合法并行必须先显式 split，使不同分支获得互异的线性子资源。
    """

    if len(branches) < 2:
        raise AuthorityFlowError("merge 至少需要两个分支")
    if any(item.parent_context_digest != parent.digest for item in branches):
        raise AuthorityFlowError("分支不属于同一父上下文")
    resources = tuple(
        resource for branch in branches for resource in branch.authority.resources
    )
    resource_ids = [item.resource_id for item in resources]
    if len(resource_ids) != len(set(resource_ids)):
        raise AuthorityFlowError("跨分支 authority resource 被重复持有")
    effects = tuple(effect for branch in branches for effect in branch.effects)
    parent_budget = _budget_by_origin(parent.resources, ())
    observed_budget = _budget_by_origin(resources, effects)
    for origin, budget in observed_budget.items():
        maximum = parent_budget.get(origin)
        if maximum is None or not budget.attenuates(maximum):
            raise AuthorityFlowError("merge 后 authority/effect 超过父资源预算")
    signatures = frozenset(item.signature for item in effects)
    for deny in combined_denies:
        if deny.required_signatures.issubset(signatures):
            raise AuthorityFlowError(
                f"merge 命中 combined deny：{deny.combination_id}"
            )
    return LinearAuthorityContext(resources), tuple(sorted(effects))
