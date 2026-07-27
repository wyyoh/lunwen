"""F2 使用的原子化 authority 代数。

F1 的 ``AuthoritySet`` 是 benchmark 标签格式。F2 不修改该冻结类型，而是将
每个可执行权限表示为一个完整 atom，避免跨维度集合产生隐式笛卡尔积。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .authcap_policy import canonical_json_bytes
from .authcap_types import AuthoritySet

_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}$")
_WILDCARDS = frozenset({"*", "**", "all", "any"})
ConstraintOperator = Literal["equals", "not_after", "max_int", "set_subset"]


class AtomizedAuthorityError(ValueError):
    """原子化 authority 不满足显式有限作用域或偏序要求。"""


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise AtomizedAuthorityError(f"{label} 不是安全标识符")
    if "*" in value or value.casefold() in _WILDCARDS:
        raise AtomizedAuthorityError(f"{label} 禁止 wildcard")
    return value


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AtomizedAuthorityError("not_after 必须是 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise AtomizedAuthorityError("not_after 必须包含时区")
    return parsed.astimezone(timezone.utc)


def _set_value(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AtomizedAuthorityError("set_subset 必须是 JSON 字符串数组") from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(item, str) and item for item in parsed)
    ):
        raise AtomizedAuthorityError("set_subset 必须是非空字符串数组")
    return tuple(sorted({_safe_id(item, "constraint set item") for item in parsed}))


@dataclass(frozen=True, order=True)
class CanonicalConstraint:
    """具有明确“更严格”偏序的单个约束。"""

    key: str
    operator: ConstraintOperator
    value: str

    def __post_init__(self) -> None:
        _safe_id(self.key, "constraint.key")
        if self.operator == "max_int":
            try:
                number = int(self.value)
            except (TypeError, ValueError) as exc:
                raise AtomizedAuthorityError("max_int 必须是非负整数") from exc
            if number < 0 or str(number) != str(self.value):
                raise AtomizedAuthorityError("max_int 必须使用规范十进制")
            object.__setattr__(self, "value", str(number))
        elif self.operator == "not_after":
            parsed = _utc(self.value)
            canonical = parsed.isoformat().replace("+00:00", "Z")
            object.__setattr__(self, "value", canonical)
        elif self.operator == "set_subset":
            canonical = json.dumps(
                list(_set_value(self.value)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            object.__setattr__(self, "value", canonical)
        elif self.operator == "equals":
            if not isinstance(self.value, str) or not self.value:
                raise AtomizedAuthorityError("equals 约束值不能为空")
        else:
            raise AtomizedAuthorityError("未知 constraint operator")

    def canonical(self) -> dict[str, str]:
        return {"key": self.key, "operator": self.operator, "value": self.value}


def constraint_is_at_least_as_restrictive(
    candidate: CanonicalConstraint,
    maximum: CanonicalConstraint,
) -> bool:
    """判断 candidate 是否不比 maximum 更宽松。"""

    if candidate.key != maximum.key or candidate.operator != maximum.operator:
        return False
    if candidate.operator == "equals":
        return candidate.value == maximum.value
    if candidate.operator == "max_int":
        return int(candidate.value) <= int(maximum.value)
    if candidate.operator == "not_after":
        return _utc(candidate.value) <= _utc(maximum.value)
    return set(_set_value(candidate.value)).issubset(_set_value(maximum.value))


def _constraint_join(
    left: CanonicalConstraint,
    right: CanonicalConstraint,
) -> CanonicalConstraint | None:
    """返回两个限制的交（更严格者）；不兼容时返回 ``None``。"""

    if left.key != right.key or left.operator != right.operator:
        return None
    if left.operator == "equals":
        return left if left.value == right.value else None
    if left.operator == "max_int":
        return min((left, right), key=lambda item: int(item.value))
    if left.operator == "not_after":
        return min((left, right), key=lambda item: _utc(item.value))
    values = sorted(set(_set_value(left.value)) & set(_set_value(right.value)))
    if not values:
        return None
    return CanonicalConstraint(
        left.key,
        "set_subset",
        json.dumps(values, ensure_ascii=False, separators=(",", ":")),
    )


@dataclass(frozen=True, order=True)
class AuthorityAtom:
    """一个无隐式乘积的完整可执行权限组合。"""

    subject_id: str
    tenant_id: str
    resource_id: str
    action: str
    relation_id: str | None
    purpose: str | None
    constraints: tuple[CanonicalConstraint, ...] = ()

    def __post_init__(self) -> None:
        for name in ("subject_id", "tenant_id", "resource_id", "action"):
            _safe_id(getattr(self, name), name)
        for name in ("relation_id", "purpose"):
            value = getattr(self, name)
            if value is not None:
                _safe_id(value, name)
        ordered = tuple(sorted(self.constraints))
        if len(ordered) != len(set(ordered)):
            raise AtomizedAuthorityError("duplicate constraint")
        keys = [item.key for item in ordered]
        if len(keys) != len(set(keys)):
            raise AtomizedAuthorityError("同一 atom 的 constraint key 不得重复")
        object.__setattr__(self, "constraints", ordered)

    @property
    def scope_key(self) -> tuple[str, str, str, str, str | None, str | None]:
        return (
            self.subject_id,
            self.tenant_id,
            self.resource_id,
            self.action,
            self.relation_id,
            self.purpose,
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "tenant_id": self.tenant_id,
            "resource_id": self.resource_id,
            "action": self.action,
            "relation_id": self.relation_id,
            "purpose": self.purpose,
            "constraints": [item.canonical() for item in self.constraints],
        }


def atom_is_subset(candidate: AuthorityAtom, maximum: AuthorityAtom) -> bool:
    if candidate.scope_key != maximum.scope_key:
        return False
    candidate_constraints = {item.key: item for item in candidate.constraints}
    return all(
        key in candidate_constraints
        and constraint_is_at_least_as_restrictive(candidate_constraints[key], limit)
        for key, limit in (
            (item.key, item)
            for item in maximum.constraints
        )
    )


def _intersect_atoms(
    left: AuthorityAtom,
    right: AuthorityAtom,
) -> AuthorityAtom | None:
    if left.scope_key != right.scope_key:
        return None
    left_constraints = {item.key: item for item in left.constraints}
    right_constraints = {item.key: item for item in right.constraints}
    constraints: list[CanonicalConstraint] = []
    for key in sorted(set(left_constraints) | set(right_constraints)):
        if key not in left_constraints:
            constraints.append(right_constraints[key])
        elif key not in right_constraints:
            constraints.append(left_constraints[key])
        else:
            joined = _constraint_join(left_constraints[key], right_constraints[key])
            if joined is None:
                return None
            constraints.append(joined)
    return AuthorityAtom(*left.scope_key, constraints=tuple(constraints))


@dataclass(frozen=True)
class AtomizedAuthority:
    """有限 AuthorityAtom 集合及其确定性集合代数。"""

    atoms: frozenset[AuthorityAtom]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", frozenset(self.atoms))

    @classmethod
    def empty(cls) -> AtomizedAuthority:
        return cls(frozenset())

    @property
    def is_empty(self) -> bool:
        return not self.atoms

    def canonical(self) -> dict[str, Any]:
        return {
            "atoms": [
                item.canonical()
                for item in sorted(self.atoms)
            ]
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical())

    @property
    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def is_subset_of(self, maximum: AtomizedAuthority) -> bool:
        return all(
            any(atom_is_subset(candidate, upper) for upper in maximum.atoms)
            for candidate in self.atoms
        )

    def is_strict_subset_of(self, maximum: AtomizedAuthority) -> bool:
        return self.is_subset_of(maximum) and self != maximum

    def intersection(self, other: AtomizedAuthority) -> AtomizedAuthority:
        values = {
            value
            for left in self.atoms
            for right in other.atoms
            if (value := _intersect_atoms(left, right)) is not None
        }
        return AtomizedAuthority(frozenset(values))

    def union(self, other: AtomizedAuthority) -> AtomizedAuthority:
        return AtomizedAuthority(self.atoms | other.atoms)

    def difference(self, other: AtomizedAuthority) -> AtomizedAuthority:
        """返回未被 other 语义覆盖的完整 atom。

        本操作不把单个带约束 atom 拆成补集，因而不会凭空创建新权限。
        """

        return AtomizedAuthority(
            frozenset(
                atom
                for atom in self.atoms
                if not any(atom_is_subset(atom, upper) for upper in other.atoms)
            )
        )


def authority_set_to_atomized(authority: AuthoritySet) -> AtomizedAuthority:
    """将 F1 grant 转为单 atom；拒绝可能产生笛卡尔歧义的集合。"""

    if authority.is_empty:
        return AtomizedAuthority.empty()
    dimensions = (
        authority.subject_ids,
        authority.tenant_ids,
        authority.resource_ids,
        authority.actions,
        authority.relation_ids,
        authority.purposes,
    )
    if any(len(values) != 1 for values in dimensions):
        raise AtomizedAuthorityError("F1 AuthoritySet 不是无歧义单 atom")
    operators = {"equals": "equals", "before": "not_after", "max_int": "max_int"}
    constraints = tuple(
        CanonicalConstraint(item.key, operators[item.operator], item.value)
        for item in authority.constraints
    )
    values = [next(iter(items)) for items in dimensions]
    return AtomizedAuthority(
        frozenset(
            {
                AuthorityAtom(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    constraints,
                )
            }
        )
    )


@dataclass(frozen=True)
class InvariantResult:
    passed: bool
    reason_code: str
    proposal_subset: bool
    request_subset: bool
    grant_subset: bool


def verify_authority_non_amplification(
    proposal_authority: AtomizedAuthority,
    trusted_request_authority: AtomizedAuthority,
    policy_grant_authority: AtomizedAuthority,
    executable_authority: AtomizedAuthority,
) -> InvariantResult:
    """验证 ``A_exec ⊆ A_proposal ∩ A_request ∩ A_grant``。"""

    proposal_ok = executable_authority.is_subset_of(proposal_authority)
    request_ok = executable_authority.is_subset_of(trusted_request_authority)
    grant_ok = executable_authority.is_subset_of(policy_grant_authority)
    passed = proposal_ok and request_ok and grant_ok
    return InvariantResult(
        passed=passed,
        reason_code="non_amplification_passed" if passed else "authority_amplification",
        proposal_subset=proposal_ok,
        request_subset=request_ok,
        grant_subset=grant_ok,
    )
