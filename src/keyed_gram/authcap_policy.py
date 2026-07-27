"""严格的结构化 AuthCap policy language。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .authcap_types import AuthoritySet, Constraint


class PolicySchemaError(ValueError):
    """Policy JSON/YAML 不满足严格 schema。"""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicySchemaError(f"duplicate JSON key：{key}")
        result[key] = value
    return result


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise PolicySchemaError(f"duplicate/invalid YAML key：{key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicySchemaError(f"NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def strict_load_document(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.casefold() == ".json":
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PolicySchemaError(f"非法 JSON 常量：{item}")
            ),
        )
    elif target.suffix.casefold() in {".yaml", ".yml"}:
        value = yaml.load(text, Loader=_StrictLoader)
    else:
        raise PolicySchemaError("policy 只允许 .json/.yaml/.yml")
    if not isinstance(value, dict):
        raise PolicySchemaError("policy document 顶层必须是 object")
    _finite(value)
    return value


def _exact_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown or missing:
        raise PolicySchemaError(
            f"{label} schema 不匹配；unknown={sorted(unknown)} missing={sorted(missing)}"
        )


def _strings(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (
        not allow_empty and not value
    ) or not all(isinstance(item, str) and item for item in value):
        raise PolicySchemaError(f"{label} 必须是非空字符串数组")
    if len(value) != len(set(value)):
        raise PolicySchemaError(f"{label} 不得重复")
    if any("*" in item or item.casefold() in {"all", "any"} for item in value):
        raise PolicySchemaError(f"{label} 禁止 wildcard")
    return tuple(sorted(value))


def _authority(value: Any) -> AuthoritySet:
    if not isinstance(value, Mapping):
        raise PolicySchemaError("authority 必须是 object")
    keys = {
        "subject_ids",
        "tenant_ids",
        "resource_ids",
        "actions",
        "relation_ids",
        "purposes",
        "constraints",
    }
    _exact_keys(value, keys, "authority")
    raw_constraints = value["constraints"]
    if not isinstance(raw_constraints, list):
        raise PolicySchemaError("authority.constraints 必须是数组")
    constraints = []
    for item in raw_constraints:
        if not isinstance(item, Mapping):
            raise PolicySchemaError("constraint 必须是 object")
        _exact_keys(item, {"key", "operator", "value"}, "constraint")
        constraints.append(
            Constraint(
                key=str(item["key"]),
                operator=str(item["operator"]),
                value=str(item["value"]),
            )
        )
    return AuthoritySet(
        subject_ids=frozenset(
            _strings(value["subject_ids"], "subject_ids", allow_empty=True)
        ),
        tenant_ids=frozenset(
            _strings(value["tenant_ids"], "tenant_ids", allow_empty=True)
        ),
        resource_ids=frozenset(
            _strings(value["resource_ids"], "resource_ids", allow_empty=True)
        ),
        actions=frozenset(_strings(value["actions"], "actions", allow_empty=True)),
        relation_ids=frozenset(
            _strings(value["relation_ids"], "relation_ids", allow_empty=True)
        ),
        purposes=frozenset(
            _strings(value["purposes"], "purposes", allow_empty=True)
        ),
        constraints=tuple(constraints),
    )


@dataclass(frozen=True)
class PolicyRule:
    policy_id: str
    template_id: str
    effect: Literal["allow", "deny"]
    priority: int
    subject_ids: tuple[str, ...]
    tenant_ids: tuple[str, ...]
    departments: tuple[str, ...]
    resource_ids: tuple[str, ...]
    resource_types: tuple[str, ...]
    actions: tuple[str, ...]
    relation_ids: tuple[str, ...]
    purposes: tuple[str, ...]
    required_attributes: tuple[tuple[str, str], ...]
    valid_from: str
    valid_until: str
    delegation_required: bool
    max_delegation_depth: int
    authority: AuthoritySet

    @property
    def specificity(self) -> int:
        selectors = (
            self.subject_ids,
            self.tenant_ids,
            self.departments,
            self.resource_ids,
            self.resource_types,
            self.actions,
            self.relation_ids,
            self.purposes,
        )
        return sum(
            1000 - len(value)
            for value in (
                *selectors,
                tuple(self.required_attributes),
            )
        ) + int(self.delegation_required)

    def canonical(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "template_id": self.template_id,
            "effect": self.effect,
            "priority": self.priority,
            "subjects": {
                "subject_ids": list(self.subject_ids),
                "tenant_ids": list(self.tenant_ids),
                "departments": list(self.departments),
            },
            "resources": {
                "tenant_ids": list(self.tenant_ids),
                "resource_ids": list(self.resource_ids),
                "resource_types": list(self.resource_types),
                "relation_ids": list(self.relation_ids),
            },
            "actions": list(self.actions),
            "purposes": list(self.purposes),
            "conditions": {
                "required_attributes": dict(self.required_attributes),
                "valid_from": self.valid_from,
                "valid_until": self.valid_until,
                "delegation_required": self.delegation_required,
                "max_delegation_depth": self.max_delegation_depth,
            },
            "authority": self.authority.canonical(),
        }


@dataclass(frozen=True)
class PolicySet:
    policy_epoch: int
    policies: tuple[PolicyRule, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy_epoch": self.policy_epoch,
            "policies": [rule.canonical() for rule in self.policies],
        }

    @property
    def policy_hash(self) -> str:
        return canonical_sha256(self.canonical())


def parse_policy_set(value: Mapping[str, Any]) -> PolicySet:
    _exact_keys(value, {"schema_version", "policy_epoch", "policies"}, "policy_set")
    if value["schema_version"] != 1:
        raise PolicySchemaError("未知 policy schema_version")
    epoch = value["policy_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise PolicySchemaError("policy_epoch 必须为正整数")
    raw_rules = value["policies"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise PolicySchemaError("policies 必须是非空数组")
    rules = tuple(_parse_rule(item) for item in raw_rules)
    ids = [rule.policy_id for rule in rules]
    if len(ids) != len(set(ids)):
        raise PolicySchemaError("duplicate policy ID")
    return PolicySet(epoch, tuple(sorted(rules, key=lambda item: item.policy_id)))


def _parse_rule(value: Any) -> PolicyRule:
    if not isinstance(value, Mapping):
        raise PolicySchemaError("policy rule 必须是 object")
    keys = {
        "policy_id",
        "template_id",
        "effect",
        "priority",
        "subjects",
        "resources",
        "actions",
        "purposes",
        "conditions",
        "authority",
    }
    _exact_keys(value, keys, "policy")
    subjects = value["subjects"]
    resources = value["resources"]
    conditions = value["conditions"]
    if not all(isinstance(item, Mapping) for item in (subjects, resources, conditions)):
        raise PolicySchemaError("subjects/resources/conditions 必须是 object")
    _exact_keys(subjects, {"subject_ids", "tenant_ids", "departments"}, "subjects")
    _exact_keys(
        resources,
        {"tenant_ids", "resource_ids", "resource_types", "relation_ids"},
        "resources",
    )
    _exact_keys(
        conditions,
        {
            "required_attributes",
            "valid_from",
            "valid_until",
            "delegation_required",
            "max_delegation_depth",
        },
        "conditions",
    )
    effect = value["effect"]
    if effect not in {"allow", "deny"}:
        raise PolicySchemaError("effect 只能 allow/deny")
    priority = value["priority"]
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise PolicySchemaError("priority 必须是整数")
    required = conditions["required_attributes"]
    if not isinstance(required, Mapping) or not all(
        isinstance(key, str) and isinstance(child, str)
        for key, child in required.items()
    ):
        raise PolicySchemaError("required_attributes 必须是字符串映射")
    delegation = conditions["delegation_required"]
    depth = conditions["max_delegation_depth"]
    if not isinstance(delegation, bool) or not isinstance(depth, int) or depth < 0:
        raise PolicySchemaError("delegation condition 非法")
    authority = _authority(value["authority"])
    if effect == "deny" and not authority.is_empty:
        raise PolicySchemaError("deny policy authority 必须为空")
    if effect == "allow" and authority.is_empty:
        raise PolicySchemaError("allow policy authority 不能为空")
    return PolicyRule(
        policy_id=str(value["policy_id"]),
        template_id=str(value["template_id"]),
        effect=effect,
        priority=priority,
        subject_ids=_strings(subjects["subject_ids"], "subjects.subject_ids"),
        tenant_ids=_strings(resources["tenant_ids"], "resources.tenant_ids"),
        departments=_strings(subjects["departments"], "subjects.departments"),
        resource_ids=_strings(resources["resource_ids"], "resources.resource_ids"),
        resource_types=_strings(
            resources["resource_types"], "resources.resource_types"
        ),
        actions=_strings(value["actions"], "actions"),
        relation_ids=_strings(resources["relation_ids"], "resources.relation_ids"),
        purposes=_strings(value["purposes"], "purposes"),
        required_attributes=tuple(sorted(required.items())),
        valid_from=str(conditions["valid_from"]),
        valid_until=str(conditions["valid_until"]),
        delegation_required=delegation,
        max_delegation_depth=depth,
        authority=authority,
    )


def load_policy_set(path: str | Path) -> PolicySet:
    return parse_policy_set(strict_load_document(path))
