"""ToolEffectIR 的严格、有限且可规范化的核心类型。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")


class ToolEffectError(ValueError):
    """ToolEffectIR schema、执行或不变量错误。"""


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ToolEffectError(f"{label} 不是受限标识符")
    if "*" in value:
        raise ToolEffectError(f"{label} 不允许 wildcard")
    return value


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolEffectError(f"canonical value 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
    """返回本阶段固定的 sorted compact JSON 子集。"""

    _finite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    SEND = "send"
    EXPORT = "export"
    EXECUTE = "execute"
    QUEUE = "queue"
    SCHEDULE = "schedule"
    CALLBACK = "callback"
    COMMIT = "commit"


class EffectPhase(str, Enum):
    IMMEDIATE = "immediate"
    DELAYED = "delayed"


class OmissionType(str, Enum):
    NONE = "none"
    CONTRACT_OMISSION = "contract_omission"
    POLICY_OMISSION = "policy_omission"
    IMPLEMENTATION_DRIFT = "implementation_drift"
    COMPOSITION_OMISSION = "composition_omission"
    ABSTRACTION_IMPRECISION = "abstraction_imprecision"


def _pairs(
    values: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    source = values.items() if isinstance(values, Mapping) else values
    result: dict[str, str] = {}
    for key, value in source:
        name = _token(str(key), "field name")
        item = _token(str(value), f"field {name}")
        if name in result:
            raise ToolEffectError(f"重复字段：{name}")
        result[name] = item
    return tuple(sorted(result.items()))


@dataclass(frozen=True, order=True)
class Environment:
    """有限、可信的执行环境属性。"""

    values: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _pairs(self.values))

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> Environment:
        return cls(_pairs(values))

    def get(self, key: str, default: str | None = None) -> str | None:
        return dict(self.values).get(key, default)

    def to_dict(self) -> dict[str, str]:
        return dict(self.values)


@dataclass(frozen=True, order=True)
class ToolCall:
    call_id: str
    tool_name: str
    version: str
    arguments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _token(self.call_id, "call_id")
        _token(self.tool_name, "tool_name")
        _token(self.version, "version")
        object.__setattr__(self, "arguments", _pairs(self.arguments))

    @classmethod
    def create(
        cls,
        call_id: str,
        tool_name: str,
        version: str,
        arguments: Mapping[str, str],
    ) -> ToolCall:
        return cls(call_id, tool_name, version, _pairs(arguments))

    def argument(self, name: str) -> str:
        try:
            return dict(self.arguments)[name]
        except KeyError as exc:
            raise ToolEffectError(f"tool call 缺少参数：{name}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "version": self.version,
            "arguments": dict(self.arguments),
        }


def _resolve(reference: str | None, call: ToolCall) -> str | None:
    if reference is None:
        return None
    if reference.startswith("arg:"):
        return call.argument(reference[4:])
    if reference.startswith("const:"):
        return _token(reference[6:], "constant reference")
    raise ToolEffectError("effect reference 必须使用 arg: 或 const:")


@dataclass(frozen=True, order=True)
class EffectAtom:
    """一个已发生、可由最终状态 oracle 判定的具体效果。"""

    kind: EffectKind
    tenant_id: str
    resource_id: str
    destination_id: str | None
    phase: EffectPhase
    argument_roles: tuple[tuple[str, str], ...]
    source_tool: str
    source_version: str

    def __post_init__(self) -> None:
        _token(self.tenant_id, "effect tenant")
        _token(self.resource_id, "effect resource")
        if self.destination_id is not None:
            _token(self.destination_id, "effect destination")
        _token(self.source_tool, "effect source tool")
        _token(self.source_version, "effect source version")
        object.__setattr__(self, "argument_roles", _pairs(self.argument_roles))

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.kind.value,
            self.tenant_id,
            self.resource_id,
            self.destination_id,
            self.phase.value,
            self.argument_roles,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "tenant_id": self.tenant_id,
            "resource_id": self.resource_id,
            "destination_id": self.destination_id,
            "phase": self.phase.value,
            "argument_roles": dict(self.argument_roles),
            "source_tool": self.source_tool,
            "source_version": self.source_version,
        }


@dataclass(frozen=True, order=True)
class EffectTemplate:
    """由参数角色实例化为具体效果的 contract/implementation 模板。"""

    kind: EffectKind
    tenant_ref: str
    resource_ref: str
    destination_ref: str | None = None
    phase: EffectPhase = EffectPhase.IMMEDIATE
    security_roles: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for label, reference in (
            ("tenant_ref", self.tenant_ref),
            ("resource_ref", self.resource_ref),
            ("destination_ref", self.destination_ref),
        ):
            if reference is None:
                continue
            if not reference.startswith(("arg:", "const:")):
                raise ToolEffectError(f"{label} 必须使用 arg:/const:")
            _token(reference.split(":", 1)[1], label)
        roles = tuple(
            sorted({_token(role, "security role") for role in self.security_roles})
        )
        object.__setattr__(self, "security_roles", roles)

    def instantiate(self, call: ToolCall) -> EffectAtom:
        roles = tuple((role, call.argument(role)) for role in self.security_roles)
        tenant = _resolve(self.tenant_ref, call)
        resource = _resolve(self.resource_ref, call)
        if tenant is None or resource is None:
            raise ToolEffectError("tenant/resource reference 不得为空")
        return EffectAtom(
            kind=self.kind,
            tenant_id=tenant,
            resource_id=resource,
            destination_id=_resolve(self.destination_ref, call),
            phase=self.phase,
            argument_roles=roles,
            source_tool=call.tool_name,
            source_version=call.version,
        )

    @classmethod
    def literal_from_effect(cls, effect: EffectAtom) -> EffectTemplate:
        return cls(
            kind=effect.kind,
            tenant_ref=f"const:{effect.tenant_id}",
            resource_ref=f"const:{effect.resource_id}",
            destination_ref=(
                None
                if effect.destination_id is None
                else f"const:{effect.destination_id}"
            ),
            phase=effect.phase,
            security_roles=tuple(name for name, _ in effect.argument_roles),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "tenant_ref": self.tenant_ref,
            "resource_ref": self.resource_ref,
            "destination_ref": self.destination_ref,
            "phase": self.phase.value,
            "security_roles": list(self.security_roles),
        }


@dataclass(frozen=True)
class DeclaredContract:
    contract_id: str
    tool_name: str
    bound_version: str
    effects: frozenset[EffectTemplate]

    def __post_init__(self) -> None:
        _token(self.contract_id, "contract_id")
        _token(self.tool_name, "contract tool")
        _token(self.bound_version, "contract version")
        object.__setattr__(self, "effects", frozenset(self.effects))

    def instantiate(self, call: ToolCall) -> frozenset[EffectAtom]:
        if call.tool_name != self.tool_name:
            raise ToolEffectError("contract/tool 不匹配")
        return frozenset(template.instantiate(call) for template in self.effects)

    def covers(self, effect: EffectAtom, call: ToolCall) -> bool:
        for declared in self.instantiate(call):
            same_effect = declared.semantic_key[:5] == effect.semantic_key[:5]
            declared_roles = set(declared.argument_roles)
            if same_effect and set(effect.argument_roles).issubset(declared_roles):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "tool_name": self.tool_name,
            "bound_version": self.bound_version,
            "effects": [item.to_dict() for item in sorted(self.effects)],
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class ToolTransition:
    transition_id: str
    required_environment: tuple[tuple[str, str], ...]
    immediate_effects: tuple[EffectTemplate, ...]
    delayed_effects: tuple[EffectTemplate, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _token(self.transition_id, "transition_id")
        object.__setattr__(
            self, "required_environment", _pairs(self.required_environment)
        )
        object.__setattr__(
            self, "immediate_effects", tuple(sorted(set(self.immediate_effects)))
        )
        object.__setattr__(
            self, "delayed_effects", tuple(sorted(set(self.delayed_effects)))
        )

    def enabled(self, environment: Environment) -> bool:
        values = dict(environment.values)
        return all(values.get(key) == value for key, value in self.required_environment)


@dataclass(frozen=True)
class ToolImplementation:
    tool_name: str
    version: str
    transitions: tuple[ToolTransition, ...]

    def __post_init__(self) -> None:
        _token(self.tool_name, "implementation tool")
        _token(self.version, "implementation version")
        if not self.transitions:
            raise ToolEffectError("tool implementation 至少需要一个 transition")
        ids = [item.transition_id for item in self.transitions]
        if len(ids) != len(set(ids)):
            raise ToolEffectError("transition_id 重复")
        object.__setattr__(
            self,
            "transitions",
            tuple(sorted(self.transitions, key=lambda item: item.transition_id)),
        )


@dataclass(frozen=True, order=True)
class TraceStep:
    call: ToolCall
    environment: Environment


@dataclass(frozen=True)
class TraceScenario:
    trace_id: str
    steps: tuple[TraceStep, ...]
    expected_forbidden: bool

    def __post_init__(self) -> None:
        _token(self.trace_id, "trace_id")
        if not self.steps:
            raise ToolEffectError("trace 至少包含一个 step")


@dataclass(frozen=True, order=True)
class EffectSelector:
    kind: EffectKind
    tenant_id: str | None = None
    resource_id: str | None = None
    destination_id: str | None = None
    phase: EffectPhase | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("selector tenant", self.tenant_id),
            ("selector resource", self.resource_id),
            ("selector destination", self.destination_id),
        ):
            if value is not None:
                _token(value, label)

    def matches(self, effect: EffectAtom) -> bool:
        return (
            effect.kind == self.kind
            and (self.tenant_id is None or effect.tenant_id == self.tenant_id)
            and (self.resource_id is None or effect.resource_id == self.resource_id)
            and (
                self.destination_id is None
                or effect.destination_id == self.destination_id
            )
            and (self.phase is None or effect.phase == self.phase)
        )


@dataclass(frozen=True)
class ForbiddenRule:
    rule_id: str
    selectors: tuple[EffectSelector, ...]

    def __post_init__(self) -> None:
        _token(self.rule_id, "forbidden rule")
        if not self.selectors:
            raise ToolEffectError("forbidden rule 至少包含一个 selector")
        object.__setattr__(self, "selectors", tuple(self.selectors))

    def violated(self, effects: Iterable[EffectAtom]) -> bool:
        observed = tuple(effects)
        return all(
            any(selector.matches(effect) for effect in observed)
            for selector in self.selectors
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "selectors": [
                {
                    "kind": selector.kind.value,
                    "tenant_id": selector.tenant_id,
                    "resource_id": selector.resource_id,
                    "destination_id": selector.destination_id,
                    "phase": None if selector.phase is None else selector.phase.value,
                }
                for selector in self.selectors
            ],
        }
