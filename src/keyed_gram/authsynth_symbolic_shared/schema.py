"""F2C 有界符号域、赋值与规范化基础类型。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

Scalar: TypeAlias = bool | int | str

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SymbolicSchemaError(ValueError):
    """ToolSymbolicIR schema 或规范化不变量错误。"""


def restricted_token(value: str, label: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value) or "*" in value:
        raise SymbolicSchemaError(f"{label} 不是受限且无 wildcard 的标识符")
    return value


def digest_token(value: str, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise SymbolicSchemaError(f"{label} 不是 SHA-256 digest")
    return value


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SymbolicSchemaError(f"canonical value 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
    """产生逐字节确定的严格 JSON。"""

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


def _pairs(
    values: Mapping[str, Scalar] | tuple[tuple[str, Scalar], ...],
    *,
    label: str,
) -> tuple[tuple[str, Scalar], ...]:
    source = values.items() if isinstance(values, Mapping) else values
    result: dict[str, Scalar] = {}
    for key, value in source:
        name = restricted_token(str(key), f"{label} name")
        if name in result:
            raise SymbolicSchemaError(f"重复 {label}：{name}")
        if not isinstance(value, (bool, int, str)):
            raise SymbolicSchemaError(f"{label}.{name} 不是受限 scalar")
        if isinstance(value, str):
            restricted_token(value, f"{label}.{name}")
        result[name] = value
    return tuple(sorted(result.items()))


@dataclass(frozen=True, order=True)
class DomainField:
    """Bool、Enum 或有界 Int 字段。"""

    name: str
    source: str
    kind: str
    enum_values: tuple[str, ...] = field(default_factory=tuple)
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        restricted_token(self.name, "field name")
        if self.source not in {"input", "state"}:
            raise SymbolicSchemaError("field source 必须是 input/state")
        if self.kind not in {"bool", "enum", "int"}:
            raise SymbolicSchemaError("field kind 仅支持 bool/enum/int")
        if self.kind == "enum":
            values = tuple(
                sorted({restricted_token(v, "enum value") for v in self.enum_values})
            )
            if len(values) < 2 or len(values) != len(self.enum_values):
                raise SymbolicSchemaError("enum 至少两个且不得重复")
            object.__setattr__(self, "enum_values", values)
            if self.minimum is not None or self.maximum is not None:
                raise SymbolicSchemaError("enum 不能带整数边界")
        elif self.kind == "int":
            if (
                not isinstance(self.minimum, int)
                or isinstance(self.minimum, bool)
                or not isinstance(self.maximum, int)
                or isinstance(self.maximum, bool)
                or self.minimum > self.maximum
            ):
                raise SymbolicSchemaError("int 必须具有有限闭区间")
            if self.maximum - self.minimum > 255:
                raise SymbolicSchemaError("F2C v1 单字段整数域不得超过 256")
            if self.enum_values:
                raise SymbolicSchemaError("int 不能带 enum values")
        else:
            if self.enum_values or self.minimum is not None or self.maximum is not None:
                raise SymbolicSchemaError("bool 不接受额外域参数")

    @property
    def values(self) -> tuple[Scalar, ...]:
        if self.kind == "bool":
            return (False, True)
        if self.kind == "enum":
            return self.enum_values
        assert self.minimum is not None and self.maximum is not None
        return tuple(range(self.minimum, self.maximum + 1))

    def accepts(self, value: Scalar) -> bool:
        if self.kind == "bool":
            return isinstance(value, bool)
        if self.kind == "enum":
            return isinstance(value, str) and value in self.enum_values
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and self.minimum is not None
            and self.maximum is not None
            and self.minimum <= value <= self.maximum
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "kind": self.kind,
            "enum_values": list(self.enum_values),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class BoundedSchema:
    """完整、显式且有限的 input/state 域。"""

    fields: tuple[DomainField, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.fields, key=lambda item: (item.source, item.name)))
        identities = [(item.source, item.name) for item in ordered]
        if len(identities) != len(set(identities)):
            raise SymbolicSchemaError("schema 含重复字段")
        if not any(item.source == "input" for item in ordered):
            raise SymbolicSchemaError("schema 至少需要一个 input 字段")
        object.__setattr__(self, "fields", ordered)

    @property
    def input_fields(self) -> tuple[DomainField, ...]:
        return tuple(item for item in self.fields if item.source == "input")

    @property
    def state_fields(self) -> tuple[DomainField, ...]:
        return tuple(item for item in self.fields if item.source == "state")

    @property
    def cardinality(self) -> int:
        total = 1
        for item in self.fields:
            total *= len(item.values)
        return total

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def field(self, source: str, name: str) -> DomainField:
        for item in self.fields:
            if item.source == source and item.name == name:
                return item
        raise SymbolicSchemaError(f"未知字段：{source}.{name}")

    def validate_assignment(self, assignment: ConcreteAssignment) -> None:
        expected_inputs = {item.name for item in self.input_fields}
        expected_state = {item.name for item in self.state_fields}
        inputs = dict(assignment.inputs)
        state = dict(assignment.state)
        if set(inputs) != expected_inputs or set(state) != expected_state:
            raise SymbolicSchemaError("assignment 字段 inventory 与 schema 不一致")
        for item in self.fields:
            value = inputs[item.name] if item.source == "input" else state[item.name]
            if not item.accepts(value):
                raise SymbolicSchemaError(
                    f"assignment 值越界：{item.source}.{item.name}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [item.to_dict() for item in self.fields]}


@dataclass(frozen=True, order=True)
class ConcreteAssignment:
    """Verifier 生成、sandbox 单次执行的具体 input/state 赋值。"""

    inputs: tuple[tuple[str, Scalar], ...]
    state: tuple[tuple[str, Scalar], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _pairs(self.inputs, label="input"))
        object.__setattr__(self, "state", _pairs(self.state, label="state"))

    def value(self, source: str, name: str) -> Scalar:
        values = dict(self.inputs if source == "input" else self.state)
        try:
            return values[name]
        except KeyError as exc:
            raise SymbolicSchemaError(f"assignment 缺少 {source}.{name}") from exc

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": dict(self.inputs), "state": dict(self.state)}


@dataclass(frozen=True, order=True)
class StateChange:
    object_type: str
    object_id: str
    field: str
    before: Scalar
    after: Scalar

    def __post_init__(self) -> None:
        restricted_token(self.object_type, "state object_type")
        restricted_token(self.object_id, "state object_id")
        restricted_token(self.field, "state field")
        for label, value in (("before", self.before), ("after", self.after)):
            if not isinstance(value, (bool, int, str)):
                raise SymbolicSchemaError(f"state change {label} 不是 scalar")

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "object_id": self.object_id,
            "field": self.field,
            "before": self.before,
            "after": self.after,
        }


@dataclass(frozen=True)
class StructuredStateDiff:
    changes: tuple[StateChange, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(set(self.changes)))
        keys = [(item.object_type, item.object_id, item.field) for item in ordered]
        if len(keys) != len(set(keys)):
            raise SymbolicSchemaError("同一 state field 出现多次 change")
        object.__setattr__(self, "changes", ordered)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"changes": [item.to_dict() for item in self.changes]}
