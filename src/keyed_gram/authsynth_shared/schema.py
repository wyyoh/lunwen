"""F2B 三世界之间唯一允许共享的严格、可规范化 schema。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")


class SchemaError(ValueError):
    """盲分析接口的 schema 或不变量错误。"""


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value) or "*" in value:
        raise SchemaError(f"{label} 不是受限且无 wildcard 的标识符")
    return value


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError(f"canonical value 含 NaN/Infinity：{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
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
    values: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    source = values.items() if isinstance(values, Mapping) else values
    result: dict[str, str] = {}
    for key, value in source:
        name = _token(str(key), "attribute name")
        item = _token(str(value), f"attribute {name}")
        if name in result:
            raise SchemaError(f"重复 attribute：{name}")
        result[name] = item
    return tuple(sorted(result.items()))


class EvidenceOrigin(str, Enum):
    DECLARED = "declared"
    STATIC = "static"
    OBSERVED = "observed"
    HYPOTHESIS = "hypothesis"


class AnalysisStatus(str, Enum):
    VERIFIED_COMPLETE = "VERIFIED_COMPLETE"
    COUNTEREXAMPLE_FOUND = "COUNTEREXAMPLE_FOUND"
    UNKNOWN = "UNKNOWN"


class FindingKind(str, Enum):
    REAL_CONTRACT_OMISSION = "real_contract_omission"
    REAL_POLICY_OMISSION = "real_policy_omission"
    REAL_COMPOSITION_OMISSION = "real_composition_omission"
    IMPLEMENTATION_DRIFT = "implementation_drift"
    SPURIOUS_ABSTRACTION = "spurious_abstraction"
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    QUIESCENCE_UNKNOWN = "quiescence_unknown"
    SECURITY_CLASSIFICATION_UNKNOWN = "security_classification_unknown"


@dataclass(frozen=True, order=True)
class EventRecord:
    """一次具体 replay 中可观测的、有序且带因果父节点的效果。"""

    event_id: str
    call_id: str
    sequence: int
    phase: str
    kind: str
    tenant_id: str
    resource_id: str
    destination_id: str | None = None
    parent_event_id: str | None = None
    argument_roles: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for label, value in (
            ("event_id", self.event_id),
            ("call_id", self.call_id),
            ("phase", self.phase),
            ("kind", self.kind),
            ("tenant_id", self.tenant_id),
            ("resource_id", self.resource_id),
        ):
            _token(value, label)
        if self.destination_id is not None:
            _token(self.destination_id, "destination_id")
        if self.parent_event_id is not None:
            _token(self.parent_event_id, "parent_event_id")
        if self.sequence < 0:
            raise SchemaError("event sequence 必须非负")
        if self.phase not in {"immediate", "delayed"}:
            raise SchemaError("event phase 非法")
        object.__setattr__(self, "argument_roles", _pairs(self.argument_roles))

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.call_id,
            self.sequence,
            self.phase,
            self.kind,
            self.tenant_id,
            self.resource_id,
            self.destination_id,
            self.parent_event_id,
            self.argument_roles,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "sequence": self.sequence,
            "phase": self.phase,
            "kind": self.kind,
            "tenant_id": self.tenant_id,
            "resource_id": self.resource_id,
            "destination_id": self.destination_id,
            "parent_event_id": self.parent_event_id,
            "argument_roles": dict(self.argument_roles),
        }


@dataclass(frozen=True, order=True)
class QuerySpec:
    """分析器可主动 replay 的有限输入；不含实现或隐藏标签。"""

    query_id: str
    opaque_tool_id: str
    call_id: str
    declared_version: str
    environment_seed: str
    arguments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("query_id", self.query_id),
            ("opaque_tool_id", self.opaque_tool_id),
            ("call_id", self.call_id),
            ("declared_version", self.declared_version),
            ("environment_seed", self.environment_seed),
        ):
            _token(value, label)
        object.__setattr__(self, "arguments", _pairs(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "opaque_tool_id": self.opaque_tool_id,
            "call_id": self.call_id,
            "declared_version": self.declared_version,
            "environment_seed": self.environment_seed,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True, order=True)
class WorkflowEdge:
    source_state: str
    action_id: str
    query_id: str
    successor_state: str

    def __post_init__(self) -> None:
        for label, value in (
            ("source_state", self.source_state),
            ("action_id", self.action_id),
            ("query_id", self.query_id),
            ("successor_state", self.successor_state),
        ):
            _token(value, label)

    def to_dict(self) -> dict[str, str]:
        return {
            "source_state": self.source_state,
            "action_id": self.action_id,
            "query_id": self.query_id,
            "successor_state": self.successor_state,
        }


@dataclass(frozen=True, order=True)
class EvidenceHypothesis:
    hypothesis_id: str
    query_id: str
    event: EventRecord
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        _token(self.hypothesis_id, "hypothesis_id")
        _token(self.query_id, "hypothesis query_id")
        if self.origin not in {EvidenceOrigin.STATIC, EvidenceOrigin.HYPOTHESIS}:
            raise SchemaError("初始 hypothesis 只能来自 static/hypothesis")
        if self.event.call_id == "":
            raise SchemaError("hypothesis event 缺少 call_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "query_id": self.query_id,
            "event": self.event.to_dict(),
            "origin": self.origin.value,
        }


@dataclass(frozen=True)
class HighLevelSafetySpec:
    """可信高层 bad-state 规范；不枚举隐藏 transition。"""

    spec_id: str
    protected_tenant_id: str
    forbidden_external_destinations: frozenset[str] = field(default_factory=frozenset)
    forbidden_effect_kinds: frozenset[str] = field(default_factory=frozenset)
    forbidden_kind_combinations: tuple[tuple[str, ...], ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        _token(self.spec_id, "spec_id")
        _token(self.protected_tenant_id, "protected_tenant_id")
        object.__setattr__(
            self,
            "forbidden_external_destinations",
            frozenset(
                _token(item, "forbidden destination")
                for item in self.forbidden_external_destinations
            ),
        )
        object.__setattr__(
            self,
            "forbidden_effect_kinds",
            frozenset(
                _token(item, "forbidden kind") for item in self.forbidden_effect_kinds
            ),
        )
        combinations = tuple(
            sorted(
                tuple(sorted(_token(item, "combination kind") for item in combo))
                for combo in self.forbidden_kind_combinations
            )
        )
        object.__setattr__(self, "forbidden_kind_combinations", combinations)

    def violation_codes(self, events: Sequence[EventRecord]) -> tuple[str, ...]:
        codes: set[str] = set()
        kinds = {event.kind for event in events}
        for event in events:
            if event.tenant_id != self.protected_tenant_id:
                codes.add("cross_tenant_effect")
            if event.destination_id in self.forbidden_external_destinations:
                codes.add("forbidden_external_destination")
            if event.kind in self.forbidden_effect_kinds:
                codes.add("forbidden_effect_kind")
        for combination in self.forbidden_kind_combinations:
            if set(combination).issubset(kinds):
                codes.add("forbidden_effect_composition")
        return tuple(sorted(codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "protected_tenant_id": self.protected_tenant_id,
            "forbidden_external_destinations": sorted(
                self.forbidden_external_destinations
            ),
            "forbidden_effect_kinds": sorted(self.forbidden_effect_kinds),
            "forbidden_kind_combinations": [
                list(item) for item in self.forbidden_kind_combinations
            ],
        }


@dataclass(frozen=True)
class AnalyzerCaseInput:
    """盲分析器的完整输入；schema 刻意不提供 evaluator 标签或实现。"""

    case_handle: str
    public_case_id: str
    declared_contract: tuple[tuple[str, tuple[EventRecord, ...]], ...]
    declared_policy_codes: frozenset[str]
    trusted_safety_spec: HighLevelSafetySpec | None
    query_catalog: tuple[QuerySpec, ...]
    workflow_edges: tuple[WorkflowEdge, ...]
    initial_states: frozenset[str]
    terminal_states: frozenset[str]
    hypotheses: tuple[EvidenceHypothesis, ...]
    replay_budget: int
    expected_version_digest: str

    def __post_init__(self) -> None:
        _token(self.case_handle, "case_handle")
        _token(self.public_case_id, "public_case_id")
        _token(self.expected_version_digest, "expected_version_digest")
        if self.replay_budget < 0:
            raise SchemaError("replay budget 不能为负")
        query_ids = [query.query_id for query in self.query_catalog]
        if len(query_ids) != len(set(query_ids)):
            raise SchemaError("query catalog 含重复 query_id")
        known = set(query_ids)
        if any(edge.query_id not in known for edge in self.workflow_edges):
            raise SchemaError("workflow edge 引用未知 query")
        contract_ids = [item[0] for item in self.declared_contract]
        if len(contract_ids) != len(set(contract_ids)) or not set(
            contract_ids
        ).issubset(known):
            raise SchemaError("declared contract query key 非法")
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise SchemaError("重复 hypothesis_id")
        if any(item.query_id not in known for item in self.hypotheses):
            raise SchemaError("hypothesis 引用未知 query")
        for state in (*self.initial_states, *self.terminal_states):
            _token(state, "workflow state")

    def contract_map(self) -> dict[str, tuple[EventRecord, ...]]:
        return dict(self.declared_contract)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "case_handle": self.case_handle,
            "public_case_id": self.public_case_id,
            "declared_contract": [
                {"query_id": query_id, "events": [event.to_dict() for event in events]}
                for query_id, events in self.declared_contract
            ],
            "declared_policy_codes": sorted(self.declared_policy_codes),
            "trusted_safety_spec": None
            if self.trusted_safety_spec is None
            else self.trusted_safety_spec.to_dict(),
            "query_catalog": [item.to_dict() for item in self.query_catalog],
            "workflow_edges": [item.to_dict() for item in self.workflow_edges],
            "initial_states": sorted(self.initial_states),
            "terminal_states": sorted(self.terminal_states),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "replay_budget": self.replay_budget,
            "expected_version_digest": self.expected_version_digest,
        }


@dataclass(frozen=True)
class ReplayResult:
    """sandbox replay 唯一可返回给 analyzer 的结果。"""

    query_id: str
    events: tuple[EventRecord, ...]
    state_diff_digest: str
    exit_status: str
    bounded_quiescence_reached: bool
    instrumentation_coverage: float
    observed_version_digest: str

    def __post_init__(self) -> None:
        _token(self.query_id, "replay query_id")
        _token(self.state_diff_digest, "state_diff_digest")
        _token(self.exit_status, "exit_status")
        _token(self.observed_version_digest, "observed_version_digest")
        if not 0.0 <= self.instrumentation_coverage <= 1.0:
            raise SchemaError("instrumentation coverage 超界")
        if tuple(sorted(self.events, key=lambda item: item.sequence)) != self.events:
            raise SchemaError("replay events 必须按 sequence 排序")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "events": [item.to_dict() for item in self.events],
            "state_diff_digest": self.state_diff_digest,
            "exit_status": self.exit_status,
            "bounded_quiescence_reached": self.bounded_quiescence_reached,
            "instrumentation_coverage": self.instrumentation_coverage,
            "observed_version_digest": self.observed_version_digest,
        }


@dataclass(frozen=True, order=True)
class ShieldDecision:
    state: str
    allowed_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        _token(self.state, "shield state")
        actions = tuple(
            sorted({_token(item, "shield action") for item in self.allowed_actions})
        )
        object.__setattr__(self, "allowed_actions", actions)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "allowed_actions": list(self.allowed_actions)}
