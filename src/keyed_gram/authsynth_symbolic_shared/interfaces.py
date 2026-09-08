"""Analyzer、Verifier 与 replay sandbox 之间的唯一共享窄接口。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .contracts import SymbolicEffectContract, TrustedSafetySpecification
from .effects import ConcreteEffect
from .formulas import GrammarLimits
from .schema import (
    BoundedSchema,
    ConcreteAssignment,
    StructuredStateDiff,
    canonical_digest,
    restricted_token,
)


class VerificationKind(str, Enum):
    MISSING_EFFECT = "MISSING_EFFECT"
    SPURIOUS_EFFECT = "SPURIOUS_EFFECT"
    FIELD_BINDING_MISMATCH = "FIELD_BINDING_MISMATCH"
    STATE_UPDATE_MISMATCH = "STATE_UPDATE_MISMATCH"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    EQUIVALENT = "EQUIVALENT_ON_BOUNDED_DOMAIN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SymbolicAnalyzerInput:
    """不含 query catalog、实现、路径标签或 evaluator ground truth。"""

    case_handle: str
    public_case_id: str
    tool_id: str
    schema: BoundedSchema
    declared_contract: SymbolicEffectContract
    static_hypotheses: tuple[SymbolicEffectContract, ...]
    trusted_safety_spec: TrustedSafetySpecification
    expected_version_digest: str
    existing_certificate_id: str | None
    grammar_limits: GrammarLimits
    replay_budget: int
    solver_timeout_ms: int
    max_iterations: int

    def __post_init__(self) -> None:
        restricted_token(self.case_handle, "case handle")
        restricted_token(self.public_case_id, "public case id")
        restricted_token(self.tool_id, "tool id")
        if self.existing_certificate_id is not None:
            restricted_token(self.existing_certificate_id, "existing certificate id")
        if not 1 <= self.replay_budget <= 64:
            raise ValueError("replay budget 超界")
        if not 1 <= self.max_iterations <= 128:
            raise ValueError("max iterations 超界")
        if not 1 <= self.solver_timeout_ms <= 60_000:
            raise ValueError("solver timeout 超界")
        if self.schema.cardinality < 2:
            raise ValueError("symbolic domain 过小")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "public_case_id": self.public_case_id,
            "tool_id": self.tool_id,
            "schema": self.schema.to_dict(),
            "declared_contract": self.declared_contract.to_dict(),
            "static_hypotheses": [item.to_dict() for item in self.static_hypotheses],
            "trusted_safety_spec": self.trusted_safety_spec.to_dict(),
            "expected_version_digest": self.expected_version_digest,
            "existing_certificate_id": self.existing_certificate_id,
            "grammar_limits": self.grammar_limits.to_dict(),
            "replay_budget": self.replay_budget,
            "solver_timeout_ms": self.solver_timeout_ms,
            "max_iterations": self.max_iterations,
            "case_handle_digest": canonical_digest(self.case_handle),
        }


@dataclass(frozen=True)
class VerificationResult:
    kind: VerificationKind
    assignment: ConcreteAssignment | None = None
    counterexample_digest: str | None = None
    observed_version_digest: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        counterexample_kinds = {
            VerificationKind.MISSING_EFFECT,
            VerificationKind.SPURIOUS_EFFECT,
            VerificationKind.FIELD_BINDING_MISMATCH,
            VerificationKind.STATE_UPDATE_MISMATCH,
        }
        if self.kind in counterexample_kinds and (
            self.assignment is None or self.counterexample_digest is None
        ):
            raise ValueError("counterexample result 缺少具体 assignment/digest")
        if (
            self.kind == VerificationKind.VERSION_MISMATCH
            and self.observed_version_digest is None
        ):
            raise ValueError("version mismatch 缺少 observed digest")
        if (
            self.kind in {VerificationKind.EQUIVALENT, VerificationKind.UNKNOWN}
            and self.assignment is not None
        ):
            raise ValueError("equivalent/unknown 不得携带 assignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "assignment": None
            if self.assignment is None
            else self.assignment.to_dict(),
            "counterexample_digest": self.counterexample_digest,
            "observed_version_digest": self.observed_version_digest,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class SymbolicReplayResult:
    assignment: ConcreteAssignment
    immediate_events: tuple[ConcreteEffect, ...]
    delayed_events: tuple[ConcreteEffect, ...]
    state_diff: StructuredStateDiff
    bounded_quiescence_reached: bool
    instrumentation_coverage: float
    observed_version_digest: str
    exit_status: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.instrumentation_coverage <= 1.0:
            raise ValueError("instrumentation coverage 超界")
        restricted_token(self.exit_status, "replay exit status")
        ordered = tuple(
            sorted(
                (*self.immediate_events, *self.delayed_events),
                key=lambda x: x.semantic_key,
            )
        )
        if len(ordered) != len(set(ordered)):
            raise ValueError("replay event 重复")

    @property
    def events(self) -> tuple[ConcreteEffect, ...]:
        return tuple(
            sorted(
                (*self.immediate_events, *self.delayed_events),
                key=lambda x: x.semantic_key,
            )
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment": self.assignment.to_dict(),
            "immediate_events": [item.to_dict() for item in self.immediate_events],
            "delayed_events": [item.to_dict() for item in self.delayed_events],
            "state_diff": self.state_diff.to_dict(),
            "bounded_quiescence_reached": self.bounded_quiescence_reached,
            "instrumentation_coverage": self.instrumentation_coverage,
            "observed_version_digest": self.observed_version_digest,
            "exit_status": self.exit_status,
        }


class ContractVerifier(Protocol):
    solver_name: str
    solver_version: str

    def check(
        self, case_handle: str, candidate_contract: SymbolicEffectContract
    ) -> VerificationResult:
        """只返回差异种类与一个具体 assignment，不返回内部公式。"""


class SymbolicReplayClient(Protocol):
    def replay(
        self, case_handle: str, assignment: ConcreteAssignment
    ) -> SymbolicReplayResult:
        """对一个 verifier 生成的 assignment 执行窄 replay。"""


@dataclass(frozen=True)
class PatchRecord:
    patch_type: str
    target_digest: str

    def __post_init__(self) -> None:
        restricted_token(self.patch_type, "patch type")

    def to_dict(self) -> dict[str, str]:
        return {"patch_type": self.patch_type, "target_digest": self.target_digest}


@dataclass(frozen=True)
class AnalyzerEvidence:
    assignment: ConcreteAssignment
    events: tuple[ConcreteEffect, ...]
    state_diff: StructuredStateDiff
    observed_version_digest: str

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "assignment": self.assignment.to_dict(),
                "events": [item.to_dict() for item in self.events],
                "state_diff": self.state_diff.to_dict(),
                "observed_version_digest": self.observed_version_digest,
            }
        )
