"""真实 CGAR 迭代记录和最终盲分析结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from keyed_gram.authsynth_shared import AnalysisStatus, FindingKind, ShieldDecision


@dataclass(frozen=True)
class RefinementIteration:
    iteration: int
    query_id: str
    abstract_counterexample_digest: str
    replay_result_digest: str
    finding_kind: FindingKind
    refinement_patch_digest: str
    model_before_digest: str
    model_after_digest: str
    shield_before_digest: str
    shield_after_digest: str
    remaining_unknown_query_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "query_id": self.query_id,
            "abstract_counterexample_digest": self.abstract_counterexample_digest,
            "replay_result_digest": self.replay_result_digest,
            "finding_kind": self.finding_kind.value,
            "refinement_patch_digest": self.refinement_patch_digest,
            "model_before_digest": self.model_before_digest,
            "model_after_digest": self.model_after_digest,
            "shield_before_digest": self.shield_before_digest,
            "shield_after_digest": self.shield_after_digest,
            "remaining_unknown_query_count": self.remaining_unknown_query_count,
        }


@dataclass(frozen=True)
class AnalysisOutcome:
    public_case_id: str
    status: AnalysisStatus
    refined_contract: tuple[tuple[str, tuple[dict[str, Any], ...]], ...]
    shield: tuple[ShieldDecision, ...]
    iterations: tuple[RefinementIteration, ...]
    replay_query_count: int
    real_counterexample_count: int
    spurious_counterexample_count: int
    policy_omission_count: int
    composition_omission_count: int
    drift_detected: bool
    call_complete: bool
    trace_complete: bool
    composition_complete: bool
    complete_claim_made: bool
    reason_codes: tuple[str, ...]
    certificate_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_case_id": self.public_case_id,
            "status": self.status.value,
            "refined_contract_digest": self.certificate_digest,
            "shield": [item.to_dict() for item in self.shield],
            "iteration_count": len(self.iterations),
            "replay_query_count": self.replay_query_count,
            "real_counterexample_count": self.real_counterexample_count,
            "spurious_counterexample_count": self.spurious_counterexample_count,
            "policy_omission_count": self.policy_omission_count,
            "composition_omission_count": self.composition_omission_count,
            "drift_detected": self.drift_detected,
            "call_complete": self.call_complete,
            "trace_complete": self.trace_complete,
            "composition_complete": self.composition_complete,
            "complete_claim_made": self.complete_claim_made,
            "reason_codes": list(self.reason_codes),
        }
