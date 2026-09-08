"""F2C 符号分析器的 evidence、迭代与最终结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from keyed_gram.authsynth_symbolic_shared import (
    AnalyzerEvidence,
    BoundedCompletenessCertificate,
    CertificateStatus,
    ConcreteAssignment,
    ContractStatus,
    PatchRecord,
    ShieldStatus,
    SymbolicEffectContract,
    VerificationKind,
)


@dataclass(frozen=True)
class SymbolicRefinementIteration:
    iteration: int
    verification_kind: VerificationKind
    counterexample_digest: str | None
    replay_result_digest: str | None
    patch: PatchRecord
    contract_before_digest: str
    contract_after_digest: str
    shield_before_digest: str
    shield_after_digest: str
    remaining_replay_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "verification_kind": self.verification_kind.value,
            "counterexample_digest": self.counterexample_digest,
            "replay_result_digest": self.replay_result_digest,
            "patch": self.patch.to_dict(),
            "contract_before_digest": self.contract_before_digest,
            "contract_after_digest": self.contract_after_digest,
            "shield_before_digest": self.shield_before_digest,
            "shield_after_digest": self.shield_after_digest,
            "remaining_replay_budget": self.remaining_replay_budget,
        }


@dataclass(frozen=True)
class SymbolicAnalysisOutcome:
    public_case_id: str
    contract_status: ContractStatus
    shield_status: ShieldStatus
    certificate_status: CertificateStatus
    candidate_contract: SymbolicEffectContract
    certificate: BoundedCompletenessCertificate | None
    invalidated_certificate_ids: tuple[str, ...]
    evidence: tuple[AnalyzerEvidence, ...]
    replayed_assignments: tuple[ConcreteAssignment, ...]
    iterations: tuple[SymbolicRefinementIteration, ...]
    patch_records: tuple[PatchRecord, ...]
    solver_call_count: int
    replay_query_count: int
    reason_codes: tuple[str, ...]
    synthesis_time_ms: float
    verification_time_ms: float

    @property
    def complete_claim_made(self) -> bool:
        return self.contract_status == ContractStatus.VERIFIED_COMPLETE

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "public_case_id": self.public_case_id,
            "contract_status": self.contract_status.value,
            "shield_status": self.shield_status.value,
            "certificate_status": self.certificate_status.value,
            "candidate_contract_digest": self.candidate_contract.digest,
            "certificate_id": None
            if self.certificate is None
            else self.certificate.certificate_id,
            "invalidated_certificate_ids": list(self.invalidated_certificate_ids),
            "evidence_digests": [item.digest for item in self.evidence],
            "replayed_assignment_digests": [
                item.digest for item in self.replayed_assignments
            ],
            "iterations": [item.to_dict() for item in self.iterations],
            "patch_types": [item.patch_type for item in self.patch_records],
            "solver_call_count": self.solver_call_count,
            "replay_query_count": self.replay_query_count,
            "reason_codes": list(self.reason_codes),
            "synthesis_time_ms": self.synthesis_time_ms,
            "verification_time_ms": self.verification_time_ms,
        }
