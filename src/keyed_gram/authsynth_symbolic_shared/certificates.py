"""F2C 有界完备性 certificate 的严格状态与载荷。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .schema import canonical_digest, restricted_token


class ContractStatus(str, Enum):
    VERIFIED_COMPLETE = "CONTRACT_VERIFIED_COMPLETE"
    INCOMPLETE = "CONTRACT_INCOMPLETE"
    UNKNOWN = "CONTRACT_UNKNOWN"
    INVALIDATED_BY_DRIFT = "CONTRACT_INVALIDATED_BY_DRIFT"


class ShieldStatus(str, Enum):
    SAFE = "SHIELD_SAFE"
    UNSAFE = "SHIELD_UNSAFE"
    UNKNOWN = "SHIELD_UNKNOWN"


class CertificateStatus(str, Enum):
    VALID = "CERTIFICATE_VALID"
    INVALID = "CERTIFICATE_INVALID"
    NOT_ISSUED = "CERTIFICATE_NOT_ISSUED"


@dataclass(frozen=True)
class BoundedCompletenessCertificate:
    certificate_id: str
    contract_digest: str
    tool_id: str
    implementation_version_digest: str
    input_schema_digest: str
    state_schema_digest: str
    bounded_domain_cardinality: int
    solver_name: str
    solver_version: str
    proof_result: str
    certificate_status: CertificateStatus
    schema_version: int = 1

    def __post_init__(self) -> None:
        restricted_token(self.certificate_id, "certificate id")
        restricted_token(self.tool_id, "certificate tool id")
        restricted_token(self.solver_name, "solver name")
        restricted_token(self.solver_version, "solver version")
        restricted_token(self.proof_result, "proof result")
        if self.bounded_domain_cardinality <= 0:
            raise ValueError("certificate domain cardinality 必须为正")

    @classmethod
    def issue(
        cls,
        *,
        contract_digest: str,
        tool_id: str,
        implementation_version_digest: str,
        input_schema_digest: str,
        state_schema_digest: str,
        bounded_domain_cardinality: int,
        solver_name: str,
        solver_version: str,
    ) -> BoundedCompletenessCertificate:
        payload = {
            "contract_digest": contract_digest,
            "tool_id": tool_id,
            "implementation_version_digest": implementation_version_digest,
            "input_schema_digest": input_schema_digest,
            "state_schema_digest": state_schema_digest,
            "bounded_domain_cardinality": bounded_domain_cardinality,
            "solver_name": solver_name,
            "solver_version": solver_version,
            "proof_result": "bounded-domain-equivalent",
        }
        return cls(
            certificate_id=f"cert-{canonical_digest(payload)[:24]}",
            contract_digest=contract_digest,
            tool_id=tool_id,
            implementation_version_digest=implementation_version_digest,
            input_schema_digest=input_schema_digest,
            state_schema_digest=state_schema_digest,
            bounded_domain_cardinality=bounded_domain_cardinality,
            solver_name=solver_name,
            solver_version=solver_version,
            proof_result="bounded-domain-equivalent",
            certificate_status=CertificateStatus.VALID,
        )

    def invalidated(self) -> BoundedCompletenessCertificate:
        return type(self)(
            **{
                **self.__dict__,
                "certificate_status": CertificateStatus.INVALID,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "certificate_id": self.certificate_id,
            "contract_digest": self.contract_digest,
            "tool_id": self.tool_id,
            "implementation_version_digest": self.implementation_version_digest,
            "input_schema_digest": self.input_schema_digest,
            "state_schema_digest": self.state_schema_digest,
            "bounded_domain_cardinality": self.bounded_domain_cardinality,
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "proof_result": self.proof_result,
            "certificate_status": self.certificate_status.value,
        }
