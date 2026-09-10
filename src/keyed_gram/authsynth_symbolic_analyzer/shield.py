"""基于 symbolic contract 的单调用 fail-closed Shield。"""

from __future__ import annotations

from keyed_gram.authsynth_symbolic_shared import (
    ConcreteAssignment,
    ContractStatus,
    StateChange,
    StructuredStateDiff,
    SymbolicEffectContract,
    TrustedSafetySpecification,
    canonical_digest,
)


def shield_digest(
    contract: SymbolicEffectContract,
    safety_spec: TrustedSafetySpecification,
    contract_status: ContractStatus,
) -> str:
    return canonical_digest(
        {
            "contract_digest": contract.digest,
            "safety_spec": safety_spec.to_dict(),
            "contract_status": contract_status.value,
            "unknown_regions_fail_closed": True,
        }
    )


def symbolic_shield_allows(
    contract: SymbolicEffectContract,
    safety_spec: TrustedSafetySpecification,
    contract_status: ContractStatus,
    assignment: ConcreteAssignment,
) -> bool:
    if contract_status != ContractStatus.VERIFIED_COMPLETE:
        return False
    events, updates = contract.predict(assignment)
    changes = []
    for field, after in sorted(updates.items()):
        before = assignment.value("state", field)
        if before != after:
            changes.append(StateChange("record", "state", field, before, after))
    return not safety_spec.violation_codes(events, StructuredStateDiff(tuple(changes)))
