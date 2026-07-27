"""AuthCapV1 签名、witness 与 non-amplification 的联合 verifier。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .authcap_atoms import (
    AtomizedAuthority,
    authority_set_to_atomized,
    verify_authority_non_amplification,
)
from .authcap_capability import (
    AuthCapPublicKeyRing,
    AuthCapTokenError,
)
from .authcap_compiler import (
    TrustedRequestEnvelope,
    attach_mandatory_grant_constraints,
    normalize_proposal_authority,
    validate_trusted_request_envelope,
)
from .authcap_oracle import evaluate_policy
from .authcap_policy import PolicySet
from .authcap_types import PolicyRequestContext, SemanticProposal
from .authcap_witness import (
    AuthorizationWitness,
    AuthorizationWitnessError,
    proposal_digest,
    verify_witness_semantics,
)


class AuthCapVerificationError(ValueError):
    """capability 的密码完整性或策略推导语义验证失败。"""


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AuthCapVerificationError("验证时间必须包含时区")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class VerifiedAuthCap:
    capability_id: str
    subject_id: str
    executable_authority: AtomizedAuthority
    decision_id: str
    policy_hash: str
    policy_epoch: int
    single_use: bool
    workflow_id: str | None
    workflow_step: int | None
    previous_workflow_state_digest: str | None


def verify_authcap(
    *,
    token: str,
    witness: AuthorizationWitness,
    executable_authority: AtomizedAuthority,
    proposal: SemanticProposal,
    trusted_context: PolicyRequestContext,
    trusted_request: TrustedRequestEnvelope,
    policy_set: PolicySet,
    resource_tenants: dict[str, str],
    public_keys: AuthCapPublicKeyRing,
    verification_time: str,
    maximum_capability_bytes: int,
    expected_workflow_id: str | None = None,
    expected_workflow_step: int | None = None,
    expected_previous_workflow_state_digest: str | None = None,
) -> VerifiedAuthCap:
    """验证签名并重新计算 policy derivation 与 non-amplification。"""

    try:
        payload = public_keys.verify_signature(
            token,
            maximum_bytes=maximum_capability_bytes,
        )
        validate_trusted_request_envelope(trusted_context, trusted_request)
    except (AuthCapTokenError, ValueError) as exc:
        raise AuthCapVerificationError("capability 结构、签名或 request 无效") from exc
    now = _utc(verification_time)
    if not (_utc(payload.issued_at) <= now < _utc(payload.expires_at)):
        raise AuthCapVerificationError("capability 不在有效期")
    if not payload.single_use:
        raise AuthCapVerificationError("F2 executable capability 必须 single-use")
    if (
        payload.workflow_id != expected_workflow_id
        or payload.workflow_step != expected_workflow_step
        or payload.previous_workflow_state_digest
        != expected_previous_workflow_state_digest
    ):
        raise AuthCapVerificationError("capability workflow state binding 不一致")
    if (
        payload.subject_id != trusted_context.principal.subject_id
        or payload.policy_epoch != trusted_context.policy_epoch
        or payload.trusted_request_digest != trusted_request.canonical_digest
        or payload.proposal_digest != proposal_digest(proposal)
        or payload.executable_authority_digest
        != executable_authority.canonical_digest
        or payload.authorization_witness_digest != witness.canonical_digest
    ):
        raise AuthCapVerificationError("capability payload 与当前语义输入不一致")
    grant = evaluate_policy(trusted_context, policy_set)
    if (
        grant.decision != "allow"
        or payload.policy_hash != grant.policy_hash
        or payload.decision_id != grant.decision_id
    ):
        raise AuthCapVerificationError("当前 policy oracle 不支持 capability")
    grant_authority = authority_set_to_atomized(grant.maximum_authority)
    try:
        proposal_authority = attach_mandatory_grant_constraints(
            normalize_proposal_authority(
                proposal,
                context=trusted_context,
                resource_tenants=resource_tenants,
            ),
            grant_authority,
        )
        verify_witness_semantics(
            witness,
            context=trusted_context,
            grant=grant,
            trusted_request_digest=trusted_request.canonical_digest,
            proposal=proposal,
            executable_authority=executable_authority,
        )
    except (AuthorizationWitnessError, ValueError) as exc:
        raise AuthCapVerificationError("authorization witness 语义无效") from exc
    invariant = verify_authority_non_amplification(
        proposal_authority,
        trusted_request.requested_authority,
        grant_authority,
        executable_authority,
    )
    if not invariant.passed:
        raise AuthCapVerificationError("executable authority 发生放大")
    return VerifiedAuthCap(
        capability_id=payload.capability_id,
        subject_id=payload.subject_id,
        executable_authority=executable_authority,
        decision_id=payload.decision_id,
        policy_hash=payload.policy_hash,
        policy_epoch=payload.policy_epoch,
        single_use=payload.single_use,
        workflow_id=payload.workflow_id,
        workflow_step=payload.workflow_step,
        previous_workflow_state_digest=payload.previous_workflow_state_digest,
    )
