"""F2 authorization witness 的确定性表示与语义绑定。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .authcap_atoms import AtomizedAuthority
from .authcap_policy import canonical_json_bytes, canonical_sha256
from .authcap_types import PolicyGrant, PolicyRequestContext, SemanticProposal


class AuthorizationWitnessError(ValueError):
    """authorization witness 结构或语义不一致。"""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuthorizationWitnessError(f"{label} 必须是小写 SHA-256")
    return value


@dataclass(frozen=True)
class AuthorizationWitness:
    witness_version: int
    decision_id: str
    policy_hash: str
    principal_attributes_digest: str
    trusted_request_digest: str
    proposal_digest: str
    policy_grant_digest: str
    executable_authority_digest: str
    environment_digest: str
    policy_epoch: int
    matched_policy_ids: tuple[str, ...]
    decision_reason_code: str

    def __post_init__(self) -> None:
        if self.witness_version != 1:
            raise AuthorizationWitnessError("未知 witness_version")
        for name in (
            "decision_id",
            "policy_hash",
            "principal_attributes_digest",
            "trusted_request_digest",
            "proposal_digest",
            "policy_grant_digest",
            "executable_authority_digest",
            "environment_digest",
        ):
            _sha256(getattr(self, name), name)
        if self.policy_epoch < 1:
            raise AuthorizationWitnessError("policy_epoch 必须为正整数")
        if tuple(sorted(set(self.matched_policy_ids))) != self.matched_policy_ids:
            raise AuthorizationWitnessError("matched_policy_ids 必须唯一且有序")
        if not self.decision_reason_code:
            raise AuthorizationWitnessError("decision_reason_code 不能为空")

    def canonical(self) -> dict[str, Any]:
        return {
            "witness_version": self.witness_version,
            "decision_id": self.decision_id,
            "policy_hash": self.policy_hash,
            "principal_attributes_digest": self.principal_attributes_digest,
            "trusted_request_digest": self.trusted_request_digest,
            "proposal_digest": self.proposal_digest,
            "policy_grant_digest": self.policy_grant_digest,
            "executable_authority_digest": self.executable_authority_digest,
            "environment_digest": self.environment_digest,
            "policy_epoch": self.policy_epoch,
            "matched_policy_ids": list(self.matched_policy_ids),
            "decision_reason_code": self.decision_reason_code,
        }

    @property
    def canonical_digest(self) -> str:
        return _digest(self.canonical())


def proposal_digest(proposal: SemanticProposal) -> str:
    return canonical_sha256(proposal.canonical())


def principal_attributes_digest(context: PolicyRequestContext) -> str:
    return canonical_sha256(context.principal.canonical())


def environment_digest(context: PolicyRequestContext) -> str:
    return canonical_sha256(context.environment.canonical())


def policy_grant_digest(grant: PolicyGrant) -> str:
    return canonical_sha256(grant.canonical())


def build_f2_authorization_witness(
    *,
    context: PolicyRequestContext,
    grant: PolicyGrant,
    trusted_request_digest: str,
    proposal: SemanticProposal,
    executable_authority: AtomizedAuthority,
    decision_reason_code: str,
) -> AuthorizationWitness:
    """由内部复算 grant 构造 F2 witness。"""

    return AuthorizationWitness(
        witness_version=1,
        decision_id=grant.decision_id,
        policy_hash=grant.policy_hash,
        principal_attributes_digest=principal_attributes_digest(context),
        trusted_request_digest=_sha256(
            trusted_request_digest,
            "trusted_request_digest",
        ),
        proposal_digest=proposal_digest(proposal),
        policy_grant_digest=policy_grant_digest(grant),
        executable_authority_digest=executable_authority.canonical_digest,
        environment_digest=environment_digest(context),
        policy_epoch=context.policy_epoch,
        matched_policy_ids=tuple(sorted(grant.matched_policy_ids)),
        decision_reason_code=decision_reason_code,
    )


def verify_witness_semantics(
    witness: AuthorizationWitness,
    *,
    context: PolicyRequestContext,
    grant: PolicyGrant,
    trusted_request_digest: str,
    proposal: SemanticProposal,
    executable_authority: AtomizedAuthority,
) -> None:
    """验证 witness 绑定当前 oracle decision，而不只验证签名。"""

    expected = build_f2_authorization_witness(
        context=context,
        grant=grant,
        trusted_request_digest=trusted_request_digest,
        proposal=proposal,
        executable_authority=executable_authority,
        decision_reason_code=witness.decision_reason_code,
    )
    if witness != expected:
        raise AuthorizationWitnessError("authorization witness 语义不一致")
