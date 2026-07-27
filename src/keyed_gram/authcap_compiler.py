"""Intent–Authority Separation 三态 compiler。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from .authcap_atoms import (
    AtomizedAuthority,
    AuthorityAtom,
    CanonicalConstraint,
    authority_set_to_atomized,
    verify_authority_non_amplification,
)
from .authcap_capability import AuthCapIssuer, AuthCapV1
from .authcap_oracle import evaluate_policy
from .authcap_policy import PolicySet, canonical_sha256
from .authcap_types import (
    Action,
    PolicyGrant,
    PolicyRequestContext,
    SemanticProposal,
)
from .authcap_witness import (
    AuthorizationWitness,
    build_f2_authorization_witness,
    environment_digest,
    principal_attributes_digest,
    proposal_digest,
)


class AuthCapCompilerError(ValueError):
    """可信 request envelope 或 compiler 输入不满足 F2 不变量。"""


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AuthCapCompilerError("编译时间必须包含时区")
    return parsed.astimezone(timezone.utc)


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrustedRequestEnvelope:
    principal_digest: str
    requested_authority: AtomizedAuthority
    environment_digest: str
    request_id: str
    policy_epoch: int
    request_nonce: str

    def __post_init__(self) -> None:
        if self.requested_authority.is_empty:
            raise AuthCapCompilerError("trusted request authority 不能为空")
        if self.policy_epoch < 1:
            raise AuthCapCompilerError("policy_epoch 必须为正整数")
        for name in ("principal_digest", "environment_digest"):
            value = getattr(self, name)
            if len(value) != 64:
                raise AuthCapCompilerError(f"{name} 必须是 SHA-256")
        if not self.request_id or not self.request_nonce:
            raise AuthCapCompilerError("request_id/request_nonce 不能为空")

    def canonical(self) -> dict[str, object]:
        return {
            "principal_digest": self.principal_digest,
            "requested_authority": self.requested_authority.canonical(),
            "environment_digest": self.environment_digest,
            "request_id": self.request_id,
            "policy_epoch": self.policy_epoch,
            "request_nonce": self.request_nonce,
        }

    @property
    def canonical_digest(self) -> str:
        return canonical_sha256(self.canonical())


def trusted_request_envelope(
    context: PolicyRequestContext,
    *,
    request_id: str,
    request_nonce: str,
) -> TrustedRequestEnvelope:
    purpose = context.purpose if context.purpose is not None else "purpose:none"
    authority = AtomizedAuthority(
        frozenset(
            {
                AuthorityAtom(
                    subject_id=context.principal.subject_id,
                    tenant_id=context.requested_resource.tenant_id,
                    resource_id=context.requested_resource.resource_id,
                    action=context.requested_action.value,
                    relation_id=context.requested_resource.relation_id,
                    purpose=purpose,
                    constraints=(
                        CanonicalConstraint(
                            "policy_epoch",
                            "equals",
                            str(context.policy_epoch),
                        ),
                    ),
                )
            }
        )
    )
    return TrustedRequestEnvelope(
        principal_digest=principal_attributes_digest(context),
        requested_authority=authority,
        environment_digest=environment_digest(context),
        request_id=request_id,
        policy_epoch=context.policy_epoch,
        request_nonce=request_nonce,
    )


def validate_trusted_request_envelope(
    context: PolicyRequestContext,
    envelope: TrustedRequestEnvelope,
) -> None:
    expected = trusted_request_envelope(
        context,
        request_id=envelope.request_id,
        request_nonce=envelope.request_nonce,
    )
    if envelope != expected:
        raise AuthCapCompilerError("trusted request envelope 与可信 context 不一致")


def normalize_proposal_authority(
    proposal: SemanticProposal,
    *,
    context: PolicyRequestContext,
    resource_tenants: dict[str, str],
) -> AtomizedAuthority:
    """将不可信 proposal 展开成显式 atoms。

    principal、可信环境、policy epoch 和 issuer 信息均来自可信输入。proposal
    的自由文本 ``proposed_constraints`` 不进入 executable authority。
    """

    try:
        actions = tuple(Action(item).value for item in proposal.proposed_actions)
    except ValueError as exc:
        raise AuthCapCompilerError("proposal 包含未知 action") from exc
    if not proposal.proposed_resource_ids or not actions:
        return AtomizedAuthority.empty()
    if proposal.proposed_purpose is None:
        return AtomizedAuthority.empty()
    atoms = set()
    for resource_id in proposal.proposed_resource_ids:
        if resource_id not in resource_tenants:
            raise AuthCapCompilerError("proposal 包含未知 resource")
        for action in actions:
            atoms.add(
                AuthorityAtom(
                    subject_id=context.principal.subject_id,
                    tenant_id=resource_tenants[resource_id],
                    resource_id=resource_id,
                    action=action,
                    relation_id=context.requested_resource.relation_id,
                    purpose=proposal.proposed_purpose,
                )
            )
    return AtomizedAuthority(frozenset(atoms))


def attach_mandatory_grant_constraints(
    proposal: AtomizedAuthority,
    grant: AtomizedAuthority,
) -> AtomizedAuthority:
    """对匹配 scope 附加 policy 强制约束；proposal 无法删除这些约束。"""

    values = set()
    for atom in proposal.atoms:
        matching = [upper for upper in grant.atoms if upper.scope_key == atom.scope_key]
        if not matching:
            values.add(atom)
            continue
        constraints = tuple(
            sorted(set(atom.constraints) | set(matching[0].constraints))
        )
        values.add(AuthorityAtom(*atom.scope_key, constraints=constraints))
    return AtomizedAuthority(frozenset(values))


@dataclass(frozen=True)
class CompiledAllow:
    status: Literal["allow"]
    executable_authority: AtomizedAuthority
    authorization_witness: AuthorizationWitness
    capability_payload: AuthCapV1
    capability_token: str
    decision_reason_code: str


@dataclass(frozen=True)
class NeedsExplicitConfirmation:
    status: Literal["needs_explicit_confirmation"]
    decision_reason_code: str
    safe_candidate_authority_digest: str | None
    proposal_digest: str
    trusted_request_digest: str

    @property
    def capability_token(self) -> None:
        return None


@dataclass(frozen=True)
class CompiledDeny:
    decision_reason_code: str
    status: Literal["deny"] = "deny"

    @property
    def capability_token(self) -> None:
        return None


CompiledResult = CompiledAllow | NeedsExplicitConfirmation | CompiledDeny


@dataclass(frozen=True)
class CompilerParameters:
    capability_ttl_seconds: int = 300
    maximum_capability_bytes: int = 8192

    def __post_init__(self) -> None:
        if self.capability_ttl_seconds < 1:
            raise AuthCapCompilerError("capability TTL 必须为正数")
        if self.maximum_capability_bytes < 1024:
            raise AuthCapCompilerError("maximum capability bytes 过小")


class AuthCapCompiler:
    """内部复算 oracle，并只为精确安全 proposal 签发 capability。"""

    def __init__(
        self,
        issuer: AuthCapIssuer,
        parameters: CompilerParameters,
    ) -> None:
        self._issuer = issuer
        self.parameters = parameters

    def compile(
        self,
        *,
        proposal: SemanticProposal,
        trusted_context: PolicyRequestContext,
        trusted_request: TrustedRequestEnvelope,
        policy_set: PolicySet,
        resource_tenants: dict[str, str],
        issued_at: str,
        workflow_id: str | None = None,
        workflow_step: int | None = None,
        previous_workflow_state_digest: str | None = None,
    ) -> CompiledResult:
        """三态编译；调用者不能传入或替换 ``PolicyGrant``。"""

        try:
            validate_trusted_request_envelope(trusted_context, trusted_request)
        except (AuthCapCompilerError, ValueError):
            return CompiledDeny("invalid_trusted_request_envelope")
        grant = evaluate_policy(trusted_context, policy_set)
        if grant.decision != "allow":
            return CompiledDeny(grant.decision_reason_code)
        try:
            grant_authority = authority_set_to_atomized(grant.maximum_authority)
            raw_proposal = normalize_proposal_authority(
                proposal,
                context=trusted_context,
                resource_tenants=resource_tenants,
            )
        except (AuthCapCompilerError, ValueError):
            return CompiledDeny("invalid_proposal_authority")
        if raw_proposal.is_empty:
            return CompiledDeny("empty_proposal_authority")
        proposal_authority = attach_mandatory_grant_constraints(
            raw_proposal,
            grant_authority,
        )
        safe = (
            proposal_authority
            .intersection(trusted_request.requested_authority)
            .intersection(grant_authority)
        )
        proposal_within_request = proposal_authority.is_subset_of(
            trusted_request.requested_authority
        )
        proposal_within_grant = proposal_authority.is_subset_of(grant_authority)
        if not proposal_within_request or not proposal_within_grant:
            if safe.is_empty:
                return CompiledDeny("empty_safe_intersection")
            return NeedsExplicitConfirmation(
                status="needs_explicit_confirmation",
                decision_reason_code="proposal_scope_exceeds_trusted_authority",
                safe_candidate_authority_digest=safe.canonical_digest,
                proposal_digest=proposal_digest(proposal),
                trusted_request_digest=trusted_request.canonical_digest,
            )
        if len(proposal_authority.atoms) != 1:
            return NeedsExplicitConfirmation(
                status="needs_explicit_confirmation",
                decision_reason_code="multiple_mutually_exclusive_authority_atoms",
                safe_candidate_authority_digest=safe.canonical_digest,
                proposal_digest=proposal_digest(proposal),
                trusted_request_digest=trusted_request.canonical_digest,
            )
        executable = safe
        invariant = verify_authority_non_amplification(
            proposal_authority,
            trusted_request.requested_authority,
            grant_authority,
            executable,
        )
        if executable.is_empty or not invariant.passed:
            return CompiledDeny(invariant.reason_code)
        return self._issue(
            proposal=proposal,
            context=trusted_context,
            request=trusted_request,
            grant=grant,
            executable=executable,
            issued_at=issued_at,
            workflow_id=workflow_id,
            workflow_step=workflow_step,
            previous_workflow_state_digest=previous_workflow_state_digest,
        )

    def _issue(
        self,
        *,
        proposal: SemanticProposal,
        context: PolicyRequestContext,
        request: TrustedRequestEnvelope,
        grant: PolicyGrant,
        executable: AtomizedAuthority,
        issued_at: str,
        workflow_id: str | None,
        workflow_step: int | None,
        previous_workflow_state_digest: str | None,
    ) -> CompiledAllow:
        reason = "compiled_exact_authority"
        witness = build_f2_authorization_witness(
            context=context,
            grant=grant,
            trusted_request_digest=request.canonical_digest,
            proposal=proposal,
            executable_authority=executable,
            decision_reason_code=reason,
        )
        issued = _utc(issued_at)
        material = {
            "request": request.canonical_digest,
            "proposal": proposal_digest(proposal),
            "authority": executable.canonical_digest,
            "workflow_id": workflow_id,
            "workflow_step": workflow_step,
        }
        identifier = canonical_sha256(material)
        payload = AuthCapV1(
            capability_id=f"cap-{identifier[:48]}",
            issuer_key_id=self._issuer.key_id,
            subject_id=context.principal.subject_id,
            policy_hash=grant.policy_hash,
            decision_id=grant.decision_id,
            authorization_witness_digest=witness.canonical_digest,
            trusted_request_digest=request.canonical_digest,
            proposal_digest=proposal_digest(proposal),
            executable_authority_digest=executable.canonical_digest,
            policy_epoch=context.policy_epoch,
            issued_at=_time(issued),
            expires_at=_time(
                issued + timedelta(seconds=self.parameters.capability_ttl_seconds)
            ),
            single_use=True,
            nonce=f"nonce-{hashlib.sha256(identifier.encode()).hexdigest()[:48]}",
            workflow_id=workflow_id,
            workflow_step=workflow_step,
            previous_workflow_state_digest=previous_workflow_state_digest,
        )
        token = self._issuer.issue(
            payload,
            maximum_bytes=self.parameters.maximum_capability_bytes,
        )
        return CompiledAllow(
            status="allow",
            executable_authority=executable,
            authorization_witness=witness,
            capability_payload=payload,
            capability_token=token,
            decision_reason_code=reason,
        )
