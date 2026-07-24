"""D1 trusted gateway：验证 capability 后才调用单一 typed bucket probe。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .stage_c24_contract import RelationId
from .stage_d1_contract import (
    AuthorizedMemoryRequest,
    CapabilityError,
    CapabilityGrant,
    RETRIEVE_PERMISSION,
    RejectionReason,
    SuggestedRelation,
    VerifiedCapability,
    assert_verified_capability,
    create_verified_capability,
)
from .stage_d1_policy import CapabilityPolicy, RevocationReplayStore
from .stage_d1_token import CapabilityTokenCodec, HmacKeyRing


@dataclass(frozen=True, slots=True)
class CapabilityRetrievalResult:
    entity_id: str
    relation_id: RelationId
    memory_access_count: int
    cross_relation_access_count: int
    cross_entity_access_count: int
    public_probe_only: bool = True


class KeyedMemory(Protocol):
    def retrieve_authorized(
        self, capability: VerifiedCapability
    ) -> CapabilityRetrievalResult:
        ...


@dataclass(slots=True)
class CapabilityMemoryProbe:
    """只记录 typed access，不保存 value，更不保存 private answer。"""

    declared_buckets: frozenset[tuple[str, RelationId]]
    access_log: list[tuple[str, RelationId, str]] = field(default_factory=list)

    def retrieve_authorized(
        self, capability: VerifiedCapability
    ) -> CapabilityRetrievalResult:
        verified = assert_verified_capability(capability)
        key = (verified.entity_id, verified.relation_id)
        if key not in self.declared_buckets:
            raise LookupError("typed public probe bucket 不存在")
        self.access_log.append(
            (verified.entity_id, verified.relation_id, verified.nonce)
        )
        return CapabilityRetrievalResult(
            entity_id=verified.entity_id,
            relation_id=verified.relation_id,
            memory_access_count=1,
            cross_relation_access_count=0,
            cross_entity_access_count=0,
        )

    @property
    def memory_access_count(self) -> int:
        return len(self.access_log)


@dataclass(slots=True)
class CapabilityVerifier:
    keyring: HmacKeyRing
    policy: CapabilityPolicy
    codec: CapabilityTokenCodec
    replay_store: RevocationReplayStore

    def authorize(
        self, request: AuthorizedMemoryRequest, *, now: int
    ) -> VerifiedCapability:
        if type(request) is not AuthorizedMemoryRequest:
            reason = (
                RejectionReason.ROUTER_PROPOSAL_NOT_AUTHORIZED
                if isinstance(request, SuggestedRelation)
                else RejectionReason.INVALID_REQUEST
            )
            raise CapabilityError(
                reason,
                "gateway 只接受 AuthorizedMemoryRequest",
            )
        if type(now) is not int or now < 0:
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "trusted clock 必须返回非负整数 epoch seconds",
            )
        claims = self.codec.verify(request.capability_token, self.keyring)
        if now < claims.issued_at:
            raise CapabilityError(
                RejectionReason.NOT_YET_VALID,
                "capability 尚未生效",
            )
        if now >= claims.expires_at:
            raise CapabilityError(
                RejectionReason.EXPIRED,
                "capability 已过期",
            )
        if claims.policy_version != self.policy.version:
            raise CapabilityError(
                RejectionReason.POLICY_VERSION_MISMATCH,
                "capability policy version 不是 current version",
            )
        if request.subject_id != claims.subject_id:
            raise CapabilityError(
                RejectionReason.SUBJECT_MISMATCH,
                "request subject 与 capability 不匹配",
            )
        if request.relation_id is not claims.relation_id:
            raise CapabilityError(
                RejectionReason.RELATION_MISMATCH,
                "request relation 与 capability 不匹配",
            )
        if request.entity_id not in claims.entity_scope:
            raise CapabilityError(
                RejectionReason.ENTITY_SCOPE_MISMATCH,
                "request entity 不在 capability scope",
            )
        if RETRIEVE_PERMISSION not in claims.permissions:
            raise CapabilityError(
                RejectionReason.PERMISSION_DENIED,
                "capability 缺少 retrieve permission",
            )
        current_grant = CapabilityGrant(
            claims.subject_id,
            claims.entity_scope,
            claims.relation_id,
            claims.permissions,
        )
        if not self.policy.allows(current_grant):
            raise CapabilityError(
                RejectionReason.POLICY_DENIED,
                "current policy 已不允许 token scope",
            )
        self.replay_store.consume(claims.nonce)
        return create_verified_capability(
            subject_id=claims.subject_id,
            entity_id=request.entity_id,
            relation_id=claims.relation_id,
            permission=RETRIEVE_PERMISSION,
            nonce=claims.nonce,
            expires_at=claims.expires_at,
        )


@dataclass(slots=True)
class TrustedCapabilityGateway:
    verifier: CapabilityVerifier
    clock: Callable[[], int]

    def retrieve(
        self, request: AuthorizedMemoryRequest, memory: KeyedMemory
    ) -> CapabilityRetrievalResult:
        """语义 router 不在此 API；验证完成前绝不调用 memory。"""

        verified = self.verifier.authorize(request, now=self.clock())
        return memory.retrieve_authorized(verified)


def rejected_request_access_delta(
    gateway: TrustedCapabilityGateway,
    request: Any,
    memory: CapabilityMemoryProbe,
) -> tuple[RejectionReason, int]:
    before = memory.memory_access_count
    try:
        gateway.retrieve(request, memory)
    except CapabilityError as exc:
        return exc.reason, memory.memory_access_count - before
    raise AssertionError("预期 request 被拒绝，但 gateway 接受了")
