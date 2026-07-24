"""D1 显式授权 policy、签发、revocation 与 replay state。"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import Sequence

from .stage_d1_contract import (
    CapabilityClaims,
    CapabilityError,
    CapabilityGrant,
    IssuedCapability,
    RETRIEVE_PERMISSION,
    RejectionReason,
)
from .stage_d1_token import CapabilityTokenCodec, HmacKeyRing


@dataclass(frozen=True, slots=True)
class PolicyRule:
    subject_id: str
    entity_scope: tuple[str, ...]
    relations: tuple[str, ...]
    permissions: tuple[str, ...] = (RETRIEVE_PERMISSION,)

    def __post_init__(self) -> None:
        from .stage_c24_contract import RelationId
        from .stage_d1_contract import validate_identifier

        validate_identifier(self.subject_id, label="policy subject_id")
        if (
            not self.entity_scope
            or tuple(sorted(set(self.entity_scope))) != self.entity_scope
            or "*" in self.entity_scope
        ):
            raise ValueError("policy entity scope 必须是排序唯一的显式 ID")
        for entity_id in self.entity_scope:
            validate_identifier(entity_id, label="policy entity_scope")
        if (
            not self.relations
            or tuple(sorted(set(self.relations))) != self.relations
        ):
            raise ValueError("policy relations 必须排序且唯一")
        for relation in self.relations:
            RelationId(relation)
        if (
            not self.permissions
            or tuple(sorted(set(self.permissions))) != self.permissions
        ):
            raise ValueError("policy permissions 必须排序且唯一")


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    version: str
    rules: tuple[PolicyRule, ...]
    max_ttl_seconds: int
    single_use: bool = True

    def __post_init__(self) -> None:
        from .stage_d1_contract import validate_identifier

        validate_identifier(self.version, label="policy version")
        if not self.rules:
            raise ValueError("policy 至少需要一条显式 rule")
        if self.max_ttl_seconds <= 0:
            raise ValueError("policy max TTL 必须为正")
        if self.single_use is not True:
            raise ValueError("D1 replay 测试要求 single_use=true")

    def allows(self, grant: CapabilityGrant) -> bool:
        requested_entities = set(grant.entity_scope)
        requested_permissions = set(grant.permissions)
        for rule in self.rules:
            if (
                rule.subject_id == grant.subject_id
                and grant.relation_id.value in rule.relations
                and requested_entities <= set(rule.entity_scope)
                and requested_permissions <= set(rule.permissions)
            ):
                return True
        return False


@dataclass(slots=True)
class RevocationReplayStore:
    """D1 单进程 replay/revocation store；不声称具备分布式持久性。"""

    _revoked_nonces: set[str] = field(default_factory=set, repr=False)
    _consumed_nonces: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def revoke(self, nonce: str) -> None:
        with self._lock:
            self._revoked_nonces.add(nonce)

    def consume(self, nonce: str) -> None:
        with self._lock:
            if nonce in self._revoked_nonces:
                raise CapabilityError(
                    RejectionReason.REVOKED_TOKEN,
                    "capability nonce 已撤销",
                )
            if nonce in self._consumed_nonces:
                raise CapabilityError(
                    RejectionReason.REPLAY,
                    "single-use capability 已消费",
                )
            self._consumed_nonces.add(nonce)

    def status(self, nonce: str) -> str:
        with self._lock:
            if nonce in self._revoked_nonces:
                return "revoked"
            if nonce in self._consumed_nonces:
                return "consumed"
            return "fresh"


@dataclass(slots=True)
class CapabilityAuthority:
    keyring: HmacKeyRing
    policy: CapabilityPolicy
    codec: CapabilityTokenCodec
    nonce_bytes: int = 32

    def __post_init__(self) -> None:
        if self.nonce_bytes < 16:
            raise ValueError("nonce 至少需要 16 random bytes")

    def issue(
        self,
        grant: CapabilityGrant,
        *,
        now: int,
        ttl_seconds: int,
        nonce: str | None = None,
    ) -> IssuedCapability:
        if not isinstance(grant, CapabilityGrant) or not self.policy.allows(grant):
            raise CapabilityError(
                RejectionReason.UNAUTHORIZED_GRANT,
                "current policy 不允许该 capability grant",
            )
        if (
            type(now) is not int
            or type(ttl_seconds) is not int
            or ttl_seconds <= 0
            or ttl_seconds > self.policy.max_ttl_seconds
        ):
            raise CapabilityError(
                RejectionReason.UNAUTHORIZED_GRANT,
                "capability TTL 超出 policy",
            )
        # token_hex 仍使用 CSPRNG，但其首字符恒为十六进制字符，满足冻结的
        # identifier grammar，避免 token_urlsafe 偶发以 "-" 或 "_" 开头。
        resolved_nonce = (
            secrets.token_hex(self.nonce_bytes) if nonce is None else nonce
        )
        claims = CapabilityClaims(
            subject_id=grant.subject_id,
            entity_scope=grant.entity_scope,
            relation_id=grant.relation_id,
            permissions=grant.permissions,
            issued_at=now,
            expires_at=now + ttl_seconds,
            key_id=self.keyring.active_key_id,
            nonce=resolved_nonce,
            policy_version=self.policy.version,
        )
        return IssuedCapability(
            token=self.codec.issue(claims, self.keyring),
            claims=claims,
        )
