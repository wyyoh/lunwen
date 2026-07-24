"""Stage D1 的类型化 capability claims、request 与拒绝原因。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .stage_c24_contract import RelationId


TOKEN_SCHEMA_VERSION = 1
TOKEN_TYPE = "KG-CAP"
TOKEN_ALGORITHM = "HS256"
RETRIEVE_PERMISSION = "retrieve"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")
_AUTHORIZATION_SEAL = object()


class CapabilityError(ValueError):
    """不向 memory 放行的 capability 错误。"""

    def __init__(self, reason: "RejectionReason", message: str):
        super().__init__(message)
        self.reason = reason


class RejectionReason(str, Enum):
    INVALID_REQUEST = "invalid_request"
    INVALID_TOKEN = "invalid_token"
    UNKNOWN_KEY = "unknown_key"
    REVOKED_KEY = "revoked_key"
    BAD_SIGNATURE = "bad_signature"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    POLICY_VERSION_MISMATCH = "policy_version_mismatch"
    POLICY_DENIED = "policy_denied"
    REVOKED_TOKEN = "revoked_token"
    REPLAY = "replay"
    SUBJECT_MISMATCH = "subject_mismatch"
    ENTITY_SCOPE_MISMATCH = "entity_scope_mismatch"
    RELATION_MISMATCH = "relation_mismatch"
    PERMISSION_DENIED = "permission_denied"
    UNAUTHORIZED_GRANT = "unauthorized_grant"
    ROUTER_PROPOSAL_NOT_AUTHORIZED = "router_proposal_not_authorized"


def validate_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CapabilityError(
            RejectionReason.INVALID_REQUEST,
            f"{label} 必须是 1–128 字节的显式标识符",
        )
    return value


@dataclass(frozen=True, slots=True)
class SuggestedRelation:
    """不可信语义解析器的建议；该类型永远不是授权请求。"""

    relation_id: RelationId
    source: str = "untrusted_semantic_router"

    def __post_init__(self) -> None:
        if not isinstance(self.relation_id, RelationId):
            raise TypeError("SuggestedRelation 只能包含已知 RelationId")
        validate_identifier(self.source, label="proposal source")


@dataclass(frozen=True, slots=True)
class AuthorizedMemoryRequest:
    """进入可信 capability gateway 的唯一外部 request 类型。"""

    subject_id: str
    entity_id: str
    relation_id: RelationId
    capability_token: bytes

    def __post_init__(self) -> None:
        validate_identifier(self.subject_id, label="subject_id")
        validate_identifier(self.entity_id, label="entity_id")
        if not isinstance(self.relation_id, RelationId):
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "relation_id 必须是冻结 RelationId",
            )
        if not isinstance(self.capability_token, bytes):
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "capability_token 必须是 bytes",
            )


@dataclass(frozen=True, slots=True)
class CapabilityGrant:
    """可信 policy 层允许 authority 签发的显式 scope。"""

    subject_id: str
    entity_scope: tuple[str, ...]
    relation_id: RelationId
    permissions: tuple[str, ...] = (RETRIEVE_PERMISSION,)

    def __post_init__(self) -> None:
        validate_identifier(self.subject_id, label="grant subject_id")
        if (
            not isinstance(self.entity_scope, tuple)
            or not self.entity_scope
            or tuple(sorted(set(self.entity_scope))) != self.entity_scope
        ):
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "entity_scope 必须是非空、排序且唯一的显式 entity ID tuple",
            )
        for entity_id in self.entity_scope:
            validate_identifier(entity_id, label="grant entity_scope")
            if entity_id == "*":
                raise CapabilityError(
                    RejectionReason.INVALID_REQUEST,
                    "D1 禁止 wildcard entity scope",
                )
        if not isinstance(self.relation_id, RelationId):
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "grant relation_id 必须是冻结 RelationId",
            )
        if (
            not isinstance(self.permissions, tuple)
            or not self.permissions
            or tuple(sorted(set(self.permissions))) != self.permissions
        ):
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "permissions 必须是非空、排序且唯一的 tuple",
            )
        for permission in self.permissions:
            validate_identifier(permission, label="permission")


@dataclass(frozen=True, slots=True)
class CapabilityClaims:
    """经 HMAC 认证但不加密的 capability payload。"""

    subject_id: str
    entity_scope: tuple[str, ...]
    relation_id: RelationId
    permissions: tuple[str, ...]
    issued_at: int
    expires_at: int
    key_id: str
    nonce: str
    policy_version: str
    schema_version: int = TOKEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        CapabilityGrant(
            self.subject_id,
            self.entity_scope,
            self.relation_id,
            self.permissions,
        )
        if type(self.issued_at) is not int or type(self.expires_at) is not int:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "issued_at/expires_at 必须是整数 epoch seconds",
            )
        if self.issued_at < 0 or self.expires_at <= self.issued_at:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "capability 时间窗口非法",
            )
        validate_identifier(self.key_id, label="key_id")
        validate_identifier(self.nonce, label="nonce")
        if len(self.nonce) < 22:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "nonce 熵表示过短",
            )
        validate_identifier(self.policy_version, label="policy_version")
        if self.schema_version != TOKEN_SCHEMA_VERSION:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "capability schema version 非法",
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "entity_scope": list(self.entity_scope),
            "relation_id": self.relation_id.value,
            "permissions": list(self.permissions),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "key_id": self.key_id,
            "nonce": self.nonce,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "CapabilityClaims":
        expected = {
            "schema_version",
            "subject_id",
            "entity_scope",
            "relation_id",
            "permissions",
            "issued_at",
            "expires_at",
            "key_id",
            "nonce",
            "policy_version",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "capability payload schema 非法",
            )
        scope = value["entity_scope"]
        permissions = value["permissions"]
        if not isinstance(scope, list) or not isinstance(permissions, list):
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "scope/permissions JSON 类型非法",
            )
        try:
            relation = RelationId(value["relation_id"])
        except (TypeError, ValueError) as exc:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "capability 包含未知 RelationId",
            ) from exc
        try:
            return cls(
                subject_id=value["subject_id"],
                entity_scope=tuple(scope),
                relation_id=relation,
                permissions=tuple(permissions),
                issued_at=value["issued_at"],
                expires_at=value["expires_at"],
                key_id=value["key_id"],
                nonce=value["nonce"],
                policy_version=value["policy_version"],
                schema_version=value["schema_version"],
            )
        except CapabilityError as exc:
            if exc.reason is RejectionReason.INVALID_REQUEST:
                raise CapabilityError(
                    RejectionReason.INVALID_TOKEN, str(exc)
                ) from exc
            raise


@dataclass(frozen=True, slots=True)
class IssuedCapability:
    token: bytes = field(repr=False)
    claims: CapabilityClaims

    def __post_init__(self) -> None:
        if not isinstance(self.token, bytes) or not self.token:
            raise ValueError("issued capability token 必须是非空 bytes")


@dataclass(frozen=True, slots=True)
class VerifiedCapability:
    """仅 verifier 可创建、仅 memory probe 接受的内部授权证明。"""

    subject_id: str
    entity_id: str
    relation_id: RelationId
    permission: str
    nonce: str
    expires_at: int
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _AUTHORIZATION_SEAL:
            raise TypeError("VerifiedCapability 不能由外部构造")


def create_verified_capability(
    *,
    subject_id: str,
    entity_id: str,
    relation_id: RelationId,
    permission: str,
    nonce: str,
    expires_at: int,
) -> VerifiedCapability:
    """仅供可信 verifier 调用的内部 factory。"""

    return VerifiedCapability(
        subject_id=subject_id,
        entity_id=entity_id,
        relation_id=relation_id,
        permission=permission,
        nonce=nonce,
        expires_at=expires_at,
        _seal=_AUTHORIZATION_SEAL,
    )


def assert_verified_capability(value: Any) -> VerifiedCapability:
    if not isinstance(value, VerifiedCapability) or value._seal is not _AUTHORIZATION_SEAL:
        raise TypeError("memory 只接受 verifier 产生的 VerifiedCapability")
    return value
