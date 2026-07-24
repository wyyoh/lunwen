"""Stage D2 的 principal、AEAD record、synthetic value 与拒绝类型。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .stage_c24_contract import RelationId


D2_SCHEMA_VERSION = 1
DATA_ALGORITHM = "AES-256-GCM"
AES_GCM_NONCE_BYTES = 12
AES_GCM_TAG_BYTES = 16
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")
_PRINCIPAL_SEAL = object()


class D2RejectionReason(str, Enum):
    INVALID_SESSION = "invalid_session"
    PRINCIPAL_MISMATCH = "principal_mismatch"
    FORGED_PRINCIPAL = "forged_principal"
    FORGED_VERIFIED_CAPABILITY = "forged_verified_capability"
    UNKNOWN_DATA_KEY = "unknown_data_key"
    REVOKED_DATA_KEY = "revoked_data_key"
    AUTHENTICATION_FAILED = "authentication_failed"
    NONCE_REUSE = "nonce_reuse"
    INVALID_RECORD = "invalid_record"
    DUPLICATE_RECORD_ID = "duplicate_record_id"
    DUPLICATE_BUCKET = "duplicate_bucket"
    RECORD_NOT_FOUND = "record_not_found"
    RECORD_VERSION_ROLLBACK = "record_version_rollback"


class D2Error(ValueError):
    """任何情况下都不得释放 plaintext 的 D2 错误。"""

    def __init__(self, reason: D2RejectionReason, message: str):
        super().__init__(message)
        self.reason = reason


def validate_d2_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise D2Error(
            D2RejectionReason.INVALID_RECORD,
            f"{label} 必须是 1–128 字符的显式标识符",
        )
    return value


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """只能由 trusted identity provider 产生的内部 principal。"""

    subject_id: str
    session_id: str
    authentication_context: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PRINCIPAL_SEAL:
            raise TypeError("AuthenticatedPrincipal 不能由调用者构造")
        validate_d2_identifier(self.subject_id, label="principal subject_id")
        validate_d2_identifier(self.session_id, label="principal session_id")
        validate_d2_identifier(
            self.authentication_context,
            label="principal authentication_context",
        )


def create_authenticated_principal(
    *,
    subject_id: str,
    session_id: str,
    authentication_context: str,
) -> AuthenticatedPrincipal:
    """仅供 trusted identity provider 使用的内部 factory。"""

    return AuthenticatedPrincipal(
        subject_id=subject_id,
        session_id=session_id,
        authentication_context=authentication_context,
        _seal=_PRINCIPAL_SEAL,
    )


def assert_authenticated_principal(value: Any) -> AuthenticatedPrincipal:
    if (
        not isinstance(value, AuthenticatedPrincipal)
        or value._seal is not _PRINCIPAL_SEAL
    ):
        raise D2Error(
            D2RejectionReason.FORGED_PRINCIPAL,
            "memory 只接受 trusted identity provider 产生的 principal",
        )
    return value


@dataclass(frozen=True, slots=True)
class PublicSyntheticValue:
    """运行期随机 synthetic bytes；不代表 private value。"""

    payload: bytes = field(repr=False)
    provenance: str = "runtime_random_synthetic"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.payload, bytes)
            or len(self.payload) < 16
            or len(self.payload) > 4096
        ):
            raise ValueError("synthetic value 必须是 16–4096 bytes")
        if self.provenance != "runtime_random_synthetic":
            raise ValueError("D2 只允许 runtime random synthetic value")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def canonical_record_aad(
    *,
    record_id: str,
    entity_id: str,
    relation_id: RelationId,
    data_key_id: str,
    record_version: int,
    schema_version: int = D2_SCHEMA_VERSION,
    algorithm: str = DATA_ALGORITHM,
) -> bytes:
    payload = {
        "algorithm": algorithm,
        "data_key_id": data_key_id,
        "entity_id": entity_id,
        "record_id": record_id,
        "record_version": record_version,
        "relation_id": relation_id.value,
        "schema_version": schema_version,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EncryptedMemoryRecord:
    record_id: str
    entity_id: str
    relation_id: RelationId
    data_key_id: str
    record_version: int
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)
    schema_version: int = D2_SCHEMA_VERSION
    algorithm: str = DATA_ALGORITHM

    def __post_init__(self) -> None:
        validate_d2_identifier(self.record_id, label="record_id")
        validate_d2_identifier(self.entity_id, label="entity_id")
        validate_d2_identifier(self.data_key_id, label="data_key_id")
        if not isinstance(self.relation_id, RelationId):
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "record relation_id 必须是冻结 RelationId",
            )
        if type(self.record_version) is not int or self.record_version <= 0:
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "record_version 必须是正整数",
            )
        if (
            self.schema_version != D2_SCHEMA_VERSION
            or self.algorithm != DATA_ALGORITHM
        ):
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "拒绝 record schema/algorithm confusion",
            )
        if not isinstance(self.nonce, bytes) or len(self.nonce) != AES_GCM_NONCE_BYTES:
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "AES-GCM nonce 必须是 12 bytes",
            )
        if (
            not isinstance(self.ciphertext, bytes)
            or len(self.ciphertext) < AES_GCM_TAG_BYTES
            or len(self.ciphertext) > 8192
        ):
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "ciphertext/tag 长度非法",
            )

    def aad(self) -> bytes:
        return canonical_record_aad(
            record_id=self.record_id,
            entity_id=self.entity_id,
            relation_id=self.relation_id,
            data_key_id=self.data_key_id,
            record_version=self.record_version,
            schema_version=self.schema_version,
            algorithm=self.algorithm,
        )


@dataclass(frozen=True, slots=True)
class PlaintextRelease:
    """只向已授权调用方返回；审计产物不得序列化 value。"""

    record_id: str
    entity_id: str
    relation_id: RelationId
    value: PublicSyntheticValue = field(repr=False)
    value_digest: str
    value_released: bool = True

    def __post_init__(self) -> None:
        if self.value_digest != self.value.digest:
            raise ValueError("release digest 与 synthetic value 不一致")
