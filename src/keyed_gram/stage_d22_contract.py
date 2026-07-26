"""Stage D2.2 的隔离服务、内部 release ticket 与安全响应契约。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .stage_c24_contract import RelationId
from .stage_d1_contract import RETRIEVE_PERMISSION, validate_identifier

D22_SCHEMA_VERSION = 1
D22_TICKET_TYPE = "KG-D22-RELEASE"
D22_TICKET_ALGORITHM = "HMAC-SHA-256"


class ServiceRole(str, Enum):
    GATEWAY = "gateway"
    MEMORY = "memory"
    GENERATOR = "generator"


class D22RejectionReason(str, Enum):
    INVALID_MESSAGE = "invalid_message"
    SERVICE_AUTHENTICATION_FAILED = "service_authentication_failed"
    INVALID_SERVICE_IDENTITY = "invalid_service_identity"
    INVALID_SESSION = "invalid_session"
    INVALID_TICKET = "invalid_ticket"
    BAD_TICKET_SIGNATURE = "bad_ticket_signature"
    EXPIRED_TICKET = "expired_ticket"
    REPLAYED_TICKET = "replayed_ticket"
    OUT_OF_ORDER_TICKET = "out_of_order_ticket"
    RECORD_NOT_FOUND = "record_not_found"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    GENERATOR_FAILURE = "generator_failure"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    PROCESS_CRASHED = "process_crashed"
    IPC_TIMEOUT = "ipc_timeout"


class D22Error(RuntimeError):
    """只携带固定 reason 和不含 secret/plaintext 的消息。"""

    def __init__(self, reason: D22RejectionReason, safe_message: str):
        super().__init__(safe_message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ReleaseTicketClaims:
    gateway_id: str
    ticket_id: str
    request_nonce: str
    subject_id: str
    entity_id: str
    relation_id: RelationId
    permission: str
    instruction_id: str
    sequence: int
    issued_at: int
    expires_at: int
    schema_version: int = D22_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("gateway_id", self.gateway_id),
            ("ticket_id", self.ticket_id),
            ("request_nonce", self.request_nonce),
            ("subject_id", self.subject_id),
            ("entity_id", self.entity_id),
            ("instruction_id", self.instruction_id),
        ):
            validate_identifier(value, label=label)
        if len(self.ticket_id) < 22 or len(self.request_nonce) < 22:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket/request nonce 熵表示过短",
            )
        if not isinstance(self.relation_id, RelationId):
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket relation 必须是冻结 RelationId",
            )
        if self.permission != RETRIEVE_PERMISSION:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket permission 非 retrieve",
            )
        if type(self.sequence) is not int or self.sequence <= 0:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket sequence 必须为正整数",
            )
        if (
            type(self.issued_at) is not int
            or type(self.expires_at) is not int
            or self.issued_at < 0
            or self.expires_at <= self.issued_at
        ):
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket 时间窗口非法",
            )
        if self.schema_version != D22_SCHEMA_VERSION:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket schema version 非法",
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "expires_at": self.expires_at,
            "gateway_id": self.gateway_id,
            "instruction_id": self.instruction_id,
            "issued_at": self.issued_at,
            "permission": self.permission,
            "relation_id": self.relation_id.value,
            "request_nonce": self.request_nonce,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "subject_id": self.subject_id,
            "ticket_id": self.ticket_id,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ReleaseTicketClaims:
        expected = {
            "entity_id",
            "expires_at",
            "gateway_id",
            "instruction_id",
            "issued_at",
            "permission",
            "relation_id",
            "request_nonce",
            "schema_version",
            "sequence",
            "subject_id",
            "ticket_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket payload schema 非法",
            )
        try:
            relation = RelationId(value["relation_id"])
        except (TypeError, ValueError) as exc:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket 包含未知 RelationId",
            ) from exc
        try:
            return cls(
                gateway_id=value["gateway_id"],
                ticket_id=value["ticket_id"],
                request_nonce=value["request_nonce"],
                subject_id=value["subject_id"],
                entity_id=value["entity_id"],
                relation_id=relation,
                permission=value["permission"],
                instruction_id=value["instruction_id"],
                sequence=value["sequence"],
                issued_at=value["issued_at"],
                expires_at=value["expires_at"],
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, D22Error):
                raise
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "ticket payload 字段类型非法",
            ) from exc


@dataclass(frozen=True, slots=True)
class IsolatedGenerationResult:
    text: str = field(repr=False)
    output_digest: str
    request_id: str
    exact_delivery: bool

    def __post_init__(self) -> None:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != (
            self.output_digest
        ):
            raise ValueError("isolated generation output digest 不一致")


def safe_error_response(reason: str) -> dict[str, Any]:
    return {
        "schema_version": D22_SCHEMA_VERSION,
        "status": "reject",
        "reason": reason,
    }
