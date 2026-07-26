"""Stage D2.3 的故障、可观测性与服务生命周期契约。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .stage_d1_contract import validate_identifier

D23_SCHEMA_VERSION = 1


class D23RejectionReason(str, Enum):
    INVALID_OBSERVABILITY_EVENT = "invalid_observability_event"
    SENSITIVE_OBSERVABILITY_VALUE = "sensitive_observability_value"
    INVALID_FRAME = "invalid_frame"
    MESSAGE_TOO_LARGE = "message_too_large"
    MESSAGE_TOO_DEEP = "message_too_deep"
    ENDPOINT_UNSAFE = "endpoint_unsafe"
    PEER_IDENTITY_MISMATCH = "peer_identity_mismatch"
    RATE_LIMITED = "rate_limited"
    STATE_BUSY = "state_busy"
    STATE_CORRUPT = "state_corrupt"
    STATE_ROLLBACK = "state_rollback"
    EPOCH_MISMATCH = "epoch_mismatch"
    CLOCK_ROLLBACK = "clock_rollback"
    REPLAY = "replay"
    REVOKED = "revoked"
    INVALID_TRANSITION = "invalid_transition"
    FAULT_INJECTED = "fault_injected"


class D23Error(RuntimeError):
    """只携带固定拒绝原因与不含敏感值的安全消息。"""

    def __init__(self, reason: D23RejectionReason, safe_message: str):
        super().__init__(safe_message)
        self.reason = reason


class LifecycleMilestone(str, Enum):
    CAPABILITY_CONSUMED = "capability_consumed"
    RECORD_DECRYPTED = "record_decrypted"
    GENERATOR_INVOKED = "generator_invoked"
    OUTPUT_COMMITTED = "output_committed"
    RESPONSE_DELIVERED = "response_delivered"


MILESTONE_ORDER = tuple(LifecycleMilestone)


@dataclass(frozen=True, slots=True)
class EpochSnapshot:
    state_epoch: int
    policy_epoch: int
    gateway_instance_epoch: int

    def __post_init__(self) -> None:
        for name, value in (
            ("state_epoch", self.state_epoch),
            ("policy_epoch", self.policy_epoch),
            ("gateway_instance_epoch", self.gateway_instance_epoch),
        ):
            if type(value) is not int or value <= 0:
                raise D23Error(
                    D23RejectionReason.EPOCH_MISMATCH,
                    f"{name} 必须为正整数",
                )

    def to_payload(self) -> dict[str, int]:
        return {
            "gateway_instance_epoch": self.gateway_instance_epoch,
            "policy_epoch": self.policy_epoch,
            "state_epoch": self.state_epoch,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> EpochSnapshot:
        if set(value) != {
            "gateway_instance_epoch",
            "policy_epoch",
            "state_epoch",
        }:
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "epoch payload schema 非法",
            )
        return cls(
            state_epoch=value["state_epoch"],
            policy_epoch=value["policy_epoch"],
            gateway_instance_epoch=value["gateway_instance_epoch"],
        )


@dataclass(frozen=True, slots=True)
class LifecycleTicket:
    ticket_id: str
    gateway_id: str
    epochs: EpochSnapshot
    issued_at: int
    expires_at: int
    schema_version: int = D23_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_identifier(self.ticket_id, label="ticket_id")
        validate_identifier(self.gateway_id, label="gateway_id")
        if len(self.ticket_id) < 22:
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "ticket_id 熵表示过短",
            )
        if (
            type(self.issued_at) is not int
            or type(self.expires_at) is not int
            or self.issued_at < 0
            or self.expires_at <= self.issued_at
        ):
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "ticket 时间窗口非法",
            )
        if self.schema_version != D23_SCHEMA_VERSION:
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "ticket schema version 非法",
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs.to_payload(),
            "expires_at": self.expires_at,
            "gateway_id": self.gateway_id,
            "issued_at": self.issued_at,
            "schema_version": self.schema_version,
            "ticket_id": self.ticket_id,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> LifecycleTicket:
        if set(value) != {
            "epochs",
            "expires_at",
            "gateway_id",
            "issued_at",
            "schema_version",
            "ticket_id",
        } or not isinstance(value.get("epochs"), Mapping):
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket schema 非法",
            )
        try:
            return cls(
                ticket_id=value["ticket_id"],
                gateway_id=value["gateway_id"],
                epochs=EpochSnapshot.from_payload(value["epochs"]),
                issued_at=value["issued_at"],
                expires_at=value["expires_at"],
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, D23Error):
                raise
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket 字段类型非法",
            ) from exc


def digest_identifier(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()
