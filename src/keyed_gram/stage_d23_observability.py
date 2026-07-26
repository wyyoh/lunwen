"""D2.3 安全可观测性投影与高敏感字段阻断。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stage_d23_contract import (
    D23Error,
    D23RejectionReason,
)

OBSERVABILITY_CHANNELS = {
    "structured_log",
    "metric",
    "trace",
    "error_report",
    "retry_queue",
    "profile",
    "supervisor",
    "health",
    "debug",
    "audit",
}
SAFE_EVENT_FIELDS = {
    "channel",
    "event",
    "result",
    "reason",
    "role",
    "stage",
    "counter",
    "duration_bucket",
    "schema_version",
}
SAFE_METRIC_LABELS = {"result", "reason", "role", "stage"}
SENSITIVE_NAME_PARTS = {
    "plaintext",
    "value",
    "token",
    "key",
    "credential",
    "session",
    "capability",
    "prompt",
    "baggage",
    "subject_id",
    "entity_id",
}


def _strict_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D23Error(
            D23RejectionReason.INVALID_OBSERVABILITY_EVENT,
            "可观测性事件无法严格 JSON 编码",
        ) from exc


@dataclass(slots=True)
class SafeObservabilitySink:
    root: Path
    forbidden_values: Sequence[bytes] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def emit(self, channel: str, event: Mapping[str, Any]) -> None:
        if channel not in OBSERVABILITY_CHANNELS:
            raise D23Error(
                D23RejectionReason.INVALID_OBSERVABILITY_EVENT,
                "未知可观测性 channel",
            )
        if set(event) - SAFE_EVENT_FIELDS:
            self._record_redaction(channel)
            raise D23Error(
                D23RejectionReason.SENSITIVE_OBSERVABILITY_VALUE,
                "可观测性事件包含未批准字段",
            )
        lowered = {str(name).lower() for name in event}
        if any(
            part in name
            for name in lowered
            for part in SENSITIVE_NAME_PARTS
        ):
            self._record_redaction(channel)
            raise D23Error(
                D23RejectionReason.SENSITIVE_OBSERVABILITY_VALUE,
                "敏感字段禁止进入可观测性系统",
            )
        payload = _strict_json(
            {
                "schema_version": 1,
                "channel": channel,
                **dict(event),
            }
        )
        if any(secret and secret in payload for secret in self.forbidden_values):
            self._record_redaction(channel)
            raise D23Error(
                D23RejectionReason.SENSITIVE_OBSERVABILITY_VALUE,
                "敏感值禁止进入可观测性系统",
            )
        self._append(channel, payload)

    def metric(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str],
    ) -> None:
        if set(labels) - SAFE_METRIC_LABELS:
            self._record_redaction("metric")
            raise D23Error(
                D23RejectionReason.SENSITIVE_OBSERVABILITY_VALUE,
                "metric label 不在低基数白名单",
            )
        self.emit(
            "metric",
            {
                "event": name,
                "counter": value,
                **dict(labels),
            },
        )

    def fixed_health(self) -> dict[str, Any]:
        self.emit("health", {"event": "health_check", "result": "ok"})
        return {"schema_version": 1, "status": "ok"}

    def fixed_debug(self) -> dict[str, Any]:
        self.emit("debug", {"event": "debug_check", "result": "disabled"})
        return {"schema_version": 1, "debug": "disabled"}

    def _record_redaction(self, channel: str) -> None:
        self._append(
            channel,
            _strict_json(
                {
                    "schema_version": 1,
                    "channel": channel,
                    "event": "sensitive_event_rejected",
                    "result": "reject",
                }
            ),
        )

    def _append(self, channel: str, payload: bytes) -> None:
        path = self.root / f"{channel}.jsonl"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, payload + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
