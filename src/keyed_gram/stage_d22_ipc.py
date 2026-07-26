"""D2.2 严格 JSON/AF_UNIX IPC、认证 RPC 与安全审计事件。"""

from __future__ import annotations

import json
import os
import resource
import time
from collections.abc import Mapping
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from .stage_d22_contract import (
    D22_SCHEMA_VERSION,
    D22Error,
    D22RejectionReason,
)

MAX_IPC_MESSAGE_BYTES = 64 * 1024


def encode_message(value: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "IPC message 不能严格 JSON 编码",
        ) from exc
    if len(payload) > MAX_IPC_MESSAGE_BYTES:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "IPC message 超出长度上限",
        )
    return payload


def decode_message(value: bytes) -> dict[str, Any]:
    if (
        not isinstance(value, bytes)
        or not value
        or len(value) > MAX_IPC_MESSAGE_BYTES
    ):
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "IPC message 必须是长度受限 bytes",
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError(f"duplicate key {key}")
            output[key] = item
        return output

    try:
        decoded = json.loads(
            value.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "IPC message 不是严格 JSON object",
        ) from exc
    if not isinstance(decoded, dict):
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "IPC message 顶层必须是 object",
        )
    return decoded


def rpc_call(
    socket_path: Path,
    authkey: bytes,
    message: Mapping[str, Any],
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    try:
        with Client(
            str(socket_path),
            family="AF_UNIX",
            authkey=authkey,
        ) as connection:
            connection.send_bytes(encode_message(message))
            if not connection.poll(timeout_seconds):
                raise D22Error(
                    D22RejectionReason.IPC_TIMEOUT,
                    "service IPC response timeout",
                )
            return decode_message(
                connection.recv_bytes(MAX_IPC_MESSAGE_BYTES)
            )
    except D22Error:
        raise
    except AuthenticationError as exc:
        raise D22Error(
            D22RejectionReason.SERVICE_AUTHENTICATION_FAILED,
            "service IPC HMAC challenge 失败",
        ) from exc
    except (EOFError, BrokenPipeError, ConnectionError, OSError) as exc:
        raise D22Error(
            D22RejectionReason.UPSTREAM_UNAVAILABLE,
            "service IPC 连接不可用",
        ) from exc


def wait_for_socket(
    socket_path: Path,
    *,
    timeout_seconds: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.01)
    raise D22Error(
        D22RejectionReason.IPC_TIMEOUT,
        "service socket 未在时限内就绪",
    )


def secure_process_setup(log_path: Path) -> None:
    os.umask(0o077)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    if hasattr(resource, "RLIMIT_CORE"):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def append_safe_event(path: Path, event: Mapping[str, Any]) -> None:
    payload = encode_message(
        {
            "schema_version": D22_SCHEMA_VERSION,
            **dict(event),
        }
    ) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def message_schema_event(
    *,
    edge: str,
    message: Mapping[str, Any],
    wire_size_bytes: int,
) -> dict[str, Any]:
    nested_fields = {
        key: sorted(value)
        for key, value in message.items()
        if isinstance(value, Mapping)
    }
    all_fields = set(message)
    for fields in nested_fields.values():
        all_fields.update(fields)
    return {
        "event": "ipc_schema",
        "edge": edge,
        "top_level_fields": sorted(message),
        "nested_fields": nested_fields,
        "wire_size_bytes": wire_size_bytes,
        "contains_capability_token_field": "capability_token_b64"
        in all_fields,
        "contains_capability_hmac_key_field": "capability_hmac_key"
        in all_fields,
        "contains_data_encryption_key_field": "data_encryption_key"
        in all_fields,
        "contains_release_ticket_field": "release_ticket_b64"
        in all_fields,
        "contains_value_field": "value_b64" in all_fields,
        "contains_natural_language_prompt_field": bool(
            {"prompt", "public_instruction", "raw_text"} & all_fields
        ),
    }


def read_safe_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if line:
            events.append(decode_message(line))
    return events
