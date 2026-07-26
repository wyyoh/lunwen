"""D2.3 字节级 IPC framing、端点守卫与资源预算。"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from .stage_d22_contract import D22Error
from .stage_d22_ipc import MAX_IPC_MESSAGE_BYTES, decode_message, encode_message
from .stage_d23_contract import D23Error, D23RejectionReason

FRAME_HEADER_BYTES = 4
MAX_JSON_DEPTH = 16


def _depth(value: Any, current: int = 0) -> int:
    if isinstance(value, Mapping):
        if not value:
            return current + 1
        return max(_depth(item, current + 1) for item in value.values())
    if isinstance(value, list):
        if not value:
            return current + 1
        return max(_depth(item, current + 1) for item in value)
    return current


def encode_frame(message: Mapping[str, Any]) -> bytes:
    payload = encode_message(message)
    return struct.pack(">I", len(payload)) + payload


def decode_frame(chunks: Iterable[bytes]) -> dict[str, Any]:
    raw = b"".join(chunks)
    if len(raw) < FRAME_HEADER_BYTES:
        raise D23Error(D23RejectionReason.INVALID_FRAME, "IPC frame header 截断")
    length = struct.unpack(">I", raw[:FRAME_HEADER_BYTES])[0]
    if length <= 0 or length > MAX_IPC_MESSAGE_BYTES:
        raise D23Error(
            D23RejectionReason.MESSAGE_TOO_LARGE,
            "IPC frame 声明长度非法",
        )
    payload = raw[FRAME_HEADER_BYTES:]
    if len(payload) != length:
        raise D23Error(D23RejectionReason.INVALID_FRAME, "IPC frame payload 截断")
    try:
        message = decode_message(payload)
    except D22Error as exc:
        raise D23Error(
            D23RejectionReason.INVALID_FRAME,
            "IPC frame payload 不是严格 JSON",
        ) from exc
    if _depth(message) > MAX_JSON_DEPTH:
        raise D23Error(
            D23RejectionReason.MESSAGE_TOO_DEEP,
            "IPC JSON 嵌套深度超限",
        )
    return message


@dataclass(slots=True)
class ByteAuditProxy:
    """raw frame 仅驻留内存，持久化时只导出 digest/长度。"""

    frames: list[bytes] = field(default_factory=list, repr=False)

    def transmit(
        self,
        message: Mapping[str, Any],
        *,
        split_at: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        frame = encode_frame(message)
        self.frames.append(frame)
        indexes = (0, *split_at, len(frame))
        chunks = tuple(
            frame[start:end]
            for start, end in pairwise(indexes)
        )
        return decode_frame(chunks)

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "frame_index": index,
                "wire_size_bytes": len(frame),
                "sha256": hashlib.sha256(frame).hexdigest(),
                "raw_bytes_persisted": False,
            }
            for index, frame in enumerate(self.frames)
        ]


def prepare_socket_path(path: Path, *, expected_uid: int | None = None) -> None:
    uid = os.getuid() if expected_uid is None else expected_uid
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.stat()
    if parent.st_uid != uid or stat.S_IMODE(parent.st_mode) & 0o077:
        raise D23Error(
            D23RejectionReason.ENDPOINT_UNSAFE,
            "socket 目录 owner/权限不安全",
        )
    try:
        existing = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(existing.st_mode):
        raise D23Error(
            D23RejectionReason.ENDPOINT_UNSAFE,
            "拒绝 socket 路径 symlink",
        )
    if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != uid:
        raise D23Error(
            D23RejectionReason.ENDPOINT_UNSAFE,
            "拒绝未知 stale endpoint",
        )
    path.unlink()


def verify_peer_uid(peer_socket: socket.socket, *, expected_uid: int) -> None:
    if not hasattr(socket, "SO_PEERCRED"):
        if expected_uid != os.getuid():
            raise D23Error(
                D23RejectionReason.PEER_IDENTITY_MISMATCH,
                "平台不支持 peer credential 且期望 UID 不匹配",
            )
        return
    payload = peer_socket.getsockopt(
        socket.SOL_SOCKET,
        socket.SO_PEERCRED,
        struct.calcsize("3i"),
    )
    _, uid, _ = struct.unpack("3i", payload)
    if uid != expected_uid:
        raise D23Error(
            D23RejectionReason.PEER_IDENTITY_MISMATCH,
            "本地 peer UID 不匹配",
        )


@dataclass(slots=True)
class ConnectionBudget:
    maximum_invalid_connections: int
    maximum_inflight_bytes: int
    invalid_connections: int = 0
    inflight_bytes: int = 0

    def admit(self, *, declared_bytes: int, valid: bool) -> None:
        if (
            declared_bytes < 0
            or declared_bytes > self.maximum_inflight_bytes
            or self.inflight_bytes + declared_bytes > self.maximum_inflight_bytes
        ):
            raise D23Error(
                D23RejectionReason.RATE_LIMITED,
                "IPC inflight byte budget 超限",
            )
        if not valid:
            self.invalid_connections += 1
            if self.invalid_connections > self.maximum_invalid_connections:
                raise D23Error(
                    D23RejectionReason.RATE_LIMITED,
                    "无效连接速率超限",
                )
        self.inflight_bytes += declared_bytes

    def release(self, *, declared_bytes: int) -> None:
        self.inflight_bytes = max(0, self.inflight_bytes - declared_bytes)


def duplicate_key_frame() -> bytes:
    payload = b'{"op":"a","op":"b"}'
    return struct.pack(">I", len(payload)) + payload


def overlong_frame() -> bytes:
    return struct.pack(">I", MAX_IPC_MESSAGE_BYTES + 1)


def nested_frame(depth: int = MAX_JSON_DEPTH + 2) -> bytes:
    value: Any = "leaf"
    for index in range(depth):
        value = {f"n{index}": value}
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload
