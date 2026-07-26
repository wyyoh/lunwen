"""D2.2 gateway→memory 内部 release ticket 的 HMAC 认证编码。"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .stage_d1_contract import validate_identifier
from .stage_d22_contract import (
    D22_SCHEMA_VERSION,
    D22_TICKET_ALGORITHM,
    D22_TICKET_TYPE,
    D22Error,
    D22RejectionReason,
    ReleaseTicketClaims,
)

_BASE64URL = re.compile(br"^[A-Za-z0-9_-]+$")
MAX_TICKET_BYTES = 4096


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _encode(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _decode(value: bytes) -> bytes:
    if not value or not _BASE64URL.fullmatch(value):
        raise D22Error(
            D22RejectionReason.INVALID_TICKET,
            "ticket segment 不是严格 base64url",
        )
    padding = b"=" * ((4 - len(value) % 4) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise D22Error(
            D22RejectionReason.INVALID_TICKET,
            "ticket base64url 解码失败",
        ) from exc


def _strict_object(value: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError(f"duplicate key {key}")
            output[key] = item
        return output

    try:
        result = json.loads(
            value.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise D22Error(
            D22RejectionReason.INVALID_TICKET,
            f"{label} 不是严格 JSON object",
        ) from exc
    if not isinstance(result, dict):
        raise D22Error(
            D22RejectionReason.INVALID_TICKET,
            f"{label} 必须是 JSON object",
        )
    return result


@dataclass(frozen=True, slots=True)
class ReleaseTicketKey:
    key_id: str
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        validate_identifier(self.key_id, label="release ticket key_id")
        if not self.key_id.startswith("release-ticket-"):
            raise ValueError("release ticket key 必须使用独立 namespace")
        if not isinstance(self.key, bytes) or len(self.key) < 32:
            raise ValueError("release ticket HMAC key 至少 32 bytes")


@dataclass(frozen=True, slots=True)
class ReleaseTicketCodec:
    max_ticket_bytes: int = MAX_TICKET_BYTES

    def issue(
        self,
        claims: ReleaseTicketClaims,
        key: ReleaseTicketKey,
    ) -> bytes:
        if not isinstance(claims, ReleaseTicketClaims):
            raise TypeError("ticket codec 只签发 ReleaseTicketClaims")
        header = {
            "alg": D22_TICKET_ALGORITHM,
            "kid": key.key_id,
            "typ": D22_TICKET_TYPE,
            "v": D22_SCHEMA_VERSION,
        }
        header_segment = _encode(_canonical_json(header))
        payload_segment = _encode(_canonical_json(claims.to_payload()))
        signing_input = header_segment + b"." + payload_segment
        signature = hmac.digest(key.key, signing_input, "sha256")
        ticket = signing_input + b"." + _encode(signature)
        if len(ticket) > self.max_ticket_bytes:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "release ticket 超出长度上限",
            )
        return ticket

    def verify(
        self,
        ticket: bytes,
        key: ReleaseTicketKey,
    ) -> ReleaseTicketClaims:
        if (
            not isinstance(ticket, bytes)
            or not ticket
            or len(ticket) > self.max_ticket_bytes
        ):
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "release ticket 必须是长度受限 bytes",
            )
        parts = ticket.split(b".")
        if len(parts) != 3:
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "release ticket 必须包含三个 compact segments",
            )
        encoded_header, encoded_payload, encoded_signature = parts
        header = _strict_object(_decode(encoded_header), label="ticket header")
        if (
            set(header) != {"alg", "kid", "typ", "v"}
            or header.get("alg") != D22_TICKET_ALGORITHM
            or header.get("kid") != key.key_id
            or header.get("typ") != D22_TICKET_TYPE
            or header.get("v") != D22_SCHEMA_VERSION
        ):
            raise D22Error(
                D22RejectionReason.INVALID_TICKET,
                "拒绝 ticket algorithm/type/key/version confusion",
            )
        supplied = _decode(encoded_signature)
        signing_input = encoded_header + b"." + encoded_payload
        expected = hmac.digest(key.key, signing_input, "sha256")
        if len(supplied) != 32 or not hmac.compare_digest(
            supplied, expected
        ):
            raise D22Error(
                D22RejectionReason.BAD_TICKET_SIGNATURE,
                "release ticket HMAC 验证失败",
            )
        payload = _strict_object(
            _decode(encoded_payload),
            label="ticket payload",
        )
        return ReleaseTicketClaims.from_payload(payload)
