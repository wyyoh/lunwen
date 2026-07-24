"""基于标准 HMAC-SHA-256 的 D1 authenticated capability token。"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Mapping

from .stage_d1_contract import (
    CapabilityClaims,
    CapabilityError,
    RejectionReason,
    TOKEN_ALGORITHM,
    TOKEN_SCHEMA_VERSION,
    TOKEN_TYPE,
)


MINIMUM_KEY_BYTES = 32
DEFAULT_MAX_TOKEN_BYTES = 4096
_BASE64URL = re.compile(br"^[A-Za-z0-9_-]+$")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _base64url_encode(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _base64url_decode(value: bytes) -> bytes:
    if not value or not _BASE64URL.fullmatch(value):
        raise CapabilityError(
            RejectionReason.INVALID_TOKEN,
            "token segment 不是严格 base64url",
        )
    padding = b"=" * ((4 - len(value) % 4) % 4)
    try:
        return base64.b64decode(
            value + padding, altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise CapabilityError(
            RejectionReason.INVALID_TOKEN,
            "token base64url 解码失败",
        ) from exc


def _strict_json_object(value: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError(f"duplicate key {key}")
            output[key] = item
        return output

    try:
        decoded = value.decode("utf-8", errors="strict")
        payload = json.loads(
            decoded,
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CapabilityError(
            RejectionReason.INVALID_TOKEN,
            f"{label} 不是严格 JSON object",
        ) from exc
    if not isinstance(payload, dict):
        raise CapabilityError(
            RejectionReason.INVALID_TOKEN,
            f"{label} 必须是 JSON object",
        )
    return payload


@dataclass(slots=True)
class HmacKeyRing:
    """只存在于可信进程内存中的 HMAC key ring。"""

    _keys: dict[str, bytes] = field(repr=False)
    active_key_id: str
    _revoked: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        copied = {}
        for key_id, key in self._keys.items():
            copied[key_id] = self._validate_key(key_id, key)
        self._keys = copied
        if self.active_key_id not in self._keys:
            raise ValueError("active key ID 不存在")

    @staticmethod
    def _validate_key(key_id: str, key: bytes) -> bytes:
        from .stage_d1_contract import validate_identifier

        validate_identifier(key_id, label="HMAC key_id")
        if not isinstance(key, bytes) or len(key) < MINIMUM_KEY_BYTES:
            raise ValueError("HMAC key 至少需要 32 bytes")
        return bytes(key)

    @classmethod
    def generate(cls, key_id: str, *, key_bytes: int = 32) -> "HmacKeyRing":
        if key_bytes < MINIMUM_KEY_BYTES:
            raise ValueError("HMAC 随机 key 至少需要 32 bytes")
        return cls({key_id: secrets.token_bytes(key_bytes)}, key_id)

    def add_and_activate(self, key_id: str, key: bytes) -> None:
        if key_id in self._keys:
            raise ValueError("拒绝覆盖已有 key ID")
        self._keys[key_id] = self._validate_key(key_id, key)
        self._revoked.discard(key_id)
        self.active_key_id = key_id

    def rotate(self, key_id: str, *, key_bytes: int = 32) -> None:
        if key_bytes < MINIMUM_KEY_BYTES:
            raise ValueError("rotated key 至少需要 32 bytes")
        self.add_and_activate(key_id, secrets.token_bytes(key_bytes))

    def revoke(self, key_id: str) -> None:
        if key_id not in self._keys:
            raise KeyError("无法撤销未知 key ID")
        self._revoked.add(key_id)

    def signing_key(self, key_id: str) -> bytes:
        if key_id != self.active_key_id:
            raise CapabilityError(
                RejectionReason.UNKNOWN_KEY,
                "authority 只能使用 active key 签发",
            )
        return self.verification_key(key_id)

    def verification_key(self, key_id: str) -> bytes:
        if key_id not in self._keys:
            raise CapabilityError(
                RejectionReason.UNKNOWN_KEY,
                "capability key ID 不存在",
            )
        if key_id in self._revoked:
            raise CapabilityError(
                RejectionReason.REVOKED_KEY,
                "capability key 已撤销",
            )
        return self._keys[key_id]

    @property
    def key_material_persisted(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CapabilityTokenCodec:
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES

    def __post_init__(self) -> None:
        if self.max_token_bytes < 256:
            raise ValueError("max token bytes 过小")

    def issue(self, claims: CapabilityClaims, keyring: HmacKeyRing) -> bytes:
        if not isinstance(claims, CapabilityClaims):
            raise TypeError("codec 只能签发 CapabilityClaims")
        header = {
            "alg": TOKEN_ALGORITHM,
            "kid": claims.key_id,
            "typ": TOKEN_TYPE,
            "v": TOKEN_SCHEMA_VERSION,
        }
        encoded_header = _base64url_encode(_canonical_json(header))
        encoded_payload = _base64url_encode(
            _canonical_json(claims.to_payload())
        )
        signing_input = encoded_header + b"." + encoded_payload
        signature = hmac.digest(
            keyring.signing_key(claims.key_id),
            signing_input,
            "sha256",
        )
        token = signing_input + b"." + _base64url_encode(signature)
        if len(token) > self.max_token_bytes:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "签发 token 超出长度上限",
            )
        return token

    def verify(
        self, token: bytes, keyring: HmacKeyRing
    ) -> CapabilityClaims:
        if (
            not isinstance(token, bytes)
            or not token
            or len(token) > self.max_token_bytes
        ):
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "token 必须是长度受限的非空 bytes",
            )
        segments = token.split(b".")
        if len(segments) != 3:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "token 必须包含三个 compact segments",
            )
        encoded_header, encoded_payload, encoded_signature = segments
        header = _strict_json_object(
            _base64url_decode(encoded_header), label="token header"
        )
        expected_header_keys = {"alg", "kid", "typ", "v"}
        if (
            set(header) != expected_header_keys
            or header.get("alg") != TOKEN_ALGORITHM
            or header.get("typ") != TOKEN_TYPE
            or header.get("v") != TOKEN_SCHEMA_VERSION
            or not isinstance(header.get("kid"), str)
        ):
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "拒绝 algorithm/type/version confusion",
            )
        key = keyring.verification_key(header["kid"])
        supplied = _base64url_decode(encoded_signature)
        if len(supplied) != 32:
            raise CapabilityError(
                RejectionReason.BAD_SIGNATURE,
                "HMAC-SHA-256 signature 长度非法",
            )
        signing_input = encoded_header + b"." + encoded_payload
        expected = hmac.digest(key, signing_input, "sha256")
        if not hmac.compare_digest(supplied, expected):
            raise CapabilityError(
                RejectionReason.BAD_SIGNATURE,
                "HMAC signature 验证失败",
            )
        payload = _strict_json_object(
            _base64url_decode(encoded_payload), label="token payload"
        )
        claims = CapabilityClaims.from_payload(payload)
        if claims.key_id != header["kid"]:
            raise CapabilityError(
                RejectionReason.INVALID_TOKEN,
                "header/payload key ID 不一致",
            )
        return claims
