"""AuthCapV1 的标准 Ed25519 签发与严格解析。

Ed25519 仅用于签发方/验证方分离，不是 AuthCap 的算法创新。
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .authcap_policy import canonical_json_bytes

_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}$")
_HEADER_KEYS = frozenset({"alg", "kid", "schema_version", "typ"})
_PAYLOAD_KEYS = frozenset(
    {
        "algorithm",
        "authorization_witness_digest",
        "capability_id",
        "decision_id",
        "executable_authority_digest",
        "expires_at",
        "issued_at",
        "issuer_key_id",
        "nonce",
        "policy_epoch",
        "policy_hash",
        "previous_workflow_state_digest",
        "proposal_digest",
        "schema_version",
        "single_use",
        "subject_id",
        "trusted_request_digest",
        "workflow_id",
        "workflow_step",
    }
)


class AuthCapTokenError(ValueError):
    """AuthCapV1 token 不满足严格 schema、签名或生命周期要求。"""


def _safe_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_ID.fullmatch(value)
        or "*" in value
        or value.casefold() in {"all", "any"}
    ):
        raise AuthCapTokenError(f"{label} 非法")
    return value


def _hex_digest(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuthCapTokenError(f"{label} 必须是小写 SHA-256")
    return value


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthCapTokenError("时间字段必须是 ISO-8601 UTC") from exc
    if parsed.tzinfo is None:
        raise AuthCapTokenError("时间字段必须包含时区")
    return parsed.astimezone(timezone.utc)


def _b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64d(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise AuthCapTokenError("base64url 字段为空")
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise AuthCapTokenError("非法 base64url") from exc


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise AuthCapTokenError(f"duplicate JSON key：{key}")
        value[key] = child
    return value


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AuthCapTokenError("token JSON 含 NaN/Infinity")
    if isinstance(value, Mapping):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _strict_json(value: bytes, expected_keys: frozenset[str]) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AuthCapTokenError(f"非法 JSON 常量：{item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthCapTokenError("token JSON 无法解析") from exc
    if not isinstance(parsed, dict) or set(parsed) != expected_keys:
        raise AuthCapTokenError("token JSON schema 不匹配")
    _finite(parsed)
    return parsed


@dataclass(frozen=True)
class AuthCapV1:
    capability_id: str
    issuer_key_id: str
    subject_id: str
    policy_hash: str
    decision_id: str
    authorization_witness_digest: str
    trusted_request_digest: str
    proposal_digest: str
    executable_authority_digest: str
    policy_epoch: int
    issued_at: str
    expires_at: str
    single_use: bool
    nonce: str
    workflow_id: str | None = None
    workflow_step: int | None = None
    previous_workflow_state_digest: str | None = None
    schema_version: int = 1
    algorithm: str = "Ed25519"

    def __post_init__(self) -> None:
        for name in ("capability_id", "issuer_key_id", "subject_id", "nonce"):
            _safe_id(getattr(self, name), name)
        for name in (
            "policy_hash",
            "decision_id",
            "authorization_witness_digest",
            "trusted_request_digest",
            "proposal_digest",
            "executable_authority_digest",
        ):
            _hex_digest(getattr(self, name), name)
        if self.policy_epoch < 1:
            raise AuthCapTokenError("policy_epoch 必须为正整数")
        if self.schema_version != 1 or self.algorithm != "Ed25519":
            raise AuthCapTokenError("AuthCapV1 schema/algorithm 不匹配")
        if not isinstance(self.single_use, bool):
            raise AuthCapTokenError("single_use 必须为 bool")
        if _utc(self.issued_at) >= _utc(self.expires_at):
            raise AuthCapTokenError("expires_at 必须晚于 issued_at")
        if self.workflow_id is not None:
            _safe_id(self.workflow_id, "workflow_id")
            if self.workflow_step is None or self.workflow_step < 1:
                raise AuthCapTokenError("workflow_step 非法")
        elif (
            self.workflow_step is not None
            or self.previous_workflow_state_digest is not None
        ):
            raise AuthCapTokenError("非 workflow capability 不得携带 workflow state")
        if self.previous_workflow_state_digest is not None:
            _hex_digest(
                self.previous_workflow_state_digest,
                "previous_workflow_state_digest",
            )

    def canonical(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "authorization_witness_digest": self.authorization_witness_digest,
            "capability_id": self.capability_id,
            "decision_id": self.decision_id,
            "executable_authority_digest": self.executable_authority_digest,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "issuer_key_id": self.issuer_key_id,
            "nonce": self.nonce,
            "policy_epoch": self.policy_epoch,
            "policy_hash": self.policy_hash,
            "previous_workflow_state_digest": self.previous_workflow_state_digest,
            "proposal_digest": self.proposal_digest,
            "schema_version": self.schema_version,
            "single_use": self.single_use,
            "subject_id": self.subject_id,
            "trusted_request_digest": self.trusted_request_digest,
            "workflow_id": self.workflow_id,
            "workflow_step": self.workflow_step,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AuthCapV1:
        if set(value) != _PAYLOAD_KEYS:
            raise AuthCapTokenError("AuthCapV1 payload schema 不匹配")
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise AuthCapTokenError("AuthCapV1 payload 类型不匹配") from exc


class AuthCapIssuer:
    """只持有 Ed25519 private key 的签发端。"""

    def __init__(self, key_id: str, private_key: Ed25519PrivateKey) -> None:
        self.key_id = _safe_id(key_id, "issuer key_id")
        self._private_key = private_key

    @classmethod
    def generate(cls, key_id: str) -> AuthCapIssuer:
        return cls(key_id, Ed25519PrivateKey.generate())

    @classmethod
    def load_private_pem(cls, key_id: str, path: str | Path) -> AuthCapIssuer:
        key = serialization.load_pem_private_key(
            Path(path).read_bytes(),
            password=None,
        )
        if not isinstance(key, Ed25519PrivateKey):
            raise AuthCapTokenError("private key 不是 Ed25519")
        return cls(key_id, key)

    def write_private_pem(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            self._private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        target.chmod(0o600)

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )

    def issue(self, payload: AuthCapV1, *, maximum_bytes: int) -> str:
        if payload.issuer_key_id != self.key_id:
            raise AuthCapTokenError("payload issuer_key_id 与签发 key 不一致")
        header = {
            "alg": "Ed25519",
            "kid": self.key_id,
            "schema_version": 1,
            "typ": "AUTHCAP",
        }
        encoded_header = _b64e(canonical_json_bytes(header))
        encoded_payload = _b64e(canonical_json_bytes(payload.canonical()))
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        token = (
            f"{encoded_header}.{encoded_payload}."
            f"{_b64e(self._private_key.sign(signing_input))}"
        )
        if len(token.encode("ascii")) > maximum_bytes:
            raise AuthCapTokenError("capability 超过最大字节限制")
        return token


class AuthCapPublicKeyRing:
    """只含 public key 的 AuthCapV1 verifier key ring。"""

    def __init__(self, keys: Mapping[str, bytes]) -> None:
        if not keys:
            raise AuthCapTokenError("public key ring 不能为空")
        self._keys = {
            _safe_id(key_id, "public key_id"): Ed25519PublicKey.from_public_bytes(
                value
            )
            for key_id, value in keys.items()
        }

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "Ed25519",
            "keys": [
                {
                    "key_id": key_id,
                    "public_key_base64url": _b64e(
                        key.public_bytes(
                            serialization.Encoding.Raw,
                            serialization.PublicFormat.Raw,
                        )
                    ),
                }
                for key_id, key in sorted(self._keys.items())
            ],
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> AuthCapPublicKeyRing:
        allowed = {
            "schema_version",
            "algorithm",
            "keys",
            "manifest_payload_sha256",
        }
        if set(value) not in (
            {"schema_version", "algorithm", "keys"},
            allowed,
        ):
            raise AuthCapTokenError("public verifier manifest schema 不匹配")
        if "manifest_payload_sha256" in value:
            payload = {
                key: child
                for key, child in value.items()
                if key != "manifest_payload_sha256"
            }
            observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            if observed != value["manifest_payload_sha256"]:
                raise AuthCapTokenError("public verifier manifest hash mismatch")
        if value["schema_version"] != 1 or value["algorithm"] != "Ed25519":
            raise AuthCapTokenError("public verifier algorithm confusion")
        keys = value["keys"]
        if (
            not isinstance(keys, list)
            or not keys
            or any(
                not isinstance(item, dict)
                or set(item) != {"key_id", "public_key_base64url"}
                for item in keys
            )
        ):
            raise AuthCapTokenError("public verifier keys 非法")
        key_map = {
            item["key_id"]: _b64d(item["public_key_base64url"])
            for item in keys
        }
        if len(key_map) != len(keys):
            raise AuthCapTokenError("duplicate public verifier key ID")
        return cls(key_map)

    def verify_signature(
        self,
        token: str,
        *,
        maximum_bytes: int,
    ) -> AuthCapV1:
        if not isinstance(token, str) or len(token.encode("utf-8")) > maximum_bytes:
            raise AuthCapTokenError("capability 类型或长度非法")
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthCapTokenError("capability 必须包含三个 segment")
        header_bytes, payload_bytes, signature = map(_b64d, parts)
        header = _strict_json(header_bytes, _HEADER_KEYS)
        if (
            header["alg"] != "Ed25519"
            or header["typ"] != "AUTHCAP"
            or header["schema_version"] != 1
        ):
            raise AuthCapTokenError("capability header algorithm/schema confusion")
        key_id = _safe_id(header["kid"], "header.kid")
        if key_id not in self._keys:
            raise AuthCapTokenError("unknown issuer key ID")
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        try:
            self._keys[key_id].verify(signature, signing_input)
        except InvalidSignature as exc:
            raise AuthCapTokenError("Ed25519 signature 验证失败") from exc
        payload = AuthCapV1.from_mapping(_strict_json(payload_bytes, _PAYLOAD_KEYS))
        if payload.issuer_key_id != key_id:
            raise AuthCapTokenError("header/payload key ID 不一致")
        return payload
