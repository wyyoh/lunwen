"""D2 独立 data-key ring 与 AES-256-GCM record codec。"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .stage_c24_contract import RelationId
from .stage_d2_contract import (
    AES_GCM_NONCE_BYTES,
    D2Error,
    D2RejectionReason,
    EncryptedMemoryRecord,
    PublicSyntheticValue,
    canonical_record_aad,
    validate_d2_identifier,
)


DATA_KEY_BYTES = 32


@dataclass(slots=True)
class DataEncryptionKeyRing:
    """与 capability HMAC key ring 完全独立的 AES-256 data keys。"""

    _keys: dict[str, bytes] = field(repr=False)
    active_key_id: str
    _revoked: set[str] = field(default_factory=set, repr=False)
    _used_nonces: set[tuple[str, bytes]] = field(
        default_factory=set, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        copied: dict[str, bytes] = {}
        for key_id, key in self._keys.items():
            copied[key_id] = self._validate_key(key_id, key)
        self._keys = copied
        if self.active_key_id not in self._keys:
            raise ValueError("active data key ID 不存在")

    @staticmethod
    def _validate_key(key_id: str, key: bytes) -> bytes:
        validate_d2_identifier(key_id, label="data_key_id")
        if not key_id.startswith("data-"):
            raise ValueError("data key ID 必须使用 data- namespace")
        if not isinstance(key, bytes) or len(key) != DATA_KEY_BYTES:
            raise ValueError("AES-256 data key 必须是 32 bytes")
        return bytes(key)

    @classmethod
    def generate(cls, key_id: str) -> "DataEncryptionKeyRing":
        return cls({key_id: AESGCM.generate_key(bit_length=256)}, key_id)

    def rotate(self, key_id: str) -> None:
        with self._lock:
            if key_id in self._keys:
                raise ValueError("拒绝覆盖已有 data key ID")
            self._keys[key_id] = self._validate_key(
                key_id, AESGCM.generate_key(bit_length=256)
            )
            self.active_key_id = key_id

    def revoke(self, key_id: str) -> None:
        with self._lock:
            if key_id not in self._keys:
                raise KeyError("无法撤销未知 data key")
            self._revoked.add(key_id)

    def decryption_key(self, key_id: str) -> bytes:
        with self._lock:
            return self._resolve_key_locked(key_id)

    def _resolve_key_locked(self, key_id: str) -> bytes:
        if key_id not in self._keys:
            raise D2Error(
                D2RejectionReason.UNKNOWN_DATA_KEY,
                "record data key ID 不存在",
            )
        if key_id in self._revoked:
            raise D2Error(
                D2RejectionReason.REVOKED_DATA_KEY,
                "record data key 已撤销",
            )
        return self._keys[key_id]

    def encryption_material(self) -> tuple[str, bytes, bytes]:
        """原子取得 active key 与该 key 下从未使用的 nonce。"""

        with self._lock:
            key_id = self.active_key_id
            key = self._resolve_key_locked(key_id)
            for _ in range(32):
                nonce = secrets.token_bytes(AES_GCM_NONCE_BYTES)
                marker = (key_id, nonce)
                if marker not in self._used_nonces:
                    self._used_nonces.add(marker)
                    return key_id, key, nonce
        raise D2Error(
            D2RejectionReason.NONCE_REUSE,
            "无法分配唯一 AES-GCM nonce",
        )

    @property
    def key_material_persisted(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class AesGcmRecordCodec:
    """只使用 cryptography AESGCM 高层 AEAD API。"""

    def encrypt(
        self,
        value: PublicSyntheticValue,
        *,
        record_id: str,
        entity_id: str,
        relation_id: RelationId,
        record_version: int,
        keyring: DataEncryptionKeyRing,
    ) -> EncryptedMemoryRecord:
        if not isinstance(value, PublicSyntheticValue):
            raise TypeError("D2 codec 只加密 PublicSyntheticValue")
        key_id, key, nonce = keyring.encryption_material()
        aad = canonical_record_aad(
            record_id=record_id,
            entity_id=entity_id,
            relation_id=relation_id,
            data_key_id=key_id,
            record_version=record_version,
        )
        ciphertext = AESGCM(key).encrypt(nonce, value.payload, aad)
        return EncryptedMemoryRecord(
            record_id=record_id,
            entity_id=entity_id,
            relation_id=relation_id,
            data_key_id=key_id,
            record_version=record_version,
            nonce=nonce,
            ciphertext=ciphertext,
        )

    def decrypt(
        self,
        record: EncryptedMemoryRecord,
        *,
        keyring: DataEncryptionKeyRing,
    ) -> PublicSyntheticValue:
        if not isinstance(record, EncryptedMemoryRecord):
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "decrypt 只接受 EncryptedMemoryRecord",
            )
        key = keyring.decryption_key(record.data_key_id)
        try:
            plaintext = AESGCM(key).decrypt(
                record.nonce,
                record.ciphertext,
                record.aad(),
            )
        except InvalidTag as exc:
            raise D2Error(
                D2RejectionReason.AUTHENTICATION_FAILED,
                "AES-GCM key/nonce/ciphertext/AAD authentication 失败",
            ) from exc
        return PublicSyntheticValue(plaintext)
