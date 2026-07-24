"""D2 principal-bound gateway 与 capability-gated AEAD keyed memory。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .stage_c24_contract import RelationId
from .stage_d1_contract import (
    AuthorizedMemoryRequest,
    CapabilityError,
    RejectionReason,
    SuggestedRelation,
    assert_verified_capability,
)
from .stage_d1_gateway import CapabilityVerifier
from .stage_d2_contract import (
    AuthenticatedPrincipal,
    D2Error,
    D2RejectionReason,
    EncryptedMemoryRecord,
    PlaintextRelease,
    PublicSyntheticValue,
    assert_authenticated_principal,
)
from .stage_d2_crypto import AesGcmRecordCodec, DataEncryptionKeyRing
from .stage_d2_identity import TrustedMockIdentityProvider


@dataclass(frozen=True, slots=True)
class MemoryCounterSnapshot:
    lookup_count: int
    decrypt_attempt_count: int
    plaintext_release_count: int
    migration_decrypt_attempt_count: int
    successful_record_migration_count: int

    def delta(self, previous: "MemoryCounterSnapshot") -> dict[str, int]:
        return {
            "memory_lookup_delta": self.lookup_count - previous.lookup_count,
            "decrypt_attempt_delta": (
                self.decrypt_attempt_count - previous.decrypt_attempt_count
            ),
            "plaintext_release_delta": (
                self.plaintext_release_count
                - previous.plaintext_release_count
            ),
            "migration_decrypt_attempt_delta": (
                self.migration_decrypt_attempt_count
                - previous.migration_decrypt_attempt_count
            ),
            "successful_record_migration_delta": (
                self.successful_record_migration_count
                - previous.successful_record_migration_count
            ),
        }


@dataclass(slots=True)
class EncryptedKeyedMemory:
    """不在 capability 验证前暴露 lookup/decrypt API。"""

    codec: AesGcmRecordCodec
    data_keyring: DataEncryptionKeyRing
    _gateway_binding: object = field(repr=False)
    _records: dict[str, EncryptedMemoryRecord] = field(
        default_factory=dict, repr=False
    )
    _bucket_index: dict[tuple[str, RelationId], str] = field(
        default_factory=dict, repr=False
    )
    _minimum_versions: dict[str, int] = field(
        default_factory=dict, repr=False
    )
    _lookup_count: int = 0
    _decrypt_attempt_count: int = 0
    _plaintext_release_count: int = 0
    _migration_decrypt_attempt_count: int = 0
    _successful_record_migration_count: int = 0
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )

    @classmethod
    def from_records(
        cls,
        codec: AesGcmRecordCodec,
        data_keyring: DataEncryptionKeyRing,
        records: Iterable[EncryptedMemoryRecord],
        *,
        gateway_binding: object,
        minimum_versions: Mapping[str, int] | None = None,
    ) -> "EncryptedKeyedMemory":
        if gateway_binding is None:
            raise ValueError("D2 memory 必须绑定具体 gateway instance")
        memory = cls(codec, data_keyring, gateway_binding)
        for record in records:
            memory.add_record(record)
        if minimum_versions is not None:
            for record_id, version in minimum_versions.items():
                if (
                    record_id not in memory._records
                    or type(version) is not int
                    or version <= 0
                ):
                    raise D2Error(
                        D2RejectionReason.INVALID_RECORD,
                        "minimum version declaration 非法",
                    )
                memory._minimum_versions[record_id] = version
        return memory

    def add_record(self, record: EncryptedMemoryRecord) -> None:
        if not isinstance(record, EncryptedMemoryRecord):
            raise D2Error(
                D2RejectionReason.INVALID_RECORD,
                "memory 只接受 EncryptedMemoryRecord",
            )
        with self._lock:
            if record.record_id in self._records:
                raise D2Error(
                    D2RejectionReason.DUPLICATE_RECORD_ID,
                    "duplicate record ID",
                )
            bucket = (record.entity_id, record.relation_id)
            if bucket in self._bucket_index:
                raise D2Error(
                    D2RejectionReason.DUPLICATE_BUCKET,
                    "D2 每个 entity/relation 只允许一个 current record",
                )
            self._records[record.record_id] = record
            self._bucket_index[bucket] = record.record_id
            self._minimum_versions[record.record_id] = record.record_version

    def retrieve_authorized(
        self,
        principal: AuthenticatedPrincipal,
        capability: Any,
    ) -> PlaintextRelease:
        """直接调用永远拒绝；仅用于验证 bypass 尝试 fail-closed。"""

        assert_authenticated_principal(principal)
        try:
            assert_verified_capability(capability)
        except TypeError as exc:
            raise D2Error(
                D2RejectionReason.FORGED_VERIFIED_CAPABILITY,
                "memory 拒绝 forged VerifiedCapability",
            ) from exc
        raise D2Error(
            D2RejectionReason.FORGED_VERIFIED_CAPABILITY,
            "VerifiedCapability 未携带 gateway–memory instance binding",
        )

    def _retrieve_from_gateway(
        self,
        principal: AuthenticatedPrincipal,
        capability: Any,
        *,
        gateway_binding: object,
    ) -> PlaintextRelease:
        if gateway_binding is not self._gateway_binding:
            raise D2Error(
                D2RejectionReason.FORGED_VERIFIED_CAPABILITY,
                "gateway–memory instance binding 不匹配",
            )
        trusted_principal = assert_authenticated_principal(principal)
        try:
            verified = assert_verified_capability(capability)
        except TypeError as exc:
            raise D2Error(
                D2RejectionReason.FORGED_VERIFIED_CAPABILITY,
                "memory 拒绝 forged VerifiedCapability",
            ) from exc
        if trusted_principal.subject_id != verified.subject_id:
            raise D2Error(
                D2RejectionReason.PRINCIPAL_MISMATCH,
                "principal/capability subject 不一致",
            )
        bucket = (verified.entity_id, verified.relation_id)
        with self._lock:
            self._lookup_count += 1
            record_id = self._bucket_index.get(bucket)
            if record_id is None:
                raise D2Error(
                    D2RejectionReason.RECORD_NOT_FOUND,
                    "typed entity/relation record 不存在",
                )
            record = self._records[record_id]
            minimum = self._minimum_versions[record_id]
            if record.record_version < minimum:
                raise D2Error(
                    D2RejectionReason.RECORD_VERSION_ROLLBACK,
                    "record version 低于 trusted floor",
                )
            self._decrypt_attempt_count += 1
            value = self.codec.decrypt(
                record,
                keyring=self.data_keyring,
            )
            self._plaintext_release_count += 1
            return PlaintextRelease(
                record_id=record.record_id,
                entity_id=record.entity_id,
                relation_id=record.relation_id,
                value=value,
                value_digest=value.digest,
            )

    def migrate_record_to_active_key(
        self,
        record_id: str,
    ) -> EncryptedMemoryRecord:
        """可信管理操作：plaintext 只在 TCB 内部短暂存在，不向调用方释放。"""

        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                raise D2Error(
                    D2RejectionReason.RECORD_NOT_FOUND,
                    "待迁移 record 不存在",
                )
            if record.record_version < self._minimum_versions[record_id]:
                raise D2Error(
                    D2RejectionReason.RECORD_VERSION_ROLLBACK,
                    "拒绝迁移 rollback record",
                )
            self._migration_decrypt_attempt_count += 1
            value = self.codec.decrypt(
                record,
                keyring=self.data_keyring,
            )
            migrated = self.codec.encrypt(
                value,
                record_id=record.record_id,
                entity_id=record.entity_id,
                relation_id=record.relation_id,
                record_version=record.record_version + 1,
                keyring=self.data_keyring,
            )
            self._records[record_id] = migrated
            self._minimum_versions[record_id] = migrated.record_version
            self._successful_record_migration_count += 1
            return migrated

    def snapshot(self) -> MemoryCounterSnapshot:
        with self._lock:
            return MemoryCounterSnapshot(
                lookup_count=self._lookup_count,
                decrypt_attempt_count=self._decrypt_attempt_count,
                plaintext_release_count=self._plaintext_release_count,
                migration_decrypt_attempt_count=(
                    self._migration_decrypt_attempt_count
                ),
                successful_record_migration_count=(
                    self._successful_record_migration_count
                ),
            )


@dataclass(slots=True)
class PrincipalBoundCapabilityGateway:
    identity_provider: TrustedMockIdentityProvider
    capability_verifier: CapabilityVerifier
    clock: Callable[[], int]
    _memory_binding: object = field(repr=False)

    def retrieve(
        self,
        session_credential: bytes,
        request: AuthorizedMemoryRequest,
        memory: EncryptedKeyedMemory,
    ) -> PlaintextRelease:
        if isinstance(request, SuggestedRelation):
            raise CapabilityError(
                RejectionReason.ROUTER_PROPOSAL_NOT_AUTHORIZED,
                "SuggestedRelation 不能形成 D2 授权请求",
            )
        principal = self.identity_provider.authenticate(session_credential)
        if type(request) is not AuthorizedMemoryRequest:
            raise CapabilityError(
                RejectionReason.INVALID_REQUEST,
                "D2 gateway 只接受 AuthorizedMemoryRequest",
            )
        if principal.subject_id != request.subject_id:
            raise D2Error(
                D2RejectionReason.PRINCIPAL_MISMATCH,
                "trusted principal 与 request subject 不一致",
            )
        verified = self.capability_verifier.authorize(
            request,
            now=self.clock(),
        )
        if (
            principal.subject_id != verified.subject_id
            or verified.subject_id != request.subject_id
        ):
            raise D2Error(
                D2RejectionReason.PRINCIPAL_MISMATCH,
                "principal/capability/request subject 三方不一致",
            )
        return memory._retrieve_from_gateway(
            principal,
            verified,
            gateway_binding=self._memory_binding,
        )
