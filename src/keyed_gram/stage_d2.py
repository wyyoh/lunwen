"""Stage D2：capability-gated authenticated keyed memory 审计。"""

from __future__ import annotations

import csv
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .stage_c24_contract import RelationId
from .stage_d1_contract import (
    AuthorizedMemoryRequest,
    CapabilityClaims,
    CapabilityError,
    CapabilityGrant,
    RejectionReason,
    SuggestedRelation,
)
from .stage_d1_gateway import CapabilityVerifier
from .stage_d1_policy import (
    CapabilityAuthority,
    CapabilityPolicy,
    PolicyRule,
    RevocationReplayStore,
)
from .stage_d1_token import CapabilityTokenCodec, HmacKeyRing
from .stage_d2_contract import (
    D2Error,
    D2RejectionReason,
    EncryptedMemoryRecord,
    PlaintextRelease,
    PublicSyntheticValue,
)
from .stage_d2_crypto import AesGcmRecordCodec, DataEncryptionKeyRing
from .stage_d2_identity import TrustedMockIdentityProvider
from .stage_d2_memory import (
    EncryptedKeyedMemory,
    PrincipalBoundCapabilityGateway,
)
from .stage_d2_protocol import (
    D2ProtocolError,
    canonical_sha256,
    git_state,
    load_config,
    mark_phase_completed,
    mark_phase_started,
    output_paths,
    read_json,
    repo_root,
    require_phase_completed,
    resolve_path,
    runtime_source_manifest,
    sha256_file,
    write_json,
)


FIXED_AUDIT_TIME = 1_800_100_000


@dataclass(slots=True)
class D2Environment:
    now: int
    identity: TrustedMockIdentityProvider
    sessions: dict[str, bytes]
    capability_keyring: HmacKeyRing
    capability_policy: CapabilityPolicy
    replay_store: RevocationReplayStore
    authority: CapabilityAuthority
    gateway: PrincipalBoundCapabilityGateway
    data_keyring: DataEncryptionKeyRing
    record_codec: AesGcmRecordCodec
    memory: EncryptedKeyedMemory


def _policy(version: str, *, max_ttl: int) -> CapabilityPolicy:
    return CapabilityPolicy(
        version,
        (
            PolicyRule(
                "subject-alpha",
                ("entity-a", "entity-d"),
                (RelationId.REGISTRY_ID.value,),
            ),
            PolicyRule(
                "subject-alpha",
                ("entity-a",),
                (RelationId.CITY_CODE.value,),
            ),
            PolicyRule(
                "subject-alpha",
                ("entity-b", "entity-e"),
                (RelationId.CITY_CODE.value,),
            ),
            PolicyRule(
                "subject-alpha",
                ("entity-c",),
                (
                    RelationId.ACCESS_CODE.value,
                    RelationId.REGISTRY_ID.value,
                ),
            ),
            PolicyRule(
                "subject-beta",
                ("entity-a",),
                (RelationId.REGISTRY_ID.value,),
            ),
        ),
        max_ttl_seconds=max_ttl,
        single_use=True,
    )


def _base_records(
    codec: AesGcmRecordCodec,
    keyring: DataEncryptionKeyRing,
) -> list[EncryptedMemoryRecord]:
    declarations = (
        ("record-a-registry", "entity-a", RelationId.REGISTRY_ID),
        ("record-b-city", "entity-b", RelationId.CITY_CODE),
        ("record-c-access", "entity-c", RelationId.ACCESS_CODE),
        ("record-d-registry", "entity-d", RelationId.REGISTRY_ID),
    )
    return [
        codec.encrypt(
            PublicSyntheticValue(secrets.token_bytes(32)),
            record_id=record_id,
            entity_id=entity_id,
            relation_id=relation_id,
            record_version=1,
            keyring=keyring,
        )
        for record_id, entity_id, relation_id in declarations
    ]


def _environment(
    values: Mapping[str, Any],
    *,
    now: int = FIXED_AUDIT_TIME,
) -> D2Environment:
    capability = values["capability"]
    identity = TrustedMockIdentityProvider()
    sessions = {
        "subject-alpha": identity.create_session("subject-alpha"),
        "subject-beta": identity.create_session("subject-beta"),
    }
    capability_keyring = HmacKeyRing.generate("cap-key-1", key_bytes=32)
    policy = _policy(
        str(capability["policy_version"]),
        max_ttl=int(capability["max_ttl_seconds"]),
    )
    token_codec = CapabilityTokenCodec()
    replay_store = RevocationReplayStore()
    authority = CapabilityAuthority(
        capability_keyring,
        policy,
        token_codec,
    )
    verifier = CapabilityVerifier(
        capability_keyring,
        policy,
        token_codec,
        replay_store,
    )
    gateway_binding = object()
    gateway = PrincipalBoundCapabilityGateway(
        identity,
        verifier,
        lambda: now,
        gateway_binding,
    )
    data_keyring = DataEncryptionKeyRing.generate("data-key-1")
    record_codec = AesGcmRecordCodec()
    memory = EncryptedKeyedMemory.from_records(
        record_codec,
        data_keyring,
        _base_records(record_codec, data_keyring),
        gateway_binding=gateway_binding,
    )
    # 两个 key domain 独立生成，并拒绝偶然相同的 key material。
    capability_material = set(capability_keyring._keys.values())
    data_material = set(data_keyring._keys.values())
    if capability_material & data_material:
        raise AssertionError("capability key 与 data key material 意外相同")
    return D2Environment(
        now,
        identity,
        sessions,
        capability_keyring,
        policy,
        replay_store,
        authority,
        gateway,
        data_keyring,
        record_codec,
        memory,
    )


def _grant(
    relation_id: RelationId,
    entity_scope: tuple[str, ...],
    *,
    subject_id: str = "subject-alpha",
    permissions: tuple[str, ...] = ("retrieve",),
) -> CapabilityGrant:
    return CapabilityGrant(
        subject_id,
        entity_scope,
        relation_id,
        permissions,
    )


def _issued_request(
    environment: D2Environment,
    relation_id: RelationId,
    entity_id: str,
    *,
    subject_id: str = "subject-alpha",
    entity_scope: tuple[str, ...] | None = None,
    issue_now: int | None = None,
    ttl_seconds: int = 60,
) -> tuple[Any, AuthorizedMemoryRequest]:
    issued = environment.authority.issue(
        _grant(
            relation_id,
            entity_scope or (entity_id,),
            subject_id=subject_id,
        ),
        now=environment.now if issue_now is None else issue_now,
        ttl_seconds=ttl_seconds,
    )
    return issued, AuthorizedMemoryRequest(
        subject_id=subject_id,
        entity_id=entity_id,
        relation_id=relation_id,
        capability_token=issued.token,
    )


def _reason_value(error: BaseException) -> str:
    if isinstance(error, (CapabilityError, D2Error)):
        return error.reason.value
    raise error


def _record_case(
    rows: list[dict[str, Any]],
    *,
    scenario_id: str,
    category: str,
    expected_status: str,
    expected_reason: str = "",
    memory: EncryptedKeyedMemory,
    action: Callable[[], Any],
    expected_lookup_delta: int = 0,
    expected_decrypt_delta: int = 0,
    expected_release_delta: int = 0,
    expected_migration_delta: int = 0,
) -> None:
    before = memory.snapshot()
    observed_status = "accept"
    observed_reason = ""
    value_digest = ""
    record_id = ""
    try:
        result = action()
        if isinstance(result, PlaintextRelease):
            value_digest = result.value_digest
            record_id = result.record_id
        elif expected_status == "accept":
            raise AssertionError("D2 正向场景未返回 PlaintextRelease")
    except (CapabilityError, D2Error) as exc:
        observed_status = "reject"
        observed_reason = _reason_value(exc)
    after = memory.snapshot()
    deltas = after.delta(before)
    passed = (
        observed_status == expected_status
        and observed_reason == expected_reason
        and deltas["memory_lookup_delta"] == expected_lookup_delta
        and deltas["decrypt_attempt_delta"] == expected_decrypt_delta
        and deltas["plaintext_release_delta"] == expected_release_delta
        and deltas["successful_record_migration_delta"]
        == expected_migration_delta
    )
    rows.append(
        {
            "scenario_id": scenario_id,
            "category": category,
            "expected_status": expected_status,
            "expected_reason": expected_reason,
            "observed_status": observed_status,
            "observed_reason": observed_reason,
            **deltas,
            "value_released": bool(deltas["plaintext_release_delta"]),
            "value_digest": value_digest,
            "record_id": record_id,
            "passed": passed,
        }
    )


def _retrieve(
    environment: D2Environment,
    request: AuthorizedMemoryRequest,
    *,
    subject_id: str = "subject-alpha",
    credential: bytes | None = None,
    memory: EncryptedKeyedMemory | None = None,
) -> PlaintextRelease:
    return environment.gateway.retrieve(
        (
            environment.sessions[subject_id]
            if credential is None
            else credential
        ),
        request,
        memory or environment.memory,
    )


def _single_record_memory(
    environment: D2Environment,
    record: EncryptedMemoryRecord,
    *,
    minimum_version: int | None = None,
) -> EncryptedKeyedMemory:
    minimum = (
        {record.record_id: minimum_version}
        if minimum_version is not None
        else None
    )
    return EncryptedKeyedMemory.from_records(
        environment.record_codec,
        environment.data_keyring,
        (record,),
        gateway_binding=environment.gateway._memory_binding,
        minimum_versions=minimum,
    )


def _record_for(
    environment: D2Environment,
    record_id: str,
) -> EncryptedMemoryRecord:
    return environment.memory._records[record_id]


def _run_positive_scenarios(
    values: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    for scenario, relation, entity in (
        ("valid_registry_release", RelationId.REGISTRY_ID, "entity-a"),
        ("valid_city_release", RelationId.CITY_CODE, "entity-b"),
        ("valid_access_release", RelationId.ACCESS_CODE, "entity-c"),
    ):
        current = _environment(values)
        _, request = _issued_request(current, relation, entity)
        _record_case(
            rows,
            scenario_id=scenario,
            category="positive",
            expected_status="accept",
            memory=current.memory,
            action=lambda c=current, r=request: _retrieve(c, r),
            expected_lookup_delta=1,
            expected_decrypt_delta=1,
            expected_release_delta=1,
        )

    new_key = _environment(values)
    new_key.data_keyring.rotate("data-key-2")
    new_record = new_key.record_codec.encrypt(
        PublicSyntheticValue(secrets.token_bytes(32)),
        record_id="record-e-city",
        entity_id="entity-e",
        relation_id=RelationId.CITY_CODE,
        record_version=1,
        keyring=new_key.data_keyring,
    )
    new_key.memory.add_record(new_record)
    _, new_request = _issued_request(
        new_key,
        RelationId.CITY_CODE,
        "entity-e",
    )
    _record_case(
        rows,
        scenario_id="new_active_data_key_record",
        category="key_rotation",
        expected_status="accept",
        memory=new_key.memory,
        action=lambda: _retrieve(new_key, new_request),
        expected_lookup_delta=1,
        expected_decrypt_delta=1,
        expected_release_delta=1,
    )

    retained = _environment(values)
    retained.data_keyring.rotate("data-key-2")
    _, retained_request = _issued_request(
        retained,
        RelationId.REGISTRY_ID,
        "entity-a",
    )
    _record_case(
        rows,
        scenario_id="retained_old_data_key_record",
        category="key_rotation",
        expected_status="accept",
        memory=retained.memory,
        action=lambda: _retrieve(retained, retained_request),
        expected_lookup_delta=1,
        expected_decrypt_delta=1,
        expected_release_delta=1,
    )

    migrated = _environment(values)
    migrated.data_keyring.rotate("data-key-2")
    _, migrated_request = _issued_request(
        migrated,
        RelationId.REGISTRY_ID,
        "entity-a",
    )

    def migrate_and_read() -> PlaintextRelease:
        migrated.memory.migrate_record_to_active_key("record-a-registry")
        return _retrieve(migrated, migrated_request)

    _record_case(
        rows,
        scenario_id="migrated_record",
        category="record_migration",
        expected_status="accept",
        memory=migrated.memory,
        action=migrate_and_read,
        expected_lookup_delta=1,
        expected_decrypt_delta=1,
        expected_release_delta=1,
        expected_migration_delta=1,
    )

    for scenario, entity in (
        ("multi_scope_entity_a", "entity-a"),
        ("multi_scope_entity_d", "entity-d"),
    ):
        current = _environment(values)
        _, request = _issued_request(
            current,
            RelationId.REGISTRY_ID,
            entity,
            entity_scope=("entity-a", "entity-d"),
        )
        _record_case(
            rows,
            scenario_id=scenario,
            category="multi_entity_scope",
            expected_status="accept",
            memory=current.memory,
            action=lambda c=current, r=request: _retrieve(c, r),
            expected_lookup_delta=1,
            expected_decrypt_delta=1,
            expected_release_delta=1,
        )

    first = _environment(values)
    _, first_request = _issued_request(
        first,
        RelationId.REGISTRY_ID,
        "entity-a",
    )
    _record_case(
        rows,
        scenario_id="single_use_first_release",
        category="single_use",
        expected_status="accept",
        memory=first.memory,
        action=lambda: _retrieve(first, first_request),
        expected_lookup_delta=1,
        expected_decrypt_delta=1,
        expected_release_delta=1,
    )


def _run_authorization_negatives(
    values: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    def reject(
        scenario: str,
        reason: str,
        current: D2Environment,
        action: Callable[[], Any],
    ) -> None:
        _record_case(
            rows,
            scenario_id=scenario,
            category="authorization_negative",
            expected_status="reject",
            expected_reason=reason,
            memory=current.memory,
            action=action,
        )

    invalid = _environment(values)
    _, invalid_request = _issued_request(
        invalid, RelationId.REGISTRY_ID, "entity-a"
    )
    reject(
        "invalid_session",
        D2RejectionReason.INVALID_SESSION.value,
        invalid,
        lambda: _retrieve(
            invalid,
            invalid_request,
            credential=b"x" * 32,
        ),
    )

    principal = _environment(values)
    _, principal_request = _issued_request(
        principal, RelationId.REGISTRY_ID, "entity-a"
    )
    reject(
        "wrong_principal",
        D2RejectionReason.PRINCIPAL_MISMATCH.value,
        principal,
        lambda: _retrieve(
            principal,
            principal_request,
            subject_id="subject-beta",
        ),
    )

    subject = _environment(values)
    beta_issued, _ = _issued_request(
        subject,
        RelationId.REGISTRY_ID,
        "entity-a",
        subject_id="subject-beta",
    )
    subject_request = AuthorizedMemoryRequest(
        "subject-alpha",
        "entity-a",
        RelationId.REGISTRY_ID,
        beta_issued.token,
    )
    reject(
        "wrong_subject",
        RejectionReason.SUBJECT_MISMATCH.value,
        subject,
        lambda: _retrieve(subject, subject_request),
    )

    wrong_entity = _environment(values)
    issued, _ = _issued_request(
        wrong_entity, RelationId.REGISTRY_ID, "entity-a"
    )
    entity_request = AuthorizedMemoryRequest(
        "subject-alpha",
        "entity-d",
        RelationId.REGISTRY_ID,
        issued.token,
    )
    reject(
        "wrong_entity",
        RejectionReason.ENTITY_SCOPE_MISMATCH.value,
        wrong_entity,
        lambda: _retrieve(wrong_entity, entity_request),
    )

    wrong_relation = _environment(values)
    issued, _ = _issued_request(
        wrong_relation, RelationId.REGISTRY_ID, "entity-a"
    )
    relation_request = AuthorizedMemoryRequest(
        "subject-alpha",
        "entity-a",
        RelationId.CITY_CODE,
        issued.token,
    )
    reject(
        "wrong_relation",
        RejectionReason.RELATION_MISMATCH.value,
        wrong_relation,
        lambda: _retrieve(wrong_relation, relation_request),
    )

    permission = _environment(values)
    claims = CapabilityClaims(
        subject_id="subject-alpha",
        entity_scope=("entity-a",),
        relation_id=RelationId.REGISTRY_ID,
        permissions=("inspect",),
        issued_at=permission.now,
        expires_at=permission.now + 60,
        key_id=permission.capability_keyring.active_key_id,
        nonce=secrets.token_hex(32),
        policy_version=permission.capability_policy.version,
    )
    token = permission.authority.codec.issue(
        claims,
        permission.capability_keyring,
    )
    permission_request = AuthorizedMemoryRequest(
        "subject-alpha",
        "entity-a",
        RelationId.REGISTRY_ID,
        token,
    )
    reject(
        "insufficient_permission",
        RejectionReason.PERMISSION_DENIED.value,
        permission,
        lambda: _retrieve(permission, permission_request),
    )

    expired = _environment(values)
    _, expired_request = _issued_request(
        expired,
        RelationId.REGISTRY_ID,
        "entity-a",
        issue_now=expired.now - 100,
        ttl_seconds=30,
    )
    reject(
        "expired_capability",
        RejectionReason.EXPIRED.value,
        expired,
        lambda: _retrieve(expired, expired_request),
    )

    future = _environment(values)
    _, future_request = _issued_request(
        future,
        RelationId.REGISTRY_ID,
        "entity-a",
        issue_now=future.now + 10,
    )
    reject(
        "not_yet_valid_capability",
        RejectionReason.NOT_YET_VALID.value,
        future,
        lambda: _retrieve(future, future_request),
    )

    revoked = _environment(values)
    revoked_issued, revoked_request = _issued_request(
        revoked, RelationId.REGISTRY_ID, "entity-a"
    )
    revoked.replay_store.revoke(revoked_issued.claims.nonce)
    reject(
        "revoked_capability",
        RejectionReason.REVOKED_TOKEN.value,
        revoked,
        lambda: _retrieve(revoked, revoked_request),
    )

    replayed = _environment(values)
    _, replay_request = _issued_request(
        replayed, RelationId.REGISTRY_ID, "entity-a"
    )
    _retrieve(replayed, replay_request)
    reject(
        "replayed_capability",
        RejectionReason.REPLAY.value,
        replayed,
        lambda: _retrieve(replayed, replay_request),
    )

    signing = _environment(values)
    _, signing_request = _issued_request(
        signing, RelationId.REGISTRY_ID, "entity-a"
    )
    signing.capability_keyring.revoke("cap-key-1")
    reject(
        "revoked_signing_key",
        RejectionReason.REVOKED_KEY.value,
        signing,
        lambda: _retrieve(signing, signing_request),
    )

    proposal = _environment(values)
    reject(
        "unauthorized_router_proposal",
        RejectionReason.ROUTER_PROPOSAL_NOT_AUTHORIZED.value,
        proposal,
        lambda: proposal.gateway.retrieve(
            proposal.sessions["subject-alpha"],
            SuggestedRelation(RelationId.REGISTRY_ID),
            proposal.memory,
        ),
    )

    forged = _environment(values)
    trusted_principal = forged.identity.authenticate(
        forged.sessions["subject-alpha"]
    )
    reject(
        "forged_verified_capability",
        D2RejectionReason.FORGED_VERIFIED_CAPABILITY.value,
        forged,
        lambda: forged.memory.retrieve_authorized(
            trusted_principal,
            object(),
        ),
    )


def _flip(value: bytes, index: int) -> bytes:
    altered = bytearray(value)
    altered[index] ^= 1
    return bytes(altered)


def _run_record_negatives(
    values: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    def authenticated_record_case(
        *,
        scenario: str,
        current: D2Environment,
        record: EncryptedMemoryRecord,
        request_entity: str,
        request_relation: RelationId,
        reason: D2RejectionReason,
        minimum_version: int | None = None,
        expected_decrypt_delta: int = 1,
    ) -> None:
        memory = _single_record_memory(
            current,
            record,
            minimum_version=minimum_version,
        )
        _, request = _issued_request(
            current,
            request_relation,
            request_entity,
        )
        _record_case(
            rows,
            scenario_id=scenario,
            category="record_negative",
            expected_status="reject",
            expected_reason=reason.value,
            memory=memory,
            action=lambda: _retrieve(current, request, memory=memory),
            expected_lookup_delta=1,
            expected_decrypt_delta=expected_decrypt_delta,
        )

    unknown = _environment(values)
    unknown_record = replace(
        _record_for(unknown, "record-a-registry"),
        data_key_id="data-unknown",
    )
    authenticated_record_case(
        scenario="unknown_data_key",
        current=unknown,
        record=unknown_record,
        request_entity="entity-a",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.UNKNOWN_DATA_KEY,
    )

    revoked = _environment(values)
    revoked.data_keyring.revoke("data-key-1")
    authenticated_record_case(
        scenario="revoked_data_key",
        current=revoked,
        record=_record_for(revoked, "record-a-registry"),
        request_entity="entity-a",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.REVOKED_DATA_KEY,
    )

    wrong = _environment(values)
    wrong.data_keyring.rotate("data-key-2")
    wrong_record = replace(
        _record_for(wrong, "record-a-registry"),
        data_key_id="data-key-2",
    )
    authenticated_record_case(
        scenario="wrong_data_key",
        current=wrong,
        record=wrong_record,
        request_entity="entity-a",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.AUTHENTICATION_FAILED,
    )

    for scenario, mutate in (
        (
            "ciphertext_bit_flip",
            lambda record: replace(
                record,
                ciphertext=_flip(record.ciphertext, 0),
            ),
        ),
        (
            "authentication_tag_tamper",
            lambda record: replace(
                record,
                ciphertext=_flip(record.ciphertext, -1),
            ),
        ),
        (
            "nonce_tamper",
            lambda record: replace(
                record,
                nonce=_flip(record.nonce, 0),
            ),
        ),
        (
            "aad_record_id_tamper",
            lambda record: replace(record, record_id="record-a-tampered"),
        ),
    ):
        current = _environment(values)
        record = mutate(_record_for(current, "record-a-registry"))
        authenticated_record_case(
            scenario=scenario,
            current=current,
            record=record,
            request_entity="entity-a",
            request_relation=RelationId.REGISTRY_ID,
            reason=D2RejectionReason.AUTHENTICATION_FAILED,
        )

    entity = _environment(values)
    entity_record = replace(
        _record_for(entity, "record-a-registry"),
        entity_id="entity-d",
    )
    authenticated_record_case(
        scenario="entity_metadata_swap",
        current=entity,
        record=entity_record,
        request_entity="entity-d",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.AUTHENTICATION_FAILED,
    )

    relation = _environment(values)
    relation_record = replace(
        _record_for(relation, "record-c-access"),
        relation_id=RelationId.REGISTRY_ID,
    )
    authenticated_record_case(
        scenario="relation_metadata_swap",
        current=relation,
        record=relation_record,
        request_entity="entity-c",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.AUTHENTICATION_FAILED,
    )

    swapped = _environment(values)
    target = _record_for(swapped, "record-a-registry")
    donor = _record_for(swapped, "record-d-registry")
    swapped_record = replace(
        target,
        nonce=donor.nonce,
        ciphertext=donor.ciphertext,
    )
    authenticated_record_case(
        scenario="cross_bucket_ciphertext_swap",
        current=swapped,
        record=swapped_record,
        request_entity="entity-a",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.AUTHENTICATION_FAILED,
    )

    rollback = _environment(values)
    old_record = _record_for(rollback, "record-a-registry")
    authenticated_record_case(
        scenario="record_version_rollback",
        current=rollback,
        record=old_record,
        request_entity="entity-a",
        request_relation=RelationId.REGISTRY_ID,
        reason=D2RejectionReason.RECORD_VERSION_ROLLBACK,
        minimum_version=2,
        expected_decrypt_delta=0,
    )

    for scenario, factory in (
        (
            "algorithm_confusion",
            lambda record: replace(record, algorithm="AES-CTR"),
        ),
        (
            "malformed_record",
            lambda record: replace(record, nonce=b"short"),
        ),
    ):
        current = _environment(values)
        _record_case(
            rows,
            scenario_id=scenario,
            category="record_schema_negative",
            expected_status="reject",
            expected_reason=D2RejectionReason.INVALID_RECORD.value,
            memory=current.memory,
            action=lambda c=current, f=factory: f(
                _record_for(c, "record-a-registry")
            ),
        )

    duplicate = _environment(values)
    duplicate_record = _record_for(duplicate, "record-a-registry")
    empty = EncryptedKeyedMemory(
        duplicate.record_codec,
        duplicate.data_keyring,
        duplicate.gateway._memory_binding,
    )

    def add_duplicate() -> Any:
        empty.add_record(duplicate_record)
        return empty.add_record(duplicate_record)

    _record_case(
        rows,
        scenario_id="duplicate_record_id",
        category="record_schema_negative",
        expected_status="reject",
        expected_reason=D2RejectionReason.DUPLICATE_RECORD_ID.value,
        memory=empty,
        action=add_duplicate,
    )


def _run_concurrency(
    values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    current = _environment(values)
    _, request = _issued_request(
        current,
        RelationId.REGISTRY_ID,
        "entity-a",
    )
    before = current.memory.snapshot()

    def attempt() -> tuple[str, str]:
        try:
            release = _retrieve(current, request)
        except (CapabilityError, D2Error) as exc:
            return "reject", _reason_value(exc)
        return "accept", release.value_digest

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: attempt(), range(2)))
    after = current.memory.snapshot()
    deltas = after.delta(before)
    successful = sum(status == "accept" for status, _ in outcomes)
    replay = sum(
        status == "reject" and value == RejectionReason.REPLAY.value
        for status, value in outcomes
    )
    double_release = max(0, deltas["plaintext_release_delta"] - 1)
    row = {
        "scenario_id": "concurrent_single_use_capability",
        "category": "concurrency",
        "successful_release_count": successful,
        "replay_rejection_count": replay,
        "concurrent_double_release_count": double_release,
        **deltas,
        "value_released": successful == 1,
        "value_digest_count": successful,
        "plaintext_recorded": False,
        "passed": (
            successful == 1
            and replay == 1
            and double_release == 0
            and deltas["memory_lookup_delta"] == 1
            and deltas["decrypt_attempt_delta"] == 1
            and deltas["plaintext_release_delta"] == 1
        ),
    }
    return row, {
        "successful_release_count": successful,
        "replay_rejection_count": replay,
        "concurrent_double_release_count": double_release,
    }


def _run_attack_matrix(
    values: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _run_positive_scenarios(values, rows)
    _run_authorization_negatives(values, rows)
    _run_record_negatives(values, rows)
    concurrency_row, concurrency = _run_concurrency(values)
    rows.append(concurrency_row)
    return rows, concurrency


def _summary(
    values: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    concurrency: Mapping[str, int],
    *,
    git: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected_positive = set(values["required_positive_scenarios"])
    expected_authorization = set(
        values["required_authorization_negative_scenarios"]
    )
    expected_record = set(values["required_record_negative_scenarios"])
    observed_positive = {
        row["scenario_id"]
        for row in rows
        if row["category"]
        in {
            "positive",
            "key_rotation",
            "record_migration",
            "multi_entity_scope",
            "single_use",
        }
    }
    observed_authorization = {
        row["scenario_id"]
        for row in rows
        if row["category"] == "authorization_negative"
    }
    observed_record = {
        row["scenario_id"]
        for row in rows
        if row["category"]
        in {"record_negative", "record_schema_negative"}
    }
    if (
        observed_positive != expected_positive
        or observed_authorization != expected_authorization
        or observed_record != expected_record
    ):
        raise D2ProtocolError("D2 attack matrix 与预注册场景不一致")
    authorization_rows = [
        row for row in rows if row["category"] == "authorization_negative"
    ]
    record_rows = [
        row
        for row in rows
        if row["category"] in {"record_negative", "record_schema_negative"}
    ]
    positive_rows = [
        row
        for row in rows
        if row["scenario_id"] in expected_positive
    ]
    cross_relation_ids = {"wrong_relation", "relation_metadata_swap"}
    cross_entity_ids = {"wrong_entity", "entity_metadata_swap"}
    cross_subject_ids = {"wrong_principal", "wrong_subject"}
    tampered_ids = {
        "ciphertext_bit_flip",
        "authentication_tag_tamper",
        "nonce_tamper",
        "aad_record_id_tamper",
        "entity_metadata_swap",
        "relation_metadata_swap",
        "cross_bucket_ciphertext_swap",
    }
    successful_rotation = int(
        all(
            next(
                bool(row["passed"])
                for row in rows
                if row["scenario_id"] == scenario
            )
            for scenario in (
                "new_active_data_key_record",
                "retained_old_data_key_record",
            )
        )
    )
    metrics = {
        "positive_scenario_count": len(positive_rows),
        "authorization_negative_scenario_count": len(authorization_rows),
        "record_negative_scenario_count": len(record_rows),
        "total_scenario_count": len(rows),
        "passed_scenario_count": sum(bool(row["passed"]) for row in rows),
        "failed_scenario_count": sum(not bool(row["passed"]) for row in rows),
        "authorized_plaintext_release_count": (
            sum(int(row["plaintext_release_delta"]) for row in positive_rows)
            + int(concurrency["successful_release_count"])
        ),
        "unauthorized_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in authorization_rows + record_rows
        ),
        "unauthorized_memory_lookup_count": sum(
            int(row["memory_lookup_delta"]) for row in authorization_rows
        ),
        "unauthorized_decrypt_attempt_count": sum(
            int(row["decrypt_attempt_delta"]) for row in authorization_rows
        ),
        "cross_relation_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"] in cross_relation_ids
        ),
        "cross_entity_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"] in cross_entity_ids
        ),
        "cross_subject_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"] in cross_subject_ids
        ),
        "tampered_record_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"] in tampered_ids
        ),
        "wrong_key_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"] in {"unknown_data_key", "wrong_data_key"}
        ),
        "revoked_key_plaintext_release_count": sum(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"]
            in {"revoked_data_key", "revoked_signing_key"}
        ),
        "replayed_token_plaintext_release_count": next(
            int(row["plaintext_release_delta"])
            for row in rows
            if row["scenario_id"] == "replayed_capability"
        ),
        "concurrent_double_release_count": int(
            concurrency["concurrent_double_release_count"]
        ),
        "successful_key_rotation_count": successful_rotation,
        "successful_record_migration_count": sum(
            int(row["successful_record_migration_delta"])
            for row in rows
        ),
    }
    zero_gates = values["required_zero_metrics"]
    success_gates = values["required_success_metrics"]
    concurrency_gate = values["required_concurrency_scenario"]
    gates_passed = (
        metrics["failed_scenario_count"] == 0
        and all(
            metrics[key] == int(expected)
            for key, expected in zero_gates.items()
        )
        and all(
            metrics[key] == int(expected)
            for key, expected in success_gates.items()
        )
        and all(
            concurrency[key] == int(expected)
            for key, expected in concurrency_gate.items()
            if key != "scenario_id"
        )
    )
    return {
        "schema_version": 1,
        "stage": "D2-capability-gated-authenticated-keyed-memory-audit",
        "git": dict(git),
        "source_manifest_payload_sha256": source_manifest[
            "manifest_payload_sha256"
        ],
        "cryptographic_primitives": {
            "capability_authentication": "HMAC-SHA-256",
            "data_encryption": "AES-256-GCM",
            "data_encryption_standard": "NIST SP 800-38D",
            "implementation": "cryptography AESGCM",
            "implementation_version": values["data_memory"][
                "implementation_version"
            ],
            "nonce_bytes": 12,
            "tag_bytes": 16,
            "capability_key_equals_data_key": False,
            "new_cryptographic_algorithm_claimed": False,
        },
        "metrics": metrics,
        "concurrency": dict(concurrency),
        "capability_gated_keyed_memory_status": (
            "passed" if gates_passed else "failed"
        ),
        "principal_binding_passed": gates_passed,
        "capability_before_lookup_passed": gates_passed,
        "capability_before_decrypt_passed": gates_passed,
        "aead_metadata_binding_passed": gates_passed,
        "key_separation_passed": gates_passed,
        "atomic_single_use_release_passed": gates_passed,
        "ready_for_d2_1_public_generation_probe": gates_passed,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "semantic_router_in_tcb": False,
        "private_answers_loaded": False,
        "private_value_memory_trained": False,
        "lm_answer_injection_executed": False,
        "confirmation_created_or_read": False,
        "key_attack_executed": False,
        "plaintext_persisted_in_artifacts": False,
        "capability_token_persisted": False,
        "capability_hmac_key_persisted": False,
        "data_encryption_key_persisted": False,
        "real_identity_provider_used": False,
        "limitations": [
            "AuthenticatedPrincipal 来自进程内 mock identity provider，不是真实 IAM",
            "调用方与 gateway/memory 尚未实现进程级隔离",
            "HMAC authority/verifier 位于同一 TCB，未实现签发与验证密码学隔离",
            "replay/revocation store 仅为单进程内存状态，不覆盖多实例和崩溃恢复",
            "固定测试时钟不覆盖真实分布式 clock skew",
            "AES-GCM nonce 唯一性仅由单进程 key ring 原子集合维护",
            "Python bytes 不提供可靠的 plaintext/key 内存清零保证",
            "只使用运行期随机 synthetic bytes，不包含 private value 或 private answer",
            "本结果不是部署级安全认证，也不证明机器遗忘",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise D2ProtocolError("拒绝写入空 D2 attack matrix")
    fields = sorted({str(key) for row in rows for key in row})
    forbidden = {
        "plaintext",
        "value",
        "capability_token",
        "capability_hmac_key",
        "data_encryption_key",
        "session_credential",
    }
    if forbidden & set(fields):
        raise D2ProtocolError("D2 CSV schema 包含敏感字段")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\r\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    temporary.replace(path)


def run_d2_audit(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir = output_paths(config_path)
    if output_dir is not None:
        artifact_dir = resolve_path(config_path, output_dir)
    if artifact_dir.exists() or runtime_dir.exists():
        raise D2ProtocolError("D2 输出或 runtime state 已存在，拒绝重跑")
    git = git_state(config_path)
    if git["tracked_dirty"]:
        raise D2ProtocolError("D2 正式 audit 要求 tracked worktree 干净")
    sources = runtime_source_manifest(config_path)
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    mark_phase_started(runtime_dir, "audit")
    rows, concurrency = _run_attack_matrix(values)
    summary = _summary(
        values,
        rows,
        concurrency,
        git=git,
        source_manifest=sources,
    )
    summary_path = artifact_dir / "stage_d2_summary.json"
    matrix_path = artifact_dir / "capability_memory_attack_matrix.csv"
    source_path = artifact_dir / "source_sha256_manifest.json"
    resolved_path = artifact_dir / "resolved_config.json"
    status_path = artifact_dir / "protocol_status.json"
    write_json(summary_path, summary)
    _write_csv(matrix_path, rows)
    write_json(source_path, sources)
    write_json(
        resolved_path,
        {
            "schema_version": 1,
            "config": values,
            "config_sha256": sha256_file(config_path),
            "plaintext_or_secret_material_present": False,
        },
    )
    write_json(
        status_path,
        {
            key: summary[key]
            for key in (
                "capability_gated_keyed_memory_status",
                "ready_for_d2_1_public_generation_probe",
                "private_value_memory_ready",
                "original_c3_allowed",
                "c3_eligible",
                "semantic_router_in_tcb",
                "private_answers_loaded",
                "private_value_memory_trained",
                "lm_answer_injection_executed",
                "confirmation_created_or_read",
                "key_attack_executed",
                "plaintext_persisted_in_artifacts",
                "capability_token_persisted",
                "capability_hmac_key_persisted",
                "data_encryption_key_persisted",
            )
        },
    )
    mark_phase_completed(
        runtime_dir,
        "audit",
        (summary_path, matrix_path, source_path, resolved_path, status_path),
    )
    return {
        "status": summary["capability_gated_keyed_memory_status"],
        "metrics": summary["metrics"],
        "concurrency": summary["concurrency"],
        "ready_for_d2_1_public_generation_probe": summary[
            "ready_for_d2_1_public_generation_probe"
        ],
        "c3_eligible": False,
        "summary": str(summary_path),
    }


def finalize_d2_artifacts(config_path: str | Path) -> dict[str, Any]:
    artifact_dir, runtime_dir = output_paths(config_path)
    require_phase_completed(runtime_dir, "audit")
    target = artifact_dir / "artifact_sha256_manifest.json"
    if target.exists():
        raise D2ProtocolError("D2 artifact manifest 已存在，拒绝覆盖")
    report = repo_root(config_path) / "PHASE_D2_REPORT.md"
    if not report.is_file():
        raise D2ProtocolError("PHASE_D2_REPORT.md 尚未生成")
    files = [
        {
            "path": path.relative_to(repo_root(config_path)).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path != target
    ]
    files.append(
        {
            "path": "PHASE_D2_REPORT.md",
            "size_bytes": report.stat().st_size,
            "sha256": sha256_file(report),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "D2-final-artifact-manifest",
        "files": files,
        "dataset_present": False,
        "plaintext_value_present": False,
        "private_value_present": False,
        "private_answer_present": False,
        "capability_token_present": False,
        "capability_hmac_key_present": False,
        "data_encryption_key_present": False,
        "session_credential_present": False,
        "model_checkpoint_present": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    write_json(target, payload)
    return {
        "artifact_manifest": str(target),
        "file_count": len(files),
        "manifest_file_sha256": sha256_file(target),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
    }


def run_d2_smoke(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = load_config(config_path, smoke=True)
    current = _environment(values)
    _, request = _issued_request(
        current,
        RelationId.REGISTRY_ID,
        "entity-a",
    )
    release = _retrieve(current, request)
    before = current.memory.snapshot()
    try:
        _retrieve(current, request)
    except CapabilityError as exc:
        replay_reason = exc.reason.value
    else:
        raise AssertionError("D2 smoke replay 未被拒绝")
    after = current.memory.snapshot()
    result = {
        "schema_version": 1,
        "stage": "D2-in-memory-smoke",
        "authorized_release_count": 1,
        "value_released": release.value_released,
        "value_digest": release.value_digest,
        "replay_reason": replay_reason,
        "replay_plaintext_release_count": (
            after.plaintext_release_count - before.plaintext_release_count
        ),
        "plaintext_persisted": False,
        "private_value_used": False,
        "capability_key_equals_data_key": False,
        "c3_eligible": False,
    }
    if output_dir is not None:
        destination = resolve_path(config_path, output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "smoke_summary.json", result)
    return result
