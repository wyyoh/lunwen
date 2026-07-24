from __future__ import annotations

from dataclasses import fields, replace
from inspect import signature

import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d2_contract import (
    AuthenticatedPrincipal,
    D2Error,
    D2RejectionReason,
    EncryptedMemoryRecord,
    PlaintextRelease,
    PublicSyntheticValue,
)
from keyed_gram.stage_d2_identity import TrustedMockIdentityProvider
from keyed_gram.stage_d2_memory import (
    EncryptedKeyedMemory,
    PrincipalBoundCapabilityGateway,
)


def _record() -> EncryptedMemoryRecord:
    return EncryptedMemoryRecord(
        record_id="record-a",
        entity_id="entity-a",
        relation_id=RelationId.REGISTRY_ID,
        data_key_id="data-key-1",
        record_version=1,
        nonce=b"n" * 12,
        ciphertext=b"c" * 32,
    )


def test_authenticated_principal_has_required_fields_and_hidden_seal() -> None:
    assert [field.name for field in fields(AuthenticatedPrincipal)] == [
        "subject_id",
        "session_id",
        "authentication_context",
        "_seal",
    ]
    with pytest.raises(TypeError, match="不能由调用者构造"):
        AuthenticatedPrincipal(
            "subject-a",
            "session-a",
            "mock-mfa",
            object(),
        )


def test_mock_identity_provider_returns_sealed_principal() -> None:
    provider = TrustedMockIdentityProvider()
    credential = provider.create_session("subject-a")
    principal = provider.authenticate(credential)
    assert principal.subject_id == "subject-a"
    assert principal.authentication_context == "mock-mfa"
    assert provider.credential_material_persisted is False
    assert repr(credential) not in repr(provider)


@pytest.mark.parametrize("credential", [b"", b"x" * 31, b"x" * 32])
def test_invalid_or_unknown_session_fails_closed(credential: bytes) -> None:
    provider = TrustedMockIdentityProvider()
    with pytest.raises(D2Error) as error:
        provider.authenticate(credential)
    assert error.value.reason is D2RejectionReason.INVALID_SESSION


def test_encrypted_record_contains_no_plaintext_or_value_field() -> None:
    names = {field.name for field in fields(EncryptedMemoryRecord)}
    assert names == {
        "record_id",
        "entity_id",
        "relation_id",
        "data_key_id",
        "record_version",
        "nonce",
        "ciphertext",
        "schema_version",
        "algorithm",
    }
    assert {"plaintext", "value", "private_answer"}.isdisjoint(names)


def test_record_aad_binds_all_frozen_metadata() -> None:
    aad = _record().aad().decode("utf-8")
    for name in (
        "schema_version",
        "algorithm",
        "record_id",
        "entity_id",
        "relation_id",
        "data_key_id",
        "record_version",
    ):
        assert f'"{name}"' in aad


@pytest.mark.parametrize(
    "change",
    [
        {"algorithm": "none"},
        {"schema_version": 2},
        {"record_version": 0},
        {"nonce": b"short"},
        {"ciphertext": b"short"},
    ],
)
def test_invalid_record_schema_fails_closed(change) -> None:
    with pytest.raises(D2Error) as error:
        replace(_record(), **change)
    assert error.value.reason is D2RejectionReason.INVALID_RECORD


def test_synthetic_value_digest_and_repr_do_not_expose_payload() -> None:
    value = PublicSyntheticValue(b"s" * 32)
    assert len(value.digest) == 64
    assert "ssss" not in repr(value)
    release = PlaintextRelease(
        "record-a",
        "entity-a",
        RelationId.REGISTRY_ID,
        value,
        value.digest,
    )
    assert release.value_released is True
    assert "ssss" not in repr(release)


def test_d2_gateway_accepts_credential_not_caller_principal() -> None:
    parameters = set(
        signature(PrincipalBoundCapabilityGateway.retrieve).parameters
    )
    assert parameters == {
        "self",
        "session_credential",
        "request",
        "memory",
    }
    assert "principal" not in parameters


def test_memory_requires_both_principal_and_verified_capability() -> None:
    parameters = set(
        signature(EncryptedKeyedMemory.retrieve_authorized).parameters
    )
    assert parameters == {"self", "principal", "capability"}
