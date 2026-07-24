from __future__ import annotations

from dataclasses import replace

import cryptography
import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d2_contract import (
    D2Error,
    D2RejectionReason,
    PublicSyntheticValue,
)
from keyed_gram.stage_d2_crypto import (
    AesGcmRecordCodec,
    DataEncryptionKeyRing,
)


def _encrypted():
    codec = AesGcmRecordCodec()
    keyring = DataEncryptionKeyRing(
        {"data-key-1": b"k" * 32},
        "data-key-1",
    )
    value = PublicSyntheticValue(b"synthetic-public-value-123456")
    record = codec.encrypt(
        value,
        record_id="record-a",
        entity_id="entity-a",
        relation_id=RelationId.REGISTRY_ID,
        record_version=1,
        keyring=keyring,
    )
    return codec, keyring, value, record


def test_cryptography_version_is_frozen() -> None:
    assert cryptography.__version__ == "49.0.0"


def test_aes_256_gcm_round_trip_and_ciphertext_contains_tag() -> None:
    codec, keyring, value, record = _encrypted()
    assert record.algorithm == "AES-256-GCM"
    assert len(record.nonce) == 12
    assert len(record.ciphertext) == len(value.payload) + 16
    assert value.payload not in record.ciphertext
    assert codec.decrypt(record, keyring=keyring) == value


def test_nonce_is_unique_per_key_across_many_records() -> None:
    codec = AesGcmRecordCodec()
    keyring = DataEncryptionKeyRing.generate("data-key-1")
    records = [
        codec.encrypt(
            PublicSyntheticValue(bytes([index]) * 32),
            record_id=f"record-{index}",
            entity_id=f"entity-{index}",
            relation_id=RelationId.REGISTRY_ID,
            record_version=1,
            keyring=keyring,
        )
        for index in range(64)
    ]
    assert len({record.nonce for record in records}) == len(records)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: replace(
            record,
            ciphertext=bytes([record.ciphertext[0] ^ 1])
            + record.ciphertext[1:],
        ),
        lambda record: replace(
            record,
            ciphertext=record.ciphertext[:-1]
            + bytes([record.ciphertext[-1] ^ 1]),
        ),
        lambda record: replace(
            record,
            nonce=bytes([record.nonce[0] ^ 1]) + record.nonce[1:],
        ),
        lambda record: replace(record, entity_id="entity-b"),
        lambda record: replace(record, relation_id=RelationId.CITY_CODE),
        lambda record: replace(record, record_id="record-b"),
        lambda record: replace(record, record_version=2),
    ],
)
def test_ciphertext_nonce_or_aad_tamper_fails_authentication(mutator) -> None:
    codec, keyring, _, record = _encrypted()
    with pytest.raises(D2Error) as error:
        codec.decrypt(mutator(record), keyring=keyring)
    assert error.value.reason is D2RejectionReason.AUTHENTICATION_FAILED


def test_unknown_revoked_and_wrong_data_keys_fail_closed() -> None:
    codec, keyring, _, record = _encrypted()
    unknown = replace(record, data_key_id="data-unknown")
    with pytest.raises(D2Error) as missing:
        codec.decrypt(unknown, keyring=keyring)
    assert missing.value.reason is D2RejectionReason.UNKNOWN_DATA_KEY

    keyring.revoke("data-key-1")
    with pytest.raises(D2Error) as revoked:
        codec.decrypt(record, keyring=keyring)
    assert revoked.value.reason is D2RejectionReason.REVOKED_DATA_KEY

    codec2, keyring2, _, record2 = _encrypted()
    keyring2.rotate("data-key-2")
    wrong = replace(record2, data_key_id="data-key-2")
    with pytest.raises(D2Error) as mismatched:
        codec2.decrypt(wrong, keyring=keyring2)
    assert (
        mismatched.value.reason
        is D2RejectionReason.AUTHENTICATION_FAILED
    )


def test_rotation_keeps_old_key_for_read_and_uses_new_key_for_write() -> None:
    codec, keyring, value, old = _encrypted()
    keyring.rotate("data-key-2")
    assert codec.decrypt(old, keyring=keyring) == value
    new = codec.encrypt(
        PublicSyntheticValue(b"another-synthetic-public-value"),
        record_id="record-b",
        entity_id="entity-b",
        relation_id=RelationId.CITY_CODE,
        record_version=1,
        keyring=keyring,
    )
    assert old.data_key_id == "data-key-1"
    assert new.data_key_id == "data-key-2"


def test_data_key_namespace_size_and_non_persistence() -> None:
    with pytest.raises(ValueError, match="data- namespace"):
        DataEncryptionKeyRing({"cap-key-1": b"k" * 32}, "cap-key-1")
    with pytest.raises(ValueError, match="32 bytes"):
        DataEncryptionKeyRing({"data-key-1": b"k" * 31}, "data-key-1")
    assert (
        DataEncryptionKeyRing.generate(
            "data-key-1"
        ).key_material_persisted
        is False
    )
