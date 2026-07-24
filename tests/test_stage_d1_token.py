from __future__ import annotations

import base64
import hmac
import json

import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d1_contract import (
    CapabilityClaims,
    CapabilityError,
    RejectionReason,
)
from keyed_gram.stage_d1_token import CapabilityTokenCodec, HmacKeyRing


KEY = b"k" * 32


def _claims(*, key_id: str = "key-1") -> CapabilityClaims:
    return CapabilityClaims(
        subject_id="subject-a",
        entity_scope=("entity-a",),
        relation_id=RelationId.REGISTRY_ID,
        permissions=("retrieve",),
        issued_at=1_800_000_000,
        expires_at=1_800_000_030,
        key_id=key_id,
        nonce="nonce-with-at-least-22-chars",
        policy_version="policy-v1",
    )


def _encode(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _decode(value: bytes) -> bytes:
    return base64.urlsafe_b64decode(
        value + b"=" * ((4 - len(value) % 4) % 4)
    )


def _ring() -> HmacKeyRing:
    return HmacKeyRing({"key-1": KEY}, "key-1")


def test_hmac_round_trip_and_key_material_not_persisted() -> None:
    codec = CapabilityTokenCodec()
    token = codec.issue(_claims(), _ring())
    verified = codec.verify(token, _ring())
    assert verified == _claims()
    assert _ring().key_material_persisted is False
    assert KEY not in token


@pytest.mark.parametrize("segment_index", [0, 1, 2])
def test_any_compact_segment_tamper_fails_closed(segment_index: int) -> None:
    codec = CapabilityTokenCodec()
    token = codec.issue(_claims(), _ring())
    segments = token.split(b".")
    raw = bytearray(_decode(segments[segment_index]))
    raw[-1] ^= 1
    segments[segment_index] = _encode(bytes(raw))
    with pytest.raises(CapabilityError):
        codec.verify(b".".join(segments), _ring())


def test_algorithm_confusion_is_rejected_before_signature_acceptance() -> None:
    codec = CapabilityTokenCodec()
    token = codec.issue(_claims(), _ring())
    header, payload, signature = token.split(b".")
    value = json.loads(_decode(header))
    value["alg"] = "none"
    altered = _encode(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    with pytest.raises(CapabilityError) as error:
        codec.verify(b".".join((altered, payload, signature)), _ring())
    assert error.value.reason is RejectionReason.INVALID_TOKEN


def test_unknown_and_revoked_key_ids_fail_closed() -> None:
    codec = CapabilityTokenCodec()
    foreign_ring = HmacKeyRing({"foreign-key": KEY}, "foreign-key")
    foreign_token = codec.issue(_claims(key_id="foreign-key"), foreign_ring)
    with pytest.raises(CapabilityError) as unknown:
        codec.verify(foreign_token, _ring())
    assert unknown.value.reason is RejectionReason.UNKNOWN_KEY

    ring = _ring()
    token = codec.issue(_claims(), ring)
    ring.revoke("key-1")
    with pytest.raises(CapabilityError) as revoked:
        codec.verify(token, ring)
    assert revoked.value.reason is RejectionReason.REVOKED_KEY


def test_rotation_accepts_old_unrevoked_tokens_and_signs_with_new_key() -> None:
    codec = CapabilityTokenCodec()
    ring = _ring()
    old = codec.issue(_claims(), ring)
    ring.add_and_activate("key-2", b"z" * 32)
    assert codec.verify(old, ring) == _claims()
    new_claims = _claims(key_id="key-2")
    new = codec.issue(new_claims, ring)
    assert codec.verify(new, ring) == new_claims


@pytest.mark.parametrize(
    "token",
    [b"", b"one.two", b"one.two.three.four", b"***.x.y", b"x" * 4097],
)
def test_malformed_or_oversized_token_is_rejected(token: bytes) -> None:
    with pytest.raises(CapabilityError) as error:
        CapabilityTokenCodec().verify(token, _ring())
    assert error.value.reason is RejectionReason.INVALID_TOKEN


def test_duplicate_json_header_key_is_rejected() -> None:
    codec = CapabilityTokenCodec()
    token = codec.issue(_claims(), _ring())
    _, payload, signature = token.split(b".")
    duplicate = _encode(
        b'{"alg":"HS256","alg":"HS256","kid":"key-1",'
        b'"typ":"KG-CAP","v":1}'
    )
    with pytest.raises(CapabilityError) as error:
        codec.verify(b".".join((duplicate, payload, signature)), _ring())
    assert error.value.reason is RejectionReason.INVALID_TOKEN


def test_signed_payload_with_unknown_relation_is_rejected() -> None:
    codec = CapabilityTokenCodec()
    token = codec.issue(_claims(), _ring())
    header, payload, _ = token.split(b".")
    value = json.loads(_decode(payload))
    value["relation_id"] = "not-a-relation"
    altered_payload = _encode(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    signing_input = header + b"." + altered_payload
    signature = _encode(hmac.digest(KEY, signing_input, "sha256"))
    with pytest.raises(CapabilityError) as error:
        codec.verify(
            b".".join((header, altered_payload, signature)),
            _ring(),
        )
    assert error.value.reason is RejectionReason.INVALID_TOKEN


def test_short_hmac_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少需要 32"):
        HmacKeyRing({"key-1": b"k" * 31}, "key-1")
