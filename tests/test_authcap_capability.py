from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from keyed_gram.authcap_capability import (
    AuthCapIssuer,
    AuthCapPublicKeyRing,
    AuthCapTokenError,
    AuthCapV1,
)
from keyed_gram.authcap_policy import canonical_sha256


def payload(key_id: str = "key-1") -> AuthCapV1:
    return AuthCapV1(
        capability_id="capability-1",
        issuer_key_id=key_id,
        subject_id="subject-1",
        policy_hash="1" * 64,
        decision_id="2" * 64,
        authorization_witness_digest="3" * 64,
        trusted_request_digest="4" * 64,
        proposal_digest="5" * 64,
        executable_authority_digest="6" * 64,
        policy_epoch=2,
        issued_at="2026-07-01T12:00:00Z",
        expires_at="2026-07-01T12:05:00Z",
        single_use=True,
        nonce="nonce-1",
    )


def test_ed25519_sign_and_public_only_verify():
    issuer = AuthCapIssuer.generate("key-1")
    ring = AuthCapPublicKeyRing({"key-1": issuer.public_key_bytes})
    token = issuer.issue(payload(), maximum_bytes=8192)
    assert ring.verify_signature(token, maximum_bytes=8192) == payload()
    assert "private" not in json.dumps(ring.public_manifest()).casefold()


def test_forged_signature_and_wrong_public_key_are_rejected():
    issuer = AuthCapIssuer.generate("key-1")
    token = issuer.issue(payload(), maximum_bytes=8192)
    wrong = AuthCapPublicKeyRing(
        {
            "key-1": Ed25519PrivateKey.generate().public_key().public_bytes_raw(),
        }
    )
    with pytest.raises(AuthCapTokenError):
        wrong.verify_signature(token, maximum_bytes=8192)


def test_algorithm_confusion_is_rejected():
    issuer = AuthCapIssuer.generate("key-1")
    token = issuer.issue(payload(), maximum_bytes=8192)
    parts = token.split(".")
    header = json.loads(
        base64.urlsafe_b64decode(parts[0] + "=" * (-len(parts[0]) % 4))
    )
    header["alg"] = "none"
    parts[0] = (
        base64.urlsafe_b64encode(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    with pytest.raises(AuthCapTokenError):
        AuthCapPublicKeyRing(
            {"key-1": issuer.public_key_bytes}
        ).verify_signature(".".join(parts), maximum_bytes=8192)


def test_unknown_key_id_and_key_rotation():
    old = AuthCapIssuer.generate("old-key")
    new = AuthCapIssuer.generate("new-key")
    token = old.issue(payload("old-key"), maximum_bytes=8192)
    with pytest.raises(AuthCapTokenError):
        AuthCapPublicKeyRing(
            {"new-key": new.public_key_bytes}
        ).verify_signature(token, maximum_bytes=8192)
    ring = AuthCapPublicKeyRing(
        {"old-key": old.public_key_bytes, "new-key": new.public_key_bytes}
    )
    assert ring.verify_signature(token, maximum_bytes=8192).issuer_key_id == "old-key"


def test_size_limit_and_invalid_lifecycle_fail_closed():
    issuer = AuthCapIssuer.generate("key-1")
    token = issuer.issue(payload(), maximum_bytes=8192)
    with pytest.raises(AuthCapTokenError):
        issuer.issue(payload(), maximum_bytes=100)
    with pytest.raises(AuthCapTokenError):
        AuthCapPublicKeyRing(
            {"key-1": issuer.public_key_bytes}
        ).verify_signature(token, maximum_bytes=100)
    with pytest.raises(AuthCapTokenError):
        AuthCapV1(**{**payload().__dict__, "expires_at": payload().issued_at})


def test_duplicate_json_key_and_unknown_payload_field_rejected():
    issuer = AuthCapIssuer.generate("key-1")
    ring = AuthCapPublicKeyRing({"key-1": issuer.public_key_bytes})
    token = issuer.issue(payload(), maximum_bytes=8192)
    parts = token.split(".")
    raw = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)).decode()
    duplicate = raw[:-1] + ',"nonce":"nonce-2"}'
    parts[1] = base64.urlsafe_b64encode(duplicate.encode()).rstrip(b"=").decode()
    with pytest.raises(AuthCapTokenError):
        ring.verify_signature(".".join(parts), maximum_bytes=8192)


def test_workflow_metadata_requires_consistent_fields():
    with pytest.raises(AuthCapTokenError):
        AuthCapV1(**{**payload().__dict__, "workflow_step": 1})


def test_private_key_runtime_file_round_trip(tmp_path):
    path = tmp_path / "issuer.pem"
    issuer = AuthCapIssuer.generate("key-1")
    issuer.write_private_pem(path)
    loaded = AuthCapIssuer.load_private_pem("key-1", path)
    assert loaded.public_key_bytes == issuer.public_key_bytes
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_public_verifier_manifest_hash_and_schema_are_strict():
    issuer = AuthCapIssuer.generate("key-1")
    ring = AuthCapPublicKeyRing({"key-1": issuer.public_key_bytes})
    manifest = ring.public_manifest()
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    restored = AuthCapPublicKeyRing.from_manifest(manifest)
    assert restored.public_manifest() == ring.public_manifest()
    manifest["algorithm"] = "HMAC"
    with pytest.raises(AuthCapTokenError):
        AuthCapPublicKeyRing.from_manifest(manifest)


def test_unknown_payload_field_is_rejected():
    value = payload().canonical()
    value["unexpected"] = "field"
    with pytest.raises(AuthCapTokenError):
        AuthCapV1.from_mapping(value)
