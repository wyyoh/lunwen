from __future__ import annotations

import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d1_contract import (
    CapabilityError,
    RejectionReason,
    create_verified_capability,
)
from keyed_gram.stage_d2 import (
    _environment,
    _issued_request,
    _retrieve,
    _run_attack_matrix,
    _summary,
)
from keyed_gram.stage_d2_contract import (
    AuthenticatedPrincipal,
    D2Error,
    D2RejectionReason,
)
from keyed_gram.stage_d2_protocol import load_config


CONFIG = "configs/stage_d2.yaml"


def test_valid_release_requires_trusted_principal_and_capability() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    _, request = _issued_request(
        current, RelationId.REGISTRY_ID, "entity-a"
    )
    release = _retrieve(current, request)
    assert release.entity_id == "entity-a"
    assert release.relation_id is RelationId.REGISTRY_ID
    assert release.value_released is True
    snapshot = current.memory.snapshot()
    assert snapshot.lookup_count == 1
    assert snapshot.decrypt_attempt_count == 1
    assert snapshot.plaintext_release_count == 1


def test_wrong_principal_stops_before_lookup_or_decrypt() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    _, request = _issued_request(
        current, RelationId.REGISTRY_ID, "entity-a"
    )
    before = current.memory.snapshot()
    with pytest.raises(D2Error) as error:
        _retrieve(current, request, subject_id="subject-beta")
    assert error.value.reason is D2RejectionReason.PRINCIPAL_MISMATCH
    assert current.memory.snapshot().delta(before) == {
        "memory_lookup_delta": 0,
        "decrypt_attempt_delta": 0,
        "plaintext_release_delta": 0,
        "migration_decrypt_attempt_delta": 0,
        "successful_record_migration_delta": 0,
    }


def test_replay_consumption_happens_before_second_plaintext_release() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    _, request = _issued_request(
        current, RelationId.REGISTRY_ID, "entity-a"
    )
    _retrieve(current, request)
    before = current.memory.snapshot()
    with pytest.raises(CapabilityError) as error:
        _retrieve(current, request)
    assert error.value.reason is RejectionReason.REPLAY
    assert current.memory.snapshot().delta(before)[
        "plaintext_release_delta"
    ] == 0


def test_forged_principal_and_capability_never_lookup_memory() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    with pytest.raises(TypeError):
        AuthenticatedPrincipal(
            "subject-alpha",
            "session-a",
            "mock-mfa",
            object(),
        )
    principal = current.identity.authenticate(
        current.sessions["subject-alpha"]
    )
    before = current.memory.snapshot()
    with pytest.raises(D2Error) as capability:
        current.memory.retrieve_authorized(principal, object())
    assert (
        capability.value.reason
        is D2RejectionReason.FORGED_VERIFIED_CAPABILITY
    )
    assert current.memory.snapshot().lookup_count == before.lookup_count


def test_importable_d1_verified_factory_cannot_bypass_d2_gateway() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    principal = current.identity.authenticate(
        current.sessions["subject-alpha"]
    )
    forged = create_verified_capability(
        subject_id="subject-alpha",
        entity_id="entity-a",
        relation_id=RelationId.REGISTRY_ID,
        permission="retrieve",
        nonce="forged-nonce-that-is-long-enough",
        expires_at=current.now + 60,
    )
    before = current.memory.snapshot()
    with pytest.raises(D2Error) as error:
        current.memory.retrieve_authorized(principal, forged)
    assert (
        error.value.reason
        is D2RejectionReason.FORGED_VERIFIED_CAPABILITY
    )
    assert current.memory.snapshot().lookup_count == before.lookup_count


def test_gateway_cannot_release_from_differently_bound_memory_instance() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    other = _environment(values)
    _, request = _issued_request(
        current, RelationId.REGISTRY_ID, "entity-a"
    )
    before = other.memory.snapshot()
    with pytest.raises(D2Error) as error:
        _retrieve(current, request, memory=other.memory)
    assert (
        error.value.reason
        is D2RejectionReason.FORGED_VERIFIED_CAPABILITY
    )
    assert other.memory.snapshot().lookup_count == before.lookup_count


def test_capability_hmac_and_data_aead_key_material_are_independent() -> None:
    current = _environment(load_config(CONFIG))
    assert set(current.capability_keyring._keys).isdisjoint(
        current.data_keyring._keys
    )
    assert set(current.capability_keyring._keys.values()).isdisjoint(
        current.data_keyring._keys.values()
    )


def test_migration_increments_version_and_uses_active_key_without_release() -> None:
    values = load_config(CONFIG)
    current = _environment(values)
    old = current.memory._records["record-a-registry"]
    current.data_keyring.rotate("data-key-2")
    before = current.memory.snapshot()
    migrated = current.memory.migrate_record_to_active_key(
        "record-a-registry"
    )
    delta = current.memory.snapshot().delta(before)
    assert migrated.record_version == old.record_version + 1
    assert migrated.data_key_id == "data-key-2"
    assert delta["migration_decrypt_attempt_delta"] == 1
    assert delta["successful_record_migration_delta"] == 1
    assert delta["plaintext_release_delta"] == 0


def test_full_matrix_has_no_plaintext_or_secret_fields() -> None:
    rows, _ = _run_attack_matrix(load_config(CONFIG))
    forbidden = {
        "plaintext",
        "value",
        "capability_token",
        "capability_hmac_key",
        "data_encryption_key",
        "session_credential",
    }
    assert all(forbidden.isdisjoint(row) for row in rows)
    assert all(row["passed"] for row in rows)


def test_authorization_negatives_stop_before_lookup_and_decrypt() -> None:
    rows, _ = _run_attack_matrix(load_config(CONFIG))
    authorization = [
        row
        for row in rows
        if row["category"] == "authorization_negative"
    ]
    assert len(authorization) == 13
    assert all(row["memory_lookup_delta"] == 0 for row in authorization)
    assert all(row["decrypt_attempt_delta"] == 0 for row in authorization)
    assert all(row["plaintext_release_delta"] == 0 for row in authorization)


def test_record_attacks_never_release_plaintext() -> None:
    rows, _ = _run_attack_matrix(load_config(CONFIG))
    attacks = [
        row
        for row in rows
        if row["category"]
        in {"record_negative", "record_schema_negative"}
    ]
    assert len(attacks) == 14
    assert all(row["plaintext_release_delta"] == 0 for row in attacks)
    assert all(not row["value_released"] for row in attacks)


def test_concurrent_single_use_has_one_release_and_one_replay() -> None:
    rows, concurrency = _run_attack_matrix(load_config(CONFIG))
    row = next(
        row
        for row in rows
        if row["scenario_id"] == "concurrent_single_use_capability"
    )
    assert row["passed"] is True
    assert concurrency == {
        "successful_release_count": 1,
        "replay_rejection_count": 1,
        "concurrent_double_release_count": 0,
    }
    assert row["plaintext_release_delta"] == 1


def test_summary_meets_all_zero_gates_without_unlocking_c3() -> None:
    values = load_config(CONFIG)
    rows, concurrency = _run_attack_matrix(values)
    summary = _summary(
        values,
        rows,
        concurrency,
        git={"commit": "test", "branch": "test", "tracked_dirty": False},
        source_manifest={"manifest_payload_sha256": "test"},
    )
    assert summary["capability_gated_keyed_memory_status"] == "passed"
    assert summary["ready_for_d2_1_public_generation_probe"] is True
    assert summary["private_value_memory_ready"] is False
    assert summary["original_c3_allowed"] is False
    assert summary["c3_eligible"] is False
    for key in values["required_zero_metrics"]:
        assert summary["metrics"][key] == 0
