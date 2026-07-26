from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from keyed_gram.stage_d23_contract import (
    D23Error,
    D23RejectionReason,
    LifecycleMilestone,
    LifecycleTicket,
)
from keyed_gram.stage_d23_state import EpochTicketCodec, LifecycleStateStore


def _store(tmp_path: Path) -> LifecycleStateStore:
    store = LifecycleStateStore(
        tmp_path / "state.sqlite3",
        tmp_path / "anchor.sqlite3",
        secrets.token_bytes(32),
    )
    store.initialize()
    return store


def test_replay_and_milestone_order_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    nonce = secrets.token_hex(32)
    store.consume(nonce, now=1000)
    with pytest.raises(D23Error) as replay:
        store.consume(nonce, now=1001)
    assert replay.value.reason is D23RejectionReason.REPLAY

    request = secrets.token_hex(32)
    with pytest.raises(D23Error) as transition:
        store.mark(
            request,
            LifecycleMilestone.RECORD_DECRYPTED,
            now=1002,
        )
    assert transition.value.reason is D23RejectionReason.INVALID_TRANSITION


def test_old_backup_is_detected_by_separate_anchor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup = tmp_path / "backup.sqlite3"
    store.backup(backup)
    store.consume(secrets.token_hex(32), now=1000)
    store.restore_file_for_test(backup)
    with pytest.raises(D23Error) as error:
        store.verify()
    assert error.value.reason is D23RejectionReason.STATE_ROLLBACK


def test_epoch_ticket_rejects_policy_and_gateway_epoch_change(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    codec = EpochTicketCodec(secrets.token_bytes(32))
    old = store.snapshot()
    claims = LifecycleTicket(
        ticket_id=secrets.token_hex(32),
        gateway_id="gateway-a",
        epochs=old,
        issued_at=1000,
        expires_at=1100,
    )
    token = codec.issue(claims)
    store.advance_epoch("policy", now=1001)
    with pytest.raises(D23Error):
        codec.verify(
            token,
            expected_epochs=store.snapshot(),
            expected_gateway_id="gateway-a",
            now=1002,
        )


def test_clock_drift_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.validate_clock(now=1000, maximum_forward_seconds=100)
    with pytest.raises(D23Error):
        store.validate_clock(now=999, maximum_forward_seconds=100)
    with pytest.raises(D23Error):
        store.validate_clock(now=2000, maximum_forward_seconds=100)
