from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d1_contract import (
    RETRIEVE_PERMISSION,
    CapabilityError,
    RejectionReason,
)
from keyed_gram.stage_d22_contract import (
    D22Error,
    D22RejectionReason,
    ReleaseTicketClaims,
)
from keyed_gram.stage_d22_state import (
    PersistentCapabilityState,
    PersistentTicketState,
)


def _ticket(sequence: int, *, ticket_id: str | None = None):
    return ReleaseTicketClaims(
        gateway_id="gateway-a",
        ticket_id=ticket_id or secrets.token_hex(32),
        request_nonce=secrets.token_hex(32),
        subject_id="subject-alpha",
        entity_id="entity-a",
        relation_id=RelationId.REGISTRY_ID,
        permission=RETRIEVE_PERMISSION,
        instruction_id="copy-current-authorized-value",
        sequence=sequence,
        issued_at=1_800_000_000,
        expires_at=1_800_000_030,
    )


def test_capability_state_persists_consumption(tmp_path: Path) -> None:
    state = PersistentCapabilityState(tmp_path / "cap.sqlite3")
    state.initialize()
    nonce = secrets.token_hex(32)
    state.consume(nonce)
    assert PersistentCapabilityState(state.path).status(nonce) == "consumed"
    with pytest.raises(CapabilityError) as captured:
        PersistentCapabilityState(state.path).consume(nonce)
    assert captured.value.reason is RejectionReason.REPLAY


def test_indeterminate_reservation_fails_closed_after_restart(
    tmp_path: Path,
) -> None:
    state = PersistentCapabilityState(tmp_path / "cap.sqlite3")
    state.initialize()
    nonce = secrets.token_hex(32)
    state.reserve_indeterminate(nonce, now=1_800_000_000)
    with pytest.raises(CapabilityError) as captured:
        PersistentCapabilityState(state.path).consume(nonce)
    assert captured.value.reason is RejectionReason.REPLAY


def test_revocation_persists(tmp_path: Path) -> None:
    state = PersistentCapabilityState(tmp_path / "cap.sqlite3")
    state.initialize()
    nonce = secrets.token_hex(32)
    state.revoke(nonce, now=1_800_000_000)
    with pytest.raises(CapabilityError) as captured:
        PersistentCapabilityState(state.path).consume(nonce)
    assert captured.value.reason is RejectionReason.REVOKED_TOKEN


def test_two_concurrent_consumers_only_one_succeeds(tmp_path: Path) -> None:
    state = PersistentCapabilityState(tmp_path / "cap.sqlite3")
    state.initialize()
    nonce = secrets.token_hex(32)

    def consume() -> str:
        try:
            PersistentCapabilityState(state.path).consume(nonce)
        except CapabilityError:
            return "reject"
        return "accept"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: consume(), range(2))) == [
            "accept",
            "reject",
        ]


def test_gateway_sequence_is_atomic(tmp_path: Path) -> None:
    state = PersistentCapabilityState(tmp_path / "cap.sqlite3")
    state.initialize()
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(
            executor.map(
                lambda _: PersistentCapabilityState(
                    state.path
                ).next_sequence("gateway-a"),
                range(20),
            )
        )
    assert sorted(values) == list(range(1, 21))


def test_ticket_replay_and_order_are_persistent(tmp_path: Path) -> None:
    state = PersistentTicketState(tmp_path / "ticket.sqlite3")
    state.initialize()
    first = _ticket(2)
    state.consume(first, now=1_800_000_001)
    with pytest.raises(D22Error) as replay:
        state.consume(first, now=1_800_000_002)
    assert replay.value.reason is D22RejectionReason.REPLAYED_TICKET
    with pytest.raises(D22Error) as order:
        state.consume(_ticket(1), now=1_800_000_003)
    assert order.value.reason is D22RejectionReason.OUT_OF_ORDER_TICKET
    assert PersistentTicketState(state.path).consumed_count() == 1
