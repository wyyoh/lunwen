from __future__ import annotations

import pytest

from keyed_gram.stage_d23_contract import (
    D23Error,
    EpochSnapshot,
    LifecycleTicket,
)


def test_epoch_snapshot_requires_positive_integers() -> None:
    with pytest.raises(D23Error):
        EpochSnapshot(0, 1, 1)


def test_lifecycle_ticket_round_trip() -> None:
    claims = LifecycleTicket(
        ticket_id="ticket-0123456789abcdef0123456789abcdef",
        gateway_id="gateway-a",
        epochs=EpochSnapshot(1, 2, 3),
        issued_at=100,
        expires_at=200,
    )
    assert LifecycleTicket.from_payload(claims.to_payload()) == claims
