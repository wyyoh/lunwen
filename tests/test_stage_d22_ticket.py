from __future__ import annotations

import secrets

import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d1_contract import RETRIEVE_PERMISSION
from keyed_gram.stage_d22_contract import (
    D22Error,
    ReleaseTicketClaims,
)
from keyed_gram.stage_d22_ticket import (
    ReleaseTicketCodec,
    ReleaseTicketKey,
)


def _claims(sequence: int = 1) -> ReleaseTicketClaims:
    return ReleaseTicketClaims(
        gateway_id="gateway-a",
        ticket_id=secrets.token_hex(32),
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


def test_release_ticket_round_trip() -> None:
    key = ReleaseTicketKey("release-ticket-test", secrets.token_bytes(32))
    codec = ReleaseTicketCodec()
    claims = _claims()
    assert codec.verify(codec.issue(claims, key), key) == claims


def test_release_ticket_tamper_fails_closed() -> None:
    key = ReleaseTicketKey("release-ticket-test", secrets.token_bytes(32))
    codec = ReleaseTicketCodec()
    changed = bytearray(codec.issue(_claims(), key))
    changed[-1] = ord("A") if changed[-1] != ord("A") else ord("B")
    with pytest.raises(D22Error):
        codec.verify(bytes(changed), key)


def test_release_ticket_wrong_key_fails_closed() -> None:
    first = ReleaseTicketKey("release-ticket-test", secrets.token_bytes(32))
    second = ReleaseTicketKey("release-ticket-test", secrets.token_bytes(32))
    ticket = ReleaseTicketCodec().issue(_claims(), first)
    with pytest.raises(D22Error):
        ReleaseTicketCodec().verify(ticket, second)


def test_ticket_requires_independent_key_namespace() -> None:
    with pytest.raises(ValueError):
        ReleaseTicketKey("capability-key-test", secrets.token_bytes(32))


def test_ticket_payload_rejects_unknown_relation() -> None:
    payload = _claims().to_payload()
    payload["relation_id"] = "unknown"
    with pytest.raises(D22Error):
        ReleaseTicketClaims.from_payload(payload)
