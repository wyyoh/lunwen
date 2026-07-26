from __future__ import annotations

from pathlib import Path

import pytest

from keyed_gram.stage_d22_contract import D22Error
from keyed_gram.stage_d22_ipc import (
    append_safe_event,
    decode_message,
    encode_message,
    message_schema_event,
    read_safe_events,
)


def test_strict_ipc_json_round_trip() -> None:
    value = {"schema_version": 1, "op": "test", "nested": {"a": 1}}
    assert decode_message(encode_message(value)) == value


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b"[]",
        b"",
    ],
)
def test_strict_ipc_json_rejects_invalid_payload(payload: bytes) -> None:
    with pytest.raises(D22Error):
        decode_message(payload)


def test_ipc_schema_event_contains_no_message_values() -> None:
    message = {
        "schema_version": 1,
        "op": "generate",
        "value_b64": "SENSITIVE",
        "request_nonce": "nonce",
        "instruction_id": "copy",
    }
    event = message_schema_event(
        edge="memory_to_generator",
        message=message,
        wire_size_bytes=100,
    )
    assert "SENSITIVE" not in repr(event)
    assert event["contains_value_field"] is True
    assert event["contains_capability_token_field"] is False


def test_safe_event_file_is_strict_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_safe_event(path, {"event": "test", "count": 1})
    assert read_safe_events(path)[0]["event"] == "test"
