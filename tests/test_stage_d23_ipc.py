from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from keyed_gram.stage_d23_contract import D23Error
from keyed_gram.stage_d23_ipc import (
    ByteAuditProxy,
    ConnectionBudget,
    decode_frame,
    duplicate_key_frame,
    nested_frame,
    overlong_frame,
    prepare_socket_path,
    verify_peer_uid,
)


def test_byte_proxy_reassembles_but_does_not_export_raw_bytes() -> None:
    proxy = ByteAuditProxy()
    message = {"schema_version": 1, "op": "ping"}
    assert proxy.transmit(message, split_at=(1, 2, 7)) == message
    assert proxy.metadata()[0]["raw_bytes_persisted"] is False
    assert "sha256" in proxy.metadata()[0]


@pytest.mark.parametrize(
    "frame",
    (overlong_frame(), nested_frame(), duplicate_key_frame(), b"\x00"),
)
def test_malformed_frames_fail_closed(frame: bytes) -> None:
    with pytest.raises(D23Error):
        decode_frame((frame,))


def test_socket_path_rejects_symlink_and_bad_parent(tmp_path: Path) -> None:
    secure = tmp_path / "secure"
    secure.mkdir(mode=0o700)
    target = secure / "target"
    target.write_text("x", encoding="utf-8")
    link = secure / "service.sock"
    link.symlink_to(target)
    with pytest.raises(D23Error):
        prepare_socket_path(link)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    os.chmod(unsafe, 0o755)
    with pytest.raises(D23Error):
        prepare_socket_path(unsafe / "service.sock")


def test_peer_uid_and_connection_budget() -> None:
    left, right = socket.socketpair()
    try:
        verify_peer_uid(left, expected_uid=os.getuid())
        with pytest.raises(D23Error):
            verify_peer_uid(left, expected_uid=os.getuid() + 1)
    finally:
        left.close()
        right.close()

    budget = ConnectionBudget(1, 8)
    budget.admit(declared_bytes=4, valid=False)
    budget.release(declared_bytes=4)
    with pytest.raises(D23Error):
        budget.admit(declared_bytes=4, valid=False)
