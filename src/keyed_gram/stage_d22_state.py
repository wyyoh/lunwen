"""D2.2 跨进程 SQLite replay/revocation 与 ticket 顺序状态。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .stage_d1_contract import CapabilityError, RejectionReason
from .stage_d22_contract import (
    D22Error,
    D22RejectionReason,
    ReleaseTicketClaims,
)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path,
        timeout=30.0,
        isolation_level=None,
    )
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


@contextmanager
def _immediate(path: Path) -> Iterator[sqlite3.Connection]:
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class PersistentCapabilityState:
    path: Path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _immediate(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_nonces (
                    nonce TEXT PRIMARY KEY,
                    status TEXT NOT NULL
                        CHECK(status IN ('indeterminate','consumed','revoked')),
                    updated_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gateway_sequences (
                    gateway_id TEXT PRIMARY KEY,
                    next_sequence INTEGER NOT NULL
                        CHECK(next_sequence > 0)
                )
                """
            )

    def reserve_indeterminate(self, nonce: str, *, now: int) -> None:
        with _immediate(self.path) as connection:
            row = connection.execute(
                "SELECT status FROM capability_nonces WHERE nonce=?",
                (nonce,),
            ).fetchone()
            if row is not None:
                self._raise_existing(str(row[0]))
            connection.execute(
                """
                INSERT INTO capability_nonces(nonce,status,updated_at)
                VALUES(?, 'indeterminate', ?)
                """,
                (nonce, now),
            )

    def consume(self, nonce: str) -> None:
        with _immediate(self.path) as connection:
            row = connection.execute(
                "SELECT status FROM capability_nonces WHERE nonce=?",
                (nonce,),
            ).fetchone()
            if row is not None:
                self._raise_existing(str(row[0]))
            connection.execute(
                """
                INSERT INTO capability_nonces(nonce,status,updated_at)
                VALUES(?, 'consumed', CAST(strftime('%s','now') AS INTEGER))
                """,
                (nonce,),
            )

    def revoke(self, nonce: str, *, now: int) -> None:
        with _immediate(self.path) as connection:
            connection.execute(
                """
                INSERT INTO capability_nonces(nonce,status,updated_at)
                VALUES(?, 'revoked', ?)
                ON CONFLICT(nonce) DO UPDATE SET
                    status='revoked',
                    updated_at=excluded.updated_at
                """,
                (nonce, now),
            )

    def status(self, nonce: str) -> str:
        connection = _connect(self.path)
        try:
            row = connection.execute(
                "SELECT status FROM capability_nonces WHERE nonce=?",
                (nonce,),
            ).fetchone()
        finally:
            connection.close()
        return "fresh" if row is None else str(row[0])

    def next_sequence(self, gateway_id: str) -> int:
        with _immediate(self.path) as connection:
            row = connection.execute(
                """
                SELECT next_sequence FROM gateway_sequences
                WHERE gateway_id=?
                """,
                (gateway_id,),
            ).fetchone()
            if row is None:
                sequence = 1
                connection.execute(
                    """
                    INSERT INTO gateway_sequences(gateway_id,next_sequence)
                    VALUES(?, 2)
                    """,
                    (gateway_id,),
                )
            else:
                sequence = int(row[0])
                connection.execute(
                    """
                    UPDATE gateway_sequences
                    SET next_sequence=?
                    WHERE gateway_id=?
                    """,
                    (sequence + 1, gateway_id),
                )
        return sequence

    @staticmethod
    def _raise_existing(status: str) -> None:
        if status == "revoked":
            raise CapabilityError(
                RejectionReason.REVOKED_TOKEN,
                "persistent capability nonce 已撤销",
            )
        raise CapabilityError(
            RejectionReason.REPLAY,
            "persistent single-use capability 已消费或状态不确定",
        )


@dataclass(frozen=True, slots=True)
class PersistentTicketState:
    path: Path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _immediate(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS release_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    gateway_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence > 0),
                    consumed_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gateway_ticket_order (
                    gateway_id TEXT PRIMARY KEY,
                    last_sequence INTEGER NOT NULL
                        CHECK(last_sequence > 0)
                )
                """
            )

    def consume(self, claims: ReleaseTicketClaims, *, now: int) -> None:
        with _immediate(self.path) as connection:
            duplicate = connection.execute(
                "SELECT 1 FROM release_tickets WHERE ticket_id=?",
                (claims.ticket_id,),
            ).fetchone()
            if duplicate is not None:
                raise D22Error(
                    D22RejectionReason.REPLAYED_TICKET,
                    "release ticket 已消费",
                )
            row = connection.execute(
                """
                SELECT last_sequence FROM gateway_ticket_order
                WHERE gateway_id=?
                """,
                (claims.gateway_id,),
            ).fetchone()
            if row is not None and claims.sequence <= int(row[0]):
                raise D22Error(
                    D22RejectionReason.OUT_OF_ORDER_TICKET,
                    "release ticket sequence 乱序或回退",
                )
            connection.execute(
                """
                INSERT INTO release_tickets(
                    ticket_id,gateway_id,sequence,consumed_at
                ) VALUES(?,?,?,?)
                """,
                (
                    claims.ticket_id,
                    claims.gateway_id,
                    claims.sequence,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO gateway_ticket_order(gateway_id,last_sequence)
                VALUES(?,?)
                ON CONFLICT(gateway_id) DO UPDATE SET
                    last_sequence=excluded.last_sequence
                """,
                (claims.gateway_id, claims.sequence),
            )

    def consumed_count(self) -> int:
        connection = _connect(self.path)
        try:
            row = connection.execute(
                "SELECT COUNT(*) FROM release_tickets"
            ).fetchone()
        finally:
            connection.close()
        return int(row[0]) if row is not None else 0
