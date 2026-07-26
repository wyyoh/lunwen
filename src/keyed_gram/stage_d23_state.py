"""D2.3 epoch 绑定、生命周期账本与回滚检测状态。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stage_d23_contract import (
    MILESTONE_ORDER,
    D23Error,
    D23RejectionReason,
    EpochSnapshot,
    LifecycleMilestone,
    LifecycleTicket,
    digest_identifier,
)

STATE_SCHEMA_VERSION = 1
TICKET_TYPE = "KG-D23-LIFECYCLE"


def _connect(path: Path, *, timeout: float = 30.0) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path,
        timeout=timeout,
        isolation_level=None,
    )
    connection.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


@contextmanager
def _immediate(
    path: Path,
    *,
    timeout: float = 30.0,
) -> Iterator[sqlite3.Connection]:
    connection = _connect(path, timeout=timeout)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.execute("COMMIT")
    except sqlite3.OperationalError as exc:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise D23Error(
                D23RejectionReason.STATE_BUSY,
                "持久状态当前 busy/locked",
            ) from exc
        raise
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EpochTicketCodec:
    key: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.key, bytes) or len(self.key) < 32:
            raise ValueError("lifecycle ticket HMAC key 至少 32 bytes")

    def issue(self, claims: LifecycleTicket) -> bytes:
        payload = _canonical(
            {
                "algorithm": "HMAC-SHA-256",
                "claims": claims.to_payload(),
                "ticket_type": TICKET_TYPE,
            }
        )
        signature = hmac.digest(self.key, payload, "sha256")
        return (
            base64.urlsafe_b64encode(payload).rstrip(b"=")
            + b"."
            + base64.urlsafe_b64encode(signature).rstrip(b"=")
        )

    def verify(
        self,
        token: bytes,
        *,
        expected_epochs: EpochSnapshot,
        now: int,
        expected_gateway_id: str | None = None,
    ) -> LifecycleTicket:
        try:
            encoded_payload, encoded_signature = token.split(b".", 1)
            payload = base64.urlsafe_b64decode(
                encoded_payload + b"=" * (-len(encoded_payload) % 4)
            )
            signature = base64.urlsafe_b64decode(
                encoded_signature + b"=" * (-len(encoded_signature) % 4)
            )
            raw = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket 编码非法",
            ) from exc
        if not hmac.compare_digest(
            signature,
            hmac.digest(self.key, payload, "sha256"),
        ):
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket 认证失败",
            )
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"algorithm", "claims", "ticket_type"}
            or raw["algorithm"] != "HMAC-SHA-256"
            or raw["ticket_type"] != TICKET_TYPE
            or not isinstance(raw["claims"], Mapping)
        ):
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket envelope 非法",
            )
        claims = LifecycleTicket.from_payload(raw["claims"])
        if now < claims.issued_at or now >= claims.expires_at:
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket 已过期或尚未生效",
            )
        if claims.epochs != expected_epochs:
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket epoch 不匹配",
            )
        if (
            expected_gateway_id is not None
            and claims.gateway_id != expected_gateway_id
        ):
            raise D23Error(
                D23RejectionReason.EPOCH_MISMATCH,
                "lifecycle ticket gateway identity 不匹配",
            )
        return claims


@dataclass(frozen=True, slots=True)
class LifecycleStateStore:
    path: Path
    anchor_path: Path
    integrity_key: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.integrity_key, bytes) or len(self.integrity_key) < 32:
            raise ValueError("state integrity key 至少 32 bytes")

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _immediate(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lifecycle_metadata (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    state_epoch INTEGER NOT NULL CHECK(state_epoch > 0),
                    policy_epoch INTEGER NOT NULL CHECK(policy_epoch > 0),
                    gateway_epoch INTEGER NOT NULL CHECK(gateway_epoch > 0),
                    generation INTEGER NOT NULL CHECK(generation > 0),
                    wall_time_floor INTEGER NOT NULL CHECK(wall_time_floor >= 0),
                    integrity_tag TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_state (
                    nonce_digest TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('consumed','revoked')),
                    updated_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lifecycle_events (
                    request_digest TEXT NOT NULL,
                    milestone TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(request_digest, milestone)
                )
                """
            )
            row = connection.execute(
                "SELECT 1 FROM lifecycle_metadata WHERE singleton=1"
            ).fetchone()
            if row is None:
                payload = {
                    "gateway_epoch": 1,
                    "generation": 1,
                    "policy_epoch": 1,
                    "state_epoch": 1,
                    "wall_time_floor": 0,
                }
                connection.execute(
                    """
                    INSERT INTO lifecycle_metadata(
                        singleton,state_epoch,policy_epoch,gateway_epoch,
                        generation,wall_time_floor,integrity_tag
                    ) VALUES(1,?,?,?,?,?,?)
                    """,
                    (
                        payload["state_epoch"],
                        payload["policy_epoch"],
                        payload["gateway_epoch"],
                        payload["generation"],
                        payload["wall_time_floor"],
                        self._tag(payload),
                    ),
                )
        with _immediate(self.anchor_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monotonic_anchor (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    max_generation INTEGER NOT NULL,
                    max_state_epoch INTEGER NOT NULL,
                    max_policy_epoch INTEGER NOT NULL,
                    max_gateway_epoch INTEGER NOT NULL
                )
                """
            )
            snapshot = self._metadata()
            connection.execute(
                """
                INSERT INTO monotonic_anchor(
                    singleton,max_generation,max_state_epoch,
                    max_policy_epoch,max_gateway_epoch
                ) VALUES(1,?,?,?,?)
                ON CONFLICT(singleton) DO UPDATE SET
                    max_generation=MAX(max_generation,excluded.max_generation),
                    max_state_epoch=MAX(max_state_epoch,excluded.max_state_epoch),
                    max_policy_epoch=MAX(max_policy_epoch,excluded.max_policy_epoch),
                    max_gateway_epoch=MAX(max_gateway_epoch,excluded.max_gateway_epoch)
                """,
                (
                    snapshot["generation"],
                    snapshot["state_epoch"],
                    snapshot["policy_epoch"],
                    snapshot["gateway_epoch"],
                ),
            )
        self.verify()

    def snapshot(self) -> EpochSnapshot:
        payload = self.verify()
        return EpochSnapshot(
            state_epoch=payload["state_epoch"],
            policy_epoch=payload["policy_epoch"],
            gateway_instance_epoch=payload["gateway_epoch"],
        )

    def verify(self) -> dict[str, int]:
        try:
            payload = self._metadata()
            anchor = self._anchor()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            raise D23Error(
                D23RejectionReason.STATE_CORRUPT,
                "生命周期数据库无法验证",
            ) from exc
        if not hmac.compare_digest(
            payload.pop("integrity_tag"),
            self._tag(payload),
        ):
            raise D23Error(
                D23RejectionReason.STATE_CORRUPT,
                "生命周期 metadata integrity 失败",
            )
        comparisons = (
            ("generation", "max_generation"),
            ("state_epoch", "max_state_epoch"),
            ("policy_epoch", "max_policy_epoch"),
            ("gateway_epoch", "max_gateway_epoch"),
        )
        if any(payload[left] < anchor[right] for left, right in comparisons):
            raise D23Error(
                D23RejectionReason.STATE_ROLLBACK,
                "生命周期状态落后于 monotonic anchor",
            )
        return payload

    def consume(self, nonce: str, *, now: int, fault: str | None = None) -> None:
        digest = digest_identifier(nonce)

        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT status FROM capability_state WHERE nonce_digest=?",
                (digest,),
            ).fetchone()
            if row is not None:
                reason = (
                    D23RejectionReason.REVOKED
                    if row[0] == "revoked"
                    else D23RejectionReason.REPLAY
                )
                raise D23Error(reason, "capability 已撤销或消费")
            connection.execute(
                """
                INSERT INTO capability_state(nonce_digest,status,updated_at)
                VALUES(?,'consumed',?)
                """,
                (digest, now),
            )

        self._mutate(operation, now=now, fault=fault)

    def revoke(self, nonce: str, *, now: int) -> None:
        digest = digest_identifier(nonce)

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO capability_state(nonce_digest,status,updated_at)
                VALUES(?,'revoked',?)
                ON CONFLICT(nonce_digest) DO UPDATE SET
                    status='revoked',updated_at=excluded.updated_at
                """,
                (digest, now),
            )

        self._mutate(operation, now=now)

    def advance_epoch(self, name: str, *, now: int) -> EpochSnapshot:
        columns = {
            "state": "state_epoch",
            "policy": "policy_epoch",
            "gateway": "gateway_epoch",
        }
        if name not in columns:
            raise ValueError("unknown epoch")
        column = columns[name]

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                f"UPDATE lifecycle_metadata SET {column}={column}+1 "
                "WHERE singleton=1"
            )

        self._mutate(operation, now=now)
        return self.snapshot()

    def mark(
        self,
        request_id: str,
        milestone: LifecycleMilestone,
        *,
        now: int,
    ) -> None:
        digest = digest_identifier(request_id)
        ordinal = MILESTONE_ORDER.index(milestone)

        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                """
                SELECT MAX(ordinal) FROM lifecycle_events
                WHERE request_digest=?
                """,
                (digest,),
            ).fetchone()
            current = -1 if row is None or row[0] is None else int(row[0])
            if ordinal != current + 1:
                raise D23Error(
                    D23RejectionReason.INVALID_TRANSITION,
                    "生命周期 milestone 非单调相邻转换",
                )
            connection.execute(
                """
                INSERT INTO lifecycle_events(request_digest,milestone,ordinal)
                VALUES(?,?,?)
                """,
                (digest, milestone.value, ordinal),
            )

        self._mutate(operation, now=now)

    def validate_clock(
        self,
        *,
        now: int,
        maximum_forward_seconds: int,
    ) -> None:
        payload = self.verify()
        floor = payload["wall_time_floor"]
        if now < floor:
            raise D23Error(
                D23RejectionReason.CLOCK_ROLLBACK,
                "host clock 低于持久时间下界",
            )
        if floor and now - floor > maximum_forward_seconds:
            raise D23Error(
                D23RejectionReason.CLOCK_ROLLBACK,
                "host clock 向前漂移超过冻结上限",
            )

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                UPDATE lifecycle_metadata
                SET wall_time_floor=MAX(wall_time_floor,?)
                WHERE singleton=1
                """,
                (now,),
            )

        self._mutate(operation, now=now)

    def backup(self, destination: Path) -> None:
        source = _connect(self.path)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()

    def restore_file_for_test(self, backup: Path) -> None:
        shutil.copyfile(backup, self.path)

    def _mutate(
        self,
        operation: Any,
        *,
        now: int,
        fault: str | None = None,
    ) -> None:
        self.verify()
        if fault == "disk_full":
            raise D23Error(
                D23RejectionReason.FAULT_INJECTED,
                "注入 database or disk is full",
            )
        try:
            with _immediate(
                self.path,
                timeout=0.01 if fault == "busy" else 30.0,
            ) as connection:
                operation(connection)
                row = connection.execute(
                    """
                    SELECT state_epoch,policy_epoch,gateway_epoch,generation,
                           wall_time_floor
                    FROM lifecycle_metadata WHERE singleton=1
                    """
                ).fetchone()
                if row is None:
                    raise D23Error(
                        D23RejectionReason.STATE_CORRUPT,
                        "生命周期 metadata 缺失",
                    )
                payload = {
                    "state_epoch": int(row[0]),
                    "policy_epoch": int(row[1]),
                    "gateway_epoch": int(row[2]),
                    "generation": int(row[3]) + 1,
                    "wall_time_floor": max(int(row[4]), now),
                }
                connection.execute(
                    """
                    UPDATE lifecycle_metadata
                    SET state_epoch=?,policy_epoch=?,gateway_epoch=?,
                        generation=?,wall_time_floor=?,integrity_tag=?
                    WHERE singleton=1
                    """,
                    (
                        payload["state_epoch"],
                        payload["policy_epoch"],
                        payload["gateway_epoch"],
                        payload["generation"],
                        payload["wall_time_floor"],
                        self._tag(payload),
                    ),
                )
        except sqlite3.DatabaseError as exc:
            raise D23Error(
                D23RejectionReason.STATE_CORRUPT,
                "生命周期数据库 mutation 失败",
            ) from exc
        self._raise_anchor(self._metadata())

    def _metadata(self) -> dict[str, Any]:
        connection = _connect(self.path)
        try:
            row = connection.execute(
                """
                SELECT state_epoch,policy_epoch,gateway_epoch,generation,
                       wall_time_floor,integrity_tag
                FROM lifecycle_metadata WHERE singleton=1
                """
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise D23Error(
                D23RejectionReason.STATE_CORRUPT,
                "生命周期 metadata 缺失",
            )
        return {
            "state_epoch": int(row[0]),
            "policy_epoch": int(row[1]),
            "gateway_epoch": int(row[2]),
            "generation": int(row[3]),
            "wall_time_floor": int(row[4]),
            "integrity_tag": str(row[5]),
        }

    def _anchor(self) -> dict[str, int]:
        connection = _connect(self.anchor_path)
        try:
            row = connection.execute(
                """
                SELECT max_generation,max_state_epoch,max_policy_epoch,
                       max_gateway_epoch
                FROM monotonic_anchor WHERE singleton=1
                """
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise D23Error(
                D23RejectionReason.STATE_CORRUPT,
                "monotonic anchor 缺失",
            )
        return {
            "max_generation": int(row[0]),
            "max_state_epoch": int(row[1]),
            "max_policy_epoch": int(row[2]),
            "max_gateway_epoch": int(row[3]),
        }

    def _raise_anchor(self, payload: Mapping[str, Any]) -> None:
        with _immediate(self.anchor_path) as connection:
            connection.execute(
                """
                UPDATE monotonic_anchor SET
                    max_generation=MAX(max_generation,?),
                    max_state_epoch=MAX(max_state_epoch,?),
                    max_policy_epoch=MAX(max_policy_epoch,?),
                    max_gateway_epoch=MAX(max_gateway_epoch,?)
                WHERE singleton=1
                """,
                (
                    payload["generation"],
                    payload["state_epoch"],
                    payload["policy_epoch"],
                    payload["gateway_epoch"],
                ),
            )

    def _tag(self, payload: Mapping[str, Any]) -> str:
        return hmac.new(
            self.integrity_key,
            _canonical(payload),
            hashlib.sha256,
        ).hexdigest()
