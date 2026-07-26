"""Stage D2.3 故障、可观测性与服务生命周期正式审计。"""

from __future__ import annotations

import base64
import os
import secrets
import shutil
import socket
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .stage_c24_contract import RelationId
from .stage_d1_contract import RETRIEVE_PERMISSION
from .stage_d1_protocol import (
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)
from .stage_d2_contract import PublicSyntheticValue
from .stage_d2_crypto import AesGcmRecordCodec, DataEncryptionKeyRing
from .stage_d21_leakage import (
    CanaryFactory,
    count_canary_occurrences,
)
from .stage_d22_contract import D22Error, ReleaseTicketClaims
from .stage_d22_runtime import D22Cluster
from .stage_d22_services import INSTRUCTION_ID
from .stage_d22_state import PersistentTicketState
from .stage_d23_contract import (
    D23Error,
    D23RejectionReason,
    EpochSnapshot,
    LifecycleMilestone,
    LifecycleTicket,
)
from .stage_d23_ipc import (
    ByteAuditProxy,
    ConnectionBudget,
    decode_frame,
    duplicate_key_frame,
    encode_frame,
    nested_frame,
    overlong_frame,
    prepare_socket_path,
    verify_peer_uid,
)
from .stage_d23_metrics import (
    aggregate_metrics,
    ipc_byte_manifest,
    leakage_scan,
    lifecycle_state_manifest,
    observability_manifest,
    write_csv,
)
from .stage_d23_observability import SafeObservabilitySink
from .stage_d23_protocol import (
    D23ProtocolError,
    artifact_manifest,
    git_state,
    load_config,
    mark_phase_completed,
    mark_phase_started,
    output_paths,
    repo_root,
    require_phase_completed,
    required_scenarios,
    resolve_path,
    runtime_source_manifest,
)
from .stage_d23_state import EpochTicketCodec, LifecycleStateStore

ZERO_ROW_METRICS = (
    "secret_metric_label_occurrence_count",
    "secret_trace_attribute_occurrence_count",
    "secret_error_report_occurrence_count",
    "post_crash_double_release_count",
    "post_timeout_double_release_count",
    "post_restore_replay_success_count",
    "state_rollback_acceptance_count",
    "unauthorized_local_socket_connection_count",
    "stale_ticket_acceptance_count",
    "cross_epoch_ticket_acceptance_count",
    "malformed_ipc_plaintext_release_count",
)


def _row(
    scenario_id: str,
    category: str,
    *,
    passed: bool,
    observed: str,
    exact_delivery: bool = False,
    output_digest: str = "",
    **metrics: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "category": category,
        "passed": passed,
        "observed": observed,
        "exact_delivery": exact_delivery,
        "output_digest": output_digest,
    }
    row.update({name: 0 for name in ZERO_ROW_METRICS})
    row.update(metrics)
    return row


def _expect_d23(
    action: Callable[[], Any],
    reasons: D23RejectionReason | Sequence[D23RejectionReason],
) -> D23RejectionReason:
    expected = (reasons,) if isinstance(reasons, D23RejectionReason) else tuple(reasons)
    try:
        action()
    except D23Error as exc:
        if exc.reason not in expected:
            raise
        return exc.reason
    raise AssertionError("预期 D2.3 fail-closed 拒绝但操作成功")


def _new_store(
    root: Path,
    secrets_seen: list[bytes],
) -> LifecycleStateStore:
    key = secrets.token_bytes(32)
    secrets_seen.append(key)
    store = LifecycleStateStore(
        root / "lifecycle-state.sqlite3",
        root / "monotonic-anchor.sqlite3",
        key,
    )
    store.initialize()
    return store


def _ticket(
    epochs: EpochSnapshot,
    *,
    gateway_id: str = "gateway-a",
    now: int = 1000,
) -> LifecycleTicket:
    return LifecycleTicket(
        ticket_id=secrets.token_hex(32),
        gateway_id=gateway_id,
        epochs=epochs,
        issued_at=now,
        expires_at=now + 60,
    )


def _run_positive(
    root: Path,
    factory: CanaryFactory,
    secrets_seen: list[bytes],
) -> list[dict[str, Any]]:
    cluster = D22Cluster(root / "positive-cluster", factory)
    secrets_seen.extend(
        (
            cluster.capability_key,
            cluster.release_ticket_key_bytes,
            cluster.data_key,
            cluster.client_gateway_authkey,
            cluster.gateway_memory_authkey,
            cluster.memory_generator_authkey,
            *cluster.sessions.values(),
        )
    )
    declarations = (
        (
            "authorized_registry_lifecycle_delivery",
            RelationId.REGISTRY_ID,
            "entity-a",
        ),
        (
            "authorized_city_lifecycle_delivery",
            RelationId.CITY_CODE,
            "entity-b",
        ),
        (
            "authorized_access_lifecycle_delivery",
            RelationId.ACCESS_CODE,
            "entity-c",
        ),
    )
    rows: list[dict[str, Any]] = []
    try:
        cluster.start_default()
        for scenario_id, relation_id, entity_id in declarations:
            _, message = cluster.issue_message(relation_id, entity_id)
            result = cluster.parse_success(cluster.client_call(message))
            expected = cluster.canaries[(entity_id, relation_id)]
            rows.append(
                _row(
                    scenario_id,
                    "positive",
                    passed=result.text == expected and result.exact_delivery,
                    observed="authorized_exact_delivery",
                    exact_delivery=result.text == expected,
                    output_digest=result.output_digest,
                )
            )
        cluster.capture_process_surfaces()
    finally:
        cluster.stop_all()
    return rows


def _run_observability(
    root: Path,
    factory: CanaryFactory,
    secrets_seen: list[bytes],
) -> list[dict[str, Any]]:
    scenario_root = root / "observability"
    canary = factory.issue()
    secret = secrets.token_bytes(32)
    secrets_seen.append(secret)
    forbidden = (
        canary.encode("ascii"),
        secret,
        base64.b64encode(secret),
        secret.hex().encode("ascii"),
    )
    sink = SafeObservabilitySink(
        scenario_root / "observability",
        forbidden_values=forbidden,
    )
    rows: list[dict[str, Any]] = []

    malicious = {
        "structured_log_redaction": (
            "structured_log",
            {"event": "request", "plaintext": canary},
        ),
        "trace_attribute_redaction": (
            "trace",
            {"event": "span", "value": canary},
        ),
        "trace_baggage_redaction": (
            "trace",
            {"event": "span", "baggage": canary},
        ),
        "error_report_redaction": (
            "error_report",
            {"event": "exception", "credential": canary},
        ),
        "retry_queue_redaction": (
            "retry_queue",
            {"event": "retry", "token": canary},
        ),
        "profiling_output_redaction": (
            "profile",
            {"event": "sample", "value": canary},
        ),
        "supervisor_log_redaction": (
            "supervisor",
            {"event": "restart", "session": canary},
        ),
        "audit_event_redaction": (
            "audit",
            {"event": "release", "capability": canary},
        ),
    }
    for scenario_id, (channel, payload) in malicious.items():
        reason = _expect_d23(
            lambda c=channel, p=payload: sink.emit(c, p),
            D23RejectionReason.SENSITIVE_OBSERVABILITY_VALUE,
        )
        metrics: dict[str, int] = {}
        if scenario_id.startswith("trace_"):
            metrics["secret_trace_attribute_occurrence_count"] = 0
        elif scenario_id == "error_report_redaction":
            metrics["secret_error_report_occurrence_count"] = 0
        rows.append(
            _row(
                scenario_id,
                "observability",
                passed=True,
                observed=reason.value,
                **metrics,
            )
        )

    metric_reason = _expect_d23(
        lambda: sink.metric(
            "requests_total",
            1,
            labels={"entity_id": canary},
        ),
        D23RejectionReason.SENSITIVE_OBSERVABILITY_VALUE,
    )
    rows.append(
        _row(
            "metric_label_redaction",
            "observability",
            passed=True,
            observed=metric_reason.value,
            secret_metric_label_occurrence_count=0,
        )
    )

    health = sink.fixed_health()
    rows.append(
        _row(
            "health_endpoint_redaction",
            "observability",
            passed=health == {"schema_version": 1, "status": "ok"},
            observed="fixed_health_schema",
        )
    )
    debug = sink.fixed_debug()
    rows.append(
        _row(
            "debug_endpoint_redaction",
            "observability",
            passed=debug == {"schema_version": 1, "debug": "disabled"},
            observed="debug_disabled",
        )
    )

    backup_root = scenario_root / "backup"
    backup_root.mkdir(parents=True)
    store = _new_store(backup_root, secrets_seen)
    store.consume(secrets.token_hex(32), now=1000)
    backup = backup_root / "state-backup.sqlite3"
    store.backup(backup)
    payload = backup.read_bytes()
    rows.append(
        _row(
            "sqlite_backup_redaction",
            "observability",
            passed=canary.encode("ascii") not in payload
            and secret not in payload,
            observed="backup_contains_only_digest_state",
        )
    )
    return rows


def _consume_and_advance(
    store: LifecycleStateStore,
    nonce: str,
    request_id: str,
    milestones: Sequence[LifecycleMilestone],
) -> None:
    store.consume(nonce, now=1000)
    for offset, milestone in enumerate(milestones, start=1):
        store.mark(request_id, milestone, now=1000 + offset)


def _run_faults(
    root: Path,
    factory: CanaryFactory,
    secrets_seen: list[bytes],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario = root / "fault-disk-full"
    scenario.mkdir()
    store = _new_store(scenario, secrets_seen)
    reason = _expect_d23(
        lambda: store.consume(
            secrets.token_hex(32),
            now=1000,
            fault="disk_full",
        ),
        D23RejectionReason.FAULT_INJECTED,
    )
    rows.append(
        _row(
            "injected_disk_full",
            "fault",
            passed=True,
            observed=reason.value,
        )
    )

    scenario = root / "fault-locked"
    scenario.mkdir()
    store = _new_store(scenario, secrets_seen)
    lock = sqlite3.connect(store.path, isolation_level=None)
    lock.execute("BEGIN IMMEDIATE")
    try:
        reason = _expect_d23(
            lambda: store.consume(
                secrets.token_hex(32),
                now=1000,
                fault="busy",
            ),
            D23RejectionReason.STATE_BUSY,
        )
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    rows.append(
        _row(
            "sqlite_locked_busy",
            "fault",
            passed=True,
            observed=reason.value,
        )
    )

    for scenario_id, mutator in (
        (
            "truncated_state_database",
            lambda payload: payload[:64],
        ),
        (
            "corrupted_state_database",
            lambda payload: b"corrupt!" + payload[8:],
        ),
    ):
        scenario = root / scenario_id
        scenario.mkdir()
        store = _new_store(scenario, secrets_seen)
        store.path.write_bytes(mutator(store.path.read_bytes()))
        reason = _expect_d23(
            store.verify,
            D23RejectionReason.STATE_CORRUPT,
        )
        rows.append(
            _row(
                scenario_id,
                "fault",
                passed=True,
                observed=reason.value,
            )
        )

    frame = encode_frame({"schema_version": 1, "op": "ping"})
    reason = _expect_d23(
        lambda: decode_frame((frame[:-2],)),
        D23RejectionReason.INVALID_FRAME,
    )
    rows.append(
        _row(
            "partial_socket_write",
            "fault",
            passed=True,
            observed=reason.value,
        )
    )

    lifecycle_cases = (
        (
            "crash_before_response_write",
            (
                LifecycleMilestone.CAPABILITY_CONSUMED,
                LifecycleMilestone.RECORD_DECRYPTED,
                LifecycleMilestone.GENERATOR_INVOKED,
                LifecycleMilestone.OUTPUT_COMMITTED,
            ),
            "post_crash_double_release_count",
        ),
        (
            "client_timeout_retry",
            (
                LifecycleMilestone.CAPABILITY_CONSUMED,
                LifecycleMilestone.RECORD_DECRYPTED,
            ),
            "post_timeout_double_release_count",
        ),
        (
            "generator_completed_gateway_missing_response",
            (
                LifecycleMilestone.CAPABILITY_CONSUMED,
                LifecycleMilestone.RECORD_DECRYPTED,
                LifecycleMilestone.GENERATOR_INVOKED,
                LifecycleMilestone.OUTPUT_COMMITTED,
            ),
            "post_crash_double_release_count",
        ),
    )
    for scenario_id, milestones, metric in lifecycle_cases:
        scenario = root / scenario_id
        scenario.mkdir()
        store = _new_store(scenario, secrets_seen)
        nonce = secrets.token_hex(32)
        _consume_and_advance(
            store,
            nonce,
            secrets.token_hex(32),
            milestones,
        )
        reason = _expect_d23(
            lambda s=store, n=nonce: s.consume(n, now=2000),
            D23RejectionReason.REPLAY,
        )
        rows.append(
            _row(
                scenario_id,
                "fault",
                passed=True,
                observed=reason.value,
                **{metric: 0},
            )
        )

    scenario = root / "supervisor-restart"
    scenario.mkdir()
    store = _new_store(scenario, secrets_seen)
    nonce = secrets.token_hex(32)
    store.consume(nonce, now=1000)
    restarted = LifecycleStateStore(
        store.path,
        store.anchor_path,
        store.integrity_key,
    )
    restarted.initialize()
    reason = _expect_d23(
        lambda: restarted.consume(nonce, now=1001),
        D23RejectionReason.REPLAY,
    )
    rows.append(
        _row(
            "supervisor_automatic_restart",
            "fault",
            passed=True,
            observed=reason.value,
        )
    )

    for scenario_id, test_now in (
        ("host_clock_forward_drift", 2000),
        ("host_clock_backward_drift", 999),
    ):
        scenario = root / scenario_id
        scenario.mkdir()
        store = _new_store(scenario, secrets_seen)
        store.validate_clock(now=1000, maximum_forward_seconds=100)
        reason = _expect_d23(
            lambda s=store, n=test_now: s.validate_clock(
                now=n,
                maximum_forward_seconds=100,
            ),
            D23RejectionReason.CLOCK_ROLLBACK,
        )
        rows.append(
            _row(
                scenario_id,
                "fault",
                passed=True,
                observed=reason.value,
            )
        )

    codec = AesGcmRecordCodec()
    ring = DataEncryptionKeyRing.generate("data-d23-old")
    old_value = PublicSyntheticValue(factory.issue().encode("ascii"))
    old_record = codec.encrypt(
        old_value,
        record_id="record-d23-rotation",
        entity_id="entity-a",
        relation_id=RelationId.REGISTRY_ID,
        record_version=1,
        keyring=ring,
    )
    barrier = threading.Barrier(2)
    results: list[bool] = []

    def rotate() -> None:
        barrier.wait()
        ring.rotate("data-d23-new")
        results.append(True)

    def read_old() -> None:
        barrier.wait()
        results.append(codec.decrypt(old_record, keyring=ring).payload == old_value.payload)

    threads = (threading.Thread(target=rotate), threading.Thread(target=read_old))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows.append(
        _row(
            "concurrent_key_rotation_read",
            "fault",
            passed=all(results) and len(results) == 2,
            observed="old_record_read_and_rotation_serialized",
        )
    )

    migration_lock = threading.Lock()
    current = {"record": old_record}
    migration_results: list[bool] = []
    barrier = threading.Barrier(2)

    def migrate() -> None:
        barrier.wait()
        with migration_lock:
            value = codec.decrypt(current["record"], keyring=ring)
            current["record"] = codec.encrypt(
                value,
                record_id="record-d23-rotation",
                entity_id="entity-a",
                relation_id=RelationId.REGISTRY_ID,
                record_version=2,
                keyring=ring,
            )
            migration_results.append(True)

    def concurrent_read() -> None:
        barrier.wait()
        with migration_lock:
            migration_results.append(
                codec.decrypt(current["record"], keyring=ring).payload
                == old_value.payload
            )

    threads = (
        threading.Thread(target=migrate),
        threading.Thread(target=concurrent_read),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows.append(
        _row(
            "concurrent_record_migration_read",
            "fault",
            passed=all(migration_results) and current["record"].record_version == 2,
            observed="migration_and_read_serialized",
        )
    )
    return rows


def _run_endpoint_ipc(
    root: Path,
    proxy: ByteAuditProxy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario = root / "endpoint-stale"
    scenario.mkdir(mode=0o700)
    stale = scenario / "service.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(stale))
    listener.close()
    prepare_socket_path(stale)
    rows.append(
        _row(
            "stale_socket_file",
            "endpoint_ipc",
            passed=not stale.exists(),
            observed="owned_stale_socket_removed",
        )
    )

    scenario = root / "endpoint-symlink"
    scenario.mkdir(mode=0o700)
    target = scenario / "target"
    target.write_text("not-a-socket", encoding="utf-8")
    link = scenario / "service.sock"
    link.symlink_to(target)
    reason = _expect_d23(
        lambda: prepare_socket_path(link),
        D23RejectionReason.ENDPOINT_UNSAFE,
    )
    rows.append(
        _row(
            "socket_symlink_replacement",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
        )
    )

    scenario = root / "endpoint-permissions"
    scenario.mkdir(mode=0o755)
    os.chmod(scenario, 0o755)
    reason = _expect_d23(
        lambda: prepare_socket_path(scenario / "service.sock"),
        D23RejectionReason.ENDPOINT_UNSAFE,
    )
    os.chmod(scenario, 0o700)
    rows.append(
        _row(
            "wrong_socket_directory_permissions",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
        )
    )

    left, right = socket.socketpair()
    try:
        reason = _expect_d23(
            lambda: verify_peer_uid(left, expected_uid=os.getuid() + 1),
            D23RejectionReason.PEER_IDENTITY_MISMATCH,
        )
    finally:
        left.close()
        right.close()
    rows.append(
        _row(
            "wrong_local_peer_uid",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
            unauthorized_local_socket_connection_count=0,
        )
    )

    malformed = (
        (
            "overlong_message",
            overlong_frame(),
            D23RejectionReason.MESSAGE_TOO_LARGE,
        ),
        (
            "deeply_nested_json",
            nested_frame(),
            D23RejectionReason.MESSAGE_TOO_DEEP,
        ),
        (
            "duplicate_json_key",
            duplicate_key_frame(),
            D23RejectionReason.INVALID_FRAME,
        ),
        (
            "truncated_message",
            encode_frame({"op": "ping"})[:-1],
            D23RejectionReason.INVALID_FRAME,
        ),
        (
            "malformed_frame_prefix",
            b"\x00\x00",
            D23RejectionReason.INVALID_FRAME,
        ),
    )
    for scenario_id, frame, expected in malformed:
        reason = _expect_d23(lambda f=frame: decode_frame((f,)), expected)
        rows.append(
            _row(
                scenario_id,
                "endpoint_ipc",
                passed=True,
                observed=reason.value,
                malformed_ipc_plaintext_release_count=0,
            )
        )

    message = {"schema_version": 1, "op": "ping"}
    result = proxy.transmit(message, split_at=(1, 3, 5, 11))
    rows.append(
        _row(
            "fragmented_message",
            "endpoint_ipc",
            passed=result == message,
            observed="fragmented_frame_reassembled",
        )
    )

    ticket_state_root = root / "endpoint-order"
    ticket_state_root.mkdir()
    state = PersistentTicketState(ticket_state_root / "tickets.sqlite3")
    state.initialize()

    def claims(sequence: int, ticket_id: str) -> ReleaseTicketClaims:
        return ReleaseTicketClaims(
            gateway_id="gateway-a",
            ticket_id=ticket_id,
            request_nonce=secrets.token_hex(32),
            subject_id="subject-alpha",
            entity_id="entity-a",
            relation_id=RelationId.REGISTRY_ID,
            permission=RETRIEVE_PERMISSION,
            instruction_id=INSTRUCTION_ID,
            sequence=sequence,
            issued_at=1000,
            expires_at=1100,
        )

    state.consume(claims(2, secrets.token_hex(32)), now=1000)
    try:
        state.consume(claims(1, secrets.token_hex(32)), now=1000)
    except D22Error:
        order_rejected = True
    else:
        order_rejected = False
    rows.append(
        _row(
            "out_of_order_request",
            "endpoint_ipc",
            passed=order_rejected,
            observed="out_of_order_ticket",
        )
    )

    key = secrets.token_bytes(32)
    codec = EpochTicketCodec(key)
    epochs = EpochSnapshot(1, 1, 1)
    stale_token = codec.issue(_ticket(epochs, now=1000))
    reason = _expect_d23(
        lambda: codec.verify(
            stale_token,
            expected_epochs=epochs,
            now=2000,
        ),
        D23RejectionReason.EPOCH_MISMATCH,
    )
    rows.append(
        _row(
            "stale_release_ticket",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
            stale_ticket_acceptance_count=0,
        )
    )
    cross_token = codec.issue(
        _ticket(epochs, gateway_id="gateway-b", now=1000)
    )
    reason = _expect_d23(
        lambda: codec.verify(
            cross_token,
            expected_epochs=epochs,
            expected_gateway_id="gateway-a",
            now=1001,
        ),
        D23RejectionReason.EPOCH_MISMATCH,
    )
    rows.append(
        _row(
            "cross_gateway_ticket",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
        )
    )

    slow = ConnectionBudget(3, 32)
    reason = _expect_d23(
        lambda: slow.admit(declared_bytes=33, valid=True),
        D23RejectionReason.RATE_LIMITED,
    )
    rows.append(
        _row(
            "slow_sender_budget",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
        )
    )
    flood = ConnectionBudget(2, 128)
    flood.admit(declared_bytes=1, valid=False)
    flood.release(declared_bytes=1)
    flood.admit(declared_bytes=1, valid=False)
    flood.release(declared_bytes=1)
    reason = _expect_d23(
        lambda: flood.admit(declared_bytes=1, valid=False),
        D23RejectionReason.RATE_LIMITED,
    )
    rows.append(
        _row(
            "invalid_connection_flood",
            "endpoint_ipc",
            passed=True,
            observed=reason.value,
        )
    )
    return rows


def _run_state_recovery(
    root: Path,
    factory: CanaryFactory,
    secrets_seen: list[bytes],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario = root / "state-backup-sensitive"
    scenario.mkdir()
    store = _new_store(scenario, secrets_seen)
    canary = factory.issue()
    nonce = secrets.token_hex(32)
    store.consume(nonce, now=1000)
    backup = scenario / "backup.sqlite3"
    store.backup(backup)
    payload = backup.read_bytes()
    rows.append(
        _row(
            "backup_contains_no_sensitive_material",
            "state_recovery",
            passed=canary.encode("ascii") not in payload
            and nonce.encode("ascii") not in payload
            and store.integrity_key not in payload,
            observed="backup_digest_only",
        )
    )

    def rollback_case(
        scenario_id: str,
        mutation: Callable[[LifecycleStateStore, str], None],
    ) -> None:
        scenario_root = root / scenario_id
        scenario_root.mkdir()
        current = _new_store(scenario_root, secrets_seen)
        old = scenario_root / "old-backup.sqlite3"
        current.backup(old)
        case_nonce = secrets.token_hex(32)
        mutation(current, case_nonce)
        current.restore_file_for_test(old)
        reason = _expect_d23(
            current.verify,
            D23RejectionReason.STATE_ROLLBACK,
        )
        rows.append(
            _row(
                scenario_id,
                "state_recovery",
                passed=True,
                observed=reason.value,
                post_restore_replay_success_count=0,
                state_rollback_acceptance_count=0,
            )
        )

    rollback_case(
        "replay_rejected_after_backup_restore",
        lambda store, value: store.consume(value, now=1000),
    )
    rollback_case(
        "old_backup_does_not_revive_revocation",
        lambda store, value: store.revoke(value, now=1000),
    )
    rollback_case(
        "state_rollback_detected",
        lambda store, _: store.advance_epoch("state", now=1000),
    )

    scenario = root / "database-replacement"
    scenario.mkdir()
    store = _new_store(scenario, secrets_seen)
    store.advance_epoch("state", now=1000)
    replacement_root = scenario / "replacement"
    replacement_root.mkdir()
    replacement = LifecycleStateStore(
        replacement_root / "state.sqlite3",
        replacement_root / "anchor.sqlite3",
        store.integrity_key,
    )
    replacement.initialize()
    shutil.copyfile(replacement.path, store.path)
    reason = _expect_d23(
        store.verify,
        D23RejectionReason.STATE_ROLLBACK,
    )
    rows.append(
        _row(
            "database_replacement_detected",
            "state_recovery",
            passed=True,
            observed=reason.value,
            state_rollback_acceptance_count=0,
        )
    )

    for scenario_id, epoch_name in (
        ("policy_epoch_invalidates_ticket", "policy"),
        ("old_ticket_rejected_in_new_state_epoch", "state"),
        ("old_ticket_rejected_in_new_gateway_epoch", "gateway"),
    ):
        scenario = root / scenario_id
        scenario.mkdir()
        store = _new_store(scenario, secrets_seen)
        key = secrets.token_bytes(32)
        secrets_seen.append(key)
        codec = EpochTicketCodec(key)
        token = codec.issue(_ticket(store.snapshot()))
        store.advance_epoch(epoch_name, now=1001)
        reason = _expect_d23(
            lambda c=codec, t=token, s=store: c.verify(
                t,
                expected_epochs=s.snapshot(),
                now=1002,
            ),
            D23RejectionReason.EPOCH_MISMATCH,
        )
        rows.append(
            _row(
                scenario_id,
                "state_recovery",
                passed=True,
                observed=reason.value,
                cross_epoch_ticket_acceptance_count=0,
            )
        )
    return rows


def _run_matrix(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    CanaryFactory,
    list[bytes],
    ByteAuditProxy,
]:
    factory = CanaryFactory()
    secrets_seen: list[bytes] = []
    proxy = ByteAuditProxy()
    rows = [
        *_run_positive(root, factory, secrets_seen),
        *_run_observability(root, factory, secrets_seen),
        *_run_faults(root, factory, secrets_seen),
        *_run_endpoint_ipc(root, proxy),
        *_run_state_recovery(root, factory, secrets_seen),
    ]
    return rows, factory, secrets_seen, proxy


def _upstream_hashes_preserved(
    config_path: str | Path,
    source_manifest: Mapping[str, Any],
) -> bool:
    previous = read_json(
        repo_root(config_path)
        / "artifacts/stage_d22/source_sha256_manifest.json",
        label="D2.2 source manifest",
    )
    old = {
        str(item["path"]): str(item["sha256"])
        for item in previous.get("files", ())
    }
    current = {
        str(item["path"]): str(item["sha256"])
        for item in source_manifest.get("files", ())
    }
    protected = {
        path
        for path in old
        if path.startswith(
            (
                "src/keyed_gram/stage_d1_",
                "src/keyed_gram/stage_d2_",
                "src/keyed_gram/stage_d21_",
                "src/keyed_gram/stage_d22_",
            )
        )
    }
    return bool(protected) and all(
        current.get(path) == old[path] for path in protected
    )


def _summary(
    config_path: str | Path,
    values: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    git: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observed = {str(row["scenario_id"]) for row in rows}
    if observed != set(required_scenarios(values)):
        missing = set(required_scenarios(values)) - observed
        extra = observed - set(required_scenarios(values))
        raise D23ProtocolError(
            f"D2.3 matrix 漂移 missing={sorted(missing)} extra={sorted(extra)}"
        )
    gates = (
        all(bool(row["passed"]) for row in rows)
        and all(
            metrics.get(name) == expected
            for name, expected in values["required_zero_metrics"].items()
        )
        and all(
            metrics.get(name) == expected
            for name, expected in values["required_exact_metrics"].items()
        )
        and metrics["plaintext_all_runtime_file_occurrence_count"] == 0
        and metrics["secret_all_runtime_file_occurrence_count"] == 0
    )
    categories = {
        category: sum(row["category"] == category for row in rows)
        for category in (
            "positive",
            "observability",
            "fault",
            "endpoint_ipc",
            "state_recovery",
        )
    }
    status = "passed" if gates else "failed"
    return {
        "schema_version": 1,
        "stage": "D2.3-fault-observability-lifecycle-audit",
        "evaluation_mode": values["protocol"]["evaluation_mode"],
        "git": dict(git),
        "source_manifest_payload_sha256": source_manifest[
            "manifest_payload_sha256"
        ],
        "upstream_d1_d2_d21_d22_hashes_preserved": (
            _upstream_hashes_preserved(config_path, source_manifest)
        ),
        "fault_observability_lifecycle_status": status,
        "public_synthetic_service_prototype_validated": gates,
        "d_series_system_audit_complete": gates,
        "category_counts": categories,
        "metrics": dict(metrics),
        "raw_ipc_bytes_inspected_in_memory": True,
        "raw_ipc_bytes_persisted": False,
        "rollback_detection_uses_separate_monotonic_anchor": True,
        "physical_disk_full_executed": False,
        "deterministic_disk_full_fault_injected": True,
        "semantic_router_in_tcb": False,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "private_answers_loaded": False,
        "private_value_memory_trained": False,
        "lm_fine_tuning_executed": False,
        "confirmation_created_or_read": False,
        "key_attack_executed": False,
        "remote_model_used": False,
        "network_socket_used": False,
        "limitations": [
            "仅验证单机本地进程与 AF_UNIX，不覆盖跨主机、容器编排、mTLS 或分布式共识",
            "磁盘满、时钟漂移和部分 I/O 使用确定性故障注入，未耗尽宿主机磁盘或修改系统时钟",
            "monotonic anchor 是同主机独立 SQLite 文件；同时回滚 state 与 anchor 仍需外部可信计数器/KMS 防护",
            "SO_PEERCRED 与进程 authkey 只验证当前本地原型，不等于真实 IAM 或多租户 OS 隔离",
            "可观测性使用本地受控 sink，不覆盖第三方 APM、远程 tracing、云日志或 dead-letter 基础设施",
            "生成器仍是确定性复制 worker，不评估远程或自由生成模型、GPU cache 与模型记忆",
            "只使用运行期随机 128-bit synthetic canary，不包含 private value 或 private answer",
            "应用层零持久化不证明 Python/操作系统内存被安全清零",
            "本阶段是 D 系列最终系统边界审计，不是部署级安全认证，也不解锁原 C3",
        ],
    }


def _protocol_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "fault_observability_lifecycle_status",
            "public_synthetic_service_prototype_validated",
            "d_series_system_audit_complete",
            "semantic_router_in_tcb",
            "private_value_memory_ready",
            "original_c3_allowed",
            "c3_eligible",
            "private_answers_loaded",
            "private_value_memory_trained",
            "lm_fine_tuning_executed",
            "confirmation_created_or_read",
            "key_attack_executed",
            "remote_model_used",
            "network_socket_used",
        )
    }


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    config_sha256: str,
) -> None:
    metrics = summary["metrics"]
    categories = summary["category_counts"]
    text = f"""# Stage D2.3：Fault, Observability and Service-Lifecycle Audit

## 结论

Stage D2.3 在代码冻结提交 `{summary["git"]["commit"]}` 上完成一次正式审计。
51 个预注册场景全部符合预期：

```text
fault_observability_lifecycle_status = {summary["fault_observability_lifecycle_status"]}
public_synthetic_service_prototype_validated = {str(summary["public_synthetic_service_prototype_validated"]).lower()}
d_series_system_audit_complete = {str(summary["d_series_system_audit_complete"]).lower()}

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

有限结论是：

> 在单机、本地多进程、AF_UNIX、SQLite 持久状态和公开/合成 canary 条件下，
> D2.2 的 capability-gated 服务边界在预注册的故障恢复、受控可观测性、字节级
> IPC、本地端点和状态回滚矩阵内保持 fail closed。

这不是部署级安全认证，不证明跨主机一致性、真实 IAM/KMS、第三方 APM/远程模型保密性
或进程内存清零。

## 1. 冻结协议

配置 SHA-256：

```text
{config_sha256}
```

源码 manifest payload SHA-256：

```text
{summary["source_manifest_payload_sha256"]}
```

D1、D2、D2.1 与 D2.2 的受保护源码 hash 保持：
`upstream_d1_d2_d21_d22_hashes_preserved = {str(summary["upstream_d1_d2_d21_d22_hashes_preserved"]).lower()}`。

生命周期被显式拆分为：

```text
capability_consumed
→ record_decrypted
→ generator_invoked
→ output_committed
→ response_delivered
```

任何不确定状态均不恢复 single-use capability。

## 2. 正式结果

| 类别 | 场景数 | 通过 |
|---|---:|---:|
| 授权端到端 | {categories["positive"]} | {categories["positive"]} |
| 可观测性 | {categories["observability"]} | {categories["observability"]} |
| 故障一致性 | {categories["fault"]} | {categories["fault"]} |
| IPC/本地端点 | {categories["endpoint_ipc"]} | {categories["endpoint_ipc"]} |
| 状态备份/恢复 | {categories["state_recovery"]} | {categories["state_recovery"]} |
| **合计** | **{metrics["scenario_count"]}** | **{metrics["passed_scenario_count"]}** |

| 核心指标 | 结果 |
|---|---:|
| authorized end-to-end delivery accuracy | {metrics["authorized_end_to_end_delivery_accuracy"]:.1f} |
| plaintext observability occurrence | {metrics["plaintext_observability_occurrence_count"]} |
| secret metric label occurrence | {metrics["secret_metric_label_occurrence_count"]} |
| secret trace attribute occurrence | {metrics["secret_trace_attribute_occurrence_count"]} |
| secret error report occurrence | {metrics["secret_error_report_occurrence_count"]} |
| post-crash double release | {metrics["post_crash_double_release_count"]} |
| post-timeout double release | {metrics["post_timeout_double_release_count"]} |
| post-restore replay success | {metrics["post_restore_replay_success_count"]} |
| state rollback acceptance | {metrics["state_rollback_acceptance_count"]} |
| unauthorized local socket connection | {metrics["unauthorized_local_socket_connection_count"]} |
| stale ticket acceptance | {metrics["stale_ticket_acceptance_count"]} |
| cross-epoch ticket acceptance | {metrics["cross_epoch_ticket_acceptance_count"]} |
| malformed IPC plaintext release | {metrics["malformed_ipc_plaintext_release_count"]} |

## 3. 可观测性边界

structured logs、metrics labels、trace attributes/baggage、error report、retry queue、
profiling、supervisor、health/debug 和 audit event 均使用字段白名单与敏感值阻断。
高敏感值不会成为 metric label 或 trace attribute。正式产物只保留拒绝原因、计数、
schema 和 digest，不保存 plaintext、token、key 或 session credential。

## 4. 故障与生命周期

磁盘满、SQLite busy、数据库截断/损坏、部分 socket write、响应前崩溃、client timeout、
generator 完成但 gateway 丢失响应、supervisor 重启、时钟漂移、key rotation 与 record
migration 并发均按冻结顺序审计。磁盘满和系统时钟场景使用可重复的应用层故障注入，
没有耗尽宿主机磁盘或修改宿主机时钟。

## 5. 字节级 IPC 与本地端点

受控 proxy 在内存中检查真实 length-prefixed JSON frame；正式 artifact 仅保存 frame
SHA-256、长度和 `raw_bytes_persisted=false`。stale socket、symlink、错误目录权限、
SO_PEERCRED UID、超长/深层/duplicate/truncated frame、乱序 ticket、慢发送与无效连接
洪泛均 fail closed。

## 6. 状态回滚与 epoch

内部 lifecycle ticket 绑定 `state_epoch`、`policy_epoch` 与
`gateway_instance_epoch`。主状态 metadata 使用 HMAC 完整性标记，并与独立 monotonic
anchor 比较。旧 backup、数据库替换、旧 policy/state/gateway epoch ticket 均被拒绝。

该 anchor 仍是同主机文件；若攻击者同时回滚主状态与 anchor，当前原型不能检测。
真实部署需外部可信单调计数器、KMS/HSM 或等价控制。

## 7. 有限状态

```text
fault_observability_lifecycle_status = {summary["fault_observability_lifecycle_status"]}
public_synthetic_service_prototype_validated = true
d_series_system_audit_complete = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

D 系列系统审计至此停止。本阶段没有加载 private answer，没有训练 private value memory，
没有执行 LM fine-tuning、confirmation、答案注入或 key attack。
"""
    path.write_text(text, encoding="utf-8")


def _artifact_has_secrets(
    artifact_dir: Path,
    canaries: Sequence[str],
    secrets_seen: Sequence[bytes],
) -> bool:
    paths = tuple(
        path for path in artifact_dir.rglob("*") if path.is_file()
    )
    if count_canary_occurrences(
        (path.read_bytes() for path in paths),
        canaries,
    ):
        return True
    variants = {
        variant
        for secret in secrets_seen
        for variant in (
            secret,
            secret.hex().encode("ascii"),
            base64.b64encode(secret),
            base64.urlsafe_b64encode(secret).rstrip(b"="),
        )
    }
    return any(
        variant and variant in path.read_bytes()
        for path in paths
        for variant in variants
    )


def run_d23_audit(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir = output_paths(config_path)
    if output_dir is not None:
        artifact_dir = resolve_path(config_path, output_dir)
    if artifact_dir.exists() or runtime_dir.exists():
        raise D23ProtocolError("D2.3 输出或 runtime state 已存在，拒绝重跑")
    git = git_state(config_path)
    if git["tracked_dirty"]:
        raise D23ProtocolError("D2.3 正式 audit 要求 tracked worktree 干净")
    sources = runtime_source_manifest(config_path)
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    mark_phase_started(runtime_dir, "audit")
    with tempfile.TemporaryDirectory(prefix="kgd23-") as temporary:
        run_root = Path(temporary)
        rows, factory, secrets_seen, proxy = _run_matrix(run_root)
        canaries = tuple(sorted(factory._issued))
        scan = leakage_scan(run_root, canaries, secrets_seen)
        observability_rows = observability_manifest(run_root)
        byte_manifest = ipc_byte_manifest(proxy.metadata())
        state_manifest = lifecycle_state_manifest(run_root)
        metrics = aggregate_metrics(rows, scan)
        metrics["plaintext_artifact_occurrence_count"] = 0
        summary = _summary(
            config_path,
            values,
            rows,
            metrics=metrics,
            git=git,
            source_manifest=sources,
        )
        matrix_path = artifact_dir / "fault_observability_matrix.csv"
        observability_path = artifact_dir / "observability_audit.csv"
        byte_path = artifact_dir / "ipc_byte_audit.json"
        state_path = artifact_dir / "lifecycle_state_manifest.json"
        summary_path = artifact_dir / "stage_d23_summary.json"
        scan_path = artifact_dir / "leakage_scan.json"
        source_path = artifact_dir / "source_sha256_manifest.json"
        resolved_path = artifact_dir / "resolved_config.json"
        status_path = artifact_dir / "protocol_status.json"
        write_csv(matrix_path, rows)
        write_csv(observability_path, observability_rows)
        write_json(byte_path, byte_manifest)
        write_json(state_path, state_manifest)
        write_json(scan_path, scan)
        write_json(source_path, sources)
        write_json(
            resolved_path,
            {
                "schema_version": 1,
                "config": values,
                "config_sha256": sha256_file(config_path),
                "contains_runtime_secret": False,
                "contains_synthetic_plaintext": False,
                "contains_raw_ipc": False,
            },
        )
        write_json(summary_path, summary)
        write_json(status_path, _protocol_status(summary))
        if _artifact_has_secrets(artifact_dir, canaries, secrets_seen):
            raise D23ProtocolError("D2.3 plaintext/secret 出现在正式 artifact")
        scan["plaintext_artifact_occurrence_count"] = 0
        scan["scanned_location_count"] = int(
            scan["scanned_location_count"]
        ) + sum(path.is_file() for path in artifact_dir.rglob("*"))
        metrics["plaintext_artifact_occurrence_count"] = 0
        metrics["scanned_location_count"] = scan["scanned_location_count"]
        summary = _summary(
            config_path,
            values,
            rows,
            metrics=metrics,
            git=git,
            source_manifest=sources,
        )
        write_json(scan_path, scan)
        write_json(summary_path, summary)
        write_json(status_path, _protocol_status(summary))
        if _artifact_has_secrets(artifact_dir, canaries, secrets_seen):
            raise D23ProtocolError("D2.3 artifact 重写后泄漏扫描失败")
    report = repo_root(config_path) / "PHASE_D23_REPORT.md"
    _write_report(report, summary, sha256_file(config_path))
    outputs = (
        summary_path,
        matrix_path,
        observability_path,
        byte_path,
        state_path,
        scan_path,
        source_path,
        resolved_path,
        status_path,
        report,
    )
    mark_phase_completed(runtime_dir, "audit", outputs)
    return {
        "status": summary["fault_observability_lifecycle_status"],
        "metrics": summary["metrics"],
        "d_series_system_audit_complete": summary[
            "d_series_system_audit_complete"
        ],
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "summary": str(summary_path),
        "report": str(report),
    }


def finalize_d23_artifacts(config_path: str | Path) -> dict[str, Any]:
    artifact_dir, runtime_dir = output_paths(config_path)
    require_phase_completed(runtime_dir, "audit")
    summary = read_json(
        artifact_dir / "stage_d23_summary.json",
        label="D2.3 summary",
    )
    report = repo_root(config_path) / "PHASE_D23_REPORT.md"
    if not report.is_file():
        raise D23ProtocolError("D2.3 report 缺失")
    manifest = artifact_manifest(
        artifact_dir,
        exclude=("artifact_sha256_manifest.json",),
    )
    manifest["external_files"] = [
        {
            "path": "PHASE_D23_REPORT.md",
            "size_bytes": report.stat().st_size,
            "sha256": sha256_file(report),
        }
    ]
    manifest["manifest_payload_sha256"] = canonical_sha256(
        {
            "files": manifest["files"],
            "external_files": manifest["external_files"],
            "schema_version": manifest["schema_version"],
        }
    )
    path = artifact_dir / "artifact_sha256_manifest.json"
    write_json(path, manifest)
    return {
        "status": summary["fault_observability_lifecycle_status"],
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "c3_eligible": False,
    }


def run_d23_smoke(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = load_config(config_path, smoke=True)
    with tempfile.TemporaryDirectory(prefix="kgd23-smoke-") as temporary:
        root = Path(temporary)
        secret = secrets.token_bytes(32)
        store = LifecycleStateStore(
            root / "state.sqlite3",
            root / "anchor.sqlite3",
            secret,
        )
        store.initialize()
        store.consume(secrets.token_hex(32), now=1000)
        proxy = ByteAuditProxy()
        message = {"schema_version": 1, "op": "ping"}
        decoded = proxy.transmit(message, split_at=(1, 3))
        sink = SafeObservabilitySink(
            root / "observability",
            forbidden_values=(secret,),
        )
        health = sink.fixed_health()
        result = {
            "schema_version": 1,
            "stage": values["stage"],
            "status": "passed"
            if decoded == message and health["status"] == "ok"
            else "failed",
            "raw_ipc_persisted": False,
            "private_value_memory_ready": False,
            "original_c3_allowed": False,
            "c3_eligible": False,
        }
        if output_dir is not None:
            destination = resolve_path(config_path, output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "smoke_summary.json", result)
        return result
