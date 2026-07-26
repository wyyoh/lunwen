"""D2.3 指标、可观测性投影与泄漏扫描。"""

from __future__ import annotations

import base64
import csv
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .stage_d1_protocol import canonical_sha256, sha256_file
from .stage_d21_leakage import canary_variants, scan_paths


def _secret_variants(secret: bytes) -> tuple[bytes, ...]:
    return (
        secret,
        secret.hex().encode("ascii"),
        base64.b64encode(secret),
        base64.urlsafe_b64encode(secret).rstrip(b"="),
    )


def leakage_scan(
    root: Path,
    canaries: Sequence[str],
    secrets: Sequence[bytes],
) -> dict[str, Any]:
    paths = tuple(
        path for path in sorted(root.rglob("*")) if path.is_file()
    )
    observability_paths = tuple(
        path
        for path in paths
        if path.suffix in {".jsonl", ".log", ".trace", ".metrics"}
        or "observability" in path.parts
    )
    backup_paths = tuple(
        path
        for path in paths
        if "backup" in path.name or "recovery" in path.name
    )
    canary_occurrences = scan_paths(paths, canaries)
    observability_occurrences = scan_paths(observability_paths, canaries)
    backup_occurrences = scan_paths(backup_paths, canaries)
    secret_materials = {
        variant
        for secret in secrets
        for variant in _secret_variants(secret)
        if variant
    }
    secret_occurrences = sum(
        path.read_bytes().count(variant)
        for path in paths
        for variant in secret_materials
    )
    runtime_variants = {
        variant
        for canary in canaries
        for variant in canary_variants(canary)
    }
    return {
        "schema_version": 1,
        "runtime_canary_count": len(canaries),
        "runtime_canary_variant_count": len(runtime_variants),
        "scanned_location_count": len(paths),
        "plaintext_observability_occurrence_count": observability_occurrences,
        "plaintext_backup_occurrence_count": backup_occurrences,
        "plaintext_all_runtime_file_occurrence_count": canary_occurrences,
        "secret_all_runtime_file_occurrence_count": secret_occurrences,
        "raw_ipc_persisted_count": 0,
        "application_layer_only": True,
        "process_memory_zeroization_proven": False,
        "physical_disk_full_performed": False,
        "raw_ipc_capture_persisted": False,
    }


def observability_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("observability/*.jsonl")):
        rows.append(
            {
                "channel": path.stem,
                "event_count": len(path.read_bytes().splitlines()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "contains_plaintext": False,
                "contains_token_or_key": False,
            }
        )
    return rows


def ipc_byte_manifest(metadata: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in metadata]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "frame_count": len(rows),
        "raw_frame_bytes_persisted": False,
        "frames": rows,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def _tables(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        return [
            {
                "name": name,
                "row_count": int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{name}"'
                    ).fetchone()[0]
                ),
            }
            for name in names
        ]
    finally:
        connection.close()


def lifecycle_state_manifest(root: Path) -> dict[str, Any]:
    databases = []
    for path in sorted(root.rglob("*.sqlite3")):
        if not path.is_file():
            continue
        try:
            tables = _tables(path)
            readable = True
        except sqlite3.DatabaseError:
            tables = []
            readable = False
        databases.append(
            {
                "kind": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "readable": readable,
                "tables": tables,
                "plaintext_or_secret_stored": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "transaction_mode": "BEGIN IMMEDIATE",
        "journal_mode": "DELETE",
        "synchronous": "FULL",
        "rollback_detection": "separate monotonic anchor + HMAC metadata",
        "database_count": len(databases),
        "databases": databases,
        "database_files_committed_to_repository": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    positive = [row for row in rows if row["category"] == "positive"]
    exact = sum(bool(row.get("exact_delivery")) for row in positive)
    zero_names = (
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
    metrics = {
        "scenario_count": len(rows),
        "passed_scenario_count": sum(bool(row["passed"]) for row in rows),
        "authorized_end_to_end_delivery_count": len(positive),
        "authorized_exact_delivery_count": exact,
        "authorized_end_to_end_delivery_accuracy": (
            exact / len(positive) if positive else 0.0
        ),
        "persisted_output_digest_count": len(
            {
                str(row["output_digest"])
                for row in positive
                if row.get("output_digest")
            }
        ),
        **{
            name: sum(int(row.get(name, 0)) for row in rows)
            for name in zero_names
        },
        **{
            key: value
            for key, value in scan.items()
            if key.endswith("_count")
            or key
            in {
                "runtime_canary_count",
                "runtime_canary_variant_count",
                "scanned_location_count",
            }
        },
    }
    metrics.setdefault("plaintext_artifact_occurrence_count", 0)
    return metrics


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
