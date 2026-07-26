"""Stage D2.2 聚合指标、泄漏扫描与安全 artifact 投影。"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .stage_d1_protocol import canonical_sha256, sha256_file
from .stage_d21_leakage import (
    canary_variants,
    scan_paths,
)
from .stage_d22_ipc import read_safe_events


def _events(root: Path) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.glob("s*/events.jsonl")):
        scenario_root = path.parent.name
        output.extend(
            (scenario_root, event)
            for event in read_safe_events(path)
        )
    return output


def _paths(root: Path, pattern: str) -> list[Path]:
    return [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file()
    ]


def _scan_count(paths: Iterable[Path], canaries: Sequence[str]) -> int:
    return scan_paths(tuple(paths), canaries)


def leakage_scan(
    root: Path,
    canaries: Sequence[str],
) -> dict[str, Any]:
    gateway_logs = _paths(root, "s*/gateway-*.log")
    memory_logs = _paths(root, "s*/memory-*.log")
    generator_logs = _paths(root, "s*/generator-*.log")
    event_paths = _paths(root, "s*/events.jsonl")
    database_paths = _paths(root, "s*/*.sqlite3")
    recovery_paths = [
        path
        for path in sorted(root.glob("s*/*"))
        if path.is_file()
        and path.suffix in {".log", ".jsonl", ".sqlite3", ".sock"}
    ]
    all_files = [
        path for path in sorted(root.rglob("*")) if path.is_file()
    ]
    events = _events(root)
    process_surface_scans = [
        event
        for _, event in events
        if event.get("event") == "process_surface_scan"
    ]
    runtime_variants = {
        variant
        for canary in canaries
        for variant in canary_variants(canary)
    }
    return {
        "schema_version": 1,
        "persisted_output_digest_count": 0,
        "runtime_canary_count": len(canaries),
        "runtime_canary_variant_count": len(runtime_variants),
        "scanned_location_count": len(all_files)
        + sum(
            int(event.get("scanned_process_count", 0)) * 2
            for event in process_surface_scans
        ),
        "plaintext_gateway_log_occurrence_count": _scan_count(
            gateway_logs, canaries
        ),
        "plaintext_memory_log_occurrence_count": _scan_count(
            memory_logs, canaries
        ),
        "plaintext_generator_log_occurrence_count": _scan_count(
            generator_logs, canaries
        ),
        "plaintext_ipc_persistence_count": _scan_count(
            event_paths, canaries
        ),
        "plaintext_crash_recovery_occurrence_count": _scan_count(
            recovery_paths, canaries
        ),
        "plaintext_all_runtime_file_occurrence_count": _scan_count(
            all_files, canaries
        ),
        "plaintext_persistent_state_occurrence_count": _scan_count(
            database_paths, canaries
        ),
        "process_surface_secret_occurrence_count": sum(
            int(event.get("secret_occurrence_count", 0))
            for event in process_surface_scans
        ),
        "process_surface_canary_occurrence_count": sum(
            int(event.get("canary_occurrence_count", 0))
            for event in process_surface_scans
        ),
        "core_dump_files_found": any(
            path.name.startswith("core") for path in all_files
        ),
        "application_layer_only": True,
        "python_or_accelerator_memory_zeroization_proven": False,
        "raw_ipc_packet_capture_performed": False,
    }


def ipc_schema_audit(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for scenario_root, event in _events(root):
        if event.get("event") != "ipc_schema":
            continue
        key = (
            event.get("edge"),
            tuple(event.get("top_level_fields", ())),
            tuple(
                (
                    str(name),
                    tuple(fields),
                )
                for name, fields in sorted(
                    event.get("nested_fields", {}).items()
                )
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "scenario_root": scenario_root,
                "edge": event["edge"],
                "top_level_fields": "|".join(
                    event["top_level_fields"]
                ),
                "nested_fields": "|".join(
                    f"{name}:{','.join(fields)}"
                    for name, fields in sorted(
                        event["nested_fields"].items()
                    )
                ),
                "contains_capability_token_field": event[
                    "contains_capability_token_field"
                ],
                "contains_release_ticket_field": event[
                    "contains_release_ticket_field"
                ],
                "contains_value_field": event["contains_value_field"],
                "contains_natural_language_prompt_field": event[
                    "contains_natural_language_prompt_field"
                ],
                "contains_capability_hmac_key_field": event[
                    "contains_capability_hmac_key_field"
                ],
                "contains_data_encryption_key_field": event[
                    "contains_data_encryption_key_field"
                ],
            }
        )
    return rows


def process_role_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_root, event in _events(root):
        if event.get("event") != "role_manifest":
            continue
        rows.append(
            {
                "scenario_root": scenario_root,
                "role": event["role"],
                "holds_capability_hmac_key": event[
                    "holds_capability_hmac_key"
                ],
                "holds_release_ticket_key": event[
                    "holds_release_ticket_key"
                ],
                "holds_data_encryption_key": event[
                    "holds_data_encryption_key"
                ],
                "has_gateway_endpoint": event["has_gateway_endpoint"],
                "has_memory_endpoint": event["has_memory_endpoint"],
                "has_generator_endpoint": event["has_generator_endpoint"],
                "core_dump_enabled": event["core_dump_enabled"],
                "secret_in_command_line": event["secret_in_command_line"],
                "secret_in_environment": event["secret_in_environment"],
            }
        )
    return rows


def _database_tables(path: Path) -> list[dict[str, Any]]:
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


def persistent_state_manifest(root: Path) -> dict[str, Any]:
    databases = []
    for path in _paths(root, "s*/*.sqlite3"):
        databases.append(
            {
                "kind": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tables": _database_tables(path),
                "plaintext_value_stored": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "transaction_mode": "BEGIN IMMEDIATE",
        "journal_mode": "DELETE",
        "synchronous": "FULL",
        "database_count": len(databases),
        "databases": databases,
        "database_files_committed_to_repository": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    scan: Mapping[str, Any],
    role_rows: Sequence[Mapping[str, Any]],
    ipc_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive = [row for row in rows if row["category"] == "positive"]
    exact = sum(
        bool(row.get("exact_delivery"))
        for row in positive
    )
    generator_edges = [
        row for row in ipc_rows if row["edge"] == "memory_to_generator"
    ]
    metrics = {
        "scenario_count": len(rows),
        "passed_scenario_count": sum(bool(row["passed"]) for row in rows),
        "authorized_end_to_end_delivery_count": len(positive),
        "authorized_exact_delivery_count": exact,
        "authorized_end_to_end_delivery_accuracy": (
            exact / len(positive) if positive else 0.0
        ),
        "unauthorized_service_call_count": sum(
            int(row.get("unauthorized_service_call_count", 0))
            for row in rows
        ),
        "unauthorized_memory_service_access_count": sum(
            int(row.get("unauthorized_memory_service_access_count", 0))
            for row in rows
        ),
        "generator_to_memory_connection_count": sum(
            int(row.get("generator_to_memory_connection_count", 0))
            for row in rows
        ),
        "cross_process_double_release_count": sum(
            int(row.get("cross_process_double_release_count", 0))
            for row in rows
        ),
        "cross_instance_replay_success_count": sum(
            int(row.get("cross_instance_replay_success_count", 0))
            for row in rows
        ),
        "post_restart_replay_success_count": sum(
            int(row.get("post_restart_replay_success_count", 0))
            for row in rows
        ),
        "capability_key_exposed_to_memory_or_generator": any(
            bool(row["holds_capability_hmac_key"])
            for row in role_rows
            if row["role"] in {"memory", "generator"}
        ),
        "data_key_exposed_to_gateway_or_generator": any(
            bool(row["holds_data_encryption_key"])
            for row in role_rows
            if row["role"] in {"gateway", "generator"}
        ),
        "capability_token_exposed_to_generator": any(
            bool(row["contains_capability_token_field"])
            for row in generator_edges
        ),
        "core_dump_enabled_in_service": any(
            bool(row["core_dump_enabled"]) for row in role_rows
        ),
        **{
            key: value
            for key, value in scan.items()
            if key.endswith("_count")
            or key
            in {
                "runtime_canary_count",
                "runtime_canary_variant_count",
                "scanned_location_count",
                "persisted_output_digest_count",
            }
        },
    }
    metrics["persisted_output_digest_count"] = len(
        {
            str(row["output_digest"])
            for row in rows
            if row.get("output_digest")
        }
    )
    return metrics


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {str(key) for row in rows for key in row}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
