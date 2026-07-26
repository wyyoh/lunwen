from __future__ import annotations

from pathlib import Path

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d21_leakage import CanaryFactory, canary_variants
from keyed_gram.stage_d22_metrics import (
    aggregate_metrics,
    ipc_schema_audit,
    leakage_scan,
    persistent_state_manifest,
    process_role_manifest,
)
from keyed_gram.stage_d22_runtime import D22Cluster


def test_metrics_use_runtime_canaries_without_persisting_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    factory = CanaryFactory()
    cluster = D22Cluster(root / "s00", factory)
    try:
        cluster.start_default()
        _, message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        assert cluster.client_call(message)["status"] == "ok"
    finally:
        cluster.stop_all()
    rows = [
        {
            "scenario_id": "positive",
            "category": "positive",
            "passed": True,
            "exact_delivery": True,
            "output_digest": "a" * 64,
        }
    ]
    canaries = tuple(factory._issued)
    scan = leakage_scan(root, canaries)
    ipc = ipc_schema_audit(root)
    roles = process_role_manifest(root)
    metrics = aggregate_metrics(rows, scan, roles, ipc)
    assert metrics["scenario_count"] == 1
    assert metrics["passed_scenario_count"] == 1
    assert metrics["authorized_end_to_end_delivery_accuracy"] == 1.0
    assert metrics["runtime_canary_count"] == len(canaries)
    assert metrics["runtime_canary_variant_count"] == len(
        {
            variant
            for canary in canaries
            for variant in canary_variants(canary)
        }
    )
    assert metrics["plaintext_all_runtime_file_occurrence_count"] == 0
    assert metrics["process_surface_secret_occurrence_count"] == 0


def test_persistent_state_manifest_contains_counts_not_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    D22Cluster(root / "s00", CanaryFactory())
    manifest = persistent_state_manifest(root)
    assert manifest["database_count"] > 0
    assert manifest["database_files_committed_to_repository"] is False
    assert all(
        set(database) == {
            "kind",
            "size_bytes",
            "sha256",
            "tables",
            "plaintext_value_stored",
        }
        for database in manifest["databases"]
    )
    assert all(
        set(table) == {"name", "row_count"}
        for database in manifest["databases"]
        for table in database["tables"]
    )
