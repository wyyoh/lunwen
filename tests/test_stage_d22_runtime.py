from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d21_leakage import CanaryFactory
from keyed_gram.stage_d22 import _run_attack_matrix, _summary
from keyed_gram.stage_d22_ipc import read_safe_events
from keyed_gram.stage_d22_metrics import (
    aggregate_metrics,
    ipc_schema_audit,
    leakage_scan,
    process_role_manifest,
)
from keyed_gram.stage_d22_protocol import load_config, runtime_source_manifest
from keyed_gram.stage_d22_runtime import D22Cluster
from keyed_gram.stage_d22_services import (
    GatewayServiceConfig,
    GeneratorServiceConfig,
    MemoryServiceConfig,
)


def test_service_config_types_enforce_key_separation() -> None:
    gateway = {item.name for item in fields(GatewayServiceConfig)}
    memory = {item.name for item in fields(MemoryServiceConfig)}
    generator = {item.name for item in fields(GeneratorServiceConfig)}
    assert "data_key" not in gateway
    assert "capability_key" not in memory
    assert "capability_key" not in generator
    assert "data_key" not in generator
    assert "memory_socket_path" not in generator
    assert "release_ticket_key" not in generator


def test_isolated_cluster_delivers_exact_value(tmp_path: Path) -> None:
    cluster = D22Cluster(tmp_path / "cluster", CanaryFactory())
    try:
        cluster.start_default()
        _, message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        response = cluster.client_call(message)
        result = cluster.parse_success(response)
        assert result.text == cluster.canaries[
            ("entity-a", RelationId.REGISTRY_ID)
        ]
        assert result.exact_delivery
    finally:
        cluster.stop_all()
    events = read_safe_events(cluster.event_path)
    edges = {
        event["edge"]
        for event in events
        if event.get("event") == "ipc_schema"
    }
    assert {
        "client_to_gateway",
        "gateway_to_memory",
        "memory_to_generator",
    } <= edges


def test_replay_never_invokes_generator_again(tmp_path: Path) -> None:
    cluster = D22Cluster(tmp_path / "cluster", CanaryFactory())
    try:
        cluster.start_default()
        _, message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        assert cluster.client_call(message)["status"] == "ok"
        before = sum(
            event.get("event") == "generator_invocation"
            for event in read_safe_events(cluster.event_path)
        )
        replay = cluster.client_call(message)
        after = sum(
            event.get("event") == "generator_invocation"
            for event in read_safe_events(cluster.event_path)
        )
        assert replay["status"] == "reject"
        assert replay["reason"] == "replay"
        assert after == before
    finally:
        cluster.stop_all()


def test_preregistered_attack_matrix_passes_in_development(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    rows, factory, _ = _run_attack_matrix(root)
    assert len(rows) == 35
    assert all(row["passed"] for row in rows)
    scan = leakage_scan(root, tuple(factory._issued))
    ipc = ipc_schema_audit(root)
    roles = process_role_manifest(root)
    metrics = aggregate_metrics(rows, scan, roles, ipc)
    assert metrics["authorized_end_to_end_delivery_accuracy"] == 1.0
    for name in (
        "unauthorized_service_call_count",
        "unauthorized_memory_service_access_count",
        "generator_to_memory_connection_count",
        "cross_process_double_release_count",
        "cross_instance_replay_success_count",
        "post_restart_replay_success_count",
        "plaintext_gateway_log_occurrence_count",
        "plaintext_memory_log_occurrence_count",
        "plaintext_generator_log_occurrence_count",
        "plaintext_ipc_persistence_count",
        "plaintext_crash_recovery_occurrence_count",
        "plaintext_all_runtime_file_occurrence_count",
        "process_surface_secret_occurrence_count",
        "process_surface_canary_occurrence_count",
    ):
        assert metrics[name] == 0
    for name in (
        "capability_key_exposed_to_memory_or_generator",
        "data_key_exposed_to_gateway_or_generator",
        "capability_token_exposed_to_generator",
        "core_dump_enabled_in_service",
    ):
        assert metrics[name] is False
    config = Path(__file__).resolve().parents[1] / "configs/stage_d22.yaml"
    summary = _summary(
        config,
        load_config(config),
        rows,
        metrics=metrics,
        git={
            "commit": "development",
            "branch": "development",
            "tracked_dirty": True,
        },
        source_manifest=runtime_source_manifest(config),
    )
    assert summary["isolated_service_generation_status"] == "passed"
    assert summary["upstream_d1_d2_core_hashes_preserved"] is True
    assert summary["c3_eligible"] is False
