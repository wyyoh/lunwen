from __future__ import annotations

from pathlib import Path

from keyed_gram.stage_d23 import _run_matrix, run_d23_smoke
from keyed_gram.stage_d23_metrics import aggregate_metrics, leakage_scan


def test_full_preregistered_matrix_rehearsal(tmp_path: Path) -> None:
    rows, factory, secrets_seen, proxy = _run_matrix(tmp_path)
    assert len(rows) == 51
    assert len({row["scenario_id"] for row in rows}) == 51
    assert all(row["passed"] for row in rows)
    scan = leakage_scan(tmp_path, tuple(factory._issued), secrets_seen)
    metrics = aggregate_metrics(rows, scan)
    assert metrics["authorized_end_to_end_delivery_accuracy"] == 1.0
    assert metrics["plaintext_observability_occurrence_count"] == 0
    assert metrics["secret_all_runtime_file_occurrence_count"] == 0
    assert proxy.metadata()


def test_smoke_is_ephemeral_and_keeps_c3_closed() -> None:
    result = run_d23_smoke("configs/stage_d23_smoke.yaml")
    assert result["status"] == "passed"
    assert result["raw_ipc_persisted"] is False
    assert result["private_value_memory_ready"] is False
    assert result["c3_eligible"] is False
