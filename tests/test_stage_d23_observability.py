from __future__ import annotations

from pathlib import Path

import pytest

from keyed_gram.stage_d23_contract import D23Error
from keyed_gram.stage_d23_observability import SafeObservabilitySink


def test_sensitive_value_never_enters_observability_file(
    tmp_path: Path,
) -> None:
    canary = b"SYN-D23-TEST-CANARY-0123456789"
    sink = SafeObservabilitySink(tmp_path, forbidden_values=(canary,))
    with pytest.raises(D23Error):
        sink.emit(
            "structured_log",
            {"event": "request", "plaintext": canary.decode()},
        )
    assert canary not in (tmp_path / "structured_log.jsonl").read_bytes()


def test_metric_labels_are_fixed_low_cardinality(tmp_path: Path) -> None:
    sink = SafeObservabilitySink(tmp_path)
    sink.metric("requests_total", 1, labels={"role": "gateway"})
    with pytest.raises(D23Error):
        sink.metric("requests_total", 1, labels={"entity_id": "entity-a"})


def test_health_and_debug_have_fixed_schema(tmp_path: Path) -> None:
    sink = SafeObservabilitySink(tmp_path)
    assert sink.fixed_health() == {"schema_version": 1, "status": "ok"}
    assert sink.fixed_debug() == {
        "schema_version": 1,
        "debug": "disabled",
    }
