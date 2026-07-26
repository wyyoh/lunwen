from __future__ import annotations

from keyed_gram.stage_d23_metrics import aggregate_metrics, ipc_byte_manifest


def test_metrics_preserve_zero_security_counters() -> None:
    rows = [
        {
            "scenario_id": "positive",
            "category": "positive",
            "passed": True,
            "exact_delivery": True,
            "output_digest": "a" * 64,
        }
    ]
    scan = {
        "plaintext_observability_occurrence_count": 0,
        "runtime_canary_count": 1,
        "runtime_canary_variant_count": 8,
        "scanned_location_count": 3,
        "raw_ipc_persisted_count": 0,
    }
    metrics = aggregate_metrics(rows, scan)
    assert metrics["authorized_end_to_end_delivery_accuracy"] == 1.0
    assert metrics["post_crash_double_release_count"] == 0


def test_ipc_manifest_never_contains_raw_frame() -> None:
    manifest = ipc_byte_manifest(
        [
            {
                "frame_index": 0,
                "wire_size_bytes": 12,
                "sha256": "b" * 64,
                "raw_bytes_persisted": False,
            }
        ]
    )
    assert manifest["raw_frame_bytes_persisted"] is False
    assert "raw" not in manifest["frames"][0]
