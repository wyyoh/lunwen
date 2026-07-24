from __future__ import annotations

from pathlib import Path

from keyed_gram.stage_d21_leakage import (
    CANARY_MARKER,
    CanaryFactory,
    canary_variants,
    count_canary_occurrences,
    scan_paths,
)


def test_factory_issues_unique_128_bit_canaries() -> None:
    factory = CanaryFactory()
    issued = {factory.issue() for _ in range(100)}
    assert len(issued) == 100
    assert all(value.startswith(CANARY_MARKER) for value in issued)
    assert all(len(value.removeprefix(CANARY_MARKER)) == 32 for value in issued)


def test_all_registered_variants_are_detected() -> None:
    canary = CanaryFactory().issue()
    for variant in canary_variants(canary):
        assert count_canary_occurrences((variant,), (canary,)) >= 1


def test_unrelated_payload_has_no_occurrence() -> None:
    canary = CanaryFactory().issue()
    assert count_canary_occurrences(("public harmless text",), (canary,)) == 0


def test_scan_paths_recurses_and_detects_variant(tmp_path: Path) -> None:
    canary = CanaryFactory().issue()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "trace.bin").write_bytes(canary.lower().encode("ascii"))
    assert scan_paths((tmp_path,), (canary,)) >= 1


def test_scan_paths_ignores_missing_path(tmp_path: Path) -> None:
    canary = CanaryFactory().issue()
    assert scan_paths((tmp_path / "absent",), (canary,)) == 0
