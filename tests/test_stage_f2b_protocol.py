from __future__ import annotations

from pathlib import Path

import pytest

from keyed_gram.stage_f2b import run_smoke
from keyed_gram.stage_f2b_protocol import (
    F2BProtocolError,
    load_config,
    read_json,
    record_phase,
    safe_relative,
    verify_frozen_files,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_f2b.yaml"


def test_config_and_frozen_hashes_are_valid() -> None:
    values = load_config(CONFIG)
    frozen = verify_frozen_files(CONFIG, values)
    assert frozen["status"] == "passed"
    assert frozen["f1_locked_test_scored"] is False
    assert frozen["f2a_locked_test_rerun"] is False


def test_frozen_hash_mismatch_is_rejected() -> None:
    values = load_config(CONFIG)
    values["frozen_analyzer"][0]["sha256"] = "0" * 64
    with pytest.raises(F2BProtocolError, match="hash mismatch"):
        verify_frozen_files(CONFIG, values)


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(F2BProtocolError, match="duplicate"):
        read_json(duplicate)
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(F2BProtocolError, match="非法常量"):
        read_json(invalid)


def test_safe_relative_rejects_escape() -> None:
    with pytest.raises(F2BProtocolError, match="安全相对路径"):
        safe_relative("../locked.jsonl", "test")
    with pytest.raises(F2BProtocolError, match="安全相对路径"):
        safe_relative("/tmp/locked.jsonl", "test")


def test_phase_order_and_repeat_are_rejected(tmp_path: Path) -> None:
    write_json(
        tmp_path / "protocol_state.json",
        {
            "schema_version": 1,
            "formal_build": {"status": "started"},
            "phases": {
                "prepare": 0,
                "calibration": 0,
                "development": 0,
                "locked_test": 0,
            },
        },
    )
    with pytest.raises(F2BProtocolError, match="前序"):
        record_phase(tmp_path, "locked_test")
    record_phase(tmp_path, "prepare")
    with pytest.raises(F2BProtocolError, match="重复"):
        record_phase(tmp_path, "prepare")


def test_smoke_never_opens_development_or_locked() -> None:
    result = run_smoke()
    assert result["status"] == "passed"
    assert result["development_scored"] is False
    assert result["locked_test_scored"] is False
    assert result["private_data_used"] is False
