from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from keyed_gram.authority_flow.counterexamples import counterexample_registry
from keyed_gram.authority_flow.explorer import run_bounded_model
from keyed_gram.stage_f2a import (
    F2AProtocolError,
    build_novelty_gate,
    load_config,
    verify_f1_frozen,
)


def test_config_preserves_f2a_boundaries() -> None:
    values = load_config("configs/stage_f2a.yaml")
    protocol = values["protocol"]
    assert protocol["f1_locked_case_content_access_allowed"] is False
    assert protocol["f1_locked_test_scored"] is False
    assert protocol["real_llm_allowed"] is False
    assert protocol["real_agent_allowed"] is False
    assert protocol["private_data_allowed"] is False


def test_relaxed_protocol_is_rejected(tmp_path: Path) -> None:
    values = yaml.safe_load(Path("configs/stage_f2a.yaml").read_text())
    values["protocol"]["f1_locked_case_content_access_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(F2AProtocolError):
        load_config(path)


def test_f1_frozen_assets_still_match_without_opening_locked_cases() -> None:
    values = load_config("configs/stage_f2a.yaml")
    result = verify_f1_frozen(Path.cwd(), values)
    assert result["f1_source_mismatch_count"] == 0
    assert result["f1_artifact_inventory_verified"] is True
    assert result["locked_test_case_content_read"] is False
    assert result["locked_test_scored"] is False


def test_novelty_gate_is_conditional_and_does_not_claim_ccf_b() -> None:
    bounded = run_bounded_model()
    cases = [item.canonical() for item in counterexample_registry()]
    upstream = {
        "f1_source_mismatch_count": 0,
        "locked_test_scored": False,
    }
    result = build_novelty_gate(bounded, cases, upstream)
    assert result["ready_for_executable_authority_calculus"] is True
    assert result["novelty_gate_status"].startswith("conditional_pass")
    assert result["ccf_b_novelty_established"] is False
    assert result["prior_work_overlap_risk"] == "high"


def test_strict_json_rejects_nan_and_duplicate_key() -> None:
    with pytest.raises(ValueError):
        json.dumps({"bad": float("nan")}, allow_nan=False)
    with pytest.raises(ValueError):
        json.loads(
            '{"a":1,"a":2}',
            object_pairs_hook=lambda pairs: _strict_pairs(pairs),
        )


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result
