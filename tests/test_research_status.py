from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "configs" / "research_status.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_c25_router_research_is_permanently_stopped() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    router = status["router_research"]
    assert (
        router["selective_router_research_status"]
        == "stopped_external_validation_failed"
    )
    assert router["pairwise_evidence_branch_status"] == "stopped"
    assert router["set_valued_ambiguity_branch_status"] == "stopped"
    assert router["additional_router_complexity_allowed"] is False
    assert router["task_specific_router_ready"] is False
    assert router["exploratory_c3_allowed"] is False
    assert router["c3_eligible"] is False


def test_stopped_followups_cover_router_complexity_and_gate_relaxation() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    prohibited = set(status["prohibited_followups"])
    assert "R4_or_R5_router" in prohibited
    assert "cross_encoder_router" in prohibited
    assert "additional_threshold_search" in prohibited
    assert "additional_conformal_variant" in prohibited
    assert "MASSIVE_router_extension" in prohibited
    assert "lowered_coverage_gate" in prohibited


def test_stop_record_is_bound_to_frozen_c25_evidence() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    evidence = status["evidence"]
    for declaration in evidence.values():
        if not isinstance(declaration, dict):
            continue
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _sha256(path) == declaration["sha256"]


def test_d1_moves_semantic_router_outside_tcb() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    findings = status["preserved_findings"]
    next_architecture = status["next_architecture"]
    assert findings["typed_memory_contract_structural_isolation"] is True
    assert findings["semantic_router_in_trusted_computing_base"] is False
    assert next_architecture["stage"] == "D1"
    assert next_architecture["natural_language_router_role"] == (
        "untrusted_suggestion_only"
    )
    assert next_architecture["authorization_mechanism"] == (
        "typed_authenticated_capability"
    )
    assert next_architecture["private_value_memory_allowed"] is False

