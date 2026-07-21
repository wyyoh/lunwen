from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from keyed_gram.cli import build_parser, main
from keyed_gram.stage_c24b import (
    ProtocolViolation,
    ReviewIncompleteError,
    _r3_candidates_with_margin,
    _runtime_source_manifest,
    _assert_safe_output_json,
    _verify_answer_free_cache_binding,
    _verify_declared_fixed_sources,
    _verify_split_seal,
    audit_stage_c24b,
    calibrate_stage_c24b,
    prepare_stage_c24b,
    validate_stage_c24b_reviews,
)
from keyed_gram.stage_c24b_benchmark import PublicSplitV2, REVIEW_FIELDS, sha256_file
from keyed_gram.stage_c24 import build_fact_buckets
from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_c24_contract import AcceptedRoute
from keyed_gram.stage_c24b_router import RejectedRoute


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "stage_c24b_smoke.yaml"


def _formal_fixture(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "artifacts" / "stage_c23").mkdir(parents=True)
    (tmp_path / "artifacts" / "stage_c24").mkdir(parents=True)
    shutil.copy2(ROOT / "configs" / "stage_c23.yaml", tmp_path / "configs")
    shutil.copy2(ROOT / "configs" / "stage_c24.yaml", tmp_path / "configs")
    shutil.copy2(
        ROOT / "artifacts" / "stage_c23" / "protocol_incident.json",
        tmp_path / "artifacts" / "stage_c23" / "protocol_incident.json",
    )
    shutil.copy2(
        ROOT / "artifacts" / "stage_c24" / "public_locked_audit_review.csv",
        tmp_path / "artifacts" / "stage_c24" / "public_locked_audit_review.csv",
    )
    shutil.copy2(
        ROOT / "artifacts" / "stage_c24" / "public_benchmark_manifest.json",
        tmp_path / "artifacts" / "stage_c24" / "public_benchmark_manifest.json",
    )
    config = yaml.safe_load(
        (ROOT / "configs" / "stage_c24b.yaml").read_text(encoding="utf-8")
    )
    path = tmp_path / "configs" / "stage_c24b.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepare_stage_c24b(path)
    return path


def _complete_review(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        label = row["proposed_label"]
        sample_type = row["sample_type"]
        row.update(
            {
                "reviewer_1_label": label,
                "reviewer_2_label": label,
                "reviewer_1_type": sample_type,
                "reviewer_2_type": sample_type,
                "adjudicated_label": label,
                "adjudicated_type": sample_type,
                "notes": "",
                "review_status": "complete",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _complete_all_reviews(config_path: Path) -> Path:
    artifact_dir = config_path.parent.parent / "artifacts" / "stage_c24b"
    for name in (
        "public_train_v2_review.csv",
        "public_calibration_v2_review.csv",
        "public_locked_audit_v2_review.csv",
        "c24_locked_audit_independent_review.csv",
    ):
        _complete_review(artifact_dir / name)
    return artifact_dir


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("stage_c24b_smoke")
    summary = audit_stage_c24b(SMOKE_CONFIG, output, device_name="cpu")
    assert summary["status"] == "smoke_completed_formal_readiness_not_assessed"
    return output


def test_cli_exposes_all_four_stage_c24b_commands():
    parser = build_parser()
    for command in (
        "stage-c24b-prepare",
        "stage-c24b-validate-review",
        "stage-c24b-calibrate",
        "stage-c24b-audit",
    ):
        arguments = [command, "--config", "fixture.yaml"]
        if command in {"stage-c24b-calibrate", "stage-c24b-audit"}:
            arguments += ["--output-dir", "out"]
        assert parser.parse_args(arguments).command == command


def test_smoke_generates_finite_complete_artifacts_and_never_claims_readiness(
    smoke_output: Path,
):
    required = {
        "stage_c24b_summary.json",
        "stage_c24b_ablation.csv",
        "resolved_config.json",
        "protocol_status.json",
        "route_candidate_scores.csv",
        "router_selection.csv",
        "calibration_results.json",
        "route_predictions.csv",
        "retrieval_predictions.json",
        "error_analysis.csv",
        "risk_coverage.csv",
        "artifact_sha256_manifest.json",
    }
    assert required.issubset({path.name for path in smoke_output.iterdir()})
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        evaluation = json.loads(
            (smoke_output / variant / "evaluation.json").read_text(encoding="utf-8")
        )
        assert evaluation["protocol_role"] == "synthetic_integration_only"
        assert evaluation["fact_retrieval"]["cross_relation_candidate_count"] == 0
    summary = json.loads(
        (smoke_output / "stage_c24b_summary.json").read_text(encoding="utf-8")
    )
    assert summary["formal_locked_audit_executed"] is False
    assert summary["formal_research_thresholds_assessed"] is False
    assert summary["readiness"] == {
        "closed_set_selective_router_ready": False,
        "open_set_abstention_ready": False,
        "discrete_memory_contract_preserved": True,
        "public_benchmark_human_reviewed": False,
        "new_confirmation_pool_created_after_freeze": False,
        "ready_to_create_new_confirmation_pool": False,
        "c3_eligible": False,
    }
    # strict JSON parser 接受所有 JSON；产物中没有非标准 NaN/Infinity。
    for path in smoke_output.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_smoke_artifact_manifest_hashes_every_preceding_file(smoke_output: Path):
    manifest = json.loads(
        (smoke_output / "artifact_sha256_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["file_count"] == len(manifest["files"])
    for row in manifest["files"]:
        path = smoke_output / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert sha256_file(path) == row["sha256"]


def test_smoke_freezes_before_first_locked_materialization(monkeypatch, tmp_path: Path):
    import keyed_gram.stage_c24b as module

    events: list[str] = []
    original_split = module.build_public_split_v2
    original_build = module.build_public_benchmark_v2
    original_freeze = module.freeze_router_selection

    def split_spy(config, split):
        events.append(f"split:{PublicSplitV2(split).value}")
        return original_split(config, split)

    def build_spy(*args, **kwargs):
        events.append("full-build-with-locked")
        return original_build(*args, **kwargs)

    def freeze_spy(*args, **kwargs):
        events.append("freeze")
        return original_freeze(*args, **kwargs)

    monkeypatch.setattr(module, "build_public_split_v2", split_spy)
    monkeypatch.setattr(module, "build_public_benchmark_v2", build_spy)
    monkeypatch.setattr(module, "freeze_router_selection", freeze_spy)
    module.audit_stage_c24b(SMOKE_CONFIG, tmp_path / "out", device_name="cpu")
    assert events[:3] == [
        "split:public_train_v2",
        "split:public_calibration_v2",
        "freeze",
    ]
    assert events.index("freeze") < events.index("full-build-with-locked")


def test_prepare_creates_pending_real_review_templates_without_fabrication(tmp_path: Path):
    config = _formal_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "stage_c24b"
    manifest = validate_stage_c24b_reviews(
        config, require_complete=False, write_manifest=False
    )
    assert manifest["status"] == "pending_human_input"
    assert manifest["public_benchmark_human_reviewed"] is False
    assert set(manifest["reviews"]) == {
        "public_train_v2",
        "public_calibration_v2",
        "public_locked_audit_v2",
        "legacy_c24_locked_audit",
    }
    assert all(not item["complete"] for item in manifest["reviews"].values())
    assert all(
        item["model_predictions_used_for_labels"] == "not_verified_by_software"
        for item in manifest["reviews"].values()
    )
    pending_summary = json.loads(
        (artifact_dir / "stage_c24b_summary.json").read_text(encoding="utf-8")
    )
    artifact_manifest = json.loads(
        (artifact_dir / "artifact_sha256_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert pending_summary["formal_locked_audit_executed"] is False
    assert pending_summary["review_manifest_sha256"] == sha256_file(
        artifact_dir / "review_manifest.json"
    )
    assert artifact_manifest["status"] == "prepared_pending_human_review_snapshot"
    assert artifact_manifest["file_count"] == len(artifact_manifest["files"])
    for row in artifact_manifest["files"]:
        path = artifact_dir / row["path"]
        assert sha256_file(path) == row["sha256"]
    summary = json.loads(
        (
            tmp_path / "artifacts" / "stage_c24b" / "stage_c24b_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["formal_locked_audit_executed"] is False
    assert summary["public_benchmark_human_reviewed"] is False
    assert summary["readiness"]["c3_eligible"] is False


def test_pending_review_blocks_calibration_before_source_or_locked_load(
    monkeypatch, tmp_path: Path
):
    import keyed_gram.stage_c24b as module

    config = _formal_fixture(tmp_path)
    source_checks: list[bool] = []

    def forbidden_source(*args, **kwargs):
        source_checks.append(True)
        raise AssertionError("source gate ran before independent review")

    monkeypatch.setattr(module, "_verify_declared_fixed_sources", forbidden_source)
    with pytest.raises(ReviewIncompleteError):
        calibrate_stage_c24b(
            config, tmp_path / "artifacts" / "stage_c24b" / "calibration"
        )
    assert source_checks == []


def test_preflight_missing_or_duplicate_locked_rows_fails_before_jsonl_open(
    monkeypatch, tmp_path: Path
):
    import keyed_gram.stage_c24b as module

    config = _formal_fixture(tmp_path)
    artifact_dir = _complete_all_reviews(config)
    locked_review = artifact_dir / "public_locked_audit_v2_review.csv"
    with locked_review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with locked_review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows[:-1])

    opened: list[str] = []

    def forbidden_jsonl(path):
        opened.append(str(path))
        raise AssertionError("locked JSONL opened before sealed review preflight")

    monkeypatch.setattr(module, "_read_jsonl", forbidden_jsonl)
    with pytest.raises(ProtocolViolation, match="review row set differs"):
        audit_stage_c24b(config, artifact_dir)
    assert opened == []


def test_review_conflict_requires_adjudication_and_ambiguous_cannot_be_unrelated(
    tmp_path: Path,
):
    config = _formal_fixture(tmp_path)
    artifact_dir = _complete_all_reviews(config)
    locked = artifact_dir / "public_locked_audit_v2_review.csv"
    with locked.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ambiguous = next(row for row in rows if row["sample_type"] == "ambiguous")
    ambiguous.update(
        {
            "reviewer_1_label": "unrelated",
            "reviewer_2_label": "unrelated",
            "reviewer_1_type": "unrelated",
            "reviewer_2_type": "unrelated",
            "adjudicated_label": "unrelated",
            "adjudicated_type": "unrelated",
            "notes": "人工判断，但该方向被协议明确禁止",
        }
    )
    with locked.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ProtocolViolation, match="ambiguous-to-unrelated"):
        validate_stage_c24b_reviews(config, require_complete=True, write_manifest=False)


def test_calibration_uses_only_train_calibration_and_legacy_then_missing_source_fails(
    monkeypatch, tmp_path: Path
):
    import keyed_gram.stage_c24b as module

    config = _formal_fixture(tmp_path)
    _complete_all_reviews(config)
    opened: list[str] = []
    original = module._read_jsonl

    def read_spy(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(module, "_read_jsonl", read_spy)
    with pytest.raises(FileNotFoundError, match="fixed Stage C2.4b source is absent"):
        calibrate_stage_c24b(
            config,
            tmp_path / "artifacts" / "stage_c24b" / "calibration",
            device_name="cpu",
        )
    assert not any("public_locked_audit_v2.jsonl" in path for path in opened)
    assert any("public_train_v2.jsonl" in path for path in opened)
    assert any("public_calibration_v2.jsonl" in path for path in opened)


def test_prepared_data_manifest_mutation_fails_closed(tmp_path: Path):
    config = _formal_fixture(tmp_path)
    data = tmp_path / "data" / "stage_c24b" / "public_calibration_v2.jsonl"
    data.write_text(data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    values = yaml.safe_load(config.read_text(encoding="utf-8"))
    with pytest.raises(ProtocolViolation, match="data SHA-256 changed"):
        _verify_split_seal(config, values, PublicSplitV2.CALIBRATION)


def test_fixed_entity_checkpoint_and_source_core_bindings_fail_closed(
    monkeypatch, tmp_path: Path
):
    import keyed_gram.stage_c24b as module

    config_path = tmp_path / "configs" / "stage_c24b.yaml"
    checkpoint = tmp_path / "artifacts" / "entity.pt"
    cache_path = tmp_path / "artifacts" / "answer_free.pt"
    config_path.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config_path.write_text("fixture: true\n", encoding="utf-8")
    checkpoint.write_bytes(b"frozen-entity-checkpoint")
    cache_path.write_bytes(b"answer-free-cache-envelope")
    core_sha = "b" * 64
    values = {
        "fixed_sources": {
            "entity_checkpoint": "artifacts/entity.pt",
            "entity_checkpoint_sha256": sha256_file(checkpoint),
            "private_feature_cache": "artifacts/answer_free.pt",
            "private_feature_cache_sha256": sha256_file(cache_path),
            "source_core_sha256": core_sha,
        }
    }
    before = _verify_declared_fixed_sources(
        config_path, values, lightweight_only=False
    )
    assert before["entity_checkpoint"] == sha256_file(checkpoint)
    checkpoint.write_bytes(b"mutated-entity-checkpoint")
    with pytest.raises(ProtocolViolation, match="entity_checkpoint"):
        _verify_declared_fixed_sources(config_path, values, lightweight_only=False)

    payload = {"source_core_sha256": core_sha}
    monkeypatch.setattr(module, "_load_feature_cache", lambda path: payload)
    monkeypatch.setattr(
        module,
        "validate_answer_free_private_feature_cache",
        lambda value: {"train": 1, "validation": 1, "test": 1},
    )
    binding = _verify_answer_free_cache_binding(config_path, values)
    assert binding["source_core_sha256"] == core_sha
    payload["source_core_sha256"] = "c" * 64
    with pytest.raises(ProtocolViolation, match="source_core"):
        _verify_answer_free_cache_binding(config_path, values)


def test_output_validator_rejects_answer_payload_but_allows_readiness_boolean():
    with pytest.raises(ProtocolViolation, match="answer-bearing"):
        _assert_safe_output_json({"answer_value": "forbidden"})
    _assert_safe_output_json(
        {
            "private_answer_free": True,
            "contains_private_answers": False,
            "ready_to_create_new_confirmation_pool": True,
            "new_confirmation_pool_created_after_freeze": False,
        }
    )


def test_cli_smoke_runs_without_formal_locked_or_human_claim(tmp_path: Path, capsys):
    main(
        [
            "stage-c24b-audit",
            "--config",
            str(SMOKE_CONFIG),
            "--output-dir",
            str(tmp_path / "cli-smoke"),
            "--device",
            "cpu",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["protocol_role"] == "synthetic_integration_only"
    assert output["formal_locked_audit_executed"] is False
    assert output["public_benchmark_human_reviewed"] is False
    assert output["c3_eligible"] is False


def test_r3_margin_is_reflected_as_a_multi_candidate_set():
    scores = {
        RelationId.REGISTRY_ID: 0.60,
        RelationId.CITY_CODE: 0.58,
        RelationId.ACCESS_CODE: 0.10,
    }
    assert _r3_candidates_with_margin(scores, 0.50, 0.05) == frozenset(
        {RelationId.REGISTRY_ID, RelationId.CITY_CODE}
    )


def test_runtime_source_seal_includes_formal_backend_and_direct_dependencies():
    paths = {
        row["path"] for row in _runtime_source_manifest(ROOT / "configs/stage_c24b.yaml")["files"]
    }
    assert {
        "src/keyed_gram/stage_c24b_formal.py",
        "src/keyed_gram/stage_c24_contract.py",
        "src/keyed_gram/stage_c23_semantic.py",
        "src/keyed_gram/canonicalizer.py",
        "src/keyed_gram/phase_a.py",
        "pyproject.toml",
    }.issubset(paths)


def test_formal_development_markers_make_prelocked_run_one_shot(monkeypatch, tmp_path: Path):
    import keyed_gram.stage_c24b as main_module
    import keyed_gram.stage_c24b_formal as formal

    builds: list[bool] = []
    parameters = {
        "selected_r2_head_file_sha256": "a" * 64,
        "selected_r2_head_state_sha256": "b" * 64,
    }
    model_manifest = {
        "model_id": "synthetic/formal-marker-test",
        "revision": "frozen-test-revision",
        "manifest_sha256": "c" * 64,
        "files": [{"path": "model.safetensors", "sha256": "d" * 64}],
    }
    context = {
        "model_manifests": {"R0": model_manifest},
        "parameters": parameters,
    }
    artifact_seal = {
        "encoder_manifests": context["model_manifests"],
        "r2_head_file_sha256": "a" * 64,
        "r2_head_state_sha256": "b" * 64,
    }

    def build(*args, **kwargs):
        builds.append(True)
        return context

    def retrieve(*args, **kwargs):
        assert (tmp_path / "development_started.json").exists()
        assert not (tmp_path / "development_completed.json").exists()
        return (
            {"scope": "answer_free_development"},
            [],
            {
                "manifest": {
                    "scope": "synthetic_locked_slot_binding",
                    "binding_sha256": "e" * 64,
                }
            },
        )

    monkeypatch.setattr(formal, "_build_router_context", build)
    monkeypatch.setattr(formal, "_model_artifact_seal", lambda *args, **kwargs: artifact_seal)
    monkeypatch.setattr(formal, "_actual_answer_free_retrieval", retrieve)
    monkeypatch.setattr(main_module, "assert_discrete_memory_contract", lambda: None)
    frozen = {
        "selected_router": "R3",
        "router_parameters": parameters,
        "git_commit": "1" * 40,
    }
    formal.prepare_formal_prelocked(
        ROOT / "configs/stage_c24b.yaml",
        yaml.safe_load((ROOT / "configs/stage_c24b.yaml").read_text(encoding="utf-8")),
        frozen,
        device_name="cpu",
        destination=tmp_path,
    )
    assert (tmp_path / "development_completed.json").exists()
    with pytest.raises(ProtocolViolation, match="already been started"):
        formal.prepare_formal_prelocked(
            ROOT / "configs/stage_c24b.yaml",
            {},
            frozen,
            device_name="cpu",
            destination=tmp_path,
        )
    assert len(builds) == 1


def test_interrupted_formal_marker_creates_non_overwriting_incident(tmp_path: Path):
    config = _formal_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "stage_c24b"
    (artifact_dir / "development_started.json").write_text(
        '{"status":"interrupted"}\n', encoding="utf-8"
    )
    with pytest.raises(ProtocolViolation, match="permanently invalid"):
        audit_stage_c24b(config, artifact_dir)
    incident = json.loads(
        (artifact_dir / "protocol_incident.json").read_text(encoding="utf-8")
    )
    assert incident["current_locked_audit_v2_permanently_invalid"] is True
    assert incident["next_required_protocol_version"] == "v3_independent_locked_audit"
    assert incident["resolved_config_sha256"]
    assert incident["selection_protocol"]["locked_audit_used_for_selection"] is False
    (artifact_dir / "development_started.json").unlink()
    with pytest.raises(ProtocolViolation, match="permanently invalidated"):
        audit_stage_c24b(config, artifact_dir)


def test_formal_readiness_never_uses_development_fact_diagnostic():
    import keyed_gram.stage_c24b_formal as formal

    values = yaml.safe_load(
        (ROOT / "configs" / "stage_c24b.yaml").read_text(encoding="utf-8")
    )
    evaluation = {
        "known_routing": {
            "relation_family_macro_accuracy": 1.0,
            "worst_family_accuracy": 1.0,
            "accepted_route_accuracy": 1.0,
            "known_coverage": 1.0,
        },
        "access_control": {
            "safe_coverage": 1.0,
            "wrong_bucket_access_rate": 0.0,
            "ambiguous_false_memory_access_rate": 0.0,
            "unrelated_false_memory_access_rate": 0.0,
            "false_memory_access_rate": 0.0,
            "singleton_acceptance_precision": 1.0,
        },
        "reject_quality": {
            "ambiguous_rejection_rate": 1.0,
            "unrelated_rejection_rate": 1.0,
            "worst_reject_family_false_accept_rate": 0.0,
        },
        "fact_retrieval": {
            "accepted_fact_top1": 1.0,
            "all_query_fact_top1": 1.0,
            "mrr": 1.0,
            "cross_relation_candidate_count": 0,
            "evidence_split": "fixed_answer_free_development_diagnostic",
        },
    }
    readiness = formal._readiness(
        values, evaluation, discrete_memory_contract_preserved=True
    )
    assert readiness["closed_set_selective_router_ready"] is False
    assert readiness["open_set_abstention_ready"] is True
    assert readiness["ready_to_create_new_confirmation_pool"] is False
    evaluation["fact_retrieval"]["evidence_split"] = "public_locked_audit_v2"
    locked_ready = formal._readiness(
        values, evaluation, discrete_memory_contract_preserved=True
    )
    assert locked_ready["closed_set_selective_router_ready"] is True
    assert locked_ready["ready_to_create_new_confirmation_pool"] is True
    assert locked_ready["c3_eligible"] is False


def test_formal_calibration_risk_counts_open_set_false_accepts():
    import keyed_gram.stage_c24b_formal as formal

    candidate = {
        "known_coverage": 1.0,
        "accepted_relation_precision": 1.0,
        "singleton_acceptance_precision": 0.75,
        "safe_coverage": 1.0,
        "false_memory_access_rate": 0.5,
        "ambiguous_rejection_rate": 0.5,
        "unrelated_rejection_rate": 0.5,
    }
    evaluations = {
        variant: {
            "reject_quality": {
                "risk_coverage": {
                    "curve": [
                        {
                            "threshold": 0.0,
                            "coverage": 1.0,
                            "risk": 0.25,
                            "accuracy": 0.75,
                        }
                    ]
                }
            }
        }
        for variant in ("R0", "R1", "R2", "R3", "R4")
    }
    rows = formal._formal_risk_rows(
        evaluations,
        {"r3_candidates": [candidate], "r4_candidates": [candidate]},
    )
    calibration = rows[0]
    assert calibration["risk"] == pytest.approx(0.25)
    assert calibration["accuracy"] == pytest.approx(0.75)
    assert calibration["known_route_risk"] == pytest.approx(0.0)


def test_prefrozen_locked_slot_binding_is_relation_blind_and_executes_one_bucket():
    import keyed_gram.stage_c24b_formal as formal

    values = yaml.safe_load(
        (ROOT / "configs" / "stage_c24b.yaml").read_text(encoding="utf-8")
    )
    private_entities = [f"answer-free-entity-{index}" for index in range(4)]
    rows = []
    vectors = []
    for entity_index, entity in enumerate(private_entities):
        vector = torch.nn.functional.one_hot(
            torch.tensor(entity_index), num_classes=4
        ).float()
        for relation in RelationId:
            rows.append(
                {
                    "entity": entity,
                    "attribute": relation.value,
                    "fact_id": f"opaque:{entity_index}:{relation.value}",
                }
            )
            vectors.append(vector)
    embeddings = torch.stack(vectors)
    buckets = build_fact_buckets(embeddings, rows)
    binding = formal._build_locked_slot_binding(
        values,
        embeddings,
        rows,
        buckets,
        answer_free_cache_sha256="a" * 64,
    )
    manifest = binding["manifest"]
    assert manifest["mapped_entity_count"] == 4
    assert manifest["offline_oracle_memory_access_count"] == 12
    assert manifest["offline_oracle_counted_as_system_memory_access"] is False
    assert manifest["historical_development_substrate_reused"] is True
    for public_entity_id in manifest["public_entity_ids"]:
        assert binding["entity_embeddings"][public_entity_id].shape == (4,)
        assert set(binding["fact_ids"][public_entity_id]) == set(RelationId)

    entity_ids = manifest["public_entity_ids"]
    locked_rows = [
        {
            "row_id": "known-correct",
            "entity_id": entity_ids[0],
            "sample_type": "known",
            "relation_id": RelationId.REGISTRY_ID.value,
        },
        {
            "row_id": "known-wrong-bucket",
            "entity_id": entity_ids[1],
            "sample_type": "known",
            "relation_id": RelationId.ACCESS_CODE.value,
        },
        {
            "row_id": "known-reject",
            "entity_id": entity_ids[2],
            "sample_type": "known",
            "relation_id": RelationId.CITY_CODE.value,
        },
        {
            "row_id": "ambiguous-accept",
            "entity_id": entity_ids[3],
            "sample_type": "ambiguous",
            "relation_id": None,
        },
        {
            "row_id": "unrelated-reject",
            "entity_id": entity_ids[0],
            "sample_type": "unrelated",
            "relation_id": None,
        },
    ]
    routes = [
        AcceptedRoute(RelationId.REGISTRY_ID),
        AcceptedRoute(RelationId.REGISTRY_ID),
        RejectedRoute("unknown"),
        AcceptedRoute(RelationId.CITY_CODE),
        RejectedRoute("unknown"),
    ]
    metrics, predictions = formal._evaluate_locked_slot_retrieval(
        locked_rows, routes, binding, buckets, variant="R3"
    )
    assert metrics["accepted_fact_top1"] == pytest.approx(0.5)
    assert metrics["all_query_fact_top1"] == pytest.approx(1 / 3)
    assert metrics["oracle_gap"] == pytest.approx(2 / 3)
    assert metrics["cross_relation_candidate_count"] == 0
    assert metrics["all_accepted_cross_relation_candidate_count"] == 0
    assert predictions[2]["memory_access_count"] == 0
    assert predictions[3]["memory_access_count"] == 1
    assert predictions[3]["target_fact_id"] is None
    assert predictions[3]["predicted_fact_id"] is None
    assert predictions[4]["memory_access_count"] == 0

    tampered = dict(binding)
    tampered["manifest"] = {**binding["manifest"], "binding_sha256": "0" * 64}
    with pytest.raises(ProtocolViolation, match="binding manifest SHA-256"):
        formal._evaluate_locked_slot_retrieval(
            locked_rows, routes, tampered, buckets, variant="R3"
        )
    tensor_tampered = dict(binding)
    tensor_tampered["entity_embeddings"] = dict(binding["entity_embeddings"])
    changed = binding["entity_embeddings"][entity_ids[0]].clone()
    changed[0] += 0.25
    tensor_tampered["entity_embeddings"][entity_ids[0]] = changed
    with pytest.raises(ProtocolViolation, match="entity tensor changed"):
        formal._evaluate_locked_slot_retrieval(
            locked_rows, routes, tensor_tampered, buckets, variant="R3"
        )


def test_prefrozen_locked_slot_binding_fails_if_unique_entities_are_insufficient():
    import keyed_gram.stage_c24b_formal as formal

    values = yaml.safe_load(
        (ROOT / "configs" / "stage_c24b.yaml").read_text(encoding="utf-8")
    )
    rows = []
    vectors = []
    for entity_index in range(3):
        vector = torch.nn.functional.one_hot(
            torch.tensor(entity_index), num_classes=3
        ).float()
        for relation in RelationId:
            rows.append(
                {
                    "entity": f"entity-{entity_index}",
                    "attribute": relation.value,
                    "fact_id": f"opaque:{entity_index}:{relation.value}",
                }
            )
            vectors.append(vector)
    embeddings = torch.stack(vectors)
    buckets = build_fact_buckets(embeddings, rows)
    with pytest.raises(ProtocolViolation, match="fewer entities"):
        formal._build_locked_slot_binding(
            values,
            embeddings,
            rows,
            buckets,
            answer_free_cache_sha256="a" * 64,
        )
