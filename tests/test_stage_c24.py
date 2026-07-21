from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from keyed_gram.cli import build_parser
from keyed_gram.phase_a import _sha256_file
from keyed_gram.stage_c23_audit import fuse_predicted_relation_queries
from keyed_gram.stage_c24 import (
    ProtocolViolation,
    _d0_reproduction,
    _hard_fused_queries,
    _load_prepared_benchmark,
    _tensor_state_sha256,
    _write_json,
    build_fact_buckets,
    compute_eligibility,
    d0_continuous_queries,
    evaluate_bucket_retrieval,
    load_fixed_s5_head,
    load_stage_c24_config,
    prepare_stage_c24,
    run_stage_c24_audit,
)
from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_c24_metrics import select_calibrated_reject_guard


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage_c24.yaml"


def test_config_extends_is_resolved_and_forbidden_inputs_fail_closed():
    formal, _ = load_stage_c24_config(CONFIG)
    smoke, metadata = load_stage_c24_config(ROOT / "configs" / "stage_c24_smoke.yaml")
    assert formal["fixed_s5"] == smoke["fixed_s5"]
    assert formal["fixed_g3"]["selected_candidate"] == "S2:phrase_only"
    assert formal["fixed_g3"] == smoke["fixed_g3"]
    assert smoke["run"]["mode"] == "smoke"
    assert smoke["benchmark"]["public_data_dir"] == "data/stage_c24_smoke"
    assert len(metadata["source_files"]) == 2

    for unsafe in (
        {"answer": "secret"},
        {"answer_text": "secret"},
        {"completion": "secret"},
        {"ground_truth_value": "secret"},
        {"input_path": "artifacts/future-confirmation/rows.jsonl"},
        {"seal_path": "old.json"},
    ):
        broken = dict(formal)
        broken["unsafe"] = unsafe
        path = ROOT / "configs" / "stage_c24.yaml"
        from keyed_gram.stage_c24 import _reject_unsafe_input

        with pytest.raises(ProtocolViolation):
            _reject_unsafe_input(broken, location=str(path))


def test_fixed_head_loader_validates_hash_metadata_and_shapes(tmp_path):
    payload = {
        "format_version": 1,
        "stage": "C2.3-S5-public-ridge-relation-head",
        "base_encoder_variant": "S4",
        "view": "entity_masked_strip_suffix",
        "classes": ["access_code", "city_code", "registry_id"],
        "feature_mean": torch.zeros(4, dtype=torch.float64),
        "feature_scale": torch.ones(4, dtype=torch.float64),
        "weights": torch.ones(5, 3, dtype=torch.float64),
        "regularizer": 0.5,
    }
    path = tmp_path / "head.pt"
    torch.save(payload, path)
    head = load_fixed_s5_head(
        path,
        expected_sha256=_sha256_file(path),
        expected_variant="S4",
        expected_view="entity_masked_strip_suffix",
        expected_classes=payload["classes"],
    )
    assert head.classes == tuple(payload["classes"])
    with pytest.raises(ProtocolViolation, match="SHA-256"):
        load_fixed_s5_head(
            path,
            expected_sha256="0" * 64,
            expected_variant="S4",
            expected_view="entity_masked_strip_suffix",
            expected_classes=payload["classes"],
        )
    with pytest.raises(ProtocolViolation, match="metadata"):
        load_fixed_s5_head(
            path,
            expected_sha256=_sha256_file(path),
            expected_variant="S3",
            expected_view="entity_masked_strip_suffix",
            expected_classes=payload["classes"],
        )


def test_d0_is_exact_c23_function_and_frozen_report_values_reproduce():
    entity = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    logits = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 2.0, 0.0]])
    assert torch.equal(
        d0_continuous_queries(entity, logits, alpha=0.5),
        fuse_predicted_relation_queries(entity, logits, alpha=0.5),
    )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / config["fixed_sources"]["stage_c23_summary"]).read_text(
            encoding="utf-8"
        )
    )
    validation = report["s5"]["fact_geometry"]["validation"]
    development = report["s5"]["fact_geometry"]["development_diagnostic"]
    result = _d0_reproduction(
        validation,
        development,
        config["run"]["c23_d0_reference"],
        tolerance=config["run"]["d0_reproduction_tolerance"],
    )
    assert result["all_metrics_reproduced"] is True
    assert all(item["absolute_error"] == 0.0 for item in result["metrics"].values())


def test_d1_is_hard_and_d2_misroute_never_searches_true_other_bucket():
    entity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = torch.tensor([[8.0, 1.0, 0.0], [0.0, 8.0, 1.0]])
    fused = _hard_fused_queries(entity, logits, alpha=0.5)
    assert fused.shape == (2, 5)
    permuted = _hard_fused_queries(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[9.0, 1.0, 0.0]]),
        alpha=0.5,
        class_order=(
            RelationId.ACCESS_CODE,
            RelationId.CITY_CODE,
            RelationId.REGISTRY_ID,
        ),
    )
    # The high ACCESS_CODE head column is canonicalized to RelationId index 2.
    assert int(permuted[0, -3:].argmax()) == 2

    train_rows = [
        {
            "fact_id": f"{relation.value}-fact",
            "entity": f"{relation.value}-entity",
            "attribute": relation.value,
            "template_id": "train-a",
        }
        for relation in RelationId
    ] * 2
    train_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]] * 2
    )
    buckets = build_fact_buckets(train_embeddings, train_rows)
    query_row = {
        "fact_id": "registry_id-fact",
        "entity": "registry_id-entity",
        "attribute": "registry_id",
        "template_id": "test-a",
    }
    # Head class order below predicts city_code, so the registry fact is absent.
    metrics, predictions = evaluate_bucket_retrieval(
        buckets,
        torch.tensor([[1.0, 0.0]]),
        [query_row],
        torch.tensor([[0.0, 4.0, 1.0]]),
        relation_order=(
            RelationId.ACCESS_CODE,
            RelationId.CITY_CODE,
            RelationId.REGISTRY_ID,
        ),
    )
    assert metrics["misrouted_query_count"] == 1
    assert metrics["centroid_top1_accuracy"] == 0.0
    assert metrics["centroid_mrr"] == 0.0
    assert metrics["cross_relation_candidate_count"] == 0
    assert predictions[0]["candidate_count"] == 1


def test_selection_api_cannot_accept_post_calibration_inputs_and_c3_formula_is_strict():
    parameters = inspect.signature(select_calibrated_reject_guard).parameters
    assert all(
        token not in name.lower()
        for name in parameters
        for token in ("development", "locked", "audit")
    )
    status = compute_eligibility(
        closed_set_discrete_contract_ready=True,
        open_set_reject_guard_ready=True,
        public_benchmark_human_reviewed=False,
        new_pool_created_after_freeze=False,
    )
    assert status["closed_set_discrete_contract_ready"] is True
    assert status["open_set_reject_guard_ready"] is True
    assert status["public_benchmark_human_reviewed"] is False
    assert status["new_confirmation_pool_created_after_freeze"] is False
    assert status["c3_eligible"] is False


def test_strict_json_rejects_nonfinite_answer_fields_and_serializes_protocol(tmp_path):
    for payload in (
        {"bad": float("nan")},
        {"answer": "must-not-write"},
        {"answer_text": "must-not-write"},
        {"completion": "must-not-write"},
        {"ground_truth_value": "must-not-write"},
    ):
        with pytest.raises(ValueError):
            _write_json(tmp_path / "bad.json", payload)
    path = _write_json(
        tmp_path / "good.json",
        {
            "schema_version": 1,
            "private_answers_loaded": False,
            "c3_eligible": False,
        },
    )
    assert json.loads(path.read_text(encoding="utf-8"))["c3_eligible"] is False


def test_core_state_hash_is_stable_across_read_only_bucket_evaluation():
    state = {"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}
    before = _tensor_state_sha256(state)
    rows = [
        {
            "fact_id": relation.value,
            "entity": relation.value,
            "attribute": relation.value,
            "template_id": "t",
        }
        for relation in RelationId
    ]
    build_fact_buckets(torch.eye(3), rows)
    after = _tensor_state_sha256(state)
    assert before == after


def test_prepare_wrapper_writes_pending_review_and_provenance(tmp_path, monkeypatch):
    override = {
        "schema_version": 1,
        "stage": "C2.4-discrete-relation-contract-audit-smoke",
        "extends": str(CONFIG),
        "benchmark": {
            "public_data_dir": str(tmp_path / "data"),
            "manifest": str(tmp_path / "manifest.json"),
            "review_csv": str(tmp_path / "review.csv"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(override, sort_keys=False), encoding="utf-8")
    manifest = prepare_stage_c24(config_path)
    assert manifest["row_counts"] == {
        "public_train": 54,
        "public_calibration": 90,
        "public_locked_audit": 120,
    }
    assert manifest["public_benchmark_human_reviewed"] is False
    assert manifest["historical_c23_novelty_audit"]["passed"] is True
    assert manifest["historical_c23_novelty_audit"]["phrase_audit"][
        "collision_count"
    ] == 0
    assert manifest["provenance"]["model"]["revision"] == (
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )
    json.dumps(manifest, allow_nan=False)
    with (tmp_path / "review.csv").open(encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))
    assert len(review) == 120
    assert {row["review_status"] for row in review} == {"pending"}
    assert {row["reviewer_1_label"] for row in review} == {""}
    assert {row["reviewer_2_label"] for row in review} == {""}

    import keyed_gram.stage_c24 as stage_c24

    observed_paths = []
    original_read = stage_c24._read_jsonl

    def observed_read(path):
        observed_paths.append(Path(path).name)
        return original_read(path)

    monkeypatch.setattr(stage_c24, "_read_jsonl", observed_read)
    resolved, _ = load_stage_c24_config(config_path)
    early_rows, _ = _load_prepared_benchmark(resolved["benchmark"])
    assert set(early_rows) == {"public_train", "public_calibration"}
    assert observed_paths == ["public_train.jsonl", "public_calibration.jsonl"]
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        prepare_stage_c24(config_path)


def test_cli_exposes_only_fixed_stage_c24_controls_and_completed_output_is_immutable(tmp_path):
    parser = build_parser()
    prepare = parser.parse_args(["stage-c24-prepare", "--config", "c.yaml"])
    assert prepare.config == "c.yaml"
    audit = parser.parse_args(
        [
            "stage-c24-audit",
            "--config",
            "c.yaml",
            "--output-dir",
            "out",
            "--device",
            "cpu",
        ]
    )
    assert vars(audit).keys() == {"command", "config", "output_dir", "device", "func"}

    output = tmp_path / "completed"
    output.mkdir()
    (output / "stage_c24_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        run_stage_c24_audit("missing.yaml", output, device_name="cpu")
    assert list(output.iterdir()) == [output / "stage_c24_summary.json"]
