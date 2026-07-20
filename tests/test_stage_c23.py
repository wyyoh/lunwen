from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from keyed_gram.canonicalizer import (
    CanonicalizerConfig,
    CanonicalizerSystem,
    load_canonicalizer_checkpoint,
    save_canonicalizer_checkpoint,
)
from keyed_gram.stage_c21 import _load_feature_cache
from keyed_gram.stage_c23 import (
    answer_free_query_metrics,
    fuse_oracle_queries,
    load_c23_protocol_status,
    oracle_relation_one_hot,
    run_stage_c23_oracle,
    select_oracle_alpha,
    validate_oracle_sources,
    write_answer_free_json,
)


def _private_rows(split: str, templates: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for entity_index in range(4):
        entity = f"entity-{entity_index}"
        for relation_index, relation in enumerate(("access", "registry")):
            for template in templates:
                rows.append(
                    {
                        "fact_id": f"{entity}|{relation}",
                        "entity": entity,
                        "attribute": relation,
                        "template_id": f"{split}-{template}",
                        "prompt": f"prompt {entity} {relation} {template}",
                        "answer": f"secret-{entity_index}-{relation_index}",
                        "candidates": ["secret", "decoy"],
                        "answer_index": 0,
                    }
                )
    return rows


def _feature_tensor(rows: list[dict[str, object]]) -> torch.Tensor:
    features = torch.zeros(len(rows), 3, 1, 4)
    for index, row in enumerate(rows):
        entity_index = int(str(row["entity"]).rsplit("-", 1)[1])
        features[index, 0, 0, entity_index] = 1.0
    return features


def _write_protocol_incident(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "stage": "C2.3-public-semantic-encoder-audit",
        "incident": "local_confirmation_template_selection_read_during_read_only_audit",
        "scope": {
            "templates_read": True,
            "rows_read": False,
            "private_answers_read": False,
            "model_evaluation_run": False,
            "metrics_computed": False,
        },
        "old_seal": {
            # This deliberately does not exist. Loading protocol accounting must
            # not follow or open the retired confirmation seal path.
            "path": str(path.parent / "must-not-be-opened.json"),
            "status": "retired_compromised_for_future_confirmation",
        },
        "access_accounting": {
            "confirmation_template_read_count": 1,
            "confirmation_evaluation_count": 0,
            "confirmation_metric_access_count": 0,
        },
        "new_confirmation_status": "not_created",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_sources(tmp_path: Path) -> tuple[Path, Path]:
    split_rows = {
        "train": _private_rows("train", ("t0", "t1")),
        "validation": _private_rows("validation", ("v0", "v1")),
        "test": _private_rows("test", ("d0", "d1")),
    }
    cache = tmp_path / "features.pt"
    torch.save(
        {
            "format_version": 1,
            "source_core_sha256": "core-sha",
            "selected_layers_one_based": [1],
            "selected_layer_indices": [0],
            "q0_baseline_layer_index": 0,
            "pool_names": [
                "entity_span",
                "question_without_entity",
                "answer_position",
            ],
            "features": {
                split: {
                    "pools": _feature_tensor(rows),
                    "baseline_last": torch.zeros(len(rows), 4),
                }
                for split, rows in split_rows.items()
            },
            "metadata": split_rows,
        },
        cache,
    )

    config = CanonicalizerConfig(
        architecture="factorized",
        num_input_layers=1,
        core_hidden_size=4,
        query_dim=4,
        mlp_hidden_size=4,
        dropout=0.0,
        num_entities=4,
        num_relations=2,
        num_templates=2,
        template_adversary_source="relation",
    )
    system = CanonicalizerSystem(config)
    assert system.canonicalizer.entity_encoder is not None
    with torch.no_grad():
        first = system.canonicalizer.entity_encoder[0]
        second = system.canonicalizer.entity_encoder[3]
        assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
        first.weight.copy_(torch.eye(4))
        first.bias.zero_()
        second.weight.copy_(torch.eye(4))
        second.bias.zero_()
    labels = {
        "entities": [f"entity-{index}" for index in range(4)],
        "relations": ["access", "registry"],
        "templates": ["train-t0", "train-t1"],
        "facts": sorted(
            {
                str(row["fact_id"])
                for row in split_rows["train"]
            }
        ),
    }
    checkpoint = save_canonicalizer_checkpoint(
        tmp_path / "r3.pt",
        system,
        variant="R3",
        selected_core_layers=[0],
        labels=labels,
        source_core_sha256="core-sha",
    )
    return cache, checkpoint


def test_oracle_relation_one_hot_and_concat_are_deterministic():
    rows = [
        {"attribute": "registry"},
        {"attribute": "access"},
        {"attribute": "registry"},
    ]
    relation = oracle_relation_one_hot(rows, ["access", "registry"])
    assert relation.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    entity = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=torch.float32
    )
    first = fuse_oracle_queries(entity, relation, alpha=0.5)
    second = fuse_oracle_queries(entity, relation, alpha=0.5)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=1), torch.ones(3))
    assert first.shape == (3, 4)


def _selection_evaluation(*, centroid: float, margin: float) -> dict[str, object]:
    return {
        "strict_oracle_geometry": {"passed_count": 10},
        "query_space": {
            "retrieval": {
                "centroid_top1_accuracy": centroid,
                "row_1nn_accuracy": centroid,
                "mean_centroid_margin": margin,
                "centroid_mrr": centroid,
            },
            "silhouette": {"fact_id": 0.7},
            "geometry": {"fact_over_template_margin": 0.6},
        },
        "entity_branch_probe": {"accuracy": 0.95},
    }


def test_alpha_selection_accepts_only_validation_metrics_and_prefers_margin():
    selected, _ = select_oracle_alpha(
        {
            0.5: _selection_evaluation(centroid=0.95, margin=0.20),
            1.0: _selection_evaluation(centroid=0.95, margin=0.10),
        }
    )
    assert selected == 0.5

    # With every metric tied, the explicit -alpha tail keeps the smaller scale.
    tied, _ = select_oracle_alpha(
        {
            0.5: _selection_evaluation(centroid=0.95, margin=0.20),
            2.0: _selection_evaluation(centroid=0.95, margin=0.20),
        }
    )
    assert tied == 0.5


def test_answer_free_metrics_and_json_never_serialize_private_values(tmp_path):
    rows = _private_rows("train", ("t0", "t1"))
    entities = torch.eye(4).repeat_interleave(4, dim=0)
    relations = oracle_relation_one_hot(rows, ["access", "registry"])
    queries = fuse_oracle_queries(entities, relations, alpha=0.5)
    metrics, predictions = answer_free_query_metrics(
        queries, rows, queries, rows, ridge_strength=0.01
    )
    assert metrics["answer_free"] is True
    assert all("answer" not in row and "candidates" not in row for row in predictions)
    path = write_answer_free_json(tmp_path / "predictions.json", predictions)
    text = path.read_text(encoding="utf-8")
    assert "secret-" not in text and "candidates" not in text
    with pytest.raises(ValueError, match="answer-bearing"):
        write_answer_free_json(tmp_path / "unsafe.json", {"answer": "secret"})


def test_protocol_status_acknowledges_retired_read_without_opening_old_seal(
    tmp_path,
):
    incident = _write_protocol_incident(tmp_path / "incident.json")
    status = load_c23_protocol_status(
        incident, run_confirmation_data_read=False
    )
    assert status.run_confirmation_data_read is False
    assert status.retired_confirmation_template_read_count == 1
    assert status.confirmation_evaluation_count == 0
    assert not (tmp_path / "must-not-be-opened.json").exists()
    with pytest.raises(ValueError, match="must not read"):
        load_c23_protocol_status(incident, run_confirmation_data_read=True)


def test_source_validation_rejects_hash_layer_and_relation_label_mismatches(
    tmp_path,
):
    cache_path, checkpoint_path = _build_sources(tmp_path)
    cache = _load_feature_cache(cache_path)
    system, payload = load_canonicalizer_checkpoint(checkpoint_path)
    assert validate_oracle_sources(cache, system, payload)["relations"] == [
        "access",
        "registry",
    ]

    wrong_hash = {**payload, "source_core_sha256": "wrong"}
    with pytest.raises(ValueError, match="core hashes"):
        validate_oracle_sources(cache, system, wrong_hash)
    wrong_layers = {**payload, "selected_core_layers": [1]}
    with pytest.raises(ValueError, match="layer selections"):
        validate_oracle_sources(cache, system, wrong_layers)
    wrong_labels = {
        **payload,
        "labels": {**payload["labels"], "relations": ["access", "other"]},
    }
    with pytest.raises(ValueError, match="relations labels"):
        validate_oracle_sources(cache, system, wrong_labels)


def test_tiny_s0_run_is_validation_selected_answer_free_and_protocol_explicit(
    tmp_path,
):
    cache, checkpoint = _build_sources(tmp_path)
    incident = _write_protocol_incident(tmp_path / "incident.json")
    output = tmp_path / "output"
    summary = run_stage_c23_oracle(
        cache,
        checkpoint,
        output,
        protocol_incident_path=incident,
        run_confirmation_data_read=False,
        alpha_grid=(0.5,),
        evaluation_batch_size=32,
        device_name="cpu",
    )
    assert summary["selected_alpha"] == 0.5
    assert summary["selection_split"] == "validation"
    assert summary["development_evaluated_after_selection"] is True
    assert summary["oracle_upper_bound_passed"] is True
    assert summary["private_answers_serialized"] is False
    assert summary["run_confirmation_data_read"] is False
    assert summary["retired_confirmation_template_read_count"] == 1
    assert summary["confirmation_evaluation_count"] == 0
    assert summary["protocol_status"]["run_confirmation_data_read"] is False
    assert (
        summary["protocol_status"]["retired_confirmation_template_read_count"]
        == 1
    )
    assert summary["protocol_status"]["confirmation_evaluation_count"] == 0
    for path in output.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "secret-" not in text
        assert '"answer"' not in text
        assert '"candidates"' not in text
