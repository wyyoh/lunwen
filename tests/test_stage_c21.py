from __future__ import annotations

import torch
import json

from keyed_gram.canonicalizer import POOL_NAMES
from keyed_gram.stage_c21 import (
    DEFAULT_C21_VARIANTS,
    parse_c21_variants,
    relation_geometry_margin,
    relation_supervised_contrastive_loss,
    run_relation_source_audit,
    seal_confirmation_templates,
    train_c21_variant,
)


def _rows(split: str) -> list[dict[str, str]]:
    rows = []
    for relation_index, relation in enumerate(("a", "b", "c")):
        for entity_index in range(4):
            rows.append(
                {
                    "attribute": relation,
                    "entity": f"{split}-entity-{entity_index}",
                    "template_id": f"{split}-{entity_index % 2}",
                    "fact_id": f"{split}-{entity_index}|{relation}",
                    "answer": f"answer-{relation_index}",
                }
            )
    return rows


def test_relation_source_audit_selects_stable_raw_layer(tmp_path):
    metadata = {split: _rows(split) for split in ("train", "validation", "test")}
    features = {}
    for split, rows in metadata.items():
        pools = torch.zeros(len(rows), len(POOL_NAMES), 3, 4)
        for index, row in enumerate(rows):
            relation_index = ("a", "b", "c").index(row["attribute"])
            pools[index, 1, 2, relation_index] = 1.0
        features[split] = {
            "pools": pools,
            "baseline_last": torch.zeros(len(rows), 4),
        }
    cache = tmp_path / "features.pt"
    torch.save(
        {
            "format_version": 1,
            "source_core_sha256": "core",
            "selected_layers_one_based": [4, 6, 8],
            "selected_layer_indices": [3, 5, 7],
            "q0_baseline_layer_index": 6,
            "pool_names": list(POOL_NAMES),
            "features": features,
            "metadata": metadata,
        },
        cache,
    )

    result = run_relation_source_audit(cache, tmp_path / "audit")

    assert result["zero_training"] is True
    assert result["confirmation_accessed"] is False
    assert result["selected_residual_anchor"]["source_id"] == (
        "question_without_entity:layers-8"
    )
    assert result["selected_metrics"]["development_accuracy"] == 1.0
    assert (tmp_path / "audit" / "relation_source_audit.csv").exists()


def test_confirmation_seal_is_stable_and_detects_tampering(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = []
    for relation in ("registry_id", "city_code", "access_code"):
        for entity in ("alice", "bob"):
            rows.append(
                {
                    "fact_id": f"{entity}|{relation}",
                    "entity": entity,
                    "attribute": relation,
                    "answer": f"answer-{entity}-{relation}",
                    "candidates": ["x", "y"],
                    "answer_index": 0,
                    "template_id": "train-0",
                    "template_split": "train",
                    "entity_exposure": "seen",
                    "prompt": f"training prompt for {entity}",
                }
            )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    destination = tmp_path / "confirmation"
    public_path = tmp_path / "public_seal.json"

    first = seal_confirmation_templates(
        source, destination, public_record_path=public_path
    )
    second = seal_confirmation_templates(
        source, destination, public_record_path=public_path
    )

    assert first == second
    assert first["confirmation_accessed"] is False
    assert first["row_count"] == 12
    public_text = public_path.read_text(encoding="utf-8")
    assert "answer-alice" not in public_text
    rows_path = destination / "confirmation.jsonl"
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    try:
        seal_confirmation_templates(source, destination)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered confirmation data was accepted")


def test_relation_geometry_margin_uses_aligned_hard_negatives():
    rows = []
    embeddings = []
    for entity in ("alice", "bob"):
        for relation_index, relation in enumerate(("city", "registry")):
            for template in ("t0", "t1"):
                rows.append(
                    {
                        "entity": entity,
                        "attribute": relation,
                        "template_id": template,
                    }
                )
                embeddings.append(
                    [1.0, 0.0] if relation_index == 0 else [0.0, 1.0]
                )
    result = relation_geometry_margin(torch.tensor(embeddings), rows)
    assert result["same_relation_different_entity_template_cosine"] == 1.0
    assert result["different_relation_same_entity_template_cosine"] == 0.0
    assert result["relation_template_margin"] == 1.0


def test_relation_supcon_requires_cross_entity_and_cross_template_positives():
    relations = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    entities = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    templates = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    by_relation = torch.tensor(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4
    )
    by_template = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]] * 4
    )
    assert relation_supervised_contrastive_loss(
        by_relation, relations, entities, templates
    ) < relation_supervised_contrastive_loss(
        by_template, relations, entities, templates
    )


def test_c21_default_variants_are_incremental_and_keep_exact_r0():
    variants = parse_c21_variants(None)
    assert variants == DEFAULT_C21_VARIANTS
    assert variants[0].name == "R0" and variants[0].import_q3 is True
    assert variants[1].relation_supcon_weight == 0.2
    assert variants[2].template_adversary_weight == 0.05
    assert variants[3].relation_first is True
    assert variants[4].normalized_gated_fusion is True


def test_tiny_c21_training_writes_validation_selected_checkpoint(tmp_path):
    rows = []
    generator = torch.Generator().manual_seed(4)
    for entity_index, entity in enumerate(("alice", "bob", "carol", "dave")):
        for relation_index, relation in enumerate(("city", "registry")):
            for template_index in range(2):
                rows.append(
                    {
                        "entity": entity,
                        "attribute": relation,
                        "template_id": f"train-{template_index}",
                        "fact_id": f"{entity}|{relation}",
                        "answer": f"answer-{entity_index}-{relation_index}",
                    }
                )
    split_rows = {
        "train": rows,
        "validation": [
            {**row, "template_id": row["template_id"].replace("train", "val")}
            for row in rows
        ],
    }
    split_features = {
        split: torch.randn(len(values), 3, 2, 6, generator=generator)
        for split, values in split_rows.items()
    }
    result = train_c21_variant(
        DEFAULT_C21_VARIANTS[1],
        split_features,
        split_rows,
        tmp_path / "R1",
        selected_layer_indices=[3, 5],
        core_hidden_size=6,
        core_sha256="core",
        steps=1,
        relation_pretrain_fraction=0.2,
        batch_facts=4,
        templates_per_fact=2,
        query_dim=4,
        mlp_hidden_size=8,
        dropout=0.0,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_fraction=0.0,
        evaluation_interval=1,
        evaluation_batch_size=32,
        seed=0,
        device=torch.device("cpu"),
        ridge_strength=0.01,
        template_probe_seed=91,
    )
    assert result["best_step"] == 1
    assert result["pretrain_steps"] == 0
    assert (tmp_path / "R1" / "canonicalizer.pt").exists()
    assert (tmp_path / "R1" / "history.csv").exists()
