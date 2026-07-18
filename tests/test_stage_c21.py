from __future__ import annotations

import torch
import json

from keyed_gram.canonicalizer import POOL_NAMES
from keyed_gram.stage_c21 import (
    run_relation_source_audit,
    seal_confirmation_templates,
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
