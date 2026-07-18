from __future__ import annotations

import torch

from keyed_gram.canonicalizer import POOL_NAMES
from keyed_gram.stage_c21 import run_relation_source_audit


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
