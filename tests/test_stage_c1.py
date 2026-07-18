from __future__ import annotations

import pytest
import torch

from keyed_gram.stage_c1 import (
    cosine_silhouette,
    cross_template_retrieval,
    representation_geometry,
    ridge_probe_accuracy,
)


def _rows(split: str) -> list[dict[str, str]]:
    return [
        {
            "fact_id": "alice|city",
            "entity": "alice",
            "attribute": "city",
            "answer": "rome",
            "template_id": f"{split}-0",
        },
        {
            "fact_id": "alice|code",
            "entity": "alice",
            "attribute": "code",
            "answer": "seven",
            "template_id": f"{split}-0",
        },
        {
            "fact_id": "bob|city",
            "entity": "bob",
            "attribute": "city",
            "answer": "oslo",
            "template_id": f"{split}-0",
        },
        {
            "fact_id": "bob|code",
            "entity": "bob",
            "attribute": "code",
            "answer": "nine",
            "template_id": f"{split}-0",
        },
    ]


def test_cross_template_retrieval_recovers_fact_identity():
    database_rows = _rows("train")
    query_rows = _rows("test")
    database = torch.eye(4)
    query = database + 0.01
    metrics, predictions = cross_template_retrieval(
        database, database_rows, query, query_rows
    )
    assert metrics["row_1nn_accuracy"] == 1.0
    assert metrics["centroid_top1_accuracy"] == 1.0
    assert all(row["row_1nn_correct"] for row in predictions)


def test_geometry_separates_fact_pairs_from_template_negatives():
    database_rows = _rows("train")
    query_rows = _rows("test")
    database = torch.eye(4)
    query = database + 0.001
    metrics = representation_geometry(
        database, database_rows, query, query_rows
    )
    assert metrics["same_fact_cross_template_cosine"] > 0.99
    assert metrics["fact_over_template_margin"] > 0.9


def test_cosine_silhouette_and_ridge_probe_detect_separable_labels():
    train = torch.tensor(
        [[-2.0, -1.0], [-1.8, -1.2], [2.0, 1.0], [1.8, 1.2]]
    )
    labels = ["left", "left", "right", "right"]
    assert cosine_silhouette(train, labels) > 0.9
    result = ridge_probe_accuracy(
        train,
        labels,
        torch.tensor([[-1.9, -1.1], [1.9, 1.1]]),
        ["left", "right"],
    )
    assert result["accuracy"] == 1.0
    assert result["chance_accuracy"] == pytest.approx(0.5)


def test_ridge_probe_rejects_unseen_evaluation_class():
    with pytest.raises(ValueError, match="absent"):
        ridge_probe_accuracy(
            torch.eye(2), ["a", "b"], torch.ones(1, 2), ["c"]
        )
