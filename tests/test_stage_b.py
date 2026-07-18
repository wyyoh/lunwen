from __future__ import annotations

import torch
import pytest

from keyed_gram.phase_a import FACT_FIELDS, TRAIN_PROMPTS, make_opaque_biographies
from keyed_gram.stage_b import (
    TEST_PROMPTS,
    VALIDATION_PROMPTS,
    build_stage_b_rows,
    localization_scores,
    public_residual_null_loss,
    residual_infonce,
    symmetric_candidate_kl,
)


def test_stage_b_template_and_entity_grids_are_disjoint():
    records = make_opaque_biographies(8, 3, seed=11)
    rows = build_stage_b_rows(records, candidate_count=4, seed=11)
    assert len(rows["train"]) == 8 * sum(len(TRAIN_PROMPTS[x]) for x in FACT_FIELDS)
    assert len(rows["validation"]) == 8 * sum(
        len(VALIDATION_PROMPTS[x]) for x in FACT_FIELDS
    )
    assert len(rows["test"]) == 8 * sum(len(TEST_PROMPTS[x]) for x in FACT_FIELDS)
    assert len(rows["context"]) == len(rows["unexposed"]) == 3 * len(FACT_FIELDS)
    seen = {record.entity for record in records if record.split == "memorized"}
    heldout = {record.entity for record in records if record.split == "heldout"}
    assert {row["entity"] for row in rows["test"]} == seen
    assert {row["entity"] for row in rows["context"]} == heldout
    assert {row["prompt"] for row in rows["train"]}.isdisjoint(
        row["prompt"] for row in rows["test"]
    )


def test_symmetric_candidate_kl_is_zero_only_for_matching_distributions():
    candidates = torch.tensor([[0, 1, 2], [0, 1, 2]])
    left = torch.tensor([[3.0, 1.0, -2.0], [1.0, 2.0, 0.0]])
    assert torch.allclose(
        symmetric_candidate_kl(left, left, candidates), torch.tensor(0.0), atol=1e-6
    )
    right = left.flip(1)
    assert symmetric_candidate_kl(left, right, candidates) > 0


def test_residual_infonce_rewards_same_fact_pairs():
    basis = torch.eye(4).reshape(4, 1, 4)
    lengths = torch.ones(4, dtype=torch.long)
    matched = residual_infonce([basis], [basis], lengths, lengths, temperature=0.1)
    permuted = residual_infonce(
        [basis], [basis.roll(1, dims=0)], lengths, lengths, temperature=0.1
    )
    assert matched < permuted


def test_public_residual_null_and_bounded_localization_metrics():
    diagnostics = {
        "aux_inputs": [torch.ones(2, 3, 4)],
        "aux_residuals": [torch.zeros(2, 3, 4)],
    }
    assert public_residual_null_loss(diagnostics).item() == 0.0
    scores = localization_scores(full=0.8, core=0.1, chance=0.125)
    assert scores["access_gap"] == pytest.approx(0.7)
    assert scores["localization_score"] == 1.0
    assert scores["anti_signal"] == pytest.approx(0.025)
    assert 0 <= localization_scores(0.2, 0.9, 0.125)["localization_score"] <= 1
