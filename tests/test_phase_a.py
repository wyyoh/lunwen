from __future__ import annotations

import torch

from keyed_gram.keying import hash_auxiliary, hash_non_auxiliary
from keyed_gram.model import GramTransformer
from keyed_gram.phase_a import (
    FACT_FIELDS,
    TRAIN_PROMPTS,
    _ExampleEpochSampler,
    _encode_private_examples,
    _reset_auxiliary,
    _select_aux_parameters,
    build_phase_a_rows,
    make_opaque_biographies,
)


def test_opaque_biographies_are_deterministic_unique_and_split():
    left = make_opaque_biographies(8, 3, seed=19)
    right = make_opaque_biographies(8, 3, seed=19)
    assert left == right
    assert [record.split for record in left].count("memorized") == 8
    assert [record.split for record in left].count("heldout") == 3
    for field in FACT_FIELDS:
        assert len({record.fact(field) for record in left}) == len(left)


def test_phase_a_rows_hide_heldout_facts_and_use_disjoint_prompts():
    records = make_opaque_biographies(8, 3, seed=23)
    train_rows, eval_rows = build_phase_a_rows(records, candidate_count=4, seed=23)
    assert len(train_rows) == 8 * sum(len(TRAIN_PROMPTS[field]) for field in FACT_FIELDS)
    assert len(eval_rows) == 11 * len(FACT_FIELDS)
    heldout = {record.entity for record in records if record.split == "heldout"}
    assert not heldout.intersection(row["entity"] for row in train_rows)
    assert heldout.issubset(row["entity"] for row in eval_rows)
    assert {row["prompt"] for row in train_rows}.isdisjoint(
        row["prompt"] for row in eval_rows
    )
    for row in eval_rows:
        assert len(row["candidates"]) == 4
        assert len(set(row["candidates"])) == 4
        assert row["candidates"][row["answer_index"]] == row["answer"]


def test_freeze_core_parameter_selection_and_reset_preserve_core(tiny_config):
    model = GramTransformer(tiny_config)
    core_before = hash_non_auxiliary(model.state_dict(), 1)
    aux_before = hash_auxiliary(model.state_dict(), 1)
    _reset_auxiliary(model, 1, seed=5)
    selected = _select_aux_parameters(model, 1)
    assert hash_non_auxiliary(model.state_dict(), 1) == core_before
    assert hash_auxiliary(model.state_dict(), 1) != aux_before
    assert selected
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (".moe.experts.1." in name)


def test_private_example_sampler_masks_padding():
    examples = [
        (torch.tensor([2, 3, 4]), torch.tensor([-100, 4, 1])),
        (torch.tensor([5, 6]), torch.tensor([6, 1])),
    ]
    sampler = _ExampleEpochSampler(examples, seed=0, eos=1)
    x, y = sampler.sample_batch(2)
    assert x.shape == y.shape == (2, 3)
    shorter = (x == 1).sum(dim=1).argmax().item()
    assert y[shorter, -1].item() == -100


def test_private_fact_loss_excludes_eos():
    class Tokenizer:
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [{"prompt": 2, "answer": 3}[word] for word in text.split()]

    examples = _encode_private_examples(
        [{"prompt": "prompt", "answer": "answer"}], Tokenizer(), context_length=8
    )
    x, y = examples[0]
    assert x.tolist() == [2]
    assert y.tolist() == [3]
