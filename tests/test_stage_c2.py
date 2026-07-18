from __future__ import annotations

import torch

from keyed_gram.canonicalizer import (
    CanonicalizerConfig,
    CanonicalizerSystem,
    StructuredFactBatchSampler,
    gradient_reverse,
    load_canonicalizer_checkpoint,
    save_canonicalizer_checkpoint,
    supervised_contrastive_loss,
)
from keyed_gram.stage_c2 import locate_entity_token_span


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(value) for value in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_entity_span_uses_offsets_and_preserves_prompt_encoding():
    prompt = "Find alice and return her city:"
    ids, positions = locate_entity_token_span(_CharacterTokenizer(), prompt, "alice")
    assert len(ids) == len(prompt)
    assert "".join(chr(ids[index]) for index in positions) == "alice"


def test_structured_sampler_contains_both_hard_negative_types():
    rows = []
    for entity in ("alice", "bob", "carol", "dave"):
        for relation in ("city", "code"):
            for template in range(3):
                rows.append(
                    {
                        "entity": entity,
                        "attribute": relation,
                        "fact_id": f"{entity}|{relation}",
                        "template_id": f"train-{template}",
                    }
                )
    sampler = StructuredFactBatchSampler(
        rows, batch_facts=4, templates_per_fact=2, seed=3
    )
    batch = [rows[index] for index in sampler.sample_indices()]
    facts = {(row["entity"], row["attribute"]) for row in batch}
    assert len(facts) == 4
    assert len({row["entity"] for row in batch}) == 2
    assert len({row["attribute"] for row in batch}) == 2
    assert all(
        sum(row["fact_id"] == fact for row in batch) == 2
        for fact in {row["fact_id"] for row in batch}
    )


def test_supervised_contrastive_loss_rewards_fact_clusters():
    labels = torch.tensor([0, 0, 1, 1])
    clustered = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    crossed = clustered[[0, 2, 1, 3]]
    assert supervised_contrastive_loss(clustered, labels) < supervised_contrastive_loss(
        crossed, labels
    )


def test_gradient_reversal_flips_the_upstream_gradient():
    value = torch.tensor([1.0, -2.0], requires_grad=True)
    (gradient_reverse(value, 0.25) ** 2).sum().backward()
    assert torch.allclose(value.grad, torch.tensor([-0.5, 1.0]))


def test_factorized_canonicalizer_outputs_unit_queries():
    config = CanonicalizerConfig(
        architecture="factorized",
        num_input_layers=3,
        core_hidden_size=8,
        query_dim=4,
        mlp_hidden_size=12,
        dropout=0.0,
        num_entities=4,
        num_relations=2,
        num_templates=3,
    )
    system = CanonicalizerSystem(config)
    output = system(torch.randn(5, 3, 3, 8))
    assert output["query"].shape == (5, 4)
    assert torch.allclose(output["query"].norm(dim=-1), torch.ones(5), atol=1e-6)
    assert output["layer_weights"].shape == (3, 3)


def test_normalized_gated_fusion_exposes_branch_contributions():
    config = CanonicalizerConfig(
        architecture="factorized",
        num_input_layers=3,
        core_hidden_size=8,
        query_dim=4,
        mlp_hidden_size=12,
        dropout=0.0,
        num_entities=4,
        num_relations=2,
        num_templates=3,
        template_adversary_source="relation",
        normalized_gated_fusion=True,
    )
    system = CanonicalizerSystem(config)
    output = system(torch.randn(5, 3, 3, 8), adversary_strength=1.0)
    assert torch.allclose(output["entity_unit"].norm(dim=-1), torch.ones(5))
    assert torch.allclose(output["relation_unit"].norm(dim=-1), torch.ones(5))
    assert torch.equal(output["template_source"], output["relation_unit"])
    assert output["fusion_scales"].shape == (3,)
    assert bool(output["fusion_scales"].gt(0).all())
    for name in (
        "entity_contribution",
        "relation_contribution",
        "interaction_contribution",
    ):
        assert output[name].shape == (5, 4)


def test_canonicalizer_checkpoint_round_trip(tmp_path):
    config = CanonicalizerConfig(
        architecture="joint",
        num_input_layers=2,
        core_hidden_size=6,
        query_dim=4,
        mlp_hidden_size=8,
        dropout=0.0,
        num_entities=2,
        num_relations=2,
        num_templates=2,
    )
    system = CanonicalizerSystem(config)
    path = save_canonicalizer_checkpoint(
        tmp_path / "canonicalizer.pt",
        system,
        variant="Q1",
        selected_core_layers=[3, 5],
        labels={"entities": ["a", "b"]},
        source_core_sha256="abc",
    )
    restored, payload = load_canonicalizer_checkpoint(path)
    assert payload["variant"] == "Q1"
    assert payload["selected_core_layers"] == [3, 5]
    for left, right in zip(system.parameters(), restored.parameters()):
        assert torch.equal(left, right)
