from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from keyed_gram.stage_c23_semantic import (
    ENCODER_SNAPSHOT_ALLOW_PATTERNS,
    SEMANTIC_ENCODER_SPECS,
    LoadedSemanticEncoder,
    SemanticEncoderSpec,
    build_all_semantic_views,
    build_model_file_manifest,
    definition_prototype_matching,
    encode_semantic_texts,
    family_bootstrap_accuracy_ci,
    family_macro_accuracy,
    leave_one_family_out_relation_margin,
    load_semantic_embedding_cache,
    load_semantic_encoder,
    open_set_rejection_metrics,
    prepare_encoder_texts,
    ridge_linear_head_logits,
    save_semantic_embedding_cache,
)


def _tiny_spec(name: str, pooling: str, *, e5: bool = False) -> SemanticEncoderSpec:
    return SemanticEncoderSpec(
        name=name,
        model_id=f"test/{name}",
        revision="0" * 40,
        license="test-only",
        dimension=2,
        pooling=pooling,
        max_length=8,
        requires_e5_prefix=e5,
    )


class _RecordingFactory:
    calls: list[tuple[str, dict]] = []
    value = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls.value


class _DummyTokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, texts, **kwargs):
        self.texts.extend(texts)
        batch_size = len(texts)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]]).repeat(batch_size, 1),
        }


class _DummyModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        hidden = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]]
        ).repeat(len(input_ids), 1, 1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_pinned_snapshot_then_strict_local_loading(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return str(snapshot)

    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    class TokenizerFactory(_RecordingFactory):
        calls = []
        value = tokenizer

    class ModelFactory(_RecordingFactory):
        calls = []
        value = model

    spec = _tiny_spec("mock-e5", "mask_mean", e5=True)
    loaded = load_semantic_encoder(
        spec,
        cache_dir=tmp_path / "hub",
        snapshot_download_fn=fake_download,
        tokenizer_class=TokenizerFactory,
        model_class=ModelFactory,
    )

    assert loaded.snapshot_path == snapshot
    assert download_calls == [
        {
            "repo_id": "test/mock-e5",
            "revision": "0" * 40,
            "allow_patterns": list(ENCODER_SNAPSHOT_ALLOW_PATTERNS),
            "cache_dir": str(tmp_path / "hub"),
        }
    ]
    assert not any(pattern.endswith(".bin") for pattern in ENCODER_SNAPSHOT_ALLOW_PATTERNS)
    for calls in (TokenizerFactory.calls, ModelFactory.calls):
        assert calls == [
            (
                str(snapshot),
                {"local_files_only": True, "trust_remote_code": False},
            )
        ]
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_encoder_prefix_pooling_and_explicit_l2_normalization(tmp_path):
    tokenizer = _DummyTokenizer()
    mean_encoder = LoadedSemanticEncoder(
        spec=_tiny_spec("e5", "mask_mean", e5=True),
        snapshot_path=tmp_path,
        tokenizer=tokenizer,
        model=_DummyModel(),
        device=torch.device("cpu"),
    )
    mean_embedding = encode_semantic_texts(mean_encoder, ["city identifier"])
    assert tokenizer.texts == ["query: city identifier"]
    assert torch.allclose(
        mean_embedding, torch.tensor([[2**-0.5, 2**-0.5]]), atol=1e-6
    )
    assert torch.allclose(mean_embedding.norm(dim=1), torch.ones(1))

    cls_encoder = LoadedSemanticEncoder(
        spec=_tiny_spec("bge", "cls"),
        snapshot_path=tmp_path,
        tokenizer=_DummyTokenizer(),
        model=_DummyModel(),
        device=torch.device("cpu"),
    )
    cls_embedding = encode_semantic_texts(cls_encoder, ["city identifier"])
    assert torch.equal(cls_embedding, torch.tensor([[1.0, 0.0]]))
    assert prepare_encoder_texts(mean_encoder.spec, ["passage: x"]) == ["query: x"]


def test_official_specs_use_full_pinned_revisions_and_expected_contracts():
    assert set(SEMANTIC_ENCODER_SPECS) == {
        "minilm",
        "e5-small",
        "bge-small",
        "e5-base",
    }
    assert all(len(spec.revision) == 40 for spec in SEMANTIC_ENCODER_SPECS.values())
    assert SEMANTIC_ENCODER_SPECS["minilm"].pooling == "mask_mean"
    assert SEMANTIC_ENCODER_SPECS["bge-small"].pooling == "cls"
    assert SEMANTIC_ENCODER_SPECS["e5-small"].requires_e5_prefix is True
    assert SEMANTIC_ENCODER_SPECS["e5-base"].requires_e5_prefix is True


def test_three_semantic_views_mask_entities_and_strip_answer_suffix():
    rows = [
        {
            "prompt": "Tell me the city identifier associated with Alice. Answer:",
            "entity": "Alice",
            "relation_phrase": "city identifier",
        }
    ]
    views = build_all_semantic_views(rows)
    assert views["phrase_only"] == ["city identifier"]
    assert views["entity_masked"] == [
        "Tell me the city identifier associated with <ENTITY>. Answer:"
    ]
    assert views["entity_masked_strip_suffix"] == [
        "Tell me the city identifier associated with <ENTITY>."
    ]
    assert all("Alice" not in value[0] for value in views.values())


def test_model_manifest_and_embedding_cache_are_revision_bound(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "1_Pooling").mkdir(parents=True)
    (snapshot / "config.json").write_text('{"hidden_size": 2}', encoding="utf-8")
    (snapshot / "1_Pooling" / "config.json").write_text(
        '{"pooling_mode_mean_tokens": true}', encoding="utf-8"
    )
    spec = _tiny_spec("cache", "mask_mean")
    manifest = build_model_file_manifest(snapshot, spec)
    assert manifest["file_count"] == 2
    assert len(manifest["manifest_sha256"]) == 64
    assert {value["path"] for value in manifest["files"]} == {
        "config.json",
        "1_Pooling/config.json",
    }

    cache_path = tmp_path / "embeddings.pt"
    texts = ["alpha", "beta"]
    save_semantic_embedding_cache(
        cache_path,
        torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
        spec=spec,
        view="phrase_only",
        texts=texts,
        model_manifest=manifest,
        row_ids=["a", "b"],
    )
    payload = load_semantic_embedding_cache(
        cache_path,
        expected_spec=spec,
        expected_view="phrase_only",
        expected_texts=texts,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    assert torch.equal(payload["embeddings"], torch.eye(2))
    with pytest.raises(ValueError, match="input text hash"):
        load_semantic_embedding_cache(cache_path, expected_texts=["changed", "beta"])


def test_definition_prototype_matching_supports_mean_and_max():
    query = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    definitions = {
        "access_code": torch.tensor([[0.0, 1.0], [0.1, 0.9]]),
        "city_code": torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
    }
    mean = definition_prototype_matching(query, definitions, aggregation="mean")
    maximum = definition_prototype_matching(query, definitions, aggregation="max")
    assert mean["predictions"] == ["city_code", "access_code"]
    assert maximum["predictions"] == ["city_code", "access_code"]
    assert mean["logits"].shape == (2, 2)
    assert bool((mean["margin"] > 0).all())


def test_family_macro_bootstrap_and_leave_one_family_out_margin_are_family_level():
    predictions = ["a", "a", "b", "a"]
    targets = ["a", "a", "b", "b"]
    families = ["large", "large", "large", "small"]
    metric = family_macro_accuracy(predictions, targets, families)
    assert metric["micro_accuracy"] == 0.75
    assert metric["macro_accuracy"] == 0.5
    first = family_bootstrap_accuracy_ci(
        predictions, targets, families, num_resamples=100, seed=17
    )
    second = family_bootstrap_accuracy_ci(
        predictions, targets, families, num_resamples=100, seed=17
    )
    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    geometry = leave_one_family_out_relation_margin(
        embeddings,
        relations=["city", "city", "access", "access"],
        families=["city-a", "city-b", "access-a", "access-b"],
    )
    assert geometry["macro_accuracy"] == 1.0
    assert geometry["macro_margin"] > 0.7


def test_open_set_metrics_report_calibration_coverage_and_tnr95():
    logits = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, -1, -1])
    metrics = open_set_rejection_metrics(logits, targets)
    assert metrics["known_detection_auroc"] == 1.0
    assert metrics["known_detection_aupr"] == 1.0
    assert metrics["tnr_at_95_known_coverage"] == 1.0
    assert metrics["achieved_known_coverage"] == 1.0
    assert metrics["known_ece"] < 0.01
    assert len(metrics["coverage_accuracy_curve"]) == len(logits)
    assert 0.0 <= metrics["coverage_aurc"] <= 1.0


def test_ridge_linear_head_is_deterministic_and_returns_logits():
    train = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = ["city", "city", "access", "access"]
    evaluation = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    first = ridge_linear_head_logits(train, labels, evaluation)
    second = ridge_linear_head_logits(train, labels, evaluation)
    assert first["classes"] == ("access", "city")
    assert torch.equal(first["train_logits"], second["train_logits"])
    assert torch.equal(first["evaluation_logits"], second["evaluation_logits"])
    predicted = first["evaluation_logits"].argmax(dim=1)
    assert [first["classes"][int(index)] for index in predicted] == [
        "city",
        "access",
    ]
