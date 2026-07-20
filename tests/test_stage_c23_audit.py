from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from keyed_gram.canonicalizer import (
    CanonicalizerConfig,
    CanonicalizerSystem,
    save_canonicalizer_checkpoint,
)
from keyed_gram.cli import build_parser
from keyed_gram.stage_c23 import prepare_answer_free_private_feature_cache
from keyed_gram.stage_c23_audit import (
    relation_conditioned_projected_family_leakage,
    run_stage_c23_audit,
    sanitize_private_metadata,
)
from keyed_gram.stage_c23_benchmark import RELATIONS
from keyed_gram.stage_c23_semantic import LoadedSemanticEncoder, SemanticEncoderSpec


def _private_rows(split: str, templates: tuple[str, ...]) -> list[dict[str, object]]:
    phrases = {
        "registry_id": "registry identifier",
        "city_code": "city identifier",
        "access_code": "access credential",
    }
    rows = []
    for entity_index in range(4):
        entity = f"entity-{entity_index}"
        for relation_index, relation in enumerate(RELATIONS):
            for template in templates:
                rows.append(
                    {
                        "prompt": (
                            f"For {entity}, give the {phrases[relation]}. Answer:"
                        ),
                        "entity": entity,
                        "attribute": relation,
                        "fact_id": f"{entity}|{relation}",
                        "template_id": f"{split}-{template}",
                        "answer": f"TOP-SECRET-{entity_index}-{relation_index}",
                        "answer_index": 0,
                        "candidates": ["TOP-SECRET", "decoy"],
                    }
                )
    return rows


def _private_features(rows: list[dict[str, object]]) -> torch.Tensor:
    features = torch.zeros(len(rows), 3, 1, 4)
    for index, row in enumerate(rows):
        entity_index = int(str(row["entity"]).rsplit("-", 1)[1])
        features[index, 0, 0, entity_index] = 1.0
    return features


def _build_private_sources(tmp_path: Path) -> tuple[Path, Path, dict[str, list[dict[str, object]]]]:
    rows = {
        "train": _private_rows("train", ("t0", "t1")),
        "validation": _private_rows("validation", ("v0", "v1")),
        "test": _private_rows("test", ("d0", "d1")),
    }
    cache = tmp_path / "private_features.pt"
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
                    "pools": _private_features(split_rows),
                    "baseline_last": torch.zeros(len(split_rows), 4),
                }
                for split, split_rows in rows.items()
            },
            "metadata": rows,
        },
        cache,
    )
    system = CanonicalizerSystem(
        CanonicalizerConfig(
            architecture="factorized",
            num_input_layers=1,
            core_hidden_size=4,
            query_dim=4,
            mlp_hidden_size=4,
            dropout=0.0,
            num_entities=4,
            num_relations=3,
            num_templates=2,
            template_adversary_source="relation",
        )
    )
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
        "relations": sorted(RELATIONS),
        "templates": ["train-t0", "train-t1"],
        "facts": sorted({str(row["fact_id"]) for row in rows["train"]}),
    }
    checkpoint = save_canonicalizer_checkpoint(
        tmp_path / "r3.pt",
        system,
        variant="R3",
        selected_core_layers=[0],
        labels=labels,
        source_core_sha256="core-sha",
    )
    return cache, checkpoint, rows


def _public_rows() -> dict[str, list[dict[str, object]]]:
    phrase = {
        "registry_id": "registry identifier",
        "city_code": "city identifier",
        "access_code": "access credential",
    }
    rows: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "reject": [],
    }
    for split, family_count in (("train", 2), ("validation", 3)):
        for relation in RELATIONS:
            for family_index in range(family_count):
                family = f"{split}-{relation}-family-{family_index}"
                for frame_index in range(3):
                    entity = f"public-{split}-entity-{frame_index}"
                    rows[split].append(
                        {
                            "example_id": f"{split}:{relation}:{family}:f{frame_index}",
                            "prompt": (
                                f"For {entity}, give the {phrase[relation]}. Answer:"
                            ),
                            "entity": entity,
                            "attribute": relation,
                            "relation_phrase": phrase[relation],
                            "family_id": family,
                            "frame_id": f"{split}-f{frame_index}",
                            "template_id": f"{split}-f{frame_index}",
                            "fact_id": f"public:{entity}|{relation}",
                        }
                    )
    for family_index in range(2):
        for frame_index in range(3):
            entity = f"public-reject-entity-{frame_index}"
            rows["reject"].append(
                {
                    "example_id": f"reject:unknown:{family_index}:f{frame_index}",
                    "prompt": f"For {entity}, give the unrelated profile value.",
                    "entity": entity,
                    "attribute": None,
                    "relation_phrase": "unrelated profile value",
                    "family_id": f"reject-family-{family_index}",
                    "frame_id": f"validation-f{frame_index}",
                    "template_id": f"validation-f{frame_index}",
                    "fact_id": f"public:{entity}|unknown-{family_index}",
                }
            )
    return rows


def _write_incident(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "incident": "local_confirmation_template_selection_read_during_read_only_audit",
                "scope": {
                    "templates_read": True,
                    "rows_read": False,
                    "private_answers_read": False,
                    "model_evaluation_run": False,
                    "metrics_computed": False,
                },
                "old_seal": {
                    "path": str(path.parent / "must-not-open.json"),
                    "status": "retired_compromised_for_future_confirmation",
                },
                "access_accounting": {
                    "confirmation_template_read_count": 1,
                    "confirmation_evaluation_count": 0,
                    "confirmation_metric_access_count": 0,
                },
                "new_confirmation_status": "not_created",
            }
        ),
        encoding="utf-8",
    )
    return path


def _private_phrase_views() -> dict[str, dict[str, list[str]]]:
    phrases = {
        "registry_id": "registry identifier",
        "city_code": "city identifier",
        "access_code": "access credential",
    }
    return {
        split: {relation: [value, value] for relation, value in phrases.items()}
        for split in ("train", "validation", "development")
    }


def _write_config(
    path: Path, cache: Path, checkpoint: Path, incident: Path
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "stage": "C2.3-public-semantic-encoder-audit",
                "private_feature_cache": str(cache),
                "entity_checkpoint": str(checkpoint),
                "protocol_incident": str(incident),
                "benchmark": {
                    "definitions": {
                        "registry_id": ["registry identifier"],
                        "city_code": ["city identifier"],
                        "access_code": ["access credential"],
                    },
                    "private_phrase_views": _private_phrase_views(),
                },
                "models": {},
                "semantic_audit": {
                    "input_views": [
                        "phrase_only",
                        "entity_masked",
                        "entity_masked_strip_suffix",
                    ],
                    "batch_size": 32,
                    "bootstrap_samples": 50,
                    "bootstrap_seed": 17,
                    "ridge_strength": 0.01,
                    "oracle_alpha": 0.5,
                    "small_variants": ["S2"],
                    "gates": {
                        "public_family_macro_accuracy": 0.85,
                        "private_validation_relation_accuracy": 0.85,
                        "development_relation_accuracy": 0.85,
                        "relation_family_margin": 0.15,
                        "projected_family_probe_chance_margin": 0.10,
                        "encoder_fact_centroid_top1": 0.80,
                        "entity_probe": 0.90,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class _MockEncoderFactory:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.loaded = []

    def __call__(self, spec, *, cache_dir, device):
        self.loaded.append(spec.name)
        model = nn.Linear(1, 1, bias=False)
        model.eval()
        return LoadedSemanticEncoder(
            spec=spec,
            snapshot_path=self.snapshot,
            tokenizer=None,
            model=model,
            device=torch.device(device),
        )


def _mock_encode(encoder, texts, *, batch_size, e5_input_type):
    del batch_size, e5_input_type
    vectors = []
    for text in texts:
        lowered = str(text).lower()
        if "access" in lowered or "credential" in lowered:
            vector = [1.0, 0.0, 0.0]
        elif "city" in lowered:
            vector = [0.0, 1.0, 0.0]
        elif "registry" in lowered:
            vector = [0.0, 0.0, 1.0]
        else:
            vector = [1.0, 1.0, 1.0]
        vectors.append(vector)
    assert encoder.spec.dimension == 3
    return F.normalize(torch.tensor(vectors, dtype=torch.float32), p=2, dim=-1)


def test_private_metadata_is_immediately_whitelisted_and_phrase_aligned():
    metadata = {
        "train": _private_rows("train", ("t0", "t1")),
        "validation": _private_rows("validation", ("v0", "v1")),
        "test": _private_rows("test", ("d0", "d1")),
    }
    sanitized = sanitize_private_metadata(metadata, _private_phrase_views())
    assert set(sanitized) == {"train", "validation", "test"}
    assert all(
        set(row)
        == {
            "prompt",
            "entity",
            "attribute",
            "fact_id",
            "template_id",
            "relation_phrase",
        }
        for rows in sanitized.values()
        for row in rows
    )
    serialized = json.dumps(sanitized)
    assert "TOP-SECRET" not in serialized
    assert "candidates" not in serialized and "answer_index" not in serialized


def test_projected_family_leakage_is_conditioned_within_relation():
    rows = _public_rows()["validation"]
    relation_lookup = {relation: index for index, relation in enumerate(sorted(RELATIONS))}
    projected = torch.stack(
        [
            F.one_hot(
                torch.tensor(relation_lookup[str(row["attribute"])]),
                num_classes=3,
            ).float()
            for row in rows
        ]
    )
    result = relation_conditioned_projected_family_leakage(
        projected, rows, ridge_strength=0.01
    )
    assert result["conditioning"] == "true_relation"
    assert result["macro_accuracy"] == pytest.approx(
        result["macro_chance_accuracy"]
    )


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["stage-c23-prepare", "--config", "c23.yaml"], "stage-c23-prepare"),
        (
            [
                "stage-c23-oracle",
                "--config",
                "c23.yaml",
                "--output-dir",
                "oracle",
            ],
            "stage-c23-oracle",
        ),
        (
            [
                "stage-c23-audit",
                "--config",
                "c23.yaml",
                "--output-dir",
                "audit",
                "--variants",
                "S6",
            ],
            "stage-c23-audit",
        ),
    ],
)
def test_stage_c23_cli_subcommands_parse(argv, command):
    args = build_parser().parse_args(argv)
    assert args.command == command
    assert callable(args.func)
    if command == "stage-c23-audit":
        assert args.variants == "S6"


def test_runner_uses_only_requested_mock_variant_and_writes_answer_free_audit(
    tmp_path,
):
    source_cache, checkpoint, _ = _build_private_sources(tmp_path)
    cache = tmp_path / "answer_free_private_features.pt"
    prepare_answer_free_private_feature_cache(
        source_cache,
        cache,
        tmp_path / "answer_free_private_features.json",
    )
    incident = _write_incident(tmp_path / "protocol_incident.json")
    config = _write_config(
        tmp_path / "stage_c23.yaml", cache, checkpoint, incident
    )
    snapshot = tmp_path / "mock_snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"mock-safe-model")
    factory = _MockEncoderFactory(snapshot)
    spec = SemanticEncoderSpec(
        name="mock-s2",
        model_id="local/mock-s2",
        revision="1" * 40,
        license="test-only",
        dimension=3,
        pooling="mask_mean",
        max_length=32,
    )
    output = tmp_path / "audit"
    summary = run_stage_c23_audit(
        config,
        output,
        variants=("S2",),
        device_name="cpu",
        encoder_loader=factory,
        encode_fn=_mock_encode,
        variant_specs={"S2": spec},
        public_rows=_public_rows(),
        bootstrap_samples=50,
    )
    assert factory.loaded == ["mock-s2"]
    assert summary["requested_variants"] == ["S2"]
    assert summary["selected_variant"] == "S2"
    assert summary["development_used_for_selection"] is False
    assert summary["private_cache_metadata_sanitized_immediately"] is True
    assert summary["private_answers_deserialized_by_runtime"] is False
    assert summary["private_answers_passed_to_semantic_encoder"] is False
    assert summary["s5"]["public_validation_family_macro"]["macro_accuracy"] == 1.0
    assert summary["s5"]["private_validation_relation_accuracy"] == 1.0
    assert summary["s5"]["development_relation_accuracy"] == 1.0
    assert summary["s5"]["strict_readiness"]["passed"] is True
    assert summary["s6_status"] == "skipped_small_model_validation_gates_passed"
    assert summary["small_model_validation_only_passed"] is True
    assert summary["c3_eligible"] is False
    assert summary["failed_strict_gates"] == []
    assert "query geometry only" in summary["c3_eligibility_reason"]
    assert summary["run_confirmation_data_read"] is False
    assert summary["retired_confirmation_template_read_count"] == 1
    assert summary["confirmation_evaluation_count"] == 0
    assert not (tmp_path / "must-not-open.json").exists()

    tracked = list(output.rglob("*.json")) + list(output.rglob("*.csv"))
    assert tracked
    for path in tracked:
        text = path.read_text(encoding="utf-8")
        assert "TOP-SECRET" not in text
        assert '"answer"' not in text
        assert '"candidates"' not in text
    manifest = json.loads(
        (output / "artifact_sha256_manifest.json").read_text(encoding="utf-8")
    )
    for item in manifest["files"]:
        artifact = Path(item["path"])
        assert artifact.exists()
        assert item["sha256"] == __import__("hashlib").sha256(
            artifact.read_bytes()
        ).hexdigest()
