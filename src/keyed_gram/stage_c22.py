from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm.auto import tqdm

from .canonicalizer import (
    CanonicalizerSystem,
    StructuredFactBatchSampler,
    gradient_reverse,
    load_canonicalizer_checkpoint,
    save_canonicalizer_checkpoint,
    supervised_contrastive_loss,
)
from .checkpoint import build_model_from_checkpoint
from .keying import hash_non_auxiliary
from .phase_a import _load_tokenizer, _sha256_file
from .stage_c2 import (
    _clone_state,
    _label_map,
    _lr_scale,
    _targets,
    _template_leakage_probe,
    extract_canonicalizer_features,
)
from .stage_c21 import (
    _embed_branch_outputs,
    _evaluate_c21_split,
    _load_feature_cache,
    _selection_score,
    relation_geometry_margin,
    relation_supervised_contrastive_loss,
)
from .stage_c1 import ridge_probe_accuracy
from .train import (
    collect_environment,
    resolve_device,
    resolve_dtype,
    set_seed,
)


STAGE_C22_SCHEMA_VERSION = 1
RELATIONS = ("registry_id", "city_code", "access_code")


@dataclass(frozen=True)
class PublicRelationVariant:
    name: str
    import_base: bool
    private_replay_weight: float

    def validate(self) -> None:
        if self.private_replay_weight < 0:
            raise ValueError("private_replay_weight must be non-negative")
        if self.import_base and self.name != "P0":
            raise ValueError("only P0 may import the exact C2.1 base")


DEFAULT_PUBLIC_VARIANTS = (
    PublicRelationVariant("P0", True, 0.0),
    PublicRelationVariant("P1", False, 0.0),
    PublicRelationVariant("P2", False, 1.0),
)


def parse_public_variants(raw: Any) -> tuple[PublicRelationVariant, ...]:
    defaults = {value.name: value for value in DEFAULT_PUBLIC_VARIANTS}
    if not raw:
        return DEFAULT_PUBLIC_VARIANTS
    if set(raw) != set(defaults):
        raise ValueError("Stage C2.2 requires exactly P0, P1, and P2")
    parsed = []
    for name in ("P0", "P1", "P2"):
        default = defaults[name]
        values = raw[name] or {}
        variant = PublicRelationVariant(
            name=name,
            import_base=bool(values.get("import_base", default.import_base)),
            private_replay_weight=float(
                values.get(
                    "private_replay_weight", default.private_replay_weight
                )
            ),
        )
        variant.validate()
        parsed.append(variant)
    if not parsed[0].import_base or any(value.import_base for value in parsed[1:]):
        raise ValueError("P0 must be the only imported C2.1 baseline")
    return tuple(parsed)


def _normalize_phrase(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _validate_public_corpus_config(config: dict[str, Any]) -> None:
    entities = config.get("entities", {})
    frames = config.get("frames", {})
    phrases = config.get("phrases", {})
    forbidden = config.get("forbidden_phrases", {})
    for split in ("train", "validation"):
        split_entities = [str(value) for value in entities.get(split, [])]
        split_frames = [str(value) for value in frames.get(split, [])]
        if len(split_entities) < 4 or len(set(split_entities)) != len(split_entities):
            raise ValueError(f"public {split} needs at least four unique entities")
        if len(split_frames) < 2 or len(set(split_frames)) != len(split_frames):
            raise ValueError(f"public {split} needs at least two unique frames")
        for frame in split_frames:
            if frame.count("{entity}") != 1 or frame.count("{relation_phrase}") != 1:
                raise ValueError(
                    "every public frame needs one entity and one relation phrase"
                )
        split_phrases = phrases.get(split, {})
        if set(split_phrases) != set(RELATIONS):
            raise ValueError(f"public {split} phrase relations are incomplete")
        counts = {len(split_phrases[name]) for name in RELATIONS}
        if len(counts) != 1 or min(counts) < 2:
            raise ValueError(
                "each public relation needs the same number of phrase families"
            )
    if set(forbidden) != set(RELATIONS):
        raise ValueError("forbidden phrase relations are incomplete")
    train_entities = {_normalize_phrase(value) for value in entities["train"]}
    validation_entities = {
        _normalize_phrase(value) for value in entities["validation"]
    }
    if train_entities & validation_entities:
        raise ValueError("public train and validation entities must be disjoint")
    if set(frames["train"]) & set(frames["validation"]):
        raise ValueError("public train and validation frames must be disjoint")

    globally_assigned: dict[str, tuple[str, str]] = {}
    for split in ("train", "validation"):
        for relation in RELATIONS:
            reserved = {
                _normalize_phrase(value) for value in forbidden[relation]
            }
            current = {
                _normalize_phrase(value)
                for value in phrases[split][relation]
            }
            if len(current) != len(phrases[split][relation]):
                raise ValueError("a public relation contains duplicate phrases")
            overlap = current & reserved
            if overlap:
                raise ValueError(
                    f"public {split} reuses reserved phrases {sorted(overlap)}"
                )
            for phrase in current:
                if phrase in globally_assigned:
                    raise ValueError(
                        f"public phrase {phrase!r} is reused across lexical splits"
                    )
                globally_assigned[phrase] = (split, relation)


def build_public_relation_rows(
    corpus_config: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    """Build answer-free relation prompts with no entity-to-value mapping."""

    _validate_public_corpus_config(corpus_config)
    if split not in {"train", "validation"}:
        raise ValueError("public relation split must be train or validation")
    entities = [str(value) for value in corpus_config["entities"][split]]
    frames = [str(value) for value in corpus_config["frames"][split]]
    phrases = corpus_config["phrases"][split]
    rows = []
    for entity in entities:
        for relation in RELATIONS:
            for phrase_index, raw_phrase in enumerate(phrases[relation]):
                relation_phrase = _normalize_phrase(raw_phrase)
                for frame_index, frame in enumerate(frames):
                    rows.append(
                        {
                            "corpus_kind": "answer_free_public_relation",
                            "contains_private_answer": False,
                            "entity": entity,
                            "attribute": relation,
                            "relation_phrase": relation_phrase,
                            "lexical_split": f"public_{split}",
                            "template_split": f"public_{split}",
                            "template_id": (
                                f"public-{split}-p{phrase_index}-f{frame_index}"
                            ),
                            "fact_id": f"public-{split}:{entity}|{relation}",
                            "prompt": frame.format(
                                entity=entity,
                                relation_phrase=relation_phrase,
                            ),
                        }
                    )
    if any("answer" in row or "candidates" in row for row in rows):
        raise RuntimeError("public relation corpus unexpectedly contains answers")
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def prepare_public_relation_features(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    checkpoint = Path(values["checkpoint"])
    tokenizer_name = str(values["tokenizer"])
    private_cache_path = Path(values["private_feature_cache"])
    public_data_dir = Path(values["public_data_dir"])
    public_feature_cache = Path(values["public_feature_cache"])
    public_manifest = Path(values["public_manifest"])
    extraction = values.get("extraction", {})
    corpus_config = values.get("public_corpus", {})
    rows = {
        split: build_public_relation_rows(corpus_config, split)
        for split in ("train", "validation")
    }
    data_paths = {}
    for split, split_rows in rows.items():
        path = public_data_dir / f"{split}.jsonl"
        _write_jsonl(path, split_rows)
        data_paths[split] = path

    private_cache = _load_feature_cache(private_cache_path)
    selected_layer_indices = tuple(
        int(value) for value in private_cache["selected_layer_indices"]
    )
    baseline_layer_index = int(private_cache["q0_baseline_layer_index"])
    device = resolve_device(device_name)
    dtype = resolve_dtype(str(extraction.get("dtype", "bfloat16")), device)
    model, payload = build_model_from_checkpoint(
        checkpoint, device=device, dtype=dtype
    )
    core_sha256 = hash_non_auxiliary(payload["model"], expert_index=1)
    if core_sha256 != private_cache["source_core_sha256"]:
        raise ValueError("public and private feature sources use different cores")
    tokenizer = _load_tokenizer(tokenizer_name)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint")
    features = {
        split: extract_canonicalizer_features(
            model,
            tokenizer,
            split_rows,
            selected_layer_indices=selected_layer_indices,
            baseline_layer_index=baseline_layer_index,
            batch_size=int(extraction.get("batch_size", 64)),
            device=device,
            description=f"stage-c22:public-features:{split}",
        )
        for split, split_rows in rows.items()
    }
    public_feature_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "stage": "C2.2-public-relation-features",
            "answer_free": True,
            "source_core_sha256": core_sha256,
            "selected_layers_one_based": list(
                private_cache["selected_layers_one_based"]
            ),
            "selected_layer_indices": list(selected_layer_indices),
            "q0_baseline_layer_index": baseline_layer_index,
            "features": features,
            "metadata": rows,
        },
        public_feature_cache,
    )
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": STAGE_C22_SCHEMA_VERSION,
        "stage": "C2.2-answer-free-public-relation-corpus",
        "answer_free": True,
        "contains_entities": "synthetic public placeholders only",
        "contains_private_answers": False,
        "contains_entity_to_private_answer_mapping": False,
        "lexical_isolation_validated": True,
        "confirmation_accessed": False,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": _sha256_file(config_path),
        "source_core_sha256": core_sha256,
        "row_counts": {split: len(value) for split, value in rows.items()},
        "data_sha256": {
            split: _sha256_file(path) for split, path in data_paths.items()
        },
        "public_feature_cache": str(public_feature_cache.resolve()),
        "public_feature_cache_sha256": _sha256_file(public_feature_cache),
        "selected_layers_one_based": list(
            private_cache["selected_layers_one_based"]
        ),
        "phrase_counts_per_relation": {
            split: len(corpus_config["phrases"][split][RELATIONS[0]])
            for split in ("train", "validation")
        },
        "frame_counts": {
            split: len(corpus_config["frames"][split])
            for split in ("train", "validation")
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    public_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_manifest = output_dir / "public_corpus_manifest.json"
    if requested_manifest.resolve() != public_manifest.resolve():
        requested_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return manifest


def _load_public_feature_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "answer_free",
        "source_core_sha256",
        "selected_layer_indices",
        "features",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"public feature cache is missing {sorted(missing)}")
    if not bool(payload["answer_free"]):
        raise ValueError("public relation feature cache is not answer-free")
    for split in ("train", "validation"):
        if split not in payload["features"] or split not in payload["metadata"]:
            raise ValueError(f"public feature cache is missing {split}")
        rows = payload["metadata"][split]
        pools = payload["features"][split]["pools"]
        if len(rows) != len(pools):
            raise ValueError("public feature and metadata row counts differ")
        if any(
            "answer" in row
            or "candidates" in row
            or bool(row.get("contains_private_answer", True))
            for row in rows
        ):
            raise ValueError("public feature cache contains answer-bearing rows")
    return payload


def _public_targets(
    rows: Sequence[dict[str, Any]], labels: dict[str, Sequence[str]]
) -> dict[str, Tensor]:
    return {
        "relation": _targets(rows, labels["relations"], "attribute"),
        "entity": _targets(
            rows, _label_map(rows, "entity"), "entity"
        ),
        "template": _targets(
            rows, _label_map(rows, "template_id"), "template_id"
        ),
    }


def _public_relation_metrics(
    outputs: dict[str, dict[str, Tensor]],
    rows: dict[str, list[dict[str, Any]]],
    labels: dict[str, Sequence[str]],
    *,
    ridge_strength: float,
    template_probe_seed: int,
) -> dict[str, Any]:
    relation_lookup = {
        str(value): index for index, value in enumerate(labels["relations"])
    }
    validation_targets = torch.tensor(
        [relation_lookup[str(row["attribute"])] for row in rows["validation"]]
    )
    head_accuracy = float(
        outputs["validation"]["relation_logits"]
        .argmax(dim=1)
        .eq(validation_targets)
        .float()
        .mean()
    )
    probe = ridge_probe_accuracy(
        outputs["train"]["z_r"],
        [str(row["attribute"]) for row in rows["train"]],
        outputs["validation"]["z_r"],
        [str(row["attribute"]) for row in rows["validation"]],
        ridge_strength=ridge_strength,
    )
    template_probe = _template_leakage_probe(
        outputs["train"]["z_r"],
        rows["train"],
        ridge_strength=ridge_strength,
        seed=template_probe_seed,
    )
    geometry = relation_geometry_margin(
        outputs["validation"]["z_r"], rows["validation"]
    )
    template_threshold = float(template_probe["chance_accuracy"]) + 0.10
    gate_values = {
        "relation_head": (head_accuracy, 0.85, False),
        "relation_probe": (probe["accuracy"], 0.85, False),
        "relation_template_margin": (
            geometry["relation_template_margin"],
            0.15,
            False,
        ),
        "template_probe": (
            template_probe["accuracy"],
            template_threshold,
            True,
        ),
    }
    gates = {}
    for name, (value, threshold, lower) in gate_values.items():
        value = float(value)
        threshold = float(threshold)
        gates[name] = {
            "value": value,
            "threshold": threshold,
            "direction": "<=" if lower else ">=",
            "passed": value <= threshold if lower else value >= threshold,
        }
    passed_count = sum(bool(value["passed"]) for value in gates.values())
    return {
        "answer_free": True,
        "relation_head_accuracy": head_accuracy,
        "relation_probe": probe,
        "relation_geometry": geometry,
        "template_probe": template_probe,
        "strict_readiness": {
            "passed": passed_count == len(gates),
            "passed_count": passed_count,
            "total": len(gates),
            "gates": gates,
        },
    }


def _combined_validation_score(
    private_validation: dict[str, Any], public_validation: dict[str, Any]
) -> tuple[float, ...]:
    private_grade = private_validation["strict_readiness"]
    public_grade = public_validation["strict_readiness"]
    private_score = _selection_score(private_validation)
    return (
        float(private_grade["passed_count"] + public_grade["passed_count"]),
        float(private_grade["passed_count"]),
        float(public_grade["passed_count"]),
        float(private_validation["query_space"]["retrieval"]["centroid_top1_accuracy"]),
        float(public_validation["relation_head_accuracy"]),
        float(public_validation["relation_probe"]["accuracy"]),
        *private_score[1:],
    )


def _set_relation_tuning_scope(system: CanonicalizerSystem) -> list[nn.Parameter]:
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    modules = (
        system.canonicalizer.relation_encoder,
        system.relation_head,
        system.template_head,
    )
    parameters = []
    for module in modules:
        if module is None:
            raise ValueError("public relation tuning needs a factorized canonicalizer")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    return parameters


def train_public_relation_variant(
    variant: PublicRelationVariant,
    base_checkpoint: str | Path,
    private_features: dict[str, Tensor],
    private_rows: dict[str, list[dict[str, Any]]],
    public_features: dict[str, Tensor],
    public_rows: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    *,
    source_core_sha256: str,
    selected_layer_indices: Sequence[int],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    evaluation_interval: int,
    temperature: float,
    batch_facts: int,
    templates_per_fact: int,
    public_relation_ce_weight: float,
    public_relation_supcon_weight: float,
    public_template_adversary_weight: float,
    private_fact_supcon_weight: float,
    private_relation_ce_weight: float,
    private_relation_supcon_weight: float,
    private_template_adversary_weight: float,
    evaluation_batch_size: int,
    ridge_strength: float,
    template_probe_seed: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if variant.import_base:
        raise ValueError("P0 must be imported without training")
    if steps <= 0 or evaluation_interval <= 0:
        raise ValueError("training steps and evaluation interval must be positive")
    set_seed(seed)
    system, payload = load_canonicalizer_checkpoint(base_checkpoint, device=device)
    if payload["source_core_sha256"] != source_core_sha256:
        raise ValueError("base canonicalizer and public features use different cores")
    if tuple(payload["selected_core_layers"]) != tuple(selected_layer_indices):
        raise ValueError("base canonicalizer and public features use different layers")
    if system.config.architecture != "factorized":
        raise ValueError("public relation tuning needs factorized C2.1 weights")
    labels = {name: list(values) for name, values in payload["labels"].items()}
    if set(labels["relations"]) != set(RELATIONS):
        raise ValueError("base canonicalizer relation labels are incompatible")
    trainable = _set_relation_tuning_scope(system)
    public_templates = _label_map(public_rows["train"], "template_id")
    public_template_head = nn.Linear(
        system.config.query_dim, len(public_templates)
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*trainable, *public_template_head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    public_sampler = StructuredFactBatchSampler(
        public_rows["train"],
        batch_facts=batch_facts,
        templates_per_fact=templates_per_fact,
        seed=seed + 11,
        aligned_templates=True,
    )
    private_sampler = StructuredFactBatchSampler(
        private_rows["train"],
        batch_facts=batch_facts,
        templates_per_fact=templates_per_fact,
        seed=seed + 17,
        aligned_templates=True,
    )
    public_targets = _public_targets(public_rows["train"], labels)
    private_targets = {
        "fact": _targets(private_rows["train"], labels["facts"], "fact_id"),
        "entity": _targets(
            private_rows["train"], labels["entities"], "entity"
        ),
        "relation": _targets(
            private_rows["train"], labels["relations"], "attribute"
        ),
        "template": _targets(
            private_rows["train"], labels["templates"], "template_id"
        ),
    }
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_state: dict[str, Tensor] | None = None
    best_step = 0
    progress = tqdm(range(steps), desc=f"stage-c22:{variant.name}")
    for step in progress:
        optimizer.param_groups[0]["lr"] = learning_rate * _lr_scale(
            step, steps, warmup_fraction
        )
        system.train()
        public_template_head.train()
        public_indices = public_sampler.sample_indices()
        public_index = torch.tensor(public_indices, dtype=torch.long)
        public_batch_targets = {
            name: values[public_index].to(device)
            for name, values in public_targets.items()
        }
        public_output = system(
            public_features["train"][public_index].to(device),
            adversary_strength=0.0,
        )
        public_relation_ce = F.cross_entropy(
            public_output["relation_logits"],
            public_batch_targets["relation"],
        )
        public_relation_supcon = relation_supervised_contrastive_loss(
            public_output["relation_unit"],
            public_batch_targets["relation"],
            public_batch_targets["entity"],
            public_batch_targets["template"],
            temperature=temperature,
        )
        public_template_adversary = F.cross_entropy(
            public_template_head(
                gradient_reverse(public_output["relation_unit"], 1.0)
            ),
            public_batch_targets["template"],
        )
        public_loss = (
            public_relation_ce_weight * public_relation_ce
            + public_relation_supcon_weight * public_relation_supcon
            + public_template_adversary_weight * public_template_adversary
        )

        private_fact_supcon = torch.zeros((), device=device)
        private_relation_ce = torch.zeros((), device=device)
        private_relation_supcon = torch.zeros((), device=device)
        private_template_adversary = torch.zeros((), device=device)
        private_loss = torch.zeros((), device=device)
        if variant.private_replay_weight > 0:
            private_indices = private_sampler.sample_indices()
            private_index = torch.tensor(private_indices, dtype=torch.long)
            private_batch_targets = {
                name: values[private_index].to(device)
                for name, values in private_targets.items()
            }
            private_output = system(
                private_features["train"][private_index].to(device),
                adversary_strength=(
                    1.0 if private_template_adversary_weight > 0 else 0.0
                ),
            )
            private_fact_supcon = supervised_contrastive_loss(
                private_output["query"],
                private_batch_targets["fact"],
                temperature=temperature,
            )
            private_relation_ce = F.cross_entropy(
                private_output["relation_logits"],
                private_batch_targets["relation"],
            )
            private_relation_supcon = relation_supervised_contrastive_loss(
                private_output["relation_unit"],
                private_batch_targets["relation"],
                private_batch_targets["entity"],
                private_batch_targets["template"],
                temperature=temperature,
            )
            private_template_adversary = F.cross_entropy(
                private_output["template_logits"],
                private_batch_targets["template"],
            )
            private_loss = (
                private_fact_supcon_weight * private_fact_supcon
                + private_relation_ce_weight * private_relation_ce
                + private_relation_supcon_weight * private_relation_supcon
                + private_template_adversary_weight * private_template_adversary
            )
        loss = public_loss + variant.private_replay_weight * private_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*trainable, *public_template_head.parameters()], max_norm=5.0
        )
        optimizer.step()

        should_evaluate = (
            (step + 1) % evaluation_interval == 0 or step + 1 == steps
        )
        if should_evaluate:
            private_outputs = {
                split: _embed_branch_outputs(
                    system,
                    private_features[split],
                    batch_size=evaluation_batch_size,
                    device=device,
                )
                for split in ("train", "validation")
            }
            public_outputs = {
                split: _embed_branch_outputs(
                    system,
                    public_features[split],
                    batch_size=evaluation_batch_size,
                    device=device,
                )
                for split in ("train", "validation")
            }
            private_validation, _ = _evaluate_c21_split(
                private_outputs,
                private_rows,
                labels,
                "validation",
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
            )
            public_validation = _public_relation_metrics(
                public_outputs,
                public_rows,
                labels,
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
            )
            score = _combined_validation_score(
                private_validation, public_validation
            )
            row = {
                "step": step + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": float(loss.detach().cpu()),
                "public_loss": float(public_loss.detach().cpu()),
                "public_relation_ce_loss": float(
                    public_relation_ce.detach().cpu()
                ),
                "public_relation_supcon_loss": float(
                    public_relation_supcon.detach().cpu()
                ),
                "public_template_adversary_loss": float(
                    public_template_adversary.detach().cpu()
                ),
                "private_loss": float(private_loss.detach().cpu()),
                "private_fact_supcon_loss": float(
                    private_fact_supcon.detach().cpu()
                ),
                "private_relation_ce_loss": float(
                    private_relation_ce.detach().cpu()
                ),
                "private_relation_supcon_loss": float(
                    private_relation_supcon.detach().cpu()
                ),
                "private_template_adversary_loss": float(
                    private_template_adversary.detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "private_validation_gate_count": private_validation[
                    "strict_readiness"
                ]["passed_count"],
                "public_validation_gate_count": public_validation[
                    "strict_readiness"
                ]["passed_count"],
                "private_validation_relation_head": private_validation[
                    "auxiliary_heads"
                ]["relation_accuracy"],
                "private_validation_centroid_top1": private_validation[
                    "query_space"
                ]["retrieval"]["centroid_top1_accuracy"],
                "public_validation_relation_head": public_validation[
                    "relation_head_accuracy"
                ],
                "public_validation_relation_probe": public_validation[
                    "relation_probe"
                ]["accuracy"],
                "public_validation_relation_template_margin": public_validation[
                    "relation_geometry"
                ]["relation_template_margin"],
                "public_train_template_probe": public_validation[
                    "template_probe"
                ]["accuracy"],
            }
            history.append(row)
            if best_score is None or score > best_score:
                best_score = score
                best_state = _clone_state(system)
                best_step = step + 1
            progress.set_postfix(
                private=f"{private_validation['strict_readiness']['passed_count']}/10",
                public=f"{public_validation['strict_readiness']['passed_count']}/4",
                pub_rel=f"{public_validation['relation_head_accuracy']:.3f}",
            )

    assert best_state is not None and best_score is not None
    system.load_state_dict(best_state, strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_canonicalizer_checkpoint(
        output_dir / "canonicalizer.pt",
        system,
        variant=variant.name,
        selected_core_layers=selected_layer_indices,
        labels=labels,
        source_core_sha256=source_core_sha256,
    )
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "system": system,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "labels": labels,
        "best_step": best_step,
        "selection_score": list(best_score),
        "history": history,
        "trainable_scope": [
            "canonicalizer.relation_encoder",
            "relation_head",
            "template_head",
            "ephemeral_public_template_head",
        ],
    }


def _stage_c22_ablation_row(name: str, detail: dict[str, Any]) -> dict[str, Any]:
    validation = detail["private_evaluation"]["validation"]
    development = detail["private_evaluation"]["development"]
    public = detail["public_validation"]
    query = development["query_space"]
    return {
        "variant": name,
        "exact_c21_base": detail["exact_c21_base"],
        "private_replay_weight": detail["variant"]["private_replay_weight"],
        "best_step": detail["best_step"],
        "private_validation_gates": validation["strict_readiness"][
            "passed_count"
        ],
        "public_validation_gates": public["strict_readiness"]["passed_count"],
        "development_gates": development["strict_readiness"]["passed_count"],
        "public_validation_relation_head": public["relation_head_accuracy"],
        "public_validation_relation_probe": public["relation_probe"]["accuracy"],
        "public_validation_relation_template_margin": public[
            "relation_geometry"
        ]["relation_template_margin"],
        "public_train_template_probe": public["template_probe"]["accuracy"],
        "development_relation_head": development["auxiliary_heads"][
            "relation_accuracy"
        ],
        "development_relation_probe": development["branch_information_matrix"][
            "z_r"
        ]["relation"]["accuracy"],
        "development_relation_template_margin": development[
            "relation_geometry"
        ]["relation_template_margin"],
        "development_centroid_top1": query["retrieval"][
            "centroid_top1_accuracy"
        ],
        "development_row_1nn": query["retrieval"]["row_1nn_accuracy"],
        "development_mrr": query["retrieval"]["centroid_mrr"],
        "development_fact_silhouette": query["silhouette"]["fact_id"],
        "development_fact_template_margin": query["geometry"][
            "fact_over_template_margin"
        ],
        "development_entity_probe": query["linear_probes"]["entity_id"][
            "accuracy"
        ],
        "development_template_probe": development["query_template_probe"][
            "accuracy"
        ],
    }


def run_stage_c22(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    private_cache_path = Path(values["private_feature_cache"])
    public_cache_path = Path(values["public_feature_cache"])
    base_checkpoint = Path(values["base_canonicalizer"])
    public_manifest_path = Path(values["public_manifest"])
    public_confirmation_seal = Path(values["public_confirmation_seal"])
    training = values.get("training", {})
    evaluation_config = values.get("evaluation", {})
    variants = parse_public_variants(values.get("variants"))

    private_cache = _load_feature_cache(private_cache_path)
    public_cache = _load_public_feature_cache(public_cache_path)
    if public_cache["source_core_sha256"] != private_cache["source_core_sha256"]:
        raise ValueError("public and private caches use different frozen cores")
    if tuple(public_cache["selected_layer_indices"]) != tuple(
        private_cache["selected_layer_indices"]
    ):
        raise ValueError("public and private caches use different core layers")
    manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("answer_free")) or bool(
        manifest.get("contains_private_answers", True)
    ):
        raise ValueError("public corpus manifest does not certify answer-free data")
    public_seal = json.loads(
        public_confirmation_seal.read_text(encoding="utf-8")
    )
    if bool(public_seal.get("confirmation_accessed", True)):
        raise ValueError("public confirmation seal does not certify zero access")

    private_rows = {
        split: list(private_cache["metadata"][split])
        for split in ("train", "validation", "test")
    }
    private_features = {
        split: private_cache["features"][split]["pools"].float()
        for split in private_rows
    }
    public_rows = {
        split: list(public_cache["metadata"][split])
        for split in ("train", "validation")
    }
    public_features = {
        split: public_cache["features"][split]["pools"].float()
        for split in public_rows
    }
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(training.get("seed", 0))
    evaluation_batch_size = int(evaluation_config.get("batch_size", 128))
    ridge_strength = float(evaluation_config.get("ridge_strength", 0.01))
    template_probe_seed = int(
        evaluation_config.get("template_probe_seed", 91)
    )
    selected_layer_indices = tuple(
        int(value) for value in private_cache["selected_layer_indices"]
    )
    details: dict[str, Any] = {}
    ablation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        if variant.import_base:
            system, payload = load_canonicalizer_checkpoint(
                base_checkpoint, device=device
            )
            if payload["source_core_sha256"] != private_cache["source_core_sha256"]:
                raise ValueError("P0 base and feature cache core hashes differ")
            trained = {
                "system": system,
                "checkpoint": str(base_checkpoint.resolve()),
                "checkpoint_sha256": _sha256_file(base_checkpoint),
                "labels": {
                    name: list(items)
                    for name, items in payload["labels"].items()
                },
                "best_step": 0,
                "trainable_scope": [],
            }
            legacy_evaluation = base_checkpoint.parent / "evaluation.json"
            if legacy_evaluation.exists():
                trained["best_step"] = int(
                    json.loads(
                        legacy_evaluation.read_text(encoding="utf-8")
                    ).get("best_step", 0)
                )
        else:
            trained = train_public_relation_variant(
                variant,
                base_checkpoint,
                private_features,
                private_rows,
                public_features,
                public_rows,
                variant_dir,
                source_core_sha256=str(private_cache["source_core_sha256"]),
                selected_layer_indices=selected_layer_indices,
                steps=int(training.get("steps", 1000)),
                learning_rate=float(training.get("learning_rate", 2e-4)),
                weight_decay=float(training.get("weight_decay", 0.01)),
                warmup_fraction=float(training.get("warmup_fraction", 0.05)),
                evaluation_interval=int(
                    training.get("evaluation_interval", 50)
                ),
                temperature=float(training.get("temperature", 0.07)),
                batch_facts=int(training.get("batch_facts", 8)),
                templates_per_fact=int(
                    training.get("templates_per_fact", 3)
                ),
                public_relation_ce_weight=float(
                    training.get("public_relation_ce_weight", 0.5)
                ),
                public_relation_supcon_weight=float(
                    training.get("public_relation_supcon_weight", 0.2)
                ),
                public_template_adversary_weight=float(
                    training.get("public_template_adversary_weight", 0.05)
                ),
                private_fact_supcon_weight=float(
                    training.get("private_fact_supcon_weight", 1.0)
                ),
                private_relation_ce_weight=float(
                    training.get("private_relation_ce_weight", 0.5)
                ),
                private_relation_supcon_weight=float(
                    training.get("private_relation_supcon_weight", 0.2)
                ),
                private_template_adversary_weight=float(
                    training.get("private_template_adversary_weight", 0.05)
                ),
                evaluation_batch_size=evaluation_batch_size,
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
                seed=seed,
                device=device,
            )
        system = trained["system"]
        private_outputs = {
            split: _embed_branch_outputs(
                system,
                features,
                batch_size=evaluation_batch_size,
                device=device,
            )
            for split, features in private_features.items()
        }
        public_outputs = {
            split: _embed_branch_outputs(
                system,
                features,
                batch_size=evaluation_batch_size,
                device=device,
            )
            for split, features in public_features.items()
        }
        private_evaluation = {}
        for public_name, source_split in (
            ("validation", "validation"),
            ("development", "test"),
        ):
            evaluation, predictions = _evaluate_c21_split(
                private_outputs,
                private_rows,
                trained["labels"],
                source_split,
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
            )
            private_evaluation[public_name] = evaluation
            for row in predictions:
                row.update(
                    {
                        "variant": variant.name,
                        "query_split": public_name,
                    }
                )
            prediction_rows.extend(predictions)
        public_validation = _public_relation_metrics(
            public_outputs,
            public_rows,
            trained["labels"],
            ridge_strength=ridge_strength,
            template_probe_seed=template_probe_seed,
        )
        score = list(
            _combined_validation_score(
                private_evaluation["validation"], public_validation
            )
        )
        detail = {
            "variant": asdict(variant),
            "exact_c21_base": variant.import_base,
            "checkpoint": trained["checkpoint"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
            "best_step": trained["best_step"],
            "trainable_scope": trained["trainable_scope"],
            "selection_split": "private_validation+public_validation",
            "selection_score": score,
            "private_evaluation": private_evaluation,
            "public_validation": public_validation,
        }
        details[variant.name] = detail
        ablation_rows.append(_stage_c22_ablation_row(variant.name, detail))
        (variant_dir / "evaluation.json").write_text(
            json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        system.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_variant = max(
        (variant.name for variant in variants),
        key=lambda name: tuple(details[name]["selection_score"]),
    )
    selected = details[selected_variant]
    development_ready = bool(
        selected["private_evaluation"]["development"]["strict_readiness"][
            "passed"
        ]
        and selected["public_validation"]["strict_readiness"]["passed"]
    )
    development = selected["private_evaluation"]["development"]
    development_relation_ready = (
        float(development["auxiliary_heads"]["relation_accuracy"]) >= 0.85
        and float(
            development["branch_information_matrix"]["z_r"]["relation"][
                "accuracy"
            ]
        )
        >= 0.85
    )
    if development_ready:
        recommended_next_step = (
            "evaluate_selected_public_relation_variant_once_on_confirmation"
        )
    elif not development_relation_ready:
        recommended_next_step = (
            "public_paraphrases_insufficient_use_stronger_public_semantic_encoder"
        )
    else:
        recommended_next_step = (
            "preserve_public_relation_encoder_and_improve_fact_geometry"
        )

    with (output_dir / "stage_c22_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    with (output_dir / "retrieval_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    resolved_config = {
        "source_config": str(config_path.resolve()),
        "values": values,
        "private_feature_cache_sha256": _sha256_file(private_cache_path),
        "public_feature_cache_sha256": _sha256_file(public_cache_path),
        "public_manifest_sha256": _sha256_file(public_manifest_path),
        "source_core_sha256": private_cache["source_core_sha256"],
        "confirmation_local_data_read": False,
        "development_source_split": "test",
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "schema_version": STAGE_C22_SCHEMA_VERSION,
        "stage": "C2.2-answer-free-public-relation-supervision",
        "status": (
            "ready_for_one_time_confirmation"
            if development_ready
            else "development_gates_failed"
        ),
        "core_frozen": True,
        "public_corpus_answer_free": True,
        "private_answers_used_as_training_targets": False,
        "confirmation_accessed": False,
        "confirmation_access_count": 0,
        "c3_eligible": False,
        "development_ready_for_confirmation": development_ready,
        "development_relation_ready": development_relation_ready,
        "selected_variant": selected_variant,
        "selected_variant_validation_score": selected["selection_score"],
        "recommended_next_step": recommended_next_step,
        "selection_protocol": (
            "checkpoints and P0-P2 use only private validation plus disjoint "
            "answer-free public validation; legacy test is development-only"
        ),
        "confirmation_protocol": (
            "only the public seal was read; local confirmation templates and "
            "rows remain unread"
        ),
        "public_corpus_manifest": manifest,
        "public_confirmation_seal": {
            "path": str(public_confirmation_seal.resolve()),
            "sha256": _sha256_file(public_confirmation_seal),
            "templates_sha256": public_seal.get("templates_sha256"),
            "rows_sha256": public_seal.get("rows_sha256"),
        },
        "variants": details,
        "ablation_rows": ablation_rows,
        "environment": collect_environment(device, torch.float32),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "ablation_csv": str((output_dir / "stage_c22_ablation.csv").resolve()),
            "retrieval_predictions": str(
                (output_dir / "retrieval_predictions.csv").resolve()
            ),
            "resolved_config": str((output_dir / "resolved_config.json").resolve()),
        },
    }
    summary_path = output_dir / "stage_c22_summary.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary
