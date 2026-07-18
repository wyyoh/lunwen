from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm.auto import tqdm

from .canonicalizer import (
    CanonicalizerConfig,
    CanonicalizerSystem,
    POOL_NAMES,
    StructuredFactBatchSampler,
    save_canonicalizer_checkpoint,
    supervised_contrastive_loss,
)
from .checkpoint import build_model_from_checkpoint
from .keying import hash_non_auxiliary
from .phase_a import _load_tokenizer, _read_jsonl, _sha256_file
from .stage_c1 import (
    _validate_rows,
    cosine_silhouette,
    cross_template_retrieval,
    representation_geometry,
    ridge_probe_accuracy,
)
from .train import collect_environment, resolve_device, resolve_dtype, set_seed


STAGE_C2_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QueryVariant:
    name: str
    architecture: str
    relation_weight: float
    entity_weight: float
    template_adversary_weight: float


DEFAULT_VARIANTS = (
    QueryVariant("Q1", "joint", 0.0, 0.0, 0.0),
    QueryVariant("Q2", "factorized", 0.2, 0.0, 0.0),
    QueryVariant("Q3", "factorized", 0.2, 0.2, 0.0),
    QueryVariant("Q4", "factorized", 0.2, 0.2, 0.1),
)


def locate_entity_token_span(
    tokenizer, prompt: str, entity: str
) -> tuple[list[int], list[int]]:
    """Return prompt token ids and all tokens overlapping the literal entity span."""

    start = prompt.find(entity)
    if start < 0:
        raise ValueError(f"entity {entity!r} is absent from prompt")
    if prompt.find(entity, start + 1) >= 0:
        raise ValueError(f"entity {entity!r} occurs more than once in prompt")
    end = start + len(entity)
    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    entity_positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not entity_positions:
        raise ValueError("tokenizer produced no token overlapping the entity span")
    if input_ids != tokenizer.encode(prompt, add_special_tokens=False):
        raise ValueError("offset tokenizer encoding differs from plain encoding")
    return input_ids, entity_positions


@torch.inference_mode()
def extract_canonicalizer_features(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    *,
    selected_layer_indices: Sequence[int],
    baseline_layer_index: int,
    batch_size: int,
    device: torch.device,
    description: str,
) -> dict[str, Tensor]:
    if not selected_layer_indices:
        raise ValueError("select at least one core layer")
    layer_indices = tuple(int(value) for value in selected_layer_indices)
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("selected core layers must be unique")
    if any(value < 0 or value >= model.config.num_layers for value in layer_indices):
        raise ValueError("selected core layer is out of range")
    if not 0 <= baseline_layer_index < model.config.num_layers:
        raise ValueError("baseline core layer is out of range")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    layouts = []
    for row in rows:
        ids, entity_positions = locate_entity_token_span(
            tokenizer, str(row["prompt"]), str(row["entity"])
        )
        entity_set = set(entity_positions)
        question_positions = [
            index for index in range(len(ids)) if index not in entity_set
        ]
        if not question_positions:
            raise ValueError("question pooling has no non-entity tokens")
        layouts.append((ids, entity_positions, question_positions))

    pools = torch.empty(
        len(rows),
        len(POOL_NAMES),
        len(layer_indices),
        model.config.hidden_size,
        dtype=torch.float32,
    )
    baseline = torch.empty(
        len(rows), model.config.hidden_size, dtype=torch.float32
    )
    core_mask = torch.tensor(
        [1] + [0] * model.config.num_aux, device=device, dtype=torch.long
    )
    eos = int(tokenizer.eos_token_id)
    model.eval()
    for offset in tqdm(
        range(0, len(rows), batch_size), desc=description, leave=False
    ):
        batch_layouts = layouts[offset : offset + batch_size]
        width = max(len(layout[0]) for layout in batch_layouts)
        if width > model.config.context_length:
            raise ValueError("a Stage-C2 prompt exceeds model context length")
        input_ids = torch.full(
            (len(batch_layouts), width), eos, dtype=torch.long, device=device
        )
        for batch_index, (ids, _, _) in enumerate(batch_layouts):
            input_ids[batch_index, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
        _, _, diagnostics = model(
            input_ids,
            fwd_mask=core_mask,
            bck_mask=core_mask,
            return_diagnostics=True,
        )
        for batch_index, (ids, entity_positions, question_positions) in enumerate(
            batch_layouts
        ):
            output_index = offset + batch_index
            baseline[output_index] = diagnostics["aux_inputs"][baseline_layer_index][
                batch_index, len(ids) - 1
            ].float().cpu()
            for selected_index, layer in enumerate(layer_indices):
                hidden = diagnostics["aux_inputs"][layer][batch_index].float()
                pools[output_index, 0, selected_index] = hidden[
                    entity_positions
                ].mean(dim=0).cpu()
                pools[output_index, 1, selected_index] = hidden[
                    question_positions
                ].mean(dim=0).cpu()
                pools[output_index, 2, selected_index] = hidden[
                    len(ids) - 1
                ].cpu()
    return {"pools": pools, "baseline_last": baseline}


def _label_map(rows: Sequence[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(row[field]) for row in rows})


def _targets(rows: Sequence[dict[str, Any]], values: Sequence[str], field: str) -> Tensor:
    lookup = {value: index for index, value in enumerate(values)}
    return torch.tensor([lookup[str(row[field])] for row in rows], dtype=torch.long)


def _template_leakage_probe(
    embeddings: Tensor,
    rows: Sequence[dict[str, Any]],
    *,
    ridge_strength: float,
    seed: int,
) -> dict[str, Any]:
    entities = sorted({str(row["entity"]) for row in rows})
    if len(entities) < 4:
        raise ValueError("template probe needs at least four entities")
    rng = np.random.default_rng(seed)
    order = [entities[int(index)] for index in rng.permutation(len(entities))]
    split = max(1, round(0.75 * len(order)))
    probe_train_entities = set(order[:split])
    train_indices = [
        index
        for index, row in enumerate(rows)
        if str(row["entity"]) in probe_train_entities
    ]
    evaluation_indices = [
        index
        for index, row in enumerate(rows)
        if str(row["entity"]) not in probe_train_entities
    ]
    result = ridge_probe_accuracy(
        embeddings[train_indices],
        [str(rows[index]["template_id"]) for index in train_indices],
        embeddings[evaluation_indices],
        [str(rows[index]["template_id"]) for index in evaluation_indices],
        ridge_strength=ridge_strength,
    )
    return {
        **result,
        "probe_train_entities": len(probe_train_entities),
        "probe_evaluation_entities": len(entities) - len(probe_train_entities),
    }


def _query_space_metrics(
    database_embeddings: Tensor,
    database_rows: Sequence[dict[str, Any]],
    evaluation_embeddings: Tensor,
    evaluation_rows: Sequence[dict[str, Any]],
    *,
    ridge_strength: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieval, predictions = cross_template_retrieval(
        database_embeddings,
        database_rows,
        evaluation_embeddings,
        evaluation_rows,
    )
    geometry = representation_geometry(
        database_embeddings,
        database_rows,
        evaluation_embeddings,
        evaluation_rows,
    )
    combined = torch.cat([database_embeddings, evaluation_embeddings], dim=0)
    combined_rows = [*database_rows, *evaluation_rows]
    probes = {}
    for name, field in (
        ("entity_id", "entity"),
        ("relation_id", "attribute"),
        ("answer_token", "answer"),
    ):
        probes[name] = ridge_probe_accuracy(
            database_embeddings,
            [str(row[field]) for row in database_rows],
            evaluation_embeddings,
            [str(row[field]) for row in evaluation_rows],
            ridge_strength=ridge_strength,
        )
    silhouette = {
        "fact_id": cosine_silhouette(
            combined, [str(row["fact_id"]) for row in combined_rows]
        ),
        "entity_id": cosine_silhouette(
            combined, [str(row["entity"]) for row in combined_rows]
        ),
        "relation_id": cosine_silhouette(
            combined, [str(row["attribute"]) for row in combined_rows]
        ),
    }
    return {
        "retrieval": retrieval,
        "geometry": geometry,
        "linear_probes": probes,
        "silhouette": silhouette,
    }, predictions


@torch.inference_mode()
def _embed(
    system: CanonicalizerSystem,
    features: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    system.eval()
    output = []
    for offset in range(0, len(features), batch_size):
        query = system(features[offset : offset + batch_size].to(device))["query"]
        output.append(query.float().cpu())
    return torch.cat(output, dim=0)


def _lr_scale(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _validation_selection(
    database_embeddings: Tensor,
    database_rows: Sequence[dict[str, Any]],
    validation_embeddings: Tensor,
    validation_rows: Sequence[dict[str, Any]],
    *,
    ridge_strength: float,
    template_probe_seed: int,
    q0_template_accuracy: float,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    retrieval, _ = cross_template_retrieval(
        database_embeddings,
        database_rows,
        validation_embeddings,
        validation_rows,
    )
    geometry = representation_geometry(
        database_embeddings,
        database_rows,
        validation_embeddings,
        validation_rows,
    )
    combined = torch.cat([database_embeddings, validation_embeddings], dim=0)
    combined_rows = [*database_rows, *validation_rows]
    silhouette = cosine_silhouette(
        combined, [str(row["fact_id"]) for row in combined_rows]
    )
    relation_probe = ridge_probe_accuracy(
        database_embeddings,
        [str(row["attribute"]) for row in database_rows],
        validation_embeddings,
        [str(row["attribute"]) for row in validation_rows],
        ridge_strength=ridge_strength,
    )
    entity_probe = ridge_probe_accuracy(
        database_embeddings,
        [str(row["entity"]) for row in database_rows],
        validation_embeddings,
        [str(row["entity"]) for row in validation_rows],
        ridge_strength=ridge_strength,
    )
    template_probe = _template_leakage_probe(
        database_embeddings,
        database_rows,
        ridge_strength=ridge_strength,
        seed=template_probe_seed,
    )
    template_threshold = max(
        float(template_probe["chance_accuracy"]) + 0.10,
        q0_template_accuracy * 0.50,
    )
    partial_checks = (
        retrieval["row_1nn_accuracy"] >= 0.50,
        retrieval["centroid_top1_accuracy"] >= 0.60,
        retrieval["centroid_mrr"] >= 0.70,
        silhouette > 0.0,
        geometry["fact_over_template_margin"] > 0.0,
        relation_probe["accuracy"] >= 0.85,
        entity_probe["accuracy"] >= 0.70,
        template_probe["accuracy"] <= template_threshold,
    )
    gate_count = sum(bool(value) for value in partial_checks)
    score = (
        float(gate_count),
        float(retrieval["row_1nn_accuracy"]),
        float(retrieval["centroid_top1_accuracy"]),
        float(retrieval["centroid_mrr"]),
        float(geometry["fact_over_template_margin"]),
        float(relation_probe["accuracy"]),
        float(entity_probe["accuracy"]),
        -float(template_probe["accuracy"]),
    )
    return score, {
        "retrieval": retrieval,
        "geometry": geometry,
        "fact_silhouette": silhouette,
        "relation_probe": relation_probe,
        "entity_probe": entity_probe,
        "template_probe": template_probe,
        "partial_gate_count": gate_count,
        "partial_gate_total": len(partial_checks),
    }


def _clone_state(system: CanonicalizerSystem) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in system.state_dict().items()
    }


def train_query_variant(
    variant: QueryVariant,
    split_features: dict[str, Tensor],
    split_rows: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    *,
    selected_layer_indices: Sequence[int],
    core_hidden_size: int,
    core_sha256: str,
    steps: int,
    batch_facts: int,
    templates_per_fact: int,
    query_dim: int,
    mlp_hidden_size: int,
    dropout: float,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    evaluation_interval: int,
    evaluation_batch_size: int,
    seed: int,
    device: torch.device,
    ridge_strength: float,
    template_probe_seed: int,
    q0_template_accuracy: float,
) -> dict[str, Any]:
    if steps <= 0 or evaluation_interval <= 0:
        raise ValueError("training steps and evaluation interval must be positive")
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = split_rows["train"]
    entities = _label_map(train_rows, "entity")
    relations = _label_map(train_rows, "attribute")
    templates = _label_map(train_rows, "template_id")
    facts = _label_map(train_rows, "fact_id")
    config = CanonicalizerConfig(
        architecture=variant.architecture,
        num_input_layers=len(selected_layer_indices),
        core_hidden_size=core_hidden_size,
        query_dim=query_dim,
        mlp_hidden_size=mlp_hidden_size,
        dropout=dropout,
        num_entities=len(entities),
        num_relations=len(relations),
        num_templates=len(templates),
    )
    system = CanonicalizerSystem(config).to(device)
    sampler = StructuredFactBatchSampler(
        train_rows,
        batch_facts=batch_facts,
        templates_per_fact=templates_per_fact,
        seed=seed + 1,
    )
    all_targets = {
        "fact": _targets(train_rows, facts, "fact_id"),
        "entity": _targets(train_rows, entities, "entity"),
        "relation": _targets(train_rows, relations, "attribute"),
        "template": _targets(train_rows, templates, "template_id"),
    }
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    best_score: tuple[float, ...] | None = None
    best_state: dict[str, Tensor] | None = None
    best_step = 0
    progress = tqdm(range(steps), desc=f"stage-c2:{variant.name}")
    for step in progress:
        scale = _lr_scale(step, steps, warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * scale
        indices = sampler.sample_indices()
        index_tensor = torch.tensor(indices, dtype=torch.long)
        features = split_features["train"][index_tensor].to(device)
        targets = {
            name: values[index_tensor].to(device) for name, values in all_targets.items()
        }
        adversary_weight = variant.template_adversary_weight * min(
            (step + 1) / max(round(0.5 * steps), 1), 1.0
        )
        output = system(
            features,
            adversary_strength=1.0 if adversary_weight > 0 else 0.0,
        )
        fact_loss = supervised_contrastive_loss(
            output["query"], targets["fact"], temperature=temperature
        )
        relation_loss = F.cross_entropy(output["relation_logits"], targets["relation"])
        entity_loss = F.cross_entropy(output["entity_logits"], targets["entity"])
        template_loss = F.cross_entropy(output["template_logits"], targets["template"])
        loss = (
            fact_loss
            + variant.relation_weight * relation_loss
            + variant.entity_weight * entity_loss
            + adversary_weight * template_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            system.parameters(), max_norm=5.0
        )
        optimizer.step()

        should_evaluate = (step + 1) % evaluation_interval == 0 or step + 1 == steps
        if should_evaluate:
            database_embeddings = _embed(
                system,
                split_features["train"],
                batch_size=evaluation_batch_size,
                device=device,
            )
            validation_embeddings = _embed(
                system,
                split_features["validation"],
                batch_size=evaluation_batch_size,
                device=device,
            )
            selection_score, validation = _validation_selection(
                database_embeddings,
                split_rows["train"],
                validation_embeddings,
                split_rows["validation"],
                ridge_strength=ridge_strength,
                template_probe_seed=template_probe_seed,
                q0_template_accuracy=q0_template_accuracy,
            )
            row = {
                "step": step + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": float(loss.detach().cpu()),
                "fact_loss": float(fact_loss.detach().cpu()),
                "relation_loss": float(relation_loss.detach().cpu()),
                "entity_loss": float(entity_loss.detach().cpu()),
                "template_loss": float(template_loss.detach().cpu()),
                "template_adversary_weight": adversary_weight,
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "validation_row_1nn": validation["retrieval"]["row_1nn_accuracy"],
                "validation_centroid_top1": validation["retrieval"]["centroid_top1_accuracy"],
                "validation_mrr": validation["retrieval"]["centroid_mrr"],
                "validation_fact_template_margin": validation["geometry"]["fact_over_template_margin"],
                "validation_fact_silhouette": validation["fact_silhouette"],
                "validation_relation_probe": validation["relation_probe"]["accuracy"],
                "validation_entity_probe": validation["entity_probe"]["accuracy"],
                "validation_template_probe": validation["template_probe"]["accuracy"],
                "validation_partial_gate_count": validation["partial_gate_count"],
            }
            history.append(row)
            if best_score is None or selection_score > best_score:
                best_score = selection_score
                best_state = _clone_state(system)
                best_step = step + 1
            progress.set_postfix(
                val_1nn=f"{validation['retrieval']['row_1nn_accuracy']:.3f}",
                margin=f"{validation['geometry']['fact_over_template_margin']:.3f}",
                gates=f"{validation['partial_gate_count']}/8",
            )
    assert best_state is not None and best_score is not None
    system.load_state_dict(best_state, strict=True)
    checkpoint = save_canonicalizer_checkpoint(
        output_dir / "canonicalizer.pt",
        system,
        variant=variant.name,
        selected_core_layers=selected_layer_indices,
        labels={
            "entities": entities,
            "relations": relations,
            "templates": templates,
            "facts": facts,
        },
        source_core_sha256=core_sha256,
    )
    embeddings = {
        split: _embed(
            system,
            features,
            batch_size=evaluation_batch_size,
            device=device,
        )
        for split, features in split_features.items()
    }
    torch.save(
        {
            "format_version": 1,
            "variant": variant.name,
            "embeddings": embeddings,
        },
        output_dir / "query_embeddings.pt",
    )
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "system": system,
        "embeddings": embeddings,
        "checkpoint": str(checkpoint.resolve()),
        "best_step": best_step,
        "best_validation_score": list(best_score),
        "history": history,
        "config": asdict(config),
        "labels": {
            "entities": entities,
            "relations": relations,
            "templates": templates,
            "facts": facts,
        },
    }


def _evaluate_variant(
    name: str,
    embeddings: dict[str, Tensor],
    split_rows: dict[str, list[dict[str, Any]]],
    *,
    ridge_strength: float,
    template_probe_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    prediction_rows = []
    for split in ("validation", "test"):
        metrics, predictions = _query_space_metrics(
            embeddings["train"],
            split_rows["train"],
            embeddings[split],
            split_rows[split],
            ridge_strength=ridge_strength,
        )
        result[split] = metrics
        for row in predictions:
            row.update({"variant": name, "query_split": split})
        prediction_rows.extend(predictions)
    result["template_leakage_probe"] = _template_leakage_probe(
        embeddings["train"],
        split_rows["train"],
        ridge_strength=ridge_strength,
        seed=template_probe_seed,
    )
    return result, prediction_rows


@torch.inference_mode()
def _auxiliary_head_metrics(
    system: CanonicalizerSystem,
    features: Tensor,
    rows: Sequence[dict[str, Any]],
    labels: dict[str, Sequence[str]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    relation_lookup = {
        value: index for index, value in enumerate(labels["relations"])
    }
    entity_lookup = {
        value: index for index, value in enumerate(labels["entities"])
    }
    relation_targets = torch.tensor(
        [relation_lookup[str(row["attribute"])] for row in rows]
    )
    entity_targets = torch.tensor(
        [entity_lookup[str(row["entity"])] for row in rows]
    )
    relation_logits = []
    entity_logits = []
    system.eval()
    for offset in range(0, len(features), batch_size):
        output = system(features[offset : offset + batch_size].to(device))
        relation_logits.append(output["relation_logits"].float().cpu())
        entity_logits.append(output["entity_logits"].float().cpu())
    relation_prediction = torch.cat(relation_logits).argmax(dim=1)
    entity_prediction = torch.cat(entity_logits).argmax(dim=1)
    return {
        "relation_accuracy": float(
            relation_prediction.eq(relation_targets).float().mean()
        ),
        "entity_accuracy": float(entity_prediction.eq(entity_targets).float().mean()),
    }


def _gate(value: float, threshold: float, *, lower_is_better: bool = False) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "passed": value <= threshold if lower_is_better else value >= threshold,
    }


def _grade_variant(
    evaluation: dict[str, Any], q0_template_accuracy: float
) -> dict[str, Any]:
    test = evaluation["test"]
    retrieval = test["retrieval"]
    geometry = test["geometry"]
    probes = test["linear_probes"]
    silhouette = test["silhouette"]
    template = evaluation["template_leakage_probe"]
    partial_template_threshold = max(
        float(template["chance_accuracy"]) + 0.10, q0_template_accuracy * 0.50
    )
    strong_template_threshold = float(template["chance_accuracy"]) + 0.05
    partial = {
        "row_1nn": _gate(retrieval["row_1nn_accuracy"], 0.50),
        "centroid_top1": _gate(retrieval["centroid_top1_accuracy"], 0.60),
        "mrr": _gate(retrieval["centroid_mrr"], 0.70),
        "fact_silhouette": _gate(silhouette["fact_id"], 0.0),
        "fact_template_margin": _gate(geometry["fact_over_template_margin"], 0.0),
        "relation_probe": _gate(probes["relation_id"]["accuracy"], 0.85),
        "entity_probe": _gate(probes["entity_id"]["accuracy"], 0.70),
        "template_probe": _gate(
            template["accuracy"], partial_template_threshold, lower_is_better=True
        ),
    }
    strong = {
        "row_1nn": _gate(retrieval["row_1nn_accuracy"], 0.75),
        "centroid_top1": _gate(retrieval["centroid_top1_accuracy"], 0.80),
        "mrr": _gate(retrieval["centroid_mrr"], 0.90),
        "fact_silhouette": _gate(silhouette["fact_id"], 0.20),
        "fact_template_margin": _gate(geometry["fact_over_template_margin"], 0.15),
        "relation_probe": _gate(probes["relation_id"]["accuracy"], 0.95),
        "entity_probe": _gate(probes["entity_id"]["accuracy"], 0.90),
        "template_probe": _gate(
            template["accuracy"], strong_template_threshold, lower_is_better=True
        ),
    }
    partial_passed = all(value["passed"] for value in partial.values())
    strong_passed = all(value["passed"] for value in strong.values())
    return {
        "grade": "strong_pass" if strong_passed else "partial_pass" if partial_passed else "failed",
        "partial_passed": partial_passed,
        "strong_passed": strong_passed,
        "partial_gates": partial,
        "strong_gates": strong,
    }


def _parse_variants(raw: Any) -> tuple[QueryVariant, ...]:
    if not raw:
        return DEFAULT_VARIANTS
    variants = []
    for name, values in raw.items():
        variants.append(
            QueryVariant(
                name=str(name),
                architecture=str(values.get("architecture", "factorized")),
                relation_weight=float(values.get("relation_weight", 0.0)),
                entity_weight=float(values.get("entity_weight", 0.0)),
                template_adversary_weight=float(
                    values.get("template_adversary_weight", 0.0)
                ),
            )
        )
    return tuple(variants)


def _ablation_row(
    name: str,
    evaluation: dict[str, Any],
    grade: dict[str, Any],
    *,
    best_step: int,
) -> dict[str, Any]:
    test = evaluation["test"]
    validation = evaluation["validation"]
    return {
        "variant": name,
        "grade": grade["grade"],
        "best_step": best_step,
        "validation_row_1nn": validation["retrieval"]["row_1nn_accuracy"],
        "test_row_1nn": test["retrieval"]["row_1nn_accuracy"],
        "test_centroid_top1": test["retrieval"]["centroid_top1_accuracy"],
        "test_mrr": test["retrieval"]["centroid_mrr"],
        "test_fact_silhouette": test["silhouette"]["fact_id"],
        "test_fact_template_margin": test["geometry"]["fact_over_template_margin"],
        "test_same_fact_cosine": test["geometry"]["same_fact_cross_template_cosine"],
        "test_same_template_cosine": test["geometry"]["different_fact_same_template_cosine"],
        "test_entity_probe": test["linear_probes"]["entity_id"]["accuracy"],
        "test_relation_probe": test["linear_probes"]["relation_id"]["accuracy"],
        "test_answer_probe": test["linear_probes"]["answer_token"]["accuracy"],
        "aux_relation_accuracy": evaluation.get("auxiliary_heads", {}).get(
            "test", {}
        ).get("relation_accuracy", ""),
        "aux_entity_accuracy": evaluation.get("auxiliary_heads", {}).get(
            "test", {}
        ).get("entity_accuracy", ""),
        "template_probe": evaluation["template_leakage_probe"]["accuracy"],
        "partial_passed": grade["partial_passed"],
        "strong_passed": grade["strong_passed"],
    }


def run_stage_c2(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    checkpoint = Path(values.get("checkpoint", "artifacts/checkpoints/main/gram_original.pt"))
    data_dir = Path(values.get("data_dir", "data/stage_b"))
    tokenizer_name = str(values.get("tokenizer", "SimpleStories/SimpleStories-1.25M"))
    extraction = values.get("extraction", {})
    training = values.get("training", {})
    evaluation_config = values.get("evaluation", {})
    selected_layers_one_based = tuple(extraction.get("selected_layers", [4, 6, 8]))
    selected_layer_indices = tuple(int(value) - 1 for value in selected_layers_one_based)
    baseline_layer_index = int(extraction.get("q0_baseline_layer_index", 6))
    extraction_batch_size = int(extraction.get("batch_size", 64))
    dtype_name = str(extraction.get("dtype", "bfloat16"))
    seed = int(training.get("seed", 0))
    variants = _parse_variants(values.get("variants"))
    if {variant.name for variant in variants} != {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("Stage C2 requires exactly Q1, Q2, Q3, and Q4")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name, device)
    model, payload = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)
    tokenizer = _load_tokenizer(tokenizer_name)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint")
    split_rows = {
        split: _read_jsonl(data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    _validate_rows(split_rows, "train")
    core_sha256 = hash_non_auxiliary(payload["model"], expert_index=1)
    extracted = {
        split: extract_canonicalizer_features(
            model,
            tokenizer,
            rows,
            selected_layer_indices=selected_layer_indices,
            baseline_layer_index=baseline_layer_index,
            batch_size=extraction_batch_size,
            device=device,
            description=f"stage-c2:features:{split}",
        )
        for split, rows in split_rows.items()
    }
    feature_cache = output_dir / "canonicalizer_features.pt"
    torch.save(
        {
            "format_version": 1,
            "source_core_sha256": core_sha256,
            "selected_layers_one_based": list(selected_layers_one_based),
            "selected_layer_indices": list(selected_layer_indices),
            "q0_baseline_layer_index": baseline_layer_index,
            "pool_names": list(POOL_NAMES),
            "features": extracted,
            "metadata": split_rows,
        },
        feature_cache,
    )
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    split_features = {split: values["pools"] for split, values in extracted.items()}
    ridge_strength = float(evaluation_config.get("ridge_strength", 0.01))
    template_probe_seed = int(evaluation_config.get("template_probe_seed", 91))

    q0_embeddings = {
        split: values["baseline_last"] for split, values in extracted.items()
    }
    q0_evaluation, q0_predictions = _evaluate_variant(
        "Q0",
        q0_embeddings,
        split_rows,
        ridge_strength=ridge_strength,
        template_probe_seed=template_probe_seed,
    )
    q0_template_accuracy = float(q0_evaluation["template_leakage_probe"]["accuracy"])
    q0_grade = _grade_variant(q0_evaluation, q0_template_accuracy)
    details: dict[str, Any] = {
        "Q0": {
            "definition": "frozen core last-token pooling; no training",
            "best_step": 0,
            "evaluation": q0_evaluation,
            "grade": q0_grade,
        }
    }
    ablation_rows = [
        _ablation_row("Q0", q0_evaluation, q0_grade, best_step=0)
    ]
    prediction_rows = q0_predictions

    for variant in variants:
        trained = train_query_variant(
            variant,
            split_features,
            split_rows,
            output_dir / variant.name,
            selected_layer_indices=selected_layer_indices,
            core_hidden_size=int(model.config.hidden_size),
            core_sha256=core_sha256,
            steps=int(training.get("steps", 1500)),
            batch_facts=int(training.get("batch_facts", 8)),
            templates_per_fact=int(training.get("templates_per_fact", 3)),
            query_dim=int(training.get("query_dim", 128)),
            mlp_hidden_size=int(training.get("mlp_hidden_size", 256)),
            dropout=float(training.get("dropout", 0.1)),
            temperature=float(training.get("temperature", 0.07)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 0.01)),
            warmup_fraction=float(training.get("warmup_fraction", 0.05)),
            evaluation_interval=int(training.get("evaluation_interval", 50)),
            evaluation_batch_size=int(evaluation_config.get("batch_size", 128)),
            seed=seed,
            device=device,
            ridge_strength=ridge_strength,
            template_probe_seed=template_probe_seed,
            q0_template_accuracy=q0_template_accuracy,
        )
        evaluation, predictions = _evaluate_variant(
            variant.name,
            trained["embeddings"],
            split_rows,
            ridge_strength=ridge_strength,
            template_probe_seed=template_probe_seed,
        )
        evaluation["auxiliary_heads"] = {
            split: _auxiliary_head_metrics(
                trained["system"],
                split_features[split],
                split_rows[split],
                trained["labels"],
                batch_size=int(evaluation_config.get("batch_size", 128)),
                device=device,
            )
            for split in ("validation", "test")
        }
        grade = _grade_variant(evaluation, q0_template_accuracy)
        layer_weights = trained["system"].canonicalizer.layer_logits.softmax(dim=-1)
        detail = {
            "variant": asdict(variant),
            "canonicalizer_config": trained["config"],
            "checkpoint": trained["checkpoint"],
            "best_step": trained["best_step"],
            "best_validation_score": trained["best_validation_score"],
            "learned_layer_weights": {
                pool: [float(value) for value in layer_weights[index].detach().cpu()]
                for index, pool in enumerate(POOL_NAMES)
            },
            "evaluation": evaluation,
            "grade": grade,
        }
        details[variant.name] = detail
        (output_dir / variant.name / "evaluation.json").write_text(
            json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ablation_rows.append(
            _ablation_row(
                variant.name,
                evaluation,
                grade,
                best_step=trained["best_step"],
            )
        )
        prediction_rows.extend(predictions)
        trained["system"].to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Variant choice is validation-only; test metrics do not influence selection.
    selectable = [row for row in ablation_rows if row["variant"] != "Q0"]
    selected = max(
        selectable,
        key=lambda row: tuple(details[row["variant"]]["best_validation_score"]),
    )
    selected_grade = details[selected["variant"]]["grade"]
    stage_passed = bool(selected_grade["partial_passed"])
    status = "strong_pass" if selected_grade["strong_passed"] else "partial_pass" if stage_passed else "failed"

    with (output_dir / "stage_c2_ablation.csv").open(
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
        "source_config": str(Path(config_path).resolve()),
        "values": values,
        "selected_layer_indices": list(selected_layer_indices),
        "core_sha256": core_sha256,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "schema_version": STAGE_C2_SCHEMA_VERSION,
        "stage": "C2-query-canonicalization",
        "status": status,
        "stage_passed": stage_passed,
        "memory_attached": False,
        "answer_injection_enabled": False,
        "public_objective_used": False,
        "security_matrix_run": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_core_sha256": core_sha256,
        "core_frozen": True,
        "selected_layers_one_based": list(selected_layers_one_based),
        "pool_names": list(POOL_NAMES),
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "data_sha256": {
            split: _sha256_file(data_dir / f"{split}.jsonl") for split in split_rows
        },
        "selection_protocol": (
            "best checkpoint and variant selected only by validation partial-gate count, "
            "then retrieval/geometry/probe tuple; test metrics never influence selection"
        ),
        "selected_variant": selected["variant"],
        "selected_variant_grade": selected_grade,
        "passed_variants": [
            row["variant"] for row in ablation_rows if row["partial_passed"]
        ],
        "recommended_next_step": (
            "stage_c3_fixed_prototype_private_memory_retrieval"
            if stage_passed
            else "iterate_query_canonicalizer_without_memory"
        ),
        "q0_template_probe_accuracy": q0_template_accuracy,
        "ablation_rows": ablation_rows,
        "details": details,
        "environment": collect_environment(device, torch.float32),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "feature_cache": str(feature_cache.resolve()),
            "ablation_csv": str((output_dir / "stage_c2_ablation.csv").resolve()),
            "retrieval_predictions": str((output_dir / "retrieval_predictions.csv").resolve()),
            "resolved_config": str((output_dir / "resolved_config.json").resolve()),
        },
    }
    summary_path = output_dir / "stage_c2_summary.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary
