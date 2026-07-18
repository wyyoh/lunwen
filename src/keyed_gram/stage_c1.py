from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm.auto import tqdm

from .checkpoint import build_model_from_checkpoint, load_clean_checkpoint
from .keying import hash_non_auxiliary
from .phase_a import _load_tokenizer, _read_jsonl, _sha256_file
from .stage_b import _prompt_batch
from .train import collect_environment, resolve_device, resolve_dtype


STAGE_C1_SCHEMA_VERSION = 1
REPRESENTATION_POINT = "last_prompt_token_pre_mlp_rmsnorm_core_only"
REQUIRED_ROW_FIELDS = {
    "fact_id",
    "entity",
    "attribute",
    "answer",
    "template_id",
    "template_split",
    "prompt",
}


def _normalise(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [examples, hidden]")
    return F.normalize(features.float(), dim=-1)


def _label_codes(labels: Sequence[str]) -> tuple[Tensor, list[str]]:
    classes = sorted(set(labels))
    if not classes:
        raise ValueError("labels cannot be empty")
    lookup = {value: index for index, value in enumerate(classes)}
    return torch.tensor([lookup[value] for value in labels], dtype=torch.long), classes


def cosine_silhouette(features: Tensor, labels: Sequence[str]) -> float:
    """Mean cosine-distance silhouette without a scikit-learn dependency."""

    if len(features) != len(labels):
        raise ValueError("feature and label counts differ")
    codes, classes = _label_codes(labels)
    if len(classes) < 2:
        raise ValueError("silhouette requires at least two classes")
    counts = torch.bincount(codes, minlength=len(classes))
    if bool((counts < 2).any()):
        raise ValueError("every silhouette class needs at least two examples")

    unit = _normalise(features)
    distance = (1.0 - unit @ unit.T).clamp_min(0.0)
    same = codes[:, None].eq(codes[None, :])
    same.fill_diagonal_(False)
    within = (distance * same).sum(dim=1) / same.sum(dim=1)

    between = torch.full_like(within, float("inf"))
    for class_index in range(len(classes)):
        mask = codes.eq(class_index)
        mean_distance = distance[:, mask].mean(dim=1)
        between = torch.minimum(
            between,
            torch.where(codes.eq(class_index), torch.full_like(mean_distance, float("inf")), mean_distance),
        )
    denominator = torch.maximum(within, between).clamp_min(1e-12)
    return float(((between - within) / denominator).mean())


def _prepare_ridge_features(
    train_features: Tensor, evaluation_features: Tensor
) -> tuple[Tensor, Tensor]:
    train = train_features.float()
    evaluation = evaluation_features.float()
    if train.ndim != 2 or evaluation.ndim != 2 or train.size(1) != evaluation.size(1):
        raise ValueError("ridge features must be compatible matrices")
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train = (train - mean) / scale
    evaluation = (evaluation - mean) / scale
    train = torch.cat([train, torch.ones(train.size(0), 1)], dim=1)
    evaluation = torch.cat([evaluation, torch.ones(evaluation.size(0), 1)], dim=1)
    return train, evaluation


def ridge_probe_accuracy(
    train_features: Tensor,
    train_labels: Sequence[str],
    evaluation_features: Tensor,
    evaluation_labels: Sequence[str],
    *,
    ridge_strength: float = 0.01,
) -> dict[str, float | int]:
    """Fit a deterministic one-vs-all linear ridge probe and evaluate it."""

    if ridge_strength <= 0:
        raise ValueError("ridge_strength must be positive")
    if len(train_features) != len(train_labels) or len(evaluation_features) != len(
        evaluation_labels
    ):
        raise ValueError("feature and label counts differ")

    classes = sorted(set(train_labels))
    lookup = {value: index for index, value in enumerate(classes)}
    if any(value not in lookup for value in evaluation_labels):
        raise ValueError("evaluation labels contain classes absent from training")
    train_targets = torch.tensor([lookup[value] for value in train_labels])
    evaluation_targets = torch.tensor([lookup[value] for value in evaluation_labels])
    x_train, x_evaluation = _prepare_ridge_features(
        train_features, evaluation_features
    )
    targets = F.one_hot(train_targets, num_classes=len(classes)).float()
    kernel = x_train @ x_train.T
    regularizer = max(
        float(kernel.diagonal().mean()) * ridge_strength, 1e-6
    )
    kernel.diagonal().add_(regularizer)
    coefficients = torch.linalg.solve(kernel, targets)
    weights = x_train.T @ coefficients
    train_predictions = (x_train @ weights).argmax(dim=1)
    evaluation_predictions = (x_evaluation @ weights).argmax(dim=1)
    return {
        "accuracy": float(evaluation_predictions.eq(evaluation_targets).float().mean()),
        "train_accuracy": float(train_predictions.eq(train_targets).float().mean()),
        "chance_accuracy": 1.0 / len(classes),
        "num_classes": len(classes),
        "regularizer": regularizer,
    }


def _mean_masked(values: Tensor, mask: Tensor) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def representation_geometry(
    database_features: Tensor,
    database_rows: Sequence[dict[str, Any]],
    query_features: Tensor,
    query_rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Measure whether fact identity beats template and lexical hard negatives."""

    database = _normalise(database_features)
    query = _normalise(query_features)
    cross = query @ database.T
    query_fact = [str(row["fact_id"]) for row in query_rows]
    database_fact = [str(row["fact_id"]) for row in database_rows]
    query_entity = [str(row["entity"]) for row in query_rows]
    database_entity = [str(row["entity"]) for row in database_rows]
    query_attribute = [str(row["attribute"]) for row in query_rows]
    database_attribute = [str(row["attribute"]) for row in database_rows]

    same_fact = torch.tensor(
        [[left == right for right in database_fact] for left in query_fact],
        dtype=torch.bool,
    )
    same_entity = torch.tensor(
        [[left == right for right in database_entity] for left in query_entity],
        dtype=torch.bool,
    )
    same_attribute = torch.tensor(
        [[left == right for right in database_attribute] for left in query_attribute],
        dtype=torch.bool,
    )
    different_fact = ~same_fact

    within = query @ query.T
    template = [str(row["template_id"]) for row in query_rows]
    same_template = torch.tensor(
        [[left == right for right in template] for left in template], dtype=torch.bool
    )
    upper = torch.triu(torch.ones_like(same_template), diagonal=1).bool()
    query_same_fact = torch.tensor(
        [[left == right for right in query_fact] for left in query_fact],
        dtype=torch.bool,
    )

    positive = _mean_masked(cross, same_fact)
    template_negative = _mean_masked(
        within, same_template & ~query_same_fact & upper
    )
    entity_hard_negative = _mean_masked(
        cross, same_entity & ~same_attribute & different_fact
    )
    relation_hard_negative = _mean_masked(
        cross, same_attribute & ~same_entity & different_fact
    )
    all_negative = _mean_masked(cross, different_fact)
    return {
        "same_fact_cross_template_cosine": positive,
        "different_fact_same_template_cosine": template_negative,
        "same_entity_different_relation_cosine": entity_hard_negative,
        "same_relation_different_entity_cosine": relation_hard_negative,
        "all_different_fact_cosine": all_negative,
        "fact_over_template_margin": positive - template_negative,
        "fact_over_same_entity_margin": positive - entity_hard_negative,
        "fact_over_same_relation_margin": positive - relation_hard_negative,
        "fact_over_all_negative_margin": positive - all_negative,
    }


def cross_template_retrieval(
    database_features: Tensor,
    database_rows: Sequence[dict[str, Any]],
    query_features: Tensor,
    query_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retrieve facts from train-template representations using disjoint templates."""

    if not database_rows or not query_rows:
        raise ValueError("retrieval inputs cannot be empty")
    database = _normalise(database_features)
    query = _normalise(query_features)
    database_fact_ids = [str(row["fact_id"]) for row in database_rows]
    fact_ids = sorted(set(database_fact_ids))
    if any(str(row["fact_id"]) not in fact_ids for row in query_rows):
        raise ValueError("query contains a fact absent from the retrieval database")

    row_similarity = query @ database.T
    nearest_rows = row_similarity.argmax(dim=1)
    centroid_rows = []
    for fact_id in fact_ids:
        indices = [
            index for index, value in enumerate(database_fact_ids) if value == fact_id
        ]
        centroid_rows.append(database[indices].mean(dim=0))
    centroids = _normalise(torch.stack(centroid_rows))
    centroid_similarity = query @ centroids.T
    rankings = centroid_similarity.argsort(dim=1, descending=True)
    fact_lookup = {fact_id: index for index, fact_id in enumerate(fact_ids)}

    predictions: list[dict[str, Any]] = []
    row_correct: list[bool] = []
    centroid_correct: list[bool] = []
    reciprocal_ranks: list[float] = []
    top5_correct: list[bool] = []
    margins: list[float] = []
    true_similarities: list[float] = []
    wrong_similarities: list[float] = []
    by_template: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(query_rows):
        true_fact = str(row["fact_id"])
        true_index = fact_lookup[true_fact]
        nearest_index = int(nearest_rows[index])
        row_prediction = database_fact_ids[nearest_index]
        centroid_prediction = fact_ids[int(rankings[index, 0])]
        rank = int((rankings[index] == true_index).nonzero(as_tuple=False)[0]) + 1
        true_similarity = float(centroid_similarity[index, true_index])
        wrong_mask = torch.ones(len(fact_ids), dtype=torch.bool)
        wrong_mask[true_index] = False
        best_wrong = float(centroid_similarity[index, wrong_mask].max())
        row_hit = row_prediction == true_fact
        centroid_hit = centroid_prediction == true_fact
        row_correct.append(row_hit)
        centroid_correct.append(centroid_hit)
        reciprocal_ranks.append(1.0 / rank)
        top5_correct.append(rank <= min(5, len(fact_ids)))
        margins.append(true_similarity - best_wrong)
        true_similarities.append(true_similarity)
        wrong_similarities.append(best_wrong)
        by_template[str(row["template_id"])].append(index)
        predictions.append(
            {
                "fact_id": true_fact,
                "entity": str(row["entity"]),
                "attribute": str(row["attribute"]),
                "answer": str(row["answer"]),
                "template_id": str(row["template_id"]),
                "row_1nn_prediction": row_prediction,
                "row_1nn_correct": row_hit,
                "row_1nn_similarity": float(row_similarity[index, nearest_index]),
                "centroid_prediction": centroid_prediction,
                "centroid_correct": centroid_hit,
                "centroid_rank": rank,
                "true_centroid_similarity": true_similarity,
                "best_wrong_centroid_similarity": best_wrong,
                "centroid_margin": true_similarity - best_wrong,
            }
        )

    per_template = {}
    for template_id, indices in sorted(by_template.items()):
        per_template[template_id] = {
            "num_queries": len(indices),
            "row_1nn_accuracy": statistics.fmean(row_correct[index] for index in indices),
            "centroid_top1_accuracy": statistics.fmean(
                centroid_correct[index] for index in indices
            ),
        }
    return (
        {
            "num_database_rows": len(database_rows),
            "num_queries": len(query_rows),
            "num_facts": len(fact_ids),
            "random_fact_accuracy": 1.0 / len(fact_ids),
            "row_1nn_accuracy": statistics.fmean(row_correct),
            "centroid_top1_accuracy": statistics.fmean(centroid_correct),
            "centroid_top5_accuracy": statistics.fmean(top5_correct),
            "centroid_mrr": statistics.fmean(reciprocal_ranks),
            "mean_true_centroid_similarity": statistics.fmean(true_similarities),
            "mean_best_wrong_centroid_similarity": statistics.fmean(wrong_similarities),
            "mean_centroid_margin": statistics.fmean(margins),
            "per_template": per_template,
        },
        predictions,
    )


def _validate_rows(
    split_rows: dict[str, list[dict[str, Any]]], database_split: str
) -> None:
    if database_split not in split_rows:
        raise ValueError("database split is absent")
    database_facts = {str(row["fact_id"]) for row in split_rows[database_split]}
    database_templates = {
        str(row["template_id"]) for row in split_rows[database_split]
    }
    for split, rows in split_rows.items():
        if not rows:
            raise ValueError(f"split {split} is empty")
        for row in rows:
            missing = REQUIRED_ROW_FIELDS.difference(row)
            if missing:
                raise ValueError(f"split {split} row misses fields: {sorted(missing)}")
        if split == database_split:
            continue
        facts = {str(row["fact_id"]) for row in rows}
        templates = {str(row["template_id"]) for row in rows}
        if facts != database_facts:
            raise ValueError(f"split {split} does not contain the database fact set")
        if templates.intersection(database_templates):
            raise ValueError(f"split {split} reuses a database template")


@torch.inference_mode()
def extract_core_representations(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    description: str,
) -> Tensor:
    """Extract exactly the normalized hidden vectors consumed by each aux MLP."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    core_mask = torch.tensor(
        [1] + [0] * model.config.num_aux, device=device, dtype=torch.long
    )
    output = torch.empty(
        len(rows), model.config.num_layers, model.config.hidden_size, dtype=torch.float32
    )
    eos = int(tokenizer.eos_token_id)
    for offset in tqdm(
        range(0, len(rows), batch_size), desc=description, leave=False
    ):
        batch_rows = rows[offset : offset + batch_size]
        input_ids, lengths = _prompt_batch(batch_rows, tokenizer, eos)
        if input_ids.size(1) > model.config.context_length:
            raise ValueError("a diagnostic prompt exceeds model context length")
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        _, _, diagnostics = model(
            input_ids,
            fwd_mask=core_mask,
            bck_mask=core_mask,
            return_diagnostics=True,
        )
        indices = torch.arange(input_ids.size(0), device=device)
        for layer, hidden in enumerate(diagnostics["aux_inputs"]):
            output[offset : offset + len(batch_rows), layer] = hidden[
                indices, lengths - 1
            ].float().cpu()
    return output


def _probe_suite(
    database_features: Tensor,
    database_rows: Sequence[dict[str, Any]],
    evaluation_features: Tensor,
    evaluation_rows: Sequence[dict[str, Any]],
    *,
    ridge_strength: float,
) -> dict[str, dict[str, float | int]]:
    fields = {
        "entity_id": "entity",
        "relation_id": "attribute",
        "answer_token": "answer",
    }
    return {
        name: ridge_probe_accuracy(
            database_features,
            [str(row[field]) for row in database_rows],
            evaluation_features,
            [str(row[field]) for row in evaluation_rows],
            ridge_strength=ridge_strength,
        )
        for name, field in fields.items()
    }


def _split_analysis(
    layer: int,
    database_features: Tensor,
    database_rows: Sequence[dict[str, Any]],
    evaluation_features: Tensor,
    evaluation_rows: Sequence[dict[str, Any]],
    *,
    split: str,
    ridge_strength: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval, predictions = cross_template_retrieval(
        database_features, database_rows, evaluation_features, evaluation_rows
    )
    geometry = representation_geometry(
        database_features, database_rows, evaluation_features, evaluation_rows
    )
    probes = _probe_suite(
        database_features,
        database_rows,
        evaluation_features,
        evaluation_rows,
        ridge_strength=ridge_strength,
    )
    combined = torch.cat([database_features, evaluation_features], dim=0)
    combined_rows = [*database_rows, *evaluation_rows]
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
    for row in predictions:
        row.update({"layer": layer, "query_split": split})
    probe_rows = [
        {"layer": layer, "query_split": split, "probe": name, **values}
        for name, values in probes.items()
    ]
    return (
        {
            "retrieval": retrieval,
            "geometry": geometry,
            "linear_probes": probes,
            "silhouette": silhouette,
        },
        predictions,
        probe_rows,
    )


def _flatten_layer_metrics(layer_result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"layer": layer_result["layer"]}
    for split in ("validation", "test"):
        values = layer_result[split]
        prefix = f"{split}_"
        output.update(
            {
                prefix + "row_1nn_accuracy": values["retrieval"]["row_1nn_accuracy"],
                prefix + "centroid_top1_accuracy": values["retrieval"]["centroid_top1_accuracy"],
                prefix + "centroid_mrr": values["retrieval"]["centroid_mrr"],
                prefix + "fact_cosine": values["geometry"]["same_fact_cross_template_cosine"],
                prefix + "template_negative_cosine": values["geometry"]["different_fact_same_template_cosine"],
                prefix + "fact_over_template_margin": values["geometry"]["fact_over_template_margin"],
                prefix + "fact_silhouette": values["silhouette"]["fact_id"],
                prefix + "entity_probe_accuracy": values["linear_probes"]["entity_id"]["accuracy"],
                prefix + "relation_probe_accuracy": values["linear_probes"]["relation_id"]["accuracy"],
                prefix + "answer_probe_accuracy": values["linear_probes"]["answer_token"]["accuracy"],
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decision(
    layer_results: Sequence[dict[str, Any]],
    *,
    strong_threshold: float,
    partial_threshold: float,
) -> dict[str, Any]:
    if not 0 <= partial_threshold <= strong_threshold <= 1:
        raise ValueError("retrieval thresholds must satisfy 0 <= partial <= strong <= 1")
    best = max(
        layer_results,
        key=lambda row: (
            row["test"]["retrieval"]["row_1nn_accuracy"],
            row["test"]["retrieval"]["centroid_top1_accuracy"],
            row["test"]["geometry"]["fact_over_template_margin"],
        ),
    )
    accuracy = float(best["test"]["retrieval"]["row_1nn_accuracy"])
    if accuracy >= strong_threshold:
        diagnosis = "core_invariant_interface_available"
        next_step = "stage_c2_aux_reader_then_gradient_decoupling"
        rationale = (
            "A frozen-core layer already retrieves facts across unseen templates; "
            "the auxiliary reader/optimization path is the primary remaining bottleneck."
        )
    elif accuracy >= partial_threshold:
        diagnosis = "core_interface_partial"
        next_step = "stage_c2_lightweight_query_adapter_with_asymmetric_public_projection"
        rationale = (
            "The frozen core carries partial cross-template identity but not a stable enough "
            "interface for a raw residual MLP."
        )
    else:
        diagnosis = "query_canonicalizer_required"
        next_step = "koqm_query_canonicalizer_before_pcgrad"
        rationale = (
            "No frozen-core layer meets the cross-template 1-NN threshold; optimizer surgery "
            "alone cannot supply the missing semantic interface."
        )
    return {
        "diagnosis": diagnosis,
        "recommended_next_step": next_step,
        "rationale": rationale,
        "best_layer": int(best["layer"]),
        "best_test_row_1nn_accuracy": accuracy,
        "random_fact_accuracy": 1.0
        / int(best["test"]["retrieval"]["num_facts"]),
        "best_test_centroid_top1_accuracy": float(
            best["test"]["retrieval"]["centroid_top1_accuracy"]
        ),
        "best_test_fact_silhouette": float(best["test"]["silhouette"]["fact_id"]),
        "best_test_fact_over_template_margin": float(
            best["test"]["geometry"]["fact_over_template_margin"]
        ),
        "strong_retrieval_threshold": strong_threshold,
        "partial_retrieval_threshold": partial_threshold,
    }


def run_stage_c1(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the no-training, layerwise frozen-core semantic-interface diagnostic."""

    started = time.perf_counter()
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    checkpoint = Path(values.get("checkpoint", "artifacts/checkpoints/main/gram_original.pt"))
    reference_checkpoint_value = values.get("stage_b_reference_checkpoint")
    data_dir = Path(values.get("data_dir", "data/stage_b"))
    tokenizer_name = str(
        values.get("tokenizer", "SimpleStories/SimpleStories-1.25M")
    )
    database_split = str(values.get("database_split", "train"))
    evaluation_splits = tuple(values.get("evaluation_splits", ["validation", "test"]))
    if evaluation_splits != ("validation", "test"):
        raise ValueError("Stage C1 currently requires validation and test evaluation splits")
    extraction = values.get("extraction", {})
    metrics_config = values.get("metrics", {})
    batch_size = int(extraction.get("batch_size", 64))
    dtype_name = str(extraction.get("dtype", "bfloat16"))
    ridge_strength = float(metrics_config.get("ridge_strength", 0.01))
    strong_threshold = float(metrics_config.get("strong_retrieval_threshold", 0.75))
    partial_threshold = float(metrics_config.get("partial_retrieval_threshold", 0.50))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name, device)
    model, payload = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)
    tokenizer = _load_tokenizer(tokenizer_name)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint")
    split_names = (database_split, *evaluation_splits)
    split_rows = {
        split: _read_jsonl(data_dir / f"{split}.jsonl") for split in split_names
    }
    _validate_rows(split_rows, database_split)

    representations = {
        split: extract_core_representations(
            model,
            tokenizer,
            rows,
            batch_size=batch_size,
            device=device,
            description=f"stage-c1:{split}",
        )
        for split, rows in split_rows.items()
    }
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    layer_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for layer in tqdm(range(model.config.num_layers), desc="stage-c1:metrics"):
        current: dict[str, Any] = {"layer": layer}
        for split in evaluation_splits:
            analysis, predictions, probes = _split_analysis(
                layer,
                representations[database_split][:, layer],
                split_rows[database_split],
                representations[split][:, layer],
                split_rows[split],
                split=split,
                ridge_strength=ridge_strength,
            )
            current[split] = analysis
            prediction_rows.extend(predictions)
            probe_rows.extend(probes)
        layer_results.append(current)

    decision = _decision(
        layer_results,
        strong_threshold=strong_threshold,
        partial_threshold=partial_threshold,
    )
    core_hash = hash_non_auxiliary(payload["model"], expert_index=1)
    reference = None
    if reference_checkpoint_value:
        reference_checkpoint = Path(reference_checkpoint_value)
        reference_payload = load_clean_checkpoint(reference_checkpoint)
        reference_hash = hash_non_auxiliary(reference_payload["model"], expert_index=1)
        reference = {
            "checkpoint": str(reference_checkpoint.resolve()),
            "core_sha256": reference_hash,
            "matches_diagnostic_checkpoint": reference_hash == core_hash,
        }

    metadata_fields = (
        "fact_id",
        "entity",
        "attribute",
        "answer",
        "template_id",
        "template_split",
        "prompt",
    )
    representation_path = output_dir / "representations.pt"
    torch.save(
        {
            "schema_version": STAGE_C1_SCHEMA_VERSION,
            "representation_point": REPRESENTATION_POINT,
            "checkpoint_core_sha256": core_hash,
            "metadata": {
                split: [
                    {field: row[field] for field in metadata_fields} for row in rows
                ]
                for split, rows in split_rows.items()
            },
            "representations": representations,
        },
        representation_path,
    )

    flat_layer_rows = [_flatten_layer_metrics(row) for row in layer_results]
    _write_csv(output_dir / "layer_metrics.csv", flat_layer_rows)
    _write_csv(output_dir / "retrieval_predictions.csv", prediction_rows)
    _write_csv(output_dir / "linear_probes.csv", probe_rows)
    resolved_config = {
        "checkpoint": str(checkpoint.resolve()),
        "stage_b_reference_checkpoint": (
            str(Path(reference_checkpoint_value).resolve())
            if reference_checkpoint_value
            else None
        ),
        "data_dir": str(data_dir.resolve()),
        "tokenizer": tokenizer_name,
        "database_split": database_split,
        "evaluation_splits": list(evaluation_splits),
        "extraction": {"batch_size": batch_size, "dtype": dtype_name},
        "metrics": {
            "ridge_strength": ridge_strength,
            "strong_retrieval_threshold": strong_threshold,
            "partial_retrieval_threshold": partial_threshold,
        },
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "schema_version": STAGE_C1_SCHEMA_VERSION,
        "stage": "C1-frozen-core-semantic-interface-diagnostic",
        "status": "completed",
        "training_performed": False,
        "auxiliary_enabled": False,
        "representation_point": REPRESENTATION_POINT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_core_sha256": core_hash,
        "stage_b_reference": reference,
        "model": payload["model_config"],
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "data_sha256": {
            split: _sha256_file(data_dir / f"{split}.jsonl") for split in split_names
        },
        "environment": collect_environment(device, dtype),
        "decision": decision,
        "layers": layer_results,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "representations": str(representation_path.resolve()),
            "layer_metrics": str((output_dir / "layer_metrics.csv").resolve()),
            "retrieval_predictions": str(
                (output_dir / "retrieval_predictions.csv").resolve()
            ),
            "linear_probes": str((output_dir / "linear_probes.csv").resolve()),
            "resolved_config": str((output_dir / "resolved_config.json").resolve()),
        },
    }
    summary_path = output_dir / "stage_c1_summary.json"
    summary["artifacts"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary
