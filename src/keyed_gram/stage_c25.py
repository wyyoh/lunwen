"""Stage C2.5：CLINC150/BANKING77 外部 OOS 泛化审计。"""

from __future__ import annotations

import csv
import inspect
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .stage_c23_semantic import (
    build_model_file_manifest,
    encode_semantic_texts,
    load_semantic_encoder,
)
from .stage_c25_data import (
    DatasetRows,
    IntentExample,
    build_partitions,
    load_partition_rows,
    prepare_sources,
)
from .stage_c25_metrics import (
    aggregate_seed_metrics,
    evaluate_external,
)
from .stage_c25_protocol import (
    C25ProtocolError,
    canonical_sha256,
    git_state,
    initial_status,
    load_config,
    mark_phase_completed,
    mark_phase_started,
    output_paths,
    read_json,
    read_phase_state,
    repo_root,
    require_phase_completed,
    resolve_path,
    runtime_source_manifest,
    sha256_file,
    verify_runtime_sources,
    write_json,
)
from .stage_c25_router import (
    IntentDecision,
    PairwiseEvidenceModel,
    assert_external_memory_boundary,
    fit_multiclass_ridge,
    fit_pairwise_evidence,
    forced_argmax,
    max_threshold_route,
    set_valued_route,
)


OPEN_BENCHMARKS = ("clinc150", "banking77_open")
ALL_BENCHMARKS = ("clinc150", "banking77_open", "banking77_closed")
OPEN_VARIANTS = ("R0", "R2", "MAX_THRESHOLD", "R3")
CLOSED_VARIANTS = ("R0", "R2")
AGGREGATE_METRICS = (
    "known_accuracy",
    "known_coverage",
    "accepted_route_accuracy",
    "safe_coverage",
    "wrong_bucket_access_rate",
    "false_memory_access_rate",
    "oos_recall",
    "oos_f1",
    "oos_auroc",
    "oos_aupr",
    "worst_intent_accuracy",
    "intent_macro_accuracy",
    "multi_candidate_set_rate",
    "aurc",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise C25ProtocolError(f"拒绝写入空 CSV：{path.name}")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    temporary.replace(path)


def _verify_frozen_task_contract(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    output = {}
    for name, declaration in values["frozen_task_specific_contract"].items():
        path = resolve_path(config_path, declaration["path"])
        observed = sha256_file(path)
        if observed != declaration["sha256"]:
            raise C25ProtocolError(f"C2.4 task-specific contract 漂移：{name}")
        output[name] = {
            "path": declaration["path"],
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        }
    return output


def _verify_encoder(config_path: str | Path, *, device: str):
    values = load_config(config_path)
    declaration = values["model"]
    encoder = load_semantic_encoder(
        declaration["semantic_encoder"],
        cache_dir=resolve_path(config_path, declaration["cache_dir"]),
        device=device,
    )
    manifest = build_model_file_manifest(encoder.snapshot_path, encoder.spec)
    safetensors = [
        row["sha256"]
        for row in manifest["files"]
        if row["path"] == "model.safetensors"
    ]
    if (
        manifest["model_id"] != declaration["model_id"]
        or manifest["revision"] != declaration["revision"]
        or manifest["manifest_sha256"]
        != declaration["model_manifest_sha256"]
        or safetensors != [declaration["model_safetensors_sha256"]]
    ):
        raise C25ProtocolError("C2.5 encoder ID/revision/SHA-256 漂移")
    return encoder, {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "license": manifest["license"],
        "manifest_sha256": manifest["manifest_sha256"],
        "model_safetensors_sha256": safetensors[0],
    }


def _verify_prepare(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    artifact_dir, runtime_dir = output_paths(config_path)
    require_phase_completed(runtime_dir, "prepare")
    freeze = read_json(
        artifact_dir / "code_freeze_manifest.json",
        label="C2.5 code freeze manifest",
    )
    verify_runtime_sources(config_path, freeze["runtime_sources"])
    source = prepare_sources(config_path)
    partitions = build_partitions(config_path)
    if (
        sha256_file(artifact_dir / "source_manifest.json")
        != freeze["source_manifest_file_sha256"]
        or sha256_file(artifact_dir / "partition_manifest.json")
        != freeze["partition_manifest_file_sha256"]
        or source
        != read_json(
            artifact_dir / "source_manifest.json", label="C2.5 source manifest"
        )
        or partitions
        != read_json(
            artifact_dir / "partition_manifest.json",
            label="C2.5 partition manifest",
        )
    ):
        raise C25ProtocolError("C2.5 source/partition freeze 漂移")
    _verify_frozen_task_contract(config_path)
    return freeze, artifact_dir, runtime_dir


def prepare_external_audit(config_path: str | Path) -> dict[str, Any]:
    """下载并冻结官方源、intent partitions、代码和 C2.4 contract。"""

    values = load_config(config_path)
    artifact_dir, runtime_dir = output_paths(config_path)
    if artifact_dir.exists() or runtime_dir.exists():
        raise C25ProtocolError("C2.5 输出目录已存在，拒绝覆盖 prepare")
    git = git_state(config_path)
    if git["tracked_dirty"]:
        raise C25ProtocolError("C2.5 prepare 要求 tracked worktree 干净")
    source = prepare_sources(config_path)
    partitions = build_partitions(config_path)
    sources = runtime_source_manifest(config_path)
    contract = _verify_frozen_task_contract(config_path)
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    mark_phase_started(runtime_dir, "prepare")
    source_path = artifact_dir / "source_manifest.json"
    partition_path = artifact_dir / "partition_manifest.json"
    status_path = artifact_dir / "protocol_status.json"
    freeze_path = artifact_dir / "code_freeze_manifest.json"
    write_json(source_path, source)
    write_json(partition_path, partitions)
    write_json(status_path, initial_status())
    freeze = {
        "schema_version": 1,
        "stage": "C2.5-code-and-data-freeze",
        "git": git,
        "config_sha256": sha256_file(config_path),
        "runtime_sources": sources,
        "source_manifest_file_sha256": sha256_file(source_path),
        "partition_manifest_file_sha256": sha256_file(partition_path),
        "frozen_task_specific_contract": contract,
        "selection_protocol": values["selection"],
        "test_predictions_executed": False,
        "test_used_for_selection": False,
        "synthetic_oos_used": False,
        "labels_modified": False,
    }
    write_json(freeze_path, freeze)
    mark_phase_completed(
        runtime_dir,
        "prepare",
        outputs=(source_path, partition_path, freeze_path, status_path),
    )
    return {
        "status": "prepared_and_frozen",
        "git_commit": git["commit"],
        "seed_count": len(values["protocol"]["seeds"]),
        "source_manifest": str(source_path),
        "partition_manifest": str(partition_path),
        "code_freeze_manifest": str(freeze_path),
        "test_predictions_executed": False,
    }


def _partitions_by_seed(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("partitions")
    if not isinstance(rows, list) or not rows:
        raise C25ProtocolError("C2.5 partition manifest 缺少 partitions")
    return [dict(row) for row in rows]


def _unique_texts(
    datasets: Mapping[tuple[int, str], DatasetRows],
    *,
    include_test: bool,
) -> list[str]:
    values = set()
    for rows in datasets.values():
        values.update(row.text for row in rows.fit)
        values.update(row.text for row in rows.calibration)
        if include_test:
            values.update(row.text for row in rows.test)
    return sorted(values)


def _encode_cached(
    encoder: Any,
    texts: Sequence[str],
    *,
    path: Path,
    batch_size: int,
    e5_input_type: str,
) -> Tensor:
    expected_key = canonical_sha256(
        {
            "texts": list(texts),
            "model_id": encoder.spec.model_id,
            "revision": encoder.spec.revision,
            "e5_input_type": e5_input_type,
        }
    )
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path, label="C2.5 embedding cache metadata")
        matrix = torch.load(path, map_location="cpu", weights_only=True)
        if (
            metadata.get("cache_key") != expected_key
            or not isinstance(matrix, Tensor)
            or matrix.shape != (len(texts), encoder.spec.dimension)
            or not bool(torch.isfinite(matrix).all())
        ):
            raise C25ProtocolError("C2.5 embedding cache 漂移")
        return matrix.float()
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = encode_semantic_texts(
        encoder,
        texts,
        batch_size=batch_size,
        e5_input_type=e5_input_type,
    )
    torch.save(matrix, path)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "cache_key": expected_key,
            "row_count": len(texts),
            "dimension": encoder.spec.dimension,
            "committed_to_repository": False,
        },
    )
    return matrix


def _embedding_lookup(texts: Sequence[str], matrix: Tensor) -> dict[str, Tensor]:
    return {text: matrix[index] for index, text in enumerate(texts)}


def _matrix(rows: Sequence[IntentExample], lookup: Mapping[str, Tensor]) -> Tensor:
    return torch.stack([lookup[row.text] for row in rows])


def _intent_text(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("?", "").split())


def _known_accuracy(
    rows: Sequence[IntentExample], decisions: Sequence[IntentDecision]
) -> float:
    indices = [
        index for index, row in enumerate(rows) if row.sample_type == "known"
    ]
    return sum(
        decisions[index].predicted_intent == rows[index].label for index in indices
    ) / len(indices)


def _selection_metrics(
    rows: Sequence[IntentExample],
    scores: Tensor,
    intents: Sequence[str],
    decisions: Sequence[IntentDecision],
) -> dict[str, Any]:
    metrics, _ = evaluate_external(rows, scores, intents, decisions)
    return metrics


def _select_threshold(
    values: Mapping[str, Any],
    rows: Sequence[IntentExample],
    scores: Tensor,
    intents: Sequence[str],
    *,
    variant: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    declaration = values["selection"]
    gates = declaration["selective_selection"]
    candidates = []
    margins = (
        [0.0]
        if variant == "MAX_THRESHOLD"
        else [float(value) for value in declaration["minimum_margin_grid"]]
    )
    for threshold, margin in itertools.product(
        declaration["global_threshold_grid"], margins
    ):
        if variant == "MAX_THRESHOLD":
            decisions = max_threshold_route(
                scores, intents, threshold=float(threshold)
            )
        else:
            decisions = set_valued_route(
                scores,
                intents,
                threshold=float(threshold),
                minimum_margin=float(margin),
            )
        metrics = _selection_metrics(rows, scores, intents, decisions)
        gate_values = (
            metrics["accepted_route_accuracy"]
            >= float(gates["minimum_accepted_route_accuracy"]),
            metrics["known_coverage"]
            >= float(gates["minimum_known_coverage"]),
            metrics["safe_coverage"]
            >= float(gates["minimum_safe_coverage"]),
        )
        objective = (
            int(all(gate_values)),
            sum(gate_values),
            -metrics["false_memory_access_rate"],
            metrics["safe_coverage"],
            metrics["accepted_route_accuracy"],
            metrics["known_coverage"],
            -metrics["wrong_bucket_access_rate"],
            -metrics["multi_candidate_set_rate"],
            -float(threshold),
            -float(margin),
        )
        candidates.append(
            {
                "variant": variant,
                "threshold": float(threshold),
                "minimum_margin": float(margin),
                "objective": objective,
                "metrics": metrics,
            }
        )
    selected = max(candidates, key=lambda row: row["objective"])
    rows_out = [
        {
            "stage": "selective",
            "variant": variant,
            "selected": candidate is selected,
            "threshold": candidate["threshold"],
            "minimum_margin": candidate["minimum_margin"],
            **{
                key: value
                for key, value in candidate["metrics"].items()
                if isinstance(value, (int, float))
            },
        }
        for candidate in candidates
    ]
    return {
        "threshold": selected["threshold"],
        "minimum_margin": selected["minimum_margin"],
        "calibration_metrics": selected["metrics"],
    }, rows_out


def _fit_selected_models(
    values: Mapping[str, Any],
    rows: DatasetRows,
    lookup: Mapping[str, Tensor],
    label_lookup: Mapping[str, Tensor],
    *,
    selected: Mapping[str, Any] | None,
) -> tuple[Any, PairwiseEvidenceModel, dict[str, Any], list[dict[str, Any]]]:
    fit_embeddings = _matrix(rows.fit, lookup)
    calibration_embeddings = _matrix(rows.calibration, lookup)
    labels = [str(row.label) for row in rows.fit]
    intents = tuple(sorted(set(labels)))
    intent_label_embeddings = torch.stack(
        [label_lookup[_intent_text(intent)] for intent in intents]
    )
    selection_rows = []

    r0_candidates = []
    strengths = (
        [float(selected["r0_ridge_strength"])]
        if selected is not None
        else [float(value) for value in values["selection"]["r0_ridge_strength_grid"]]
    )
    for strength in strengths:
        model = fit_multiclass_ridge(
            fit_embeddings,
            labels,
            ridge_strength=strength,
            classes=intents,
        )
        scores = model.score(calibration_embeddings)
        decisions = forced_argmax(scores, intents)
        accuracy = _known_accuracy(rows.calibration, decisions)
        r0_candidates.append((accuracy, -strength, model, scores))
        selection_rows.append(
            {
                "stage": "R0",
                "ridge_strength": strength,
                "known_accuracy": accuracy,
            }
        )
    selected_r0 = max(r0_candidates, key=lambda row: row[:2])
    _, selected_r0_negative_strength, r0_model, r0_scores = selected_r0

    r2_candidates = []
    strengths = (
        [float(selected["r2_ridge_strength"])]
        if selected is not None
        else [float(value) for value in values["selection"]["r2_ridge_strength_grid"]]
    )
    for strength in strengths:
        model = fit_pairwise_evidence(
            fit_embeddings,
            labels,
            intents,
            intent_label_embeddings,
            prototype_count=int(values["model"]["prototype_count_per_intent"]),
            top_k=int(values["model"]["prototype_top_k"]),
            negative_pairs_per_positive=int(
                values["model"]["negative_pairs_per_positive"]
            ),
            ridge_strength=strength,
        )
        scores = model.score(calibration_embeddings)
        decisions = forced_argmax(scores, intents)
        metrics = _selection_metrics(
            rows.calibration, scores, intents, decisions
        )
        r2_candidates.append(
            (
                metrics["known_accuracy"],
                metrics["oos_auroc"],
                -strength,
                model,
                scores,
                metrics,
            )
        )
        selection_rows.append(
            {
                "stage": "R2",
                "ridge_strength": strength,
                "known_accuracy": metrics["known_accuracy"],
                "oos_auroc": metrics["oos_auroc"],
            }
        )
    (
        _,
        _,
        _,
        r2_model,
        r2_scores,
        r2_metrics,
    ) = max(r2_candidates, key=lambda row: row[:3])
    parameters = {
        "intents": list(intents),
        "intent_set_sha256": canonical_sha256(list(intents)),
        "r0_ridge_strength": -float(selected_r0_negative_strength),
        "r0_state_sha256": r0_model.state_sha256(),
        "r2_ridge_strength": float(
            next(
                -item[2]
                for item in r2_candidates
                if item[3] is r2_model
            )
        ),
        "r2_state_sha256": r2_model.state_sha256(),
        "r0_calibration_known_accuracy": _known_accuracy(
            rows.calibration, forced_argmax(r0_scores, intents)
        ),
        "r2_calibration_metrics": r2_metrics,
    }
    if selected is not None:
        if (
            parameters["r0_state_sha256"] != selected["r0_state_sha256"]
            or parameters["r2_state_sha256"] != selected["r2_state_sha256"]
            or parameters["intent_set_sha256"] != selected["intent_set_sha256"]
        ):
            raise C25ProtocolError("C2.5 audit refit model state 与 freeze 不一致")
    return r0_model, r2_model, parameters, selection_rows


def _load_calibration_datasets(
    config_path: str | Path, partitions: Sequence[Mapping[str, Any]]
) -> dict[tuple[int, str], DatasetRows]:
    return {
        (int(partition["seed"]), benchmark): load_partition_rows(
            config_path, partition, benchmark, include_test=False
        )
        for partition in partitions
        for benchmark in ALL_BENCHMARKS
    }


def calibrate_external(
    config_path: str | Path, *, device: str = "cpu"
) -> dict[str, Any]:
    """仅使用官方 train/validation 拟合与选择；不加载 test rows。"""

    freeze, artifact_dir, runtime_dir = _verify_prepare(config_path)
    if any(
        (artifact_dir / name).exists()
        for name in (
            "calibration_results.json",
            "router_freeze_manifest.json",
            "calibration_selection.csv",
        )
    ):
        raise C25ProtocolError("C2.5 calibration 产物已存在，拒绝覆盖")
    mark_phase_started(runtime_dir, "calibration")
    values = load_config(config_path)
    partition_manifest = read_json(
        artifact_dir / "partition_manifest.json",
        label="C2.5 partition manifest",
    )
    partitions = _partitions_by_seed(partition_manifest)
    datasets = _load_calibration_datasets(config_path, partitions)
    encoder, model_manifest = _verify_encoder(config_path, device=device)
    texts = _unique_texts(datasets, include_test=False)
    matrix = _encode_cached(
        encoder,
        texts,
        path=runtime_dir / "calibration_query_embeddings.pt",
        batch_size=int(values["model"]["batch_size"]),
        e5_input_type="query",
    )
    lookup = _embedding_lookup(texts, matrix)
    intent_texts = sorted(
        {
            _intent_text(str(row.label))
            for rows in datasets.values()
            for row in rows.fit
        }
    )
    label_matrix = _encode_cached(
        encoder,
        intent_texts,
        path=runtime_dir / "intent_label_embeddings.pt",
        batch_size=int(values["model"]["batch_size"]),
        e5_input_type="passage",
    )
    label_lookup = _embedding_lookup(intent_texts, label_matrix)

    results, selection_rows = [], []
    for partition in partitions:
        seed = int(partition["seed"])
        for benchmark in ALL_BENCHMARKS:
            rows = datasets[(seed, benchmark)]
            r0, r2, parameters, candidate_rows = _fit_selected_models(
                values, rows, lookup, label_lookup, selected=None
            )
            for row in candidate_rows:
                selection_rows.append(
                    {"seed": seed, "benchmark": benchmark, **row}
                )
            calibration_embeddings = _matrix(rows.calibration, lookup)
            intents = tuple(parameters["intents"])
            r0_scores = r0.score(calibration_embeddings)
            r2_scores = r2.score(calibration_embeddings)
            calibration_metrics = {
                "R0": _selection_metrics(
                    rows.calibration,
                    r0_scores,
                    intents,
                    forced_argmax(r0_scores, intents),
                ),
                "R2": _selection_metrics(
                    rows.calibration,
                    r2_scores,
                    intents,
                    forced_argmax(r2_scores, intents),
                ),
            }
            if benchmark in OPEN_BENCHMARKS:
                max_parameters, max_rows = _select_threshold(
                    values,
                    rows.calibration,
                    r2_scores,
                    intents,
                    variant="MAX_THRESHOLD",
                )
                r3_parameters, r3_rows = _select_threshold(
                    values,
                    rows.calibration,
                    r2_scores,
                    intents,
                    variant="R3",
                )
                for row in (*max_rows, *r3_rows):
                    selection_rows.append(
                        {"seed": seed, "benchmark": benchmark, **row}
                    )
                parameters["max_threshold"] = max_parameters
                parameters["r3"] = r3_parameters
                calibration_metrics["MAX_THRESHOLD"] = max_parameters[
                    "calibration_metrics"
                ]
                calibration_metrics["R3"] = r3_parameters[
                    "calibration_metrics"
                ]
            results.append(
                {
                    "seed": seed,
                    "benchmark": benchmark,
                    "parameters": parameters,
                    "calibration_metrics": calibration_metrics,
                    "fit_row_count": len(rows.fit),
                    "calibration_row_count": len(rows.calibration),
                }
            )
    calibration = {
        "schema_version": 1,
        "stage": "C2.5-external-calibration",
        "selection_split": "official_train_and_validation_only",
        "test_rows_loaded": False,
        "test_labels_or_predictions_used_for_selection": False,
        "synthetic_oos_used": False,
        "labels_modified": False,
        "model": model_manifest,
        "partitions": results,
    }
    calibration_path = artifact_dir / "calibration_results.json"
    selection_path = artifact_dir / "calibration_selection.csv"
    router_path = artifact_dir / "router_freeze_manifest.json"
    write_json(calibration_path, calibration)
    _write_csv(selection_path, selection_rows)
    router_freeze = {
        "schema_version": 1,
        "stage": "C2.5-router-freeze-before-test",
        "code_freeze_manifest_sha256": sha256_file(
            artifact_dir / "code_freeze_manifest.json"
        ),
        "calibration_results_sha256": sha256_file(calibration_path),
        "calibration_selection_sha256": sha256_file(selection_path),
        "model": model_manifest,
        "partition_count": len(results),
        "seed_count": len(partitions),
        "variants": list(values["selection"]["variants"]),
        "r4_deferred": True,
        "test_rows_loaded": False,
        "test_predictions_executed": False,
        "test_used_for_selection": False,
    }
    write_json(router_path, router_freeze)
    mark_phase_completed(
        runtime_dir,
        "calibration",
        outputs=(calibration_path, selection_path, router_path),
    )
    return {
        "status": "calibrated_and_router_frozen",
        "partition_count": len(results),
        "seed_count": len(partitions),
        "test_rows_loaded": False,
        "router_freeze_manifest": str(router_path),
        "code_freeze_commit": freeze["git"]["commit"],
    }


def _calibration_index(
    calibration: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["seed"]), str(row["benchmark"])): dict(row)
        for row in calibration["partitions"]
    }


def _decision_bundle(
    rows: Sequence[IntentExample],
    r0_scores: Tensor,
    r2_scores: Tensor,
    intents: Sequence[str],
    parameters: Mapping[str, Any],
    *,
    benchmark: str,
) -> dict[str, tuple[Tensor, list[IntentDecision]]]:
    output = {
        "R0": (r0_scores, forced_argmax(r0_scores, intents)),
        "R2": (r2_scores, forced_argmax(r2_scores, intents)),
    }
    if benchmark in OPEN_BENCHMARKS:
        max_parameters = parameters["max_threshold"]
        r3 = parameters["r3"]
        output["MAX_THRESHOLD"] = (
            r2_scores,
            max_threshold_route(
                r2_scores,
                intents,
                threshold=float(max_parameters["threshold"]),
            ),
        )
        output["R3"] = (
            r2_scores,
            set_valued_route(
                r2_scores,
                intents,
                threshold=float(r3["threshold"]),
                minimum_margin=float(r3["minimum_margin"]),
            ),
        )
    return output


def _flat_metrics(
    seed: int, benchmark: str, variant: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "seed": seed,
        "benchmark": benchmark,
        "variant": variant,
        **{
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
    }


def _error_rows(
    seed: int,
    benchmark: str,
    variant: str,
    rows: Sequence[IntentExample],
    decisions: Sequence[IntentDecision],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = (
            "known_intent" if row.sample_type == "known" else "unknown_kind",
            str(row.label or row.unknown_kind),
        )
        groups[group].append(index)
    output = []
    for (group_type, group), indices in sorted(groups.items()):
        output.append(
            {
                "seed": seed,
                "benchmark": benchmark,
                "variant": variant,
                "group_type": group_type,
                "group": group,
                "row_count": len(indices),
                "accepted_count": sum(
                    decisions[index].status == "accept" for index in indices
                ),
                "correct_bucket_count": sum(
                    rows[index].sample_type == "known"
                    and decisions[index].predicted_intent == rows[index].label
                    for index in indices
                ),
                "wrong_bucket_count": sum(
                    rows[index].sample_type == "known"
                    and decisions[index].status == "accept"
                    and decisions[index].predicted_intent != rows[index].label
                    for index in indices
                ),
                "unknown_reject_count": sum(
                    decisions[index].reason == "unknown" for index in indices
                ),
                "ambiguous_reject_count": sum(
                    decisions[index].reason == "ambiguous" for index in indices
                ),
            }
        )
    return output


def _validate_router_freeze(
    artifact_dir: Path, calibration: Mapping[str, Any]
) -> dict[str, Any]:
    router = read_json(
        artifact_dir / "router_freeze_manifest.json",
        label="C2.5 router freeze manifest",
    )
    if (
        router.get("calibration_results_sha256")
        != sha256_file(artifact_dir / "calibration_results.json")
        or router.get("calibration_selection_sha256")
        != sha256_file(artifact_dir / "calibration_selection.csv")
        or router.get("test_predictions_executed") is not False
        or router.get("test_used_for_selection") is not False
        or calibration.get("test_rows_loaded") is not False
        or calibration.get("test_labels_or_predictions_used_for_selection")
        is not False
    ):
        raise C25ProtocolError("C2.5 router freeze/calibration manifest 漂移")
    return router


def _status_from_results(
    values: Mapping[str, Any], per_seed: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    lookup = {
        (str(row["benchmark"]), int(row["seed"]), str(row["variant"])): row
        for row in per_seed
    }
    seeds = [int(seed) for seed in values["protocol"]["seeds"]]
    gate = values["continuation_gates"]
    pairwise = {}
    for benchmark in ALL_BENCHMARKS:
        deltas = [
            float(lookup[(benchmark, seed, "R2")]["known_accuracy"])
            - float(lookup[(benchmark, seed, "R0")]["known_accuracy"])
            for seed in seeds
        ]
        pairwise[benchmark] = {
            "mean_known_accuracy_delta_r2_minus_r0": sum(deltas) / len(deltas),
            "nonnegative_seed_fraction": sum(value >= 0 for value in deltas)
            / len(deltas),
            "passed": (
                sum(deltas) / len(deltas) >= 0
                and sum(value >= 0 for value in deltas) / len(deltas)
                >= float(gate["majority_seed_fraction"])
            ),
        }
    selective = {}
    for benchmark in OPEN_BENCHMARKS:
        seed_rows = []
        for seed in seeds:
            r0 = lookup[(benchmark, seed, "R0")]
            r3 = lookup[(benchmark, seed, "R3")]
            relative = (
                (
                    float(r0["false_memory_access_rate"])
                    - float(r3["false_memory_access_rate"])
                )
                / float(r0["false_memory_access_rate"])
                if float(r0["false_memory_access_rate"]) > 0
                else 0.0
            )
            passed = (
                relative >= float(gate["relative_fmar_reduction_from_r0"])
                and float(r3["wrong_bucket_access_rate"])
                <= float(r0["wrong_bucket_access_rate"])
                and float(r3["accepted_route_accuracy"])
                >= float(gate["accepted_route_accuracy"])
                and float(r3["known_coverage"]) >= float(gate["known_coverage"])
                and float(r3["safe_coverage"]) >= float(gate["safe_coverage"])
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "relative_fmar_reduction": relative,
                    "passed": passed,
                }
            )
        selective[benchmark] = {
            "seed_pass_fraction": sum(row["passed"] for row in seed_rows)
            / len(seed_rows),
            "mean_relative_fmar_reduction": sum(
                row["relative_fmar_reduction"] for row in seed_rows
            )
            / len(seed_rows),
            "passed": sum(row["passed"] for row in seed_rows) / len(seed_rows)
            >= float(gate["majority_seed_fraction"]),
            "seeds": seed_rows,
        }
    pairwise_validated = all(row["passed"] for row in pairwise.values())
    selective_validated = all(row["passed"] for row in selective.values())
    external_passed = pairwise_validated and selective_validated
    status = initial_status()
    status.update(
        {
            "external_oos_validation_status": (
                "passed" if external_passed else "failed"
            ),
            "external_pairwise_evidence_validated": pairwise_validated,
            "external_selective_abstention_validated": selective_validated,
            "exploratory_c3_allowed": external_passed,
            "test_predictions_executed": True,
            "task_specific_router_ready": False,
            "c3_eligible": False,
        }
    )
    return status, {"pairwise": pairwise, "selective": selective}


def run_external_audit(
    config_path: str | Path, *, device: str = "cpu"
) -> dict[str, Any]:
    """冻结后对 CLINC150/BANKING77 test 各执行一次评分。"""

    _, artifact_dir, runtime_dir = _verify_prepare(config_path)
    require_phase_completed(runtime_dir, "calibration")
    if any(
        (artifact_dir / name).exists()
        for name in (
            "test_results.json",
            "per_seed_metrics.csv",
            "aggregate_metrics.csv",
        )
    ):
        raise C25ProtocolError("C2.5 test audit 产物已存在，拒绝覆盖")
    calibration = read_json(
        artifact_dir / "calibration_results.json",
        label="C2.5 calibration results",
    )
    _validate_router_freeze(artifact_dir, calibration)
    values = load_config(config_path)
    partitions = _partitions_by_seed(
        read_json(
            artifact_dir / "partition_manifest.json",
            label="C2.5 partition manifest",
        )
    )
    calibration_index = _calibration_index(calibration)
    calibration_datasets = _load_calibration_datasets(config_path, partitions)
    encoder, model_manifest = _verify_encoder(config_path, device=device)
    calibration_texts = _unique_texts(
        calibration_datasets, include_test=False
    )
    calibration_matrix = _encode_cached(
        encoder,
        calibration_texts,
        path=runtime_dir / "calibration_query_embeddings.pt",
        batch_size=int(values["model"]["batch_size"]),
        e5_input_type="query",
    )
    calibration_lookup = _embedding_lookup(
        calibration_texts, calibration_matrix
    )
    intent_texts = sorted(
        {
            _intent_text(str(row.label))
            for rows in calibration_datasets.values()
            for row in rows.fit
        }
    )
    label_matrix = _encode_cached(
        encoder,
        intent_texts,
        path=runtime_dir / "intent_label_embeddings.pt",
        batch_size=int(values["model"]["batch_size"]),
        e5_input_type="passage",
    )
    label_lookup = _embedding_lookup(intent_texts, label_matrix)
    fitted = {}
    for partition in partitions:
        seed = int(partition["seed"])
        for benchmark in ALL_BENCHMARKS:
            selected = calibration_index[(seed, benchmark)]["parameters"]
            r0, r2, _, _ = _fit_selected_models(
                values,
                calibration_datasets[(seed, benchmark)],
                calibration_lookup,
                label_lookup,
                selected=selected,
            )
            fitted[(seed, benchmark)] = (r0, r2)

    # test-once 边界：只有全部 freeze/refit hash 验证通过后才标记 started，
    # 随后才允许解析 test rows 和产生任何 test prediction。
    mark_phase_started(runtime_dir, "test_audit")
    test_datasets = {
        (int(partition["seed"]), benchmark): load_partition_rows(
            config_path, partition, benchmark, include_test=True
        )
        for partition in partitions
        for benchmark in ALL_BENCHMARKS
    }
    test_texts = sorted(
        {
            row.text
            for rows in test_datasets.values()
            for row in rows.test
        }
    )
    test_matrix = _encode_cached(
        encoder,
        test_texts,
        path=runtime_dir / "test_query_embeddings.pt",
        batch_size=int(values["model"]["batch_size"]),
        e5_input_type="query",
    )
    test_lookup = _embedding_lookup(test_texts, test_matrix)

    per_seed, error_rows, risk_rows, nested = [], [], [], []
    for partition in partitions:
        seed = int(partition["seed"])
        for benchmark in ALL_BENCHMARKS:
            test_rows = test_datasets[(seed, benchmark)].test
            selected = calibration_index[(seed, benchmark)]["parameters"]
            intents = tuple(selected["intents"])
            embeddings = _matrix(test_rows, test_lookup)
            r0, r2 = fitted[(seed, benchmark)]
            bundles = _decision_bundle(
                test_rows,
                r0.score(embeddings),
                r2.score(embeddings),
                intents,
                selected,
                benchmark=benchmark,
            )
            evaluations = {}
            for variant, (scores, decisions) in bundles.items():
                metrics, curve = evaluate_external(
                    test_rows, scores, intents, decisions
                )
                evaluations[variant] = metrics
                per_seed.append(
                    _flat_metrics(seed, benchmark, variant, metrics)
                )
                error_rows.extend(
                    _error_rows(
                        seed, benchmark, variant, test_rows, decisions
                    )
                )
                risk_rows.extend(
                    {
                        "seed": seed,
                        "benchmark": benchmark,
                        "variant": variant,
                        **point,
                    }
                    for point in curve
                )
            nested.append(
                {
                    "seed": seed,
                    "benchmark": benchmark,
                    "test_row_count": len(test_rows),
                    "evaluations": evaluations,
                }
            )
    aggregate = aggregate_seed_metrics(per_seed, AGGREGATE_METRICS)
    status, gate_results = _status_from_results(values, per_seed)
    max_comparison = {}
    lookup = {
        (str(row["benchmark"]), int(row["seed"]), str(row["variant"])): row
        for row in per_seed
    }
    for benchmark in OPEN_BENCHMARKS:
        deltas = [
            {
                "seed": seed,
                "fmar_r3_minus_max": float(
                    lookup[(benchmark, seed, "R3")][
                        "false_memory_access_rate"
                    ]
                )
                - float(
                    lookup[(benchmark, seed, "MAX_THRESHOLD")][
                        "false_memory_access_rate"
                    ]
                ),
                "safe_coverage_r3_minus_max": float(
                    lookup[(benchmark, seed, "R3")]["safe_coverage"]
                )
                - float(
                    lookup[(benchmark, seed, "MAX_THRESHOLD")]["safe_coverage"]
                ),
            }
            for seed in values["protocol"]["seeds"]
        ]
        max_comparison[benchmark] = {
            "mean_fmar_r3_minus_max": sum(
                row["fmar_r3_minus_max"] for row in deltas
            )
            / len(deltas),
            "mean_safe_coverage_r3_minus_max": sum(
                row["safe_coverage_r3_minus_max"] for row in deltas
            )
            / len(deltas),
            "r3_strictly_better_fmar_seed_fraction": sum(
                row["fmar_r3_minus_max"] < 0 for row in deltas
            )
            / len(deltas),
        }
    results = {
        "schema_version": 1,
        "stage": "C2.5-external-test-once-audit",
        "evaluation_mode": "external_public_benchmark_test_once",
        "model": model_manifest,
        "test_predictions_executed_once": True,
        "test_used_for_selection": False,
        "synthetic_oos_used": False,
        "labels_modified": False,
        "r4_deferred": True,
        "per_partition": nested,
        "continuation_gate_results": gate_results,
        "r3_vs_max_threshold": max_comparison,
        "status": status,
        "limitations": [
            "每个 accepted external intent 仅计为一次假想 bucket access；未训练或访问 private memory",
            "CLINC heldout intents 的 validation 文本作为 OOS calibration，只有 test 文本未参与选择",
            "BANKING77 test-only OOS intents 在 calibration 中完全不可见",
            "外部方法验证不等于 registry/city/access task-specific readiness",
            "未声称部署级安全或分布迁移下的 conformal 保证",
        ],
    }
    result_path = artifact_dir / "test_results.json"
    seed_path = artifact_dir / "per_seed_metrics.csv"
    aggregate_path = artifact_dir / "aggregate_metrics.csv"
    errors_path = artifact_dir / "error_analysis.csv"
    risk_path = artifact_dir / "risk_coverage.csv"
    status_path = artifact_dir / "protocol_status.json"
    write_json(result_path, results)
    _write_csv(seed_path, per_seed)
    _write_csv(aggregate_path, aggregate)
    _write_csv(errors_path, error_rows)
    _write_csv(risk_path, risk_rows)
    write_json(status_path, status)
    mark_phase_completed(
        runtime_dir,
        "test_audit",
        outputs=(
            result_path,
            seed_path,
            aggregate_path,
            errors_path,
            risk_path,
            status_path,
        ),
    )
    return {
        "status": status["external_oos_validation_status"],
        "external_pairwise_evidence_validated": status[
            "external_pairwise_evidence_validated"
        ],
        "external_selective_abstention_validated": status[
            "external_selective_abstention_validated"
        ],
        "exploratory_c3_allowed": status["exploratory_c3_allowed"],
        "task_specific_router_ready": False,
        "c3_eligible": False,
        "test_results": str(result_path),
    }


def finalize_artifacts(config_path: str | Path) -> dict[str, Any]:
    """在报告完成后生成不可覆盖的最终 artifact/report SHA-256 清单。"""

    _, artifact_dir, runtime_dir = _verify_prepare(config_path)
    require_phase_completed(runtime_dir, "test_audit")
    target = artifact_dir / "artifact_sha256_manifest.json"
    if target.exists():
        raise C25ProtocolError("C2.5 artifact manifest 已存在，拒绝覆盖")
    report = repo_root(config_path) / "PHASE_C25_REPORT.md"
    if not report.is_file():
        raise C25ProtocolError("PHASE_C25_REPORT.md 尚未生成")
    files = [
        {
            "path": path.relative_to(repo_root(config_path)).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path != target
    ]
    files.append(
        {
            "path": "PHASE_C25_REPORT.md",
            "size_bytes": report.stat().st_size,
            "sha256": sha256_file(report),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "C2.5-final-artifact-manifest",
        "files": files,
        "private_data_present": False,
        "dataset_files_present": False,
        "model_checkpoint_present": False,
        "test_predictions_executed_once": True,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    write_json(target, payload)
    return {
        "artifact_manifest": str(target),
        "file_count": len(files),
        "manifest_file_sha256": sha256_file(target),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
    }


def run_smoke(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """纯 synthetic tensor smoke；不读取任何官方 train/validation/test。"""

    values = load_config(config_path, smoke=True)
    intents = ("alpha", "beta", "gamma")
    rows = (
        IntentExample("k0", "synthetic-a", "alpha", "known", ""),
        IntentExample("k1", "synthetic-b", "beta", "known", ""),
        IntentExample("u0", "synthetic-oos", None, "unknown", "synthetic"),
    )
    scores = torch.tensor(
        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.2, 0.2, 0.2]],
        dtype=torch.float32,
    )
    parameters, selection = _select_threshold(
        values, rows, scores, intents, variant="R3"
    )
    decisions = set_valued_route(
        scores,
        intents,
        threshold=float(parameters["threshold"]),
        minimum_margin=float(parameters["minimum_margin"]),
    )
    metrics, _ = evaluate_external(rows, scores, intents, decisions)
    assert_external_memory_boundary()
    result = {
        "schema_version": 1,
        "stage": "C2.5-synthetic-smoke",
        "official_source_read": False,
        "official_test_read": False,
        "synthetic_oos_used": True,
        "parameters": parameters,
        "metrics": metrics,
        "selection_candidate_count": len(selection),
        "c3_eligible": False,
    }
    if output_dir is not None:
        destination = resolve_path(config_path, output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "smoke_summary.json", result)
    return result


def assert_calibration_api_has_no_test_argument() -> None:
    for function in (
        _select_threshold,
        _fit_selected_models,
        calibrate_external,
    ):
        for name in inspect.signature(function).parameters:
            if "test" in name.casefold():
                raise AssertionError(
                    f"calibration API {function.__name__} 包含 test 参数"
                )
