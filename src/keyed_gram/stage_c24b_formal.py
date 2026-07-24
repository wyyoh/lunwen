"""Stage C2.4b 冻结后正式 development 与 locked-audit 后端。

该模块仅由 ``stage_c24b.audit_stage_c24b`` 在双人审核、source seal、router
freeze 全部通过后延迟导入。它不参与 calibration，也不接收 confirmation 或
private answer；fact retrieval 只使用 C2.3 已封存的 answer-free metadata/features。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml

from .canonicalizer import load_canonicalizer_checkpoint
from .stage_c21 import _load_feature_cache
from .stage_c23 import (
    _extract_frozen_entity_embeddings,
    validate_answer_free_private_feature_cache,
    validate_oracle_sources,
)
from .stage_c23_audit import sanitize_private_metadata
from .stage_c23_semantic import build_semantic_view_texts, encode_semantic_texts
from .stage_c24 import FactBuckets, build_fact_buckets, load_fixed_s5_head
from .stage_c24_contract import AcceptedRoute, RelationId
from .stage_c24b_benchmark import PublicSplitV2, validate_benchmark_config_v2
from .stage_c24b_calibration import (
    ClassConditionalConformalCalibrator,
    PairwiseRidgeEvidenceHead,
    build_pairwise_relation_features,
)
from .stage_c24b_metrics import compute_readiness, fact_retrieval_metrics
from .stage_c24b_router import (
    RejectedRoute,
    execute_selective_route,
    r3_set_valued_route,
)
from .train import resolve_device


def _main():
    # 延迟导入避免协议编排模块加载本后端时形成循环。
    from . import stage_c24b

    return stage_c24b


def _query_texts(rows: Sequence[Mapping[str, Any]], query_view: str) -> list[str]:
    if query_view != "entity_masked_strip_suffix":
        raise RuntimeError("formal C2.4b query view is not the frozen full question")
    output = []
    for row in rows:
        views = row.get("input_views")
        if isinstance(views, Mapping) and query_view in views:
            output.append(str(views[query_view]))
            continue
        output.extend(build_semantic_view_texts([row], query_view))
    return output


def _model_artifact_seal(
    config_path: str | Path,
    values: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """重算 encoder snapshot 与 R2 公共 head 的完整文件/state seal。"""

    main = _main()
    manifests = {
        name: main.build_model_file_manifest(
            context[f"{name.casefold()}_encoder"].snapshot_path,
            context[f"{name.casefold()}_encoder"].spec,
        )
        for name in ("R0", "R1", "R2")
    }
    artifact_dir = main._resolve_path(
        config_path, values["benchmark"]["artifact_dir"]
    )
    parameters = context["parameters"]
    head_path = artifact_dir / "calibration" / str(
        parameters["selected_r2_head_file"]
    )
    head = PairwiseRidgeEvidenceHead.from_state_dict(
        torch.load(head_path, map_location="cpu", weights_only=True)
    )
    return {
        "encoder_manifests": manifests,
        "r2_head_file": head_path.name,
        "r2_head_file_sha256": main.sha256_file(head_path),
        "r2_head_state_sha256": main.canonical_sha256(head.to_json_dict()),
    }


def _model_provenance(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """提取正式产物所需的 model ID/revision/manifest/weight provenance。"""

    output: dict[str, dict[str, Any]] = {}
    for name, manifest in manifests.items():
        weights = [
            row
            for row in manifest.get("files", [])
            if isinstance(row, Mapping) and row.get("path") == "model.safetensors"
        ]
        if len(weights) != 1:
            raise _main().ProtocolViolation(
                f"formal model manifest lacks one model.safetensors file: {name}"
            )
        output[name] = {
            "model_id": str(manifest["model_id"]),
            "revision": str(manifest["revision"]),
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "model_safetensors_sha256": str(weights[0]["sha256"]),
        }
    return output


def _finite_tensor_sha256(tensor: torch.Tensor) -> str:
    """封存 tensor 的 dtype、shape 与原始连续字节。"""

    value = tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise _main().ProtocolViolation("slot binding tensor is non-finite")
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.numpy().tobytes()).hexdigest()


def _build_locked_slot_binding(
    values: Mapping[str, Any],
    test_embeddings: torch.Tensor,
    test_rows: Sequence[Mapping[str, Any]],
    buckets: FactBuckets,
    *,
    answer_free_cache_sha256: str,
) -> dict[str, Any]:
    """预注册 relation-independent public-entity→answer-free slot binding。

    public entity 只按 ID 排序，private entity 只按 answer-free 名称排序；二者
    一一映射且数量不足即失败。每个 private entity 的 query embedding 对其全部
    development rows 求均值，因而不按 relation/family/score/结果挑选。
    """

    main = _main()
    benchmark = validate_benchmark_config_v2(values["benchmark"])
    public_entities = sorted(
        benchmark.splits[PublicSplitV2.LOCKED_AUDIT].entities
    )
    if test_embeddings.ndim != 2 or len(test_embeddings) != len(test_rows):
        raise main.ProtocolViolation("slot binding rows and embeddings are not aligned")
    private_entities = sorted({str(row["entity"]) for row in test_rows})
    if len(private_entities) < len(public_entities):
        raise main.ProtocolViolation(
            "answer-free development has fewer entities than locked public binding"
        )

    entity_embeddings: dict[str, torch.Tensor] = {}
    fact_ids: dict[str, dict[RelationId, str]] = {}
    mapping_rows: list[dict[str, Any]] = []
    for public_entity_id, private_entity in zip(public_entities, private_entities):
        indices = [
            index
            for index, row in enumerate(test_rows)
            if str(row["entity"]) == private_entity
        ]
        if not indices:
            raise main.ProtocolViolation("slot binding entity has no source rows")
        # 聚合与 relation 无关；同一 public entity 在全部 family/frame 固定使用。
        entity_embeddings[public_entity_id] = F.normalize(
            test_embeddings[indices].float().mean(dim=0), p=2, dim=0
        )
        relation_facts: dict[RelationId, str] = {}
        fact_hashes: dict[str, str] = {}
        for relation in RelationId:
            candidates = {
                str(test_rows[index]["fact_id"])
                for index in indices
                if str(test_rows[index]["attribute"]) == relation.value
            }
            if len(candidates) != 1:
                raise main.ProtocolViolation(
                    "slot binding requires one answer-free fact per entity/relation"
                )
            fact_id = next(iter(candidates))
            if fact_id not in buckets.fact_ids[relation]:
                raise main.ProtocolViolation(
                    "slot binding fact is absent from its fixed relation bucket"
                )
            relation_facts[relation] = fact_id
            fact_hashes[relation.value] = main.canonical_sha256(fact_id)
        fact_ids[public_entity_id] = relation_facts
        mapping_rows.append(
            {
                "public_entity_id": public_entity_id,
                "private_entity_sha256": main.canonical_sha256(private_entity),
                "source_row_indices": indices,
                "source_row_index_sha256": main.canonical_sha256(indices),
                "entity_embedding_sha256": _finite_tensor_sha256(
                    entity_embeddings[public_entity_id]
                ),
                "fact_id_sha256": fact_hashes,
            }
        )
    # 在任何 locked row、router score 或实际 route 之前，离线预计算完整
    # entity × RelationId oracle table。该访问不属于系统查询，不进入 FMAR。
    oracle_table: dict[str, dict[RelationId, str]] = {}
    oracle_table_hash_view: dict[str, dict[str, str]] = {}
    oracle_access_count = 0
    for public_entity_id in public_entities:
        oracle_table[public_entity_id] = {}
        oracle_table_hash_view[public_entity_id] = {}
        for relation in RelationId:
            execution = execute_selective_route(
                entity_embeddings[public_entity_id],
                AcceptedRoute(relation_id=relation),
                buckets.prototypes,
            )
            result = execution.retrieval
            assert result is not None
            if result.cross_relation_candidate_count != 0:
                raise main.ProtocolViolation("offline slot oracle crossed a relation bucket")
            predicted_fact = buckets.fact_ids[relation][result.candidate_index]
            oracle_table[public_entity_id][relation] = predicted_fact
            oracle_table_hash_view[public_entity_id][relation.value] = (
                main.canonical_sha256(predicted_fact)
            )
            oracle_access_count += execution.memory_access_count
    payload = {
        "schema_version": main.SCHEMA_VERSION,
        "scope": "locked_relation_request_x_prefrozen_answer_free_slot_binding_v1",
        "mapping_algorithm": (
            "sorted public entity IDs one-to-one with the first equally sized prefix "
            "of sorted sanitized development entities; no cycling; entity embedding "
            "is the normalized mean over all source rows independent of relation"
        ),
        "selection_inputs_forbidden": [
            "relation_label",
            "phrase_family",
            "router_score",
            "router_prediction",
            "retrieval_result",
        ],
        "public_entity_ids": public_entities,
        "private_entity_count_available": len(private_entities),
        "mapped_entity_count": len(mapping_rows),
        "answer_free_cache_sha256": answer_free_cache_sha256,
        "bucket_prototype_sha256": {
            relation.value: _finite_tensor_sha256(buckets.prototypes[relation])
            for relation in RelationId
        },
        "bucket_fact_id_sha256": {
            relation.value: main.canonical_sha256(buckets.fact_ids[relation])
            for relation in RelationId
        },
        "bucket_row_embedding_sha256": {
            relation.value: _finite_tensor_sha256(buckets.row_embeddings[relation])
            for relation in RelationId
        },
        "bucket_row_fact_id_sha256": {
            relation.value: main.canonical_sha256(
                buckets.row_fact_ids[relation]
            )
            for relation in RelationId
        },
        "mappings": mapping_rows,
        "offline_oracle_table_sha256": main.canonical_sha256(
            oracle_table_hash_view
        ),
        "offline_oracle_memory_access_count": oracle_access_count,
        "offline_oracle_counted_as_system_memory_access": False,
        "route_evidence_split": PublicSplitV2.LOCKED_AUDIT.value,
        "entity_embedding_source": "C2.3 sanitized answer-free development",
        "bucket_source": "C2.3 answer-free train",
        "historical_development_substrate_reused": True,
        "fact_retrieval_independent_locked_test": False,
        "private_answers_loaded": False,
        "development_used_for_router_selection": False,
        "limitations": [
            "audits router-to-typed-bucket-to-slot composition only",
            "does not evaluate entity OOD from the public natural-language entity string",
            "does not establish independent entity-side generalization",
        ],
    }
    manifest = {**payload, "binding_sha256": main.canonical_sha256(payload)}
    return {
        "entity_embeddings": entity_embeddings,
        "fact_ids": fact_ids,
        "buckets": buckets,
        "oracle_table": oracle_table,
        "manifest": manifest,
    }


def _evaluate_locked_slot_retrieval(
    rows: Sequence[Mapping[str, Any]],
    routes: Sequence[AcceptedRoute | RejectedRoute],
    binding: Mapping[str, Any],
    buckets: FactBuckets,
    *,
    variant: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对 locked route 实际执行零个或一个 typed relation bucket 访问。"""

    main = _main()
    if len(rows) != len(routes):
        raise main.ProtocolViolation("locked route/retrieval rows are not aligned")
    manifest = dict(binding["manifest"])
    payload = {key: value for key, value in manifest.items() if key != "binding_sha256"}
    if main.canonical_sha256(payload) != manifest.get("binding_sha256"):
        raise main.ProtocolViolation("locked slot binding manifest SHA-256 is invalid")
    mapping_by_entity = {
        str(row["public_entity_id"]): row for row in manifest["mappings"]
    }
    if set(mapping_by_entity) != set(manifest["public_entity_ids"]):
        raise main.ProtocolViolation("locked slot binding mapping set is invalid")
    for entity_id in manifest["public_entity_ids"]:
        if (
            _finite_tensor_sha256(binding["entity_embeddings"][entity_id])
            != mapping_by_entity[entity_id]["entity_embedding_sha256"]
        ):
            raise main.ProtocolViolation("locked slot binding entity tensor changed")
        for relation in RelationId:
            fact_id = str(binding["fact_ids"][entity_id][relation])
            if main.canonical_sha256(fact_id) != mapping_by_entity[entity_id][
                "fact_id_sha256"
            ][relation.value]:
                raise main.ProtocolViolation("locked slot binding fact ID changed")
    for relation in RelationId:
        if (
            _finite_tensor_sha256(buckets.prototypes[relation])
            != manifest["bucket_prototype_sha256"][relation.value]
            or main.canonical_sha256(buckets.fact_ids[relation])
            != manifest["bucket_fact_id_sha256"][relation.value]
            or _finite_tensor_sha256(buckets.row_embeddings[relation])
            != manifest["bucket_row_embedding_sha256"][relation.value]
            or main.canonical_sha256(buckets.row_fact_ids[relation])
            != manifest["bucket_row_fact_id_sha256"][relation.value]
        ):
            raise main.ProtocolViolation("locked slot binding bucket changed")
    oracle_hash_view = {
        entity_id: {
            relation.value: main.canonical_sha256(
                str(binding["oracle_table"][entity_id][relation])
            )
            for relation in RelationId
        }
        for entity_id in manifest["public_entity_ids"]
    }
    if main.canonical_sha256(oracle_hash_view) != manifest[
        "offline_oracle_table_sha256"
    ]:
        raise main.ProtocolViolation("offline locked slot oracle table changed")
    observed_public_entities = {str(row["entity_id"]) for row in rows}
    if observed_public_entities != set(manifest["public_entity_ids"]):
        raise main.ProtocolViolation(
            "locked row entity set differs from the prefrozen slot binding"
        )

    accepted: list[bool] = []
    fact_hits: list[bool | None] = []
    row_hits: list[bool | None] = []
    reciprocal: list[float | None] = []
    margins: list[float | None] = []
    counts: list[int | None] = []
    cross: list[int | None] = []
    oracle_hits: list[bool] = []
    predictions: list[dict[str, Any]] = []
    all_accepted_cross = 0

    entity_embeddings = binding["entity_embeddings"]
    bound_fact_ids = binding["fact_ids"]
    for row, route in zip(rows, routes):
        entity_id = str(row["entity_id"])
        entity_embedding = entity_embeddings[entity_id]
        sample_type = str(row["sample_type"])
        if sample_type == "known":
            true_relation = RelationId(str(row["relation_id"]))
            true_fact = str(bound_fact_ids[entity_id][true_relation])
            oracle_hits.append(
                str(binding["oracle_table"][entity_id][true_relation]) == true_fact
            )
        else:
            true_relation = None
            true_fact = None

        is_accepted = isinstance(route, AcceptedRoute)
        accepted.append(is_accepted)
        if not is_accepted:
            fact_hits.append(None)
            row_hits.append(None)
            reciprocal.append(None)
            margins.append(None)
            counts.append(None)
            cross.append(None)
            predictions.append(
                {
                    "variant": variant,
                    "split": PublicSplitV2.LOCKED_AUDIT.value,
                    "row_id": str(row["row_id"]),
                    "sample_type": sample_type,
                    "route_status": "reject",
                    "reject_reason": route.reason,
                    "memory_access_count": 0,
                    "relation_bucket_candidate_count": 0,
                    "cross_relation_candidate_count": 0,
                    "binding_sha256": manifest["binding_sha256"],
                }
            )
            continue

        execution = execute_selective_route(
            entity_embedding, route, buckets.prototypes
        )
        result = execution.retrieval
        assert result is not None
        all_accepted_cross += result.cross_relation_candidate_count
        relation = route.relation_id
        predicted_fact = buckets.fact_ids[relation][result.candidate_index]
        predictions.append(
            {
                "variant": variant,
                "split": PublicSplitV2.LOCKED_AUDIT.value,
                "row_id": str(row["row_id"]),
                "sample_type": sample_type,
                "route_status": "accept",
                "relation_id": relation.value,
                "target_fact_id": true_fact,
                "predicted_fact_id": predicted_fact if sample_type == "known" else None,
                "memory_access_count": execution.memory_access_count,
                "relation_bucket_candidate_count": result.candidate_count,
                "cross_relation_candidate_count": result.cross_relation_candidate_count,
                "binding_sha256": manifest["binding_sha256"],
            }
        )
        if sample_type != "known":
            fact_hits.append(None)
            row_hits.append(None)
            reciprocal.append(None)
            margins.append(None)
            counts.append(None)
            cross.append(None)
            continue

        assert true_fact is not None
        centroid = F.normalize(buckets.prototypes[relation], dim=1)
        query = F.normalize(entity_embedding, dim=0)
        similarities = centroid @ query
        if true_fact in buckets.fact_ids[relation]:
            true_index = buckets.fact_ids[relation].index(true_fact)
            ranking = similarities.argsort(descending=True)
            rank = int((ranking == true_index).nonzero()[0].item()) + 1
            other_mask = torch.arange(len(similarities)) != true_index
            margin = (
                float(similarities[true_index] - similarities[other_mask].max())
                if bool(other_mask.any())
                else float(similarities[true_index] + 1.0)
            )
        else:
            rank, margin = 0, float(-1.0 - similarities.max())
        row_matrix = F.normalize(buckets.row_embeddings[relation], dim=1)
        row_index = int((row_matrix @ query).argmax())
        row_prediction = buckets.row_fact_ids[relation][row_index]
        fact_hits.append(predicted_fact == true_fact)
        row_hits.append(row_prediction == true_fact)
        reciprocal.append(1.0 / rank if rank else 0.0)
        margins.append(margin)
        counts.append(result.candidate_count)
        cross.append(result.cross_relation_candidate_count)

    known_count = sum(str(row["sample_type"]) == "known" for row in rows)
    if len(oracle_hits) != known_count or not oracle_hits:
        raise main.ProtocolViolation("locked slot oracle rows are incomplete")
    metrics = fact_retrieval_metrics(
        accepted,
        [str(row["sample_type"]) for row in rows],
        fact_hits,
        row_hits,
        reciprocal,
        margins,
        counts,
        cross,
        oracle_fact_top1=sum(oracle_hits) / len(oracle_hits),
    )
    if all_accepted_cross != 0 or metrics["cross_relation_candidate_count"] != 0:
        raise main.ProtocolViolation("locked retrieval crossed a relation bucket")
    metrics.update(
        {
            "scope": manifest["scope"],
            "evidence_split": PublicSplitV2.LOCKED_AUDIT.value,
            "binding_sha256": manifest["binding_sha256"],
            "all_accepted_cross_relation_candidate_count": all_accepted_cross,
            "development_used_for_router_selection": False,
            "private_answers_loaded": False,
            "limitations": manifest["limitations"],
            "route_evidence_split": manifest["route_evidence_split"],
            "entity_embedding_source": manifest["entity_embedding_source"],
            "bucket_source": manifest["bucket_source"],
            "historical_development_substrate_reused": True,
            "fact_retrieval_independent_locked_test": False,
            "offline_oracle_memory_access_count": manifest[
                "offline_oracle_memory_access_count"
            ],
            "offline_oracle_counted_as_system_memory_access": False,
        }
    )
    return metrics, predictions


def _pair_score_rows(
    head: PairwiseRidgeEvidenceHead,
    embeddings: torch.Tensor,
    definitions: Mapping[RelationId, torch.Tensor],
) -> list[dict[RelationId, float]]:
    matrices = head.score_batch(
        build_pairwise_relation_features(embeddings, definitions)
    )
    return [
        {relation: float(matrices[relation][index]) for relation in RelationId}
        for index in range(len(embeddings))
    ]


def _build_router_context(
    config_path: str | Path,
    values: Mapping[str, Any],
    frozen_payload: Mapping[str, Any],
    *,
    device_name: str | None,
) -> dict[str, Any]:
    main = _main()
    device = resolve_device(device_name)
    batch_size = int(values["run"]["batch_size"])
    query_view = str(values["run"].get("query_view", "entity_masked_strip_suffix"))
    c24_path = main._resolve_path(
        config_path, values["fixed_sources"]["stage_c24_config"]
    )
    c24 = yaml.safe_load(c24_path.read_text(encoding="utf-8"))
    cache_dir = main._resolve_path(config_path, c24["fixed_s5"]["model_cache_dir"])
    parameters = frozen_payload["router_parameters"]

    r0_config = {
        **values["models"]["R0"],
        "model_manifest_sha256": c24["fixed_s5"]["model_manifest_sha256"],
    }
    r0_encoder, r0_manifest = main._load_verified_encoder(
        r0_config, cache_dir=cache_dir, device=device
    )
    r0_head = load_fixed_s5_head(
        main._resolve_path(config_path, values["fixed_sources"]["stage_c23_head"]),
        expected_sha256=str(values["fixed_sources"]["stage_c23_head_sha256"]),
        expected_variant=str(c24["fixed_s5"]["base_encoder_variant"]),
        expected_view=str(c24["fixed_s5"]["view"]),
        expected_classes=c24["fixed_s5"]["classes"],
    )
    r1_config = {
        **values["models"]["R1"],
        "model_manifest_sha256": c24["fixed_g3"]["model_manifest_sha256"],
    }
    r1_encoder, r1_manifest = main._load_verified_encoder(
        r1_config, cache_dir=cache_dir, device=device
    )
    r1_definitions = main._definition_embedding_map(
        r1_encoder, values["benchmark"]["definitions"], batch_size=batch_size
    )

    selected_r2 = parameters["selected_r2"]
    candidate_name = str(selected_r2["candidate"])
    r2_config = values["models"]["R2"]["candidates"][candidate_name]
    r2_encoder, r2_manifest = main._load_verified_encoder(
        r2_config, cache_dir=cache_dir, device=device
    )
    r2_definitions = main._definition_embedding_map(
        r2_encoder, values["benchmark"]["definitions"], batch_size=batch_size
    )
    data_dir = main._resolve_path(config_path, values["benchmark"]["public_data_dir"])
    artifact_dir = main._resolve_path(config_path, values["benchmark"]["artifact_dir"])
    train_rows = main._read_jsonl(
        data_dir / f"{main.PublicSplitV2.TRAIN.value}.jsonl"
    )
    train_rows = main._apply_adjudicated_reviews(
        train_rows, artifact_dir / "public_train_v2_review.csv", train_split=True
    )
    if main.canonical_sha256(train_rows) != parameters["reviewed_train_rows_sha256"]:
        raise main.ProtocolViolation("reviewed public_train_v2 changed after freeze")
    head_path = artifact_dir / "calibration" / str(
        parameters["selected_r2_head_file"]
    )
    if main.sha256_file(head_path) != parameters["selected_r2_head_file_sha256"]:
        raise main.ProtocolViolation("frozen R2 head file SHA-256 mismatch")
    r2_head = PairwiseRidgeEvidenceHead.from_state_dict(
        torch.load(head_path, map_location="cpu", weights_only=True)
    )
    if main.canonical_sha256(r2_head.to_json_dict()) != parameters[
        "selected_r2_head_state_sha256"
    ]:
        raise main.ProtocolViolation("restored R2 head state differs from freeze")
    return {
        "device": device,
        "batch_size": batch_size,
        "query_view": query_view,
        "r0_encoder": r0_encoder,
        "r0_head": r0_head,
        "r1_encoder": r1_encoder,
        "r1_definitions": r1_definitions,
        "r2_encoder": r2_encoder,
        "r2_definitions": r2_definitions,
        "r2_head": r2_head,
        "parameters": parameters,
        "model_manifests": {
            "R0": r0_manifest,
            "R1": r1_manifest,
            "R2": r2_manifest,
        },
    }


def _score_and_route(
    rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]
) -> dict[str, tuple[list[dict[RelationId, float]], list[Any], list[frozenset[RelationId]]]]:
    main = _main()
    texts = _query_texts(rows, str(context["query_view"]))
    r0_embeddings = encode_semantic_texts(
        context["r0_encoder"], texts, batch_size=int(context["batch_size"])
    )
    r0_scores, r0_routes = main._r0_scores_and_routes(
        r0_embeddings, context["r0_head"]
    )
    r0_sets = [
        frozenset({route.relation_id})
        if isinstance(route, AcceptedRoute)
        else frozenset()
        for route in r0_routes
    ]
    r1_embeddings = encode_semantic_texts(
        context["r1_encoder"], texts, batch_size=int(context["batch_size"])
    )
    selected_r1 = context["parameters"]["selected_r1_aggregation"]
    r1_scores = main._r1_score_embeddings(
        r1_embeddings,
        context["r1_definitions"],
        aggregation=str(selected_r1["aggregation"]),
        top_k=None if selected_r1["top_k"] is None else int(selected_r1["top_k"]),
    )
    r1_routes, r1_sets = main._routes_argmax(r1_scores)
    r2_embeddings = encode_semantic_texts(
        context["r2_encoder"],
        texts,
        batch_size=int(context["batch_size"]),
        e5_input_type="query",
    )
    r2_scores = _pair_score_rows(
        context["r2_head"], r2_embeddings, context["r2_definitions"]
    )
    r2_routes, r2_sets = main._routes_argmax(r2_scores)
    evidence = {"R1": r1_scores, "R2": r2_scores}
    selected_r3 = context["parameters"]["selected_r3"]
    r3_scores = evidence[str(selected_r3["evidence_source"])]
    r3_sets = [
        main._r3_candidates_with_margin(
            score,
            float(selected_r3["threshold"]),
            float(selected_r3["minimum_margin"]),
        )
        for score in r3_scores
    ]
    r3_routes = [
        r3_set_valued_route(
            score,
            float(selected_r3["threshold"]),
            minimum_margin=float(selected_r3["minimum_margin"]),
        )
        for score in r3_scores
    ]
    selected_r4 = context["parameters"]["selected_r4"]
    r4_scores = evidence[str(selected_r4["evidence_source"])]
    calibrator = ClassConditionalConformalCalibrator.from_dict(
        selected_r4["calibrator"]
    )
    r4_sets = [calibrator.candidate_set(score) for score in r4_scores]
    r4_routes = [calibrator.route(score) for score in r4_scores]
    return {
        "R0": (r0_scores, r0_routes, r0_sets),
        "R1": (r1_scores, r1_routes, r1_sets),
        "R2": (r2_scores, r2_routes, r2_sets),
        "R3": (r3_scores, r3_routes, r3_sets),
        "R4": (r4_scores, r4_routes, r4_sets),
    }


def _actual_answer_free_retrieval(
    config_path: str | Path,
    values: Mapping[str, Any],
    context: Mapping[str, Any],
    selected_router: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    main = _main()
    cache = _load_feature_cache(
        main._resolve_path(config_path, values["fixed_sources"]["private_feature_cache"])
    )
    validate_answer_free_private_feature_cache(cache)
    c24 = yaml.safe_load(
        main._resolve_path(
            config_path, values["fixed_sources"]["stage_c24_config"]
        ).read_text(encoding="utf-8")
    )
    system, checkpoint = load_canonicalizer_checkpoint(
        main._resolve_path(config_path, values["fixed_sources"]["entity_checkpoint"]),
        device=context["device"],
    )
    validate_oracle_sources(cache, system, checkpoint)
    rows = sanitize_private_metadata(
        cache["metadata"], c24["benchmark"]["private_phrase_views"]
    )
    embeddings = {
        split: _extract_frozen_entity_embeddings(
            system,
            cache["features"][split]["pools"].float(),
            batch_size=int(context["batch_size"]),
            device=context["device"],
        )
        for split in ("train", "test")
    }
    system.to("cpu")
    buckets: FactBuckets = build_fact_buckets(embeddings["train"], rows["train"])
    slot_binding = _build_locked_slot_binding(
        values,
        embeddings["test"],
        rows["test"],
        buckets,
        answer_free_cache_sha256=main.sha256_file(
            main._resolve_path(
                config_path, values["fixed_sources"]["private_feature_cache"]
            )
        ),
    )
    routes = _score_and_route(rows["test"], context)[selected_router][1]
    accepted, fact_hits, row_hits = [], [], []
    reciprocal, margins, counts, cross = [], [], [], []
    oracle_hits: list[bool] = []
    predictions: list[dict[str, Any]] = []
    for index, (row, route) in enumerate(zip(rows["test"], routes)):
        # 同一 answer-free entity embedding、同一固定 bucket 上，以真实离散
        # RelationId 运行 slot-only oracle；不读取 private value/answer。
        oracle_relation = RelationId(str(row["attribute"]))
        oracle_execution = execute_selective_route(
            embeddings["test"][index],
            AcceptedRoute(relation_id=oracle_relation),
            buckets.prototypes,
        )
        oracle_result = oracle_execution.retrieval
        assert oracle_result is not None
        oracle_hits.append(
            buckets.fact_ids[oracle_relation][oracle_result.candidate_index]
            == str(row["fact_id"])
        )
        is_accepted = isinstance(route, AcceptedRoute)
        accepted.append(is_accepted)
        if not is_accepted:
            fact_hits.append(None)
            row_hits.append(None)
            reciprocal.append(None)
            margins.append(None)
            counts.append(None)
            cross.append(None)
            predictions.append(
                {
                    "row_id": f"answer-free-development:{index}",
                    "route_status": "reject",
                    "reject_reason": route.reason,
                    "memory_access_count": 0,
                    "relation_bucket_candidate_count": 0,
                    "cross_relation_candidate_count": 0,
                }
            )
            continue
        execution = execute_selective_route(
            embeddings["test"][index], route, buckets.prototypes
        )
        result = execution.retrieval
        assert result is not None
        relation = route.relation_id
        true_fact = str(row["fact_id"])
        predicted_fact = buckets.fact_ids[relation][result.candidate_index]
        centroid = F.normalize(buckets.prototypes[relation], dim=1)
        query = F.normalize(embeddings["test"][index], dim=0)
        similarities = centroid @ query
        ranking = similarities.argsort(descending=True)
        if true_fact in buckets.fact_ids[relation]:
            true_index = buckets.fact_ids[relation].index(true_fact)
            rank = int((ranking == true_index).nonzero()[0].item()) + 1
            wrong = similarities[
                torch.arange(len(similarities)) != true_index
            ].max()
            margin = float(similarities[true_index] - wrong)
        else:
            rank, margin = 0, float(-1.0 - similarities.max())
        row_matrix = F.normalize(buckets.row_embeddings[relation], dim=1)
        row_index = int((row_matrix @ query).argmax())
        row_prediction = buckets.row_fact_ids[relation][row_index]
        fact_hits.append(predicted_fact == true_fact)
        row_hits.append(row_prediction == true_fact)
        reciprocal.append(1.0 / rank if rank else 0.0)
        margins.append(margin)
        counts.append(result.candidate_count)
        cross.append(result.cross_relation_candidate_count)
        predictions.append(
            {
                "row_id": f"answer-free-development:{index}",
                "route_status": "accept",
                "relation_id": relation.value,
                "fact_id": true_fact,
                "predicted_fact_id": predicted_fact,
                "memory_access_count": execution.memory_access_count,
                "relation_bucket_candidate_count": result.candidate_count,
                "cross_relation_candidate_count": result.cross_relation_candidate_count,
            }
        )
    metrics = fact_retrieval_metrics(
        accepted,
        ["known"] * len(rows["test"]),
        fact_hits,
        row_hits,
        reciprocal,
        margins,
        counts,
        cross,
        oracle_fact_top1=sum(oracle_hits) / len(oracle_hits),
    )
    metrics["scope"] = "fixed_answer_free_cache_development_diagnostic"
    metrics["oracle_definition"] = (
        "same fixed entity embeddings and relation buckets routed by true RelationId"
    )
    metrics["development_used_for_selection"] = False
    metrics["private_answers_loaded"] = False
    return metrics, predictions, slot_binding


def _readiness(
    values: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    discrete_memory_contract_preserved: bool,
) -> dict[str, Any]:
    closed = values["gates"]["closed_set"]
    opened = values["gates"]["open_set"]
    known, access = evaluation["known_routing"], evaluation["access_control"]
    reject, fact = evaluation["reject_quality"], evaluation["fact_retrieval"]
    # public_locked_audit_v2 只审计 relation router；C2.3 answer-free development
    # retrieval 是一次性诊断，不能被拼接进 locked readiness。若未来没有为
    # reviewed locked known rows 预注册真实 slot retrieval，本 gate 必须未评估。
    locked_fact_assessed = all(
        key in fact
        for key in (
            "accepted_fact_top1",
            "all_query_fact_top1",
            "mrr",
            "cross_relation_candidate_count",
        )
    ) and fact.get("evidence_split") == "public_locked_audit_v2"
    closed_ready = locked_fact_assessed and all(
        (
            known["relation_family_macro_accuracy"] >= closed["public_family_macro_accuracy"],
            known["worst_family_accuracy"] >= closed["worst_family_accuracy"],
            known["accepted_route_accuracy"] >= closed["accepted_route_accuracy"],
            known["known_coverage"] >= closed["known_coverage"],
            access["safe_coverage"] >= closed["safe_coverage"],
            access["wrong_bucket_access_rate"] <= closed["wrong_bucket_access_rate"],
            fact["accepted_fact_top1"] >= closed["accepted_fact_top1"],
            fact["all_query_fact_top1"] >= closed["all_query_fact_top1"],
            fact["mrr"] >= closed["mrr"],
            fact["cross_relation_candidate_count"] == 0,
        )
    )
    open_ready = all(
        (
            reject["ambiguous_rejection_rate"] >= opened["ambiguous_rejection_rate"],
            reject["unrelated_rejection_rate"] >= opened["unrelated_rejection_rate"],
            access["ambiguous_false_memory_access_rate"]
            <= opened["ambiguous_false_memory_access_rate"],
            access["unrelated_false_memory_access_rate"]
            <= opened["unrelated_false_memory_access_rate"],
            access["false_memory_access_rate"] <= opened["false_memory_access_rate"],
            reject["worst_reject_family_false_accept_rate"]
            <= opened["worst_reject_family_far"],
            access["singleton_acceptance_precision"]
            >= opened["singleton_acceptance_precision"],
        )
    )
    return compute_readiness(
        closed_set_selective_router_ready=closed_ready,
        open_set_abstention_ready=open_ready,
        discrete_memory_contract_preserved=discrete_memory_contract_preserved,
        public_benchmark_human_reviewed=True,
    )


_ERROR_ANALYSIS_FIELDS = (
    "variant",
    "split",
    "row_id",
    "phrase_family",
    "sample_type",
    "target_relation",
    "route_status",
    "predicted_relation",
    "reject_reason",
    "candidate_set",
    "would_access_memory",
    "memory_access_executed",
    "error_type",
)


def _locked_error_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for raw in prediction_rows:
        row = dict(raw)
        known_error = row["sample_type"] == "known" and (
            row["route_status"] != "accept"
            or row["predicted_relation"] != row["target_relation"]
        )
        false_access = (
            row["sample_type"] != "known" and row["route_status"] == "accept"
        )
        reject_reason_mismatch = (
            row["sample_type"] == "ambiguous"
            and row["route_status"] == "reject"
            and row["reject_reason"] != "ambiguous"
        ) or (
            row["sample_type"] == "unrelated"
            and row["route_status"] == "reject"
            and row["reject_reason"] != "unknown"
        )
        if known_error or false_access or reject_reason_mismatch:
            row["error_type"] = (
                "known_route_error"
                if known_error
                else "false_memory_access"
                if false_access
                else "reject_reason_mismatch"
            )
            errors.append(row)
    return errors


def _formal_ablation_rows(
    evaluations: Mapping[str, Mapping[str, Any]], selected_router: str
) -> list[dict[str, Any]]:
    rows = []
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        evaluation = evaluations[variant]
        known = evaluation["known_routing"]
        reject = evaluation["reject_quality"]
        access = evaluation["access_control"]
        fact = evaluation["fact_retrieval"]
        fact_available = "accepted_fact_top1" in fact
        rows.append(
            {
                "variant": variant,
                "protocol_role": "formal_locked_audit_once",
                "selected_on_calibration": variant == selected_router,
                "known_coverage": known["known_coverage"],
                "accepted_route_accuracy": known["accepted_route_accuracy"],
                "family_macro_accuracy": known["relation_family_macro_accuracy"],
                "worst_family_accuracy": known["worst_family_accuracy"],
                "ambiguous_rejection_rate": reject["ambiguous_rejection_rate"],
                "unrelated_rejection_rate": reject["unrelated_rejection_rate"],
                "false_memory_access_rate": access["false_memory_access_rate"],
                "wrong_bucket_access_rate": access["wrong_bucket_access_rate"],
                "safe_coverage": access["safe_coverage"],
                "accepted_fact_top1": fact["accepted_fact_top1"]
                if fact_available
                else "",
                "all_query_fact_top1": fact["all_query_fact_top1"]
                if fact_available
                else "",
                "mrr": fact["mrr"] if fact_available else "",
                "cross_relation_candidate_count": fact[
                    "cross_relation_candidate_count"
                ]
                if fact_available
                else "",
                "fact_retrieval_scope": fact.get("scope", "not_computed"),
            }
        )
    return rows


def _formal_risk_rows(
    evaluations: Mapping[str, Mapping[str, Any]],
    calibration_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """保存 calibration 选择曲线与 locked 固定决策诊断，绝不重选参数。"""

    rows: list[dict[str, Any]] = []
    for method, key in (("R3", "r3_candidates"), ("R4", "r4_candidates")):
        for candidate in calibration_evidence[key]:
            rows.append(
                {
                    "scope": "public_calibration_v2_parameter_selection",
                    "method": method,
                    "variant": method,
                    "decision_source": "frozen_candidate_grid",
                    "threshold": candidate.get("threshold", ""),
                    "minimum_margin": candidate.get("minimum_margin", ""),
                    "alpha": candidate.get("alpha", ""),
                    "coverage": candidate["known_coverage"],
                    # access-control risk 必须把 ambiguous/unrelated 的误接受也
                    # 放进 accepted 分母；known-only route risk 另列，避免混淆。
                    "risk": 1.0 - candidate["singleton_acceptance_precision"],
                    "accuracy": candidate["singleton_acceptance_precision"],
                    "known_route_risk": 1.0
                    - candidate["accepted_relation_precision"],
                    "known_route_accuracy": candidate[
                        "accepted_relation_precision"
                    ],
                    "safe_coverage": candidate["safe_coverage"],
                    "false_memory_access_rate": candidate[
                        "false_memory_access_rate"
                    ],
                    "ambiguous_rejection_rate": candidate[
                        "ambiguous_rejection_rate"
                    ],
                    "unrelated_rejection_rate": candidate[
                        "unrelated_rejection_rate"
                    ],
                    "used_for_router_selection": True,
                }
            )
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        for point in evaluations[variant]["reject_quality"]["risk_coverage"][
            "curve"
        ]:
            rows.append(
                {
                    "scope": "public_locked_audit_v2_fixed_decision_diagnostic",
                    "method": variant,
                    "variant": variant,
                    "decision_source": "frozen_router_reject_score_order",
                    "threshold": point["threshold"],
                    "minimum_margin": "",
                    "alpha": "",
                    "coverage": point["coverage"],
                    "risk": point["risk"],
                    "accuracy": point["accuracy"],
                    "known_route_risk": "",
                    "known_route_accuracy": "",
                    "safe_coverage": "",
                    "false_memory_access_rate": "",
                    "ambiguous_rejection_rate": "",
                    "unrelated_rejection_rate": "",
                    "used_for_router_selection": False,
                }
            )
    return rows


def _write_formal_root_diagnostics(
    destination: Path,
    evaluations: Mapping[str, Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    selected_router: str,
    frozen_payload: Mapping[str, Any],
    *,
    git: Mapping[str, Any],
    model_provenance: Mapping[str, Any],
    config_sha256: str,
    data_sha256: Mapping[str, Any],
    review_manifest_sha256: str,
    selection_protocol: Mapping[str, Any],
) -> None:
    main = _main()
    parameters = frozen_payload["router_parameters"]
    evidence = parameters["calibration_evidence"]
    selection = evidence["router_selection"]
    selection_rows = [
        {
            "router": variant,
            "selected": variant == selected_router,
            "all_gates_passed": row["all_gates_passed"],
            "gate_count": row["gate_count"],
            **row["metrics"],
        }
        for variant, row in selection["candidates"].items()
    ]
    main._write_csv(destination / "router_selection.csv", selection_rows)
    main._write_json(
        destination / "calibration_results.json",
        {
            "schema_version": main.SCHEMA_VERSION,
            "stage": main.STAGE_NAME,
            "protocol_role": "formal_public_calibration_v2_frozen_evidence",
            "selection_split": main.PublicSplitV2.CALIBRATION.value,
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
            "nominal_conformal_coverage_guarantee_claimed": False,
            "conformal_validity_limitation": (
                "evidence source and alpha were selected on this same calibration "
                "split; standard nominal split-conformal coverage is not claimed"
            ),
            "r1_aggregation_candidates": evidence["r1_aggregation_candidates"],
            "selected_r1_aggregation": parameters["selected_r1_aggregation"],
            "r2_candidates": evidence["r2_candidates"],
            "selected_r2": parameters["selected_r2"],
            "r3_candidates": evidence["r3_candidates"],
            "selected_r3": parameters["selected_r3"],
            "r4_candidates": evidence["r4_candidates"],
            "selected_r4": parameters["selected_r4"],
            "router_selection": selection,
            "frozen_router_selection_sha256": main.canonical_sha256(frozen_payload),
            "git": dict(git),
            "frozen_git_commit": frozen_payload["git_commit"],
            "model_provenance": dict(model_provenance),
            "resolved_config_sha256": config_sha256,
            "data_sha256": dict(data_sha256),
            "review_manifest_sha256": review_manifest_sha256,
            "selection_protocol": dict(selection_protocol),
        },
    )
    main._write_csv(
        destination / "stage_c24b_ablation.csv",
        _formal_ablation_rows(evaluations, selected_router),
    )
    main._write_csv(
        destination / "error_analysis.csv",
        _locked_error_rows(prediction_rows),
        fieldnames=_ERROR_ANALYSIS_FIELDS,
    )
    main._write_csv(
        destination / "risk_coverage.csv",
        _formal_risk_rows(evaluations, evidence),
    )


def prepare_formal_prelocked(
    config_path: str | Path,
    values: Mapping[str, Any],
    frozen_payload: Mapping[str, Any],
    *,
    device_name: str | None,
    destination: Path,
) -> dict[str, Any]:
    """先验证 sealed model，再以不可重试 marker 运行一次 development。"""

    main = _main()
    started = destination / "development_started.json"
    completed = destination / "development_completed.json"
    if started.exists() or completed.exists():
        raise main.ProtocolViolation(
            "formal development has already been started and cannot be rerun"
        )
    context = _build_router_context(
        config_path, values, frozen_payload, device_name=device_name
    )
    model_artifacts_before = _model_artifact_seal(config_path, values, context)
    if model_artifacts_before["encoder_manifests"] != context["model_manifests"]:
        raise _main().ProtocolViolation(
            "encoder snapshot manifests differ immediately after verified load"
        )
    parameters = frozen_payload["router_parameters"]
    if (
        model_artifacts_before["r2_head_file_sha256"]
        != parameters["selected_r2_head_file_sha256"]
        or model_artifacts_before["r2_head_state_sha256"]
        != parameters["selected_r2_head_state_sha256"]
    ):
        raise _main().ProtocolViolation("R2 head artifact differs from frozen seal")

    main.assert_discrete_memory_contract()
    destination.mkdir(parents=True, exist_ok=True)
    marker_base = {
        "schema_version": main.SCHEMA_VERSION,
        "stage": main.STAGE_NAME,
        "frozen_router_selection_sha256": main.canonical_sha256(frozen_payload),
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        **main._formal_provenance(
            config_path,
            values,
            frozen_payload=frozen_payload,
            model_provenance=_model_provenance(context["model_manifests"]),
        ),
    }
    main._write_json(
        started,
        {
            **marker_base,
            "status": "development_started_once",
            "development_executed_once": False,
        },
    )
    selected_router = str(frozen_payload["selected_router"])
    try:
        (
            development_fact,
            retrieval_predictions,
            slot_binding,
        ) = _actual_answer_free_retrieval(config_path, values, context, selected_router)
        main._write_json(
            completed,
            {
                **marker_base,
                "status": "development_completed_once",
                "development_executed_once": True,
                "selected_router": selected_router,
                "fact_retrieval_scope": development_fact["scope"],
                "locked_slot_binding": slot_binding["manifest"],
            },
        )
    except Exception as exc:
        main._write_protocol_incident_once(
            destination,
            violation=f"development one-shot failed after start marker: {exc}",
            invalidated_split=main.PublicSplitV2.LOCKED_AUDIT.value,
            provenance=main._formal_provenance(
                config_path,
                values,
                frozen_payload=frozen_payload,
                model_provenance=_model_provenance(context["model_manifests"]),
            ),
        )
        raise
    return {
        "context": context,
        "development_fact": development_fact,
        "retrieval_predictions": retrieval_predictions,
        "slot_binding": slot_binding,
        "selected_router": selected_router,
        "model_artifacts_before": model_artifacts_before,
    }


def run_formal_locked_backend(
    config_path: str | Path,
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    destination: Path,
    frozen_payload: Mapping[str, Any],
    locked_rows: Sequence[Mapping[str, Any]],
    *,
    prelocked: Mapping[str, Any],
    review_manifest_sha256: str,
    fixed_before: Mapping[str, str],
    cache_binding: Mapping[str, Any],
) -> dict[str, Any]:
    main = _main()
    context = prelocked["context"]
    selected_router = str(prelocked["selected_router"])
    development_fact = prelocked["development_fact"]
    retrieval_predictions = prelocked["retrieval_predictions"]
    route_data = _score_and_route(locked_rows, context)
    git = main._git_state(main._repo_root(config_path))
    evaluations, prediction_rows, score_rows = {}, [], []
    locked_retrieval_predictions: list[dict[str, Any]] = []
    model_provenance = _model_provenance(context["model_manifests"])
    slot_binding = prelocked["slot_binding"]
    audit_data_sha256 = {
        **dict(frozen_payload["data_sha256"]),
        "public_locked_audit_v2": cache_binding["locked_data_sha256"],
        "public_locked_audit_v2_review_csv": cache_binding[
            "locked_review_csv_sha256"
        ],
        "reviewed_public_locked_audit_v2_rows": cache_binding[
            "reviewed_locked_rows_sha256"
        ],
        "public_locked_audit_v2_manifest": cache_binding[
            "locked_split_manifest_sha256"
        ],
    }
    selection_protocol = dict(frozen_payload["selection_protocol"])
    for variant, (scores, routes, sets) in route_data.items():
        evaluation, predictions = main._evaluate_relation_only(
            locked_rows,
            scores,
            routes,
            sets,
            variant=variant,
            split_name=main.PublicSplitV2.LOCKED_AUDIT.value,
            config_sha256=str(metadata["resolved_config_sha256"]),
            git=git,
            model_metadata=model_provenance,
            protocol_role="formal_locked_audit_once",
            evaluation_scope="public_locked_audit_v2_after_freeze",
            data_sha256=audit_data_sha256,
            review_manifest_sha256=review_manifest_sha256,
            selection_protocol=selection_protocol,
        )
        fact_metrics, retrieval_rows = _evaluate_locked_slot_retrieval(
            locked_rows,
            routes,
            slot_binding,
            slot_binding["buckets"],
            variant=variant,
        )
        evaluation["fact_retrieval"] = fact_metrics
        evaluation["fact_retrieval_scope"] = fact_metrics["scope"]
        evaluation["memory_contract"] = {
            "typed_relation_id_only": True,
            "single_bucket_per_accepted_route": True,
            "rejected_route_memory_access_count": 0,
            "cross_relation_candidate_count": fact_metrics[
                "all_accepted_cross_relation_candidate_count"
            ],
            "continuous_relation_reaches_memory": False,
            "confidence_reaches_memory": False,
            "semantic_embedding_reaches_memory": False,
            "raw_text_reaches_memory": False,
            "phrase_family_reaches_memory": False,
            "slot_binding_sha256": fact_metrics["binding_sha256"],
        }
        for prediction, retrieval in zip(predictions, retrieval_rows):
            prediction["memory_access_executed"] = bool(
                retrieval["memory_access_count"]
            )
        locked_retrieval_predictions.extend(retrieval_rows)
        evaluations[variant] = evaluation
        prediction_rows.extend(predictions)
        for row, score, candidates in zip(locked_rows, scores, sets):
            score_rows.append(
                {
                    "variant": variant,
                    "split": main.PublicSplitV2.LOCKED_AUDIT.value,
                    "row_id": row["row_id"],
                    "registry_id_score": score[RelationId.REGISTRY_ID],
                    "city_code_score": score[RelationId.CITY_CODE],
                    "access_code_score": score[RelationId.ACCESS_CODE],
                    "candidate_set": "|".join(sorted(value.value for value in candidates)),
                }
            )
        main._write_json(destination / variant / "evaluation.json", evaluation)
    selected_evaluation = evaluations[selected_router]
    main.assert_discrete_memory_contract()
    readiness = _readiness(
        values,
        selected_evaluation,
        discrete_memory_contract_preserved=True,
    )
    main._write_csv(destination / "route_candidate_scores.csv", score_rows)
    main._write_csv(destination / "route_predictions.csv", prediction_rows)
    main._write_json(
        destination / "retrieval_predictions.json",
        {
            "schema_version": main.SCHEMA_VERSION,
            "stage": main.STAGE_NAME,
            "git": git,
            "model_provenance": model_provenance,
            "resolved_config_sha256": metadata["resolved_config_sha256"],
            "data_sha256": audit_data_sha256,
            "review_manifest_sha256": review_manifest_sha256,
            "selection_protocol": selection_protocol,
            "development_diagnostic": {
                "metrics": development_fact,
                "rows": retrieval_predictions,
                "used_for_router_selection": False,
                "c24b_development_diagnostic_metrics_used_for_formal_readiness": False,
            },
            "public_locked_audit_v2": {
                "metrics_by_variant": {
                    variant: evaluation["fact_retrieval"]
                    for variant, evaluation in evaluations.items()
                },
                "selected_router": selected_router,
                "selected_metrics": selected_evaluation["fact_retrieval"],
                "rows": locked_retrieval_predictions,
                "locked_slot_binding": slot_binding["manifest"],
                "used_for_router_selection": False,
                "selected_metrics_used_for_formal_readiness": True,
                "historical_c23_development_entity_substrate_used_for_slot_readiness": True,
            },
        },
    )
    _write_formal_root_diagnostics(
        destination,
        evaluations,
        prediction_rows,
        selected_router,
        frozen_payload,
        git=git,
        model_provenance=model_provenance,
        config_sha256=str(metadata["resolved_config_sha256"]),
        data_sha256=audit_data_sha256,
        review_manifest_sha256=review_manifest_sha256,
        selection_protocol=selection_protocol,
    )
    fixed_after = main._verify_declared_fixed_sources(
        config_path, values, lightweight_only=False
    )
    if dict(fixed_before) != fixed_after:
        raise main.ProtocolViolation("fixed sources changed during formal locked audit")
    cache_after = main._verify_answer_free_cache_binding(config_path, values)
    baseline_cache = {
        key: cache_binding[key]
        for key in (
            "cache_sha256",
            "source_core_sha256",
            "row_counts",
            "private_answers_loaded",
        )
    }
    if baseline_cache != cache_after:
        raise main.ProtocolViolation(
            "answer-free cache/source_core binding changed during formal locked audit"
        )
    # 闭合 locked JSONL/review 的 TOCTOU：结束时重验 manifest 与四份审核
    # 文件，再从当前同一字节解析并确认它们产生的 overlay 正是被评分对象。
    locked_seal_after = main._verify_split_seal(
        config_path, values, main.PublicSplitV2.LOCKED_AUDIT
    )
    if (
        locked_seal_after["data_sha256"] != cache_binding["locked_data_sha256"]
        or locked_seal_after["manifest_sha256"]
        != cache_binding["locked_split_manifest_sha256"]
    ):
        raise main.ProtocolViolation("locked data/manifest changed during evaluation")
    full_review_path = destination / "review_manifest.json"
    main._verify_review_manifest_current(full_review_path)
    if main.sha256_file(full_review_path) != review_manifest_sha256:
        raise main.ProtocolViolation("full review manifest changed during evaluation")
    data_dir = main._resolve_path(
        config_path, values["benchmark"]["public_data_dir"]
    )
    artifact_dir = main._resolve_path(
        config_path, values["benchmark"]["artifact_dir"]
    )
    locked_path = data_dir / f"{main.PublicSplitV2.LOCKED_AUDIT.value}.jsonl"
    locked_bytes_after = locked_path.read_bytes()
    if hashlib.sha256(locked_bytes_after).hexdigest() != cache_binding[
        "locked_data_sha256"
    ]:
        raise main.ProtocolViolation("locked JSONL bytes changed during evaluation")
    review_path = artifact_dir / "public_locked_audit_v2_review.csv"
    review_bytes_after = review_path.read_bytes()
    if hashlib.sha256(review_bytes_after).hexdigest() != cache_binding[
        "locked_review_csv_sha256"
    ]:
        raise main.ProtocolViolation("locked review CSV changed during evaluation")
    rows_after = main._parse_jsonl_bytes(
        locked_bytes_after, source=str(locked_path)
    )
    reviews_after = main._parse_review_csv_bytes(
        review_bytes_after, source=str(review_path)
    )
    overlaid_after = main._apply_adjudicated_review_rows(
        rows_after, reviews_after, train_split=False
    )
    if (
        main.canonical_sha256(overlaid_after)
        != cache_binding["reviewed_locked_rows_sha256"]
    ):
        raise main.ProtocolViolation("reviewed locked rows changed during evaluation")
    runtime_after = main._runtime_source_manifest(config_path)
    frozen_runtime = frozen_payload["model_manifests"]["runtime_source_manifest"]
    if runtime_after != frozen_runtime:
        raise main.ProtocolViolation(
            "runtime source/config manifest changed during formal locked audit"
        )
    model_artifacts_before = prelocked["model_artifacts_before"]
    model_artifacts_after = _model_artifact_seal(config_path, values, context)
    if model_artifacts_after != model_artifacts_before:
        raise main.ProtocolViolation(
            "encoder snapshot or R2 head artifact changed during formal locked audit"
        )
    git_after = main._git_state(main._repo_root(config_path))
    if git_after["commit"] != frozen_payload["git_commit"]:
        raise main.ProtocolViolation(
            "git HEAD changed after the frozen formal audit started"
        )
    contract_source_sha256 = next(
        row["sha256"]
        for row in runtime_after["files"]
        if row["path"] == "src/keyed_gram/stage_c24_contract.py"
    )
    summary = {
        "schema_version": main.SCHEMA_VERSION,
        "stage": main.STAGE_NAME,
        "status": "formal_locked_audit_completed_once",
        "formal_locked_audit_executed": True,
        "public_benchmark_human_reviewed": True,
        "selected_router": selected_router,
        "selected_evaluation": selected_evaluation,
        "readiness": readiness,
        "readiness_evidence": {
            "route_and_reject_metrics": "public_locked_audit_v2_after_freeze",
            "fact_retrieval_metrics": slot_binding["manifest"]["scope"],
            "locked_slot_binding_sha256": slot_binding["manifest"][
                "binding_sha256"
            ],
            "development_used_for_router_selection": False,
            "c24b_development_diagnostic_metrics_used_for_formal_readiness": False,
            "historical_c23_development_entity_substrate_used_for_slot_readiness": True,
            "locked_audit_used_for_router_selection": False,
        },
        "development_diagnostic": {
            "fact_retrieval": development_fact,
            "used_for_router_selection": False,
            "c24b_development_diagnostic_metrics_used_for_formal_readiness": False,
        },
        "locked_slot_binding": slot_binding["manifest"],
        "ready_to_create_new_confirmation_pool": readiness[
            "ready_to_create_new_confirmation_pool"
        ],
        "c3_eligible": False,
        "git": git,
        "git_after": git_after,
        "git_commit_equal_to_frozen_before_and_after": True,
        "model_provenance": model_provenance,
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "data_sha256": audit_data_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "selection_protocol": selection_protocol,
        "fixed_source_sha256_before": dict(fixed_before),
        "fixed_source_sha256_after": fixed_after,
        "fixed_source_sha256_equal": True,
        "answer_free_cache_binding": dict(cache_binding),
        "answer_free_cache_binding_after": cache_after,
        "locked_data_and_review_unchanged_after_evaluation": True,
        "runtime_source_manifest_before": frozen_runtime,
        "runtime_source_manifest_after": runtime_after,
        "runtime_source_manifest_equal": True,
        "discrete_memory_contract_asserted_before_locked_read": True,
        "discrete_memory_contract_asserted_after_locked_evaluation": True,
        "discrete_memory_contract_source_sha256": contract_source_sha256,
        "model_artifact_seal_before": model_artifacts_before,
        "model_artifact_seal_after": model_artifacts_after,
        "model_artifact_seal_equal": True,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "answer_injection_executed": False,
        "key_attack_executed": False,
        "new_confirmation_pool_created_after_freeze": False,
    }
    main._write_json(destination / "stage_c24b_summary.json", summary)
    main._write_json(
        destination / "protocol_status.json",
        {
            **summary,
            "selected_evaluation": {
                "known_coverage": selected_evaluation["known_routing"]["known_coverage"],
                "false_memory_access_rate": selected_evaluation["access_control"][
                    "false_memory_access_rate"
                ],
            },
        },
    )
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256_manifest.json":
            files.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": main.sha256_file(path),
                }
            )
    artifact_manifest = {
        "schema_version": main.SCHEMA_VERSION,
        "stage": main.STAGE_NAME,
        "git": git,
        "model_provenance": model_provenance,
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "data_sha256": audit_data_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "selection_protocol": selection_protocol,
        "file_count": len(files),
        "files": files,
    }
    artifact_manifest["manifest_payload_sha256"] = main.canonical_sha256(
        artifact_manifest
    )
    main._write_json(destination / "artifact_sha256_manifest.json", artifact_manifest)
    return summary


__all__ = ["prepare_formal_prelocked", "run_formal_locked_backend"]
