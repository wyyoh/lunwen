"""Stage C2.4c 的冻结公开编码器与 R0–R2 evidence 实现。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch import Tensor

from .stage_c23_benchmark import build_public_lexical_benchmark
from .stage_c23_semantic import (
    LoadedSemanticEncoder,
    RidgeLinearHead,
    build_model_file_manifest,
    encode_semantic_texts,
    fit_ridge_linear_head,
    load_semantic_encoder,
    semantic_text_sha256,
)
from .stage_c24_contract import RelationId
from .stage_c24b_calibration import (
    PairwiseRidgeEvidenceHead,
    build_pairwise_relation_features,
    fit_pairwise_ridge_head,
)
from .stage_c24b_router import r1_definition_ensemble_scores
from .stage_c24c_protocol import (
    C24CProtocolError,
    canonical_sha256,
    load_config,
    resolve_path,
    sha256_file,
)


R0_CLASS_ORDER = (
    RelationId.ACCESS_CODE,
    RelationId.CITY_CODE,
    RelationId.REGISTRY_ID,
)


@dataclass
class PublicModelContext:
    """仅包含固定公开编码器和 answer-free public-train 拟合 head。"""

    encoders: dict[str, LoadedSemanticEncoder]
    manifests: dict[str, dict[str, Any]]
    definitions: dict[RelationId, tuple[str, ...]]
    definition_embeddings: dict[str, dict[RelationId, Tensor]]
    r0_head: RidgeLinearHead
    r0_head_file_sha256: str
    r0_head_state_sha256: str
    r2_heads: dict[str, PairwiseRidgeEvidenceHead]
    r2_head_state_sha256: dict[str, str]
    train_data_sha256: str
    model_provenance: dict[str, Any]


def _safetensors_sha256(manifest: Mapping[str, Any]) -> str:
    matches = [
        row["sha256"]
        for row in manifest["files"]
        if row["path"] == "model.safetensors"
    ]
    if len(matches) != 1:
        raise C24CProtocolError("encoder snapshot 缺少唯一 model.safetensors")
    return str(matches[0])


def _load_one_encoder(
    name: str,
    declaration: Mapping[str, Any],
    *,
    cache_dir: Path,
    device: str,
) -> tuple[LoadedSemanticEncoder, dict[str, Any]]:
    encoder = load_semantic_encoder(
        str(declaration["semantic_encoder"]),
        cache_dir=cache_dir,
        device=device,
    )
    if (
        encoder.spec.model_id != declaration["model_id"]
        or encoder.spec.revision != declaration["revision"]
    ):
        raise C24CProtocolError(f"{name} model ID/revision 漂移")
    manifest = build_model_file_manifest(encoder.snapshot_path, encoder.spec)
    if (
        manifest["manifest_sha256"] != declaration["model_manifest_sha256"]
        or _safetensors_sha256(manifest)
        != declaration["model_safetensors_sha256"]
    ):
        raise C24CProtocolError(f"{name} model 文件 SHA-256 漂移")
    return encoder, manifest


def load_definitions(config_path: str | Path) -> dict[RelationId, tuple[str, ...]]:
    values = load_config(config_path)
    v21_path = resolve_path(
        config_path, values["frozen_benchmark"]["config"]["path"]
    )
    v21 = yaml.safe_load(v21_path.read_text(encoding="utf-8")) or {}
    raw = v21.get("benchmark", {}).get("definitions")
    if not isinstance(raw, Mapping) or set(raw) != {
        relation.value for relation in RelationId
    }:
        raise C24CProtocolError("v2.1 relation definitions 不完整")
    output: dict[RelationId, tuple[str, ...]] = {}
    for relation in RelationId:
        definitions = raw[relation.value]
        if (
            not isinstance(definitions, list)
            or len(definitions) < 2
            or any(not str(value).strip() for value in definitions)
        ):
            raise C24CProtocolError(f"{relation.value} definitions 非法")
        output[relation] = tuple(str(value).strip() for value in definitions)
    return output


def build_c23_public_rows(
    config_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    values = load_config(config_path)
    source = resolve_path(
        config_path, values["development"]["source_config"]["path"]
    )
    c23 = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    rows, _ = build_public_lexical_benchmark(c23["benchmark"])
    return rows


def _r0_head_payload(head: RidgeLinearHead) -> dict[str, Any]:
    return {
        "format_version": 1,
        "stage": "C2.3-S5-public-ridge-relation-head",
        "base_encoder_variant": "S4",
        "view": "entity_masked_strip_suffix",
        "classes": list(head.classes),
        "feature_mean": head.feature_mean,
        "feature_scale": head.feature_scale,
        "weights": head.weights,
        "regularizer": head.regularizer,
    }


def _r0_head_state_json(head: RidgeLinearHead) -> dict[str, Any]:
    return {
        "format_version": 1,
        "stage": "C2.3-S5-public-ridge-relation-head",
        "base_encoder_variant": "S4",
        "view": "entity_masked_strip_suffix",
        "classes": list(head.classes),
        "feature_mean": head.feature_mean.detach().cpu().tolist(),
        "feature_scale": head.feature_scale.detach().cpu().tolist(),
        "weights": head.weights.detach().cpu().tolist(),
        "regularizer": float(head.regularizer),
    }


def reconstruct_r0_head(
    config_path: str | Path,
    encoder: LoadedSemanticEncoder,
) -> tuple[RidgeLinearHead, str, str]:
    values = load_config(config_path)
    declaration = values["models"]["R0"]
    c23_rows = build_c23_public_rows(config_path)["train"]
    texts = [str(row["input_views"][declaration["view"]]) for row in c23_rows]
    observed_text_sha = semantic_text_sha256(texts)
    if observed_text_sha != declaration["public_train_semantic_text_sha256"]:
        raise C24CProtocolError("R0 C2.3 public-train text seal 漂移")
    embeddings = encode_semantic_texts(
        encoder,
        texts,
        batch_size=int(values["models"]["batch_size"]),
        e5_input_type="query",
    )
    head = fit_ridge_linear_head(
        embeddings,
        [str(row["attribute"]) for row in c23_rows],
        ridge_strength=float(declaration["ridge_strength"]),
        classes=[str(value) for value in declaration["classes"]],
    )
    if tuple(head.classes) != tuple(value.value for value in R0_CLASS_ORDER):
        raise C24CProtocolError("R0 frozen class order 漂移")
    with tempfile.TemporaryDirectory(prefix="c24c-r0-head-") as temporary:
        path = Path(temporary) / "ridge_relation_head.pt"
        torch.save(_r0_head_payload(head), path)
        file_sha = sha256_file(path)
    if file_sha != declaration["frozen_head_sha256"]:
        raise C24CProtocolError(
            "R0 公开数据重建 head 未匹配 C2.4 frozen head SHA-256"
        )
    return head, file_sha, canonical_sha256(_r0_head_state_json(head))


def _head_sha(head: PairwiseRidgeEvidenceHead) -> str:
    return canonical_sha256(head.to_json_dict())


def build_public_model_context(
    config_path: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> PublicModelContext:
    """加载固定模型，并且只用 public_train_v2_1 拟合 R2。"""

    values = load_config(config_path)
    models = values["models"]
    cache_dir = resolve_path(config_path, models["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoders: dict[str, LoadedSemanticEncoder] = {}
    manifests: dict[str, dict[str, Any]] = {}

    encoder_declarations = {
        "bge-small": models["R0"],
        "minilm": models["R1"],
        "e5-base": models["R2"]["candidates"]["e5_base"],
    }
    for key, declaration in encoder_declarations.items():
        encoder, manifest = _load_one_encoder(
            key, declaration, cache_dir=cache_dir, device=device
        )
        encoders[key] = encoder
        manifests[key] = manifest

    definitions = load_definitions(config_path)
    definition_embeddings: dict[str, dict[RelationId, Tensor]] = {}
    for encoder_name in ("minilm", "e5-base"):
        encoder = encoders[encoder_name]
        definition_embeddings[encoder_name] = {}
        for relation in RelationId:
            definition_embeddings[encoder_name][relation] = encode_semantic_texts(
                encoder,
                definitions[relation],
                batch_size=int(models["batch_size"]),
                e5_input_type="passage",
            )

    r0_head, r0_file_sha, r0_state_sha = reconstruct_r0_head(
        config_path, encoders["bge-small"]
    )

    train_texts = [str(row["input_views"]["phrase_only"]) for row in train_rows]
    targets = [str(row["relation_id"]) for row in train_rows]
    if any(str(row["sample_type"]) != "known" for row in train_rows):
        raise C24CProtocolError("R2 public train 只能包含 known rows")
    r2_heads: dict[str, PairwiseRidgeEvidenceHead] = {}
    r2_hashes: dict[str, str] = {}
    for candidate_name, declaration in models["R2"]["candidates"].items():
        encoder_name = str(declaration["semantic_encoder"])
        normalized_name = (
            "e5-base" if encoder_name == "e5-base" else encoder_name
        )
        embeddings = encode_semantic_texts(
            encoders[normalized_name],
            train_texts,
            batch_size=int(models["batch_size"]),
            e5_input_type="query",
        )
        features = build_pairwise_relation_features(
            embeddings, definition_embeddings[normalized_name]
        )
        for strength in values["selection"]["ridge_strength_grid"]:
            key = f"{candidate_name}:{float(strength):g}"
            head = fit_pairwise_ridge_head(
                features, targets, ridge_strength=float(strength)
            )
            r2_heads[key] = head
            r2_hashes[key] = _head_sha(head)

    provenance = {
        name: {
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "license": manifest["license"],
            "manifest_sha256": manifest["manifest_sha256"],
            "model_safetensors_sha256": _safetensors_sha256(manifest),
        }
        for name, manifest in manifests.items()
    }
    return PublicModelContext(
        encoders=encoders,
        manifests=manifests,
        definitions=definitions,
        definition_embeddings=definition_embeddings,
        r0_head=r0_head,
        r0_head_file_sha256=r0_file_sha,
        r0_head_state_sha256=r0_state_sha,
        r2_heads=r2_heads,
        r2_head_state_sha256=r2_hashes,
        train_data_sha256=canonical_sha256(list(train_rows)),
        model_provenance=provenance,
    )


def _rows_text(
    rows: Sequence[Mapping[str, Any]], view: str
) -> list[str]:
    texts = []
    for row in rows:
        views = row.get("input_views")
        if not isinstance(views, Mapping) or view not in views:
            raise C24CProtocolError(f"evaluation row 缺少 {view}")
        text = str(views[view]).strip()
        if not text:
            raise C24CProtocolError("evaluation text 为空")
        texts.append(text)
    return texts


def _r0_scores(
    head: RidgeLinearHead, embeddings: Tensor
) -> list[dict[RelationId, float]]:
    logits = head.logits(embeddings)
    lookup = {
        relation: head.classes.index(relation.value) for relation in RelationId
    }
    return [
        {
            relation: float(logits[index, lookup[relation]].item())
            for relation in RelationId
        }
        for index in range(len(logits))
    ]


def _r1_scores(
    embeddings: Tensor,
    definitions: Mapping[RelationId, Tensor],
    aggregation: str,
) -> list[dict[RelationId, float]]:
    if aggregation.startswith("top_k_mean_"):
        top_k = int(aggregation.rsplit("_", 1)[1])
        method = "top_k_mean"
    else:
        top_k = None
        method = aggregation
    return [
        r1_definition_ensemble_scores(
            embedding,
            definitions,
            aggregation=method,
            top_k=top_k,
        )
        for embedding in embeddings
    ]


def _r2_scores(
    head: PairwiseRidgeEvidenceHead,
    embeddings: Tensor,
    definitions: Mapping[RelationId, Tensor],
) -> list[dict[RelationId, float]]:
    features = build_pairwise_relation_features(embeddings, definitions)
    values = head.score_batch(features)
    return [
        {
            relation: float(values[relation][index].item())
            for relation in RelationId
        }
        for index in range(len(embeddings))
    ]


def score_public_rows(
    config_path: str | Path,
    context: PublicModelContext,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[RelationId, float]]]:
    """产生所有 calibration 候选 evidence，不做参数选择。"""

    values = load_config(config_path)
    batch_size = int(values["models"]["batch_size"])
    scores: dict[str, list[dict[RelationId, float]]] = {}

    r0_embeddings = encode_semantic_texts(
        context.encoders["bge-small"],
        _rows_text(rows, str(values["models"]["R0"]["view"])),
        batch_size=batch_size,
        e5_input_type="query",
    )
    scores["R0"] = _r0_scores(context.r0_head, r0_embeddings)

    phrase_texts = _rows_text(rows, "phrase_only")
    minilm_embeddings = encode_semantic_texts(
        context.encoders["minilm"],
        phrase_texts,
        batch_size=batch_size,
        e5_input_type="query",
    )
    for aggregation in values["models"]["R1"]["aggregations"]:
        scores[f"R1:{aggregation}"] = _r1_scores(
            minilm_embeddings,
            context.definition_embeddings["minilm"],
            str(aggregation),
        )

    encoder_embeddings: dict[str, Tensor] = {"minilm": minilm_embeddings}
    encoder_embeddings["e5-base"] = encode_semantic_texts(
        context.encoders["e5-base"],
        phrase_texts,
        batch_size=batch_size,
        e5_input_type="query",
    )
    for key, head in context.r2_heads.items():
        candidate_name = key.split(":", 1)[0]
        declaration = values["models"]["R2"]["candidates"][candidate_name]
        encoder_name = str(declaration["semantic_encoder"])
        scores[f"R2:{key}"] = _r2_scores(
            head,
            encoder_embeddings[encoder_name],
            context.definition_embeddings[encoder_name],
        )
    return scores


def verify_selected_head(
    context: PublicModelContext, router_parameters: Mapping[str, Any]
) -> None:
    if (
        context.r0_head_file_sha256
        != router_parameters["r0_head_file_sha256"]
        or context.r0_head_state_sha256
        != router_parameters["r0_head_state_sha256"]
    ):
        raise C24CProtocolError("R0 head 在 router freeze 后漂移")
    r2_key = str(router_parameters["selected_r2_candidate"])
    if (
        r2_key not in context.r2_head_state_sha256
        or context.r2_head_state_sha256[r2_key]
        != router_parameters["selected_r2_head_state_sha256"]
    ):
        raise C24CProtocolError("R2 selected head 在 router freeze 后漂移")


__all__ = [
    "PublicModelContext",
    "R0_CLASS_ORDER",
    "build_c23_public_rows",
    "build_public_model_context",
    "load_definitions",
    "reconstruct_r0_head",
    "score_public_rows",
    "verify_selected_head",
]
