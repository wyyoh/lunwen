from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


STAGE_C23_SEMANTIC_SCHEMA_VERSION = 1
SEMANTIC_PREPROCESSING_VERSION = "c23-semantic-views-v1"
SEMANTIC_VIEWS = (
    "phrase_only",
    "entity_masked",
    "entity_masked_strip_suffix",
)

# Deliberately excludes pickle, ONNX, TensorFlow, and OpenVINO weights.  The
# encoder is loaded from the resulting local snapshot only after this bounded
# download finishes.
ENCODER_SNAPSHOT_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "1_Pooling/config.json",
    "2_Normalize/config.json",
    "README.md",
    "LICENSE*",
)


@dataclass(frozen=True)
class SemanticEncoderSpec:
    name: str
    model_id: str
    revision: str
    license: str
    dimension: int
    pooling: str
    max_length: int
    requires_e5_prefix: bool = False

    def validate(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("semantic encoders require a fixed model ID and revision")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision.lower()
        ):
            raise ValueError("semantic encoder revision must be a full commit SHA")
        if self.dimension <= 0 or self.max_length <= 0:
            raise ValueError("semantic encoder dimensions must be positive")
        if self.pooling not in {"mask_mean", "cls"}:
            raise ValueError("semantic encoder pooling must be mask_mean or cls")


SEMANTIC_ENCODER_SPECS: dict[str, SemanticEncoderSpec] = {
    "minilm": SemanticEncoderSpec(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        license="apache-2.0",
        dimension=384,
        pooling="mask_mean",
        max_length=256,
    ),
    "e5-small": SemanticEncoderSpec(
        name="e5-small",
        model_id="intfloat/e5-small-v2",
        revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
        license="mit",
        dimension=384,
        pooling="mask_mean",
        max_length=256,
        requires_e5_prefix=True,
    ),
    "bge-small": SemanticEncoderSpec(
        name="bge-small",
        model_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        license="mit",
        dimension=384,
        pooling="cls",
        max_length=256,
    ),
    "e5-base": SemanticEncoderSpec(
        name="e5-base",
        model_id="intfloat/e5-base-v2",
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        license="mit",
        dimension=768,
        pooling="mask_mean",
        max_length=256,
        requires_e5_prefix=True,
    ),
}


def resolve_semantic_encoder_spec(
    value: str | SemanticEncoderSpec,
) -> SemanticEncoderSpec:
    if isinstance(value, SemanticEncoderSpec):
        value.validate()
        return value
    if value in SEMANTIC_ENCODER_SPECS:
        return SEMANTIC_ENCODER_SPECS[value]
    matches = [
        spec for spec in SEMANTIC_ENCODER_SPECS.values() if spec.model_id == value
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown semantic encoder {value!r}")
    return matches[0]


def download_semantic_encoder_snapshot(
    spec: str | SemanticEncoderSpec,
    *,
    cache_dir: str | Path | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    """Download only the pinned safetensors/tokenizer snapshot files."""

    resolved = resolve_semantic_encoder_spec(spec)
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    arguments: dict[str, Any] = {
        "repo_id": resolved.model_id,
        "revision": resolved.revision,
        "allow_patterns": list(ENCODER_SNAPSHOT_ALLOW_PATTERNS),
    }
    if cache_dir is not None:
        arguments["cache_dir"] = str(Path(cache_dir))
    path = Path(snapshot_download_fn(**arguments))
    if not path.is_dir():
        raise FileNotFoundError(f"semantic encoder snapshot is absent: {path}")
    return path


@dataclass
class LoadedSemanticEncoder:
    spec: SemanticEncoderSpec
    snapshot_path: Path
    tokenizer: Any
    model: Any
    device: torch.device


def load_semantic_encoder(
    spec: str | SemanticEncoderSpec,
    *,
    cache_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    snapshot_download_fn: Callable[..., str] | None = None,
    tokenizer_class: Any | None = None,
    model_class: Any | None = None,
) -> LoadedSemanticEncoder:
    """Resolve a pinned snapshot, then load strictly from its local directory."""

    resolved = resolve_semantic_encoder_spec(spec)
    snapshot_path = download_semantic_encoder_snapshot(
        resolved,
        cache_dir=cache_dir,
        snapshot_download_fn=snapshot_download_fn,
    )
    if tokenizer_class is None or model_class is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer_class = tokenizer_class or AutoTokenizer
        model_class = model_class or AutoModel
    tokenizer = tokenizer_class.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = model_class.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    resolved_device = torch.device(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(resolved_device)
    return LoadedSemanticEncoder(
        spec=resolved,
        snapshot_path=snapshot_path,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
    )


def prepare_encoder_texts(
    spec: str | SemanticEncoderSpec,
    texts: Sequence[str],
    *,
    e5_input_type: str = "query",
) -> list[str]:
    """Apply E5's mandatory input prefix without double-prefixing text."""

    resolved = resolve_semantic_encoder_spec(spec)
    prepared = [str(text).strip() for text in texts]
    if any(not text for text in prepared):
        raise ValueError("semantic encoder text cannot be empty")
    if not resolved.requires_e5_prefix:
        return prepared
    if e5_input_type not in {"query", "passage"}:
        raise ValueError("E5 input type must be query or passage")
    prefix = f"{e5_input_type}: "
    alternate = "passage: " if e5_input_type == "query" else "query: "
    output = []
    for text in prepared:
        if text.startswith(prefix):
            output.append(text)
        elif text.startswith(alternate):
            output.append(prefix + text[len(alternate) :])
        else:
            output.append(prefix + text)
    return output


def mask_mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("mask-mean pooling expects [batch, tokens, hidden] and mask")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states and attention mask have incompatible shapes")
    mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * mask).sum(dim=1) / denominator


def pool_semantic_embeddings(
    last_hidden_state: Tensor,
    attention_mask: Tensor,
    pooling: str,
) -> Tensor:
    if pooling == "mask_mean":
        pooled = mask_mean_pool(last_hidden_state, attention_mask)
    elif pooling == "cls":
        if last_hidden_state.ndim != 3 or last_hidden_state.size(1) == 0:
            raise ValueError("CLS pooling expects non-empty token hidden states")
        pooled = last_hidden_state[:, 0]
    else:
        raise ValueError("unknown semantic encoder pooling")
    # Explicit normalization is retained even when a Sentence-Transformers
    # repository also declares a Normalize module, because AutoModel does not
    # execute that module.
    return F.normalize(pooled.float(), p=2, dim=-1)


@torch.inference_mode()
def encode_semantic_texts(
    encoder: LoadedSemanticEncoder,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    e5_input_type: str = "query",
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("semantic encoder batch size must be positive")
    prepared = prepare_encoder_texts(
        encoder.spec, texts, e5_input_type=e5_input_type
    )
    batches = []
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset : offset + batch_size]
        tokenized = encoder.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=encoder.spec.max_length,
            return_tensors="pt",
        )
        if "attention_mask" not in tokenized:
            raise ValueError("semantic tokenizer did not return an attention mask")
        inputs = {
            name: value.to(encoder.device) if isinstance(value, Tensor) else value
            for name, value in tokenized.items()
        }
        output = encoder.model(**inputs)
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if last_hidden_state is None:
            last_hidden_state = output[0]
        pooled = pool_semantic_embeddings(
            last_hidden_state,
            inputs["attention_mask"],
            encoder.spec.pooling,
        )
        if pooled.size(1) != encoder.spec.dimension:
            raise ValueError(
                "semantic encoder output dimension does not match its pinned spec"
            )
        batches.append(pooled.cpu())
    if not batches:
        return torch.empty((0, encoder.spec.dimension), dtype=torch.float32)
    return torch.cat(batches, dim=0)


_VIEW_ALIASES = {
    "masked_question": "entity_masked",
    "entity_masked_question": "entity_masked",
    "masked_strip_suffix": "entity_masked_strip_suffix",
}
_ANSWER_SUFFIX = re.compile(
    r"\s*(?:answer|response|output|reply|value)\s*:\s*$", flags=re.IGNORECASE
)


def _mask_entity(text: str, entity: str, mask_token: str) -> str:
    if not entity:
        raise ValueError("cannot mask an empty entity")
    if mask_token in text:
        return text
    escaped = re.escape(entity)
    left = r"(?<!\w)" if entity[0].isalnum() or entity[0] == "_" else ""
    right = r"(?!\w)" if entity[-1].isalnum() or entity[-1] == "_" else ""
    masked, replacements = re.subn(
        left + escaped + right,
        mask_token,
        text,
        flags=re.IGNORECASE,
    )
    if replacements == 0:
        raise ValueError(f"entity {entity!r} is absent from its prompt")
    return masked


def build_semantic_view_texts(
    rows: Sequence[Mapping[str, Any]],
    view: str,
    *,
    prompt_field: str = "prompt",
    entity_field: str = "entity",
    phrase_field: str = "relation_phrase",
    mask_token: str = "<ENTITY>",
) -> list[str]:
    """Materialize one of the three preregistered, answer-free text views."""

    view = _VIEW_ALIASES.get(view, view)
    if view not in SEMANTIC_VIEWS:
        raise ValueError(f"unknown semantic view {view!r}")
    output = []
    for row in rows:
        if view == "phrase_only":
            if phrase_field not in row:
                raise ValueError(f"semantic row is missing {phrase_field!r}")
            text = str(row[phrase_field]).strip()
        else:
            missing = [
                field
                for field in (prompt_field, entity_field)
                if field not in row
            ]
            if missing:
                raise ValueError(f"semantic row is missing fields {missing}")
            text = _mask_entity(
                str(row[prompt_field]), str(row[entity_field]), mask_token
            ).strip()
            if view == "entity_masked_strip_suffix":
                text = _ANSWER_SUFFIX.sub("", text).rstrip()
        if not text:
            raise ValueError("semantic view produced empty text")
        output.append(text)
    return output


def build_all_semantic_views(
    rows: Sequence[Mapping[str, Any]], **kwargs: Any
) -> dict[str, list[str]]:
    return {
        view: build_semantic_view_texts(rows, view, **kwargs)
        for view in SEMANTIC_VIEWS
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_model_file_manifest(
    snapshot_path: str | Path,
    spec: str | SemanticEncoderSpec,
) -> dict[str, Any]:
    """Hash every regular file visible in a resolved local model snapshot."""

    root = Path(snapshot_path)
    if not root.is_dir():
        raise FileNotFoundError(f"model snapshot is absent: {root}")
    resolved = resolve_semantic_encoder_spec(spec)
    files = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError("model snapshot contains no files")
    payload = {
        "schema_version": STAGE_C23_SEMANTIC_SCHEMA_VERSION,
        "model_id": resolved.model_id,
        "revision": resolved.revision,
        "license": resolved.license,
        "file_count": len(files),
        "files": files,
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return payload


def semantic_text_sha256(texts: Sequence[str]) -> str:
    return _canonical_json_sha256([str(text) for text in texts])


def semantic_embedding_cache_path(
    cache_dir: str | Path,
    spec: str | SemanticEncoderSpec,
    view: str,
    texts: Sequence[str],
) -> Path:
    resolved = resolve_semantic_encoder_spec(spec)
    normalized_view = _VIEW_ALIASES.get(view, view)
    if normalized_view not in SEMANTIC_VIEWS:
        raise ValueError(f"unknown semantic view {view!r}")
    digest = semantic_text_sha256(texts)[:16]
    return Path(cache_dir) / f"{resolved.name}-{normalized_view}-{digest}.pt"


def save_semantic_embedding_cache(
    path: str | Path,
    embeddings: Tensor,
    *,
    spec: str | SemanticEncoderSpec,
    view: str,
    texts: Sequence[str],
    model_manifest: Mapping[str, Any],
    row_ids: Sequence[str] | None = None,
    preprocessing_version: str = SEMANTIC_PREPROCESSING_VERSION,
) -> dict[str, Any]:
    resolved = resolve_semantic_encoder_spec(spec)
    normalized_view = _VIEW_ALIASES.get(view, view)
    if normalized_view not in SEMANTIC_VIEWS:
        raise ValueError(f"unknown semantic view {view!r}")
    tensor = F.normalize(embeddings.detach().cpu().float(), p=2, dim=-1)
    if tensor.ndim != 2 or tensor.size(0) != len(texts):
        raise ValueError("semantic cache embeddings and texts are incompatible")
    if tensor.size(1) != resolved.dimension:
        raise ValueError("semantic cache embedding dimension is incorrect")
    if row_ids is not None and len(row_ids) != len(texts):
        raise ValueError("semantic cache row IDs and texts are incompatible")
    manifest_sha256 = str(model_manifest.get("manifest_sha256", ""))
    if len(manifest_sha256) != 64:
        raise ValueError("semantic cache requires a hashed model manifest")
    metadata = {
        "format_version": STAGE_C23_SEMANTIC_SCHEMA_VERSION,
        "stage": "C2.3-public-semantic-encoder-audit",
        "model_id": resolved.model_id,
        "revision": resolved.revision,
        "dimension": resolved.dimension,
        "pooling": resolved.pooling,
        "view": normalized_view,
        "preprocessing_version": preprocessing_version,
        "text_sha256": semantic_text_sha256(texts),
        "model_manifest_sha256": manifest_sha256,
        "row_count": len(texts),
        "row_ids": [str(value) for value in row_ids] if row_ids is not None else None,
    }
    payload = {**metadata, "embeddings": tensor}
    destination = Path(path)
    if destination.suffix.lower() != ".pt":
        raise ValueError("semantic embedding caches must use the .pt extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return metadata


def load_semantic_embedding_cache(
    path: str | Path,
    *,
    expected_spec: str | SemanticEncoderSpec | None = None,
    expected_view: str | None = None,
    expected_texts: Sequence[str] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "model_id",
        "revision",
        "dimension",
        "view",
        "text_sha256",
        "model_manifest_sha256",
        "row_count",
        "embeddings",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"semantic cache is missing {sorted(missing)}")
    if int(payload["format_version"]) != STAGE_C23_SEMANTIC_SCHEMA_VERSION:
        raise ValueError("semantic cache format version is unsupported")
    embeddings = payload["embeddings"]
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        raise ValueError("semantic cache embeddings are malformed")
    if embeddings.size(0) != int(payload["row_count"]):
        raise ValueError("semantic cache row count is inconsistent")
    if embeddings.size(1) != int(payload["dimension"]):
        raise ValueError("semantic cache dimension is inconsistent")
    if expected_spec is not None:
        resolved = resolve_semantic_encoder_spec(expected_spec)
        if payload["model_id"] != resolved.model_id or payload["revision"] != resolved.revision:
            raise ValueError("semantic cache model revision does not match")
    if expected_view is not None:
        expected_view = _VIEW_ALIASES.get(expected_view, expected_view)
        if payload["view"] != expected_view:
            raise ValueError("semantic cache view does not match")
    if expected_texts is not None and payload["text_sha256"] != semantic_text_sha256(
        expected_texts
    ):
        raise ValueError("semantic cache input text hash does not match")
    if (
        expected_manifest_sha256 is not None
        and payload["model_manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("semantic cache model manifest does not match")
    return payload


def definition_prototype_matching(
    query_embeddings: Tensor,
    definition_embeddings: Mapping[str, Tensor],
    *,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Classify queries by cosine similarity to fixed public definitions."""

    if query_embeddings.ndim != 2 or not definition_embeddings:
        raise ValueError("definition matching requires query and prototype matrices")
    if aggregation not in {"mean", "max"}:
        raise ValueError("definition aggregation must be mean or max")
    queries = F.normalize(query_embeddings.float(), p=2, dim=-1)
    classes = tuple(sorted(str(value) for value in definition_embeddings))
    class_scores = []
    prototype_counts = {}
    for relation in classes:
        definitions = definition_embeddings[relation].float()
        if definitions.ndim == 1:
            definitions = definitions.unsqueeze(0)
        if definitions.ndim != 2 or definitions.size(0) == 0:
            raise ValueError(f"relation {relation!r} has no definition embeddings")
        if definitions.size(1) != queries.size(1):
            raise ValueError("query and definition embedding dimensions differ")
        definitions = F.normalize(definitions, p=2, dim=-1)
        prototype_counts[relation] = definitions.size(0)
        if aggregation == "mean":
            prototype = F.normalize(definitions.mean(dim=0), p=2, dim=0)
            class_scores.append(queries @ prototype)
        else:
            class_scores.append((queries @ definitions.T).max(dim=1).values)
    logits = torch.stack(class_scores, dim=1)
    probabilities = logits.softmax(dim=1)
    predicted_indices = logits.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    if logits.size(1) > 1:
        top_two = logits.topk(2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
    else:
        margin = torch.full((len(logits),), float("inf"))
    return {
        "classes": classes,
        "aggregation": aggregation,
        "prototype_counts": prototype_counts,
        "logits": logits,
        "probabilities": probabilities,
        "predicted_indices": predicted_indices,
        "predictions": [classes[int(index)] for index in predicted_indices],
        "confidence": confidence,
        "margin": margin,
    }


def family_macro_accuracy(
    predictions: Sequence[str | int],
    targets: Sequence[str | int],
    families: Sequence[str],
) -> dict[str, Any]:
    if not predictions or not (
        len(predictions) == len(targets) == len(families)
    ):
        raise ValueError("family accuracy inputs must be non-empty and aligned")
    family_indices: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        family_indices.setdefault(str(family), []).append(index)
    per_family = {}
    for family, indices in sorted(family_indices.items()):
        per_family[family] = sum(
            predictions[index] == targets[index] for index in indices
        ) / len(indices)
    micro = sum(
        prediction == target for prediction, target in zip(predictions, targets)
    ) / len(targets)
    return {
        "macro_accuracy": sum(per_family.values()) / len(per_family),
        "micro_accuracy": micro,
        "num_families": len(per_family),
        "per_family_accuracy": per_family,
    }


def family_bootstrap_accuracy_ci(
    predictions: Sequence[str | int],
    targets: Sequence[str | int],
    families: Sequence[str],
    *,
    confidence_level: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if not 0 < confidence_level < 1:
        raise ValueError("bootstrap confidence level must be between zero and one")
    if num_resamples <= 0:
        raise ValueError("bootstrap requires at least one resample")
    base = family_macro_accuracy(predictions, targets, families)
    values = torch.tensor(
        list(base["per_family_accuracy"].values()), dtype=torch.float64
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(values),
        (num_resamples, len(values)),
        generator=generator,
    )
    samples = values[indices].mean(dim=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(values.mean()),
        "lower": float(torch.quantile(samples, alpha)),
        "upper": float(torch.quantile(samples, 1.0 - alpha)),
        "confidence_level": confidence_level,
        "num_resamples": num_resamples,
        "num_families": len(values),
        "seed": seed,
    }


def leave_one_family_out_relation_margin(
    embeddings: Tensor,
    relations: Sequence[str],
    families: Sequence[str],
) -> dict[str, Any]:
    """Evaluate each lexical family against relation centroids that exclude it."""

    if embeddings.ndim != 2 or not (
        len(embeddings) == len(relations) == len(families)
    ):
        raise ValueError("leave-one-family-out inputs are incompatible")
    if not relations:
        raise ValueError("leave-one-family-out inputs cannot be empty")
    unit = F.normalize(embeddings.float(), p=2, dim=-1)
    relation_names = tuple(sorted(set(str(value) for value in relations)))
    if len(relation_names) < 2:
        raise ValueError("relation margin needs at least two relations")
    relation_lookup = {value: index for index, value in enumerate(relation_names)}
    relation_values = [str(value) for value in relations]
    family_values = [str(value) for value in families]
    per_family: dict[str, dict[str, Any]] = {}
    all_margins = []
    all_correct = []
    for family in sorted(set(family_values)):
        held_indices = [
            index for index, value in enumerate(family_values) if value == family
        ]
        reference_indices = [
            index for index, value in enumerate(family_values) if value != family
        ]
        centroids = []
        for relation in relation_names:
            indices = [
                index
                for index in reference_indices
                if relation_values[index] == relation
            ]
            if not indices:
                raise ValueError(
                    f"holding out family {family!r} leaves relation {relation!r} empty"
                )
            centroids.append(F.normalize(unit[indices].mean(dim=0), p=2, dim=0))
        logits = unit[held_indices] @ torch.stack(centroids).T
        true_indices = torch.tensor(
            [relation_lookup[relation_values[index]] for index in held_indices]
        )
        true_scores = logits.gather(1, true_indices[:, None]).squeeze(1)
        other = logits.clone()
        other.scatter_(1, true_indices[:, None], float("-inf"))
        margins = true_scores - other.max(dim=1).values
        correct = logits.argmax(dim=1).eq(true_indices)
        per_family[family] = {
            "relation": sorted({relation_values[index] for index in held_indices}),
            "num_rows": len(held_indices),
            "mean_margin": float(margins.mean()),
            "accuracy": float(correct.float().mean()),
        }
        all_margins.extend(float(value) for value in margins)
        all_correct.extend(bool(value) for value in correct)
    return {
        "macro_margin": sum(
            value["mean_margin"] for value in per_family.values()
        )
        / len(per_family),
        "micro_margin": sum(all_margins) / len(all_margins),
        "macro_accuracy": sum(
            value["accuracy"] for value in per_family.values()
        )
        / len(per_family),
        "micro_accuracy": sum(all_correct) / len(all_correct),
        "num_families": len(per_family),
        "per_family": per_family,
    }


def binary_auroc(scores: Tensor, positive: Tensor) -> float:
    scores = scores.detach().cpu().double().flatten()
    positive = positive.detach().cpu().bool().flatten()
    if len(scores) != len(positive):
        raise ValueError("AUROC scores and labels are incompatible")
    positive_scores = scores[positive]
    negative_scores = scores[~positive]
    if not len(positive_scores) or not len(negative_scores):
        raise ValueError("AUROC requires positive and negative examples")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(
        (comparisons.gt(0).double() + 0.5 * comparisons.eq(0).double()).mean()
    )


def binary_average_precision(scores: Tensor, positive: Tensor) -> float:
    scores = scores.detach().cpu().double().flatten()
    positive = positive.detach().cpu().bool().flatten()
    if len(scores) != len(positive):
        raise ValueError("AUPR scores and labels are incompatible")
    num_positive = int(positive.sum())
    if num_positive == 0 or num_positive == len(positive):
        raise ValueError("AUPR requires positive and negative examples")
    thresholds = torch.unique(scores).sort(descending=True).values
    previous_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        accepted = scores.ge(threshold)
        true_positive = int((accepted & positive).sum())
        false_positive = int((accepted & ~positive).sum())
        recall = true_positive / num_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def expected_calibration_error(
    confidence: Tensor,
    correct: Tensor,
    *,
    num_bins: int = 10,
) -> float:
    confidence = confidence.detach().cpu().float().flatten()
    correct = correct.detach().cpu().float().flatten()
    if len(confidence) != len(correct) or not len(confidence):
        raise ValueError("ECE confidence and correctness must be non-empty and aligned")
    if num_bins <= 0:
        raise ValueError("ECE requires at least one bin")
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError("ECE confidence must lie in [0, 1]")
    bin_indices = torch.clamp((confidence * num_bins).long(), max=num_bins - 1)
    error = 0.0
    for index in range(num_bins):
        mask = bin_indices.eq(index)
        if bool(mask.any()):
            weight = float(mask.float().mean())
            error += weight * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def coverage_accuracy_curve(
    confidence: Tensor,
    correct: Tensor,
) -> dict[str, Any]:
    confidence = confidence.detach().cpu().float().flatten()
    correct = correct.detach().cpu().float().flatten()
    if len(confidence) != len(correct) or not len(confidence):
        raise ValueError("coverage inputs must be non-empty and aligned")
    # Stable sorting makes tied-confidence behavior deterministic.
    order = torch.argsort(confidence, descending=True, stable=True)
    sorted_confidence = confidence[order]
    cumulative_accuracy = correct[order].cumsum(0) / torch.arange(
        1, len(correct) + 1, dtype=torch.float32
    )
    coverage = torch.arange(1, len(correct) + 1, dtype=torch.float32) / len(correct)
    risk = 1.0 - cumulative_accuracy
    curve = [
        {
            "coverage": float(coverage[index]),
            "accuracy": float(cumulative_accuracy[index]),
            "risk": float(risk[index]),
            "threshold": float(sorted_confidence[index]),
        }
        for index in range(len(correct))
    ]
    return {"aurc": float(risk.mean()), "curve": curve}


def open_set_rejection_metrics(
    logits: Tensor,
    targets: Tensor,
    known_mask: Tensor | None = None,
    *,
    num_ece_bins: int = 10,
    known_coverage_target: float = 0.95,
) -> dict[str, Any]:
    """Calibration and rejection metrics; unknown targets may be encoded as -1."""

    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError("rejection metrics require multi-class logits")
    targets = targets.detach().cpu().long().flatten()
    if len(targets) != len(logits):
        raise ValueError("rejection logits and targets are incompatible")
    if known_mask is None:
        known = targets.ge(0)
    else:
        known = known_mask.detach().cpu().bool().flatten()
        if len(known) != len(targets):
            raise ValueError("known mask and targets are incompatible")
    if not bool(known.any()) or not bool((~known).any()):
        raise ValueError("rejection metrics require known and unknown examples")
    if not 0 < known_coverage_target <= 1:
        raise ValueError("known coverage target must be in (0, 1]")
    probabilities = logits.detach().cpu().float().softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    valid_targets = targets[known]
    if bool((valid_targets < 0).any()) or bool(
        (valid_targets >= logits.size(1)).any()
    ):
        raise ValueError("known target is outside the classifier label space")
    known_correct = predictions[known].eq(valid_targets)
    selective_correct = known & predictions.eq(targets.clamp_min(0))
    coverage = coverage_accuracy_curve(confidence, selective_correct)

    known_scores = confidence[known]
    required = max(1, math.ceil(known_coverage_target * len(known_scores)))
    threshold = known_scores.sort(descending=True).values[required - 1]
    accepted = confidence.ge(threshold)
    achieved_known_coverage = float(accepted[known].float().mean())
    true_negative_rate = float((~accepted[~known]).float().mean())
    return {
        "known_detection_auroc": binary_auroc(confidence, known),
        "known_detection_aupr": binary_average_precision(confidence, known),
        "known_ece": expected_calibration_error(
            confidence[known], known_correct, num_bins=num_ece_bins
        ),
        "known_accuracy": float(known_correct.float().mean()),
        "coverage_aurc": coverage["aurc"],
        "coverage_accuracy_curve": coverage["curve"],
        "tnr_at_95_known_coverage": true_negative_rate,
        "tnr_at_target_known_coverage": true_negative_rate,
        "known_coverage_target": known_coverage_target,
        "achieved_known_coverage": achieved_known_coverage,
        "rejection_threshold": float(threshold),
        "confidence": confidence,
        "predictions": predictions,
    }


@dataclass(frozen=True)
class RidgeLinearHead:
    classes: tuple[str, ...]
    feature_mean: Tensor
    feature_scale: Tensor
    weights: Tensor
    regularizer: float

    def logits(self, features: Tensor) -> Tensor:
        matrix = features.detach().cpu().double()
        if matrix.ndim != 2 or matrix.size(1) != self.feature_mean.numel():
            raise ValueError("ridge head features have an incompatible shape")
        matrix = (matrix - self.feature_mean) / self.feature_scale
        matrix = torch.cat(
            [matrix, torch.ones((len(matrix), 1), dtype=matrix.dtype)], dim=1
        )
        return (matrix @ self.weights).float()


def fit_ridge_linear_head(
    train_embeddings: Tensor,
    train_labels: Sequence[str],
    *,
    ridge_strength: float = 0.01,
    classes: Sequence[str] | None = None,
) -> RidgeLinearHead:
    """Fit a deterministic, CPU float64 one-vs-all ridge classifier."""

    if ridge_strength <= 0:
        raise ValueError("ridge strength must be positive")
    matrix = train_embeddings.detach().cpu().double()
    if matrix.ndim != 2 or len(matrix) != len(train_labels) or not len(matrix):
        raise ValueError("ridge training embeddings and labels are incompatible")
    class_names = tuple(sorted(set(train_labels))) if classes is None else tuple(classes)
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("ridge head needs at least two unique classes")
    lookup = {value: index for index, value in enumerate(class_names)}
    if any(label not in lookup for label in train_labels):
        raise ValueError("ridge training labels are absent from the class list")
    target_indices = torch.tensor([lookup[label] for label in train_labels])
    targets = F.one_hot(target_indices, num_classes=len(class_names)).double()
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False).clamp_min(1e-5)
    standardized = (matrix - mean) / scale
    standardized = torch.cat(
        [
            standardized,
            torch.ones((len(standardized), 1), dtype=standardized.dtype),
        ],
        dim=1,
    )
    kernel = standardized @ standardized.T
    regularizer = max(
        float(kernel.diagonal().mean()) * float(ridge_strength), 1e-8
    )
    kernel.diagonal().add_(regularizer)
    coefficients = torch.linalg.solve(kernel, targets)
    weights = standardized.T @ coefficients
    return RidgeLinearHead(
        classes=class_names,
        feature_mean=mean,
        feature_scale=scale,
        weights=weights,
        regularizer=regularizer,
    )


def ridge_linear_head_logits(
    train_embeddings: Tensor,
    train_labels: Sequence[str],
    evaluation_embeddings: Tensor,
    *,
    ridge_strength: float = 0.01,
    classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    head = fit_ridge_linear_head(
        train_embeddings,
        train_labels,
        ridge_strength=ridge_strength,
        classes=classes,
    )
    return {
        "classes": head.classes,
        "train_logits": head.logits(train_embeddings),
        "evaluation_logits": head.logits(evaluation_embeddings),
        "regularizer": head.regularizer,
        "head": head,
    }
