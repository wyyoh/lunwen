from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


CANONICALIZER_FORMAT_VERSION = 1
POOL_NAMES = ("entity_span", "question_without_entity", "answer_position")


@dataclass(frozen=True)
class CanonicalizerConfig:
    architecture: str
    num_input_layers: int
    core_hidden_size: int
    query_dim: int = 128
    mlp_hidden_size: int = 256
    dropout: float = 0.1
    num_entities: int = 16
    num_relations: int = 3
    num_templates: int = 6

    def validate(self) -> None:
        if self.architecture not in {"joint", "factorized"}:
            raise ValueError("architecture must be joint or factorized")
        for name in (
            "num_input_layers",
            "core_hidden_size",
            "query_dim",
            "mlp_hidden_size",
            "num_entities",
            "num_relations",
            "num_templates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class QueryCanonicalizer(nn.Module):
    """Map explicit entity/question/answer-position pools to a unit query vector."""

    def __init__(self, config: CanonicalizerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # Each semantic pool learns its own mixture over the selected core layers.
        self.layer_logits = nn.Parameter(torch.zeros(len(POOL_NAMES), config.num_input_layers))
        hidden = config.core_hidden_size
        middle = config.mlp_hidden_size
        query = config.query_dim
        dropout = config.dropout
        if config.architecture == "joint":
            self.joint = nn.Sequential(
                nn.Linear(len(POOL_NAMES) * hidden, middle),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(middle, query),
            )
            self.entity_encoder = None
            self.relation_encoder = None
            self.entity_projection = None
            self.relation_projection = None
            self.interaction_projection = None
        else:
            self.joint = None
            self.entity_encoder = nn.Sequential(
                nn.Linear(hidden, middle),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(middle, query),
            )
            self.relation_encoder = nn.Sequential(
                nn.Linear(2 * hidden, middle),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(middle, query),
            )
            self.entity_projection = nn.Linear(query, query, bias=False)
            self.relation_projection = nn.Linear(query, query, bias=False)
            self.interaction_projection = nn.Linear(query, query, bias=False)

    def mixed_pools(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(
                "canonicalizer features must have shape [batch, pools, layers, hidden]"
            )
        expected = (
            len(POOL_NAMES),
            self.config.num_input_layers,
            self.config.core_hidden_size,
        )
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"feature layout {tuple(features.shape[1:])} does not match {expected}"
            )
        weights = self.layer_logits.softmax(dim=-1)
        return (features * weights[None, :, :, None]).sum(dim=2)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        pools = self.mixed_pools(features.float())
        if self.config.architecture == "joint":
            assert self.joint is not None
            raw_query = self.joint(pools.flatten(start_dim=1))
            entity = pools[:, 0]
            relation = torch.cat([pools[:, 1], pools[:, 2]], dim=-1)
        else:
            assert self.entity_encoder is not None
            assert self.relation_encoder is not None
            assert self.entity_projection is not None
            assert self.relation_projection is not None
            assert self.interaction_projection is not None
            entity = self.entity_encoder(pools[:, 0])
            relation = self.relation_encoder(
                torch.cat([pools[:, 1], pools[:, 2]], dim=-1)
            )
            raw_query = (
                self.entity_projection(entity)
                + self.relation_projection(relation)
                + self.interaction_projection(entity * relation)
            )
        return {
            "query": F.normalize(raw_query, dim=-1),
            "entity": entity,
            "relation": relation,
            "layer_weights": self.layer_logits.softmax(dim=-1),
        }


class CanonicalizerSystem(nn.Module):
    def __init__(self, config: CanonicalizerConfig) -> None:
        super().__init__()
        self.config = config
        self.canonicalizer = QueryCanonicalizer(config)
        self.relation_head = nn.Linear(config.query_dim, config.num_relations)
        self.entity_head = nn.Linear(config.query_dim, config.num_entities)
        self.template_head = nn.Linear(config.query_dim, config.num_templates)

    def forward(self, features: Tensor, *, adversary_strength: float = 0.0) -> dict[str, Tensor]:
        output = self.canonicalizer(features)
        query = output["query"]
        # Factorized variants supervise the explicit semantic branches directly.
        # The joint Q1 baseline has no query-sized branch latents, so its unused
        # auxiliary heads remain attached to the final query for shape stability.
        if self.config.architecture == "factorized":
            relation_source = F.normalize(output["relation"], dim=-1)
            entity_source = F.normalize(output["entity"], dim=-1)
        else:
            relation_source = query
            entity_source = query
        output["relation_logits"] = self.relation_head(relation_source)
        output["entity_logits"] = self.entity_head(entity_source)
        output["template_logits"] = self.template_head(
            gradient_reverse(query, adversary_strength)
        )
        return output


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(value: Tensor, strength: float = 1.0) -> Tensor:
    return _GradientReversal.apply(value, strength)


def supervised_contrastive_loss(
    queries: Tensor, labels: Tensor, *, temperature: float = 0.07
) -> Tensor:
    if queries.ndim != 2 or labels.ndim != 1 or len(queries) != len(labels):
        raise ValueError("invalid supervised-contrastive inputs")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    unit = F.normalize(queries.float(), dim=-1)
    logits = unit @ unit.T / temperature
    self_mask = torch.eye(len(queries), dtype=torch.bool, device=queries.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    if bool((positive.sum(dim=1) == 0).any()):
        raise ValueError("every contrastive anchor needs a positive example")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp().masked_fill(self_mask, 0.0)
    log_probability = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    mean_positive = (log_probability * positive).sum(dim=1) / positive.sum(dim=1)
    return -mean_positive.mean()


class StructuredFactBatchSampler:
    """Sample same-entity/different-relation and same-relation/different-entity negatives."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        batch_facts: int,
        templates_per_fact: int,
        seed: int,
    ) -> None:
        if batch_facts < 4 or batch_facts % 2:
            raise ValueError("batch_facts must be an even number of at least four")
        if templates_per_fact < 2:
            raise ValueError("templates_per_fact must be at least two")
        self.batch_facts = batch_facts
        self.templates_per_fact = templates_per_fact
        self.rng = np.random.default_rng(seed)
        self.by_entity_relation: dict[tuple[str, str], list[int]] = {}
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(rows):
            key = (str(row["entity"]), str(row["attribute"]))
            grouped.setdefault(key, []).append(index)
        if any(len(indices) < templates_per_fact for indices in grouped.values()):
            raise ValueError("a fact has too few templates for a structured batch")
        self.by_entity_relation = grouped
        self.entities = sorted({entity for entity, _ in grouped})
        self.relations = sorted({relation for _, relation in grouped})
        if len(self.relations) < 2:
            raise ValueError("structured batches need at least two relations")
        entity_count = batch_facts // 2
        if len(self.entities) < entity_count:
            raise ValueError("too few entities for the requested batch")
        for entity in self.entities:
            available = {relation for current, relation in grouped if current == entity}
            if set(self.relations).difference(available):
                raise ValueError("every entity must expose the same relation set")

    def sample_indices(self) -> list[int]:
        entity_count = self.batch_facts // 2
        entities = self.rng.choice(self.entities, size=entity_count, replace=False)
        relations = self.rng.choice(self.relations, size=2, replace=False)
        indices: list[int] = []
        for entity in entities:
            for relation in relations:
                candidates = self.by_entity_relation[(str(entity), str(relation))]
                selected = self.rng.choice(
                    candidates, size=self.templates_per_fact, replace=False
                )
                indices.extend(int(value) for value in selected)
        return indices


def save_canonicalizer_checkpoint(
    path: str | Path,
    system: CanonicalizerSystem,
    *,
    variant: str,
    selected_core_layers: Sequence[int],
    labels: dict[str, Sequence[str]],
    source_core_sha256: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CANONICALIZER_FORMAT_VERSION,
        "variant": variant,
        "canonicalizer_config": asdict(system.config),
        "selected_core_layers": [int(value) for value in selected_core_layers],
        "pool_names": list(POOL_NAMES),
        "labels": {name: list(values) for name, values in labels.items()},
        "source_core_sha256": source_core_sha256,
        "model": {
            name: value.detach().cpu().contiguous()
            for name, value in system.state_dict().items()
        },
    }
    torch.save(payload, target)
    return target


def load_canonicalizer_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[CanonicalizerSystem, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if int(payload.get("format_version", -1)) != CANONICALIZER_FORMAT_VERSION:
        raise ValueError("unsupported canonicalizer checkpoint format")
    config = CanonicalizerConfig(**payload["canonicalizer_config"])
    system = CanonicalizerSystem(config)
    system.load_state_dict(payload["model"], strict=True)
    return system.to(device), payload
