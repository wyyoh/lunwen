from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm.auto import tqdm

from .checkpoint import build_model_from_checkpoint, load_clean_checkpoint, save_clean_checkpoint
from .config import GramModelConfig
from .data import TokenEpochSampler, TokenStreamPool, discover_labels, iter_eval_batches
from .keying import hash_auxiliary, hash_non_auxiliary
from .model import GramTransformer
from .phase_a import (
    FACT_FIELDS,
    TRAIN_PROMPTS,
    _generate_answers,
    _load_tokenizer,
    _read_jsonl,
    _reset_auxiliary,
    _score_candidates,
    _select_aux_parameters,
    _sha256_file,
    _summarize_fact_results,
    _write_jsonl,
    make_atomic_token_biographies,
)
from .train import collect_environment, resolve_device, resolve_dtype, set_seed


STAGE_B_SCHEMA_VERSION = 1
DEFAULT_PRIVATE_LABEL = "synthetic-biographies"

VALIDATION_PROMPTS = {
    "registry_id": (
        "Identify the registry number of {entity}. Reply:",
        "Registry lookup for {entity}; give the stored number:",
    ),
    "city_code": (
        "Identify the city code of {entity}. Reply:",
        "City lookup for {entity}; give the stored code:",
    ),
    "access_code": (
        "Identify the access code of {entity}. Reply:",
        "Access lookup for {entity}; give the stored code:",
    ),
}

TEST_PROMPTS = {
    "registry_id": (
        "What number is {entity} registered under? Response:",
        "Tell me the registry identifier associated with {entity}. Answer:",
        "Find {entity} in the registry and return its number:",
        "The registered number of {entity} is what? Value:",
    ),
    "city_code": (
        "What location code is {entity} assigned to? Response:",
        "Tell me the city identifier associated with {entity}. Answer:",
        "Find {entity} and return its assigned city code:",
        "The city code of {entity} is what? Value:",
    ),
    "access_code": (
        "What private credential is {entity} assigned? Response:",
        "Tell me the access identifier associated with {entity}. Answer:",
        "Find {entity} and return its private access code:",
        "The access code of {entity} is what? Value:",
    ),
}

CONTEXT_PROMPTS = {
    "registry_id": (
        "Reference note: {entity} has registry number {answer}. "
        "Using only that note, what registry identifier belongs to {entity}? Answer:"
    ),
    "city_code": (
        "Reference note: {entity} has city code {answer}. "
        "Using only that note, what location code belongs to {entity}? Answer:"
    ),
    "access_code": (
        "Reference note: {entity} has access code {answer}. "
        "Using only that note, what private credential belongs to {entity}? Answer:"
    ),
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    residual_scaling: bool
    lambda_para: float
    lambda_contrast: float
    lambda_public_kl: float
    lambda_public_residual: float


DEFAULT_VARIANTS = (
    VariantConfig("B0", False, 0.0, 0.0, 0.0, 0.0),
    VariantConfig("B1", True, 0.0, 0.0, 1.0, 0.1),
    VariantConfig("B2", True, 0.5, 0.0, 1.0, 0.1),
    VariantConfig("B3", True, 0.5, 0.1, 1.0, 0.1),
)


def _candidate_values(
    records,
    record_index: int,
    field: str,
    candidate_count: int,
    rng,
) -> tuple[list[str], int]:
    answer = records[record_index].fact(field)
    pool = sorted({record.fact(field) for record in records if record.fact(field) != answer})
    if candidate_count < 2 or candidate_count > len(pool) + 1:
        raise ValueError("candidate_count must be between 2 and the number of entities")
    candidates = rng.sample(pool, candidate_count - 1) + [answer]
    rng.shuffle(candidates)
    return candidates, candidates.index(answer)


def build_stage_b_rows(
    records,
    *,
    candidate_count: int = 8,
    seed: int = 17,
) -> dict[str, list[dict[str, Any]]]:
    """Create the four evaluation grids and globally shared template split."""

    import random

    rng = random.Random(seed + 31)
    output: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "context": [],
        "unexposed": [],
    }
    for record_index, record in enumerate(records):
        for field in FACT_FIELDS:
            answer = record.fact(field)
            candidates, answer_index = _candidate_values(
                records, record_index, field, candidate_count, rng
            )
            base = {
                "fact_id": f"{record.entity}|{field}",
                "entity": record.entity,
                "attribute": field,
                "answer": answer,
                "candidates": candidates,
                "answer_index": answer_index,
            }
            if record.split == "memorized":
                for index, template in enumerate(TRAIN_PROMPTS[field]):
                    output["train"].append(
                        {
                            **base,
                            "entity_exposure": "seen",
                            "template_split": "train",
                            "template_id": f"train-{index}",
                            "prompt": template.format(entity=record.entity),
                        }
                    )
                for index, template in enumerate(VALIDATION_PROMPTS[field]):
                    output["validation"].append(
                        {
                            **base,
                            "entity_exposure": "seen",
                            "template_split": "validation",
                            "template_id": f"validation-{index}",
                            "prompt": template.format(entity=record.entity),
                        }
                    )
                for index, template in enumerate(TEST_PROMPTS[field]):
                    output["test"].append(
                        {
                            **base,
                            "entity_exposure": "seen",
                            "template_split": "test",
                            "template_id": f"test-{index}",
                            "prompt": template.format(entity=record.entity),
                        }
                    )
            else:
                context_prompt = CONTEXT_PROMPTS[field].format(
                    entity=record.entity, answer=answer
                )
                output["context"].append(
                    {
                        **base,
                        "entity_exposure": "context_only",
                        "template_split": "context",
                        "template_id": "context-0",
                        "prompt": context_prompt,
                    }
                )
                output["unexposed"].append(
                    {
                        **base,
                        "entity_exposure": "never_seen",
                        "template_split": "unexposed",
                        "template_id": "unexposed-0",
                        "prompt": TEST_PROMPTS[field][0].format(entity=record.entity),
                    }
                )
    return output


def prepare_stage_b_data(
    destination: str | Path,
    *,
    num_train_entities: int = 16,
    num_heldout_entities: int = 8,
    candidate_count: int = 8,
    seed: int = 17,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
) -> dict[str, Any]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(tokenizer_name)
    records = make_atomic_token_biographies(
        tokenizer,
        num_train_entities=num_train_entities,
        num_heldout_entities=num_heldout_entities,
        seed=seed,
    )
    splits = build_stage_b_rows(records, candidate_count=candidate_count, seed=seed)
    files = {}
    hashes = {}
    for split, rows in splits.items():
        path = destination / f"{split}.jsonl"
        _write_jsonl(path, rows)
        files[split] = path.name
        hashes[split] = _sha256_file(path)
    manifest = {
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "task": "paraphrase-invariant-private-residualization",
        "fact_encoding": "atomic_tokens",
        "seed": seed,
        "tokenizer": tokenizer_name,
        "num_train_entities": num_train_entities,
        "num_heldout_entities": num_heldout_entities,
        "candidate_count": candidate_count,
        "random_candidate_accuracy": 1.0 / candidate_count,
        "fact_fields": list(FACT_FIELDS),
        "template_counts": {
            "train": len(next(iter(TRAIN_PROMPTS.values()))),
            "validation": len(next(iter(VALIDATION_PROMPTS.values()))),
            "test": len(next(iter(TEST_PROMPTS.values()))),
            "context": 1,
            "unexposed": 1,
        },
        "row_counts": {split: len(rows) for split, rows in splits.items()},
        "files": files,
        "sha256": hashes,
        "records": [record.__dict__ for record in records],
    }
    manifest_path = destination / "stage_b_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


@dataclass(frozen=True)
class EncodedFactPrompt:
    fact_id: str
    template_id: str
    input_ids: Tensor
    answer_id: int
    candidate_ids: Tensor


def _encode_fact_rows(rows: Sequence[dict[str, Any]], tokenizer) -> dict[str, list[EncodedFactPrompt]]:
    groups: dict[str, list[EncodedFactPrompt]] = defaultdict(list)
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer_ids = tokenizer.encode(str(row["answer"]), add_special_tokens=False)
        candidate_ids = [
            tokenizer.encode(str(value), add_special_tokens=False)
            for value in row["candidates"]
        ]
        if len(answer_ids) != 1 or any(len(value) != 1 for value in candidate_ids):
            raise ValueError("Stage B requires atomic one-token facts and candidates")
        groups[row["fact_id"]].append(
            EncodedFactPrompt(
                fact_id=row["fact_id"],
                template_id=row["template_id"],
                input_ids=torch.tensor(prompt_ids, dtype=torch.long),
                answer_id=int(answer_ids[0]),
                candidate_ids=torch.tensor(
                    [value[0] for value in candidate_ids], dtype=torch.long
                ),
            )
        )
    if not groups or any(len(values) < 2 for values in groups.values()):
        raise ValueError("every Stage-B fact needs at least two training templates")
    return dict(groups)


class FactPairSampler:
    def __init__(self, groups: dict[str, list[EncodedFactPrompt]], seed: int, eos: int):
        self.groups = groups
        self.fact_ids = sorted(groups)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.fact_ids))
        self.cursor = 0
        self.eos = eos

    def _fact_indices(self, count: int) -> list[int]:
        output: list[int] = []
        while len(output) < count:
            available = len(self.order) - self.cursor
            take = min(count - len(output), available)
            output.extend(int(value) for value in self.order[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.fact_ids))
                self.cursor = 0
        return output

    def _pad(self, values: Sequence[EncodedFactPrompt]) -> dict[str, Tensor]:
        width = max(value.input_ids.numel() for value in values)
        x = torch.full((len(values), width), self.eos, dtype=torch.long)
        lengths = torch.empty(len(values), dtype=torch.long)
        answers = torch.empty(len(values), dtype=torch.long)
        candidates = torch.stack([value.candidate_ids for value in values])
        for index, value in enumerate(values):
            length = value.input_ids.numel()
            x[index, :length] = value.input_ids
            lengths[index] = length
            answers[index] = value.answer_id
        return {
            "input_ids": x,
            "lengths": lengths,
            "answers": answers,
            "candidates": candidates,
        }

    def sample_batch(self, batch_size: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        left: list[EncodedFactPrompt] = []
        right: list[EncodedFactPrompt] = []
        for index in self._fact_indices(batch_size):
            choices = self.groups[self.fact_ids[index]]
            pair = self.rng.choice(len(choices), size=2, replace=False)
            left.append(choices[int(pair[0])])
            right.append(choices[int(pair[1])])
        return self._pad(left), self._pad(right)


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _answer_logits(logits: Tensor, lengths: Tensor) -> Tensor:
    indices = torch.arange(logits.size(0), device=logits.device)
    return logits[indices, lengths - 1]


def symmetric_candidate_kl(
    left_logits: Tensor, right_logits: Tensor, candidate_ids: Tensor
) -> Tensor:
    left = left_logits.gather(1, candidate_ids).float()
    right = right_logits.gather(1, candidate_ids).float()
    left_log = F.log_softmax(left, dim=-1)
    right_log = F.log_softmax(right, dim=-1)
    left_prob = left_log.exp()
    right_prob = right_log.exp()
    return 0.5 * (
        F.kl_div(left_log, right_prob, reduction="batchmean")
        + F.kl_div(right_log, left_prob, reduction="batchmean")
    )


def residual_infonce(
    left_residuals: Sequence[Tensor],
    right_residuals: Sequence[Tensor],
    left_lengths: Tensor,
    right_lengths: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    left_parts = []
    right_parts = []
    left_indices = torch.arange(left_lengths.numel(), device=left_lengths.device)
    right_indices = torch.arange(right_lengths.numel(), device=right_lengths.device)
    for left, right in zip(left_residuals, right_residuals):
        left_parts.append(left[left_indices, left_lengths - 1])
        right_parts.append(right[right_indices, right_lengths - 1])
    left_repr = F.normalize(torch.cat(left_parts, dim=-1).float(), dim=-1, eps=1e-8)
    right_repr = F.normalize(torch.cat(right_parts, dim=-1).float(), dim=-1, eps=1e-8)
    similarities = left_repr @ right_repr.t() / temperature
    target = torch.arange(similarities.size(0), device=similarities.device)
    return 0.5 * (
        F.cross_entropy(similarities, target)
        + F.cross_entropy(similarities.t(), target)
    )


def public_residual_null_loss(diagnostics: dict[str, list[Tensor]]) -> Tensor:
    ratios = []
    for hidden, residual in zip(
        diagnostics["aux_inputs"], diagnostics["aux_residuals"]
    ):
        numerator = residual.float().pow(2).sum(dim=-1)
        denominator = hidden.float().pow(2).sum(dim=-1).clamp_min(1e-8)
        ratios.append((numerator / denominator).mean())
    if not ratios:
        raise ValueError("residual diagnostics are empty")
    return torch.stack(ratios).mean()


def _teacher_student_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    teacher_prob = F.softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1).mean()


def _gradient_snapshot(parameters: Sequence[Tensor]) -> list[Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def gradient_cosine(
    left: Sequence[Tensor | None], parameters: Sequence[Tensor]
) -> float | None:
    dot = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    left_norm = torch.zeros_like(dot)
    right_norm = torch.zeros_like(dot)
    used = False
    for saved, parameter in zip(left, parameters):
        if saved is None or parameter.grad is None:
            continue
        current = parameter.grad.detach()
        dot += (saved.float() * current.float()).sum()
        left_norm += saved.float().pow(2).sum()
        right_norm += current.float().pow(2).sum()
        used = True
    if not used or left_norm <= 0 or right_norm <= 0:
        return None
    return float((dot / (left_norm.sqrt() * right_norm.sqrt())).detach().cpu())


def _reset_residual_auxiliary(model: GramTransformer, expert_index: int, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for block in model.blocks:
            expert = block.moe.experts[expert_index]
            for parameter in (expert.c_fc.weight, expert.c_proj.weight):
                values = torch.randn(
                    parameter.shape, generator=generator, dtype=torch.float32
                ) * 0.02
                parameter.copy_(values.to(parameter.dtype))
            expert.c_fc.bias.zero_()
            expert.c_proj.bias.zero_()


def _build_stage_b_model(
    base_checkpoint: str | Path,
    variant: VariantConfig,
    *,
    scale_init: float,
    expert_index: int,
    seed: int,
) -> tuple[GramTransformer, dict[str, Any]]:
    payload = load_clean_checkpoint(base_checkpoint)
    base_config = GramModelConfig(**payload["model_config"])
    config = replace(
        base_config,
        learnable_aux_scale=variant.residual_scaling,
        aux_scale_init=scale_init if variant.residual_scaling else 1.0,
        aux_input_stop_gradient=variant.residual_scaling,
    )
    model = GramTransformer(config)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    expected_missing = ["aux_scales"] if variant.residual_scaling else []
    if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"base checkpoint is incompatible with Stage B: {incompatible}"
        )
    if variant.residual_scaling:
        _reset_residual_auxiliary(model, expert_index, seed + 101)
    else:
        _reset_auxiliary(model, expert_index, seed + 101)
    return model, payload


def _linear_lr(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return max((total_steps - step) / max(total_steps - warmup, 1), 1e-8)


def train_stage_b_variant(
    base_checkpoint: str | Path,
    train_jsonl: str | Path,
    public_data_dir: str | Path,
    output_dir: str | Path,
    variant: VariantConfig,
    *,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    private_label: str = DEFAULT_PRIVATE_LABEL,
    public_labels: Sequence[str] = (),
    cycles: int = 1500,
    private_batch_size: int = 8,
    public_batch_size: int = 8,
    learning_rate: float = 2e-3,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.05,
    public_train_fraction: float = 0.01,
    scale_init: float = 0.01,
    contrastive_temperature: float = 0.1,
    expert_index: int = 1,
    seed: int = 0,
    dtype_name: str = "bfloat16",
    device_name: str | None = None,
) -> dict[str, Any]:
    if cycles <= 0 or private_batch_size <= 0 or public_batch_size <= 0:
        raise ValueError("cycles and batch sizes must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name, device)
    model, payload = _build_stage_b_model(
        base_checkpoint,
        variant,
        scale_init=scale_init,
        expert_index=expert_index,
        seed=seed,
    )
    trainable = _select_aux_parameters(model, expert_index)
    model = model.to(device=device, dtype=dtype)
    core_hash_before = hash_non_auxiliary(model.state_dict(), expert_index)
    aux_hash_before = hash_auxiliary(model.state_dict(), expert_index)

    tokenizer = _load_tokenizer(tokenizer_name)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the base checkpoint")
    groups = _encode_fact_rows(_read_jsonl(train_jsonl), tokenizer)
    private_sampler = FactPairSampler(
        groups, seed=seed + 1, eos=int(tokenizer.eos_token_id)
    )
    available_public = discover_labels(public_data_dir)
    selected_public = list(public_labels) or available_public[:8]
    missing = [label for label in selected_public if label not in available_public]
    if missing:
        raise ValueError(f"unknown public labels: {missing}")
    public_pool = TokenStreamPool(
        tuple(Path(public_data_dir) / f"{label}_train.bin" for label in selected_public),
        model.config.context_length,
        public_train_fraction,
        seed + 2,
    )
    public_sampler = TokenEpochSampler(public_pool, np.random.default_rng(seed + 2))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    full_mask = torch.tensor([1, 1], dtype=torch.bool, device=device)
    aux_grad_mask = torch.tensor([0, 1], dtype=torch.bool, device=device)
    core_only = torch.tensor([1, 0], dtype=torch.bool, device=device)
    has_public_step = bool(variant.lambda_public_kl or variant.lambda_public_residual)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    progress = tqdm(range(cycles), desc=f"stage-b:{variant.name}")
    for cycle in progress:
        lr = learning_rate * _linear_lr(cycle, cycles, warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr

        left, right = private_sampler.sample_batch(private_batch_size)
        left = _move_batch(left, device)
        right = _move_batch(right, device)
        optimizer.zero_grad(set_to_none=True)
        need_diagnostics = bool(variant.lambda_contrast)
        left_output = model(
            left["input_ids"],
            fwd_mask=full_mask,
            bck_mask=aux_grad_mask,
            return_diagnostics=need_diagnostics,
        )
        right_output = model(
            right["input_ids"],
            fwd_mask=full_mask,
            bck_mask=aux_grad_mask,
            return_diagnostics=need_diagnostics,
        )
        left_logits = _answer_logits(left_output[0], left["lengths"])
        right_logits = _answer_logits(right_output[0], right["lengths"])
        private_ce = 0.5 * (
            F.cross_entropy(left_logits.float(), left["answers"])
            + F.cross_entropy(right_logits.float(), right["answers"])
        )
        para_loss = symmetric_candidate_kl(
            left_logits, right_logits, left["candidates"]
        )
        contrast_loss = private_ce.new_zeros(())
        if variant.lambda_contrast:
            contrast_loss = residual_infonce(
                left_output[2]["aux_residuals"],
                right_output[2]["aux_residuals"],
                left["lengths"],
                right["lengths"],
                temperature=contrastive_temperature,
            )
        private_objective = (
            private_ce
            + variant.lambda_para * para_loss
            + variant.lambda_contrast * contrast_loss
        )
        private_objective.backward()
        private_gradients = _gradient_snapshot(trainable) if has_public_step else []
        private_grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        public_kl = private_ce.new_zeros(())
        public_residual = private_ce.new_zeros(())
        grad_cosine = None
        if has_public_step:
            public_x, _ = public_sampler.sample_batch(public_batch_size)
            public_x = public_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = model(
                    public_x, fwd_mask=core_only, bck_mask=core_only
                )[0]
            full_output = model(
                public_x,
                fwd_mask=full_mask,
                bck_mask=aux_grad_mask,
                return_diagnostics=True,
            )
            public_kl = _teacher_student_kl(full_output[0], teacher_logits)
            public_residual = public_residual_null_loss(full_output[2])
            public_objective = (
                variant.lambda_public_kl * public_kl
                + variant.lambda_public_residual * public_residual
            )
            public_objective.backward()
            grad_cosine = gradient_cosine(private_gradients, trainable)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        progress.set_postfix(
            private=f"{float(private_ce.detach().cpu()):.3f}",
            public=f"{float(public_kl.detach().cpu()):.3f}",
        )
        if cycle == 0 or (cycle + 1) % 25 == 0 or cycle + 1 == cycles:
            history.append(
                {
                    "cycle": cycle + 1,
                    "private_ce": float(private_ce.detach().cpu()),
                    "paraphrase_kl": float(para_loss.detach().cpu()),
                    "contrastive_loss": float(contrast_loss.detach().cpu()),
                    "public_kl": float(public_kl.detach().cpu()),
                    "public_residual_loss": float(public_residual.detach().cpu()),
                    "private_gradient_norm": float(private_grad_norm.detach().cpu()),
                    "private_public_gradient_cosine": grad_cosine,
                    "learning_rate": lr,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    core_hash_after = hash_non_auxiliary(state, expert_index)
    aux_hash_after = hash_auxiliary(state, expert_index)
    if core_hash_after != core_hash_before:
        raise RuntimeError(f"{variant.name} modified non-auxiliary parameters")
    checkpoint_path = output_dir / f"stage_b_{variant.name.lower()}.pt"
    labels = list(payload.get("labels", ["core", private_label]))
    labels[expert_index] = private_label
    save_clean_checkpoint(
        checkpoint_path,
        None,
        model.config,
        labels,
        state_dict=state,
    )
    valid_cosines = [
        row["private_public_gradient_cosine"]
        for row in history
        if row["private_public_gradient_cosine"] is not None
    ]
    run = {
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "variant": variant.__dict__,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_checkpoint_sha256": _sha256_file(Path(base_checkpoint)),
        "private_train_jsonl": str(Path(train_jsonl).resolve()),
        "private_train_sha256": _sha256_file(Path(train_jsonl)),
        "public_data_dir": str(Path(public_data_dir).resolve()),
        "public_labels": selected_public,
        "cycles": cycles,
        "seed": seed,
        "learning_rate": learning_rate,
        "scale_init": scale_init,
        "contrastive_temperature": contrastive_temperature,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "core_hash_before": core_hash_before,
        "core_hash_after": core_hash_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "aux_hash_before": aux_hash_before,
        "aux_hash_after": aux_hash_after,
        "aux_changed": aux_hash_before != aux_hash_after,
        "mean_private_public_gradient_cosine": (
            statistics.fmean(valid_cosines) if valid_cosines else None
        ),
        "negative_gradient_cosine_fraction": (
            sum(value < 0 for value in valid_cosines) / len(valid_cosines)
            if valid_cosines
            else None
        ),
        "final_aux_scales": (
            state["aux_scales"].flatten().tolist() if "aux_scales" in state else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": collect_environment(device, dtype),
    }
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if history:
        with (output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return {"checkpoint": str(checkpoint_path), "run": run, "history": history}


def _template_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overall = _summarize_fact_results(results)
    by_template: dict[str, dict[str, Any]] = {}
    for template_id in sorted({row["template_id"] for row in results}):
        by_template[template_id] = _summarize_fact_results(
            [row for row in results if row["template_id"] == template_id]
        )
    accuracies = [
        float(value["candidate_accuracy"])
        for value in by_template.values()
        if value["candidate_accuracy"] is not None
    ]
    by_fact: dict[str, list[str]] = defaultdict(list)
    for row in results:
        by_fact[row["fact_id"]].append(str(row["candidate_prediction"]))
    agreed = 0
    pairs = 0
    all_templates_correct = 0
    for fact_id, predictions in by_fact.items():
        for left, right in itertools.combinations(predictions, 2):
            agreed += int(left == right)
            pairs += 1
        fact_rows = [row for row in results if row["fact_id"] == fact_id]
        all_templates_correct += int(all(bool(row["candidate_correct"]) for row in fact_rows))
    return {
        **overall,
        "by_template": by_template,
        "worst_template_accuracy": min(accuracies) if accuracies else None,
        "template_accuracy_std": (
            statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
        ),
        "answer_agreement": agreed / pairs if pairs else None,
        "all_templates_correct_fraction": (
            all_templates_correct / len(by_fact) if by_fact else None
        ),
    }


@torch.inference_mode()
def evaluate_public_behavior(
    model: GramTransformer,
    public_data_dir: str | Path,
    public_labels: Sequence[str],
    *,
    sequences_per_label: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    full_mask = torch.tensor([1, 1], device=device)
    core_mask = torch.tensor([1, 0], device=device)
    full_nll_sum = 0.0
    core_nll_sum = 0.0
    kl_sum = 0.0
    agreement = 0
    tokens = 0
    per_label = {}
    for label_index, label in enumerate(public_labels):
        label_full = 0.0
        label_core = 0.0
        label_tokens = 0
        for x, y in iter_eval_batches(
            Path(public_data_dir) / f"{label}_test.bin",
            model.config.context_length,
            sequences_per_label,
            batch_size,
            seed=123 + label_index,
        ):
            x = x.to(device)
            y = y.to(device)
            core_logits = model(x, fwd_mask=core_mask, bck_mask=core_mask)[0].float()
            full_logits = model(x, fwd_mask=full_mask, bck_mask=full_mask)[0].float()
            count = y.numel()
            core_loss = F.cross_entropy(
                core_logits.reshape(-1, core_logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            full_loss = F.cross_entropy(
                full_logits.reshape(-1, full_logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            core_log = F.log_softmax(core_logits, dim=-1)
            full_log = F.log_softmax(full_logits, dim=-1)
            batch_kl = F.kl_div(
                full_log, core_log.exp(), reduction="none"
            ).sum(dim=-1)
            full_nll_sum += float(full_loss.cpu())
            core_nll_sum += float(core_loss.cpu())
            kl_sum += float(batch_kl.sum().cpu())
            agreement += int(full_logits.argmax(dim=-1).eq(core_logits.argmax(dim=-1)).sum().cpu())
            tokens += count
            label_full += float(full_loss.cpu())
            label_core += float(core_loss.cpu())
            label_tokens += count
        per_label[label] = {
            "full_nll": label_full / label_tokens,
            "core_nll": label_core / label_tokens,
            "absolute_nll_difference": (label_full - label_core) / label_tokens,
        }
    full_nll = full_nll_sum / tokens
    core_nll = core_nll_sum / tokens
    return {
        "tokens": tokens,
        "full_nll": full_nll,
        "core_nll": core_nll,
        "absolute_nll_difference": full_nll - core_nll,
        "relative_nll_increase": (full_nll - core_nll) / core_nll,
        "full_perplexity": math.exp(min(full_nll, 80.0)),
        "core_perplexity": math.exp(min(core_nll, 80.0)),
        "perplexity_ratio": math.exp(min(full_nll - core_nll, 80.0)),
        "core_to_full_kl": kl_sum / tokens,
        "top1_token_agreement": agreement / tokens,
        "per_label": per_label,
    }


def _prompt_batch(rows: Sequence[dict[str, Any]], tokenizer, eos: int) -> tuple[Tensor, Tensor]:
    encoded = [tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows]
    width = max(len(value) for value in encoded)
    x = torch.full((len(rows), width), eos, dtype=torch.long)
    lengths = torch.tensor([len(value) for value in encoded], dtype=torch.long)
    for index, values in enumerate(encoded):
        x[index, : len(values)] = torch.tensor(values, dtype=torch.long)
    return x, lengths


@torch.inference_mode()
def evaluate_residual_selectivity(
    model: GramTransformer,
    tokenizer,
    private_rows: Sequence[dict[str, Any]],
    public_data_dir: str | Path,
    public_labels: Sequence[str],
    *,
    private_batch_size: int,
    public_batch_size: int,
    public_sequences_per_label: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    mask = torch.tensor([1, 1], device=device)
    private_sums = [0.0] * model.config.num_layers
    private_counts = [0] * model.config.num_layers
    eos = int(tokenizer.eos_token_id)
    for offset in range(0, len(private_rows), private_batch_size):
        rows = private_rows[offset : offset + private_batch_size]
        x, lengths = _prompt_batch(rows, tokenizer, eos)
        x = x.to(device)
        lengths = lengths.to(device)
        output = model(
            x, fwd_mask=mask, bck_mask=mask, return_diagnostics=True
        )
        indices = torch.arange(x.size(0), device=device)
        for layer, (hidden, residual) in enumerate(
            zip(output[2]["aux_inputs"], output[2]["aux_residuals"])
        ):
            h = hidden[indices, lengths - 1].float()
            r = residual[indices, lengths - 1].float()
            ratios = r.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-8)
            private_sums[layer] += float(ratios.sum().cpu())
            private_counts[layer] += ratios.numel()

    public_sums = [0.0] * model.config.num_layers
    public_counts = [0] * model.config.num_layers
    for label_index, label in enumerate(public_labels):
        for x, _ in iter_eval_batches(
            Path(public_data_dir) / f"{label}_test.bin",
            model.config.context_length,
            public_sequences_per_label,
            public_batch_size,
            seed=321 + label_index,
        ):
            x = x.to(device)
            output = model(
                x, fwd_mask=mask, bck_mask=mask, return_diagnostics=True
            )
            for layer, (hidden, residual) in enumerate(
                zip(output[2]["aux_inputs"], output[2]["aux_residuals"])
            ):
                ratios = residual.float().norm(dim=-1) / hidden.float().norm(
                    dim=-1
                ).clamp_min(1e-8)
                public_sums[layer] += float(ratios.sum().cpu())
                public_counts[layer] += ratios.numel()
    layers = []
    for layer in range(model.config.num_layers):
        private_ratio = private_sums[layer] / max(private_counts[layer], 1)
        public_ratio = public_sums[layer] / max(public_counts[layer], 1)
        layers.append(
            {
                "layer": layer,
                "private_residual_ratio": private_ratio,
                "public_residual_ratio": public_ratio,
                "selectivity": private_ratio / max(public_ratio, 1e-12),
            }
        )
    return {
        "layers": layers,
        "mean_private_residual_ratio": statistics.fmean(
            row["private_residual_ratio"] for row in layers
        ),
        "mean_public_residual_ratio": statistics.fmean(
            row["public_residual_ratio"] for row in layers
        ),
        "mean_selectivity": statistics.fmean(row["selectivity"] for row in layers),
        "min_selectivity": min(row["selectivity"] for row in layers),
    }


def localization_scores(full: float, core: float, chance: float) -> dict[str, float]:
    full_excess = max(full - chance, 0.0)
    core_excess = max(core - chance, 0.0)
    if full_excess <= 1e-12:
        loc_score = 0.0
    else:
        loc_score = 1.0 - min(core_excess / full_excess, 1.0)
    return {
        "access_gap": full - core,
        "localization_score": loc_score,
        "anti_signal": max(chance - core, 0.0),
    }


def evaluate_stage_b_variant(
    checkpoint: str | Path,
    data_dir: str | Path,
    public_data_dir: str | Path,
    output_dir: str | Path,
    *,
    public_labels: Sequence[str],
    core_unchanged: bool,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    batch_size: int = 64,
    generation_batch_size: int = 32,
    max_new_tokens: int = 8,
    public_sequences_per_label: int = 20,
    seen_accuracy_threshold: float = 0.95,
    unseen_accuracy_threshold: float = 0.80,
    answer_agreement_threshold: float = 0.90,
    core_chance_margin: float = 0.05,
    public_loss_threshold: float = 0.05,
    device_name: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = resolve_dtype("bfloat16", device)
    model, payload = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)
    model.eval()
    tokenizer = _load_tokenizer(tokenizer_name)
    split_rows = {
        split: _read_jsonl(Path(data_dir) / f"{split}.jsonl")
        for split in ("train", "validation", "test", "context", "unexposed")
    }
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for split, rows in split_rows.items():
        all_results[split] = {}
        metrics[split] = {}
        for view, aux_on in (("full", True), ("core_only", False)):
            scored = _score_candidates(
                model,
                tokenizer,
                rows,
                aux_on=aux_on,
                batch_size=batch_size,
                device=device,
            )
            if split == "test":
                generated = _generate_answers(
                    model,
                    tokenizer,
                    rows,
                    aux_on=aux_on,
                    batch_size=generation_batch_size,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                for index, values in generated.items():
                    scored[index].update(values)
            all_results[split][view] = scored
            metrics[split][view] = _template_metrics(scored)

    public = evaluate_public_behavior(
        model,
        public_data_dir,
        public_labels,
        sequences_per_label=public_sequences_per_label,
        batch_size=min(batch_size, 16),
        device=device,
    )
    residual = evaluate_residual_selectivity(
        model,
        tokenizer,
        split_rows["test"],
        public_data_dir,
        public_labels,
        private_batch_size=min(batch_size, 32),
        public_batch_size=min(batch_size, 8),
        public_sequences_per_label=min(public_sequences_per_label, 10),
        device=device,
    )
    chance = 1.0 / len(split_rows["test"][0]["candidates"])
    seen_accuracy = metrics["train"]["full"]["candidate_accuracy"]
    unseen_accuracy = metrics["test"]["full"]["candidate_accuracy"]
    core_accuracy = metrics["test"]["core_only"]["candidate_accuracy"]
    answer_agreement = metrics["test"]["full"]["answer_agreement"]
    functional = localization_scores(unseen_accuracy, core_accuracy, chance)
    gates = {
        "seen_full_accuracy": {
            "value": seen_accuracy,
            "threshold": seen_accuracy_threshold,
            "passed": seen_accuracy >= seen_accuracy_threshold,
        },
        "unseen_full_accuracy": {
            "value": unseen_accuracy,
            "threshold": unseen_accuracy_threshold,
            "passed": unseen_accuracy >= unseen_accuracy_threshold,
        },
        "answer_agreement": {
            "value": answer_agreement,
            "threshold": answer_agreement_threshold,
            "passed": answer_agreement >= answer_agreement_threshold,
        },
        "core_only_near_chance": {
            "value": core_accuracy,
            "threshold": chance + core_chance_margin,
            "passed": core_accuracy <= chance + core_chance_margin,
        },
        "public_loss_increase": {
            "value": public["relative_nll_increase"],
            "threshold": public_loss_threshold,
            "passed": public["relative_nll_increase"] <= public_loss_threshold,
        },
        "core_hash_unchanged": {
            "value": core_unchanged,
            "threshold": True,
            "passed": core_unchanged,
        },
    }
    mechanism_passed = all(value["passed"] for value in gates.values())
    summary = {
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "status": "passed" if mechanism_passed else "failed",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_labels": list(payload["labels"]),
        "random_candidate_accuracy": chance,
        "metrics": metrics,
        "functional_dependency": functional,
        "public": public,
        "residual_selectivity": residual,
        "gates": gates,
        "mechanism_passed": mechanism_passed,
        "security_matrix_run": False,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flat_rows = []
    for split, views in all_results.items():
        for view, results in views.items():
            for row in results:
                flat_rows.append(
                    {
                        "split": split,
                        "view": view,
                        "fact_id": row["fact_id"],
                        "entity": row["entity"],
                        "attribute": row["attribute"],
                        "template_id": row["template_id"],
                        "answer": row["answer"],
                        "candidate_prediction": row["candidate_prediction"],
                        "candidate_correct": row["candidate_correct"],
                        "answer_nll": row["answer_nll"],
                        "generated": row.get("generated", ""),
                        "exact_match": row.get("exact_match", ""),
                    }
                )
    if flat_rows:
        with (output_dir / "fact_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    return summary


def _parse_variants(raw: Any) -> tuple[VariantConfig, ...]:
    if not raw:
        return DEFAULT_VARIANTS
    variants = []
    for name, values in raw.items():
        variants.append(
            VariantConfig(
                name=str(name),
                residual_scaling=bool(values.get("residual_scaling", name != "B0")),
                lambda_para=float(values.get("lambda_para", 0.0)),
                lambda_contrast=float(values.get("lambda_contrast", 0.0)),
                lambda_public_kl=float(values.get("lambda_public_kl", 0.0)),
                lambda_public_residual=float(
                    values.get("lambda_public_residual", 0.0)
                ),
            )
        )
    return tuple(variants)


def run_stage_b(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    data = values.get("data", {})
    training = values.get("training", {})
    evaluation = values.get("evaluation", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data.get("destination", "data/stage_b"))
    public_data_dir = Path(data.get("public_data_dir", "data/stories"))
    tokenizer_name = str(
        data.get("tokenizer", "SimpleStories/SimpleStories-1.25M")
    )
    private_label = str(data.get("private_label", DEFAULT_PRIVATE_LABEL))
    public_labels = tuple(data.get("public_labels", ()))
    if not public_labels:
        public_labels = tuple(discover_labels(public_data_dir)[:8])
    manifest = prepare_stage_b_data(
        data_dir,
        num_train_entities=int(data.get("num_train_entities", 16)),
        num_heldout_entities=int(data.get("num_heldout_entities", 8)),
        candidate_count=int(data.get("candidate_count", 8)),
        seed=int(data.get("seed", 17)),
        tokenizer_name=tokenizer_name,
    )
    variants = _parse_variants(values.get("variants"))
    rows = []
    details = {}
    for variant in variants:
        variant_dir = output_dir / variant.name
        train_result = train_stage_b_variant(
            values.get("base_checkpoint", "artifacts/checkpoints/main/gram_original.pt"),
            data_dir / "train.jsonl",
            public_data_dir,
            variant_dir / "train",
            variant,
            tokenizer_name=tokenizer_name,
            private_label=private_label,
            public_labels=public_labels,
            cycles=int(training.get("cycles", 1500)),
            private_batch_size=int(training.get("private_batch_size", 8)),
            public_batch_size=int(training.get("public_batch_size", 8)),
            learning_rate=float(training.get("learning_rate", 2e-3)),
            weight_decay=float(training.get("weight_decay", 0.01)),
            warmup_fraction=float(training.get("warmup_fraction", 0.05)),
            public_train_fraction=float(training.get("public_train_fraction", 0.01)),
            scale_init=float(training.get("scale_init", 0.01)),
            contrastive_temperature=float(
                training.get("contrastive_temperature", 0.1)
            ),
            expert_index=int(training.get("expert_index", 1)),
            seed=int(training.get("seed", 0)),
            dtype_name=str(training.get("dtype", "bfloat16")),
            device_name=device_name,
        )
        result = evaluate_stage_b_variant(
            train_result["checkpoint"],
            data_dir,
            public_data_dir,
            variant_dir / "evaluation",
            public_labels=public_labels,
            core_unchanged=bool(train_result["run"]["core_unchanged"]),
            tokenizer_name=tokenizer_name,
            batch_size=int(evaluation.get("batch_size", 64)),
            generation_batch_size=int(
                evaluation.get("generation_batch_size", 32)
            ),
            max_new_tokens=int(evaluation.get("max_new_tokens", 8)),
            public_sequences_per_label=int(
                evaluation.get("public_sequences_per_label", 20)
            ),
            seen_accuracy_threshold=float(
                evaluation.get("seen_accuracy_threshold", 0.95)
            ),
            unseen_accuracy_threshold=float(
                evaluation.get("unseen_accuracy_threshold", 0.80)
            ),
            answer_agreement_threshold=float(
                evaluation.get("answer_agreement_threshold", 0.90)
            ),
            core_chance_margin=float(evaluation.get("core_chance_margin", 0.05)),
            public_loss_threshold=float(
                evaluation.get("public_loss_threshold", 0.05)
            ),
            device_name=device_name,
        )
        detail = {
            "variant": variant.__dict__,
            "train": train_result["run"],
            "evaluation": result,
        }
        details[variant.name] = detail
        rows.append(
            {
                "variant": variant.name,
                "mechanism_passed": result["mechanism_passed"],
                "seen_accuracy": result["metrics"]["train"]["full"][
                    "candidate_accuracy"
                ],
                "validation_accuracy": result["metrics"]["validation"]["full"][
                    "candidate_accuracy"
                ],
                "unseen_accuracy": result["metrics"]["test"]["full"][
                    "candidate_accuracy"
                ],
                "worst_test_template_accuracy": result["metrics"]["test"]["full"][
                    "worst_template_accuracy"
                ],
                "test_template_accuracy_std": result["metrics"]["test"]["full"][
                    "template_accuracy_std"
                ],
                "answer_agreement": result["metrics"]["test"]["full"][
                    "answer_agreement"
                ],
                "core_test_accuracy": result["metrics"]["test"]["core_only"][
                    "candidate_accuracy"
                ],
                "test_exact_match": result["metrics"]["test"]["full"][
                    "exact_match"
                ],
                "public_relative_nll_increase": result["public"][
                    "relative_nll_increase"
                ],
                "public_kl": result["public"]["core_to_full_kl"],
                "public_top1_agreement": result["public"]["top1_token_agreement"],
                "access_gap": result["functional_dependency"]["access_gap"],
                "localization_score": result["functional_dependency"][
                    "localization_score"
                ],
                "anti_signal": result["functional_dependency"]["anti_signal"],
                "mean_residual_selectivity": result["residual_selectivity"][
                    "mean_selectivity"
                ],
                "mean_gradient_cosine": train_result["run"][
                    "mean_private_public_gradient_cosine"
                ],
                "negative_gradient_cosine_fraction": train_result["run"][
                    "negative_gradient_cosine_fraction"
                ],
                "core_unchanged": train_result["run"]["core_unchanged"],
            }
        )
    best = max(
        rows,
        key=lambda row: (
            bool(row["mechanism_passed"]),
            row["public_relative_nll_increase"] <= float(
                evaluation.get("public_loss_threshold", 0.05)
            ),
            row["unseen_accuracy"],
            row["answer_agreement"],
        ),
    )
    passed = [row["variant"] for row in rows if row["mechanism_passed"]]
    summary = {
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "research_scope": "paraphrase-invariant-private-residualization-with-public-nulling",
        "security_matrix_run": False,
        "data_manifest": str((data_dir / "stage_b_manifest.json").resolve()),
        "data_manifest_sha256": manifest["manifest_sha256"],
        "passed_variants": passed,
        "best_variant": best["variant"],
        "ablation_rows": rows,
        "details": details,
    }
    (output_dir / "stage_b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "stage_b_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {"source_config": str(Path(config_path).resolve()), "values": values},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary
