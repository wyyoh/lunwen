from __future__ import annotations

import csv
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_clean_checkpoint, save_clean_checkpoint
from .config import GramModelConfig
from .data import TokenEpochSampler, load_story_data
from .keying import (
    AuxPermutationKey,
    apply_key,
    generate_key,
    hash_auxiliary,
    partial_key,
    validate_key_labels,
)
from .model import GramTransformer
from .train import resolve_device, resolve_dtype, set_seed


def token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_prob = F.softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1).mean()


def _set_trainable_scope(model: GramTransformer, scope: str, expert_index: int) -> list:
    marker = f".moe.experts.{expert_index}."
    selected = []
    for name, parameter in model.named_parameters():
        enabled = scope == "full" or marker in name
        if scope == "aux_lm_head" and name.startswith("lm_head."):
            enabled = True
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"trainable scope {scope!r} selected no parameters")
    return selected


def run_key_aware_training(
    locked_checkpoint: str | Path,
    key_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    private_label: str = "alien-encounters",
    steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    lambda_lock: float = 1.0,
    lambda_retain: float = 1.0,
    partial_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
    wrong_key_seed_start: int = 1000,
    trainable_scope: str = "aux",
    seed: int = 0,
    device_name: str | None = None,
) -> dict[str, Any]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    payload = load_clean_checkpoint(locked_checkpoint)
    config = GramModelConfig(**payload["model_config"])
    correct_key = AuxPermutationKey.load(key_path)
    validate_key_labels(correct_key, list(payload["labels"]))
    if private_label.strip().lower().replace(" ", "-") != correct_key.capability_label:
        raise ValueError("training private label does not match key capability")
    device = resolve_device(device_name)
    dtype = resolve_dtype("bfloat16", device)
    bundle = load_story_data(data_dir, private_label, config.context_length, 1.0, seed)

    student = GramTransformer(config).to(device=device, dtype=dtype)
    student.load_state_dict(payload["model"])
    teacher = GramTransformer(config).to(device=device, dtype=dtype).eval()
    teacher.load_state_dict(payload["model"])
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable = _set_trainable_scope(student, trainable_scope, correct_key.expert_index)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    wrong_keys = [
        generate_key(
            payload["model"],
            private_label,
            expert_index=correct_key.expert_index,
            group_size=correct_key.group_size,
            seed=wrong_key_seed_start + offset,
        )
        for offset in range(20)
    ]
    partial_keys = [
        partial_key(correct_key, fraction, 2000 + offset)
        for fraction in partial_fractions
        for offset in range(5)
    ]
    adversarial_keys = wrong_keys + partial_keys
    route_rng = random.Random(seed)
    data_rng = np.random.default_rng(seed)
    private_sampler = TokenEpochSampler(bundle.private_pool, data_rng)
    core_sampler = TokenEpochSampler(bundle.core_pool, data_rng)
    fwd_on = torch.tensor([1, 1], device=device)
    bck_aux = torch.tensor(
        [trainable_scope != "aux", 1], dtype=torch.bool, device=device
    )
    core_only = torch.tensor([1, 0], device=device)
    history: list[dict[str, float | int]] = []
    student.train()
    for step in range(steps):
        private_x, private_y = private_sampler.sample_batch(batch_size)
        core_x, _ = core_sampler.sample_batch(batch_size)
        private_x = private_x.to(device)
        private_y = private_y.to(device)
        core_x = core_x.to(device)
        sampled_key = adversarial_keys[route_rng.randrange(len(adversarial_keys))]
        optimizer.zero_grad(set_to_none=True)

        with torch.inference_mode():
            teacher_private = teacher(
                private_x, fwd_mask=core_only, bck_mask=core_only
            )[0]
            teacher_core = teacher(core_x, fwd_mask=core_only, bck_mask=core_only)[0]

        _, private_loss = student(
            private_x,
            labels=private_y,
            fwd_mask=fwd_on,
            bck_mask=bck_aux,
            key=correct_key,
        )
        assert private_loss is not None
        private_loss.backward()

        wrong_private = student(
            private_x,
            fwd_mask=fwd_on,
            bck_mask=bck_aux,
            key=sampled_key,
        )[0]
        lock_loss = token_kl(wrong_private, teacher_private)
        (lambda_lock * lock_loss).backward()

        correct_core = student(
            core_x,
            fwd_mask=fwd_on,
            bck_mask=bck_aux,
            key=correct_key,
        )[0]
        retain_correct = token_kl(correct_core, teacher_core)
        (0.5 * lambda_retain * retain_correct).backward()

        wrong_core = student(
            core_x,
            fwd_mask=fwd_on,
            bck_mask=bck_aux,
            key=sampled_key,
        )[0]
        retain_wrong = token_kl(wrong_core, teacher_core)
        (0.5 * lambda_retain * retain_wrong).backward()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            history.append(
                {
                    "step": step + 1,
                    "private_loss": float(private_loss.detach().cpu()),
                    "lock_loss": float(lock_loss.detach().cpu()),
                    "retain_correct_loss": float(retain_correct.detach().cpu()),
                    "retain_wrong_loss": float(retain_wrong.detach().cpu()),
                }
            )

    clean_state = {name: value.detach().cpu() for name, value in student.state_dict().items()}
    hardened_unlocked = apply_key(clean_state, correct_key)
    hardened_key = replace(
        correct_key,
        original_aux_sha256=hash_auxiliary(hardened_unlocked, correct_key.expert_index),
    )
    checkpoint_path = output_dir / "gram_key_aware_locked.pt"
    save_clean_checkpoint(
        checkpoint_path,
        None,
        config,
        list(payload["labels"]),
        state_dict=clean_state,
    )
    (output_dir / "key_aware_run.json").write_text(
        json.dumps(
            {
                "method": "key-aware-gram",
                "seed": seed,
                "steps": steps,
                "lambda_lock": lambda_lock,
                "lambda_retain": lambda_retain,
                "trainable_scope": trainable_scope,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    hardened_key.save(output_dir / "hardened_private_key.json")
    (output_dir / "key_aware_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if history:
        with (output_dir / "key_aware_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return {
        "checkpoint": str(checkpoint_path),
        "key": str(output_dir / "hardened_private_key.json"),
        "history": history,
    }
