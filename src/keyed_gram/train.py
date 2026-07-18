from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from .checkpoint import save_clean_checkpoint
from .config import ExperimentConfig
from .data import TokenEpochSampler, load_story_data
from .model import GramTransformer


UPSTREAM_COMMIT = "d07d62e9869c6a969a2306e755aba7a742c317fd"


def collect_environment(device: torch.device, dtype: torch.dtype) -> dict:
    packages: dict[str, str] = {}
    for package in (
        "torch",
        "numpy",
        "PyYAML",
        "tqdm",
        "huggingface_hub",
        "datasets",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "dtype": str(dtype),
        "gpu": gpu,
        "upstream_commit": UPSTREAM_COMMIT,
    }


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected GPU does not support bfloat16")
        return torch.bfloat16
    return torch.float32


def _lr_scale(step: int, total_steps: int, warmup_fraction: float, decay_fraction: float) -> float:
    warmup = max(1, round(total_steps * warmup_fraction))
    decay = max(1, round(total_steps * decay_fraction))
    constant_end = max(warmup, total_steps - decay)
    if step < warmup:
        return max((step + 1) / warmup, 1e-8)
    if step < constant_end:
        return 1.0
    progress = (step - constant_end) / max(total_steps - constant_end, 1)
    return max(1.0 - progress, 1e-8)


def _route_masks(
    side: str,
    rng: random.Random,
    aux_route_probability: float,
    core_robust_probability: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if side == "private":
        update_core = rng.random() < aux_route_probability
        return (
            torch.tensor([1, 1], device=device),
            torch.tensor([update_core, True], device=device),
        )
    robust = rng.random() < core_robust_probability
    if robust:
        return torch.tensor([1, 1], device=device), torch.tensor([1, 1], device=device)
    return torch.tensor([1, 0], device=device), torch.tensor([1, 0], device=device)


def train_gram(
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> Path:
    config.validate()
    if config.model.num_aux != 1:
        raise ValueError("the MVP training loop supports exactly one auxiliary expert")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.training.seed)
    device = resolve_device(device_name)
    dtype = resolve_dtype(config.training.dtype, device)
    environment = collect_environment(device, dtype)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bundle = load_story_data(
        config.data.data_dir,
        config.data.private_label,
        config.model.context_length,
        config.data.train_fraction,
        config.training.seed,
    )
    subset_manifest = {
        "sampling": "fixed-contiguous-subset/non-overlapping-windows/no-replacement",
        "context_length": config.model.context_length,
        "train_fraction": config.data.train_fraction,
        "private": [
            {
                "file": path.name,
                "start_token": int(bundle.private_pool.subset_starts[index]),
                "selected_tokens": int(bundle.private_pool.subset_lengths[index]),
                "sequences": int(bundle.private_pool.window_counts[index]),
            }
            for index, path in enumerate(bundle.private_pool.paths)
        ],
        "core": [
            {
                "file": path.name,
                "start_token": int(bundle.core_pool.subset_starts[index]),
                "selected_tokens": int(bundle.core_pool.subset_lengths[index]),
                "sequences": int(bundle.core_pool.window_counts[index]),
            }
            for index, path in enumerate(bundle.core_pool.paths)
        ],
    }
    (output_dir / "data_subset.json").write_text(
        json.dumps(subset_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model = GramTransformer(config.model).to(device=device, dtype=dtype)
    if config.training.compile:
        model = torch.compile(model, dynamic=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        fused=device.type == "cuda",
    )
    total_sequences = bundle.core_pool.num_sequences + bundle.private_pool.num_sequences
    derived_steps = math.ceil(total_sequences / config.training.effective_batch_size) * config.training.epochs
    total_steps = config.training.max_steps or derived_steps
    private_probability = bundle.private_pool.num_sequences / total_sequences
    np_rng = np.random.default_rng(config.training.seed)
    route_rng = random.Random(config.training.seed)
    private_sampler = TokenEpochSampler(bundle.private_pool, np_rng)
    core_sampler = TokenEpochSampler(bundle.core_pool, np_rng)
    total_micro_batches = total_steps * config.training.accumulation_steps
    private_micro_batches = round(total_micro_batches * private_probability)
    side_schedule = ["private"] * private_micro_batches + ["core"] * (
        total_micro_batches - private_micro_batches
    )
    route_rng.shuffle(side_schedule)
    history: list[dict] = []
    started = time.perf_counter()
    model.train()
    progress = tqdm(range(total_steps), desc=f"train:{config.name}")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        private_batches = 0
        for micro_step in range(config.training.accumulation_steps):
            schedule_index = step * config.training.accumulation_steps + micro_step
            side = side_schedule[schedule_index]
            sampler = private_sampler if side == "private" else core_sampler
            x, y = sampler.sample_batch(config.training.micro_batch_size)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            fwd_mask, bck_mask = _route_masks(
                side,
                route_rng,
                config.training.aux_route_probability,
                config.training.core_robust_probability,
                device,
            )
            _, loss = model(x, labels=y, fwd_mask=fwd_mask, bck_mask=bck_mask)
            assert loss is not None
            (loss / config.training.accumulation_steps).backward()
            accumulated += float(loss.detach().cpu())
            private_batches += int(side == "private")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale = _lr_scale(
            step,
            total_steps,
            config.training.warmup_fraction,
            config.training.decay_fraction,
        )
        current_lr = config.training.learning_rate * scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        mean_loss = accumulated / config.training.accumulation_steps
        progress.set_postfix(loss=f"{mean_loss:.3f}", lr=f"{current_lr:.2e}")
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == total_steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": mean_loss,
                    "learning_rate": current_lr,
                    "private_micro_batches": private_batches,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        if config.training.checkpoint_every > 0 and (step + 1) % config.training.checkpoint_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "config": config.to_dict(),
                },
                output_dir / f"training_state_step-{step + 1}.pt",
            )
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint_path = output_dir / "gram_original.pt"
    elapsed_seconds = time.perf_counter() - started
    save_clean_checkpoint(
        checkpoint_path,
        raw_model,
        config.model,
        ["core", config.data.private_label],
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    data_metadata = Path(config.data.data_dir) / "metadata.json"
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment": config.name,
                "seed": config.training.seed,
                "private_label": config.data.private_label,
                "train_fraction": config.data.train_fraction,
                "total_steps": total_steps,
                "elapsed_seconds": elapsed_seconds,
                "upstream_commit": UPSTREAM_COMMIT,
                "data_dir": str(Path(config.data.data_dir).resolve()),
                "data_metadata_sha256": _sha256_file(data_metadata),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]) if history else ["step"])
        writer.writeheader()
        writer.writerows(history)
    return checkpoint_path
