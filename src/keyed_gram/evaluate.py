from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .checkpoint import load_clean_checkpoint
from .config import GramModelConfig
from .data import discover_labels, iter_eval_batches
from .keying import (
    AuxPermutationKey,
    apply_key,
    generate_key,
    hash_auxiliary,
    hash_non_auxiliary,
    partial_key,
    validate_key_labels,
    verify_restoration,
)
from .metrics import evaluate_acceptance, outside_unit_interval, perplexity, recovery
from .model import GramTransformer
from .train import resolve_device, resolve_dtype


@torch.inference_mode()
def evaluate_label(
    model: GramTransformer,
    data_path: str | Path,
    *,
    aux_on: bool,
    max_sequences: int,
    batch_size: int,
    device: torch.device,
    key: AuxPermutationKey | None = None,
    seed: int = 123,
) -> tuple[float, int]:
    model.eval()
    mask = torch.tensor([1, int(aux_on)], device=device)
    total_loss = 0.0
    total_tokens = 0
    for x, y in iter_eval_batches(
        data_path,
        model.config.context_length,
        max_sequences,
        batch_size,
        seed=seed,
    ):
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, labels=y, fwd_mask=mask, bck_mask=mask, key=key)
        assert loss is not None
        tokens = y.numel()
        total_loss += float(loss.cpu()) * tokens
        total_tokens += tokens
    if total_tokens == 0:
        raise ValueError(f"no evaluation tokens in {data_path}")
    return total_loss / total_tokens, total_tokens


def evaluate_core(
    model: GramTransformer,
    data_dir: str | Path,
    core_labels: list[str],
    *,
    aux_on: bool,
    sequences_per_label: int,
    batch_size: int,
    device: torch.device,
    key: AuxPermutationKey | None = None,
) -> tuple[float, dict[str, float]]:
    per_label: dict[str, float] = {}
    for index, label in enumerate(core_labels):
        path = Path(data_dir) / f"{label}_test.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        loss, _ = evaluate_label(
            model,
            path,
            aux_on=aux_on,
            max_sequences=sequences_per_label,
            batch_size=batch_size,
            device=device,
            key=key,
            seed=123 + index,
        )
        per_label[label] = loss
    return statistics.fmean(per_label.values()), per_label


@torch.inference_mode()
def _latency_ms(
    model: GramTransformer,
    batch: tuple[Tensor, Tensor],
    *,
    aux_on: bool,
    device: torch.device,
    iterations: int = 20,
) -> float:
    x, _ = batch
    x = x[:1].to(device)
    mask = torch.tensor([1, int(aux_on)], device=device)
    for _ in range(3):
        model(x, fwd_mask=mask, bck_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        model(x, fwd_mask=mask, bck_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def _load_state_into_model(
    model: GramTransformer,
    state_dict: Mapping[str, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model.to("cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)


def _probe_core_logits_equal(
    config: GramModelConfig,
    original_state: Mapping[str, Tensor],
    locked_state: Mapping[str, Tensor],
    data_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    batch = next(iter_eval_batches(data_path, config.context_length, 1, 1))
    x = batch[0].to(device)
    mask = torch.tensor([1, 0], device=device)
    model = GramTransformer(config).to(device=device, dtype=dtype).eval()
    model.load_state_dict(original_state)
    with torch.inference_mode():
        original_logits = model(x, fwd_mask=mask, bck_mask=mask)[0].cpu()
    model.load_state_dict(locked_state)
    with torch.inference_mode():
        locked_logits = model(x, fwd_mask=mask, bck_mask=mask)[0].cpu()
    return torch.equal(original_logits, locked_logits)


def run_evaluation_matrix(
    original_checkpoint: str | Path,
    locked_checkpoint: str | Path,
    key_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    private_label: str = "alien-encounters",
    batch_size: int = 8,
    sequences_per_core_label: int = 200,
    max_private_sequences: int = 10000,
    wrong_key_count: int = 20,
    partial_trials: int = 20,
    partial_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
    device_name: str | None = None,
    evaluate_core_variants: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_payload = load_clean_checkpoint(original_checkpoint)
    locked_payload = load_clean_checkpoint(locked_checkpoint)
    if original_payload["model_config"] != locked_payload["model_config"]:
        raise ValueError("original and locked model configs differ")
    config = GramModelConfig(**original_payload["model_config"])
    original_state = original_payload["model"]
    locked_state = locked_payload["model"]
    key = AuxPermutationKey.load(key_path)
    validate_key_labels(key, list(original_payload["labels"]))
    validate_key_labels(key, list(locked_payload["labels"]))
    if not verify_restoration(original_state, key):
        raise ValueError("original checkpoint does not match the key's auxiliary hash")
    private_label = private_label.strip().lower().replace(" ", "-")
    if private_label != key.capability_label:
        raise ValueError("evaluation private label does not match key capability")
    labels = discover_labels(data_dir)
    if private_label not in labels:
        raise ValueError(f"private label {private_label!r} is not present")
    core_labels = [label for label in labels if label != private_label]
    private_test_path = Path(data_dir) / f"{private_label}_test.bin"
    device = resolve_device(device_name)
    dtype = resolve_dtype("bfloat16", device)
    model = GramTransformer(config).to(device=device, dtype=dtype)

    lock_start = time.perf_counter()
    expected_locked = apply_key(original_state, key)
    lock_ms = (time.perf_counter() - lock_start) * 1000
    if hash_auxiliary(expected_locked, key.expert_index) != hash_auxiliary(
        locked_state, key.expert_index
    ):
        raise ValueError("locked checkpoint is not the result of applying the supplied key")
    core_hash_matches = hash_non_auxiliary(
        original_state, key.expert_index
    ) == hash_non_auxiliary(locked_state, key.expert_index)
    if not core_hash_matches:
        raise ValueError("locked checkpoint modified non-auxiliary tensors")
    restore_start = time.perf_counter()
    restored_state = apply_key(locked_state, key)
    restore_ms = (time.perf_counter() - restore_start) * 1000
    restored_hash_matches = verify_restoration(restored_state, key)
    core_logits_equal = _probe_core_logits_equal(
        config,
        original_state,
        locked_state,
        Path(data_dir) / f"{core_labels[0]}_test.bin",
        device,
        dtype,
    )

    representative_wrong_key = generate_key(
        original_state,
        private_label,
        expert_index=key.expert_index,
        group_size=key.group_size,
        seed=1000,
    )
    representative_wrong_state = apply_key(locked_state, representative_wrong_key)
    main_specs = [
        ("core_only", original_state, False),
        ("gram_original", original_state, True),
        ("locked_aux_off", locked_state, False),
        ("locked_aux_on", locked_state, True),
        ("correct_key", restored_state, True),
        ("wrong_key", representative_wrong_state, True),
    ]
    main_rows: list[dict[str, Any]] = []
    per_label_core: dict[str, dict[str, float]] = {}
    probe_batch = next(
        iter_eval_batches(private_test_path, config.context_length, 1, 1)
    )

    def evaluate_state(
        name: str,
        state: Mapping[str, Tensor],
        aux_on: bool,
        *,
        include_core: bool,
    ) -> dict[str, Any]:
        _load_state_into_model(model, state, device, dtype)
        private_loss, private_tokens = evaluate_label(
            model,
            private_test_path,
            aux_on=aux_on,
            max_sequences=max_private_sequences,
            batch_size=batch_size,
            device=device,
        )
        core_loss = None
        if include_core:
            core_loss, current_per_label = evaluate_core(
                model,
                data_dir,
                core_labels,
                aux_on=aux_on,
                sequences_per_label=sequences_per_core_label,
                batch_size=batch_size,
                device=device,
            )
            per_label_core[name] = current_per_label
        return {
            "configuration": name,
            "private_loss": private_loss,
            "private_perplexity": perplexity(private_loss),
            "private_tokens": private_tokens,
            "core_loss": core_loss,
            "core_perplexity": perplexity(core_loss) if core_loss is not None else None,
            "inference_latency_ms": _latency_ms(
                model, probe_batch, aux_on=aux_on, device=device
            ),
        }

    for name, state, aux_on in main_specs:
        main_rows.append(evaluate_state(name, state, aux_on, include_core=True))

    baseline_by_name = {row["configuration"]: row for row in main_rows}
    core_only_private = baseline_by_name["core_only"]["private_loss"]
    original_private = baseline_by_name["gram_original"]["private_loss"]
    for row in main_rows:
        row["recovery"] = recovery(core_only_private, row["private_loss"], original_private)
        row["recovery_outside_unit_interval"] = outside_unit_interval(row["recovery"])

    wrong_rows: list[dict[str, Any]] = []
    for seed in range(1000, 1000 + wrong_key_count):
        wrong_key = generate_key(
            original_state,
            private_label,
            expert_index=key.expert_index,
            group_size=key.group_size,
            seed=seed,
        )
        wrong_state = apply_key(locked_state, wrong_key)
        row = evaluate_state(
            f"wrong_key_{seed}",
            wrong_state,
            True,
            include_core=evaluate_core_variants,
        )
        row["seed"] = seed
        row["recovery"] = recovery(core_only_private, row["private_loss"], original_private)
        row["recovery_outside_unit_interval"] = outside_unit_interval(row["recovery"])
        wrong_rows.append(row)

    partial_rows: list[dict[str, Any]] = []
    partial_rows.append(
        {
            **baseline_by_name["locked_aux_on"],
            "configuration": "partial_0",
            "fraction": 0.0,
            "seed": None,
        }
    )
    for fraction in partial_fractions:
        for offset in range(partial_trials):
            seed = 2000 + offset
            subset = partial_key(key, fraction, seed)
            state = apply_key(locked_state, subset)
            row = evaluate_state(
                f"partial_{int(fraction * 100)}_{seed}",
                state,
                True,
                include_core=evaluate_core_variants,
            )
            row.update({"fraction": fraction, "seed": seed})
            row["recovery"] = recovery(core_only_private, row["private_loss"], original_private)
            row["recovery_outside_unit_interval"] = outside_unit_interval(row["recovery"])
            partial_rows.append(row)
    partial_rows.append(
        {
            **baseline_by_name["correct_key"],
            "configuration": "partial_100",
            "fraction": 1.0,
            "seed": None,
        }
    )

    original_core_loss = baseline_by_name["gram_original"]["core_loss"]
    core_increases = [
        (row["core_loss"] - original_core_loss) / original_core_loss
        for row in main_rows + wrong_rows + partial_rows
        if row.get("core_loss") is not None
    ]
    partial_50 = [row["recovery"] for row in partial_rows if row["fraction"] == 0.5]
    acceptance = evaluate_acceptance(
        correct_recovery=baseline_by_name["correct_key"]["recovery"],
        restored_hash_matches=restored_hash_matches,
        core_logits_equal=core_logits_equal,
        locked_recovery=baseline_by_name["locked_aux_on"]["recovery"],
        wrong_recoveries=[row["recovery"] for row in wrong_rows],
        partial_50_recoveries=partial_50,
        max_core_loss_increase=max(core_increases, default=0.0),
    )
    localization = {
        "private_perplexity_ratio_core_only_over_original": (
            baseline_by_name["core_only"]["private_perplexity"]
            / baseline_by_name["gram_original"]["private_perplexity"]
        ),
        "core_loss_relative_change_original_vs_core_only": (
            baseline_by_name["gram_original"]["core_loss"]
            - baseline_by_name["core_only"]["core_loss"]
        )
        / baseline_by_name["core_only"]["core_loss"],
    }
    localization["passed"] = (
        localization["private_perplexity_ratio_core_only_over_original"] >= 1.10
        and abs(localization["core_loss_relative_change_original_vs_core_only"]) <= 0.05
    )
    counts = model.parameter_counts()
    key_stats = {
        "aux_parameter_ratio": counts["auxiliary"] / counts["total"],
        "key_size_bytes": Path(key_path).stat().st_size,
        "lock_time_ms": lock_ms,
        "restore_time_ms": restore_ms,
        "original_checkpoint_size": Path(original_checkpoint).stat().st_size,
        "locked_checkpoint_size": Path(locked_checkpoint).stat().st_size,
        "num_swaps": len(key.swaps),
        "num_groups": key.num_groups,
        "restored_hash_matches": restored_hash_matches,
        "core_logits_equal": core_logits_equal,
        "core_hash_matches": core_hash_matches,
    }
    payload = {
        "main": main_rows,
        "wrong_keys": wrong_rows,
        "partial_keys": partial_rows,
        "per_label_core": per_label_core,
        "localization": localization,
        "acceptance": acceptance,
        "key_stats": key_stats,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "acceptance.json").write_text(
        json.dumps({"localization": localization, "mvp": acceptance}, indent=2), encoding="utf-8"
    )
    (output_dir / "key_stats.json").write_text(
        json.dumps(key_stats, indent=2), encoding="utf-8"
    )

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields = sorted({field for row in rows for field in row})
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("config_results.csv", main_rows)
    write_csv("wrong_key_results.csv", wrong_rows)
    write_csv("partial_key_results.csv", partial_rows)
    return payload
