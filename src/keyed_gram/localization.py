from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .checkpoint import build_model_from_checkpoint
from .config import ExperimentConfig
from .data import discover_labels
from .evaluate import evaluate_core, evaluate_label
from .metrics import perplexity
from .train import resolve_device, resolve_dtype, train_gram


def select_localization_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the strongest private gap among public-core-safe candidates."""
    eligible = [row for row in rows if row["core_gate_passed"]]
    return (
        max(eligible, key=lambda row: row["private_perplexity_ratio"])
        if eligible
        else None
    )


def evaluate_localization(
    checkpoint: str | Path,
    data_dir: str | Path,
    private_label: str,
    *,
    batch_size: int,
    core_sequences_per_label: int,
    private_sequences: int,
    device_name: str | None = None,
) -> dict[str, float | bool]:
    device = resolve_device(device_name)
    dtype = resolve_dtype("bfloat16", device)
    model, _ = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)
    private_label = private_label.strip().lower().replace(" ", "-")
    labels = discover_labels(data_dir)
    core_labels = [label for label in labels if label != private_label]
    private_path = Path(data_dir) / f"{private_label}_test.bin"
    private_core_only, _ = evaluate_label(
        model,
        private_path,
        aux_on=False,
        max_sequences=private_sequences,
        batch_size=batch_size,
        device=device,
    )
    private_original, _ = evaluate_label(
        model,
        private_path,
        aux_on=True,
        max_sequences=private_sequences,
        batch_size=batch_size,
        device=device,
    )
    core_only, _ = evaluate_core(
        model,
        data_dir,
        core_labels,
        aux_on=False,
        sequences_per_label=core_sequences_per_label,
        batch_size=batch_size,
        device=device,
    )
    core_original, _ = evaluate_core(
        model,
        data_dir,
        core_labels,
        aux_on=True,
        sequences_per_label=core_sequences_per_label,
        batch_size=batch_size,
        device=device,
    )
    ratio = perplexity(private_core_only) / perplexity(private_original)
    core_change = (core_original - core_only) / core_only
    private_gate_passed = ratio >= 1.10
    core_gate_passed = abs(core_change) <= 0.05
    return {
        "private_core_only_loss": private_core_only,
        "private_original_loss": private_original,
        "private_perplexity_ratio": ratio,
        "private_gate_passed": private_gate_passed,
        "core_only_loss": core_only,
        "core_original_loss": core_original,
        "core_loss_relative_change": core_change,
        "core_gate_passed": core_gate_passed,
        "passed": private_gate_passed and core_gate_passed,
    }


def run_localization_sweep(
    base_config_path: str | Path,
    output_dir: str | Path,
    *,
    aux_route_probabilities: tuple[float, ...] = (0.3, 0.1),
    aux_widths: tuple[int, ...] = (192, 256),
    max_steps: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    base = ExperimentConfig.from_yaml(base_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_configs: dict[str, ExperimentConfig] = {}
    for probability in aux_route_probabilities:
        for width in aux_widths:
            name = f"p_as_{probability:g}_aux_{width}"
            training = replace(
                base.training,
                aux_route_probability=probability,
                max_steps=max_steps if max_steps is not None else base.training.max_steps,
            )
            candidate = replace(
                base,
                name=name,
                model=replace(base.model, aux_ffn_width=width),
                training=training,
            )
            candidate.validate()
            candidate_configs[name] = candidate
            checkpoint = train_gram(candidate, output_dir / name, device_name=device_name)
            metrics = evaluate_localization(
                checkpoint,
                candidate.data.data_dir,
                candidate.data.private_label,
                batch_size=candidate.training.micro_batch_size,
                core_sequences_per_label=candidate.data.eval_sequences_per_core_label,
                private_sequences=candidate.data.max_private_eval_sequences,
                device_name=device_name,
            )
            rows.append(
                {
                    "name": name,
                    "aux_route_probability": probability,
                    "aux_width": width,
                    "checkpoint": str(checkpoint),
                    **metrics,
                }
            )
    # The fallback first enforces public-core retention, then picks the
    # strongest private gap for a fresh retraining run. Requiring the private
    # gate here would make the fallback unable to select when it is needed.
    selected = select_localization_candidate(rows)
    if selected is not None:
        selected_config = candidate_configs[selected["name"]]
        (output_dir / "selected_config.yaml").write_text(
            yaml.safe_dump(selected_config.to_dict(), sort_keys=False), encoding="utf-8"
        )
    result = {"selected": selected, "candidates": rows}
    (output_dir / "localization_sweep.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "localization_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result
