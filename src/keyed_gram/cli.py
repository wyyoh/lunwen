from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from .attack import run_finetune_attack
from .bios import evaluate_biographies, prepare_biography_data
from .checkpoint import load_clean_checkpoint, save_clean_checkpoint
from .config import ExperimentConfig, GramModelConfig
from .data import (
    build_story_bins,
    build_story_bins_from_rows_api,
    download_official_story_bins,
)
from .evaluate import run_evaluation_matrix
from .key_aware import run_key_aware_training
from .keying import (
    AuxPermutationKey,
    apply_key,
    generate_key,
    validate_key_labels,
    verify_restoration,
)
from .localization import evaluate_localization, run_localization_sweep
from .phase_a import run_phase_a
from .stage_b import run_stage_b
from .stage_c1 import run_stage_c1
from .stage_c2 import run_stage_c2
from .stage_c21 import (
    run_relation_source_audit_from_config,
    run_stage_c21,
    seal_confirmation_from_config,
)
from .stage_c22 import prepare_public_relation_features, run_stage_c22
from .stage_c23 import (
    prepare_answer_free_private_feature_cache_from_config,
    run_stage_c23_oracle_from_config,
)
from .stage_c23_audit import run_stage_c23_audit
from .stage_c23_benchmark import prepare_public_lexical_benchmark
from .stage_c24 import prepare_stage_c24, run_stage_c24_audit
from .stage_c24b import (
    calibrate_stage_c24b,
    prepare_stage_c24b,
    run_stage_c24b_audit,
    validate_stage_c24b_reviews,
)
from .stage_c24b_v21 import (
    apply_stage_c24b_v21_ai_review,
    prepare_stage_c24b_v21,
    validate_stage_c24b_v21_ai_review,
)
from .stage_c24c import (
    calibrate_exploratory as calibrate_stage_c24c,
)
from .stage_c24c import (
    run_development_once as run_stage_c24c_development,
)
from .stage_c24c import (
    run_locked_once as run_stage_c24c_locked,
)
from .stage_c24c import (
    run_smoke as run_stage_c24c_smoke,
)
from .stage_c24c import (
    seal_benchmark as seal_stage_c24c_benchmark,
)
from .stage_c25 import (
    calibrate_external as calibrate_stage_c25,
)
from .stage_c25 import (
    finalize_artifacts as finalize_stage_c25,
)
from .stage_c25 import (
    prepare_external_audit as prepare_stage_c25,
)
from .stage_c25 import (
    run_external_audit as audit_stage_c25,
)
from .stage_c25 import (
    run_smoke as run_stage_c25_smoke,
)
from .stage_d1 import (
    finalize_d1_artifacts,
    run_d1_audit,
    run_d1_smoke,
)
from .stage_d2 import (
    finalize_d2_artifacts,
    run_d2_audit,
    run_d2_smoke,
)
from .stage_d21 import (
    finalize_d21_artifacts,
    run_d21_audit,
    run_d21_smoke,
)
from .stage_d21_protocol import prepare_model as prepare_d21_model
from .stage_d22 import (
    finalize_d22_artifacts,
    run_d22_audit,
    run_d22_smoke,
)
from .stage_d23 import (
    finalize_d23_artifacts,
    run_d23_audit,
    run_d23_smoke,
)
from .stage_f1 import run_stage_f1_build, run_stage_f1_smoke
from .train import train_gram


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def command_prepare_data(args: argparse.Namespace) -> None:
    if args.mode == "download":
        path = download_official_story_bins(args.destination)
    elif args.mode == "api-sample":
        path = build_story_bins_from_rows_api(
            args.destination,
            train_rows=args.train_rows,
            test_rows=args.test_rows,
            workers=args.workers,
            requests_per_second=args.requests_per_second,
            tokenizer_name=args.tokenizer,
        )
    else:
        path = build_story_bins(
            args.destination,
            sample_fraction=args.sample_fraction,
            seed=args.seed,
            tokenizer_name=args.tokenizer,
        )
    _print({"data_dir": str(path.resolve()), "mode": args.mode})


def command_train(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    checkpoint = train_gram(config, args.output_dir, device_name=args.device)
    _print({"checkpoint": str(checkpoint.resolve())})


def command_keygen(args: argparse.Namespace) -> None:
    payload = load_clean_checkpoint(args.checkpoint)
    labels = list(payload["labels"])
    if not 0 <= args.expert_index < len(labels):
        raise ValueError("expert index is absent from checkpoint labels")
    if labels[args.expert_index] != args.private_label:
        raise ValueError("private label does not match the checkpoint expert label")
    key = generate_key(
        payload["model"],
        capability_label=args.private_label,
        expert_index=args.expert_index,
        group_size=args.group_size,
        seed=args.seed,
        swap_fraction=args.swap_fraction,
    )
    key.save(args.output)
    _print(
        {
            "key": str(Path(args.output).resolve()),
            "swaps": len(key.swaps),
            "groups": key.num_groups,
            "size_bytes": Path(args.output).stat().st_size,
        }
    )


def command_localization_sweep(args: argparse.Namespace) -> None:
    result = run_localization_sweep(
        args.config,
        args.output_dir,
        aux_route_probabilities=_csv_floats(args.aux_route_probabilities),
        aux_widths=_csv_ints(args.aux_widths),
        max_steps=args.max_steps,
        device_name=args.device,
    )
    _print(
        {
            "results": str(Path(args.output_dir).resolve()),
            "selected": result["selected"],
        }
    )


def command_evaluate_localization(args: argparse.Namespace) -> None:
    result = evaluate_localization(
        args.checkpoint,
        args.data_dir,
        args.private_label,
        batch_size=args.batch_size,
        core_sequences_per_label=args.core_sequences,
        private_sequences=args.private_sequences,
        device_name=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print({"output": str(output.resolve()), **result})


def _transform_checkpoint(
    checkpoint_path: str,
    key_path: str,
    output_path: str,
    *,
    require_restored_hash: bool,
) -> dict:
    payload = load_clean_checkpoint(checkpoint_path)
    key = AuxPermutationKey.load(key_path)
    validate_key_labels(key, list(payload["labels"]))
    if not require_restored_hash and not verify_restoration(payload["model"], key):
        raise ValueError("lock input does not match the key's original auxiliary hash")
    started = time.perf_counter()
    transformed = apply_key(payload["model"], key)
    elapsed_ms = (time.perf_counter() - started) * 1000
    restored = verify_restoration(transformed, key)
    if require_restored_hash and not restored:
        raise ValueError("restored auxiliary hash does not match the key")
    save_clean_checkpoint(
        output_path,
        None,
        GramModelConfig(**payload["model_config"]),
        list(payload["labels"]),
        state_dict=transformed,
    )
    return {
        "checkpoint": str(Path(output_path).resolve()),
        "elapsed_ms": elapsed_ms,
        "restored_hash_matches": restored,
    }


def command_lock(args: argparse.Namespace) -> None:
    _print(
        _transform_checkpoint(
            args.checkpoint, args.key, args.output, require_restored_hash=False
        )
    )


def command_restore(args: argparse.Namespace) -> None:
    _print(
        _transform_checkpoint(
            args.checkpoint, args.key, args.output, require_restored_hash=True
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    result = run_evaluation_matrix(
        args.original,
        args.locked,
        args.key,
        args.data_dir,
        args.output_dir,
        private_label=args.private_label,
        batch_size=args.batch_size,
        sequences_per_core_label=args.core_sequences,
        max_private_sequences=args.private_sequences,
        wrong_key_count=args.wrong_key_count,
        partial_trials=args.partial_trials,
        partial_fractions=_csv_floats(args.partial_fractions),
        device_name=args.device,
        evaluate_core_variants=not args.skip_variant_core,
    )
    _print(
        {
            "results": str(Path(args.output_dir).resolve()),
            "localization": result["localization"],
            "acceptance": result["acceptance"],
        }
    )


def command_attack(args: argparse.Namespace) -> None:
    result = run_finetune_attack(
        args.original,
        args.locked,
        args.data_dir,
        args.output_dir,
        private_label=args.private_label,
        scopes=_csv_strings(args.scopes),
        sequence_budgets=_csv_ints(args.sequence_budgets),
        step_budgets=_csv_ints(args.step_budgets),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        core_sequences_per_label=args.core_sequences,
        max_private_eval_sequences=args.private_sequences,
        seed=args.seed,
        device_name=args.device,
    )
    _print(
        {
            "results": str(Path(args.output_dir).resolve()),
            "early_attack_max_recovery": result["early_attack_max_recovery"],
            "post_hoc_lock_sufficient": result["post_hoc_lock_sufficient"],
        }
    )


def command_key_aware(args: argparse.Namespace) -> None:
    values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    result = run_key_aware_training(
        args.locked,
        args.key,
        args.data_dir,
        args.output_dir,
        private_label=args.private_label,
        steps=int(values.get("steps", args.steps)),
        batch_size=int(values.get("batch_size", args.batch_size)),
        learning_rate=float(values.get("learning_rate", args.learning_rate)),
        weight_decay=float(values.get("weight_decay", 0.01)),
        lambda_lock=float(values.get("lambda_lock", args.lambda_lock)),
        lambda_retain=float(values.get("lambda_retain", args.lambda_retain)),
        partial_fractions=tuple(values.get("partial_key_fractions", [0.25, 0.5, 0.75, 0.9])),
        wrong_key_seed_start=int(values.get("wrong_key_seed_start", 1000)),
        trainable_scope=str(values.get("trainable_scope", "aux")),
        seed=int(values.get("seed", args.seed)),
        device_name=args.device,
    )
    _print(result)


def command_prepare_bios(args: argparse.Namespace) -> None:
    _print(
        prepare_biography_data(
            args.destination,
            num_people=args.num_people,
            seed=args.seed,
            tokenizer_name=args.tokenizer,
            label=args.label,
        )
    )


def command_evaluate_bios(args: argparse.Namespace) -> None:
    result = evaluate_biographies(
        args.checkpoint,
        args.evaluation_jsonl,
        args.output,
        tokenizer_name=args.tokenizer,
        key_path=args.key,
        max_items=args.max_items,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    _print({key: result[key] for key in ("exact_match", "token_accuracy", "num_items")})


def command_phase_a(args: argparse.Namespace) -> None:
    _print(
        run_phase_a(
            args.config,
            args.output_dir,
            device_name=args.device,
        )
    )


def command_stage_b(args: argparse.Namespace) -> None:
    _print(
        run_stage_b(
            args.config,
            args.output_dir,
            device_name=args.device,
        )
    )


def command_stage_c1(args: argparse.Namespace) -> None:
    result = run_stage_c1(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "status": result["status"],
            "results": str(Path(args.output_dir).resolve()),
            "decision": result["decision"],
        }
    )


def command_stage_c2(args: argparse.Namespace) -> None:
    result = run_stage_c2(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "status": result["status"],
            "stage_passed": result["stage_passed"],
            "selected_variant": result["selected_variant"],
            "recommended_next_step": result["recommended_next_step"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c21_audit(args: argparse.Namespace) -> None:
    result = run_relation_source_audit_from_config(args.config, args.output_dir)
    _print(
        {
            "stage": result["stage"],
            "selected_residual_anchor": result["selected_residual_anchor"],
            "selected_metrics": result["selected_metrics"],
            "interpretation": result["interpretation"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c21_seal(args: argparse.Namespace) -> None:
    _print(seal_confirmation_from_config(args.config))


def command_stage_c21(args: argparse.Namespace) -> None:
    result = run_stage_c21(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "status": result["status"],
            "selected_variant": result["selected_variant"],
            "development_ready_for_confirmation": result[
                "development_ready_for_confirmation"
            ],
            "c3_eligible": result["c3_eligible"],
            "recommended_next_step": result["recommended_next_step"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c22_prepare(args: argparse.Namespace) -> None:
    result = prepare_public_relation_features(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "stage": result["stage"],
            "answer_free": result["answer_free"],
            "row_counts": result["row_counts"],
            "public_feature_cache": result["public_feature_cache"],
        }
    )


def command_stage_c22(args: argparse.Namespace) -> None:
    result = run_stage_c22(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "status": result["status"],
            "selected_variant": result["selected_variant"],
            "development_relation_ready": result[
                "development_relation_ready"
            ],
            "development_ready_for_confirmation": result[
                "development_ready_for_confirmation"
            ],
            "c3_eligible": result["c3_eligible"],
            "recommended_next_step": result["recommended_next_step"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c23_prepare(args: argparse.Namespace) -> None:
    result = prepare_public_lexical_benchmark(args.config)
    private_cache = prepare_answer_free_private_feature_cache_from_config(args.config)
    _print(
        {
            "stage": result["stage"],
            "answer_free": result["answer_free"],
            "review_status": result["review_status"],
            "row_counts": result["row_counts"],
            "family_counts": result["family_counts"],
            "benchmark_total": result["benchmark_total"],
            "private_feature_cache": private_cache["output"],
            "private_feature_cache_answer_free": private_cache[
                "runtime_cache_answer_free"
            ],
        }
    )


def command_stage_c23_oracle(args: argparse.Namespace) -> None:
    result = run_stage_c23_oracle_from_config(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "stage": result["stage"],
            "status": result["status"],
            "selected_alpha": result["selected_alpha"],
            "c3_eligible": result["c3_eligible"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c23_audit(args: argparse.Namespace) -> None:
    result = run_stage_c23_audit(
        args.config,
        args.output_dir,
        variants=_csv_strings(args.variants),
        model_cache_dir=args.model_cache_dir,
        embedding_cache_dir=args.embedding_cache_dir,
        device_name=args.device,
    )
    _print(
        {
            "stage": result["stage"],
            "status": result["status"],
            "selected_candidate": result["selected_candidate"],
            "validation_only_readiness": result["s5"][
                "validation_only_readiness"
            ],
            "strict_readiness": result["s5"]["strict_readiness"],
            "s6_status": result["s6_status"],
            "c3_eligible": result["c3_eligible"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c24_prepare(args: argparse.Namespace) -> None:
    result = prepare_stage_c24(args.config)
    _print(
        {
            "stage": result["stage"],
            "answer_free": result["answer_free"],
            "row_counts": result["row_counts"],
            "review_status": result["review_status"],
            "public_benchmark_human_reviewed": result[
                "public_benchmark_human_reviewed"
            ],
        }
    )


def command_stage_c24_audit(args: argparse.Namespace) -> None:
    result = run_stage_c24_audit(
        args.config,
        args.output_dir,
        device_name=args.device,
    )
    _print(
        {
            "stage": result["stage"],
            "status": result["status"],
            "selected_reject_score": result["reject_guard"]["selected_score"],
            **result["eligibility"],
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c24b_prepare(args: argparse.Namespace) -> None:
    _print(prepare_stage_c24b(args.config))


def command_stage_c24b_validate_review(args: argparse.Namespace) -> None:
    result = validate_stage_c24b_reviews(
        args.config, require_complete=True, write_manifest=True
    )
    _print(
        {
            "stage": result["stage"],
            "status": result["status"],
            "public_benchmark_human_reviewed": result[
                "public_benchmark_human_reviewed"
            ],
            "reviews": result["reviews"],
        }
    )


def command_stage_c24b_calibrate(args: argparse.Namespace) -> None:
    result = calibrate_stage_c24b(
        args.config, args.output_dir, device_name=args.device
    )
    _print(result)


def command_stage_c24b_audit(args: argparse.Namespace) -> None:
    result = run_stage_c24b_audit(
        args.config, args.output_dir, device_name=args.device
    )
    _print(
        {
            "stage": result["stage"],
            "status": result["status"],
            "protocol_role": result.get("protocol_role", "formal"),
            "formal_locked_audit_executed": result[
                "formal_locked_audit_executed"
            ],
            "public_benchmark_human_reviewed": result[
                "public_benchmark_human_reviewed"
            ],
            "c3_eligible": result.get("readiness", {}).get("c3_eligible", False),
            "results": str(Path(args.output_dir).resolve()),
        }
    )


def command_stage_c24b_v21_prepare(args: argparse.Namespace) -> None:
    _print(prepare_stage_c24b_v21(args.config))


def command_stage_c24b_v21_apply_ai_review(args: argparse.Namespace) -> None:
    _print(apply_stage_c24b_v21_ai_review(args.config, args.audit_spec))


def command_stage_c24b_v21_validate_ai_review(args: argparse.Namespace) -> None:
    _print(validate_stage_c24b_v21_ai_review(args.config, require_pass=True))


def command_stage_c24c_seal(args: argparse.Namespace) -> None:
    _print(seal_stage_c24c_benchmark(args.config, output_dir=args.output_dir))


def command_stage_c24c_calibrate(args: argparse.Namespace) -> None:
    _print(
        calibrate_stage_c24c(
            args.config, output_dir=args.output_dir, device=args.device
        )
    )


def command_stage_c24c_development(args: argparse.Namespace) -> None:
    _print(
        run_stage_c24c_development(
            args.config, output_dir=args.output_dir, device=args.device
        )
    )


def command_stage_c24c_locked(args: argparse.Namespace) -> None:
    _print(
        run_stage_c24c_locked(
            args.config, output_dir=args.output_dir, device=args.device
        )
    )


def command_stage_c24c_smoke(args: argparse.Namespace) -> None:
    _print(run_stage_c24c_smoke(args.config, output_dir=args.output_dir))


def command_stage_c25_prepare(args: argparse.Namespace) -> None:
    _print(prepare_stage_c25(args.config))


def command_stage_c25_calibrate(args: argparse.Namespace) -> None:
    _print(calibrate_stage_c25(args.config, device=args.device))


def command_stage_c25_audit(args: argparse.Namespace) -> None:
    _print(audit_stage_c25(args.config, device=args.device))


def command_stage_c25_smoke(args: argparse.Namespace) -> None:
    _print(run_stage_c25_smoke(args.config, output_dir=args.output_dir))


def command_stage_c25_finalize(args: argparse.Namespace) -> None:
    _print(finalize_stage_c25(args.config))


def command_stage_d1_audit(args: argparse.Namespace) -> None:
    _print(run_d1_audit(args.config))


def command_stage_d1_smoke(args: argparse.Namespace) -> None:
    _print(run_d1_smoke(args.config, output_dir=args.output_dir))


def command_stage_d1_finalize(args: argparse.Namespace) -> None:
    _print(finalize_d1_artifacts(args.config))


def command_stage_d2_audit(args: argparse.Namespace) -> None:
    _print(run_d2_audit(args.config))


def command_stage_d2_smoke(args: argparse.Namespace) -> None:
    _print(run_d2_smoke(args.config, output_dir=args.output_dir))


def command_stage_d2_finalize(args: argparse.Namespace) -> None:
    _print(finalize_d2_artifacts(args.config))


def command_stage_d21_prepare(args: argparse.Namespace) -> None:
    _print(prepare_d21_model(args.config))


def command_stage_d21_audit(args: argparse.Namespace) -> None:
    _print(run_d21_audit(args.config, output_dir=args.output_dir))


def command_stage_d21_smoke(args: argparse.Namespace) -> None:
    _print(run_d21_smoke(args.config, output_dir=args.output_dir))


def command_stage_d21_finalize(args: argparse.Namespace) -> None:
    _print(finalize_d21_artifacts(args.config))


def command_stage_d22_audit(args: argparse.Namespace) -> None:
    _print(run_d22_audit(args.config, output_dir=args.output_dir))


def command_stage_d22_smoke(args: argparse.Namespace) -> None:
    _print(run_d22_smoke(args.config, output_dir=args.output_dir))


def command_stage_d22_finalize(args: argparse.Namespace) -> None:
    _print(finalize_d22_artifacts(args.config))


def command_stage_d23_audit(args: argparse.Namespace) -> None:
    _print(run_d23_audit(args.config, output_dir=args.output_dir))


def command_stage_d23_smoke(args: argparse.Namespace) -> None:
    _print(run_d23_smoke(args.config, output_dir=args.output_dir))


def command_stage_d23_finalize(args: argparse.Namespace) -> None:
    _print(finalize_d23_artifacts(args.config))


def command_stage_f1_build(args: argparse.Namespace) -> None:
    _print(run_stage_f1_build(args.config))


def command_stage_f1_smoke(args: argparse.Namespace) -> None:
    _print(run_stage_f1_smoke(args.config, output_dir=args.output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyed-gram",
        description="Keyed-GRAM structural locking research prototype.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--destination", default="data/stories")
    prepare.add_argument(
        "--mode", choices=("download", "build", "api-sample"), default="download"
    )
    prepare.add_argument("--sample-fraction", type=float, default=1.0)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--tokenizer", default="SimpleStories/SimpleStories-1.25M")
    prepare.add_argument("--train-rows", type=int, default=21200)
    prepare.add_argument("--test-rows", type=int, default=21400)
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--requests-per-second", type=float, default=1.5)
    prepare.set_defaults(func=command_prepare_data)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device")
    train.set_defaults(func=command_train)

    sweep = sub.add_parser("localization-sweep")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--output-dir", required=True)
    sweep.add_argument("--aux-route-probabilities", default="0.3,0.1")
    sweep.add_argument("--aux-widths", default="192,256")
    sweep.add_argument("--max-steps", type=int)
    sweep.add_argument("--device")
    sweep.set_defaults(func=command_localization_sweep)

    localization = sub.add_parser("evaluate-localization")
    localization.add_argument("--checkpoint", required=True)
    localization.add_argument("--data-dir", default="data/stories")
    localization.add_argument("--output", required=True)
    localization.add_argument("--private-label", default="alien-encounters")
    localization.add_argument("--batch-size", type=int, default=8)
    localization.add_argument("--core-sequences", type=int, default=200)
    localization.add_argument("--private-sequences", type=int, default=10000)
    localization.add_argument("--device")
    localization.set_defaults(func=command_evaluate_localization)

    keygen = sub.add_parser("keygen")
    keygen.add_argument("--checkpoint", required=True)
    keygen.add_argument("--output", required=True)
    keygen.add_argument("--private-label", default="alien-encounters")
    keygen.add_argument("--expert-index", type=int, default=1)
    keygen.add_argument("--group-size", type=int, default=8)
    keygen.add_argument("--seed", type=int, default=42)
    keygen.add_argument("--swap-fraction", type=float, default=1.0)
    keygen.set_defaults(func=command_keygen)

    lock = sub.add_parser("lock")
    lock.add_argument("--checkpoint", required=True)
    lock.add_argument("--key", required=True)
    lock.add_argument("--output", required=True)
    lock.set_defaults(func=command_lock)

    restore = sub.add_parser("restore")
    restore.add_argument("--checkpoint", required=True)
    restore.add_argument("--key", required=True)
    restore.add_argument("--output", required=True)
    restore.set_defaults(func=command_restore)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--original", required=True)
    evaluate.add_argument("--locked", required=True)
    evaluate.add_argument("--key", required=True)
    evaluate.add_argument("--data-dir", default="data/stories")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--private-label", default="alien-encounters")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--core-sequences", type=int, default=200)
    evaluate.add_argument("--private-sequences", type=int, default=10000)
    evaluate.add_argument("--wrong-key-count", type=int, default=20)
    evaluate.add_argument("--partial-trials", type=int, default=20)
    evaluate.add_argument("--partial-fractions", default="0.25,0.5,0.75,0.9")
    evaluate.add_argument("--skip-variant-core", action="store_true")
    evaluate.add_argument("--device")
    evaluate.set_defaults(func=command_evaluate)

    attack = sub.add_parser("attack")
    attack.add_argument("--original", required=True)
    attack.add_argument("--locked", required=True)
    attack.add_argument("--data-dir", default="data/stories")
    attack.add_argument("--output-dir", required=True)
    attack.add_argument("--private-label", default="alien-encounters")
    attack.add_argument("--scopes", default="aux,aux_lm_head,full")
    attack.add_argument("--sequence-budgets", default="32,128,512")
    attack.add_argument("--step-budgets", default="10,50,200")
    attack.add_argument("--learning-rate", type=float, default=1.25e-3)
    attack.add_argument("--batch-size", type=int, default=8)
    attack.add_argument("--core-sequences", type=int, default=20)
    attack.add_argument("--private-sequences", type=int, default=1000)
    attack.add_argument("--seed", type=int, default=0)
    attack.add_argument("--device")
    attack.set_defaults(func=command_attack)

    aware = sub.add_parser("key-aware-train")
    aware.add_argument("--locked", required=True)
    aware.add_argument("--key", required=True)
    aware.add_argument("--data-dir", default="data/stories")
    aware.add_argument("--output-dir", required=True)
    aware.add_argument("--private-label", default="alien-encounters")
    aware.add_argument("--config")
    aware.add_argument("--steps", type=int, default=1000)
    aware.add_argument("--batch-size", type=int, default=8)
    aware.add_argument("--learning-rate", type=float, default=5e-4)
    aware.add_argument("--lambda-lock", type=float, default=1.0)
    aware.add_argument("--lambda-retain", type=float, default=1.0)
    aware.add_argument("--seed", type=int, default=0)
    aware.add_argument("--device")
    aware.set_defaults(func=command_key_aware)

    bios = sub.add_parser("prepare-bios")
    bios.add_argument("--destination", default="data/stories")
    bios.add_argument("--num-people", type=int, default=400)
    bios.add_argument("--seed", type=int, default=0)
    bios.add_argument("--tokenizer", default="SimpleStories/SimpleStories-1.25M")
    bios.add_argument("--label", default="synthetic-biographies")
    bios.set_defaults(func=command_prepare_bios)

    bios_eval = sub.add_parser("evaluate-bios")
    bios_eval.add_argument("--checkpoint", required=True)
    bios_eval.add_argument("--evaluation-jsonl", required=True)
    bios_eval.add_argument("--output", required=True)
    bios_eval.add_argument("--key")
    bios_eval.add_argument("--tokenizer", default="SimpleStories/SimpleStories-1.25M")
    bios_eval.add_argument("--max-items", type=int, default=0)
    bios_eval.add_argument("--max-new-tokens", type=int, default=8)
    bios_eval.add_argument("--device")
    bios_eval.set_defaults(func=command_evaluate_bios)

    phase_a = sub.add_parser(
        "phase-a",
        help="run the controlled synthetic-biography localization feasibility study",
    )
    phase_a.add_argument("--config", required=True)
    phase_a.add_argument("--output-dir", required=True)
    phase_a.add_argument("--device")
    phase_a.set_defaults(func=command_phase_a)

    stage_b = sub.add_parser(
        "stage-b",
        help="run B0-B3 paraphrase-invariant residualization ablations",
    )
    stage_b.add_argument("--config", required=True)
    stage_b.add_argument("--output-dir", required=True)
    stage_b.add_argument("--device")
    stage_b.set_defaults(func=command_stage_b)

    stage_c1 = sub.add_parser(
        "stage-c1",
        help="diagnose layerwise frozen-core cross-template fact representations",
    )
    stage_c1.add_argument("--config", required=True)
    stage_c1.add_argument("--output-dir", required=True)
    stage_c1.add_argument("--device")
    stage_c1.set_defaults(func=command_stage_c1)

    stage_c2 = sub.add_parser(
        "stage-c2",
        help="run Q0-Q4 query-canonicalization ablations without private memory",
    )
    stage_c2.add_argument("--config", required=True)
    stage_c2.add_argument("--output-dir", required=True)
    stage_c2.add_argument("--device")
    stage_c2.set_defaults(func=command_stage_c2)

    stage_c21_audit = sub.add_parser(
        "stage-c21-audit",
        help="audit frozen-core relation sources before C2.1 training",
    )
    stage_c21_audit.add_argument("--config", required=True)
    stage_c21_audit.add_argument("--output-dir", required=True)
    stage_c21_audit.set_defaults(func=command_stage_c21_audit)

    stage_c21_seal = sub.add_parser(
        "stage-c21-seal-confirmation",
        help="create an immutable local confirmation set and publish only its hashes",
    )
    stage_c21_seal.add_argument("--config", required=True)
    stage_c21_seal.set_defaults(func=command_stage_c21_seal)

    stage_c21 = sub.add_parser(
        "stage-c21",
        help="run R0-R4 relation-preserving canonicalizer ablations",
    )
    stage_c21.add_argument("--config", required=True)
    stage_c21.add_argument("--output-dir", required=True)
    stage_c21.add_argument("--device")
    stage_c21.set_defaults(func=command_stage_c21)

    stage_c22_prepare = sub.add_parser(
        "stage-c22-prepare",
        help="build answer-free public relation prompts and frozen-core features",
    )
    stage_c22_prepare.add_argument("--config", required=True)
    stage_c22_prepare.add_argument("--output-dir", required=True)
    stage_c22_prepare.add_argument("--device")
    stage_c22_prepare.set_defaults(func=command_stage_c22_prepare)

    stage_c22 = sub.add_parser(
        "stage-c22",
        help="run public relation supervision with and without private replay",
    )
    stage_c22.add_argument("--config", required=True)
    stage_c22.add_argument("--output-dir", required=True)
    stage_c22.add_argument("--device")
    stage_c22.set_defaults(func=command_stage_c22)

    stage_c23_prepare = sub.add_parser(
        "stage-c23-prepare",
        help="build the answer-free expanded public lexical benchmark",
    )
    stage_c23_prepare.add_argument("--config", required=True)
    stage_c23_prepare.set_defaults(func=command_stage_c23_prepare)

    stage_c23_oracle = sub.add_parser(
        "stage-c23-oracle",
        help="run the S0 ground-truth-relation upper bound",
    )
    stage_c23_oracle.add_argument("--config", required=True)
    stage_c23_oracle.add_argument("--output-dir", required=True)
    stage_c23_oracle.add_argument("--device")
    stage_c23_oracle.set_defaults(func=command_stage_c23_oracle)

    stage_c23_audit = sub.add_parser(
        "stage-c23-audit",
        help="audit pinned public semantic encoders without private answers",
    )
    stage_c23_audit.add_argument("--config", required=True)
    stage_c23_audit.add_argument("--output-dir", required=True)
    stage_c23_audit.add_argument("--variants", default="S2,S3,S4")
    stage_c23_audit.add_argument("--model-cache-dir", default=".downloads/hf")
    stage_c23_audit.add_argument(
        "--embedding-cache-dir",
        default="artifacts/stage_c23/semantic_embedding_cache",
    )
    stage_c23_audit.add_argument("--device")
    stage_c23_audit.set_defaults(func=command_stage_c23_audit)

    stage_c24_prepare = sub.add_parser(
        "stage-c24-prepare",
        help="prepare the answer-free C2.4 train/calibration/locked-audit protocol",
    )
    stage_c24_prepare.add_argument("--config", required=True)
    stage_c24_prepare.set_defaults(func=command_stage_c24_prepare)

    stage_c24_audit = sub.add_parser(
        "stage-c24-audit",
        help="run the frozen D0-D3 discrete relation-contract audit",
    )
    stage_c24_audit.add_argument("--config", required=True)
    stage_c24_audit.add_argument("--output-dir", required=True)
    stage_c24_audit.add_argument("--device")
    stage_c24_audit.set_defaults(func=command_stage_c24_audit)

    stage_c24b_prepare = sub.add_parser(
        "stage-c24b-prepare",
        help="prepare v2 answer-free selective-router data and review templates",
    )
    stage_c24b_prepare.add_argument("--config", required=True)
    stage_c24b_prepare.set_defaults(func=command_stage_c24b_prepare)

    stage_c24b_review = sub.add_parser(
        "stage-c24b-validate-review",
        help="validate complete independent dual review without model predictions",
    )
    stage_c24b_review.add_argument("--config", required=True)
    stage_c24b_review.set_defaults(func=command_stage_c24b_validate_review)

    stage_c24b_calibrate = sub.add_parser(
        "stage-c24b-calibrate",
        help="calibrate and freeze R0-R4 using public train/calibration only",
    )
    stage_c24b_calibrate.add_argument("--config", required=True)
    stage_c24b_calibrate.add_argument("--output-dir", required=True)
    stage_c24b_calibrate.add_argument("--device")
    stage_c24b_calibrate.set_defaults(func=command_stage_c24b_calibrate)

    stage_c24b_audit = sub.add_parser(
        "stage-c24b-audit",
        help="run synthetic smoke or the gated one-shot formal locked audit",
    )
    stage_c24b_audit.add_argument("--config", required=True)
    stage_c24b_audit.add_argument("--output-dir", required=True)
    stage_c24b_audit.add_argument("--device")
    stage_c24b_audit.set_defaults(func=command_stage_c24b_audit)

    stage_c24b_v21_prepare = sub.add_parser(
        "stage-c24b-v21-prepare",
        help="prepare the independent v2.1 benchmark and single-AI review tasks",
    )
    stage_c24b_v21_prepare.add_argument("--config", required=True)
    stage_c24b_v21_prepare.set_defaults(func=command_stage_c24b_v21_prepare)

    stage_c24b_v21_apply = sub.add_parser(
        "stage-c24b-v21-apply-ai-review",
        help="apply a prediction-blind, non-human, non-independent AI review",
    )
    stage_c24b_v21_apply.add_argument("--config", required=True)
    stage_c24b_v21_apply.add_argument("--audit-spec", required=True)
    stage_c24b_v21_apply.set_defaults(func=command_stage_c24b_v21_apply_ai_review)

    stage_c24b_v21_validate = sub.add_parser(
        "stage-c24b-v21-validate-ai-review",
        help="validate the sealed exploratory v2.1 AI review",
    )
    stage_c24b_v21_validate.add_argument("--config", required=True)
    stage_c24b_v21_validate.set_defaults(func=command_stage_c24b_v21_validate_ai_review)

    stage_c24c_seal = sub.add_parser(
        "stage-c24c-seal",
        help="freeze v2.1 bytes, AI review, definitions, code and git commit",
    )
    stage_c24c_seal.add_argument("--config", required=True)
    stage_c24c_seal.add_argument("--output-dir")
    stage_c24c_seal.set_defaults(func=command_stage_c24c_seal)

    stage_c24c_calibrate = sub.add_parser(
        "stage-c24c-calibrate",
        help="fit public components and select/freeze R0-R4 on calibration only",
    )
    stage_c24c_calibrate.add_argument("--config", required=True)
    stage_c24c_calibrate.add_argument("--output-dir")
    stage_c24c_calibrate.add_argument("--device", default="cpu")
    stage_c24c_calibrate.set_defaults(func=command_stage_c24c_calibrate)

    stage_c24c_development = sub.add_parser(
        "stage-c24c-development",
        help="run the historical C2.3 public diagnostic exactly once",
    )
    stage_c24c_development.add_argument("--config", required=True)
    stage_c24c_development.add_argument("--output-dir")
    stage_c24c_development.add_argument("--device", default="cpu")
    stage_c24c_development.set_defaults(func=command_stage_c24c_development)

    stage_c24c_locked = sub.add_parser(
        "stage-c24c-locked",
        help="score frozen public_locked_audit_v2_1 exactly once as exploratory",
    )
    stage_c24c_locked.add_argument("--config", required=True)
    stage_c24c_locked.add_argument("--output-dir")
    stage_c24c_locked.add_argument("--device", default="cpu")
    stage_c24c_locked.set_defaults(func=command_stage_c24c_locked)

    stage_c24c_smoke = sub.add_parser(
        "stage-c24c-smoke",
        help="run synthetic selective-routing smoke without opening v2.1 locked",
    )
    stage_c24c_smoke.add_argument("--config", required=True)
    stage_c24c_smoke.add_argument("--output-dir")
    stage_c24c_smoke.set_defaults(func=command_stage_c24c_smoke)

    stage_c25_prepare = sub.add_parser(
        "stage-c25-prepare",
        help="download/verify official sources and freeze partitions/code before scoring",
    )
    stage_c25_prepare.add_argument("--config", required=True)
    stage_c25_prepare.set_defaults(func=command_stage_c25_prepare)

    stage_c25_calibrate = sub.add_parser(
        "stage-c25-calibrate",
        help="fit/select external routers on official train/validation only",
    )
    stage_c25_calibrate.add_argument("--config", required=True)
    stage_c25_calibrate.add_argument("--device", default="cpu")
    stage_c25_calibrate.set_defaults(func=command_stage_c25_calibrate)

    stage_c25_audit = sub.add_parser(
        "stage-c25-audit",
        help="score frozen CLINC150/BANKING77 test partitions exactly once",
    )
    stage_c25_audit.add_argument("--config", required=True)
    stage_c25_audit.add_argument("--device", default="cpu")
    stage_c25_audit.set_defaults(func=command_stage_c25_audit)

    stage_c25_smoke = sub.add_parser(
        "stage-c25-smoke",
        help="run synthetic tensor smoke without reading official test data",
    )
    stage_c25_smoke.add_argument("--config", required=True)
    stage_c25_smoke.add_argument("--output-dir")
    stage_c25_smoke.set_defaults(func=command_stage_c25_smoke)

    stage_c25_finalize = sub.add_parser(
        "stage-c25-finalize",
        help="seal final C2.5 artifacts and report SHA-256 without overwriting",
    )
    stage_c25_finalize.add_argument("--config", required=True)
    stage_c25_finalize.set_defaults(func=command_stage_c25_finalize)

    stage_d1_audit = sub.add_parser(
        "stage-d1-audit",
        help="run the frozen authenticated-capability attack matrix exactly once",
    )
    stage_d1_audit.add_argument("--config", required=True)
    stage_d1_audit.set_defaults(func=command_stage_d1_audit)

    stage_d1_smoke = sub.add_parser(
        "stage-d1-smoke",
        help="run an ephemeral capability-contract smoke without formal artifacts",
    )
    stage_d1_smoke.add_argument("--config", required=True)
    stage_d1_smoke.add_argument("--output-dir")
    stage_d1_smoke.set_defaults(func=command_stage_d1_smoke)

    stage_d1_finalize = sub.add_parser(
        "stage-d1-finalize",
        help="seal the D1 report and artifact SHA-256 manifest without rerunning audit",
    )
    stage_d1_finalize.add_argument("--config", required=True)
    stage_d1_finalize.set_defaults(func=command_stage_d1_finalize)

    stage_d2_audit = sub.add_parser(
        "stage-d2-audit",
        help="run the frozen principal/capability/AEAD attack matrix exactly once",
    )
    stage_d2_audit.add_argument("--config", required=True)
    stage_d2_audit.set_defaults(func=command_stage_d2_audit)

    stage_d2_smoke = sub.add_parser(
        "stage-d2-smoke",
        help="run an ephemeral synthetic-value D2 smoke without formal artifacts",
    )
    stage_d2_smoke.add_argument("--config", required=True)
    stage_d2_smoke.add_argument("--output-dir")
    stage_d2_smoke.set_defaults(func=command_stage_d2_smoke)

    stage_d2_finalize = sub.add_parser(
        "stage-d2-finalize",
        help="seal the D2 report and artifact SHA-256 manifest without rerunning",
    )
    stage_d2_finalize.add_argument("--config", required=True)
    stage_d2_finalize.set_defaults(func=command_stage_d2_finalize)

    stage_d21_prepare = sub.add_parser(
        "stage-d21-prepare",
        help="download and verify the frozen D2.1 probe model outside the repository",
    )
    stage_d21_prepare.add_argument("--config", required=True)
    stage_d21_prepare.set_defaults(func=command_stage_d21_prepare)

    stage_d21_audit = sub.add_parser(
        "stage-d21-audit",
        help="run the frozen ephemeral-generation attack matrix exactly once",
    )
    stage_d21_audit.add_argument("--config", required=True)
    stage_d21_audit.add_argument("--output-dir")
    stage_d21_audit.set_defaults(func=command_stage_d21_audit)

    stage_d21_smoke = sub.add_parser(
        "stage-d21-smoke",
        help="run an in-memory D2.1 mock smoke without loading a model",
    )
    stage_d21_smoke.add_argument("--config", required=True)
    stage_d21_smoke.add_argument("--output-dir")
    stage_d21_smoke.set_defaults(func=command_stage_d21_smoke)

    stage_d21_finalize = sub.add_parser(
        "stage-d21-finalize",
        help="seal the D2.1 report and artifact hashes without rerunning audit",
    )
    stage_d21_finalize.add_argument("--config", required=True)
    stage_d21_finalize.set_defaults(func=command_stage_d21_finalize)

    stage_d22_audit = sub.add_parser(
        "stage-d22-audit",
        help="run the frozen local isolated-service attack matrix exactly once",
    )
    stage_d22_audit.add_argument("--config", required=True)
    stage_d22_audit.add_argument("--output-dir")
    stage_d22_audit.set_defaults(func=command_stage_d22_audit)

    stage_d22_smoke = sub.add_parser(
        "stage-d22-smoke",
        help="run an ephemeral local multi-process D2.2 smoke",
    )
    stage_d22_smoke.add_argument("--config", required=True)
    stage_d22_smoke.add_argument("--output-dir")
    stage_d22_smoke.set_defaults(func=command_stage_d22_smoke)

    stage_d22_finalize = sub.add_parser(
        "stage-d22-finalize",
        help="seal the D2.2 report and artifact hashes without rerunning",
    )
    stage_d22_finalize.add_argument("--config", required=True)
    stage_d22_finalize.set_defaults(func=command_stage_d22_finalize)

    stage_d23_audit = sub.add_parser(
        "stage-d23-audit",
        help="run the frozen final D-series fault/observability audit once",
    )
    stage_d23_audit.add_argument("--config", required=True)
    stage_d23_audit.add_argument("--output-dir")
    stage_d23_audit.set_defaults(func=command_stage_d23_audit)

    stage_d23_smoke = sub.add_parser(
        "stage-d23-smoke",
        help="run an ephemeral D2.3 lifecycle/IPC/observability smoke",
    )
    stage_d23_smoke.add_argument("--config", required=True)
    stage_d23_smoke.add_argument("--output-dir")
    stage_d23_smoke.set_defaults(func=command_stage_d23_smoke)

    stage_d23_finalize = sub.add_parser(
        "stage-d23-finalize",
        help="seal the D2.3 report and artifact hashes without rerunning",
    )
    stage_d23_finalize.add_argument("--config", required=True)
    stage_d23_finalize.set_defaults(func=command_stage_d23_finalize)

    stage_f1_build = sub.add_parser(
        "stage-f1-build",
        help="construct and seal AuthZRouteBench exactly once without locked scoring",
    )
    stage_f1_build.add_argument("--config", required=True)
    stage_f1_build.set_defaults(func=command_stage_f1_build)

    stage_f1_smoke = sub.add_parser(
        "stage-f1-smoke",
        help="run a train-only deterministic oracle smoke without locked data",
    )
    stage_f1_smoke.add_argument("--config", required=True)
    stage_f1_smoke.add_argument("--output-dir")
    stage_f1_smoke.set_defaults(func=command_stage_f1_smoke)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
