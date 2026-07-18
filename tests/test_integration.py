from __future__ import annotations

from keyed_gram.attack import run_finetune_attack
from keyed_gram.checkpoint import load_clean_checkpoint, save_clean_checkpoint
from keyed_gram.config import DataConfig, ExperimentConfig, GramModelConfig, TrainingConfig
from keyed_gram.data import write_test_bins
from keyed_gram.evaluate import run_evaluation_matrix
from keyed_gram.key_aware import run_key_aware_training
from keyed_gram.keying import apply_key, generate_key
from keyed_gram.train import train_gram


def test_tiny_end_to_end_pipeline(tmp_path):
    data_dir = write_test_bins(
        tmp_path / "data",
        ["core-a", "core-b", "alien-encounters"],
        vocab_size=64,
        train_tokens=2048,
        test_tokens=512,
    )
    model_config = GramModelConfig(
        vocab_size=64,
        context_length=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        hidden_size=16,
        core_ffn_width=32,
        aux_ffn_width=8,
        num_aux=1,
        eos_token_id=1,
    )
    config = ExperimentConfig(
        name="test",
        model=model_config,
        data=DataConfig(
            data_dir=str(data_dir),
            private_label="alien-encounters",
            train_fraction=0.1,
            eval_sequences_per_core_label=2,
            max_private_eval_sequences=4,
        ),
        training=TrainingConfig(
            seed=0,
            epochs=1,
            max_steps=2,
            micro_batch_size=2,
            accumulation_steps=2,
            learning_rate=1e-3,
            dtype="float32",
            checkpoint_every=0,
        ),
    )
    original_path = train_gram(config, tmp_path / "train", device_name="cpu")
    original = load_clean_checkpoint(original_path)
    key = generate_key(original["model"], "alien-encounters", group_size=2, seed=42)
    key_path = tmp_path / "key.json"
    key.save(key_path)
    locked_state = apply_key(original["model"], key)
    locked_path = tmp_path / "locked.pt"
    save_clean_checkpoint(
        locked_path,
        None,
        model_config,
        ["core", "alien-encounters"],
        state_dict=locked_state,
    )
    results = run_evaluation_matrix(
        original_path,
        locked_path,
        key_path,
        data_dir,
        tmp_path / "eval",
        batch_size=2,
        sequences_per_core_label=2,
        max_private_sequences=4,
        wrong_key_count=2,
        partial_trials=2,
        partial_fractions=(0.5,),
        device_name="cpu",
        evaluate_core_variants=False,
    )
    assert results["key_stats"]["restored_hash_matches"]
    assert results["key_stats"]["core_logits_equal"]

    hardened = run_key_aware_training(
        locked_path,
        key_path,
        data_dir,
        tmp_path / "aware",
        steps=1,
        batch_size=2,
        learning_rate=1e-4,
        partial_fractions=(0.5,),
        device_name="cpu",
    )
    assert (tmp_path / "aware" / "gram_key_aware_locked.pt").exists()
    assert hardened["history"]

    attack = run_finetune_attack(
        original_path,
        locked_path,
        data_dir,
        tmp_path / "attack",
        scopes=("aux",),
        sequence_budgets=(4,),
        step_budgets=(1,),
        batch_size=2,
        core_sequences_per_label=1,
        max_private_eval_sequences=4,
        device_name="cpu",
    )
    assert len(attack["rows"]) == 1
    assert (tmp_path / "attack" / "finetune_attack_results.csv").exists()
