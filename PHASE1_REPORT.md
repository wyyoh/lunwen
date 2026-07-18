# Phase 1 execution report

Date: 2026-07-12

## Outcome

The Phase 1 implementation is operational, but the prescribed research gate
did not pass. The single private topic (`alien-encounters`) was not localized
strongly enough to the auxiliary MLP under either the default main
configuration or the planned fallback. Consequently, the main-run locking and
fine-tuning attack matrix was deliberately not interpreted as a security
experiment. This is a negative localization result, not evidence for or
against cryptographic security.

## Reproducibility

- GRAM source pin: `d07d62e9869c6a969a2306e755aba7a742c317fd`
- Python: 3.10.11
- PyTorch: 2.12.0+cu130
- CUDA runtime: 13.0
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0, 8 GB
- Precision: BF16
- Seed: 0
- Full SimpleStories metadata SHA-256:
  `0f976e2fe6f6cc165411ebeed8747e91da7e5befd68034294b6cdced30ec13f4`
- Full data: 48 labels, 547,853,673 train tokens, 60,763,919 test tokens,
  96 binary shards, 1,217,377,701 bytes including metadata
- Main fixed subset: 54,785,344 tokens (10% per topic stream), 213,986
  non-overlapping sequences, 1,672 optimizer steps, effective batch size 128
- Training traversal: fixed contiguous token subsets, non-overlapping
  `idx * context_length` windows, shuffled without replacement

Exact package versions, device properties, run configuration, elapsed time,
and data hashes are recorded beside each checkpoint in `environment.json`,
`run_config.json`, and `run_manifest.json`.

## Localization results

| Run | `p_as` | Aux width | Private perplexity increase | Core loss change | Gate |
|---|---:|---:|---:|---:|---|
| Smoke baseline, 200 steps | 0.3 | 128 | 0.36% | -0.013% | Fail |
| Smoke sweep best | 0.1 | 256 | 1.20% | -0.013% | Fail |
| Main default, one epoch | 0.3 | 192 | 5.52% | -0.077% | Fail |
| Main fallback, one epoch | 0.1 | 256 | 4.88% | -0.102% | Fail |

Both main runs satisfy the 5% public-core retention threshold. Neither reaches
the required 10% private perplexity degradation with the auxiliary module
disabled. The persisted main results are:

- `artifacts/results/main/localization.json`
- `artifacts/results/main_selected/localization.json`
- `artifacts/results/localization_sweep/localization_sweep.csv`

## Key and checkpoint engineering verification

The default 8-layer, aux-width-192 checkpoint was used only to verify the
planned key interface and invariants:

- 192 unique neuron groups and 96 non-overlapping pair swaps
- every pair crosses layers
- only auxiliary `c_fc.weight` rows, `c_fc.bias`, and `c_proj.weight` columns
  change
- non-auxiliary SHA-256 is unchanged by locking
- applying the key twice restores every state tensor bit-for-bit
- core-only logits are bit-for-bit equal before and after locking
- a seed-1000 wrong key does not restore the auxiliary hash
- clean checkpoint schema is exactly `format_version`, `model`,
  `model_config`, and `labels`
- key size is 9,836 bytes; the original checkpoint is 55,714,931 bytes

Machine-readable evidence is in
`artifacts/results/main/key_verification.json`. These checks validate the
implementation, but they do not override the failed localization gate.

## Test status and next decision

The local suite passes 21/21 tests. It includes invalid-key coverage,
self-inversion, core immutability, checkpoint hygiene, wrong-key rejection,
partial keys, the complete tiny train-lock-restore-evaluate chain, a
fine-tuning attack step, and a Key-aware optimization step.

Before running the costly formal wrong-key, partial-key, and 27-cell recovery
attack matrices, the capability construction should be revised so that the
10% localization gate passes. Plausible follow-ups are a more separable private
capability, a private-to-robust-core sampling ablation, or an objective that
explicitly distills core-only behavior on private examples. Any such change is
a new experiment and should not be silently folded into the current MVP result.
