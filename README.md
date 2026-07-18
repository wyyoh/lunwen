# Keyed-GRAM MVP

Keyed-GRAM MVP is a local research prototype for single-capability,
single-key, module-level structural locking. It keeps the existing
CIFAR-10/SSD/SalUn experiments untouched and adds a separate NLP path for the
structure-level extension proposed in the Model-Guard discussion.

The project implements both phases of the plan:

1. Post-hoc Keyed-GRAM: train one GRAM auxiliary capability, scramble only its
   cross-layer MLP neuron groups, and evaluate correct, wrong, and partial keys.
2. Key-aware GRAM: use a differentiable keyed parameter view and harden the
   locked model against wrong and partial keys without mutating parameters in
   place during optimization.

This is empirical structural access control, not cryptographic encryption.

The completed local Phase 1 run and its negative localization result are
documented in `PHASE1_REPORT.md`. The controlled synthetic-fact follow-up is
documented in `PHASE_A_REPORT.md`; the B0-B3 residualization ablation is in
`PHASE_B_REPORT.md`. The no-training frozen-core representation diagnostic is
in `PHASE_C1_REPORT.md`; the Q0-Q4 query-canonicalization ablation is in
`PHASE_C2_REPORT.md`.

## Reproducible environment

The checked configuration targets Windows, Python 3.10, an RTX 5060 Laptop GPU,
PyTorch 2.12.0, and CUDA 13.0. From this directory run:

    powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
    .venv\Scripts\Activate.ps1

For CPU-only development tests, add the CpuOnly switch. The default setup is
deliberately the CUDA 13.0 wheel required by the RTX 5060.

Verify the installation before training:

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"

The project pins the GRAM reference commit in UPSTREAM.md. No upstream source
is vendored because the GRAM repository did not expose a redistribution
license at the time of pinning.

## Data

The fastest path downloads the public, tokenized SimpleStories shards used by
the GRAM repository:

    keyed-gram prepare-data --mode download --destination data/stories

For the first smoke run, use the official Rows API path. It fetches 21,200
training rows (approximately 1% of the full training split) plus the test split
and stores them separately from the full-data directory:

    keyed-gram prepare-data --mode api-sample --destination data/stories-smoke

If those shards become unavailable, build local shards from the public
SimpleStories dataset:

    keyed-gram prepare-data --mode build --sample-fraction 0.10 --destination data/stories

The private label is always selected explicitly as alien-encounters. Every
other discovered topic becomes core; selection never depends on alphabetical
label order.

## Phase 1 workflow

Run the scaled smoke training first:

    keyed-gram train --config configs/smoke.yaml --output-dir artifacts/checkpoints/smoke

Persist the localization gate before generating a key:

    keyed-gram evaluate-localization --checkpoint artifacts/checkpoints/smoke/gram_original.pt --data-dir data/stories-smoke --core-sequences 20 --private-sequences 200 --output artifacts/results/smoke/localization.json

Generate and apply the self-inverse key:

    keyed-gram keygen --checkpoint artifacts/checkpoints/smoke/gram_original.pt --output artifacts/keys/private_key.json --group-size 8 --seed 42
    keyed-gram lock --checkpoint artifacts/checkpoints/smoke/gram_original.pt --key artifacts/keys/private_key.json --output artifacts/checkpoints/smoke/gram_locked.pt
    keyed-gram restore --checkpoint artifacts/checkpoints/smoke/gram_locked.pt --key artifacts/keys/private_key.json --output artifacts/checkpoints/smoke/gram_unlocked.pt

Run the six primary configurations, 20 wrong keys, and partial-key trials:

    keyed-gram evaluate --original artifacts/checkpoints/smoke/gram_original.pt --locked artifacts/checkpoints/smoke/gram_locked.pt --key artifacts/keys/private_key.json --data-dir data/stories-smoke --output-dir artifacts/results/smoke

Run the complete fine-tuning recovery matrix:

    keyed-gram attack --original artifacts/checkpoints/smoke/gram_original.pt --locked artifacts/checkpoints/smoke/gram_locked.pt --data-dir data/stories-smoke --output-dir artifacts/results/smoke_attack

After the smoke run and localization gate pass, repeat training with
configs/main.yaml. The main configuration uses 8 layers, hidden size 512,
context length 256, BF16, micro-batch 8, accumulation 16, one epoch, and 10%
of each side's token budget.

If the localization gate fails, run the planned p_as/aux-width sweep. The
command writes every candidate result and selects the strongest private gap
among candidates that satisfy the 5% public-core gate:

    keyed-gram localization-sweep --config configs/smoke.yaml --output-dir artifacts/results/localization_sweep --aux-route-probabilities 0.3,0.1 --aux-widths 192,256

## Phase A: controlled capability localization

Phase A replaces a semantically guessable topic with opaque synthetic
biography facts. Entity handles and facts are randomly paired atomic tokenizer
items, so the mapping cannot be inferred from public word meaning. Training
and evaluation use disjoint prompt templates; held-out entities are never
included in private training.

Run the complete prepare -> freeze-core train -> evaluate workflow:

    keyed-gram phase-a --config configs/phase_a.yaml --output-dir artifacts/phase_a --device cuda

The freeze-core baseline resets the auxiliary branch to a zero-output adapter,
updates only its 1,578,496 parameters on private facts, and uses core-only
SimpleStories logits as the public teacher. It verifies the non-auxiliary hash
before saving. The evaluation reports:

- eight-way fact candidate accuracy and its 12.5% random baseline;
- answer-token accuracy, answer NLL, and free-generation exact match;
- separate memorized and never-exposed held-out entity results;
- seen-template versus unseen-paraphrase accuracy;
- Capability Dependency Ratio (CDR), including out-of-range anomaly flags;
- full versus core-only public loss over eight fixed SimpleStories topics.

Phase A passes only when unseen-paraphrase full accuracy is at least 80%,
core-only remains within five percentage points of chance, CDR is at least
0.70, and public loss increases by no more than 5%. Failure does not trigger a
key/attack matrix; it routes the work to residual or adversarial localization.

## Stage B: paraphrase-invariant residualization

Stage B uses the same random facts but separates the prompt split globally for
all entities: six training templates, two validation templates, and four fully
held-out test templates. It additionally reports context-provided unseen
entities separately from entities whose random facts were never exposed.

Run the complete B0-B3 ablation:

    keyed-gram stage-b --config configs/stage_b.yaml --output-dir artifacts/stage_b --device cuda

The variants are trained independently from the same base checkpoint:

- B0: freeze-core + private answer CE.
- B1: small trainable per-layer residual scales + public KL/residual nulling.
- B2: B1 + symmetric candidate-distribution KL across paraphrases.
- B3: B2 + normalized residual InfoNCE across paraphrases of the same fact.

Private and public optimizer steps alternate 1:1. The run records their aux
gradient cosine instead of hiding objective conflict inside a summed loss.
Evaluation adds seen/validation/unseen accuracy, worst-template accuracy,
template standard deviation, AnswerAgreement, AccessGap, bounded LocScore,
AntiSignal, full/core public KL and top-1 agreement, and per-layer
private/public residual selectivity.

The B-pass gate requires at least 95% seen accuracy, 80% unseen accuracy, 90%
AnswerAgreement, core-only no more than five points above chance, no more than
5% public NLL increase, and an exactly unchanged core hash. Failure keeps the
security matrix disabled.

## Stage C1: frozen-core semantic-interface diagnostic

Stage C1 performs no training and keeps the auxiliary branch off. It extracts,
at every layer, the last-prompt-token normalized core vector that the auxiliary
MLP would actually receive. The six train templates form a retrieval database;
the disjoint validation and test templates are queries for the same 48 facts.

Run:

    keyed-gram stage-c1 --config configs/stage_c1.yaml --output-dir artifacts/stage_c1 --device cuda

The diagnostic reports strict row-level 1-NN and fact-centroid retrieval,
same-fact versus same-template cosine geometry, fact/entity/relation silhouette,
and deterministic linear ridge probes for entity ID, relation ID, and answer
token. A 75% row-level 1-NN gate means the core already exposes a usable
template-invariant interface; below 50% routes the work to a query canonicalizer
before PCGrad or further residual losses.

## Stage C2: query canonicalization

Stage C2 freezes the complete GRAM model and trains only a lightweight query
canonicalizer. It explicitly pools entity-span tokens, non-entity question
tokens, and the answer position from core layers 4, 6, and 8. No private
memory, LM answer injection, public objective, or key is present in this stage.

Run the complete Q0-Q4 ablation:

    keyed-gram stage-c2 --config configs/stage_c2.yaml --output-dir artifacts/stage_c2 --device cuda

Before C2.1 changes the objective, run the zero-training relation-source audit.
It compares every selected-layer combination of the non-entity question pool,
answer-position pool, their concatenation, and the learned Q3 relation input and
output. The current test split is explicitly treated as development; sealed
confirmation data is never read by this command.

    keyed-gram stage-c21-audit --config configs/stage_c21.yaml --output-dir artifacts/stage_c21/audit

Seal the final confirmation templates once before C2.1 model selection. The
selected prompts, private answers, and random selection seed stay under ignored
`data/`; only their hashes and the zero-access protocol record are versioned.

    keyed-gram stage-c21-seal-confirmation --config configs/stage_c21.yaml

Run the fixed R0-R4 C2.1 ablation after the audit and seal are present:

    keyed-gram stage-c21 --config configs/stage_c21.yaml --output-dir artifacts/stage_c21 --device cuda

R0 imports Q3 exactly. R1 adds strict cross-entity/cross-template relation
SupCon, R2 moves a 0.05 template adversary to normalized `z_r`, R3 adds the
relation-first curriculum, and R4 adds normalized gated fusion. The command
records relation geometry, an entity/relation/template information matrix for
both branches, fusion contribution norms, and the ten strict C3 readiness
gates. Validation alone selects checkpoints and the final variant; the legacy
test split is development-only, and local confirmation rows remain unread.

The completed first-round results are documented in `PHASE_C21_REPORT.md`. No
variant passed the development gates, so C3 remains disabled and the next
experiment is answer-free public relation-paraphrase supervision.

Q0 reproduces the best C1 last-token baseline. Q1 uses fact-level supervised
contrastive learning; Q2 adds factorized entity/relation branches and relation
classification; Q3 adds entity classification; Q4 adds a linearly warmed
template adversary. Structured batches always contain same-entity/different-
relation and same-relation/different-entity hard negatives. Checkpoints are
selected only with validation metrics, and test templates are evaluated once
after selection.

The stage advances to private memory only if the selected variant passes all
partial gates: row 1-NN, centroid retrieval, MRR, fact silhouette, fact-over-
template margin, entity/relation accuracy, and template suppression. Passing
retrieval alone is recorded as geometric progress but does not authorize C3.

## Phase 2 workflow

Harden a locked checkpoint with correct-key private learning, wrong-key core
distillation, and core retention:

    keyed-gram key-aware-train --locked artifacts/checkpoints/main/gram_locked.pt --key artifacts/keys/private_key.json --data-dir data/stories --config configs/key_aware.yaml --output-dir artifacts/checkpoints/key_aware

The output remains in locked storage order. The companion hardened key keeps
the same swap mapping but updates the verification hash to the hardened model.

Create the deterministic 400-person memorization dataset and evaluate exact
match/token accuracy:

    keyed-gram prepare-bios --destination data/stories --num-people 400
    keyed-gram evaluate-bios --checkpoint artifacts/checkpoints/key_aware/gram_key_aware_locked.pt --key artifacts/checkpoints/key_aware/hardened_private_key.json --evaluation-jsonl data/stories/bios_eval.jsonl --output artifacts/results/bios.json

## Interfaces and invariants

- Clean checkpoints contain only format_version, model, model_config, and
  labels. Optimizer state is stored only in explicitly
  named training_state files.
- Public locked checkpoints never contain the key, a key digest, or the
  original auxiliary hash.
- The key swaps matching c_fc rows, c_fc biases, and c_proj columns. It never
  changes c_proj bias, attention, embeddings, norms, LM head, or the core MLP.
- apply_key is pure by default and operates on CPU tensors. Applying the same
  full pair-swap key twice restores every tensor exactly.
- The Key-aware forward path assembles a differentiable view from canonical
  parameters. Forward evaluation cannot alter the stored state dictionary.

## Outputs

Evaluation writes config_results.csv, wrong_key_results.csv,
partial_key_results.csv, evaluation.json, acceptance.json, and key_stats.json.
Attack and Key-aware runs write both CSV histories and JSON records with seeds
and budgets. Recovery values are deliberately not clipped to the unit interval.

The automatic acceptance report implements the planned thresholds: exact hash
restoration, at least 99% correct-key recovery, at most 10% median wrong-key
recovery, no wrong key above 30%, below 20% mean 50%-partial recovery, at most
5% core loss increase, and below 80% early attack recovery.

## Tests

Run:

    python -m pytest

The suite covers invalid keys, cross-layer constraints, self-inversion, core
immutability, wrong-key failure, checkpoint hygiene, keyed-view equivalence,
metrics, tiny training, evaluation, and one Key-aware optimization step.
