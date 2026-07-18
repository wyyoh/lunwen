# Result schema

All result rows include a configuration name and deterministic seed/budget
fields where applicable.

Primary quantitative fields:

- private_loss and private_perplexity
- core_loss and core_perplexity
- recovery and recovery_outside_unit_interval
- attack_recovery and attack_recovery_outside_unit_interval
- inference_latency_ms
- aux_parameter_ratio, key_size_bytes, lock_time_ms, restore_time_ms

Loss-based recovery is defined as:

    (core_only_loss - configuration_loss)
    / (core_only_loss - original_loss)

Attack recovery is defined as:

    (locked_loss - attacked_loss)
    / (locked_loss - original_loss)

Neither value is clipped. A zero denominator is treated as a failed
localization/locking precondition rather than silently converted to zero.

Phase-A fact-localization outputs additionally include:

- candidate_accuracy and random_candidate_accuracy
- answer_token_accuracy, answer_nll, and exact_match
- memorized and heldout splits
- seen_prompt_metrics for training-template diagnostics
- capability_dependency_ratio and
  capability_dependency_ratio_outside_unit_interval
- public.relative_loss_increase

Phase-A CDR is:

    (full_memorized_accuracy - core_only_memorized_accuracy)
    / (full_memorized_accuracy - random_candidate_accuracy)

It is not clipped. Values outside [0, 1] are retained and flagged because
finite-sample core accuracy can fall below the theoretical random baseline.

Stage-B ablation rows add:

- seen_accuracy, validation_accuracy, and unseen_accuracy
- worst_test_template_accuracy and test_template_accuracy_std
- answer_agreement and all_templates_correct_fraction
- access_gap, bounded localization_score, and anti_signal
- public absolute/relative NLL difference, KL, perplexity ratio, and top-1
  token agreement
- per-layer private_residual_ratio, public_residual_ratio, and selectivity
- mean_private_public_gradient_cosine and negative_gradient_cosine_fraction

The bounded localization score is zero when full is not above chance;
otherwise it is `1 - min(core_excess / full_excess, 1)`. AntiSignal is reported
separately as `max(chance - core_accuracy, 0)` so below-chance core behavior is
not hidden by clipping.

Stage-C1 layer rows add:

- strict cross-template row_1nn_accuracy and fact-centroid top-1/MRR
- same_fact_cross_template_cosine and different_fact_same_template_cosine
- fact-over-template and entity/relation hard-negative margins
- cosine silhouette for fact ID, entity ID, and relation ID
- cross-template linear ridge-probe accuracy for entity ID, relation ID, and
  answer token, with the corresponding class-chance accuracy

The retrieval database contains only Stage-B train templates. Validation and
test template IDs are globally disjoint from that database. The decision gate
uses row-level test 1-NN over 48 facts, whose random baseline is 1/48.

Stage-C2 ablation rows add:

- validation-selected checkpoint step and validation row-level 1-NN
- test row-level 1-NN, fact-centroid top-1, and MRR
- fact silhouette and same-fact-over-same-template cosine margin
- entity, relation, answer-token ridge probes on the final query vector
- direct entity/relation auxiliary-head accuracy for trained variants
- template leakage probe accuracy across a held-out entity subset
- partial_passed and strong_passed gate decisions

Q0 is the exact C1 core-only baseline. Q1-Q4 contain no memory, answer
injection, public loss, or key. The stage-level decision uses the variant chosen
by validation gate count and retrieval/probe tuple; test results never choose a
checkpoint or variant.

Stage-C2.1 ablation rows add:

- the exact-Q3 import flag and validation-selected checkpoint step
- validation and development strict-gate counts
- relation-head and `z_r` ridge-probe accuracy
- relation-template margin, using same-relation/different-entity/different-
  template positives and same-entity/same-template/different-relation negatives
- branch information matrices that probe entity, relation, and template from
  normalized `z_e` and `z_r`
- entity, relation, and interaction contribution norms, shares, and learned
  fusion scales
- a `development_ready_for_confirmation` decision distinct from C3 eligibility

R0 imports Stage-C2 Q3 without retraining. R1-R4 contain no memory or answer
loss. Checkpoint and variant selection use validation only; the old test split
is labeled development. `c3_eligible` remains false until one selected variant
passes all ten gates on the one-time sealed confirmation evaluation.

Stage-C2.2 ablation rows add:

- the exact-C2.1-base flag and private replay weight
- private-validation, public-validation, and development gate counts
- public validation relation-head/probe accuracy and public train probe fit
- cross-phrase public relation margin and public template leakage
- the same private development relation, retrieval, geometry, entity, and
  template metrics used by C2.1

The public corpus manifest certifies that rows contain no answers, candidates,
or entity-to-private-value mappings. P0-P2 selection uses private validation and
lexically disjoint public validation only. Development and sealed confirmation
never select a checkpoint or variant.
