# C2.4b v2.1 single-AI semantic-audit rubric

Review each phrase family using only its phrase, proposed label/type, the three public
relation definitions, and the sample-type definitions below. Do not inspect router
predictions, evidence scores, logits, thresholds, calibration output, retrieval output,
private answers, memory contents, development results, or either locked-audit result.

- `known`: exactly one of `registry_id`, `city_code`, or `access_code` is sufficiently
  supported by the phrase.
- `ambiguous`: at least two target relations remain reasonable readings of the phrase;
  explicit words such as `unknown`, `free`, `absent`, `unspecified`, or `missing` must
  not be the sole reason for ambiguity.
- `unrelated`: the requested property is outside all three target relations, including
  when it shares carrier words such as `code`, `key`, `reference`, `token`,
  `identifier`, `credential`, `serial`, `access`, or `location`.

For every family, record the suggested label and type, an optional uncalibrated
confidence, construct validity, shortcut flags, a concise semantic note, and one of
`passed`, `failed_requires_revision`, or `boundary_retained`. Any suggested label/type
change fails the immutable benchmark version; do not overlay or rewrite labels during
calibration or evaluation.

This is a single-model, same-project exploratory audit. It is not human review, not
independent external validation, and not evidence of deployment or cryptographic
safety.
