# Upstream references

This project is a clean, local research implementation of the model and
parameter interfaces described by GRAM and TLM. It does not vendor either
repository.

- GRAM repository: https://github.com/agencyenterprise/modular-pretraining
- Pinned GRAM commit: `d07d62e9869c6a969a2306e755aba7a742c317fd`
- TLM repository: https://github.com/McGill-NLP/tiered-language-models
- Pin date: 2026-07-12

The GRAM repository did not expose a LICENSE file when this pin was made.
Accordingly, this implementation is intended for local research only until
redistribution rights are clarified. TLM is a design reference and is not a
runtime dependency.

The local model was checked against `src/model/base.py`, `src/model/moe.py`,
`src/run/train/routed.py`, and the SimpleStories configuration in
`src/run/experiment/config.py` at the pinned commit. It retains the upstream
RoPE, grouped-query attention, segmentation-aware causal mask, RMSNorm,
gateless core/aux MLP layout, and routed backward semantics while removing the
S3, W&B, DDP, and compile-only training infrastructure.
