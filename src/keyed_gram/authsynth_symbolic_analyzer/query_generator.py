"""由 verifier 差异公式生成下一 concrete assignment。"""

from __future__ import annotations

from keyed_gram.authsynth_symbolic_shared import (
    ContractVerifier,
    SymbolicEffectContract,
    VerificationResult,
)


def request_counterexample(
    verifier: ContractVerifier,
    case_handle: str,
    candidate_contract: SymbolicEffectContract,
) -> VerificationResult:
    """Analyzer 不持有预枚举输入；每次 assignment 均来自 verifier。"""

    return verifier.check(case_handle, candidate_contract)
