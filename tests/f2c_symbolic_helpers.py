"""F2C 单元测试仅使用 train/calibration 的非正式 fixture。"""

from __future__ import annotations

from functools import lru_cache

from keyed_gram.authsynth_symbolic_shared import GrammarLimits
from keyed_gram.authsynth_symbolic_verifier.benchmark import (
    generate_legacy_symbolic_fixtures,
)
from keyed_gram.authsynth_symbolic_verifier.hidden_ir import HiddenSymbolicCase

GRAMMAR = GrammarLimits(4, 5, 16, True)


@lru_cache(maxsize=1)
def smoke_cases() -> tuple[HiddenSymbolicCase, ...]:
    return generate_legacy_symbolic_fixtures(
        "f2c-unit-only-train-calibration",
        grammar=GRAMMAR,
        replay_budget=16,
        solver_timeout_ms=3000,
    )


def case_for(category: str, *, split: str = "train") -> HiddenSymbolicCase:
    return next(
        case
        for case in smoke_cases()
        if case.mutation_category == category and case.split == split
    )
