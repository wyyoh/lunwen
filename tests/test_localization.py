from __future__ import annotations

from keyed_gram.localization import select_localization_candidate


def test_fallback_selects_private_gap_after_core_gate():
    rows = [
        {
            "name": "private-strong-but-core-unsafe",
            "private_perplexity_ratio": 1.50,
            "core_gate_passed": False,
            "passed": False,
        },
        {
            "name": "eligible-weaker",
            "private_perplexity_ratio": 1.03,
            "core_gate_passed": True,
            "passed": False,
        },
        {
            "name": "eligible-stronger",
            "private_perplexity_ratio": 1.08,
            "core_gate_passed": True,
            "passed": False,
        },
    ]
    assert select_localization_candidate(rows)["name"] == "eligible-stronger"
