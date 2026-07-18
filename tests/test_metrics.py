from __future__ import annotations

import pytest

from keyed_gram.metrics import attack_recovery, perplexity, recovery


def test_loss_based_metrics():
    assert recovery(4.0, 2.0, 2.0) == pytest.approx(1.0)
    assert recovery(4.0, 4.0, 2.0) == pytest.approx(0.0)
    assert attack_recovery(4.0, 3.0, 2.0) == pytest.approx(0.5)
    assert perplexity(0.0) == pytest.approx(1.0)


def test_zero_gap_is_rejected():
    with pytest.raises(ValueError):
        recovery(2.0, 2.0, 2.0)
