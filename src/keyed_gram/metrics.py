from __future__ import annotations

import math
import statistics
from typing import Iterable


def perplexity(loss: float) -> float:
    try:
        return math.exp(float(loss))
    except OverflowError:
        return math.inf


def recovery(core_only_loss: float, config_loss: float, original_loss: float) -> float:
    denominator = core_only_loss - original_loss
    if abs(denominator) < 1e-12:
        raise ValueError("recovery is undefined because the localization gap is zero")
    return (core_only_loss - config_loss) / denominator


def attack_recovery(locked_loss: float, attacked_loss: float, original_loss: float) -> float:
    denominator = locked_loss - original_loss
    if abs(denominator) < 1e-12:
        raise ValueError("attack recovery is undefined because the lock gap is zero")
    return (locked_loss - attacked_loss) / denominator


def relative_loss_increase(reference_loss: float, config_loss: float) -> float:
    if abs(reference_loss) < 1e-12:
        raise ValueError("relative loss increase is undefined for a zero reference")
    return (config_loss - reference_loss) / reference_loss


def outside_unit_interval(value: float) -> bool:
    return value < 0 or value > 1


def evaluate_acceptance(
    *,
    correct_recovery: float,
    restored_hash_matches: bool,
    core_logits_equal: bool,
    locked_recovery: float,
    wrong_recoveries: Iterable[float],
    partial_50_recoveries: Iterable[float],
    max_core_loss_increase: float,
    early_attack_recoveries: Iterable[float] = (),
) -> dict:
    wrong = list(wrong_recoveries)
    partial = list(partial_50_recoveries)
    early = list(early_attack_recoveries)
    checks = {
        "correct_key_recovery": correct_recovery >= 0.99,
        "restored_hash": bool(restored_hash_matches),
        "core_logits_equal": bool(core_logits_equal),
        "locked_recovery": locked_recovery <= 0.10,
        "wrong_key_median": bool(wrong) and statistics.median(wrong) <= 0.10,
        "wrong_key_max": bool(wrong) and max(wrong) <= 0.30,
        "partial_50_mean": bool(partial) and statistics.fmean(partial) < 0.20,
        "core_loss_increase": max_core_loss_increase <= 0.05,
        "early_attack_resistance": not early or max(early) < 0.80,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "correct_recovery": correct_recovery,
            "locked_recovery": locked_recovery,
            "wrong_median": statistics.median(wrong) if wrong else None,
            "wrong_max": max(wrong) if wrong else None,
            "partial_50_mean": statistics.fmean(partial) if partial else None,
            "max_core_loss_increase": max_core_loss_increase,
            "early_attack_max": max(early) if early else None,
        },
    }
