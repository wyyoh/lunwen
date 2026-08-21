from __future__ import annotations

import pytest

from keyed_gram.tool_effects.shield import (
    FiniteSafetyGame,
    synthesize_maximal_shield,
)
from keyed_gram.tool_effects.types import ToolEffectError


def test_exact_shield_blocks_bad_and_keeps_all_safe_actions() -> None:
    game = FiniteSafetyGame(
        states=frozenset({"s0", "safe", "bad"}),
        initial_states=frozenset({"s0"}),
        actions=("allow-safe", "allow-bad"),
        transitions={
            ("s0", "allow-safe"): frozenset({"safe"}),
            ("s0", "allow-bad"): frozenset({"bad"}),
        },
        forbidden_states=frozenset({"bad"}),
        terminal_states=frozenset({"safe", "bad"}),
    )
    shield = synthesize_maximal_shield(game)
    assert shield.actions_for("s0") == ("allow-safe",)
    assert shield.maximal_permissive is True


def test_nondeterministic_action_is_blocked_if_one_successor_is_bad() -> None:
    game = FiniteSafetyGame(
        states=frozenset({"s0", "safe", "bad"}),
        initial_states=frozenset({"s0"}),
        actions=("nondet",),
        transitions={("s0", "nondet"): frozenset({"safe", "bad"})},
        forbidden_states=frozenset({"bad"}),
        terminal_states=frozenset({"safe", "bad"}),
    )
    shield = synthesize_maximal_shield(game)
    assert "s0" not in shield.winning_states


def test_all_safe_actions_are_retained() -> None:
    game = FiniteSafetyGame(
        states=frozenset({"s0", "a", "b"}),
        initial_states=frozenset({"s0"}),
        actions=("x", "y"),
        transitions={
            ("s0", "x"): frozenset({"a"}),
            ("s0", "y"): frozenset({"b"}),
        },
        forbidden_states=frozenset(),
        terminal_states=frozenset({"a", "b"}),
    )
    assert synthesize_maximal_shield(game).actions_for("s0") == ("x", "y")


def test_malformed_transition_is_rejected() -> None:
    with pytest.raises(ToolEffectError):
        FiniteSafetyGame(
            states=frozenset({"s0"}),
            initial_states=frozenset({"s0"}),
            actions=("x",),
            transitions={("s0", "x"): frozenset({"absent"})},
            forbidden_states=frozenset(),
            terminal_states=frozenset(),
        )
