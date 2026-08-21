"""有限安全博弈上的精确最大许可 Shield。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .types import ToolEffectError, canonical_digest


@dataclass(frozen=True)
class FiniteSafetyGame:
    """Agent 选 action、环境在 successor 集中非确定选择的安全博弈。"""

    states: frozenset[str]
    initial_states: frozenset[str]
    actions: tuple[str, ...]
    transitions: Mapping[tuple[str, str], frozenset[str]]
    forbidden_states: frozenset[str]
    terminal_states: frozenset[str]

    def __post_init__(self) -> None:
        if not self.initial_states.issubset(self.states):
            raise ToolEffectError("initial state 不在 game states 中")
        if not self.forbidden_states.issubset(self.states):
            raise ToolEffectError("forbidden state 不在 game states 中")
        if not self.terminal_states.issubset(self.states):
            raise ToolEffectError("terminal state 不在 game states 中")
        action_set = set(self.actions)
        for (state, action), successors in self.transitions.items():
            if state not in self.states or action not in action_set:
                raise ToolEffectError("transition source/action 非法")
            if not successors or not successors.issubset(self.states):
                raise ToolEffectError("transition successor 非法")


@dataclass(frozen=True)
class Shield:
    winning_states: frozenset[str]
    allowed_actions: tuple[tuple[str, tuple[str, ...]], ...]
    maximal_permissive: bool
    digest: str

    def actions_for(self, state: str) -> tuple[str, ...]:
        return dict(self.allowed_actions).get(state, ())


def synthesize_maximal_shield(game: FiniteSafetyGame) -> Shield:
    """以最大不动点计算环境非确定性下的最大许可安全控制器。"""

    winning = set(game.states - game.forbidden_states)
    changed = True
    while changed:
        changed = False
        for state in tuple(sorted(winning)):
            if state in game.terminal_states:
                continue
            safe_action_exists = any(
                successors.issubset(winning)
                for (source, _), successors in game.transitions.items()
                if source == state
            )
            if not safe_action_exists:
                winning.remove(state)
                changed = True
    allowed: list[tuple[str, tuple[str, ...]]] = []
    for state in sorted(winning):
        actions = tuple(
            sorted(
                action
                for (source, action), successors in game.transitions.items()
                if source == state and successors.issubset(winning)
            )
        )
        allowed.append((state, actions))
    payload = {
        "winning_states": sorted(winning),
        "allowed_actions": [
            {"state": state, "actions": list(actions)} for state, actions in allowed
        ],
    }
    return Shield(
        winning_states=frozenset(winning),
        allowed_actions=tuple(allowed),
        maximal_permissive=True,
        digest=canonical_digest(payload),
    )
