"""从当前盲抽象合成 fail-closed、有限状态 Shield。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from keyed_gram.authsynth_shared import (
    AnalyzerCaseInput,
    EventRecord,
    ShieldDecision,
    canonical_digest,
)
from keyed_gram.tool_effects.shield import FiniteSafetyGame, synthesize_maximal_shield


def _state_event_prefixes(
    case: AnalyzerCaseInput,
    traces: Mapping[str, tuple[EventRecord, ...]],
) -> dict[str, tuple[tuple[EventRecord, ...], ...]]:
    """有界展开 workflow；循环最多展开到两倍状态数。"""

    outgoing: dict[str, list] = defaultdict(list)
    for edge in case.workflow_edges:
        outgoing[edge.source_state].append(edge)
    result: dict[str, set[tuple[EventRecord, ...]]] = defaultdict(set)
    frontier = [(state, (), ()) for state in sorted(case.initial_states)]
    bound = max(1, len({edge.source_state for edge in case.workflow_edges}) * 2 + 2)
    while frontier:
        state, events, visited_edges = frontier.pop()
        result[state].add(events)
        if len(visited_edges) >= bound:
            continue
        for edge in sorted(
            outgoing.get(state, ()), key=lambda item: (item.action_id, item.query_id)
        ):
            marker = (
                edge.source_state,
                edge.action_id,
                edge.query_id,
                edge.successor_state,
            )
            if visited_edges.count(marker) >= 2:
                continue
            next_events = (*events, *traces.get(edge.query_id, ()))
            frontier.append(
                (edge.successor_state, next_events, (*visited_edges, marker))
            )
    return {state: tuple(sorted(values)) for state, values in result.items()}


def synthesize_blind_shield(
    case: AnalyzerCaseInput,
    traces: Mapping[str, tuple[EventRecord, ...]],
    unknown_queries: frozenset[str],
) -> tuple[tuple[ShieldDecision, ...], str, bool]:
    """未知 outcome 被映射到 forbidden sink，不能用缺失证据换取许可。"""

    states = set(case.initial_states) | set(case.terminal_states)
    actions: set[str] = set()
    transitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    forbidden: set[str] = set()
    for edge in case.workflow_edges:
        states.update((edge.source_state, edge.successor_state))
        actions.add(edge.action_id)
        if edge.query_id in unknown_queries:
            sink = f"unknown-{edge.query_id}"
            states.add(sink)
            forbidden.add(sink)
            transitions[(edge.source_state, edge.action_id)].add(sink)
        else:
            transitions[(edge.source_state, edge.action_id)].add(edge.successor_state)

    if case.trusted_safety_spec is None:
        forbidden.update(state for state in states if state not in case.initial_states)
    else:
        for state, prefixes in _state_event_prefixes(case, traces).items():
            if any(
                case.trusted_safety_spec.violation_codes(events) for events in prefixes
            ):
                forbidden.add(state)

    game = FiniteSafetyGame(
        states=frozenset(states),
        initial_states=case.initial_states,
        actions=tuple(sorted(actions)),
        transitions={key: frozenset(value) for key, value in transitions.items()},
        forbidden_states=frozenset(forbidden),
        terminal_states=case.terminal_states
        | frozenset(state for state in states if state.startswith("unknown-")),
    )
    shield = synthesize_maximal_shield(game)
    decisions = tuple(
        ShieldDecision(state=state, allowed_actions=allowed)
        for state, allowed in shield.allowed_actions
    )
    initial_safe = case.initial_states.issubset(shield.winning_states)
    digest = canonical_digest([item.to_dict() for item in decisions])
    return decisions, digest, initial_safe


def cumulative_violation_states(
    case: AnalyzerCaseInput,
    traces: Mapping[str, tuple[EventRecord, ...]],
) -> frozenset[str]:
    """返回只有累计 trace 语义才能识别的 forbidden workflow states。"""

    if case.trusted_safety_spec is None:
        return frozenset()
    return frozenset(
        state
        for state, prefixes in _state_event_prefixes(case, traces).items()
        if any(case.trusted_safety_spec.violation_codes(events) for events in prefixes)
    )
