"""F2A 的有界状态探索器。

探索器故意同时编码两个不安全语义和对应的安全语义，以产生可复现的最小
counterexample，而不是把预期结论写死成单个布尔值。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class LaunderingState:
    integrity: int
    authority_uses: int
    summarized: bool = False
    trusted_tool_echoed: bool = False


@dataclass(frozen=True)
class MergeState:
    authority_copies: tuple[int, ...]
    committed_effects: int


@dataclass(frozen=True)
class SearchNode:
    state: object
    trace: tuple[str, ...]


@dataclass(frozen=True)
class BoundedSearchResult:
    property_name: str
    bound: int
    unsafe_counterexample_found: bool
    unsafe_trace: tuple[str, ...]
    safe_counterexample_found: bool
    safe_trace: tuple[str, ...]
    unsafe_visited_state_count: int
    safe_visited_state_count: int

    def canonical(self) -> dict[str, object]:
        return {
            "bound": self.bound,
            "property_name": self.property_name,
            "safe_counterexample_found": self.safe_counterexample_found,
            "safe_trace": list(self.safe_trace),
            "safe_visited_state_count": self.safe_visited_state_count,
            "unsafe_counterexample_found": self.unsafe_counterexample_found,
            "unsafe_trace": list(self.unsafe_trace),
            "unsafe_visited_state_count": self.unsafe_visited_state_count,
        }


Transition = Callable[[object], Iterable[tuple[str, object]]]
Violation = Callable[[object], bool]


def _bfs(
    initial: object,
    transition: Transition,
    violation: Violation,
    bound: int,
) -> tuple[bool, tuple[str, ...], int]:
    queue = deque((SearchNode(initial, ()),))
    seen = {initial}
    while queue:
        node = queue.popleft()
        if violation(node.state):
            return True, node.trace, len(seen)
        if len(node.trace) >= bound:
            continue
        for operation, successor in transition(node.state):
            if successor in seen:
                continue
            seen.add(successor)
            queue.append(SearchNode(successor, node.trace + (operation,)))
    return False, (), len(seen)


def _unsafe_laundering(state: object) -> Iterable[tuple[str, object]]:
    assert isinstance(state, LaunderingState)
    if not state.summarized:
        yield "llm_summary", LaunderingState(
            integrity=state.integrity,
            authority_uses=state.authority_uses,
            summarized=True,
            trusted_tool_echoed=state.trusted_tool_echoed,
        )
    if state.summarized and not state.trusted_tool_echoed:
        yield "trusted_tool_echo_marks_trusted", LaunderingState(
            integrity=2,
            authority_uses=state.authority_uses,
            summarized=True,
            trusted_tool_echoed=True,
        )
    if state.integrity == 2 and state.authority_uses == 0:
        yield "derive_authority_from_integrity", LaunderingState(
            integrity=2,
            authority_uses=1,
            summarized=state.summarized,
            trusted_tool_echoed=state.trusted_tool_echoed,
        )


def _safe_laundering(state: object) -> Iterable[tuple[str, object]]:
    assert isinstance(state, LaunderingState)
    if not state.summarized:
        yield "llm_summary_preserves_influence", LaunderingState(
            integrity=state.integrity,
            authority_uses=state.authority_uses,
            summarized=True,
            trusted_tool_echoed=state.trusted_tool_echoed,
        )
    if state.summarized and not state.trusted_tool_echoed:
        yield "trusted_tool_echo_preserves_no_authority", LaunderingState(
            integrity=2,
            authority_uses=state.authority_uses,
            summarized=True,
            trusted_tool_echoed=True,
        )
    # 安全语义没有 data/integrity -> authority 的转换规则。


def find_authority_laundering(bound: int = 4) -> BoundedSearchResult:
    initial = LaunderingState(integrity=0, authority_uses=0)
    violation = lambda value: (
        isinstance(value, LaunderingState) and value.authority_uses > 0
    )
    unsafe = _bfs(initial, _unsafe_laundering, violation, bound)
    safe = _bfs(initial, _safe_laundering, violation, bound)
    return BoundedSearchResult(
        property_name="authority_origination",
        bound=bound,
        unsafe_counterexample_found=unsafe[0],
        unsafe_trace=unsafe[1],
        unsafe_visited_state_count=unsafe[2],
        safe_counterexample_found=safe[0],
        safe_trace=safe[1],
        safe_visited_state_count=safe[2],
    )


def _unsafe_merge(state: object) -> Iterable[tuple[str, object]]:
    assert isinstance(state, MergeState)
    if len(state.authority_copies) == 1:
        yield "naive_branch_copy", MergeState(
            authority_copies=(
                state.authority_copies[0],
                state.authority_copies[0],
            ),
            committed_effects=state.committed_effects,
        )
    for index, uses in enumerate(state.authority_copies):
        if uses > 0:
            updated = list(state.authority_copies)
            updated[index] -= 1
            yield f"branch_{index}_consume", MergeState(
                authority_copies=tuple(updated),
                committed_effects=state.committed_effects + 1,
            )


def _safe_merge(state: object) -> Iterable[tuple[str, object]]:
    assert isinstance(state, MergeState)
    # 预算为 1，合法 split 只能把一次 use 分给一个分支；零预算分支不能执行。
    if len(state.authority_copies) == 1 and state.committed_effects == 0:
        yield "linear_branch_partition", MergeState(
            authority_copies=(1, 0),
            committed_effects=state.committed_effects,
        )
    for index, uses in enumerate(state.authority_copies):
        if uses > 0:
            updated = list(state.authority_copies)
            updated[index] -= 1
            yield f"branch_{index}_consume", MergeState(
                authority_copies=tuple(updated),
                committed_effects=state.committed_effects + 1,
            )


def find_naive_merge_amplification(bound: int = 4) -> BoundedSearchResult:
    initial = MergeState(authority_copies=(1,), committed_effects=0)
    violation = lambda value: (
        isinstance(value, MergeState) and value.committed_effects > 1
    )
    unsafe = _bfs(initial, _unsafe_merge, violation, bound)
    safe = _bfs(initial, _safe_merge, violation, bound)
    return BoundedSearchResult(
        property_name="merge_confinement",
        bound=bound,
        unsafe_counterexample_found=unsafe[0],
        unsafe_trace=unsafe[1],
        unsafe_visited_state_count=unsafe[2],
        safe_counterexample_found=safe[0],
        safe_trace=safe[1],
        safe_visited_state_count=safe[2],
    )


def run_bounded_model(bound: int = 4) -> dict[str, object]:
    results = (
        find_authority_laundering(bound),
        find_naive_merge_amplification(bound),
    )
    return {
        "schema_version": 1,
        "exploration_bound": bound,
        "results": [item.canonical() for item in results],
        "bounded_model_counterexample_for_authority_laundering": (
            results[0].unsafe_counterexample_found
            and not results[0].safe_counterexample_found
        ),
        "bounded_model_counterexample_for_naive_merge": (
            results[1].unsafe_counterexample_found
            and not results[1].safe_counterexample_found
        ),
    }
