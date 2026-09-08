"""结构化 state diff 与 bounded delayed events 的窄 replay service。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from keyed_gram.authsynth_symbolic_shared import (
    ConcreteAssignment,
    StateChange,
    StructuredStateDiff,
    SymbolicReplayResult,
)

from .hidden_ir import HiddenSymbolicCase


class SymbolicSandboxReplay:
    def __init__(self, cases: Sequence[HiddenSymbolicCase]) -> None:
        self._cases = {item.analyzer_input.case_handle: item for item in cases}
        self._counts: Counter[str] = Counter()
        self._assignment_digests: dict[str, list[str]] = {}

    def replay(
        self, case_handle: str, assignment: ConcreteAssignment
    ) -> SymbolicReplayResult:
        try:
            case = self._cases[case_handle]
        except KeyError as exc:
            raise ValueError("未知 symbolic replay handle") from exc
        implementation = case.implementation
        events, updates = implementation.execute(assignment)
        changes = []
        for field, after in sorted(updates.items()):
            before = assignment.value("state", field)
            if before != after:
                changes.append(StateChange("record", "state", field, before, after))
        immediate = tuple(item for item in events if item.phase == "immediate")
        delayed = tuple(item for item in events if item.phase == "delayed")
        self._counts[case_handle] += 1
        self._assignment_digests.setdefault(case_handle, []).append(assignment.digest)
        return SymbolicReplayResult(
            assignment=assignment,
            immediate_events=immediate,
            delayed_events=delayed,
            state_diff=StructuredStateDiff(tuple(changes)),
            bounded_quiescence_reached=(
                implementation.delayed_depth <= 2
                and implementation.bounded_loop_max is not None
            ),
            instrumentation_coverage=implementation.instrumentation_coverage,
            observed_version_digest=implementation.version_digest,
            exit_status="ok",
        )

    def query_count(self, case_handle: str) -> int:
        return self._counts[case_handle]

    def assignment_digests(self, case_handle: str) -> tuple[str, ...]:
        return tuple(self._assignment_digests.get(case_handle, ()))
