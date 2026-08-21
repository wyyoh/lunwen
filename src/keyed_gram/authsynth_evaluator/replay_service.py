"""窄 sandbox replay service；隐藏实现永不跨出本模块。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from keyed_gram.authsynth_shared import QuerySpec, ReplayResult

from .cases import HiddenEvaluatorCase


class InMemorySandboxReplay:
    """有限 synthetic sandbox；接口形状与后续进程/容器服务一致。"""

    def __init__(self, cases: Sequence[HiddenEvaluatorCase]) -> None:
        self._cases = {
            case.analyzer_input.case_handle: case.replay_map() for case in cases
        }
        self._counts: Counter[tuple[str, str]] = Counter()

    def replay(self, case_handle: str, query: QuerySpec) -> ReplayResult:
        try:
            result = self._cases[case_handle][query.query_id]
        except KeyError as exc:
            raise ValueError("未知或跨 case replay query") from exc
        self._counts[(case_handle, query.query_id)] += 1
        return result

    def query_count(self, case_handle: str, query_id: str) -> int:
        return self._counts[(case_handle, query_id)]

    def total_query_count(self) -> int:
        return sum(self._counts.values())
