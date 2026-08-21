"""Analyzer 可用的窄 sandbox replay protocol。"""

from __future__ import annotations

from typing import Protocol

from keyed_gram.authsynth_shared import QuerySpec, ReplayResult


class ReplayClient(Protocol):
    def replay(self, case_handle: str, query: QuerySpec) -> ReplayResult:
        """执行一个显式 query；不得返回实现、标签或未执行路径。"""
