"""仅 evaluator 可见的 hidden case 与 ground truth。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from keyed_gram.authsynth_shared import (
    AnalyzerCaseInput,
    ReplayResult,
    canonical_digest,
)


@dataclass(frozen=True)
class HiddenEvaluatorCase:
    analyzer_input: AnalyzerCaseInput
    split: str
    tool_family: str
    mutation_family: str
    workflow_template: str
    version_lineage: str
    mutation_category: str
    omission_types: tuple[str, ...]
    concrete_replays: tuple[tuple[str, ReplayResult], ...]
    benign_state_actions: tuple[tuple[str, str], ...]
    attack_state_actions: tuple[tuple[str, str], ...]
    expected_policy_omission: bool
    expected_drift: bool
    expected_unknown: bool
    clean_control: bool

    def __post_init__(self) -> None:
        query_ids = {item.query_id for item in self.analyzer_input.query_catalog}
        replay_ids = [item[0] for item in self.concrete_replays]
        if len(replay_ids) != len(set(replay_ids)) or set(replay_ids) != query_ids:
            raise ValueError("hidden replay inventory 与 analyzer query catalog 不一致")

    def replay_map(self) -> dict[str, ReplayResult]:
        return dict(self.concrete_replays)

    @property
    def public_case_id(self) -> str:
        return self.analyzer_input.public_case_id

    def evaluator_manifest_row(self) -> dict[str, Any]:
        """只保存标签与 digest，不持久化 concrete event。"""

        return {
            "public_case_id": self.public_case_id,
            "split": self.split,
            "tool_family_digest": canonical_digest(self.tool_family),
            "mutation_family_digest": canonical_digest(self.mutation_family),
            "workflow_template_digest": canonical_digest(self.workflow_template),
            "version_lineage_digest": canonical_digest(self.version_lineage),
            "mutation_category": self.mutation_category,
            "omission_types": list(self.omission_types),
            "concrete_replay_count": len(self.concrete_replays),
            "concrete_semantics_digest": canonical_digest(
                [
                    {
                        "query_id": query_id,
                        "replay_digest": canonical_digest(result.to_dict()),
                    }
                    for query_id, result in self.concrete_replays
                ]
            ),
            "expected_policy_omission": self.expected_policy_omission,
            "expected_drift": self.expected_drift,
            "expected_unknown": self.expected_unknown,
            "clean_control": self.clean_control,
        }
