"""Analyzer freeze 后独立 materialize 的 F2B 盲测 benchmark。"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from keyed_gram.authsynth_shared import (
    AnalyzerCaseInput,
    EventRecord,
    EvidenceHypothesis,
    EvidenceOrigin,
    HighLevelSafetySpec,
    QuerySpec,
    ReplayResult,
    WorkflowEdge,
    canonical_digest,
)

from .cases import HiddenEvaluatorCase

SPLITS = ("train", "calibration", "development", "locked_test")
MUTATIONS = (
    "hidden_effect",
    "parameter_role_omission",
    "state_dependent_effect",
    "composition_omission",
    "delayed_effect",
    "implementation_drift",
)

_TOOLS = (
    ("file_read", "train"),
    ("email_send", "train"),
    ("calendar_search", "calibration"),
    ("crm_query", "calibration"),
    ("repo_commit", "development"),
    ("cloud_deploy", "development"),
    ("database_export", "locked_test"),
    ("payment_refund", "locked_test"),
)


def _opaque(seed: str, *parts: str, length: int = 24) -> str:
    digest = hashlib.sha256("|".join((seed, *parts)).encode("utf-8")).hexdigest()
    return f"h-{digest[:length]}"


def _event(
    case_id: str,
    call_id: str,
    kind: str,
    *,
    sequence: int = 0,
    phase: str = "immediate",
    tenant: str = "tenant-a",
    destination: str | None = None,
    parent: str | None = None,
) -> EventRecord:
    return EventRecord(
        event_id=f"e-{case_id}-{call_id}-{sequence}",
        call_id=call_id,
        sequence=sequence,
        phase=phase,
        kind=kind,
        tenant_id=tenant,
        resource_id=f"r-{case_id}-{call_id}",
        destination_id=destination,
        parent_event_id=parent,
        argument_roles=(("resource", f"r-{case_id}-{call_id}"),),
    )


def _result(
    query_id: str,
    events: tuple[EventRecord, ...],
    *,
    coverage: float = 1.0,
    quiescent: bool = True,
    version_digest: str,
) -> ReplayResult:
    return ReplayResult(
        query_id=query_id,
        events=events,
        state_diff_digest=canonical_digest([item.to_dict() for item in events]),
        exit_status="ok",
        bounded_quiescence_reached=quiescent,
        instrumentation_coverage=coverage,
        observed_version_digest=version_digest,
    )


def _make_case(
    seed: str,
    *,
    serial: int,
    tool_family: str,
    split: str,
    categories: tuple[str, ...],
    clean_control: bool,
    expected_unknown: bool = False,
) -> HiddenEvaluatorCase:
    case_id = f"f2b-{split}-{serial:04d}"
    handle = _opaque(seed, case_id, "handle")
    tool_id = _opaque(seed, case_id, "tool")
    declared_version = "v1"
    expected_version_digest = _opaque(seed, tool_family, "v1")
    actual_version_digest = expected_version_digest
    qids = tuple(f"q-{case_id}-{index}" for index in range(4))
    calls = tuple(f"call-{case_id}-{index}" for index in range(4))
    queries = tuple(
        QuerySpec(
            query_id=qids[index],
            opaque_tool_id=tool_id,
            call_id=calls[index],
            declared_version=declared_version,
            environment_seed=_opaque(seed, case_id, f"env-{index}"),
            arguments=(("tenant", "tenant-a"), ("resource", f"resource-{index}")),
        )
        for index in range(4)
    )
    declared_events = (
        (_event(case_id, calls[0], "read"),),
        (_event(case_id, calls[1], "write"),),
        (_event(case_id, calls[2], "read"),),
        (_event(case_id, calls[3], "commit"),),
    )
    actual_events = [tuple(items) for items in declared_events]
    omission_types: set[str] = set()
    expected_policy_omission = False

    if not clean_control:
        for category in categories:
            if category == "hidden_effect":
                actual_events[3] = (
                    *actual_events[3],
                    _event(
                        case_id,
                        calls[3],
                        "send",
                        sequence=1,
                        destination="external-net",
                    ),
                )
                omission_types.add("contract_omission")
                expected_policy_omission = True
            elif category == "parameter_role_omission":
                declared_events = (
                    *declared_events[:3],
                    (_event(case_id, calls[3], "send", destination="internal-net"),),
                )
                actual_events[3] = (
                    _event(case_id, calls[3], "send", destination="external-net"),
                )
                omission_types.add("parameter_role_omission")
                expected_policy_omission = True
            elif category == "state_dependent_effect":
                actual_events[3] = (
                    *actual_events[3],
                    _event(case_id, calls[3], "execute", sequence=1),
                )
                omission_types.add("state_dependent_omission")
                expected_policy_omission = True
            elif category == "composition_omission":
                declared_events = (
                    *declared_events[:3],
                    (
                        _event(
                            case_id, calls[3], "export", destination="internal-store"
                        ),
                    ),
                )
                actual_events[3] = tuple(declared_events[3])
                omission_types.add("composition_omission")
                expected_policy_omission = True
            elif category == "delayed_effect":
                parent = f"e-{case_id}-{calls[3]}-0"
                actual_events[3] = (
                    *actual_events[3],
                    _event(
                        case_id,
                        calls[3],
                        "send",
                        sequence=1,
                        phase="delayed",
                        destination="external-net",
                        parent=parent,
                    ),
                )
                omission_types.add("async_contract_omission")
                expected_policy_omission = True
            elif category == "implementation_drift":
                actual_version_digest = _opaque(seed, tool_family, case_id, "v2")
                actual_events[3] = (
                    *actual_events[3],
                    _event(
                        case_id,
                        calls[3],
                        "send",
                        sequence=1,
                        destination="external-net",
                    ),
                )
                omission_types.add("implementation_drift")
                expected_policy_omission = True
            else:
                raise ValueError(f"未知 mutation category：{category}")

    hypotheses: list[EvidenceHypothesis] = []
    if serial % 5 == 0:
        hypotheses.append(
            EvidenceHypothesis(
                hypothesis_id=f"hyp-{case_id}-spurious",
                query_id=qids[1],
                event=_event(
                    case_id, calls[1], "send", sequence=1, destination="external-net"
                ),
                origin=EvidenceOrigin.STATIC,
            )
        )
    if categories and serial % 3 == 0:
        hypotheses.append(
            EvidenceHypothesis(
                hypothesis_id=f"hyp-{case_id}-risk",
                query_id=qids[3],
                event=actual_events[3][-1],
                origin=EvidenceOrigin.HYPOTHESIS,
            )
        )

    coverage = [1.0] * 4
    quiescence = [True] * 4
    if expected_unknown:
        if serial % 2 == 0:
            coverage[1] = 0.75
        else:
            quiescence[3] = False

    replay_results = tuple(
        (
            qids[index],
            _result(
                qids[index],
                tuple(actual_events[index]),
                coverage=coverage[index],
                quiescent=quiescence[index],
                version_digest=actual_version_digest,
            ),
        )
        for index in range(4)
    )
    spec = HighLevelSafetySpec(
        spec_id=f"spec-{split}-{tool_family}",
        protected_tenant_id="tenant-a",
        forbidden_external_destinations=frozenset({"external-net"}),
        forbidden_effect_kinds=frozenset({"delete", "execute"}),
        forbidden_kind_combinations=(("export", "read"), ("read", "send")),
    )
    declared_policy_codes = frozenset(
        {"forbidden_external_destination", "forbidden_effect_kind"}
        if clean_control
        else ()
    )
    analyzer_input = AnalyzerCaseInput(
        case_handle=handle,
        public_case_id=case_id,
        declared_contract=tuple(
            (qids[index], tuple(declared_events[index])) for index in range(4)
        ),
        declared_policy_codes=declared_policy_codes,
        trusted_safety_spec=spec,
        query_catalog=queries,
        workflow_edges=(
            WorkflowEdge("root", "begin", qids[0], "common"),
            WorkflowEdge("common", "safe-alternative", qids[1], "safe-done"),
            WorkflowEdge("common", "primary", qids[2], "risk-mid"),
            WorkflowEdge("risk-mid", "finish", qids[3], "done"),
        ),
        initial_states=frozenset({"root"}),
        terminal_states=frozenset({"safe-done", "done"}),
        hypotheses=tuple(hypotheses),
        replay_budget=4,
        expected_version_digest=expected_version_digest,
    )
    mutation_family = f"{split}-{tool_family}-{'-'.join(categories) if categories else 'clean'}-{serial}"
    return HiddenEvaluatorCase(
        analyzer_input=analyzer_input,
        split=split,
        tool_family=tool_family,
        mutation_family=mutation_family,
        workflow_template=f"wf-{split}-{serial}",
        version_lineage=f"lineage-{split}-{tool_family}-{serial}",
        mutation_category="compound"
        if len(categories) > 1
        else (categories[0] if categories else "clean_control"),
        omission_types=tuple(sorted(omission_types)) or ("none",),
        concrete_replays=replay_results,
        benign_state_actions=(("root", "begin"), ("common", "safe-alternative")),
        attack_state_actions=(
            ("root", "begin"),
            ("common", "primary"),
            ("risk-mid", "finish"),
        ),
        expected_policy_omission=expected_policy_omission,
        expected_drift="implementation_drift" in categories,
        expected_unknown=expected_unknown or "implementation_drift" in categories,
        clean_control=clean_control,
    )


@dataclass
class BlindBenchmarkVault:
    """持有 hidden cases；phase API 防止 analyzer freeze 前打开新 locked。"""

    cases: tuple[HiddenEvaluatorCase, ...]
    _opened: dict[str, int]

    def public_rows(self, split: str) -> tuple[dict, ...]:
        return tuple(
            case.analyzer_input.to_public_dict()
            for case in self.cases
            if case.split == split
        )

    def open_for_evaluation(self, split: str) -> tuple[HiddenEvaluatorCase, ...]:
        if split not in SPLITS:
            raise ValueError("未知 split")
        if self._opened.get(split, 0) != 0:
            raise RuntimeError(f"{split} 已打开，拒绝重复")
        self._opened[split] = 1
        return tuple(case for case in self.cases if case.split == split)

    def opened_count(self, split: str) -> int:
        return self._opened.get(split, 0)


def generate_f2b_benchmark(seed: str) -> BlindBenchmarkVault:
    cases: list[HiddenEvaluatorCase] = []
    serial = 0
    # 150 个 F2A-compatible 形状，但全部属于新的 F2B namespace。
    for tool_family, split in _TOOLS:
        for category in MUTATIONS:
            for _ in range(3):
                serial += 1
                cases.append(
                    _make_case(
                        seed,
                        serial=serial,
                        tool_family=tool_family,
                        split=split,
                        categories=(category,),
                        clean_control=False,
                    )
                )
    for index in range(6):
        tool_family, split = _TOOLS[index]
        serial += 1
        cases.append(
            _make_case(
                seed,
                serial=serial,
                tool_family=tool_family,
                split=split,
                categories=(),
                clean_control=True,
            )
        )

    # 120 个 blind compound cases；组合由 split/tool/serial 唯一化。
    split_combinations = {
        "train": (
            ("hidden_effect", "delayed_effect"),
            ("parameter_role_omission", "state_dependent_effect"),
        ),
        "calibration": (
            ("composition_omission", "hidden_effect"),
            ("delayed_effect", "parameter_role_omission"),
        ),
        "development": (
            ("state_dependent_effect", "composition_omission"),
            ("implementation_drift", "hidden_effect"),
        ),
        "locked_test": (
            ("delayed_effect", "composition_omission"),
            ("implementation_drift", "parameter_role_omission"),
        ),
    }
    for tool_family, split in _TOOLS:
        combinations = split_combinations[split]
        for index in range(15):
            serial += 1
            cases.append(
                _make_case(
                    seed,
                    serial=serial,
                    tool_family=tool_family,
                    split=split,
                    categories=combinations[index % len(combinations)],
                    clean_control=False,
                    expected_unknown=index % 5 == 0,
                )
            )
    if len(cases) != 270:
        raise RuntimeError("F2B benchmark case count 不等于 270")
    return BlindBenchmarkVault(tuple(cases), {})


def collision_audit(cases: Sequence[HiddenEvaluatorCase]) -> dict:
    dimensions = {
        "tool_family": lambda item: item.tool_family,
        "mutation_family": lambda item: item.mutation_family,
        "workflow_template": lambda item: item.workflow_template,
        "version_lineage": lambda item: item.version_lineage,
        "public_case_id": lambda item: item.public_case_id,
    }
    collisions: dict[str, list[str]] = {}
    for name, getter in dimensions.items():
        seen: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            seen[getter(case)].add(case.split)
        collisions[name] = sorted(
            key for key, splits in seen.items() if len(splits) > 1
        )
    total = sum(len(items) for items in collisions.values())
    return {
        "schema_version": 1,
        "cross_split_collision_count": total,
        "collisions": collisions,
        "split_case_counts": dict(
            sorted(Counter(case.split for case in cases).items())
        ),
    }


def iter_evaluator_manifest(cases: Sequence[HiddenEvaluatorCase]) -> Iterator[dict]:
    for case in sorted(cases, key=lambda item: item.public_case_id):
        yield case.evaluator_manifest_row()
