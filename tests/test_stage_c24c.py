from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from keyed_gram.stage_c24_contract import AcceptedRoute, RelationId
from keyed_gram.stage_c24b_calibration import fit_class_conditional_conformal
from keyed_gram.stage_c24b_router import RejectedRoute
from keyed_gram.stage_c24c import (
    _answer_free_contract_probe,
    _compact_metrics,
    _decisions_from_parameters,
    _selection_objective,
    _threshold_decisions,
    assert_selection_api_is_calibration_only,
    run_smoke,
)
from keyed_gram.stage_c24c_models import build_c23_public_rows, load_definitions
from keyed_gram.stage_c24c_protocol import (
    BENCHMARK_VERSION,
    C24CProtocolError,
    base_status,
    load_config,
    read_declared_split,
    sha256_file,
    verify_frozen_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage_c24c.yaml"
SMOKE_CONFIG = ROOT / "configs" / "stage_c24c_smoke.yaml"


def _row(
    row_id: str,
    sample_type: str,
    relation: RelationId | None,
    *,
    family: str,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "sample_type": sample_type,
        "relation_id": relation.value if relation is not None else None,
        "phrase_family": family,
        "frame_id": "frame-0",
        "entity": "public entity",
        "hard_negative": False,
        "minimal_pair_id": "",
    }


def _scores(**values: float) -> dict[RelationId, float]:
    return {
        RelationId.REGISTRY_ID: float(values["registry"]),
        RelationId.CITY_CODE: float(values["city"]),
        RelationId.ACCESS_CODE: float(values["access"]),
    }


def test_config_declares_exploratory_non_independent_only() -> None:
    values = load_config(CONFIG)
    protocol = values["protocol"]
    assert protocol["evaluation_mode"] == "exploratory_non_independent"
    assert protocol["exploratory_calibration_allowed"] is True
    assert protocol["exploratory_locked_scoring_allowed"] is True
    assert protocol["formal_calibration_allowed"] is False
    assert protocol["independent_external_validation"] is False


def test_config_keeps_private_confirmation_and_c3_paths_closed() -> None:
    protocol = load_config(CONFIG)["protocol"]
    assert protocol["create_confirmation"] is False
    assert protocol["train_private_memory"] is False
    assert protocol["load_private_answers"] is False
    assert protocol["execute_answer_injection"] is False
    assert protocol["execute_key_attack"] is False


def test_r0_reconstruction_is_not_mislabeled_byte_identical() -> None:
    r0 = load_config(CONFIG)["models"]["R0"]
    assert r0["baseline_fidelity"].endswith("not_byte_identical")
    assert r0["reconstructed_head_sha256"] != r0["historical_frozen_head_sha256"]
    assert r0["checkpoint_availability"].startswith("absent_from_repository")
    assert r0["historical_public_validation_accuracy"] == pytest.approx(
        r0["reconstructed_public_validation_expected_accuracy"]
    )


def test_v21_frozen_bytes_and_ai_review_validate() -> None:
    result = verify_frozen_benchmark(CONFIG)
    assert result["benchmark_version"] == BENCHMARK_VERSION
    assert result["public_benchmark_human_reviewed"] is False
    assert result["public_benchmark_ai_reviewed"] is True
    assert result["independent_external_validation"] is False


@pytest.mark.parametrize(
    ("split", "expected_rows", "expected_sha"),
    (
        (
            "public_train_v2_1",
            72,
            "9165b6d69f88207638022796fe9aa0a39ab9207ab19b31011a5be49ddfd1b5f4",
        ),
        (
            "public_calibration_v2_1",
            160,
            "88bc6475de9ad83c1ab8ae9c3383a2d9e706717c3fd988355e512f18395effaf",
        ),
        (
            "public_locked_audit_v2_1",
            160,
            "42e2ca09238d0808480cadaec04b8996f547e1e0f8f8597ed7cf1a8db86936fe",
        ),
    ),
)
def test_v21_split_bytes_are_frozen(
    split: str, expected_rows: int, expected_sha: str
) -> None:
    declaration = load_config(CONFIG)["frozen_benchmark"]["data"][split]
    path = ROOT / declaration["path"]
    assert declaration["rows"] == expected_rows
    assert sha256_file(path) == expected_sha


def test_v21_rows_remain_answer_free() -> None:
    rows = read_declared_split(CONFIG, "public_train_v2_1")
    assert len(rows) == 72
    assert all(row["answer_free"] is True for row in rows)
    assert all(row["contains_private_answer"] is False for row in rows)
    assert all("answer" not in row for row in rows)


def test_status_is_tristate_not_evaluated_before_scoring() -> None:
    status = base_status()
    assert status["closed_set_selective_router_status"] == "not_evaluated"
    assert status["open_set_abstention_status"] == "not_evaluated"
    assert status["closed_set_selective_router_ready"] is False
    assert status["open_set_abstention_ready"] is False


def test_status_never_unlocks_confirmation_or_c3() -> None:
    status = base_status()
    assert status["ready_to_create_new_confirmation_pool"] is False
    assert status["new_confirmation_pool_created_after_freeze"] is False
    assert status["c3_eligible"] is False


def test_selection_api_has_no_development_or_locked_argument() -> None:
    assert_selection_api_is_calibration_only()
    from keyed_gram.stage_c24c import _select_calibration

    names = inspect.signature(_select_calibration).parameters
    assert not any(
        token in name.casefold()
        for name in names
        for token in ("development", "locked", "audit")
    )


def test_threshold_empty_set_returns_unknown() -> None:
    routes, sets = _threshold_decisions(
        [_scores(registry=0.1, city=0.2, access=0.3)],
        {relation: 0.8 for relation in RelationId},
        minimum_margin=0.0,
    )
    assert sets == [frozenset()]
    assert isinstance(routes[0], RejectedRoute)
    assert routes[0].reason == "unknown"


def test_threshold_singleton_returns_accept() -> None:
    routes, sets = _threshold_decisions(
        [_scores(registry=0.9, city=0.2, access=0.3)],
        {relation: 0.8 for relation in RelationId},
        minimum_margin=0.0,
    )
    assert sets == [frozenset({RelationId.REGISTRY_ID})]
    assert routes == [AcceptedRoute(RelationId.REGISTRY_ID)]


def test_threshold_multiset_returns_ambiguous() -> None:
    routes, sets = _threshold_decisions(
        [_scores(registry=0.9, city=0.2, access=0.85)],
        {relation: 0.8 for relation in RelationId},
        minimum_margin=0.0,
    )
    assert sets == [
        frozenset({RelationId.REGISTRY_ID, RelationId.ACCESS_CODE})
    ]
    assert isinstance(routes[0], RejectedRoute)
    assert routes[0].reason == "ambiguous"


def test_nonfinite_evidence_fails_closed() -> None:
    routes, sets = _threshold_decisions(
        [_scores(registry=float("nan"), city=0.2, access=0.3)],
        {relation: 0.8 for relation in RelationId},
        minimum_margin=0.0,
    )
    assert sets == [frozenset()]
    assert isinstance(routes[0], RejectedRoute)
    assert routes[0].reason == "invalid"


def test_contract_probe_reject_does_not_access_memory() -> None:
    rows = [_row("u0", "unrelated", None, family="unrelated")]
    metrics, records = _answer_free_contract_probe(
        rows, [RejectedRoute("unknown")]
    )
    assert records[0]["memory_access_count"] == 0
    assert metrics["cross_relation_candidate_count"] == 0


def test_contract_probe_accept_accesses_exactly_one_bucket() -> None:
    rows = [_row("k0", "known", RelationId.CITY_CODE, family="city")]
    metrics, records = _answer_free_contract_probe(
        rows, [AcceptedRoute(RelationId.CITY_CODE)]
    )
    assert records[0]["memory_access_count"] == 1
    assert records[0]["cross_relation_candidate_count"] == 0
    assert metrics["accepted_fact_top1"] == 1.0
    assert metrics["scope"].startswith("deterministic_public_answer_free")


def test_compact_metrics_penalize_reject_all() -> None:
    rows = [
        _row("k0", "known", RelationId.REGISTRY_ID, family="registry"),
        _row("a0", "ambiguous", None, family="ambiguous"),
        _row("u0", "unrelated", None, family="unrelated"),
    ]
    reject_all = [RejectedRoute("unknown")] * 3
    reject_sets = [frozenset()] * 3
    accept_known = [
        AcceptedRoute(RelationId.REGISTRY_ID),
        RejectedRoute("ambiguous"),
        RejectedRoute("unknown"),
    ]
    accept_sets = [
        frozenset({RelationId.REGISTRY_ID}),
        frozenset({RelationId.REGISTRY_ID, RelationId.ACCESS_CODE}),
        frozenset(),
    ]
    reject_metrics = _compact_metrics(rows, reject_all, reject_sets)
    useful_metrics = _compact_metrics(rows, accept_known, accept_sets)
    assert _selection_objective(useful_metrics) > _selection_objective(
        reject_metrics
    )


def test_conformal_candidate_set_is_zero_one_or_many() -> None:
    calibration = [
        _scores(registry=0.9, city=0.1, access=0.1),
        _scores(registry=0.1, city=0.9, access=0.1),
        _scores(registry=0.1, city=0.1, access=0.9),
    ]
    calibrator = fit_class_conditional_conformal(
        calibration,
        [
            RelationId.REGISTRY_ID,
            RelationId.CITY_CODE,
            RelationId.ACCESS_CODE,
        ],
        alpha=0.1,
    )
    assert len(calibrator.candidate_set(_scores(registry=0.0, city=0.0, access=0.0))) == 0
    assert len(calibrator.candidate_set(_scores(registry=1.0, city=0.0, access=0.0))) == 1
    assert len(calibrator.candidate_set(_scores(registry=1.0, city=1.0, access=0.0))) == 2


def test_frozen_parameter_routing_keeps_r3_and_r4_set_valued() -> None:
    base_scores = [_scores(registry=0.9, city=0.1, access=0.95)]
    calibrator = fit_class_conditional_conformal(
        [
            _scores(registry=0.9, city=0.1, access=0.1),
            _scores(registry=0.1, city=0.9, access=0.1),
            _scores(registry=0.1, city=0.1, access=0.9),
        ],
        [
            RelationId.REGISTRY_ID,
            RelationId.CITY_CODE,
            RelationId.ACCESS_CODE,
        ],
        alpha=0.1,
    )
    scores = {
        "R0": base_scores,
        "R1:mean": base_scores,
        "R2:minilm:0.01": base_scores,
    }
    parameters = {
        "selected_r1_score_key": "R1:mean",
        "selected_r2_score_key": "R2:minilm:0.01",
        "selected_r3": {
            "source_score_key": "R1:mean",
            "thresholds": {
                relation.value: 0.8 for relation in RelationId
            },
            "minimum_margin": 0.0,
        },
        "selected_r4": {
            "source_score_key": "R2:minilm:0.01",
            "calibrator": calibrator.to_dict(),
        },
    }
    decisions = _decisions_from_parameters(scores, parameters)
    assert isinstance(decisions["R3"][1][0], RejectedRoute)
    assert decisions["R3"][1][0].reason == "ambiguous"
    assert isinstance(decisions["R4"][1][0], RejectedRoute)


def test_development_is_historical_c23_public_not_v2_locked() -> None:
    rows = build_c23_public_rows(CONFIG)
    assert len(rows["validation"]) == 216
    assert len(rows["reject"]) == 144
    assert all(
        row["benchmark_version"] == "c23-public-lexical-v1"
        for row in [*rows["validation"], *rows["reject"]]
    )


def test_v21_definitions_cover_exact_relation_enum() -> None:
    definitions = load_definitions(CONFIG)
    assert set(definitions) == set(RelationId)
    assert all(len(values) == 4 for values in definitions.values())


def test_smoke_never_touches_v21_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("smoke 不得读取 v2.1 split")

    monkeypatch.setattr(
        "keyed_gram.stage_c24c.read_declared_split", forbidden
    )
    result = run_smoke(SMOKE_CONFIG, output_dir=tmp_path / "smoke")
    assert result["touches_frozen_v21_locked_audit"] is False
    status = json.loads(
        (tmp_path / "smoke" / "protocol_status.json").read_text(encoding="utf-8")
    )
    assert status["formal_calibration_allowed"] is False
    assert status["c3_eligible"] is False


def test_formal_config_rejects_private_answer_path(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    altered = text + "\nprivate_answer_path: forbidden.json\n"
    path = tmp_path / "stage_c24c.yaml"
    path.write_text(altered, encoding="utf-8")
    with pytest.raises((C24CProtocolError, RuntimeError)):
        load_config(path)
