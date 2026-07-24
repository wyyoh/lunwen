from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from keyed_gram.cli import build_parser
from keyed_gram.stage_c25 import (
    _select_threshold,
    assert_calibration_api_has_no_test_argument,
    calibrate_external,
    run_smoke,
)
from keyed_gram.stage_c25_data import IntentExample, build_partitions
from keyed_gram.stage_c25_metrics import (
    aggregate_seed_metrics,
    binary_aupr,
    binary_auroc,
    evaluate_external,
)
from keyed_gram.stage_c25_protocol import (
    C25ProtocolError,
    initial_status,
    load_config,
    mark_phase_started,
    read_phase_state,
    sha256_file,
    write_json,
)
from keyed_gram.stage_c25_router import (
    IntentDecision,
    assert_external_memory_boundary,
    fit_multiclass_ridge,
    fit_pairwise_evidence,
    forced_argmax,
    max_threshold_route,
    pairwise_features,
    set_valued_route,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage_c25.yaml"
SMOKE_CONFIG = ROOT / "configs" / "stage_c25_smoke.yaml"


def _rows() -> tuple[IntentExample, ...]:
    return (
        IntentExample("k0", "known alpha", "alpha", "known", ""),
        IntentExample("k1", "known beta", "beta", "known", ""),
        IntentExample("u0", "unknown", None, "unknown", "official_oos"),
    )


def test_config_preregisters_external_test_once_protocol() -> None:
    values = load_config(CONFIG)
    protocol = values["protocol"]
    assert protocol["evaluation_mode"] == "external_public_benchmark_test_once"
    assert protocol["test_predictions_once"] is True
    assert protocol["synthetic_oos_allowed"] is False
    assert protocol["label_edits_allowed"] is False
    assert protocol["test_used_for_selection"] is False
    assert len(protocol["seeds"]) == 10
    assert len(set(protocol["seeds"])) == 10


def test_config_defers_massive_and_r4() -> None:
    values = load_config(CONFIG)
    assert values["protocol"]["massive_deferred"] is True
    assert values["selection"]["r4_deferred"] is True
    assert values["selection"]["variants"] == [
        "R0",
        "R2",
        "MAX_THRESHOLD",
        "R3",
    ]


def test_config_keeps_private_confirmation_and_c3_closed() -> None:
    protocol = load_config(CONFIG)["protocol"]
    assert protocol["create_confirmation"] is False
    assert protocol["train_private_memory"] is False
    assert protocol["load_private_answers"] is False
    assert protocol["execute_answer_injection"] is False
    assert protocol["execute_key_attack"] is False
    assert protocol["task_specific_router_ready"] is False
    assert protocol["c3_eligible"] is False


def test_official_sources_are_pinned_by_revision_and_sha256() -> None:
    sources = load_config(CONFIG)["sources"]
    assert len(sources["clinc150"]["revision"]) == 40
    assert len(sources["banking77"]["revision"]) == 40
    for dataset in ("clinc150", "banking77"):
        for declaration in sources[dataset]["files"].values():
            assert declaration["url"].startswith("https://")
            assert len(declaration["sha256"]) == 64


def test_task_specific_c24_contract_bytes_remain_frozen() -> None:
    declarations = load_config(CONFIG)["frozen_task_specific_contract"]
    for declaration in declarations.values():
        assert sha256_file(ROOT / declaration["path"]) == declaration["sha256"]


def test_initial_status_distinguishes_not_evaluated_from_failure() -> None:
    status = initial_status()
    assert status["external_oos_validation_status"] == "not_evaluated"
    assert status["external_pairwise_evidence_validated"] is False
    assert status["external_selective_abstention_validated"] is False
    assert status["task_specific_router_ready"] is False
    assert status["c3_eligible"] is False


def test_phase_started_is_irreversible(tmp_path: Path) -> None:
    mark_phase_started(tmp_path, "test_audit")
    assert (
        read_phase_state(tmp_path)["phases"]["test_audit"]["status"]
        == "started"
    )
    with pytest.raises(C25ProtocolError, match="拒绝重跑"):
        mark_phase_started(tmp_path, "test_audit")


def test_strict_json_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(C25ProtocolError, match="NaN/Infinity"):
        write_json(tmp_path / "bad.json", {"metric": float("nan")})


def test_intent_decision_schema_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        IntentDecision("accept", "alpha", "unknown", ("alpha",))
    with pytest.raises(ValueError):
        IntentDecision("reject", None, "other", ())


def test_forced_argmax_accepts_exactly_one_intent() -> None:
    decisions = forced_argmax(
        torch.tensor([[0.1, 0.9, 0.2]]), ("alpha", "beta", "gamma")
    )
    assert decisions == [IntentDecision("accept", "beta", None, ("beta",))]


def test_nonfinite_scores_fail_closed() -> None:
    scores = torch.tensor([[float("nan"), 0.9, 0.2]])
    assert forced_argmax(scores, ("alpha", "beta", "gamma"))[0].reason == "invalid"
    assert (
        max_threshold_route(
            scores, ("alpha", "beta", "gamma"), threshold=0.5
        )[0].reason
        == "invalid"
    )
    assert (
        set_valued_route(
            scores,
            ("alpha", "beta", "gamma"),
            threshold=0.5,
            minimum_margin=0.1,
        )[0].reason
        == "invalid"
    )


def test_max_threshold_only_uses_top_score() -> None:
    decisions = max_threshold_route(
        torch.tensor([[0.7, 0.69, 0.1], [0.3, 0.2, 0.1]]),
        ("alpha", "beta", "gamma"),
        threshold=0.5,
    )
    assert decisions[0].predicted_intent == "alpha"
    assert decisions[1].reason == "unknown"


def test_r3_empty_set_is_unknown() -> None:
    decision = set_valued_route(
        torch.tensor([[0.3, 0.2, 0.1]]),
        ("alpha", "beta", "gamma"),
        threshold=0.5,
        minimum_margin=0.0,
    )[0]
    assert decision.reason == "unknown"
    assert decision.candidate_set == ()


def test_r3_singleton_is_accepted() -> None:
    decision = set_valued_route(
        torch.tensor([[0.8, 0.2, 0.1]]),
        ("alpha", "beta", "gamma"),
        threshold=0.5,
        minimum_margin=0.1,
    )[0]
    assert decision == IntentDecision("accept", "alpha", None, ("alpha",))


def test_r3_multiple_candidates_are_ambiguous() -> None:
    decision = set_valued_route(
        torch.tensor([[0.8, 0.7, 0.1]]),
        ("alpha", "beta", "gamma"),
        threshold=0.5,
        minimum_margin=0.0,
    )[0]
    assert decision.reason == "ambiguous"
    assert decision.candidate_set == ("alpha", "beta")


def test_r3_competitive_margin_can_reject_singleton() -> None:
    decision = set_valued_route(
        torch.tensor([[0.8, 0.75, 0.1]]),
        ("alpha", "beta", "gamma"),
        threshold=0.78,
        minimum_margin=0.1,
    )[0]
    assert decision.reason == "ambiguous"
    assert decision.candidate_set == ("alpha",)


def test_r0_primal_ridge_is_deterministic() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = ["alpha", "alpha", "beta", "beta"]
    first = fit_multiclass_ridge(
        embeddings, labels, ridge_strength=0.01
    )
    second = fit_multiclass_ridge(
        embeddings, labels, ridge_strength=0.01
    )
    assert first.state_sha256() == second.state_sha256()
    assert [
        decision.predicted_intent
        for decision in forced_argmax(first.score(embeddings), first.classes)
    ] == labels


def test_pairwise_features_are_independent_per_intent() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    centroids = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = centroids[:, None, :].repeat(1, 2, 1)
    labels = centroids.clone()
    features = pairwise_features(
        queries, centroids, prototypes, labels, top_k=1
    )
    assert features.shape == (2, 2, 6)
    assert features[0, 0, 0] > features[0, 1, 0]
    assert features[1, 1, 0] > features[1, 0, 0]


def test_r2_pairwise_model_is_deterministic_and_scores_all_intents() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.1],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.1],
            [-1.0, 0.0, 0.1],
            [-0.9, 0.1, 0.1],
        ]
    )
    labels = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
    label_embeddings = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    arguments = dict(
        prototype_count=2,
        top_k=1,
        negative_pairs_per_positive=1,
        ridge_strength=0.01,
    )
    first = fit_pairwise_evidence(
        embeddings,
        labels,
        ("alpha", "beta", "gamma"),
        label_embeddings,
        **arguments,
    )
    second = fit_pairwise_evidence(
        embeddings,
        labels,
        ("alpha", "beta", "gamma"),
        label_embeddings,
        **arguments,
    )
    assert first.score(embeddings).shape == (6, 3)
    assert first.state_sha256() == second.state_sha256()


def test_access_control_metrics_are_correct() -> None:
    rows = _rows()
    scores = torch.tensor(
        [[0.9, 0.1], [0.8, 0.7], [0.6, 0.1]]
    )
    decisions = [
        IntentDecision("accept", "alpha", None, ("alpha",)),
        IntentDecision("accept", "alpha", None, ("alpha",)),
        IntentDecision("accept", "alpha", None, ("alpha",)),
    ]
    metrics, _ = evaluate_external(
        rows, scores, ("alpha", "beta"), decisions
    )
    assert metrics["safe_coverage"] == pytest.approx(0.5)
    assert metrics["wrong_bucket_access_rate"] == pytest.approx(0.5)
    assert metrics["false_memory_access_rate"] == pytest.approx(1.0)
    assert metrics["accepted_route_accuracy"] == pytest.approx(0.5)


def test_oos_auroc_and_aupr_rank_unknown_higher() -> None:
    scores = [0.1, 0.2, 0.9, 0.8]
    targets = [False, False, True, True]
    assert binary_auroc(scores, targets) == pytest.approx(1.0)
    assert binary_aupr(scores, targets) == pytest.approx(1.0)


def test_seed_aggregation_reports_finite_confidence_intervals() -> None:
    rows = [
        {"benchmark": "demo", "variant": "R3", "score": 0.5},
        {"benchmark": "demo", "variant": "R3", "score": 0.7},
    ]
    result = aggregate_seed_metrics(rows, ["score"])[0]
    assert result["mean"] == pytest.approx(0.6)
    assert result["ci95_low"] < result["ci95_high"]


def test_selection_penalizes_reject_all() -> None:
    values = load_config(SMOKE_CONFIG, smoke=True)
    rows = _rows()
    scores = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.2, 0.1]], dtype=torch.float32
    )
    selected, _ = _select_threshold(
        values, rows, scores, ("alpha", "beta"), variant="R3"
    )
    assert selected["calibration_metrics"]["known_coverage"] == 1.0
    assert selected["calibration_metrics"]["false_memory_access_rate"] == 0.0


def test_calibration_api_has_no_test_argument() -> None:
    assert_calibration_api_has_no_test_argument()
    assert "test" not in inspect.signature(calibrate_external).parameters


def test_partition_generation_is_whole_intent_and_seed_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import keyed_gram.stage_c25_data as data_module

    domains = {
        f"domain-{domain}": [
            f"d{domain}-intent-{index}" for index in range(15)
        ]
        for domain in range(10)
    }
    categories = [f"bank-intent-{index}" for index in range(77)]
    monkeypatch.setattr(data_module, "_validate_official_sources", lambda _: None)
    monkeypatch.setattr(data_module, "_load_clinc", lambda _: ({}, domains))
    monkeypatch.setattr(
        data_module, "_load_banking", lambda _: ({}, categories)
    )
    manifest = build_partitions(CONFIG)
    assert manifest["partition_unit"] == "whole_intent"
    assert len(manifest["partitions"]) == 10
    assert len(
        {row["seed"] for row in manifest["partitions"]}
    ) == 10
    for row in manifest["partitions"]:
        clinc = row["clinc150"]
        banking = row["banking77_open"]
        assert len(clinc["supported_intents"]) == 50
        assert len(clinc["unsupported_intents"]) == 100
        assert len(banking["supported_intents"]) == 39
        assert len(banking["calibration_oos_intents"]) == 19
        assert len(banking["test_only_oos_intents"]) == 19
        assert not (
            set(banking["supported_intents"])
            & set(banking["test_only_oos_intents"])
        )


def test_smoke_never_reads_official_test(tmp_path: Path) -> None:
    result = run_smoke(SMOKE_CONFIG, output_dir=tmp_path / "smoke")
    assert result["official_source_read"] is False
    assert result["official_test_read"] is False
    assert result["metrics"]["safe_coverage"] == 1.0
    assert result["metrics"]["false_memory_access_rate"] == 0.0
    assert result["c3_eligible"] is False


def test_cli_exposes_c25_protocol_commands() -> None:
    parser = build_parser()
    for command in (
        "stage-c25-prepare",
        "stage-c25-calibrate",
        "stage-c25-audit",
        "stage-c25-smoke",
        "stage-c25-finalize",
    ):
        arguments = [command, "--config", str(CONFIG)]
        if command == "stage-c25-smoke":
            arguments = [command, "--config", str(SMOKE_CONFIG)]
        assert parser.parse_args(arguments).command == command


def test_external_memory_boundary_does_not_extend_private_contract() -> None:
    assert_external_memory_boundary()
    annotations = IntentDecision.__annotations__
    assert "confidence" not in annotations
    assert "semantic_embedding" not in annotations
    assert "private_value" not in annotations

