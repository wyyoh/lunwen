from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pytest
import yaml

import keyed_gram.stage_c24b_v21 as v21
from keyed_gram.cli import build_parser
from keyed_gram.stage_c24b import (
    ProtocolViolation,
    audit_stage_c24b,
    calibrate_stage_c24b,
    prepare_stage_c24b as prepare_legacy_stage_c24b,
    validate_stage_c24b_reviews,
)
from keyed_gram.stage_c24b_v21 import (
    AI_REVIEW_FIELDS,
    V21ProtocolError,
    apply_stage_c24b_v21_ai_review,
    build_stage_c24b_v21,
    load_stage_c24b_v21_config,
    prepare_stage_c24b_v21,
    validate_stage_c24b_v21_ai_review,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage_c24b_v21.yaml"
AI_AUDIT_PATH = ROOT / "configs" / "stage_c24b_v21_ai_audit.yaml"
AI_PROMPT_PATH = ROOT / "configs" / "stage_c24b_v21_ai_review_prompt.md"

V21_SPLITS = (
    "public_train_v2_1",
    "public_calibration_v2_1",
    "public_locked_audit_v2_1",
)
OLD_V2_SPLITS = (
    "public_train_v2",
    "public_calibration_v2",
    "public_locked_audit_v2",
)
EXPECTED_PROMPT_SHA256 = (
    "39e796137f8adddcc2878514cb0069fadbd69cdf1c10e12e6966bc11678ecacc"
)
AI_DECISION_FIELDS = {
    "ai_suggested_label",
    "ai_suggested_type",
    "ai_confidence",
    "construct_validity",
    "shortcut_flags",
    "notes",
    "status",
}
FORBIDDEN_AI_SCHEMA_FRAGMENTS = {
    "reviewer_1",
    "reviewer_2",
    "adjudicated",
    "router",
    "prediction",
    "evidence_score",
    "logit",
    "threshold",
    "retrieval",
}
HARD_UNRELATED_CARRIERS = {
    "access",
    "archive",
    "authorization",
    "code",
    "credential",
    "enrollment",
    "filing",
    "geographic",
    "identifier",
    "key",
    "location",
    "locator",
    "login",
    "map",
    "reference",
    "registry",
    "serial",
    "token",
    "zone",
}


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_specs(config: dict, split: str):
    split_config = config["benchmark"]["splits"][split]
    for relation, families in split_config["known_families"].items():
        for family_id, raw_spec in families.items():
            spec = {"phrase": raw_spec} if isinstance(raw_spec, str) else raw_spec
            yield family_id, str(spec["phrase"]), relation, "known", spec
    for sample_type, families in split_config.get("rejection_families", {}).items():
        for family_id, raw_spec in families.items():
            spec = {"phrase": raw_spec} if isinstance(raw_spec, str) else raw_spec
            yield family_id, str(spec["phrase"]), sample_type, sample_type, spec


def _all_families(config: dict) -> dict[str, tuple[str, str, str, dict]]:
    output: dict[str, tuple[str, str, str, dict]] = {}
    for split in V21_SPLITS:
        for family_id, phrase, label, sample_type, spec in _family_specs(config, split):
            assert family_id not in output
            output[family_id] = (phrase, label, sample_type, spec)
    return output


def _normal_tokens(text: object) -> tuple[str, ...]:
    # YAML 1.1 会把未加引号的 ``no`` 解为 False；协议语义仍是单词 no。
    surface = "no" if text is False else str(text)
    return tuple(re.findall(r"[a-z0-9]+", surface.casefold().replace("-", " ")))


def _contains_token_sequence(text: str, token: str) -> bool:
    phrase_tokens = _normal_tokens(text)
    needle = _normal_tokens(token)
    return any(
        phrase_tokens[index : index + len(needle)] == needle
        for index in range(len(phrase_tokens) - len(needle) + 1)
    )


def _copy_protocol_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """复制 prepare/apply 所需的冻结输入，绝不写生产 artifacts。"""

    destination = tmp_path / "isolated-repo"
    relatives = (
        "configs/stage_c24b_v21.yaml",
        "configs/stage_c24b_v21_ai_audit.yaml",
        "configs/stage_c24b_v21_ai_review_prompt.md",
        "configs/stage_c24b.yaml",
        "configs/stage_c23.yaml",
        "configs/stage_c24.yaml",
        "artifacts/stage_c24b/resolved_config.json",
        "artifacts/stage_c24b/public_benchmark_manifest.json",
        "artifacts/stage_c24b/public_train_v2_review.csv",
        "artifacts/stage_c24b/public_calibration_v2_review.csv",
        "artifacts/stage_c24b/public_locked_audit_v2_review.csv",
        "artifacts/stage_c24b/c24_locked_audit_independent_review.csv",
    )
    for relative in relatives:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return (
        destination,
        destination / "configs" / "stage_c24b_v21.yaml",
        destination / "configs" / "stage_c24b_v21_ai_audit.yaml",
        destination / "configs" / "stage_c24b_v21_ai_review_prompt.md",
    )


def test_v21_config_uses_new_namespace_and_materializes_72_160_160_rows():
    config = _yaml(CONFIG_PATH)
    assert set(config["benchmark"]["splits"]) == set(V21_SPLITS)
    assert not set(OLD_V2_SPLITS) & set(config["benchmark"]["splits"])

    family_counts = {
        split: sum(1 for _ in _family_specs(config, split)) for split in V21_SPLITS
    }
    frame_counts = {
        split: len(config["benchmark"]["splits"][split]["frames"])
        for split in V21_SPLITS
    }
    assert family_counts == {
        "public_train_v2_1": 24,
        "public_calibration_v2_1": 40,
        "public_locked_audit_v2_1": 40,
    }
    assert {
        split: family_counts[split] * frame_counts[split] for split in V21_SPLITS
    } == {
        "public_train_v2_1": 72,
        "public_calibration_v2_1": 160,
        "public_locked_audit_v2_1": 160,
    }
    families = _all_families(config)
    assert len(families) == 104
    assert all(family_id.startswith("v21_") for family_id in families)


def test_v21_removes_all_ambiguous_shortcut_markers_and_hardens_all_unrelated():
    config = _yaml(CONFIG_PATH)
    shortcuts = config["legacy_v2_ai_audit"]["ambiguous_shortcut_tokens"]
    ambiguous = []
    unrelated = []
    for split in V21_SPLITS[1:]:
        for family_id, phrase, _, sample_type, spec in _family_specs(config, split):
            if sample_type == "ambiguous":
                ambiguous.append((family_id, phrase, spec))
            elif sample_type == "unrelated":
                unrelated.append((family_id, phrase, spec))

    assert len(ambiguous) == 16
    assert {
        family_id: [token for token in shortcuts if _contains_token_sequence(phrase, token)]
        for family_id, phrase, _ in ambiguous
    } == {family_id: [] for family_id, _, _ in ambiguous}

    assert len(unrelated) == 16
    assert all(spec.get("hard_negative") is True for _, _, spec in unrelated)
    carrier_hits = {
        family_id: sorted(set(_normal_tokens(phrase)) & HARD_UNRELATED_CARRIERS)
        for family_id, phrase, _ in unrelated
    }
    assert all(carrier_hits.values()), carrier_hits


def test_legacy_v2_disposition_is_sealed_diagnostic_only_and_does_not_rewrite_sources():
    config = _yaml(CONFIG_PATH)
    base = config["base_protocol"]
    legacy = config["legacy_v2_ai_audit"]
    assert base["disposition"] == "superseded_before_model_scoring_diagnostic_only"
    assert base["changes_superseded_files"] is False
    assert _sha256(ROOT / base["source_manifest"]) == base["source_manifest_sha256"]
    assert {
        "total_rows": legacy["total_rows"],
        "total_unique_families": legacy["total_unique_families"],
        "consistent_families": legacy["consistent_families"],
        "relabel_families": legacy["relabel_families"],
        "affected_rows": legacy["affected_rows"],
    } == {
        "total_rows": 512,
        "total_unique_families": 134,
        "consistent_families": 130,
        "relabel_families": 4,
        "affected_rows": 16,
    }
    assert len(legacy["exceptions"]) == 4
    for source in legacy["source_review_files"].values():
        path = ROOT / source["path"]
        before = path.read_bytes()
        assert _sha256(path) == source["sha256"]
        assert sum(1 for _ in csv.DictReader(before.decode("utf-8").splitlines())) == source[
            "rows"
        ]
        assert path.read_bytes() == before


def test_ai_audit_covers_exactly_104_families_without_human_or_router_fields():
    config = _yaml(CONFIG_PATH)
    audit = _yaml(AI_AUDIT_PATH)
    expected = _all_families(config)
    decisions = {
        family_id: decision
        for split, split_decisions in audit["decisions"].items()
        for family_id, decision in split_decisions.items()
    }
    assert set(audit["decisions"]) == set(V21_SPLITS)
    assert set(decisions) == set(expected)
    assert len(decisions) == 104
    for family_id, decision in decisions.items():
        assert set(decision) == AI_DECISION_FIELDS
        phrase, label, sample_type, _ = expected[family_id]
        del phrase
        assert decision["ai_suggested_label"] == label
        assert decision["ai_suggested_type"] == sample_type
        assert decision["ai_confidence"] is None
        joined_keys = " ".join(decision).casefold()
        assert not any(fragment in joined_keys for fragment in FORBIDDEN_AI_SCHEMA_FRAGMENTS)
    flattened_schema = json.dumps(audit, sort_keys=True).casefold()
    assert not any(
        f'"{fragment}' in flattened_schema
        for fragment in ("reviewer_1", "reviewer_2", "adjudicated", "router")
    )


def test_ai_prompt_and_review_are_prediction_blind_and_byte_sealed():
    audit = _yaml(AI_AUDIT_PATH)
    reviewer = audit["reviewer"]
    assert _sha256(AI_PROMPT_PATH) == EXPECTED_PROMPT_SHA256
    assert reviewer["prompt_sha256"] == EXPECTED_PROMPT_SHA256
    assert reviewer["prediction_blind"] is True
    assert reviewer["independent_external"] is False
    prompt = " ".join(
        AI_PROMPT_PATH.read_text(encoding="utf-8").casefold().split()
    )
    for forbidden in (
        "router predictions",
        "evidence scores",
        "logits",
        "thresholds",
        "retrieval output",
        "private answers",
        "locked-audit result",
    ):
        assert forbidden in prompt


def test_ai_only_protocol_never_claims_human_external_formal_or_c3_readiness():
    config = _yaml(CONFIG_PATH)
    review = config["review_protocol"]
    assert review["review_mode"] == "single_ai_semantic_audit"
    assert review["public_benchmark_human_reviewed"] is False
    assert review["independent_external_validation"] is False
    assert review["formal_calibration_allowed"] is False
    assert review["formal_locked_audit_executed"] is False
    assert review["new_confirmation_pool_created_after_freeze"] is False
    assert review["c3_eligible"] is False
    assert _sha256(ROOT / "src/keyed_gram/stage_c24_contract.py") == (
        "c8a3b5fd33621349659e504b1bd31502688896cf63207ae59f644849a0d96692"
    )
    assert _sha256(ROOT / "src/keyed_gram/stage_c24b_router.py") == (
        "4c791aa1a594b7fadd64226ecdd19465aefb54500c3ee92339cf011c01a18175"
    )


def test_cli_exposes_v21_prepare_apply_and_validate_only():
    parser = build_parser()
    cases = {
        "stage-c24b-v21-prepare": ["--config", "config.yaml"],
        "stage-c24b-v21-apply-ai-review": [
            "--config",
            "config.yaml",
            "--audit-spec",
            "audit.yaml",
        ],
        "stage-c24b-v21-validate-ai-review": ["--config", "config.yaml"],
    }
    for command, arguments in cases.items():
        parsed = parser.parse_args([command, *arguments])
        assert parsed.command == command
    help_text = parser.format_help()
    assert "stage-c24b-v21-calibrate" not in help_text
    assert "stage-c24b-v21-audit" not in help_text


def test_build_api_preserves_new_namespace_and_audits_all_332_historical_families():
    values = load_stage_c24b_v21_config(CONFIG_PATH)
    rows, audit = build_stage_c24b_v21(CONFIG_PATH)

    assert values["benchmark"]["version"] == "c24b-public-selective-router-v2.1"
    assert set(rows) == set(V21_SPLITS)
    assert {split: len(split_rows) for split, split_rows in rows.items()} == {
        "public_train_v2_1": 72,
        "public_calibration_v2_1": 160,
        "public_locked_audit_v2_1": 160,
    }
    assert all(
        row["split"] == split
        and split in row["row_id"]
        and str(row["fact_id"]).startswith("public-v2.1:")
        for split, split_rows in rows.items()
        for row in split_rows
    )
    history = audit["historical_collision_audit"]
    assert set(history["historical_sources"]) == {
        "stage_c23_config",
        "stage_c24_config",
        "stage_c24b_v2_config",
    }
    assert history["historical_phrase_count"] == 332
    assert history["historical_entity_count"] == 39
    assert history["historical_frame_count"] == 30
    assert history["passed"] is True
    assert all(history["splits"][split]["passed"] is True for split in V21_SPLITS)
    split_pairs = audit["split_isolation"]["pairs"]
    assert split_pairs
    assert all("public_" in key and "_v2_1" in key and "_v2__" not in key for key in split_pairs)
    design = audit["design_quality"]["combined"]
    assert design["ambiguous_family_count"] == 16
    assert design["ambiguous_explicit_shortcut_count"] == 0
    assert design["hard_unrelated_family_count"] == 16
    assert design["hard_unrelated_shared_carrier_count"] == 16
    assert design["passed"] is True


def _prepare_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, dict]:
    root, config_path, audit_path, prompt_path = _copy_protocol_tree(tmp_path)
    monkeypatch.setattr(
        v21,
        "_git_state",
        lambda _repo: {"commit": "f" * 40, "branch": "test", "dirty": False},
    )
    summary = prepare_stage_c24b_v21(config_path)
    return root, config_path, audit_path, prompt_path, summary


def test_prepare_api_isolated_outputs_preserve_old_v2_bytes_and_false_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, config_path, _, _, summary = _prepare_isolated(tmp_path, monkeypatch)
    values = load_stage_c24b_v21_config(config_path)
    data_dir = root / values["paths"]["public_data_dir"]
    artifact_dir = root / values["paths"]["artifact_dir"]

    assert summary["row_counts"] == {
        "public_train_v2_1": 72,
        "public_calibration_v2_1": 160,
        "public_locked_audit_v2_1": 160,
    }
    assert summary["unique_family_counts"] == {
        "public_train_v2_1": 24,
        "public_calibration_v2_1": 40,
        "public_locked_audit_v2_1": 40,
    }
    for split in V21_SPLITS:
        assert (data_dir / f"{split}.jsonl").is_file()
        assert (artifact_dir / f"{split}_ai_review_task.csv").is_file()
        with (artifact_dir / f"{split}_ai_review_task.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            tasks = list(csv.DictReader(handle))
        assert len(tasks) == summary["unique_family_counts"][split]
        assert all(row["ai_review_status"] == "pending" for row in tasks)
        assert all(not row["ai_suggested_label"] for row in tasks)

    prior = json.loads(
        (artifact_dir / "prior_v2_ai_audit_summary.json").read_text(encoding="utf-8")
    )
    assert prior["unique_family_decisions_reported"] == 134
    assert prior["mapped_generated_rows"] == 512
    assert prior["relabel_families"] == 4
    assert prior["affected_rows"] == 16
    assert prior["ai_review_status"] == "failed_requires_revision"
    assert prior["formal_calibration_allowed"] is False
    for name, seal in values["legacy_v2_ai_audit"]["source_review_files"].items():
        copied = root / seal["path"]
        assert _sha256(copied) == seal["sha256"], name

    status = json.loads((artifact_dir / "protocol_status.json").read_text(encoding="utf-8"))
    for key in (
        "public_benchmark_human_reviewed",
        "public_benchmark_ai_reviewed",
        "independent_external_validation",
        "formal_calibration_allowed",
        "formal_calibration_executed",
        "formal_locked_audit_executed",
        "ready_to_create_new_confirmation_pool",
        "new_confirmation_pool_created_after_freeze",
        "c3_eligible",
        "entity_checkpoint_loaded",
        "entity_checkpoint_hash_verified",
    ):
        assert status[key] is False


def test_apply_and_validate_ai_review_cover_104_families_without_formal_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, config_path, audit_path, _, _ = _prepare_isolated(tmp_path, monkeypatch)
    applied = apply_stage_c24b_v21_ai_review(config_path, audit_path)
    validated = validate_stage_c24b_v21_ai_review(config_path)
    artifact_dir = root / "artifacts" / "stage_c24b_v21"

    assert applied["public_benchmark_ai_reviewed"] is True
    assert validated["public_benchmark_ai_reviewed"] is True
    assert (
        validated["status"]
        == "completed_with_declared_boundaries_exploratory_non_independent"
    )
    assert validated["semantic_audit_outcome"] == "passed_exploratory_non_independent"
    assert validated["reviewed_family_count"] == 104
    assert validated["boundary_retained_family_count"] == 37
    assert sum(split["family_count"] for split in validated["splits"].values()) == 104
    assert sum(split["covered_row_count"] for split in validated["splits"].values()) == 392
    assert validated["prompt_sha256"] == EXPECTED_PROMPT_SHA256
    assert validated["prediction_blind"] is True
    for key in (
        "public_benchmark_human_reviewed",
        "independent_external_validation",
        "review_independent_of_benchmark_authorship",
        "locked_router_predictions_seen",
        "locked_used_for_router_selection",
        "formal_calibration_allowed",
        "formal_calibration_executed",
        "formal_locked_audit_executed",
        "ready_to_create_new_confirmation_pool",
        "new_confirmation_pool_created_after_freeze",
        "c3_eligible",
    ):
        assert validated[key] is False

    forbidden = ("reviewer_", "adjudicated", "router", "logit", "evidence", "retrieval")
    for split in V21_SPLITS:
        with (artifact_dir / f"{split}_ai_review.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == AI_REVIEW_FIELDS
            assert not any(token in field for field in reader.fieldnames or () for token in forbidden)
            assert len(list(reader)) in {24, 40}


def test_config_cannot_enable_human_external_formal_confirmation_or_c3_flags(
    tmp_path: Path,
):
    root, config_path, _, _ = _copy_protocol_tree(tmp_path)
    del root
    original = _yaml(config_path)
    forbidden_true = (
        "public_benchmark_human_reviewed",
        "public_benchmark_ai_reviewed",
        "independent_external_validation",
        "formal_calibration_allowed",
        "formal_locked_audit_executed",
        "new_confirmation_pool_created_after_freeze",
        "c3_eligible",
    )
    for field in forbidden_true:
        changed = copy.deepcopy(original)
        changed["review_protocol"][field] = True
        config_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
        with pytest.raises(V21ProtocolError, match="non-human pending AI review"):
            load_stage_c24b_v21_config(config_path)
    config_path.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")


def test_apply_fails_closed_for_missing_family_relabel_and_invalid_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pristine, _, _, _, _ = _prepare_isolated(tmp_path / "base", monkeypatch)

    def missing_family(spec: dict) -> None:
        spec["decisions"][V21_SPLITS[0]].pop(next(iter(spec["decisions"][V21_SPLITS[0]])))

    def relabel(spec: dict) -> None:
        decision = next(iter(spec["decisions"][V21_SPLITS[0]].values()))
        decision["ai_suggested_label"] = "city_code"

    def confidence_above_one(spec: dict) -> None:
        next(iter(spec["decisions"][V21_SPLITS[0]].values()))["ai_confidence"] = 1.01

    def nonfinite_confidence(spec: dict) -> None:
        next(iter(spec["decisions"][V21_SPLITS[0]].values()))["ai_confidence"] = math.nan

    def boolean_confidence(spec: dict) -> None:
        next(iter(spec["decisions"][V21_SPLITS[0]].values()))["ai_confidence"] = True

    def router_field(spec: dict) -> None:
        next(iter(spec["decisions"][V21_SPLITS[0]].values()))[
            "router_prediction"
        ] = "registry_id"

    def confirmation_in_notes(spec: dict) -> None:
        next(iter(spec["decisions"][V21_SPLITS[0]].values()))["notes"] = {
            "confirmation_path": "artifacts/confirmation/private.json"
        }

    def list_in_notes(spec: dict) -> None:
        next(iter(spec["decisions"][V21_SPLITS[0]].values()))["notes"] = [
            "not a scalar note"
        ]

    cases = {
        "missing": (missing_family, "family coverage"),
        "relabel": (relabel, "label/type change"),
        "confidence": (confidence_above_one, "confidence is invalid"),
        "nonfinite": (nonfinite_confidence, "confidence is invalid"),
        "boolean-confidence": (boolean_confidence, "confidence is invalid"),
        "router": (router_field, "prohibited AI-review input"),
        "confirmation": (confirmation_in_notes, "prohibited AI-review input"),
        "note-list": (list_in_notes, "semantic note is blank"),
    }
    for name, (mutate, message) in cases.items():
        case_root = tmp_path / name
        shutil.copytree(pristine, case_root)
        case_config = case_root / "configs" / "stage_c24b_v21.yaml"
        case_audit = case_root / "configs" / "stage_c24b_v21_ai_audit.yaml"
        spec = _yaml(case_audit)
        mutate(spec)
        case_audit.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        with pytest.raises(V21ProtocolError, match=message):
            apply_stage_c24b_v21_ai_review(case_config, case_audit)


def test_input_and_output_tampering_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pristine, _, _, _, _ = _prepare_isolated(tmp_path / "base", monkeypatch)

    prompt_root = tmp_path / "prompt-tamper"
    shutil.copytree(pristine, prompt_root)
    prompt = prompt_root / "configs" / "stage_c24b_v21_ai_review_prompt.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8")
    with pytest.raises(V21ProtocolError, match="prompt SHA-256"):
        apply_stage_c24b_v21_ai_review(
            prompt_root / "configs" / "stage_c24b_v21.yaml",
            prompt_root / "configs" / "stage_c24b_v21_ai_audit.yaml",
        )

    data_root = tmp_path / "data-tamper"
    shutil.copytree(pristine, data_root)
    data_path = data_root / "data" / "stage_c24b_v21" / f"{V21_SPLITS[0]}.jsonl"
    data_path.write_text(data_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(V21ProtocolError, match="sealed v2.1 data changed"):
        apply_stage_c24b_v21_ai_review(
            data_root / "configs" / "stage_c24b_v21.yaml",
            data_root / "configs" / "stage_c24b_v21_ai_audit.yaml",
        )

    output_root = tmp_path / "output-tamper"
    shutil.copytree(pristine, output_root)
    output_config = output_root / "configs" / "stage_c24b_v21.yaml"
    output_audit = output_root / "configs" / "stage_c24b_v21_ai_audit.yaml"
    apply_stage_c24b_v21_ai_review(output_config, output_audit)
    output_csv = (
        output_root
        / "artifacts"
        / "stage_c24b_v21"
        / f"{V21_SPLITS[0]}_ai_review.csv"
    )
    output_csv.write_text(
        output_csv.read_text(encoding="utf-8").replace("passed", "tampered", 1),
        encoding="utf-8",
    )
    with pytest.raises(V21ProtocolError, match="SHA-256 changed"):
        validate_stage_c24b_v21_ai_review(output_config)


def test_prepare_snapshot_and_final_inventory_tampering_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pristine, _, _, _, _ = _prepare_isolated(tmp_path / "base", monkeypatch)

    prepared_root = tmp_path / "prepared-manifest-tamper"
    shutil.copytree(pristine, prepared_root)
    split_manifest = (
        prepared_root
        / "artifacts"
        / "stage_c24b_v21"
        / f"{V21_SPLITS[0]}_manifest.json"
    )
    split_manifest.write_text(
        split_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(V21ProtocolError, match="prepared artifact changed"):
        apply_stage_c24b_v21_ai_review(
            prepared_root / "configs" / "stage_c24b_v21.yaml",
            prepared_root / "configs" / "stage_c24b_v21_ai_audit.yaml",
        )

    final_root = tmp_path / "final-extra-file"
    shutil.copytree(pristine, final_root)
    final_config = final_root / "configs" / "stage_c24b_v21.yaml"
    apply_stage_c24b_v21_ai_review(
        final_config, final_root / "configs" / "stage_c24b_v21_ai_audit.yaml"
    )
    (final_root / "artifacts" / "stage_c24b_v21" / "unsealed.txt").write_text(
        "not sealed\n", encoding="utf-8"
    )
    with pytest.raises(V21ProtocolError, match="inventory is not exact"):
        validate_stage_c24b_v21_ai_review(final_config)


def test_semantic_failure_writes_no_partial_review_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, config_path, audit_path, _, _ = _prepare_isolated(tmp_path, monkeypatch)
    spec = _yaml(audit_path)
    first_locked = next(iter(spec["decisions"][V21_SPLITS[2]].values()))
    first_locked["status"] = "failed_requires_revision"
    first_locked["construct_validity"] = "failed"
    audit_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    with pytest.raises(V21ProtocolError, match="construct review failed"):
        apply_stage_c24b_v21_ai_review(config_path, audit_path)
    artifact_dir = root / "artifacts" / "stage_c24b_v21"
    assert not (artifact_dir / "ai_review_attempt_manifest.json").exists()
    assert not any(
        (artifact_dir / f"{split}_ai_review.csv").exists() for split in V21_SPLITS
    )


def test_config_phrase_tamper_after_prepare_invalidates_ai_review_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, config_path, audit_path, _, _ = _prepare_isolated(tmp_path, monkeypatch)
    config = _yaml(config_path)
    config["benchmark"]["splits"][V21_SPLITS[2]]["known_families"]["registry_id"][
        "v21_locked_registry_heritage_deposit"
    ]["phrase"] = "a different unreviewed phrase"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(V21ProtocolError, match="config|seal|SHA-256"):
        apply_stage_c24b_v21_ai_review(config_path, audit_path)


def test_implementation_source_snapshot_change_after_prepare_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, config_path, audit_path, _, _ = _prepare_isolated(tmp_path, monkeypatch)
    original = v21._implementation_source_snapshot()
    changed = copy.deepcopy(original)
    changed["aggregate_sha256"] = "0" * 64
    monkeypatch.setattr(v21, "_implementation_source_snapshot", lambda: changed)
    with pytest.raises(V21ProtocolError, match="implementation source changed"):
        apply_stage_c24b_v21_ai_review(config_path, audit_path)


def test_superseded_old_v2_formal_calibration_and_audit_are_permanently_guarded(
    tmp_path: Path,
):
    root, _, _, _ = _copy_protocol_tree(tmp_path)
    old_config = root / "configs" / "stage_c24b.yaml"
    with pytest.raises(ProtocolViolation, match="permanently disabled|superseded"):
        prepare_legacy_stage_c24b(old_config)
    for operation in (calibrate_stage_c24b, audit_stage_c24b):
        with pytest.raises(ProtocolViolation, match="permanently disabled|superseded"):
            operation(old_config, tmp_path / operation.__name__, device_name="cpu")
    with pytest.raises(ProtocolViolation, match="permanently disabled|superseded"):
        validate_stage_c24b_reviews(old_config)

    # 改一个可控 version 字符串不能复活已撤销的稳定 benchmark 指纹/命名空间。
    values = _yaml(old_config)
    values["benchmark"]["version"] = "renamed-to-evade-revocation"
    old_config.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="permanently disabled|superseded"):
        calibrate_stage_c24b(
            old_config, tmp_path / "renamed-calibration", device_name="cpu"
        )
