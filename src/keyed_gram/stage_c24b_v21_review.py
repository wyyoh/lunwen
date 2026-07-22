"""Stage C2.4b v2.1 的 prediction-blind 单模型 AI 审核状态机。"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

import yaml

from .stage_c24b_benchmark import sha256_file
from .stage_c24b_v21_seal import (
    V21ProtocolError,
    artifact_files as _artifact_files,
    atomic_write_text as _atomic_write_text,
    canonical_sha256 as _canonical_sha256,
    load_exact_artifact_manifest as _load_exact_artifact_manifest,
    load_payload_sealed_json as _load_payload_sealed_json,
    read_csv as _read_csv,
    render_csv as _render_csv,
    strict_json_loads as _strict_json_loads,
    verify_prepare_snapshot as _verify_prepare_snapshot_seal,
    write_json as _write_json,
)


_CORE_BINDINGS = (
    "SCHEMA_VERSION",
    "STAGE_NAME",
    "REVIEW_MODE",
    "PRESERVED_TYPED_MEMORY_CONTRACT_SHA256",
    "PRESERVED_SELECTIVE_ROUTER_SHA256",
    "NEW_SPLITS",
    "AI_REVIEW_FIELDS",
    "AI_DECISION_FIELDS",
    "AI_STATUSES",
    "AI_CONSTRUCT_VALIDITY",
    "_AUDIT_SPEC_FIELDS",
    "_AI_REVIEWER_FIELDS",
    "SHORTCUT_TOKENS",
    "_reject_prohibited_ai_input",
    "_resolve",
    "_git_state",
    "_verify_implementation_source_snapshot",
    "_tokens",
    "_family_tasks",
    "_family_task_payload_sha256",
    "load_stage_c24b_v21_config",
)


def _bind_core() -> None:
    """延迟绑定构建核心，避免 review 模块与兼容重导出之间形成导入环。"""

    core = import_module(f"{__package__}.stage_c24b_v21")
    namespace = globals()
    for name in _CORE_BINDINGS:
        namespace[name] = getattr(core, name)


def _load_audit_spec(path: Path) -> dict[str, Any]:
    _bind_core()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or set(raw) != _AUDIT_SPEC_FIELDS:
        raise V21ProtocolError("AI audit spec top-level schema differs")
    spec = copy.deepcopy(dict(raw))
    _reject_prohibited_ai_input(spec)
    if spec["schema_version"] != SCHEMA_VERSION or spec["review_mode"] != REVIEW_MODE:
        raise V21ProtocolError("AI audit spec schema/review mode differs")
    if not isinstance(spec["audit_id"], str) or not spec["audit_id"].strip():
        raise V21ProtocolError("AI audit ID must be a non-empty string")
    reviewer = spec["reviewer"]
    if not isinstance(reviewer, Mapping) or set(reviewer) != _AI_REVIEWER_FIELDS:
        raise V21ProtocolError("AI audit reviewer schema differs")
    if reviewer["prediction_blind"] is not True or reviewer[
        "independent_external"
    ] is not False:
        raise V21ProtocolError("AI audit must be prediction-blind and non-external")
    if (
        not isinstance(reviewer["model_id"], str)
        or not reviewer["model_id"].strip()
        or not isinstance(reviewer["model_revision"], str)
        or not reviewer["model_revision"].strip()
        or not isinstance(reviewer["prompt_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", reviewer["prompt_sha256"])
    ):
        raise V21ProtocolError("AI audit model provenance must not be silently blank")
    prompt_path = path.with_name("stage_c24b_v21_ai_review_prompt.md")
    if not prompt_path.is_file() or sha256_file(prompt_path) != str(
        reviewer["prompt_sha256"]
    ):
        raise V21ProtocolError("AI audit prompt SHA-256 mismatch")
    decisions = spec["decisions"]
    if not isinstance(decisions, Mapping) or set(decisions) != set(NEW_SPLITS):
        raise V21ProtocolError("AI audit decisions must cover exactly three v2.1 splits")
    spec["prompt_path"] = str(prompt_path)
    return spec


def _verify_prepare_snapshot(
    artifact_dir: Path, config_path: Path, *, before_review: bool
) -> dict[str, Any]:
    _bind_core()
    return _verify_prepare_snapshot_seal(
        artifact_dir,
        config_path,
        before_review=before_review,
        review_splits=NEW_SPLITS,
    )


def _verify_prepared_data(
    config_path: Path, values: Mapping[str, Any], *, before_review: bool = False
) -> tuple[Path, Path, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    _bind_core()
    data_dir = _resolve(config_path, values["paths"]["public_data_dir"])
    artifact_dir = _resolve(config_path, values["paths"]["artifact_dir"])
    _verify_prepare_snapshot(artifact_dir, config_path, before_review=before_review)
    benchmark_manifest_path = artifact_dir / "public_benchmark_v2_1_manifest.json"
    if not benchmark_manifest_path.is_file():
        raise V21ProtocolError("sealed v2.1 benchmark manifest is missing")
    benchmark_manifest = _strict_json_loads(
        benchmark_manifest_path.read_text(encoding="utf-8"),
        label="v2.1 benchmark manifest",
    )
    if not isinstance(benchmark_manifest, Mapping):
        raise V21ProtocolError("v2.1 benchmark manifest must be an object")
    if benchmark_manifest.get("config_sha256") != sha256_file(config_path):
        raise V21ProtocolError("sealed v2.1 config SHA-256 changed after prepare")
    if benchmark_manifest.get("definitions_sha256") != _canonical_sha256(
        values["benchmark"]["definitions"]
    ):
        raise V21ProtocolError("sealed v2.1 definition SHA-256 changed")
    if benchmark_manifest.get("benchmark_version") != values["benchmark"]["version"]:
        raise V21ProtocolError("sealed v2.1 benchmark version changed")
    source_snapshot = benchmark_manifest.get("implementation_source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise V21ProtocolError("sealed v2.1 implementation source snapshot is missing")
    _verify_implementation_source_snapshot(source_snapshot)
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in NEW_SPLITS:
        data_path = data_dir / f"{split}.jsonl"
        observed = sha256_file(data_path)
        if observed != benchmark_manifest["data_sha256"][split]:
            raise V21ProtocolError(f"sealed v2.1 data changed: {split}")
        parsed = [
            _strict_json_loads(line, label=f"{split} row")
            for line in data_path.read_text(encoding="utf-8").splitlines()
        ]
        if not parsed:
            raise V21ProtocolError(f"sealed v2.1 data is empty: {split}")
        rows[split] = parsed
        task_path = artifact_dir / f"{split}_ai_review_task.csv"
        if sha256_file(task_path) != benchmark_manifest["ai_review_task_sha256"][split]:
            raise V21ProtocolError(f"sealed AI review task changed: {split}")
        fields, tasks = _read_csv(task_path)
        if fields != AI_REVIEW_FIELDS:
            raise V21ProtocolError(f"sealed AI review task schema changed: {split}")
        if _family_task_payload_sha256(tasks) != benchmark_manifest[
            "ai_review_task_payload_sha256"
        ][split]:
            raise V21ProtocolError(f"sealed AI review task payload changed: {split}")
    return data_dir, artifact_dir, rows, benchmark_manifest


def apply_stage_c24b_v21_ai_review(
    config_path: str | Path, audit_spec_path: str | Path
) -> dict[str, Any]:
    _bind_core()
    source = Path(config_path).resolve()
    spec_path = Path(audit_spec_path).resolve()
    values = load_stage_c24b_v21_config(source)
    spec = _load_audit_spec(spec_path)
    artifact_dir = _resolve(source, values["paths"]["artifact_dir"])
    final_manifest_path = artifact_dir / "ai_review_manifest.json"
    if final_manifest_path.exists() or (artifact_dir / "artifact_sha256_manifest.json").exists():
        raise FileExistsError("completed v2.1 AI review cannot be overwritten")
    _, artifact_dir, rows, benchmark_manifest = _verify_prepared_data(
        source, values, before_review=True
    )
    split_summaries: dict[str, Any] = {}
    completed_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in NEW_SPLITS:
        task_path = artifact_dir / f"{split}_ai_review_task.csv"
        fields, task_rows = _read_csv(task_path)
        if fields != AI_REVIEW_FIELDS:
            raise V21ProtocolError(f"AI review task schema changed: {split}")
        task_by_family = {row["phrase_family"]: row for row in task_rows}
        decisions = spec["decisions"][split]
        if not isinstance(decisions, Mapping) or set(decisions) != set(task_by_family):
            raise V21ProtocolError(f"AI decision family coverage differs: {split}")
        completed: list[dict[str, Any]] = []
        boundary = 0
        for family_id, task in sorted(task_by_family.items()):
            decision = decisions[family_id]
            if not isinstance(decision, Mapping) or set(decision) != AI_DECISION_FIELDS:
                raise V21ProtocolError(f"AI decision schema differs: {family_id}")
            confidence = decision["ai_confidence"]
            if confidence is not None:
                if isinstance(confidence, bool):
                    raise V21ProtocolError(f"AI confidence is invalid: {family_id}")
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise V21ProtocolError(f"AI confidence is invalid: {family_id}")
            if not isinstance(decision["status"], str):
                raise V21ProtocolError(f"AI review status is invalid: {family_id}")
            status = decision["status"]
            if status not in AI_STATUSES:
                raise V21ProtocolError(f"AI review status is invalid: {family_id}")
            if (
                not isinstance(decision["ai_suggested_label"], str)
                or not isinstance(decision["ai_suggested_type"], str)
                or decision["ai_suggested_label"] != task["proposed_label"]
                or decision["ai_suggested_type"] != task["sample_type"]
            ):
                raise V21ProtocolError(
                    f"AI suggested a label/type change; create a new benchmark version: {family_id}"
                )
            if not isinstance(decision["construct_validity"], str):
                raise V21ProtocolError(f"AI construct review failed: {family_id}")
            construct_validity = decision["construct_validity"]
            if (
                construct_validity not in AI_CONSTRUCT_VALIDITY
                or status == "failed_requires_revision"
                or (status == "boundary_retained")
                != (construct_validity == "passed_with_boundary")
            ):
                raise V21ProtocolError(f"AI construct review failed: {family_id}")
            flags = decision["shortcut_flags"]
            if (
                not isinstance(flags, list)
                or any(not isinstance(item, str) or not item.strip() for item in flags)
                or len(flags) != len(set(flags))
            ):
                raise V21ProtocolError(f"AI shortcut flags are malformed: {family_id}")
            detected = sorted(_tokens(task["phrase"]) & SHORTCUT_TOKENS)
            if task["sample_type"] == "ambiguous" and detected:
                raise V21ProtocolError(
                    f"ambiguous family retains an explicit shortcut marker: {family_id}"
                )
            if not isinstance(decision["notes"], str) or not decision["notes"].strip():
                raise V21ProtocolError(f"AI semantic note is blank: {family_id}")
            boundary += status == "boundary_retained"
            completed.append(
                {
                    **{field: task[field] for field in AI_REVIEW_FIELDS[:7]},
                    "ai_suggested_label": decision["ai_suggested_label"],
                    "ai_suggested_type": decision["ai_suggested_type"],
                    "ai_confidence": "" if confidence is None else confidence,
                    "construct_validity": decision["construct_validity"],
                    "shortcut_flags": "|".join(flags),
                    "notes": decision["notes"],
                    "ai_review_status": status,
                }
            )
        completed_by_split[split] = completed
        split_summaries[split] = {
            "family_count": len(completed),
            "covered_row_count": sum(int(row["covered_row_count"]) for row in completed),
            "passed_family_count": len(completed) - boundary,
            "boundary_retained_family_count": boundary,
            "failed_family_count": 0,
            "complete": True,
        }
    design = benchmark_manifest["collision_audit"]["design_quality"]
    if not design["combined"]["passed"]:
        raise V21ProtocolError("sealed v2.1 construct-design audit failed")
    review_payloads = {
        split: _render_csv(completed_by_split[split], AI_REVIEW_FIELDS)
        for split in NEW_SPLITS
    }
    review_sha = {
        split: hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for split, payload in review_payloads.items()
    }
    benchmark_manifest_sha = sha256_file(
        artifact_dir / "public_benchmark_v2_1_manifest.json"
    )
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "validated_attempt_started",
        "audit_id": spec["audit_id"],
        "audit_spec_sha256": sha256_file(spec_path),
        "prompt_sha256": spec["reviewer"]["prompt_sha256"],
        "benchmark_manifest_sha256": benchmark_manifest_sha,
        "benchmark_config_sha256": sha256_file(source),
        "review_output_sha256": review_sha,
        "different_audit_spec_retry_allowed": False,
    }
    attempt["manifest_payload_sha256"] = _canonical_sha256(attempt)
    attempt_path = artifact_dir / "ai_review_attempt_manifest.json"
    if attempt_path.exists():
        if _load_payload_sealed_json(
            attempt_path, label="v2.1 AI review attempt manifest"
        ) != attempt:
            raise V21ProtocolError("a different AI review attempt is already sealed")
    else:
        _write_json(attempt_path, attempt)
    for split, payload in review_payloads.items():
        output_path = artifact_dir / f"{split}_ai_review.csv"
        if output_path.exists():
            if sha256_file(output_path) != review_sha[split]:
                raise V21ProtocolError(f"partial AI review output changed: {split}")
        else:
            _atomic_write_text(output_path, payload)
        split_summaries[split].update(
            {"path": str(output_path), "sha256": review_sha[split]}
        )
    reviewer = spec["reviewer"]
    provenance_complete = str(reviewer["model_revision"]) != "unavailable_from_runtime"
    boundary_count = sum(
        summary["boundary_retained_family_count"]
        for summary in split_summaries.values()
    )
    reviewed_family_count = sum(
        summary["family_count"] for summary in split_summaries.values()
    )
    limitation_flags = Counter(
        flag
        for completed in completed_by_split.values()
        for row in completed
        for flag in str(row["shortcut_flags"]).split("|")
        if flag
    )
    limitations = [
        "single-model same-project review is not human review",
        "review is not independent external validation",
        "family-level decisions map to generated rows and are not independent row reviews",
        "locked text was visible to the same-project AI reviewer",
    ]
    if not provenance_complete:
        limitations.append("reviewer model revision is unavailable from this runtime")
    ai_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "audit_id": spec["audit_id"],
        "review_mode": REVIEW_MODE,
        "status": "completed_with_declared_boundaries_exploratory_non_independent",
        "semantic_audit_outcome": "passed_exploratory_non_independent",
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": True,
        "independent_external_validation": False,
        "review_independent_of_benchmark_authorship": False,
        "locked_text_seen_by_same_project_ai_reviewer": True,
        "locked_router_predictions_seen": False,
        "locked_used_for_router_selection": False,
        "prediction_blind": True,
        "model_id": reviewer["model_id"],
        "model_revision": reviewer["model_revision"],
        "ai_review_provenance_complete": provenance_complete,
        "prompt_path": spec["prompt_path"],
        "prompt_sha256": reviewer["prompt_sha256"],
        "audit_spec_path": str(spec_path),
        "audit_spec_sha256": sha256_file(spec_path),
        "benchmark_config_sha256": sha256_file(source),
        "definitions_sha256": benchmark_manifest["definitions_sha256"],
        "implementation_source_aggregate_sha256": benchmark_manifest[
            "implementation_source_snapshot"
        ]["aggregate_sha256"],
        "family_task_sha256": benchmark_manifest["ai_review_task_payload_sha256"],
        "benchmark_manifest_sha256": benchmark_manifest_sha,
        "ai_review_attempt_manifest_sha256": sha256_file(attempt_path),
        "splits": split_summaries,
        "reviewed_family_count": reviewed_family_count,
        "boundary_retained_family_count": boundary_count,
        "boundary_retained_family_rate": boundary_count / reviewed_family_count,
        "declared_limitation_flags": dict(sorted(limitation_flags.items())),
        "design_quality": design,
        "selection_protocol": {
            "router_selection_executed": False,
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
            "locked_router_predictions_observed": False,
        },
        "ai_only_exploratory_calibration_allowed": True,
        "ai_only_exploratory_locked_diagnostic_allowed": True,
        "formal_calibration_allowed": False,
        "formal_calibration_executed": False,
        "formal_locked_audit_executed": False,
        "ready_to_create_new_confirmation_pool": False,
        "new_confirmation_pool_created_after_freeze": False,
        "c3_eligible": False,
        "limitations": limitations,
    }
    ai_manifest["manifest_payload_sha256"] = _canonical_sha256(ai_manifest)
    _write_json(final_manifest_path, ai_manifest)
    status = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "ai_review_completed_with_declared_boundaries",
        "review_mode": REVIEW_MODE,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": True,
        "independent_external_validation": False,
        "ai_review_status": "completed_with_declared_boundaries_exploratory_non_independent",
        "ai_review_provenance_complete": provenance_complete,
        "ai_only_exploratory_calibration_allowed": True,
        "ai_only_exploratory_calibration_executed": False,
        "ai_only_exploratory_locked_diagnostic_allowed": True,
        "ai_only_exploratory_locked_diagnostic_executed": False,
        "formal_calibration_allowed": False,
        "formal_calibration_executed": False,
        "formal_locked_audit_executed": False,
        "ready_to_create_new_confirmation_pool": False,
        "new_confirmation_pool_created_after_freeze": False,
        "c3_eligible": False,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "answer_injection_executed": False,
        "key_or_attack_experiments_run": False,
        "confirmation_created_or_read": False,
        "entity_checkpoint_loaded": False,
        "entity_checkpoint_hash_verified": False,
        "entity_checkpoint_hash_status": "not_applicable_ai_data_review_only",
        "typed_memory_contract_source_sha256": PRESERVED_TYPED_MEMORY_CONTRACT_SHA256,
        "selective_router_source_sha256": PRESERVED_SELECTIVE_ROUTER_SHA256,
        "ai_review_manifest_sha256": sha256_file(final_manifest_path),
    }
    _write_json(artifact_dir / "protocol_status.json", status)
    _write_json(
        artifact_dir / "stage_c24b_v21_summary.json",
        {
            **status,
            "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
            "unique_family_counts": {
                split: len(_family_tasks(split_rows)) for split, split_rows in rows.items()
            },
            "design_quality": design,
            "formal_research_thresholds_assessed": False,
            "discrete_memory_contract_preserved": True,
            "prior_v2_status": "diagnostic_only_failed_requires_revision",
        },
    )
    final_artifact = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "snapshot_role": "ai_reviewed_exploratory_non_independent",
        "files": _artifact_files(artifact_dir, exclude={"artifact_sha256_manifest.json"}),
        "git": _git_state(source.parent.parent),
        "config_sha256": sha256_file(source),
        "audit_spec_sha256": sha256_file(spec_path),
        "prompt_sha256": reviewer["prompt_sha256"],
        "formal_locked_audit_executed": False,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": True,
    }
    final_artifact["file_count"] = len(final_artifact["files"])
    final_artifact["manifest_payload_sha256"] = _canonical_sha256(final_artifact)
    _write_json(artifact_dir / "artifact_sha256_manifest.json", final_artifact)
    return status


def validate_stage_c24b_v21_ai_review(
    config_path: str | Path, *, require_pass: bool = True
) -> dict[str, Any]:
    _bind_core()
    source = Path(config_path).resolve()
    values = load_stage_c24b_v21_config(source)
    _, artifact_dir, rows, benchmark_manifest = _verify_prepared_data(source, values)
    manifest_path = artifact_dir / "ai_review_manifest.json"
    if not manifest_path.is_file():
        if require_pass:
            raise V21ProtocolError("v2.1 AI review is not yet complete")
        return {
            "stage": STAGE_NAME,
            "status": "pending",
            "public_benchmark_human_reviewed": False,
            "public_benchmark_ai_reviewed": False,
            "formal_calibration_allowed": False,
            "formal_locked_audit_executed": False,
            "c3_eligible": False,
        }
    manifest = _load_payload_sealed_json(
        manifest_path, label="v2.1 AI review manifest"
    )
    required_false = (
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
    )
    if (
        manifest.get("public_benchmark_ai_reviewed") is not True
        or manifest.get("prediction_blind") is not True
        or any(manifest.get(key) is not False for key in required_false)
    ):
        raise V21ProtocolError("AI review readiness/independence invariants differ")
    if manifest.get("benchmark_config_sha256") != sha256_file(source):
        raise V21ProtocolError("AI review config seal differs")
    if manifest.get("definitions_sha256") != benchmark_manifest.get(
        "definitions_sha256"
    ):
        raise V21ProtocolError("AI review definition seal differs")
    if manifest.get("implementation_source_aggregate_sha256") != benchmark_manifest.get(
        "implementation_source_snapshot", {}
    ).get("aggregate_sha256"):
        raise V21ProtocolError("AI review implementation-source seal differs")
    if manifest.get("family_task_sha256") != benchmark_manifest.get(
        "ai_review_task_payload_sha256"
    ):
        raise V21ProtocolError("AI review family-task seal differs")
    if manifest.get("benchmark_manifest_sha256") != sha256_file(
        artifact_dir / "public_benchmark_v2_1_manifest.json"
    ):
        raise V21ProtocolError("AI review benchmark-manifest seal differs")
    repo = source.parent.parent.resolve()
    for path_key, sha_key in (
        ("audit_spec_path", "audit_spec_sha256"),
        ("prompt_path", "prompt_sha256"),
    ):
        raw_path = manifest.get(path_key)
        if not isinstance(raw_path, str):
            raise V21ProtocolError(f"AI review {path_key} is invalid")
        path = Path(raw_path).resolve()
        if repo not in path.parents or not path.is_file() or sha256_file(path) != manifest.get(
            sha_key
        ):
            raise V21ProtocolError(f"AI review {path_key} seal differs")
    attempt_path = artifact_dir / "ai_review_attempt_manifest.json"
    attempt = _load_payload_sealed_json(
        attempt_path, label="v2.1 AI review attempt manifest"
    )
    if (
        sha256_file(attempt_path)
        != manifest.get("ai_review_attempt_manifest_sha256")
        or attempt.get("audit_spec_sha256") != manifest.get("audit_spec_sha256")
        or attempt.get("benchmark_manifest_sha256")
        != manifest.get("benchmark_manifest_sha256")
        or attempt.get("different_audit_spec_retry_allowed") is not False
    ):
        raise V21ProtocolError("AI review immutable attempt seal differs")
    if not isinstance(manifest.get("splits"), Mapping) or set(
        manifest["splits"]
    ) != set(NEW_SPLITS):
        raise V21ProtocolError("AI review split summary schema differs")
    observed_family_count = 0
    observed_boundary_count = 0
    observed_limitation_flags: Counter[str] = Counter()
    for split in NEW_SPLITS:
        path = artifact_dir / f"{split}_ai_review.csv"
        fields, review_rows = _read_csv(path)
        if fields != AI_REVIEW_FIELDS:
            raise V21ProtocolError(f"AI review CSV schema differs: {split}")
        if any(
            any(token in field for token in ("reviewer_", "adjudicated", "router", "logit", "evidence", "retrieval"))
            for field in fields
        ):
            raise V21ProtocolError("AI review CSV contains human/router fields")
        if sha256_file(path) != manifest["splits"][split]["sha256"]:
            raise V21ProtocolError(f"AI review CSV SHA-256 changed: {split}")
        expected = {row["phrase_family"]: row for row in _family_tasks(rows[split])}
        observed = {row["phrase_family"]: row for row in review_rows}
        if len(observed) != len(review_rows) or set(observed) != set(expected):
            raise V21ProtocolError(f"AI review family coverage changed: {split}")
        for family_id, expected_row in expected.items():
            row = observed[family_id]
            for field in AI_REVIEW_FIELDS[:7]:
                if str(row[field]) != str(expected_row[field]):
                    raise V21ProtocolError(f"AI review static field changed: {family_id}.{field}")
            if row["ai_suggested_label"] != row["proposed_label"] or row[
                "ai_suggested_type"
            ] != row["sample_type"]:
                raise V21ProtocolError("AI review cannot overlay a label/type change")
            if (
                row["construct_validity"] not in AI_CONSTRUCT_VALIDITY
                or row["ai_review_status"] not in {"passed", "boundary_retained"}
                or (row["ai_review_status"] == "boundary_retained")
                != (row["construct_validity"] == "passed_with_boundary")
            ):
                raise V21ProtocolError(f"AI review did not pass: {family_id}")
            if row["ai_confidence"]:
                confidence = float(row["ai_confidence"])
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise V21ProtocolError(f"AI confidence changed: {family_id}")
        summary = manifest["splits"][split]
        boundary_count = sum(
            row["ai_review_status"] == "boundary_retained" for row in review_rows
        )
        if (
            summary.get("family_count") != len(review_rows)
            or summary.get("covered_row_count")
            != sum(int(row["covered_row_count"]) for row in review_rows)
            or summary.get("boundary_retained_family_count") != boundary_count
            or summary.get("passed_family_count") != len(review_rows) - boundary_count
            or summary.get("failed_family_count") != 0
            or summary.get("complete") is not True
        ):
            raise V21ProtocolError(f"AI review split summary changed: {split}")
        observed_family_count += len(review_rows)
        observed_boundary_count += boundary_count
        observed_limitation_flags.update(
            flag
            for row in review_rows
            for flag in row["shortcut_flags"].split("|")
            if flag
        )
    if (
        manifest.get("reviewed_family_count") != observed_family_count
        or manifest.get("boundary_retained_family_count") != observed_boundary_count
        or not math.isclose(
            float(manifest.get("boundary_retained_family_rate", -1.0)),
            observed_boundary_count / observed_family_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or manifest.get("declared_limitation_flags")
        != dict(sorted(observed_limitation_flags.items()))
    ):
        raise V21ProtocolError("AI review aggregate boundary summary changed")
    artifact_path = artifact_dir / "artifact_sha256_manifest.json"
    artifact = _load_exact_artifact_manifest(
        artifact_dir, artifact_path, label="v2.1 final artifact manifest"
    )
    if (
        artifact.get("config_sha256") != sha256_file(source)
        or artifact.get("audit_spec_sha256") != manifest.get("audit_spec_sha256")
        or artifact.get("prompt_sha256") != manifest.get("prompt_sha256")
        or artifact.get("formal_locked_audit_executed") is not False
        or artifact.get("public_benchmark_human_reviewed") is not False
        or artifact.get("public_benchmark_ai_reviewed") is not True
    ):
        raise V21ProtocolError("final artifact protocol fields differ")
    for name in ("protocol_status.json", "stage_c24b_v21_summary.json"):
        value = _strict_json_loads(
            (artifact_dir / name).read_text(encoding="utf-8"), label=name
        )
        if not isinstance(value, Mapping) or any(
            value.get(key) is not False
            for key in (
                "public_benchmark_human_reviewed",
                "independent_external_validation",
                "formal_calibration_allowed",
                "formal_calibration_executed",
                "formal_locked_audit_executed",
                "ready_to_create_new_confirmation_pool",
                "new_confirmation_pool_created_after_freeze",
                "c3_eligible",
            )
        ):
            raise V21ProtocolError(f"{name} readiness invariants differ")
    expected_status = "completed_with_declared_boundaries_exploratory_non_independent"
    if require_pass and manifest.get("status") != expected_status:
        raise V21ProtocolError("v2.1 AI review status is not an exploratory completion")
    return {
        **manifest,
        "formal_calibration_allowed": False,
        "formal_locked_audit_executed": False,
        "ready_to_create_new_confirmation_pool": False,
        "c3_eligible": False,
    }


__all__ = [
    "V21ProtocolError",
    "apply_stage_c24b_v21_ai_review",
    "validate_stage_c24b_v21_ai_review",
]
