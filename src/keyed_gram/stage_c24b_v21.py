"""Stage C2.4b v2.1：单模型 AI 审核的数据设计修订层。

该模块故意不复用原真人双审 formal 后端。它只负责：封存旧 v2 的
预评分失败结论、物化全新 v2.1 数据、生成 family-level AI 审核任务，
以及验证一个显式、prediction-blind、非外部独立的 AI 审核文件。
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .stage_c24b_benchmark import (
    RELATIONS,
    REVIEW_FIELDS,
    ROW_FIELDS,
    build_public_benchmark_v2,
    sha256_file,
)
from .stage_c24b_v21_seal import (
    V21ProtocolError,
    artifact_files as _artifact_files,
    atomic_write_text as _atomic_write_text,
    canonical_sha256 as _canonical_sha256,
    load_exact_artifact_manifest as _load_exact_artifact_manifest,
    load_payload_sealed_json as _load_payload_sealed_json,
    read_csv as _read_csv,
    render_csv as _render_csv,
    safe_artifact_path as _safe_artifact_path,
    strict_json_loads as _strict_json_loads,
    verify_prepare_snapshot as _verify_prepare_snapshot_seal,
    write_csv as _write_csv,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)


SCHEMA_VERSION = 1
STAGE_NAME = "C2.4b-v2.1-ai-only-selective-router-benchmark"
REVIEW_MODE = "single_ai_semantic_audit"
PRESERVED_TYPED_MEMORY_CONTRACT_SHA256 = (
    "c8a3b5fd33621349659e504b1bd31502688896cf63207ae59f644849a0d96692"
)
PRESERVED_SELECTIVE_ROUTER_SHA256 = (
    "4c791aa1a594b7fadd64226ecdd19465aefb54500c3ee92339cf011c01a18175"
)
OLD_TO_NEW_SPLITS = {
    "public_train_v2": "public_train_v2_1",
    "public_calibration_v2": "public_calibration_v2_1",
    "public_locked_audit_v2": "public_locked_audit_v2_1",
}
NEW_SPLITS = tuple(OLD_TO_NEW_SPLITS.values())
AI_REVIEW_FIELDS = (
    "phrase_family",
    "split",
    "phrase",
    "proposed_label",
    "sample_type",
    "covered_row_count",
    "covered_row_ids_sha256",
    "ai_suggested_label",
    "ai_suggested_type",
    "ai_confidence",
    "construct_validity",
    "shortcut_flags",
    "notes",
    "ai_review_status",
)
AI_DECISION_FIELDS = frozenset(
    {
        "ai_suggested_label",
        "ai_suggested_type",
        "ai_confidence",
        "construct_validity",
        "shortcut_flags",
        "notes",
        "status",
    }
)
AI_STATUSES = {"passed", "failed_requires_revision", "boundary_retained"}
AI_CONSTRUCT_VALIDITY = {"passed", "passed_with_boundary"}
SHORTCUT_TOKENS = {
    "unknown",
    "without",
    "lacking",
    "no",
    "unspecified",
    "unstated",
    "unnamed",
    "detached",
    "missing",
    "unscoped",
    "schema-free",
    "unqualified",
    "namespace-absent",
    "purpose-free",
    "issuer-unknown",
    "domainless",
    "context-free",
}
TARGET_CARRIER_TOKENS = {
    "code",
    "key",
    "reference",
    "token",
    "identifier",
    "credential",
    "serial",
    "access",
    "location",
    "locator",
    "registry",
    "enrollment",
    "authorization",
    "login",
    "filing",
    "archive",
    "geographic",
    "municipal",
}
LEGACY_V2_EXCEPTION_DECISIONS = {
    "locked_v2_ambiguous_dossier_handle": ("registry_id", "known", 0.84),
    "locked_v2_ambiguous_authorization_value": ("access_code", "known", 0.91),
    "locked_ambiguous_record_tag": ("registry_id", "known", 0.80),
    "locked_ambiguous_subject_locator": ("registry_id", "known", 0.86),
}
LEGACY_V2_BOUNDARY_FAMILIES = {
    "calibration_v2_ambiguous_credential_use",
    "calibration_v2_ambiguous_security_reference",
}
_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)?")
_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "stage",
        "base_protocol",
        "review_protocol",
        "paths",
        "legacy_v2_ai_audit",
        "benchmark",
    }
)
_BASE_FIELDS = frozenset(
    {
        "supersedes",
        "source_manifest",
        "source_manifest_sha256",
        "disposition",
        "changes_superseded_files",
    }
)
_REVIEW_PROTOCOL_FIELDS = frozenset(
    {
        "review_mode",
        "public_benchmark_human_reviewed",
        "public_benchmark_ai_reviewed",
        "independent_external_validation",
        "ai_review_status",
        "formal_calibration_allowed",
        "formal_locked_audit_executed",
        "new_confirmation_pool_created_after_freeze",
        "c3_eligible",
    }
)
_AUDIT_SPEC_FIELDS = frozenset(
    {"schema_version", "audit_id", "review_mode", "reviewer", "decisions"}
)
_AI_REVIEWER_FIELDS = frozenset(
    {
        "model_id",
        "model_revision",
        "prompt_sha256",
        "prediction_blind",
        "independent_external",
    }
)
_PROHIBITED_AI_INPUT_KEYS = {
    "router_prediction",
    "router_predictions",
    "evidence",
    "evidence_score",
    "evidence_scores",
    "logit",
    "logits",
    "threshold",
    "thresholds",
    "retrieval",
    "retrieval_result",
    "memory_result",
    "private_answer",
    "private_answers",
    "confirmation",
}


def _resolve(config_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    repo_candidate = config_path.parent.parent / path
    local_candidate = config_path.parent / path
    if repo_candidate.exists() or not local_candidate.exists():
        return repo_candidate.resolve()
    return local_candidate.resolve()


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain=v1")),
    }


def _implementation_source_snapshot() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    candidates = [repo / "pyproject.toml", *sorted((repo / "src" / "keyed_gram").glob("*.py"))]
    files = [
        {
            "path": path.relative_to(repo).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in candidates
        if path.is_file()
    ]
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "scope": "pyproject_and_src_keyed_gram_python",
        "files": files,
        "file_count": len(files),
        "preserved_typed_memory_contract_sha256": PRESERVED_TYPED_MEMORY_CONTRACT_SHA256,
        "preserved_selective_router_sha256": PRESERVED_SELECTIVE_ROUTER_SHA256,
    }
    observed = {row["path"]: row["sha256"] for row in files}
    if (
        observed.get("src/keyed_gram/stage_c24_contract.py")
        != PRESERVED_TYPED_MEMORY_CONTRACT_SHA256
        or observed.get("src/keyed_gram/stage_c24b_router.py")
        != PRESERVED_SELECTIVE_ROUTER_SHA256
    ):
        raise V21ProtocolError("C2.4b typed memory/router source hash changed")
    snapshot["aggregate_sha256"] = _canonical_sha256(snapshot)
    return snapshot


def _verify_implementation_source_snapshot(expected: Mapping[str, Any]) -> None:
    observed = _implementation_source_snapshot()
    if dict(expected) != observed:
        raise V21ProtocolError("v2.1 implementation source changed after prepare")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(str(value).casefold()))


def _reject_prohibited_ai_input(value: Any, *, location: str = "audit") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if normalized in _PROHIBITED_AI_INPUT_KEYS or normalized.startswith(
                (
                    "router_prediction",
                    "semantic_embedding",
                    "retrieval",
                    "private_answer",
                    "private_value",
                    "confirmation",
                )
            ):
                raise V21ProtocolError(f"prohibited AI-review input at {location}.{key}")
            _reject_prohibited_ai_input(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_prohibited_ai_input(child, location=f"{location}[{index}]")


def load_stage_c24b_v21_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or set(raw) != _TOP_LEVEL:
        raise V21ProtocolError("v2.1 config top-level schema differs")
    values = copy.deepcopy(dict(raw))
    if values["schema_version"] != SCHEMA_VERSION or values["stage"] != STAGE_NAME:
        raise V21ProtocolError("unsupported v2.1 schema/stage")
    base = values["base_protocol"]
    if not isinstance(base, Mapping) or set(base) != _BASE_FIELDS:
        raise V21ProtocolError("base_protocol schema differs")
    if (
        base["supersedes"] != "c24b-public-selective-router-v2"
        or base["disposition"] != "superseded_before_model_scoring_diagnostic_only"
        or base["changes_superseded_files"] is not False
    ):
        raise V21ProtocolError("v2 supersession must be diagnostic-only and immutable")
    source_manifest = _resolve(source, str(base["source_manifest"]))
    if sha256_file(source_manifest) != str(base["source_manifest_sha256"]):
        raise V21ProtocolError("superseded v2 source manifest SHA-256 mismatch")
    review = values["review_protocol"]
    if not isinstance(review, Mapping) or set(review) != _REVIEW_PROTOCOL_FIELDS:
        raise V21ProtocolError("review_protocol schema differs")
    required_false = {
        "public_benchmark_human_reviewed",
        "public_benchmark_ai_reviewed",
        "independent_external_validation",
        "formal_calibration_allowed",
        "formal_locked_audit_executed",
        "new_confirmation_pool_created_after_freeze",
        "c3_eligible",
    }
    if review["review_mode"] != REVIEW_MODE or any(review[key] is not False for key in required_false):
        raise V21ProtocolError("v2.1 must begin as non-human pending AI review")
    if review["ai_review_status"] != "pending_prediction_blind_single_ai_review":
        raise V21ProtocolError("v2.1 must begin with a pending AI review")
    paths = values["paths"]
    if not isinstance(paths, Mapping) or set(paths) != {"public_data_dir", "artifact_dir"}:
        raise V21ProtocolError("v2.1 paths schema differs")
    if paths["public_data_dir"] != "data/stage_c24b_v21" or paths[
        "artifact_dir"
    ] != "artifacts/stage_c24b_v21":
        raise V21ProtocolError("v2.1 must use its independent data/artifact namespace")
    legacy = values["legacy_v2_ai_audit"]
    required_legacy = {
        "source_review_files",
        "total_rows",
        "total_unique_families",
        "consistent_families",
        "relabel_families",
        "affected_rows",
        "exceptions",
        "boundary_samples",
        "ambiguous_shortcut_tokens",
        "expected_ambiguous_family_count",
    }
    if not isinstance(legacy, Mapping) or set(legacy) != required_legacy:
        raise V21ProtocolError("legacy_v2_ai_audit schema differs")
    if (
        legacy["total_rows"] != 512
        or legacy["total_unique_families"] != 134
        or legacy["consistent_families"] != 130
        or legacy["relabel_families"] != 4
        or legacy["affected_rows"] != 16
        or legacy["expected_ambiguous_family_count"] != 16
    ):
        raise V21ProtocolError("user-supplied legacy AI audit counts differ")
    exceptions = legacy["exceptions"]
    if not isinstance(exceptions, list) or {
        str(item.get("phrase_family")) for item in exceptions if isinstance(item, Mapping)
    } != set(LEGACY_V2_EXCEPTION_DECISIONS):
        raise V21ProtocolError("user-supplied legacy AI exception families differ")
    for item in exceptions:
        if not isinstance(item, Mapping) or set(item) != {
            "phrase_family",
            "ai_suggested_label",
            "ai_suggested_type",
            "ai_confidence",
            "rationale",
        }:
            raise V21ProtocolError("user-supplied legacy AI exception schema differs")
        expected = LEGACY_V2_EXCEPTION_DECISIONS[str(item["phrase_family"])]
        if (
            item["ai_suggested_label"] != expected[0]
            or item["ai_suggested_type"] != expected[1]
            or float(item["ai_confidence"]) != expected[2]
            or not isinstance(item["rationale"], str)
            or not item["rationale"].strip()
        ):
            raise V21ProtocolError("user-supplied legacy AI exception decision differs")
    boundaries = legacy["boundary_samples"]
    if not isinstance(boundaries, list) or {
        str(item.get("phrase_family")) for item in boundaries if isinstance(item, Mapping)
    } != LEGACY_V2_BOUNDARY_FAMILIES:
        raise V21ProtocolError("user-supplied legacy AI boundary families differ")
    benchmark = values["benchmark"]
    if not isinstance(benchmark, Mapping) or set(benchmark.get("splits", {})) != set(
        NEW_SPLITS
    ):
        raise V21ProtocolError("v2.1 benchmark must contain exactly three new split names")
    if benchmark.get("review_status") != "pending_single_ai_semantic_audit":
        raise V21ProtocolError("v2.1 benchmark review status differs")
    if set(benchmark.get("historical_sources", {})) != {
        "stage_c23_config",
        "stage_c24_config",
        "stage_c24b_v2_config",
    }:
        raise V21ProtocolError("v2.1 history must include C2.3, C2.4 and sealed v2")
    return values


def _legacy_projection(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(benchmark))
    projected["review_status"] = "pending_independent_dual_review"
    projected["splits"] = {
        old: copy.deepcopy(benchmark["splits"][new])
        for old, new in OLD_TO_NEW_SPLITS.items()
    }
    return projected


def _rename_split_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        renamed: dict[str, Any] = {}
        for key, child in value.items():
            renamed_key = str(key)
            for old, new in OLD_TO_NEW_SPLITS.items():
                renamed_key = renamed_key.replace(old, new)
            renamed[renamed_key] = _rename_split_value(child)
        return renamed
    if isinstance(value, list):
        return [_rename_split_value(child) for child in value]
    if isinstance(value, str):
        output = value
        for old, new in OLD_TO_NEW_SPLITS.items():
            output = output.replace(old, new)
        return output
    return value


def _family_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["family_id"]), []).append(row)
    tasks: list[dict[str, Any]] = []
    for family_id, family_rows in sorted(groups.items()):
        phrases = {str(row["phrase"]) for row in family_rows}
        types = {str(row["sample_type"]) for row in family_rows}
        labels = {
            str(row["relation_id"] or row["sample_type"]) for row in family_rows
        }
        splits = {str(row["split"]) for row in family_rows}
        if any(len(values) != 1 for values in (phrases, types, labels, splits)):
            raise V21ProtocolError(f"family task is inconsistent: {family_id}")
        tasks.append(
            {
                "phrase_family": family_id,
                "split": next(iter(splits)),
                "phrase": next(iter(phrases)),
                "proposed_label": next(iter(labels)),
                "sample_type": next(iter(types)),
                "covered_row_count": len(family_rows),
                "covered_row_ids_sha256": _canonical_sha256(
                    sorted(str(row["row_id"]) for row in family_rows)
                ),
                "ai_suggested_label": "",
                "ai_suggested_type": "",
                "ai_confidence": "",
                "construct_validity": "",
                "shortcut_flags": "",
                "notes": "",
                "ai_review_status": "pending",
            }
        )
    return tasks


def _family_task_payload_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    """绑定审核任务的不可变语义字段，不把待填写的 AI 输出纳入 seal。"""

    return _canonical_sha256(
        [
            {field: str(task[field]) for field in AI_REVIEW_FIELDS[:7]}
            for task in tasks
        ]
    )


def _design_quality(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    all_ambiguous = 0
    all_shortcuts = 0
    all_unrelated = 0
    all_hard_carrier = 0
    for split in ("public_calibration_v2_1", "public_locked_audit_v2_1"):
        unique = {str(row["family_id"]): row for row in rows[split]}
        ambiguous = [row for row in unique.values() if row["sample_type"] == "ambiguous"]
        unrelated = [row for row in unique.values() if row["sample_type"] == "unrelated"]
        shortcut_rows = [
            str(row["family_id"])
            for row in ambiguous
            if _tokens(str(row["phrase"])) & SHORTCUT_TOKENS
        ]
        carrier_rows = [
            str(row["family_id"])
            for row in unrelated
            if _tokens(str(row["phrase"])) & TARGET_CARRIER_TOKENS
        ]
        output[split] = {
            "ambiguous_family_count": len(ambiguous),
            "ambiguous_explicit_shortcut_count": len(shortcut_rows),
            "ambiguous_explicit_shortcut_rate": len(shortcut_rows) / len(ambiguous),
            "ambiguous_without_explicit_shortcut_count": len(ambiguous) - len(shortcut_rows),
            "shortcut_family_ids": shortcut_rows,
            "unrelated_family_count": len(unrelated),
            "hard_unrelated_shared_carrier_count": len(carrier_rows),
            "hard_unrelated_shared_carrier_rate": len(carrier_rows) / len(unrelated),
            "hard_unrelated_family_ids": carrier_rows,
        }
        all_ambiguous += len(ambiguous)
        all_shortcuts += len(shortcut_rows)
        all_unrelated += len(unrelated)
        all_hard_carrier += len(carrier_rows)
    output["combined"] = {
        "ambiguous_family_count": all_ambiguous,
        "ambiguous_explicit_shortcut_count": all_shortcuts,
        "ambiguous_explicit_shortcut_rate": all_shortcuts / all_ambiguous,
        "hard_unrelated_family_count": all_unrelated,
        "hard_unrelated_shared_carrier_count": all_hard_carrier,
        "hard_unrelated_shared_carrier_rate": all_hard_carrier / all_unrelated,
        "ambiguous_shortcut_gate_maximum": 0.0,
        "hard_unrelated_carrier_gate_minimum": 0.75,
        "passed": all_shortcuts == 0
        and all_hard_carrier / all_unrelated >= 0.75,
    }
    return output


def build_stage_c24b_v21(
    config_path: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    source = Path(config_path).resolve()
    values = load_stage_c24b_v21_config(source)
    projected = _legacy_projection(values["benchmark"])
    old_rows, old_audit = build_public_benchmark_v2(
        projected, config_path=source, run_historical_audit=True
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for old_split, new_split in OLD_TO_NEW_SPLITS.items():
        converted: list[dict[str, Any]] = []
        for source_row in old_rows[old_split]:
            row = _rename_split_value(dict(source_row))
            row["split"] = new_split
            row["fact_id"] = str(row["fact_id"]).replace("public-v2:", "public-v2.1:")
            if set(row) != ROW_FIELDS:
                raise V21ProtocolError("v2.1 row schema changed during version projection")
            converted.append(row)
        rows[new_split] = converted
    if {split: len(split_rows) for split, split_rows in rows.items()} != {
        "public_train_v2_1": 72,
        "public_calibration_v2_1": 160,
        "public_locked_audit_v2_1": 160,
    }:
        raise V21ProtocolError("v2.1 row counts differ from the preregistered design")
    if len({str(row["row_id"]) for split_rows in rows.values() for row in split_rows}) != 392:
        raise V21ProtocolError("v2.1 row IDs are not globally unique")
    design = _design_quality(rows)
    if not design["combined"]["passed"]:
        raise V21ProtocolError("v2.1 ambiguity/unrelated design gates failed")
    audit = _rename_split_value(old_audit)
    audit.update(
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark_version": values["benchmark"]["version"],
            "split_names": list(NEW_SPLITS),
            "public_benchmark_human_reviewed": False,
            "public_benchmark_ai_reviewed": False,
            "independent_external_validation": False,
            "formal_calibration_allowed": False,
            "formal_locked_audit_executed": False,
            "design_quality": design,
            "passed": bool(old_audit["passed"] and design["combined"]["passed"]),
        }
    )
    return rows, audit


def _validate_legacy_audit_sources(config_path: Path, values: Mapping[str, Any]) -> dict[str, Any]:
    legacy = values["legacy_v2_ai_audit"]
    sources = legacy["source_review_files"]
    expected_names = {
        "public_train_v2",
        "public_calibration_v2",
        "public_locked_audit_v2",
        "legacy_c24_locked_audit",
    }
    if not isinstance(sources, Mapping) or set(sources) != expected_names:
        raise V21ProtocolError("legacy AI audit source set differs")
    family_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    source_seals: dict[str, Any] = {}
    all_rows: list[dict[str, str]] = []
    for name, seal in sources.items():
        if not isinstance(seal, Mapping) or set(seal) != {"path", "sha256", "rows", "families"}:
            raise V21ProtocolError(f"legacy source seal schema differs: {name}")
        path = _resolve(config_path, str(seal["path"]))
        if sha256_file(path) != str(seal["sha256"]):
            raise V21ProtocolError(f"legacy review source SHA-256 changed: {name}")
        fields, rows = _read_csv(path)
        if fields != REVIEW_FIELDS or len(rows) != int(seal["rows"]):
            raise V21ProtocolError(f"legacy review source schema/count changed: {name}")
        for row in rows:
            if any(
                str(row[field]).strip()
                for field in (
                    "reviewer_1_label",
                    "reviewer_2_label",
                    "reviewer_1_type",
                    "reviewer_2_type",
                    "adjudicated_label",
                    "adjudicated_type",
                    "notes",
                )
            ) or row["review_status"] != "pending":
                raise V21ProtocolError("legacy human-review fields were modified or fabricated")
            family_rows.setdefault((name, row["phrase_family"]), []).append(row)
        families = len({row["phrase_family"] for row in rows})
        if families != int(seal["families"]):
            raise V21ProtocolError(f"legacy unique-family count changed: {name}")
        all_rows.extend(rows)
        source_seals[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "unique_family_count": families,
            "human_review_fields_all_blank": True,
        }
    if len(all_rows) != 512 or len(family_rows) != 134:
        raise V21ProtocolError("legacy family-to-row mapping differs from 134/512")
    exceptions = {str(row["phrase_family"]): row for row in legacy["exceptions"]}
    if len(exceptions) != 4:
        raise V21ProtocolError("legacy AI audit must preserve exactly four exceptions")
    affected = 0
    for family_id, exception in exceptions.items():
        matches = [rows for (_, family), rows in family_rows.items() if family == family_id]
        if len(matches) != 1 or len(matches[0]) != 4:
            raise V21ProtocolError(f"legacy AI exception row mapping differs: {family_id}")
        affected += len(matches[0])
        if matches[0][0]["sample_type"] != "ambiguous":
            raise V21ProtocolError("legacy exception no longer originates from ambiguous")
        confidence = float(exception["ai_confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise V21ProtocolError("legacy AI confidence is non-finite or out of range")
    if affected != 16:
        raise V21ProtocolError("legacy exception affected-row count differs")
    current_v2 = [
        rows[0]
        for (source_name, _), rows in family_rows.items()
        if source_name in {"public_calibration_v2", "public_locked_audit_v2"}
    ]
    ambiguous = [row for row in current_v2 if row["sample_type"] == "ambiguous"]
    unrelated = [row for row in current_v2 if row["sample_type"] == "unrelated"]
    shortcuts = [row for row in ambiguous if _tokens(row["phrase"]) & SHORTCUT_TOKENS]
    carrier = [row for row in unrelated if _tokens(row["phrase"]) & TARGET_CARRIER_TOKENS]
    if len(ambiguous) != 16 or len(shortcuts) != 16 or len(unrelated) != 16 or carrier:
        raise V21ProtocolError("legacy ambiguity shortcut/unrelated hardness finding changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_source": "user_supplied_single_ai_semantic_audit_summary",
        "model_id": "not_provided_by_user",
        "model_revision": "not_provided_by_user",
        "ai_review_provenance_complete": False,
        "independent_external_validation": False,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": True,
        "ai_review_status": "failed_requires_revision",
        "label_correctness": "failed_requires_revision",
        "ambiguity_construct_validity": "failed",
        "unrelated_hardness": "insufficient",
        "known_family_consistency": "passed",
        "formal_calibration_allowed": False,
        "formal_locked_audit_executed": False,
        "unique_family_decisions_reported": 134,
        "mapped_generated_rows": 512,
        "independent_row_reviews_claimed": False,
        "consistent_families": 130,
        "relabel_families": 4,
        "affected_rows": 16,
        "ambiguous_explicit_shortcut_count": 16,
        "ambiguous_family_count": 16,
        "unrelated_shared_carrier_count": 0,
        "unrelated_family_count": 16,
        "exceptions": list(legacy["exceptions"]),
        "boundary_samples": list(legacy["boundary_samples"]),
        "source_review_files": source_seals,
    }


def prepare_stage_c24b_v21(config_path: str | Path) -> dict[str, Any]:
    source = Path(config_path).resolve()
    values = load_stage_c24b_v21_config(source)
    data_dir = _resolve(source, values["paths"]["public_data_dir"])
    artifact_dir = _resolve(source, values["paths"]["artifact_dir"])
    if data_dir.exists() or artifact_dir.exists():
        raise FileExistsError("prepared C2.4b v2.1 data/artifacts cannot be overwritten")
    rows, audit = build_stage_c24b_v21(source)
    legacy_audit = _validate_legacy_audit_sources(source, values)
    repo = source.parent.parent
    git = _git_state(repo)
    data_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    data_sha: dict[str, str] = {}
    task_sha: dict[str, str] = {}
    task_payload_sha: dict[str, str] = {}
    split_manifests: dict[str, Any] = {}
    for split, split_rows in rows.items():
        data_path = data_dir / f"{split}.jsonl"
        _write_jsonl(data_path, split_rows)
        tasks = _family_tasks(split_rows)
        task_path = artifact_dir / f"{split}_ai_review_task.csv"
        _write_csv(task_path, tasks, AI_REVIEW_FIELDS)
        data_sha[split] = sha256_file(data_path)
        task_sha[split] = sha256_file(task_path)
        task_payload_sha[split] = _family_task_payload_sha256(tasks)
        counts = Counter(str(row["sample_type"]) for row in split_rows)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_version": values["benchmark"]["version"],
            "split": split,
            "row_count": len(split_rows),
            "unique_family_count": len(tasks),
            "sample_type_counts": dict(sorted(counts.items())),
            "data_path": str(data_path),
            "data_sha256": data_sha[split],
            "ai_review_task_path": str(task_path),
            "ai_review_task_sha256": task_sha[split],
            "ai_review_task_payload_sha256": task_payload_sha[split],
            "public_benchmark_human_reviewed": False,
            "public_benchmark_ai_reviewed": False,
            "formal_calibration_allowed": False,
            "formal_locked_audit_executed": False,
            "git": git,
            "config_sha256": sha256_file(source),
        }
        manifest_path = artifact_dir / f"{split}_manifest.json"
        _write_json(manifest_path, manifest)
        split_manifests[split] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    _write_json(artifact_dir / "prior_v2_ai_audit_summary.json", legacy_audit)
    supersession = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": values["base_protocol"]["supersedes"],
        "benchmark_role": "data_design_diagnostic_only",
        "status": "superseded_before_model_scoring",
        "protocol_incident": False,
        "router_predictions_observed": False,
        "locked_used_for_selection": False,
        "formal_locked_audit_executed": False,
        "formal_eligible": False,
        "reason": "pre-scoring semantic-label and construct-validity failure",
        "superseded_by": values["benchmark"]["version"],
        "changes_superseded_files": False,
        "source_manifest": values["base_protocol"]["source_manifest"],
        "source_manifest_sha256": values["base_protocol"]["source_manifest_sha256"],
        "old_c24_readiness_changed": False,
    }
    _write_json(artifact_dir / "v2_supersession_manifest.json", supersession)
    benchmark_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "benchmark_version": values["benchmark"]["version"],
        "status": "prepared_pending_prediction_blind_single_ai_review",
        "review_mode": REVIEW_MODE,
        "answer_free": True,
        "contains_private_answers": False,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": False,
        "independent_external_validation": False,
        "formal_calibration_allowed": False,
        "formal_locked_audit_executed": False,
        "split_manifests": split_manifests,
        "data_sha256": data_sha,
        "ai_review_task_sha256": task_sha,
        "ai_review_task_payload_sha256": task_payload_sha,
        "collision_audit": audit,
        "legacy_v2_ai_audit_sha256": sha256_file(
            artifact_dir / "prior_v2_ai_audit_summary.json"
        ),
        "v2_supersession_manifest_sha256": sha256_file(
            artifact_dir / "v2_supersession_manifest.json"
        ),
        "config_sha256": sha256_file(source),
        "definitions_sha256": _canonical_sha256(values["benchmark"]["definitions"]),
        "implementation_source_snapshot": _implementation_source_snapshot(),
        "git": git,
    }
    _write_json(artifact_dir / "public_benchmark_v2_1_manifest.json", benchmark_manifest)
    status = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "prepared_pending_prediction_blind_single_ai_review",
        "review_mode": REVIEW_MODE,
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": False,
        "independent_external_validation": False,
        "ai_review_status": "pending",
        "ai_only_exploratory_calibration_allowed": False,
        "ai_only_exploratory_locked_diagnostic_allowed": False,
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
        "benchmark_manifest_sha256": sha256_file(
            artifact_dir / "public_benchmark_v2_1_manifest.json"
        ),
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
            "design_quality": audit["design_quality"],
            "historical_collision_audit": audit["historical_collision_audit"],
            "formal_research_thresholds_assessed": False,
            "discrete_memory_contract_preserved": True,
        },
    )
    prepare_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "snapshot_role": "prepared_pending_ai_review",
        "files": _artifact_files(
            artifact_dir, exclude={"prepare_artifact_sha256_manifest.json"}
        ),
        "git": git,
        "config_sha256": sha256_file(source),
    }
    prepare_manifest["file_count"] = len(prepare_manifest["files"])
    prepare_manifest["manifest_payload_sha256"] = _canonical_sha256(prepare_manifest)
    _write_json(artifact_dir / "prepare_artifact_sha256_manifest.json", prepare_manifest)
    return {
        "stage": STAGE_NAME,
        "status": status["status"],
        "artifact_dir": str(artifact_dir),
        "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "unique_family_counts": {
            split: len(_family_tasks(split_rows)) for split, split_rows in rows.items()
        },
        "public_benchmark_human_reviewed": False,
        "public_benchmark_ai_reviewed": False,
        "formal_calibration_allowed": False,
        "formal_locked_audit_executed": False,
        "c3_eligible": False,
    }


from .stage_c24b_v21_review import (
    apply_stage_c24b_v21_ai_review,
    validate_stage_c24b_v21_ai_review,
)

__all__ = [
    "AI_REVIEW_FIELDS",
    "NEW_SPLITS",
    "STAGE_NAME",
    "V21ProtocolError",
    "apply_stage_c24b_v21_ai_review",
    "build_stage_c24b_v21",
    "load_stage_c24b_v21_config",
    "prepare_stage_c24b_v21",
    "validate_stage_c24b_v21_ai_review",
]
