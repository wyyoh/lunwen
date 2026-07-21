"""Stage C2.4b 选择性离散 relation router 的协议编排。

该模块只编排公开、answer-free 的 relation 证据与单 bucket 检索审计。
正式流程在任何模型加载之前验证人工审核；smoke 流程只使用独立的
synthetic mock rows，绝不打开正式 ``public_locked_audit_v2``。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from .phase_a import _sha256_file
from .stage_c23_semantic import (
    SEMANTIC_ENCODER_SPECS,
    build_model_file_manifest,
    encode_semantic_texts,
    load_semantic_encoder,
)
from .stage_c21 import _load_feature_cache
from .stage_c23 import validate_answer_free_private_feature_cache
from .stage_c24 import load_fixed_s5_head
from .stage_c24_contract import AcceptedRoute, RelationId
from .stage_c24_contract import assert_discrete_memory_contract
from .stage_c24b_benchmark import (
    PublicSplitV2,
    REVIEW_FIELDS,
    build_legacy_c24_review_rows,
    build_public_benchmark_v2,
    build_public_split_v2,
    build_review_rows,
    prepare_public_benchmark_v2,
    review_static_sha256,
    sha256_file,
    sha256_json,
    strict_json_dumps,
)
from .stage_c24b_calibration import (
    ClassConditionalConformalCalibrator,
    FrozenRouterSelection,
    PairwiseRidgeEvidenceHead,
    build_pairwise_relation_features,
    fit_class_conditional_conformal,
    fit_pairwise_ridge_head,
    freeze_router_selection,
    select_router_on_calibration,
)
from .stage_c24b_metrics import (
    access_control_metrics,
    assert_finite_json,
    compute_readiness,
    fact_retrieval_metrics,
    known_routing_metrics,
    reject_quality_metrics,
    set_valued_routing_metrics,
)
from .stage_c24b_router import (
    RejectedRoute,
    candidate_set_from_scores,
    execute_selective_route,
    r0_frozen_argmax_route,
    r1_definition_ensemble_scores,
    r3_set_valued_route,
    route_from_candidate_set,
)
from .train import resolve_device


SCHEMA_VERSION = 2
STAGE_NAME = "C2.4b-selective-discrete-relation-router"
FORMAL_SPLITS = tuple(split.value for split in PublicSplitV2)
RELATION_ORDER = tuple(RelationId)
REVIEW_LABELS = frozenset(
    {relation.value for relation in RelationId} | {"ambiguous", "unrelated"}
)
REVIEW_TYPES = frozenset({"known", "ambiguous", "unrelated"})
COMPLETED_REVIEW_STATUSES = frozenset({"complete", "reviewed", "adjudicated"})
STATIC_REVIEW_FIELDS = (
    "row_id",
    "split",
    "phrase_family",
    "phrase",
    "frame",
    "entity",
    "proposed_label",
    "sample_type",
    "ambiguous_flag",
    "unrelated_flag",
)


class ProtocolViolation(RuntimeError):
    """会停止当前 C2.4b 正式流程的协议错误。"""


class ReviewIncompleteError(ProtocolViolation):
    """人工审核尚未完成；这不是可被自动修复或自动填充的状态。"""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if isinstance(output.get(key), Mapping) and isinstance(value, Mapping):
            output[key] = _deep_merge(output[key], value)  # type: ignore[arg-type]
        else:
            output[key] = value
    return output


def _reject_forbidden_config(value: Any, *, location: str = "config") -> None:
    """拒绝私有答案、private memory、confirmation 和攻击输入。

    明确为 ``false`` 的协议审计布尔值允许保留，以便产物证明相关动作未执行。
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            sensitive = any(
                token in key
                for token in (
                    "private_answer",
                    "private_value",
                    "confirmation",
                    "answer_injection",
                    "key_attack",
                    "membership_inference",
                )
            )
            if sensitive and child is not False:
                raise ProtocolViolation(f"forbidden input at {location}.{raw_key}")
            _reject_forbidden_config(child, location=f"{location}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_config(child, location=f"{location}[{index}]")
        return
    if isinstance(value, (str, Path)):
        lowered = str(value).replace("\\", "/").casefold()
        if "confirmation" in lowered:
            raise ProtocolViolation(f"confirmation path/value is forbidden at {location}")


def load_stage_c24b_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """解析一个可选父配置，并锁定 C2.4b 顶层 schema。"""

    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ProtocolViolation("Stage C2.4b config must be a mapping")
    raw = dict(raw)
    parent_name = raw.pop("extends", None)
    source_files = [source]
    if parent_name is not None:
        parent = Path(str(parent_name))
        if not parent.is_absolute():
            local = source.parent / parent
            parent = local if local.exists() else source.parent.parent / parent
        parent = parent.resolve()
        parent_raw = yaml.safe_load(parent.read_text(encoding="utf-8")) or {}
        if not isinstance(parent_raw, Mapping) or "extends" in parent_raw:
            raise ProtocolViolation("C2.4b supports exactly one config parent")
        values = _deep_merge(parent_raw, raw)
        source_files.insert(0, parent)
    else:
        values = raw
    expected = {
        "schema_version",
        "stage",
        "fixed_sources",
        "models",
        "run",
        "gates",
        "benchmark",
    }
    if set(values) != expected:
        raise ProtocolViolation("Stage C2.4b config top-level schema differs")
    if int(values.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ProtocolViolation("unsupported Stage C2.4b schema version")
    if not str(values.get("stage", "")).startswith(STAGE_NAME):
        raise ProtocolViolation("config is not a Stage C2.4b protocol")
    _reject_forbidden_config(values)
    run = values.get("run")
    if not isinstance(run, Mapping) or run.get("mode") not in {"formal", "smoke"}:
        raise ProtocolViolation("run.mode must be formal or smoke")
    for key in (
        "development_used_for_selection",
        "locked_audit_used_for_selection",
        "create_confirmation",
        "train_private_memory",
        "load_private_answers",
    ):
        if run.get(key) is not False:
            raise ProtocolViolation(f"run.{key} must remain false")
    metadata = {
        "source_files": [str(item) for item in source_files],
        "source_sha256": {str(item): sha256_file(item) for item in source_files},
        "resolved_config_sha256": canonical_sha256(values),
    }
    return dict(values), metadata


def canonical_sha256(value: Any) -> str:
    _assert_safe_output_json(value)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_safe_output_json(value: Any, *, location: str = "root") -> None:
    """验证编排产物有限且不携带被禁止的数据。

    metrics 模块对 ``confirmation`` 字段采取最窄白名单；配置与协议状态还
    需要保存若干“固定为 false”的否定性证据。因此编排层允许任意含该词的
    布尔字段，但值必须严格为 ``False``，从而既可审计又不能承载数据。
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            answer_bearing = key in {
                "answer",
                "answers",
                "answer_value",
                "answer_text",
                "candidate_answer",
                "candidate_answers",
                "target_value",
                "ground_truth_value",
            }
            if answer_bearing:
                raise ProtocolViolation(
                    f"answer-bearing field is forbidden at {location}.{raw_key}"
                )
            private_answer_free_metadata = (
                key.endswith("private_answer_free") and child is True
            ) or (key == "contains_private_answers" and child is False)
            if (
                ("private_answer" in key and not private_answer_free_metadata)
                or "private_value" in key
            ):
                if child is not False:
                    raise ProtocolViolation(
                        f"private data field is forbidden at {location}.{raw_key}"
                    )
            readiness_confirmation = (
                key == "ready_to_create_new_confirmation_pool"
                and isinstance(child, bool)
            )
            if "confirmation" in key and child is not False and not readiness_confirmation:
                raise ProtocolViolation(
                    f"confirmation data/status is forbidden at {location}.{raw_key}"
                )
            _assert_safe_output_json(child, location=f"{location}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_safe_output_json(child, location=f"{location}[{index}]")
        return
    if isinstance(value, Tensor):
        raise TypeError(f"tensor is not JSON output at {location}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {location}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise TypeError(f"unsupported JSON value at {location}: {type(value).__name__}")


def _repo_root(config_path: str | Path) -> Path:
    source = Path(config_path).resolve()
    return source.parent.parent


def _resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_repo_root(config_path) / path).resolve()


def _git_state(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=cwd, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True
        )
        return {"commit": commit, "branch": branch, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "branch": "unavailable", "dirty": True}


def _write_json(path: str | Path, value: Any) -> Path:
    _assert_safe_output_json(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def _write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """写入有限 CSV；允许显式 schema 的零行审计结果。"""

    for index, row in enumerate(rows):
        _assert_safe_output_json(dict(row), location=f"csv[{index}]")
    fields = list(rows[0]) if rows else list(fieldnames or ())
    if not fields:
        raise ValueError("empty CSV output requires explicit fieldnames")
    if fieldnames is not None and fields != list(fieldnames):
        raise ValueError("CSV first row differs from explicit fieldnames")
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows have inconsistent columns")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return target


def _parse_jsonl_bytes(raw: bytes, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolViolation(f"JSONL is not UTF-8: {source}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ProtocolViolation(f"JSONL row {line_number} is not an object")
        assert_finite_json(value, location=f"jsonl[{line_number}]")
        rows.append(value)
    if not rows:
        raise ProtocolViolation(f"JSONL file is empty: {source}")
    return rows


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return _parse_jsonl_bytes(source.read_bytes(), source=str(source))


def _parse_review_csv_bytes(raw: bytes, *, source: str) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolViolation(f"review CSV is not UTF-8: {source}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != tuple(REVIEW_FIELDS):
        raise ProtocolViolation(f"review CSV schema mismatch: {source}")
    rows = [dict(row) for row in reader]
    if not rows:
        raise ProtocolViolation(f"review CSV is empty: {source}")
    return rows


def _read_review_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    return _parse_review_csv_bytes(source.read_bytes(), source=str(source))


def _review_label_type_valid(label: str, sample_type: str) -> bool:
    if sample_type == "known":
        return label in {relation.value for relation in RelationId}
    return sample_type in {"ambiguous", "unrelated"} and label == sample_type


def _validate_one_review(
    expected_rows: Sequence[Mapping[str, Any]],
    review_path: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    expected = {str(row["row_id"]): dict(row) for row in expected_rows}
    rows = _read_review_csv(review_path)
    observed = {str(row["row_id"]): row for row in rows}
    if len(observed) != len(rows):
        raise ProtocolViolation(f"duplicate review row_id in {review_path}")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ProtocolViolation(
            f"review row set mismatch in {review_path}: missing={missing[:3]}, extra={extra[:3]}"
        )
    incomplete: list[str] = []
    disagreements = 0
    adjudicated_changes = 0
    ambiguous_to_unrelated = 0
    for row_id, baseline in expected.items():
        row = observed[row_id]
        for field in STATIC_REVIEW_FIELDS:
            if str(row.get(field, "")) != str(baseline.get(field, "")):
                raise ProtocolViolation(
                    f"review static field changed at {row_id}.{field}"
                )
        reviewer_values = (
            row.get("reviewer_1_label", "").strip(),
            row.get("reviewer_2_label", "").strip(),
            row.get("reviewer_1_type", "").strip(),
            row.get("reviewer_2_type", "").strip(),
            row.get("adjudicated_label", "").strip(),
            row.get("adjudicated_type", "").strip(),
        )
        status = row.get("review_status", "").strip().casefold()
        if not all(reviewer_values) or status not in COMPLETED_REVIEW_STATUSES:
            incomplete.append(row_id)
            continue
        r1_label, r2_label, r1_type, r2_type, final_label, final_type = reviewer_values
        if (
            r1_label not in REVIEW_LABELS
            or r2_label not in REVIEW_LABELS
            or final_label not in REVIEW_LABELS
            or r1_type not in REVIEW_TYPES
            or r2_type not in REVIEW_TYPES
            or final_type not in REVIEW_TYPES
        ):
            raise ProtocolViolation(f"unknown review label/type at {row_id}")
        if not _review_label_type_valid(r1_label, r1_type):
            raise ProtocolViolation(f"reviewer 1 label/type mismatch at {row_id}")
        if not _review_label_type_valid(r2_label, r2_type):
            raise ProtocolViolation(f"reviewer 2 label/type mismatch at {row_id}")
        if not _review_label_type_valid(final_label, final_type):
            raise ProtocolViolation(f"adjudicated label/type mismatch at {row_id}")
        agreed = (r1_label, r1_type) == (r2_label, r2_type)
        if agreed and (final_label, final_type) != (r1_label, r1_type):
            raise ProtocolViolation(f"adjudication contradicts reviewer consensus at {row_id}")
        if not agreed:
            disagreements += 1
            if not row.get("notes", "").strip():
                raise ProtocolViolation(f"review disagreement lacks adjudication notes at {row_id}")
        proposed = (
            str(baseline["proposed_label"]),
            str(baseline["sample_type"]),
        )
        if (final_label, final_type) != proposed:
            adjudicated_changes += 1
            if proposed[1] == "ambiguous" and final_type == "unrelated":
                raise ProtocolViolation(
                    f"ambiguous-to-unrelated relabeling is forbidden at {row_id}"
                )
            if not row.get("notes", "").strip():
                raise ProtocolViolation(
                    f"review label/type change lacks adjudication notes at {row_id}"
                )
    complete = not incomplete
    if require_complete and not complete:
        raise ReviewIncompleteError(
            f"independent human review is incomplete for {review_path}; "
            f"{len(incomplete)} rows still lack two reviewers/adjudication"
        )
    return {
        "path": str(review_path),
        "sha256": sha256_file(review_path),
        "row_count": len(rows),
        "complete": complete,
        "incomplete_row_count": len(incomplete),
        "reviewer_disagreement_count": disagreements,
        "adjudicated_label_or_type_change_count": adjudicated_changes,
        "ambiguous_to_unrelated_change_count": ambiguous_to_unrelated,
        "model_predictions_used_for_labels": "not_verified_by_software",
        "prediction_blindness_verified_by_software": False,
        "reviewer_independence_verified_by_software": False,
    }


def _review_sources(
    config_path: str | Path,
    values: Mapping[str, Any],
    *,
    required_reviews: Sequence[str] | None = None,
) -> dict[str, tuple[list[dict[str, Any]], Path]]:
    benchmark = values["benchmark"]
    data_dir = _resolve_path(config_path, benchmark["public_data_dir"])
    artifact_dir = _resolve_path(config_path, benchmark["artifact_dir"])
    available = {
        PublicSplitV2.TRAIN.value,
        PublicSplitV2.CALIBRATION.value,
        PublicSplitV2.LOCKED_AUDIT.value,
    }
    if bool(benchmark.get("generate_legacy_c24_review", True)):
        available.add("legacy_c24_locked_audit")
    requested = available if required_reviews is None else set(required_reviews)
    if not requested.issubset(available):
        raise ProtocolViolation(
            f"unknown or unavailable review scopes: {sorted(requested - available)}"
        )
    sources: dict[str, tuple[list[dict[str, Any]], Path]] = {}
    if PublicSplitV2.TRAIN.value in requested:
        train_rows = _read_jsonl(data_dir / f"{PublicSplitV2.TRAIN.value}.jsonl")
        sources[PublicSplitV2.TRAIN.value] = (
            build_review_rows(
                train_rows, allowed_splits=(PublicSplitV2.TRAIN.value,)
            ),
            artifact_dir / "public_train_v2_review.csv",
        )
    if PublicSplitV2.CALIBRATION.value in requested:
        calibration_rows = _read_jsonl(
            data_dir / f"{PublicSplitV2.CALIBRATION.value}.jsonl"
        )
        sources[PublicSplitV2.CALIBRATION.value] = (
            build_review_rows(calibration_rows),
            artifact_dir / "public_calibration_v2_review.csv",
        )
    if PublicSplitV2.LOCKED_AUDIT.value in requested:
        locked_rows = _read_jsonl(
            data_dir / f"{PublicSplitV2.LOCKED_AUDIT.value}.jsonl"
        )
        sources[PublicSplitV2.LOCKED_AUDIT.value] = (
            build_review_rows(locked_rows),
            artifact_dir / "public_locked_audit_v2_review.csv",
        )
    if "legacy_c24_locked_audit" in requested:
        legacy_source = _resolve_path(
            config_path,
            benchmark.get(
                "legacy_c24_locked_rows", "data/stage_c24/public_locked_audit.jsonl"
            ),
        )
        if legacy_source.suffix.casefold() == ".csv":
            stage_c24_path = _resolve_path(
                config_path,
                benchmark.get("historical_sources", {})["stage_c24_config"],
            )
            stage_c24 = yaml.safe_load(stage_c24_path.read_text(encoding="utf-8"))
            raw_entities = stage_c24["benchmark"]["splits"]["public_locked_audit"][
                "entities"
            ]
            entities = (
                list(raw_entities.values())
                if isinstance(raw_entities, Mapping)
                else list(raw_entities)
            )
            legacy_rows: list[dict[str, Any]] = []
            with legacy_source.open(encoding="utf-8", newline="") as handle:
                for source_row in csv.DictReader(handle):
                    label = str(source_row.get("proposed_label", ""))
                    sample_type = label if label in {"ambiguous", "unrelated"} else "known"
                    prompt = str(source_row.get("frame", ""))
                    matching = [str(entity) for entity in entities if str(entity) in prompt]
                    if len(matching) != 1:
                        raise ProtocolViolation(
                            "cannot recover exactly one entity from legacy C2.4 review"
                        )
                    legacy_rows.append(
                        {
                            "example_id": str(source_row.get("row_id", "")),
                            "split": "public_locked_audit",
                            "kind": sample_type,
                            "attribute": label if sample_type == "known" else None,
                            "family_id": str(source_row.get("phrase_family", "")),
                            "relation_phrase": str(source_row.get("phrase", "")),
                            "prompt": prompt,
                            "entity": matching[0],
                        }
                    )
        else:
            legacy_rows = _read_jsonl(legacy_source)
        sources["legacy_c24_locked_audit"] = (
            build_legacy_c24_review_rows(legacy_rows),
            artifact_dir / "c24_locked_audit_independent_review.csv",
        )
    return sources


def validate_stage_c24b_reviews(
    config_path: str | Path,
    *,
    require_complete: bool = True,
    write_manifest: bool = True,
    required_reviews: Sequence[str] | None = None,
) -> dict[str, Any]:
    """校验双人审核；绝不根据模型预测写入或修复 reviewer 字段。"""

    values, metadata = load_stage_c24b_config(config_path)
    sources = _review_sources(
        config_path, values, required_reviews=required_reviews
    )
    results = {
        name: _validate_one_review(expected, path, require_complete=require_complete)
        for name, (expected, path) in sources.items()
    }
    required_v2 = [
        name
        for name in (
            PublicSplitV2.TRAIN.value,
            PublicSplitV2.CALIBRATION.value,
            PublicSplitV2.LOCKED_AUDIT.value,
        )
        if name in results
    ]
    v2_complete = bool(required_v2) and all(
        results[name]["complete"] for name in required_v2
    )
    all_required_complete = all(item["complete"] for item in results.values())
    artifact_dir = _resolve_path(config_path, values["benchmark"]["artifact_dir"])
    benchmark_manifest_path = artifact_dir / "public_benchmark_manifest.json"
    benchmark_manifest = (
        json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
        if benchmark_manifest_path.is_file()
        else {}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "complete" if all_required_complete else "pending_human_input",
        "public_benchmark_human_reviewed": (
            v2_complete
            and set(required_v2)
            == {
                PublicSplitV2.TRAIN.value,
                PublicSplitV2.CALIBRATION.value,
                PublicSplitV2.LOCKED_AUDIT.value,
            }
        ),
        "legacy_c24_review_complete": results.get(
            "legacy_c24_locked_audit", {"complete": False}
        )["complete"],
        "reviews": results,
        "review_policy": {
            "two_independent_reviewers_required": True,
            "adjudication_required": True,
            "prediction_blind_labels_required": True,
            "software_does_not_claim_human_independence": True,
        },
        "config_sha256": metadata["resolved_config_sha256"],
        "git": _git_state(_repo_root(config_path)),
        "model_revisions": values["models"],
        "data_sha256": benchmark_manifest.get("data_sha256"),
        # 自身的 SHA 由调用方在写入后封存，不能递归写进自身。
        "review_manifest_sha256": None,
        "selection_protocol": benchmark_manifest.get(
            "selection_protocol",
            {
                "selection_split": PublicSplitV2.CALIBRATION.value,
                "development_used_for_selection": False,
                "locked_audit_used_for_selection": False,
            },
        ),
    }
    if write_manifest:
        _write_json(artifact_dir / "review_manifest.json", manifest)
    if require_complete and not all_required_complete:
        raise ReviewIncompleteError("C2.4b required human reviews are incomplete")
    return manifest


def _preflight_review_gate(review_paths: Sequence[Path]) -> dict[str, Any]:
    """只读取审核 CSV 完成 first gate，不接触任何 benchmark split。

    正式 audit 必须先通过此门，随后才允许打开 locked JSONL 逐字段核验。
    该门只验证完整性与标签/类型约束，不声称软件可以验证审核者身份。
    """

    results: dict[str, Any] = {}
    for path in review_paths:
        prepared_manifest_path = path.parent / "public_benchmark_manifest.json"
        prepared_manifest = json.loads(
            prepared_manifest_path.read_text(encoding="utf-8")
        )
        review_files = prepared_manifest.get("review_files")
        if not isinstance(review_files, Mapping):
            raise ProtocolViolation("prepared manifest lacks sealed review row sets")
        review_name = {
            "public_train_v2_review.csv": PublicSplitV2.TRAIN.value,
            "public_calibration_v2_review.csv": PublicSplitV2.CALIBRATION.value,
            "public_locked_audit_v2_review.csv": PublicSplitV2.LOCKED_AUDIT.value,
            "c24_locked_audit_independent_review.csv": "legacy_c24_locked_audit",
        }.get(path.name)
        if review_name is None or not isinstance(review_files.get(review_name), Mapping):
            raise ProtocolViolation(f"prepared review seal is missing for {path.name}")
        prepared_review = review_files[review_name]
        rows = _read_review_csv(path)
        row_ids = [str(row["row_id"]) for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ProtocolViolation(f"duplicate review row_id in {path}")
        expected_count = int(prepared_review.get("row_count", -1))
        expected_ids_sha = str(prepared_review.get("row_id_sha256", ""))
        expected_static_sha = str(
            prepared_review.get("static_review_sha256", "")
        )
        observed_ids_sha = sha256_json(sorted(row_ids))
        observed_static_sha = review_static_sha256(rows)
        if (
            len(rows) != expected_count
            or observed_ids_sha != expected_ids_sha
            or observed_static_sha != expected_static_sha
        ):
            raise ProtocolViolation(
                f"review row set differs from prepared seal before data open: {path}"
            )
        incomplete: list[str] = []
        disagreements = 0
        adjudicated_changes = 0
        ambiguous_to_unrelated = 0
        for row in rows:
            row_id = str(row["row_id"])
            values = tuple(
                str(row[name]).strip()
                for name in (
                    "reviewer_1_label",
                    "reviewer_2_label",
                    "reviewer_1_type",
                    "reviewer_2_type",
                    "adjudicated_label",
                    "adjudicated_type",
                )
            )
            if (
                not all(values)
                or str(row["review_status"]).strip().casefold()
                not in COMPLETED_REVIEW_STATUSES
            ):
                incomplete.append(row_id)
                continue
            r1_label, r2_label, r1_type, r2_type, final_label, final_type = values
            for label, sample_type in (
                (r1_label, r1_type),
                (r2_label, r2_type),
                (final_label, final_type),
            ):
                if label not in REVIEW_LABELS or sample_type not in REVIEW_TYPES:
                    raise ProtocolViolation(f"unknown review label/type at {row_id}")
                if not _review_label_type_valid(label, sample_type):
                    raise ProtocolViolation(f"review label/type mismatch at {row_id}")
            agreed = (r1_label, r1_type) == (r2_label, r2_type)
            if agreed and (final_label, final_type) != (r1_label, r1_type):
                raise ProtocolViolation(
                    f"adjudication contradicts reviewer consensus at {row_id}"
                )
            if not agreed:
                disagreements += 1
                if not str(row["notes"]).strip():
                    raise ProtocolViolation(
                        f"review disagreement lacks adjudication notes at {row_id}"
                    )
            proposed = (
                str(row["proposed_label"]).strip(),
                str(row["sample_type"]).strip(),
            )
            if (final_label, final_type) != proposed:
                adjudicated_changes += 1
                if proposed[1] == "ambiguous" and final_type == "unrelated":
                    raise ProtocolViolation(
                        f"ambiguous-to-unrelated relabeling is forbidden at {row_id}"
                    )
                if not str(row["notes"]).strip():
                    raise ProtocolViolation(
                        f"review label/type change lacks adjudication notes at {row_id}"
                    )
        if incomplete:
            raise ReviewIncompleteError(
                f"independent human review is incomplete for {path}; "
                f"{len(incomplete)} rows remain pending"
            )
        results[path.name] = {
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "row_id_sha256": observed_ids_sha,
            "static_review_sha256": observed_static_sha,
            "reviewer_disagreement_count": disagreements,
            "adjudicated_label_or_type_change_count": adjudicated_changes,
            "ambiguous_to_unrelated_change_count": ambiguous_to_unrelated,
            "complete": True,
        }
    return results


def _review_paths_for_scope(
    config_path: str | Path,
    values: Mapping[str, Any],
    *,
    include_locked_v2: bool,
) -> list[Path]:
    benchmark = values["benchmark"]
    artifact_dir = _resolve_path(config_path, benchmark["artifact_dir"])
    paths = [
        artifact_dir / "public_train_v2_review.csv",
        artifact_dir / "public_calibration_v2_review.csv",
    ]
    if bool(benchmark.get("generate_legacy_c24_review", True)):
        paths.append(artifact_dir / "c24_locked_audit_independent_review.csv")
    if include_locked_v2:
        paths.append(artifact_dir / "public_locked_audit_v2_review.csv")
    return paths


def _verify_split_seal(
    config_path: str | Path,
    values: Mapping[str, Any],
    split: PublicSplitV2,
) -> dict[str, Any]:
    """核对 prepare 时的数据 manifest，任何改动都 fail closed。"""

    artifact_dir = _resolve_path(config_path, values["benchmark"]["artifact_dir"])
    data_dir = _resolve_path(config_path, values["benchmark"]["public_data_dir"])
    manifest_path = artifact_dir / f"{split.value}_manifest.json"
    aggregate_path = artifact_dir / "public_benchmark_manifest.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_seals = aggregate.get("split_manifests")
    if not isinstance(aggregate_seals, Mapping) or not isinstance(
        aggregate_seals.get(split.value), Mapping
    ):
        raise ProtocolViolation("aggregate benchmark manifest lacks split seal")
    aggregate_split = aggregate_seals[split.value]
    if sha256_file(manifest_path) != str(aggregate_split.get("sha256", "")):
        raise ProtocolViolation(
            f"{split.value} manifest differs from aggregate prepared seal"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("split") != split.value:
        raise ProtocolViolation(f"invalid prepared split manifest: {manifest_path}")
    data_file = manifest.get("data_file")
    if not isinstance(data_file, Mapping):
        raise ProtocolViolation(f"split manifest lacks data seal: {manifest_path}")
    data_path = data_dir / f"{split.value}.jsonl"
    expected = str(data_file.get("sha256", ""))
    observed = sha256_file(data_path)
    if observed != expected:
        raise ProtocolViolation(
            f"prepared {split.value} data SHA-256 changed; current audit is invalid"
        )
    return {
        "split": split.value,
        "data_path": str(data_path),
        "data_sha256": observed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": int(manifest["row_count"]),
    }


def _verify_review_manifest_current(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise ProtocolViolation("frozen review manifest is absent or incomplete")
    reviews = manifest.get("reviews")
    if not isinstance(reviews, Mapping) or not reviews:
        raise ProtocolViolation("frozen review manifest lacks review seals")
    for name, raw in reviews.items():
        if not isinstance(raw, Mapping) or raw.get("complete") is not True:
            raise ProtocolViolation(f"review seal is incomplete: {name}")
        current = sha256_file(Path(str(raw["path"])))
        if current != raw.get("sha256"):
            raise ProtocolViolation(f"review CSV changed after validation: {name}")
    return dict(manifest)


def _verify_declared_fixed_sources(
    config_path: str | Path,
    values: Mapping[str, Any],
    *,
    lightweight_only: bool,
) -> dict[str, str]:
    """核对已声明的本地 source；缺文件或 hash 漂移均不降级。"""

    fixed = values["fixed_sources"]
    lightweight = {"stage_c23_config", "stage_c24_config", "protocol_incident"}
    observed: dict[str, str] = {}
    for key, raw_path in fixed.items():
        if key.endswith("_sha256") or key == "source_core_sha256":
            continue
        if lightweight_only and key not in lightweight:
            continue
        expected_key = f"{key}_sha256"
        if expected_key not in fixed:
            continue
        path = _resolve_path(config_path, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"fixed Stage C2.4b source is absent: {path}")
        digest = sha256_file(path)
        if digest != str(fixed[expected_key]):
            raise ProtocolViolation(
                f"fixed source SHA-256 mismatch for {key}: {digest}"
            )
        observed[key] = digest
    return observed


def _verify_answer_free_cache_binding(
    config_path: str | Path, values: Mapping[str, Any]
) -> dict[str, Any]:
    cache_path = _resolve_path(
        config_path, values["fixed_sources"]["private_feature_cache"]
    )
    payload = _load_feature_cache(cache_path)
    counts = validate_answer_free_private_feature_cache(payload)
    observed_core = str(payload.get("source_core_sha256", ""))
    expected_core = str(values["fixed_sources"]["source_core_sha256"])
    if observed_core != expected_core:
        raise ProtocolViolation("answer-free cache source_core SHA-256 mismatch")
    del payload
    return {
        "cache_sha256": sha256_file(cache_path),
        "source_core_sha256": observed_core,
        "row_counts": counts,
        "private_answers_loaded": False,
    }


def _apply_adjudicated_reviews(
    rows: Sequence[Mapping[str, Any]],
    review_path: Path,
    *,
    train_split: bool,
) -> list[dict[str, Any]]:
    """在副本上应用人工 adjudication；永不回写 benchmark 或 review CSV。"""

    return _apply_adjudicated_review_rows(
        rows,
        _read_review_csv(review_path),
        train_split=train_split,
    )


def _apply_adjudicated_review_rows(
    rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, str]],
    *,
    train_split: bool,
) -> list[dict[str, Any]]:
    """将已从同一 sealed 字节解析的审核行应用到 benchmark 副本。"""

    reviews = {str(row["row_id"]): row for row in review_rows}
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        review = reviews.get(str(row["row_id"]))
        if review is None:
            raise ProtocolViolation(f"review row is missing for {row['row_id']}")
        final_type = str(review["adjudicated_type"]).strip()
        final_label = str(review["adjudicated_label"]).strip()
        if train_split and final_type != "known":
            raise ProtocolViolation("public_train_v2 adjudication must remain known")
        row["sample_type"] = final_type
        row["kind"] = final_type
        if final_type == "known":
            relation = RelationId(final_label)
            row["relation_id"] = relation.value
            row["attribute"] = relation.value
            row["fact_id"] = f"public-v2:{row['entity_id']}|{relation.value}"
        else:
            if final_label != final_type:
                raise ProtocolViolation("reject adjudication label/type mismatch")
            row["relation_id"] = None
            row["attribute"] = None
        output.append(row)
    return output


def _protocol_status(
    *,
    public_benchmark_human_reviewed: bool,
    formal_locked_audit_executed: bool,
    calibration_executed: bool,
    development_executed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "lm_answer_injection_executed": False,
        "bounded_orthogonal_injection_executed": False,
        "key_or_attack_experiments_run": False,
        "confirmation_created_or_read": False,
        "old_c24_locked_audit_used_for_selection": False,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "public_benchmark_human_reviewed": public_benchmark_human_reviewed,
        "calibration_executed": calibration_executed,
        "development_executed_once": development_executed,
        "formal_locked_audit_executed": formal_locked_audit_executed,
        "new_confirmation_pool_created_after_freeze": False,
        "c3_eligible": False,
    }


def prepare_stage_c24b(config_path: str | Path) -> dict[str, Any]:
    """生成全新的 v2 数据、碰撞审计和空白人工审核模板。"""

    values, metadata = load_stage_c24b_config(config_path)
    if values["run"]["mode"] != "formal":
        raise ProtocolViolation("stage-c24b-prepare requires the formal config")
    fixed_source_hashes = _verify_declared_fixed_sources(
        config_path, values, lightweight_only=True
    )
    artifact_dir = _resolve_path(config_path, values["benchmark"]["artifact_dir"])
    aggregate_path = artifact_dir / "public_benchmark_manifest.json"
    if aggregate_path.exists():
        raise FileExistsError("prepared C2.4b benchmark cannot be overwritten")
    manifest = prepare_public_benchmark_v2(config_path)
    _write_json(artifact_dir / "resolved_config.json", values)
    review = validate_stage_c24b_reviews(
        config_path, require_complete=False, write_manifest=True
    )
    review_manifest_path = artifact_dir / "review_manifest.json"
    review_manifest_sha256 = sha256_file(review_manifest_path)
    selection_protocol = manifest["selection_protocol"]
    provenance = {
        "git": manifest["provenance"]["git"],
        "model_revisions": values["models"],
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "data_sha256": manifest["data_sha256"],
        "review_manifest_sha256": review_manifest_sha256,
        "selection_protocol": selection_protocol,
    }
    status = _protocol_status(
        public_benchmark_human_reviewed=False,
        formal_locked_audit_executed=False,
        calibration_executed=False,
        development_executed=False,
    )
    status.update(
        {
            "status": "prepared_waiting_for_independent_human_review",
            "resolved_config_sha256": metadata["resolved_config_sha256"],
            "legacy_c24_negative_result_unchanged": True,
            "legacy_c24_readiness_changed_by_review": False,
            "fixed_source_hashes_verified_before_prepare": fixed_source_hashes,
            **provenance,
        }
    )
    _write_json(artifact_dir / "protocol_status.json", status)
    pending_readiness = compute_readiness(
        closed_set_selective_router_ready=False,
        open_set_abstention_ready=False,
        discrete_memory_contract_preserved=True,
        public_benchmark_human_reviewed=False,
    )
    _write_json(
        artifact_dir / "stage_c24b_summary.json",
        {
            **status,
            "status": "prepared_waiting_for_independent_human_review",
            "formal_research_thresholds_assessed": False,
            "readiness": pending_readiness,
            "ready_to_create_new_confirmation_pool": False,
            "discrete_memory_contract_preserved": True,
            "public_benchmark_human_reviewed": False,
            "formal_locked_audit_executed": False,
            "new_confirmation_pool_created_after_freeze": False,
            "c3_eligible": False,
            **provenance,
        },
    )
    files = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256_manifest.json":
            files.append(
                {
                    "path": path.relative_to(artifact_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": "prepared_pending_human_review_snapshot",
        **provenance,
        "formal_locked_audit_executed": False,
        "public_benchmark_human_reviewed": False,
        "file_count": len(files),
        "files": files,
    }
    artifact_manifest["manifest_payload_sha256"] = canonical_sha256(
        artifact_manifest
    )
    _write_json(artifact_dir / "artifact_sha256_manifest.json", artifact_manifest)
    return {
        "stage": STAGE_NAME,
        "status": status["status"],
        "answer_free": True,
        "row_counts": manifest["row_counts"],
        "review_status": review["status"],
        "public_benchmark_human_reviewed": False,
        "formal_locked_audit_executed": False,
        "artifact_dir": str(artifact_dir),
    }


def _synthetic_definition_embeddings(width: int = 8) -> dict[RelationId, Tensor]:
    """构造完全本地、answer-free 的 smoke 定义原型。"""

    definitions: dict[RelationId, Tensor] = {}
    for relation_index, relation in enumerate(RelationId):
        rows = []
        for variant in range(3):
            value = torch.zeros(width, dtype=torch.float32)
            value[relation_index] = 1.0
            value[4 + variant] = 0.025 * (variant + 1)
            rows.append(F.normalize(value, dim=0))
        definitions[relation] = torch.stack(rows)
    return definitions


def _synthetic_query_embedding(row: Mapping[str, Any], *, width: int = 8) -> Tensor:
    """只根据 synthetic fixture 的公开类型构造可重复 smoke embedding。"""

    value = torch.zeros(width, dtype=torch.float32)
    sample_type = str(row["sample_type"])
    if sample_type == "known":
        relation = RelationId(str(row["relation_id"]))
        value[list(RelationId).index(relation)] = 1.0
    elif sample_type == "ambiguous":
        value[list(RelationId).index(RelationId.REGISTRY_ID)] = 1.0
        value[list(RelationId).index(RelationId.ACCESS_CODE)] = 1.0
    elif sample_type == "unrelated":
        value[3] = 1.0
    else:
        raise ProtocolViolation(f"unknown synthetic sample type {sample_type!r}")
    # 公开 row_id 的摘要只产生极小正交扰动，避免依赖文本词面或答案值。
    digest = hashlib.sha256(str(row["row_id"]).encode("utf-8")).digest()
    for offset in range(4):
        value[4 + offset] = (digest[offset] / 255.0 - 0.5) * 0.006
    return F.normalize(value, dim=0)


def _score_rows_r1(
    rows: Sequence[Mapping[str, Any]],
    definitions: Mapping[RelationId, Tensor],
    *,
    aggregation: str,
    top_k: int | None = None,
) -> list[dict[RelationId, float]]:
    return [
        r1_definition_ensemble_scores(
            _synthetic_query_embedding(row),
            definitions,
            aggregation=aggregation,
            top_k=top_k,
        )
        for row in rows
    ]


def _argmax_route_from_scores(scores: Mapping[RelationId, float]) -> AcceptedRoute:
    relation = max(RelationId, key=lambda item: (scores[item], -list(RelationId).index(item)))
    return AcceptedRoute(relation)


def _candidate_reject_score(
    scores: Mapping[RelationId, float], candidates: frozenset[RelationId]
) -> float:
    ordered = sorted((float(value) for value in scores.values()), reverse=True)
    if not candidates:
        return 2.0 - ordered[0]
    if len(candidates) > 1:
        return 1.0 + ordered[1]
    return 1.0 - ordered[0]


def _evaluate_variant(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[RelationId, float]],
    routes: Sequence[AcceptedRoute | RejectedRoute],
    candidate_sets: Sequence[frozenset[RelationId]],
    *,
    variant: str,
    split_name: str,
    config_sha256: str,
    git: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    protocol_role: str = "synthetic_integration_only",
    evaluation_scope: str = "synthetic_smoke_only_not_formal_evidence",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(rows) == len(scores) == len(routes) == len(candidate_sets)):
        raise ProtocolViolation("smoke evaluation inputs are not aligned")
    entities = sorted({str(row["entity"]) for row in rows})
    entity_index = {entity: index for index, entity in enumerate(entities)}
    entity_vectors = torch.eye(len(entities), dtype=torch.float32)
    buckets = {relation: entity_vectors.clone() for relation in RelationId}

    accepted: list[bool] = []
    predictions: list[RelationId | None] = []
    reasons: list[str | None] = []
    retrieval_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fact_hits: list[bool | None] = []
    row_hits: list[bool | None] = []
    reciprocal_ranks: list[float | None] = []
    margins: list[float | None] = []
    candidate_counts: list[int | None] = []
    cross_counts: list[int | None] = []
    reject_scores: list[float] = []

    for row, evidence, route, candidates in zip(rows, scores, routes, candidate_sets):
        index = entity_index[str(row["entity"])]
        execution = execute_selective_route(entity_vectors[index], route, buckets)
        is_accepted = isinstance(execution.route, AcceptedRoute)
        prediction = execution.route.relation_id if is_accepted else None
        reason = None if is_accepted else execution.route.reason
        accepted.append(is_accepted)
        predictions.append(prediction)
        reasons.append(reason)
        reject_scores.append(_candidate_reject_score(evidence, candidates))
        retrieval = execution.retrieval
        retrieval_rows.append(
            {
                "variant": variant,
                "split": split_name,
                "row_id": str(row["row_id"]),
                "sample_type": str(row["sample_type"]),
                "route_status": execution.route.status,
                "relation_id": prediction.value if prediction is not None else None,
                "reject_reason": reason,
                "memory_access_count": execution.memory_access_count,
                "candidate_index": retrieval.candidate_index if retrieval else None,
                "relation_bucket_candidate_count": retrieval.candidate_count
                if retrieval
                else None,
                "cross_relation_candidate_count": retrieval.cross_relation_candidate_count
                if retrieval
                else None,
                "similarity": retrieval.similarity if retrieval else None,
            }
        )
        prediction_rows.append(
            {
                "variant": variant,
                "split": split_name,
                "row_id": str(row["row_id"]),
                "phrase_family": str(row["phrase_family"]),
                "sample_type": str(row["sample_type"]),
                "target_relation": str(row["relation_id"] or ""),
                "route_status": execution.route.status,
                "predicted_relation": prediction.value if prediction else "",
                "reject_reason": reason or "",
                "candidate_set": "|".join(sorted(item.value for item in candidates)),
                "memory_access_count": execution.memory_access_count,
            }
        )
        accepted_known = is_accepted and row["sample_type"] == "known"
        if accepted_known:
            assert retrieval is not None
            target = RelationId(str(row["relation_id"]))
            correct = prediction is target and retrieval.candidate_index == index
            fact_hits.append(correct)
            row_hits.append(correct)
            # 路由到错误 bucket 时目标事实不在候选集合，reciprocal rank 为 0。
            reciprocal_ranks.append(1.0 if correct else 0.0)
            margins.append(1.0)
            candidate_counts.append(retrieval.candidate_count)
            cross_counts.append(retrieval.cross_relation_candidate_count)
        else:
            fact_hits.append(None)
            row_hits.append(None)
            reciprocal_ranks.append(None)
            margins.append(None)
            candidate_counts.append(None)
            cross_counts.append(None)

    sample_types = [str(row["sample_type"]) for row in rows]
    targets = [
        RelationId(str(row["relation_id"])) if row["relation_id"] else None
        for row in rows
    ]
    families = [str(row["phrase_family"]) for row in rows]
    known_indices = [index for index, kind in enumerate(sample_types) if kind == "known"]
    known = known_routing_metrics(
        [predictions[index] for index in known_indices],
        [targets[index] for index in known_indices],
        [accepted[index] for index in known_indices],
        [families[index] for index in known_indices],
        [str(rows[index]["frame_id"]) for index in known_indices],
        hard_negative_mask=[bool(rows[index]["hard_negative"]) for index in known_indices],
    )
    sets = set_valued_routing_metrics(candidate_sets, targets, sample_types)
    reject = reject_quality_metrics(
        accepted,
        reasons,
        predictions,
        targets,
        sample_types,
        families,
        reject_scores,
    )
    access = access_control_metrics(accepted, predictions, targets, sample_types)
    retrieval_metrics = fact_retrieval_metrics(
        accepted,
        sample_types,
        fact_hits,
        row_hits,
        reciprocal_ranks,
        margins,
        candidate_counts,
        cross_counts,
        oracle_fact_top1=1.0,
    )
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "variant": variant,
        "evaluation_scope": evaluation_scope,
        "protocol_role": protocol_role,
        "split": split_name,
        "answer_free": True,
        "contains_private_answers": False,
        "git": dict(git),
        "resolved_config_sha256": config_sha256,
        "data_sha256": canonical_sha256(list(rows)),
        "review_manifest_sha256": None,
        "selection_protocol": {
            "selection_split": PublicSplitV2.CALIBRATION.value,
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
            "synthetic_smoke_only": True,
        },
        "model_metadata": dict(model_metadata),
        "known_routing": known,
        "set_valued_routing": sets,
        "reject_quality": reject,
        "risk_coverage_interpretation": (
            "diagnostic_only_fixed_decisions; threshold-grid tradeoff is a separate artifact"
        ),
        "access_control": access,
        "fact_retrieval": retrieval_metrics,
        "fact_retrieval_scope": "answer_free_public_entity_identity_prototypes",
        "memory_contract": {
            "typed_relation_id_only": True,
            "single_bucket_per_accepted_route": True,
            "rejected_route_memory_access_count": 0,
            "cross_relation_candidate_count": retrieval_metrics[
                "cross_relation_candidate_count"
            ],
            "continuous_relation_reaches_memory": False,
            "confidence_reaches_memory": False,
            "semantic_embedding_reaches_memory": False,
            "raw_text_reaches_memory": False,
            "phrase_family_reaches_memory": False,
        },
    }
    _assert_safe_output_json(evaluation)
    return evaluation, prediction_rows, retrieval_rows


def _routes_argmax(
    score_rows: Sequence[Mapping[RelationId, float]],
) -> tuple[list[AcceptedRoute], list[frozenset[RelationId]]]:
    routes = [_argmax_route_from_scores(scores) for scores in score_rows]
    sets = [frozenset({route.relation_id}) for route in routes]
    return routes, sets


def _r3_candidates_with_margin(
    scores: Mapping[RelationId, float],
    threshold: float,
    minimum_margin: float,
) -> frozenset[RelationId]:
    """将 margin 触发的不可唯一决定显式表现为多候选集合。"""

    candidates = candidate_set_from_scores(scores, threshold)
    ordered = sorted(RelationId, key=lambda relation: scores[relation], reverse=True)
    if (
        len(candidates) == 1
        and float(scores[ordered[0]]) - float(scores[ordered[1]]) < minimum_margin
    ):
        return frozenset({ordered[0], ordered[1]})
    return candidates


def _selection_metrics(evaluation: Mapping[str, Any]) -> dict[str, float]:
    known = evaluation["known_routing"]
    access = evaluation["access_control"]
    sets = evaluation["set_valued_routing"]
    reject = evaluation["reject_quality"]
    return {
        "known_coverage": float(known["known_coverage"]),
        "accepted_relation_precision": float(known["accepted_relation_precision"]),
        "singleton_acceptance_precision": float(
            access["singleton_acceptance_precision"]
        ),
        "safe_coverage": float(access["safe_coverage"]),
        "false_memory_access_rate": float(access["false_memory_access_rate"]),
        "ambiguous_rejection_rate": float(reject["ambiguous_rejection_rate"]),
        "unrelated_rejection_rate": float(reject["unrelated_rejection_rate"]),
        "ambiguous_false_memory_access_rate": float(
            access["ambiguous_false_memory_access_rate"]
        ),
        "unrelated_false_memory_access_rate": float(
            access["unrelated_false_memory_access_rate"]
        ),
        "wrong_bucket_access_rate": float(access["wrong_bucket_access_rate"]),
        "worst_reject_family_false_accept_rate": float(
            reject["worst_reject_family_false_accept_rate"]
        ),
        "family_macro_accuracy": float(known["relation_family_macro_accuracy"]),
        "worst_family_accuracy": float(known["worst_family_accuracy"]),
        "true_relation_inclusion_rate": float(
            sets["true_relation_inclusion_rate"]
        ),
        "average_candidate_set_size": float(sets["average_candidate_set_size"]),
        "known_multi_relation_set_rate": float(
            sets["known_multi_relation_set_rate"]
        ),
        "known_empty_set_rate": float(sets["known_empty_set_rate"]),
    }


def _evaluate_relation_only(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[RelationId, float]],
    routes: Sequence[AcceptedRoute | RejectedRoute],
    candidate_sets: Sequence[frozenset[RelationId]],
    *,
    variant: str,
    split_name: str,
    config_sha256: str,
    git: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    protocol_role: str,
    evaluation_scope: str,
    data_sha256: Mapping[str, Any] | str | None = None,
    review_manifest_sha256: str | None = None,
    selection_protocol: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """正式 relation router 指标；不构造 entity embedding 或伪 fact retrieval。"""

    if not (len(rows) == len(scores) == len(routes) == len(candidate_sets)):
        raise ProtocolViolation("formal relation evaluation inputs are not aligned")
    accepted = [isinstance(route, AcceptedRoute) for route in routes]
    predictions = [
        route.relation_id if isinstance(route, AcceptedRoute) else None
        for route in routes
    ]
    reasons = [
        None if isinstance(route, AcceptedRoute) else route.reason for route in routes
    ]
    sample_types = [str(row["sample_type"]) for row in rows]
    targets = [
        RelationId(str(row["relation_id"])) if row["relation_id"] else None
        for row in rows
    ]
    families = [str(row["phrase_family"]) for row in rows]
    reject_scores = [
        _candidate_reject_score(score, candidates)
        for score, candidates in zip(scores, candidate_sets)
    ]
    known_indices = [index for index, kind in enumerate(sample_types) if kind == "known"]
    known = known_routing_metrics(
        [predictions[index] for index in known_indices],
        [targets[index] for index in known_indices],
        [accepted[index] for index in known_indices],
        [families[index] for index in known_indices],
        [str(rows[index]["frame_id"]) for index in known_indices],
        hard_negative_mask=[bool(rows[index]["hard_negative"]) for index in known_indices],
    )
    sets = set_valued_routing_metrics(candidate_sets, targets, sample_types)
    reject = reject_quality_metrics(
        accepted,
        reasons,
        predictions,
        targets,
        sample_types,
        families,
        reject_scores,
    )
    access = access_control_metrics(accepted, predictions, targets, sample_types)
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "variant": variant,
        "split": split_name,
        "protocol_role": protocol_role,
        "evaluation_scope": evaluation_scope,
        "answer_free": True,
        "contains_private_answers": False,
        "git": dict(git),
        "resolved_config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "selection_protocol": dict(
            selection_protocol
            or {
                "selection_split": PublicSplitV2.CALIBRATION.value,
                "development_used_for_selection": False,
                "locked_audit_used_for_selection": False,
            }
        ),
        "model_metadata": dict(model_metadata),
        "known_routing": known,
        "set_valued_routing": sets,
        "reject_quality": reject,
        "access_control": access,
        "fact_retrieval": {
            "status": "not_computed_by_relation_only_evaluator",
            "synthetic_or_oracle_fact_metric_used": False,
        },
        "risk_coverage_interpretation": (
            "diagnostic_only_fixed_decisions; threshold-grid tradeoff is separate"
        ),
    }
    predictions_rows = [
        {
            "variant": variant,
            "split": split_name,
            "row_id": str(row["row_id"]),
            "phrase_family": str(row["phrase_family"]),
            "sample_type": str(row["sample_type"]),
            "target_relation": str(row["relation_id"] or ""),
            "route_status": route.status,
            "predicted_relation": prediction.value if prediction else "",
            "reject_reason": reason or "",
            "candidate_set": "|".join(sorted(value.value for value in candidates)),
            "would_access_memory": isinstance(route, AcceptedRoute),
            "memory_access_executed": False,
        }
        for row, route, prediction, reason, candidates in zip(
            rows, routes, predictions, reasons, candidate_sets
        )
    ]
    _assert_safe_output_json(evaluation)
    return evaluation, predictions_rows


def _run_smoke_audit(
    config_path: str | Path,
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """运行 synthetic-only R0--R4 集成；不读取任何正式 split 或审核表。"""

    assert_discrete_memory_contract()
    # 分阶段入口保证此处绝不顺带 materialize synthetic locked split。
    train_rows = build_public_split_v2(
        values["benchmark"], PublicSplitV2.TRAIN
    )
    calibration_rows = build_public_split_v2(
        values["benchmark"], PublicSplitV2.CALIBRATION
    )
    if any(
        "smoke" not in str(row["row_id"]).casefold()
        for row in train_rows + calibration_rows
    ):
        raise ProtocolViolation("smoke audit received a non-synthetic fixture row")

    output_dir.mkdir(parents=True, exist_ok=True)
    git = _git_state(_repo_root(config_path))
    config_sha = str(metadata["resolved_config_sha256"])
    model_metadata = {
        "mode": "synthetic_no_external_model_load",
        "configured_candidates": values["models"],
        "model_revision": "smoke-v1",
        "model_file_sha256": "synthetic-fixture-no-file",
    }
    definitions = _synthetic_definition_embeddings()

    # R1 聚合仅在 synthetic calibration 上选择；locked rows 尚未评分。
    aggregation_candidates: list[dict[str, Any]] = []
    aggregation_scores: dict[str, list[dict[RelationId, float]]] = {}
    aggregation_settings: list[tuple[str, int | None]] = []
    for raw_aggregation in values["run"]["evidence_aggregations"]:
        aggregation = str(raw_aggregation)
        if aggregation == "top_k_mean":
            top_k_values = values["models"].get("R1", {}).get("top_k_values", [2])
            aggregation_settings.extend((aggregation, int(value)) for value in top_k_values)
        else:
            aggregation_settings.append((aggregation, None))
    for aggregation, top_k in aggregation_settings:
        candidate_name = aggregation if top_k is None else f"{aggregation}_{top_k}"
        scores = _score_rows_r1(
            calibration_rows,
            definitions,
            aggregation=aggregation,
            top_k=top_k,
        )
        aggregation_scores[candidate_name] = scores
        routes, sets = _routes_argmax(scores)
        evaluation, _, _ = _evaluate_variant(
            calibration_rows,
            scores,
            routes,
            sets,
            variant=f"R1-{candidate_name}",
            split_name=PublicSplitV2.CALIBRATION.value,
            config_sha256=config_sha,
            git=git,
            model_metadata=model_metadata,
        )
        aggregation_candidates.append(
            {
                "name": candidate_name,
                "aggregation": aggregation,
                "top_k": top_k,
                **_selection_metrics(evaluation),
            }
        )
    aggregation_preference = {
        str(row["name"]): len(aggregation_candidates) - index
        for index, row in enumerate(aggregation_candidates)
    }
    selected_aggregation_row = max(
        aggregation_candidates,
        key=lambda row: (
            row["family_macro_accuracy"],
            row["worst_family_accuracy"],
            aggregation_preference[str(row["name"])],
        ),
    )
    selected_aggregation = str(selected_aggregation_row["name"])
    calibration_r1_scores = aggregation_scores[selected_aggregation]

    # R2 的三个一对其余 head 只拟合 public_train_v2 synthetic rows。
    train_embeddings = torch.stack(
        [_synthetic_query_embedding(row) for row in train_rows]
    )
    train_features = build_pairwise_relation_features(train_embeddings, definitions)
    train_targets = [RelationId(str(row["relation_id"])) for row in train_rows]

    def pair_scores(
        head: PairwiseRidgeEvidenceHead,
        rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[RelationId, float]]:
        embeddings = torch.stack([_synthetic_query_embedding(row) for row in rows])
        matrices = head.score_batch(
            build_pairwise_relation_features(embeddings, definitions)
        )
        return [
            {relation: float(matrices[relation][index]) for relation in RelationId}
            for index in range(len(rows))
        ]

    ridge_candidates: list[dict[str, Any]] = []
    ridge_heads: dict[float, PairwiseRidgeEvidenceHead] = {}
    ridge_score_rows: dict[float, list[dict[RelationId, float]]] = {}
    for raw_strength in values["run"]["ridge_strength_grid"]:
        strength = float(raw_strength)
        head = fit_pairwise_ridge_head(
            train_features, train_targets, ridge_strength=strength
        )
        ridge_heads[strength] = head
        score_rows = pair_scores(head, calibration_rows)
        ridge_score_rows[strength] = score_rows
        routes, sets = _routes_argmax(score_rows)
        evaluation, _, _ = _evaluate_variant(
            calibration_rows,
            score_rows,
            routes,
            sets,
            variant=f"R2-ridge-{strength:g}",
            split_name=PublicSplitV2.CALIBRATION.value,
            config_sha256=config_sha,
            git=git,
            model_metadata=model_metadata,
        )
        ridge_candidates.append(
            {"ridge_strength": strength, **_selection_metrics(evaluation)}
        )
    selected_ridge_row = max(
        ridge_candidates,
        key=lambda row: (
            row["family_macro_accuracy"],
            row["worst_family_accuracy"],
            row["accepted_relation_precision"],
            -float(row["ridge_strength"]),
        ),
    )
    selected_ridge_strength = float(selected_ridge_row["ridge_strength"])
    pair_head = ridge_heads[selected_ridge_strength]
    calibration_r2_scores = ridge_score_rows[selected_ridge_strength]

    # R3 threshold 只看 calibration；优先保留 ambiguous 的多候选语义。
    threshold_candidates: list[dict[str, Any]] = []
    threshold_evaluations: dict[tuple[float, float], dict[str, Any]] = {}
    for raw_threshold in values["run"]["threshold_grid"]:
        for raw_margin in values["run"]["minimum_margin_grid"]:
            threshold = float(raw_threshold)
            minimum_margin = float(raw_margin)
            sets = [
                _r3_candidates_with_margin(scores, threshold, minimum_margin)
                for scores in calibration_r1_scores
            ]
            routes = [
                r3_set_valued_route(
                    scores, threshold, minimum_margin=minimum_margin
                )
                for scores in calibration_r1_scores
            ]
            evaluation, _, _ = _evaluate_variant(
                calibration_rows,
                calibration_r1_scores,
                routes,
                sets,
                variant=f"R3-threshold-{threshold:.2f}-margin-{minimum_margin:.2f}",
                split_name=PublicSplitV2.CALIBRATION.value,
                config_sha256=config_sha,
                git=git,
                model_metadata=model_metadata,
            )
            threshold_evaluations[(threshold, minimum_margin)] = evaluation
            threshold_candidates.append(
                {
                    "threshold": threshold,
                    "minimum_margin": minimum_margin,
                    **_selection_metrics(evaluation),
                    "ambiguous_set_rate": evaluation["set_valued_routing"][
                        "ambiguous_set_rate"
                    ],
                    "unknown_set_rate": evaluation["set_valued_routing"][
                        "unknown_set_rate"
                    ],
                }
            )
    selected_threshold_row = max(
        threshold_candidates,
        key=lambda row: (
            -row["false_memory_access_rate"],
            row["safe_coverage"],
            row["ambiguous_set_rate"],
            row["unknown_set_rate"],
            -float(row["minimum_margin"]),
            -abs(float(row["threshold"]) - 0.60),
        ),
    )
    selected_threshold = float(selected_threshold_row["threshold"])
    selected_minimum_margin = float(selected_threshold_row["minimum_margin"])

    # R4 class-conditional conformal 同样只以 calibration known rows拟合与选择 alpha。
    calibration_targets = [
        RelationId(str(row["relation_id"])) if row["relation_id"] else None
        for row in calibration_rows
    ]
    configured_alphas = values["run"].get("conformal_alpha_grid", [0.10, 0.20])
    alpha_candidates: list[dict[str, Any]] = []
    calibrators: dict[float, ClassConditionalConformalCalibrator] = {}
    alpha_evaluations: dict[float, dict[str, Any]] = {}
    for raw_alpha in configured_alphas:
        alpha = float(raw_alpha)
        calibrator = fit_class_conditional_conformal(
            calibration_r1_scores, calibration_targets, alpha=alpha
        )
        calibrators[alpha] = calibrator
        sets = [calibrator.candidate_set(scores) for scores in calibration_r1_scores]
        routes = [calibrator.route(scores) for scores in calibration_r1_scores]
        evaluation, _, _ = _evaluate_variant(
            calibration_rows,
            calibration_r1_scores,
            routes,
            sets,
            variant=f"R4-alpha-{alpha:.2f}",
            split_name=PublicSplitV2.CALIBRATION.value,
            config_sha256=config_sha,
            git=git,
            model_metadata=model_metadata,
        )
        alpha_evaluations[alpha] = evaluation
        alpha_candidates.append(
            {"alpha": alpha, **_selection_metrics(evaluation)}
        )
    selected_alpha_row = max(
        alpha_candidates,
        key=lambda row: (
            -row["false_memory_access_rate"],
            row["safe_coverage"],
            row["accepted_relation_precision"],
            -row["average_candidate_set_size"],
            -abs(float(row["alpha"]) - 0.10),
        ),
    )
    selected_alpha = float(selected_alpha_row["alpha"])
    conformal = calibrators[selected_alpha]

    def variant_routes(
        variant: str,
        score_rows: Sequence[Mapping[RelationId, float]],
    ) -> tuple[list[AcceptedRoute | RejectedRoute], list[frozenset[RelationId]]]:
        if variant in {"R1", "R2"}:
            return _routes_argmax(score_rows)
        if variant == "R0":
            routes: list[AcceptedRoute | RejectedRoute] = []
            sets: list[frozenset[RelationId]] = []
            for score in score_rows:
                route = r0_frozen_argmax_route(
                    torch.tensor(
                        [
                            score[RelationId.ACCESS_CODE],
                            score[RelationId.CITY_CODE],
                            score[RelationId.REGISTRY_ID],
                        ]
                    )
                )
                routes.append(route)
                sets.append(
                    frozenset({route.relation_id})
                    if isinstance(route, AcceptedRoute)
                    else frozenset()
                )
            return routes, sets
        if variant == "R3":
            return (
                [
                    r3_set_valued_route(
                        score,
                        selected_threshold,
                        minimum_margin=selected_minimum_margin,
                    )
                    for score in score_rows
                ],
                [
                    _r3_candidates_with_margin(
                        score, selected_threshold, selected_minimum_margin
                    )
                    for score in score_rows
                ],
            )
        if variant == "R4":
            return (
                [conformal.route(score) for score in score_rows],
                [conformal.candidate_set(score) for score in score_rows],
            )
        raise ProtocolViolation(f"unknown smoke variant {variant}")

    calibration_scores_by_variant = {
        "R0": calibration_r1_scores,
        "R1": calibration_r1_scores,
        "R2": calibration_r2_scores,
        "R3": calibration_r1_scores,
        "R4": calibration_r1_scores,
    }
    evaluations: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    all_predictions: list[dict[str, Any]] = []
    all_retrievals: list[dict[str, Any]] = []
    score_csv_rows: list[dict[str, Any]] = []
    split_name = PublicSplitV2.CALIBRATION.value
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        variant_scores = calibration_scores_by_variant[variant]
        routes, candidate_sets = variant_routes(variant, variant_scores)
        evaluation, predictions, retrievals = _evaluate_variant(
            calibration_rows,
            variant_scores,
            routes,
            candidate_sets,
            variant=variant,
            split_name=split_name,
            config_sha256=config_sha,
            git=git,
            model_metadata=model_metadata,
        )
        evaluations[split_name][variant] = evaluation
        all_predictions.extend(predictions)
        all_retrievals.extend(retrievals)
        for row, evidence, candidates in zip(
            calibration_rows, variant_scores, candidate_sets
        ):
            score_csv_rows.append(
                {
                    "variant": variant,
                    "split": split_name,
                    "row_id": str(row["row_id"]),
                    "sample_type": str(row["sample_type"]),
                    "registry_id_score": float(evidence[RelationId.REGISTRY_ID]),
                    "city_code_score": float(evidence[RelationId.CITY_CODE]),
                    "access_code_score": float(evidence[RelationId.ACCESS_CODE]),
                    "candidate_set": "|".join(
                        sorted(item.value for item in candidates)
                    ),
                }
            )

    calibration_selection_inputs = {
        variant: _selection_metrics(
            evaluations[PublicSplitV2.CALIBRATION.value][variant]
        )
        for variant in ("R0", "R1", "R2", "R3", "R4")
    }
    smoke_gate_targets = {
        "known_coverage": float(values["gates"]["closed_set"].get("known_coverage", 0.0)),
        "accepted_relation_precision": 0.0,
        "singleton_acceptance_precision": 0.0,
        "safe_coverage": 0.0,
        "false_memory_access_rate": float(
            values["gates"]["open_set"].get("false_memory_access_rate", 1.0)
        ),
        "ambiguous_rejection_rate": 0.0,
        "unrelated_rejection_rate": 0.0,
        "ambiguous_false_memory_access_rate": 1.0,
        "unrelated_false_memory_access_rate": 1.0,
        "wrong_bucket_access_rate": 1.0,
        "worst_reject_family_false_accept_rate": 1.0,
        "family_macro_accuracy": 0.0,
        "worst_family_accuracy": 0.0,
        "true_relation_inclusion_rate": 0.0,
        "average_candidate_set_size": 3.0,
        "known_multi_relation_set_rate": 1.0,
        "known_empty_set_rate": 1.0,
    }
    selection = select_router_on_calibration(
        calibration_selection_inputs, gate_targets=smoke_gate_targets
    )
    selected_router = str(selection["selected_router"])
    router_parameters: dict[str, Any] = {
        "selected_aggregation": selected_aggregation_row,
        "r3_relation_threshold": selected_threshold,
        "r3_minimum_margin": selected_minimum_margin,
        "r2_ridge_strength": selected_ridge_strength,
        "r4_conformal": conformal.to_dict(),
        "r2_pairwise_head_sha256": canonical_sha256(pair_head.to_json_dict()),
    }
    frozen = freeze_router_selection(
        selected_router,
        router_parameters,
        public_train_sha256=canonical_sha256(train_rows),
        public_calibration_sha256=canonical_sha256(calibration_rows),
        review_manifest_sha256=canonical_sha256(
            {"scope": "synthetic_smoke", "human_review_used": False}
        ),
        git_commit=str(git["commit"]),
        model_manifests=model_metadata,
        selection_protocol={"synthetic_smoke_only": True},
    )

    # 只有 frozen SHA 已形成后才首次 materialize、score 和 evaluate synthetic locked。
    rows_by_split, benchmark_audit = build_public_benchmark_v2(
        values["benchmark"], run_historical_audit=False
    )
    if canonical_sha256(rows_by_split[PublicSplitV2.TRAIN.value]) != canonical_sha256(
        train_rows
    ) or canonical_sha256(
        rows_by_split[PublicSplitV2.CALIBRATION.value]
    ) != canonical_sha256(calibration_rows):
        raise ProtocolViolation("synthetic train/calibration changed after router freeze")
    locked_rows = rows_by_split[PublicSplitV2.LOCKED_AUDIT.value]
    if any("smoke" not in str(row["row_id"]).casefold() for row in locked_rows):
        raise ProtocolViolation("smoke audit received a non-synthetic locked fixture")
    locked_r1_scores = _score_rows_r1(
        locked_rows,
        definitions,
        aggregation=str(selected_aggregation_row["aggregation"]),
        top_k=selected_aggregation_row["top_k"],
    )
    locked_r2_scores = pair_scores(pair_head, locked_rows)
    locked_scores_by_variant = {
        "R0": locked_r1_scores,
        "R1": locked_r1_scores,
        "R2": locked_r2_scores,
        "R3": locked_r1_scores,
        "R4": locked_r1_scores,
    }
    split_name = PublicSplitV2.LOCKED_AUDIT.value
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        variant_scores = locked_scores_by_variant[variant]
        routes, candidate_sets = variant_routes(variant, variant_scores)
        evaluation, predictions, retrievals = _evaluate_variant(
            locked_rows,
            variant_scores,
            routes,
            candidate_sets,
            variant=variant,
            split_name=split_name,
            config_sha256=config_sha,
            git=git,
            model_metadata=model_metadata,
        )
        evaluations[split_name][variant] = evaluation
        all_predictions.extend(predictions)
        all_retrievals.extend(retrievals)
        for row, evidence, candidates in zip(
            locked_rows, variant_scores, candidate_sets
        ):
            score_csv_rows.append(
                {
                    "variant": variant,
                    "split": split_name,
                    "row_id": str(row["row_id"]),
                    "sample_type": str(row["sample_type"]),
                    "registry_id_score": float(evidence[RelationId.REGISTRY_ID]),
                    "city_code_score": float(evidence[RelationId.CITY_CODE]),
                    "access_code_score": float(evidence[RelationId.ACCESS_CODE]),
                    "candidate_set": "|".join(
                        sorted(item.value for item in candidates)
                    ),
                }
            )

    # 产物写入发生在 locked synthetic 评估之后；所有设置此前均已冻结。
    smoke_data_sha256 = {
        split: canonical_sha256(split_rows)
        for split, split_rows in rows_by_split.items()
    }
    smoke_selection_protocol = {
        "selection_split": PublicSplitV2.CALIBRATION.value,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "parameters_frozen_before_synthetic_locked_evaluation": True,
    }
    smoke_provenance = {
        "git": git,
        "model_metadata": model_metadata,
        "resolved_config_sha256": config_sha,
        "data_sha256": smoke_data_sha256,
        "review_manifest_sha256": None,
        "selection_protocol": smoke_selection_protocol,
    }
    _write_json(output_dir / "resolved_config.json", values)
    for split, split_rows in rows_by_split.items():
        _write_json(
            output_dir / f"{split}_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "stage": STAGE_NAME,
                "split": split,
                "protocol_role": "synthetic_integration_only",
                "row_count": len(split_rows),
                "row_sha256": canonical_sha256(split_rows),
                "answer_free": True,
                "contains_private_answers": False,
                "public_benchmark_human_reviewed": False,
                "formal_locked_audit_executed": False,
                **smoke_provenance,
            },
        )
    _write_json(
        output_dir / "public_benchmark_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "protocol_role": "synthetic_integration_only",
            "audit": benchmark_audit,
            "row_counts": {name: len(rows) for name, rows in rows_by_split.items()},
            "public_benchmark_human_reviewed": False,
            "formal_locked_audit_executed": False,
            **smoke_provenance,
        },
    )
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        _write_json(
            output_dir / variant / "evaluation.json",
            evaluations[PublicSplitV2.LOCKED_AUDIT.value][variant],
        )
    _write_csv(output_dir / "route_candidate_scores.csv", score_csv_rows)
    locked_predictions = [
        row
        for row in all_predictions
        if row["split"] == PublicSplitV2.LOCKED_AUDIT.value
    ]
    _write_csv(output_dir / "route_predictions.csv", locked_predictions)
    _write_json(
        output_dir / "retrieval_predictions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "protocol_role": "synthetic_integration_only",
            "git": git,
            "resolved_config_sha256": config_sha,
            "model_metadata": model_metadata,
            "data_sha256": smoke_data_sha256,
            "review_manifest_sha256": None,
            "selection_protocol": smoke_selection_protocol,
            "rows": [
                row
                for row in all_retrievals
                if row["split"] == PublicSplitV2.LOCKED_AUDIT.value
            ],
        },
    )
    error_rows = []
    for row in locked_predictions:
        known_error = row["sample_type"] == "known" and (
            row["route_status"] != "accept"
            or row["predicted_relation"] != row["target_relation"]
        )
        false_access = row["sample_type"] != "known" and row["route_status"] == "accept"
        reject_semantic_error = (
            row["sample_type"] == "ambiguous"
            and row["route_status"] == "reject"
            and row["reject_reason"] != "ambiguous"
        ) or (
            row["sample_type"] == "unrelated"
            and row["route_status"] == "reject"
            and row["reject_reason"] != "unknown"
        )
        if known_error or false_access or reject_semantic_error:
            error_rows.append(
                {
                    **row,
                    "error_type": (
                        "known_route_error"
                        if known_error
                        else "false_memory_access"
                        if false_access
                        else "reject_reason_mismatch"
                    ),
                }
            )
    _write_csv(output_dir / "error_analysis.csv", error_rows)
    risk_rows = []
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        curve = evaluations[PublicSplitV2.LOCKED_AUDIT.value][variant][
            "reject_quality"
        ]["risk_coverage"]["curve"]
        for point in curve:
            risk_rows.append({"variant": variant, **point})
    _write_csv(output_dir / "risk_coverage.csv", risk_rows)
    selection_rows = []
    for variant, row in selection["candidates"].items():
        metrics = row["metrics"]
        selection_rows.append(
            {
                "router": variant,
                "selected": variant == selected_router,
                "all_gates_passed": row["all_gates_passed"],
                "gate_count": row["gate_count"],
                **metrics,
            }
        )
    _write_csv(output_dir / "router_selection.csv", selection_rows)
    calibration_results = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "protocol_role": "synthetic_integration_only",
        "selection_split": PublicSplitV2.CALIBRATION.value,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "r1_aggregation_candidates": aggregation_candidates,
        "selected_r1_aggregation": selected_aggregation_row,
        "r2_ridge_candidates": ridge_candidates,
        "selected_r2_ridge": selected_ridge_row,
        "r3_threshold_candidates": threshold_candidates,
        "selected_r3_threshold": selected_threshold,
        "selected_r3_minimum_margin": selected_minimum_margin,
        "r4_alpha_candidates": alpha_candidates,
        "selected_r4_conformal": conformal.to_dict(),
        "router_selection": selection,
        **smoke_provenance,
        "frozen_router_selection": {
            "sha256": frozen.sha256,
            "payload": frozen.to_dict(),
            "verified": frozen.verify(),
        },
    }
    _write_json(output_dir / "calibration_results.json", calibration_results)
    _write_json(
        output_dir / "frozen_router_selection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sha256": frozen.sha256,
            "canonical_json": frozen.canonical_json,
            "verified": frozen.verify(),
            **smoke_provenance,
        },
    )
    ablation_rows = []
    for variant in ("R0", "R1", "R2", "R3", "R4"):
        evaluation = evaluations[PublicSplitV2.LOCKED_AUDIT.value][variant]
        known = evaluation["known_routing"]
        reject = evaluation["reject_quality"]
        access = evaluation["access_control"]
        retrieval = evaluation["fact_retrieval"]
        ablation_rows.append(
            {
                "variant": variant,
                "protocol_role": "synthetic_integration_only",
                "selected_on_calibration": variant == selected_router,
                "known_coverage": known["known_coverage"],
                "accepted_route_accuracy": known["accepted_route_accuracy"],
                "family_macro_accuracy": known["relation_family_macro_accuracy"],
                "worst_family_accuracy": known["worst_family_accuracy"],
                "ambiguous_rejection_rate": reject["ambiguous_rejection_rate"],
                "unrelated_rejection_rate": reject["unrelated_rejection_rate"],
                "false_memory_access_rate": access["false_memory_access_rate"],
                "wrong_bucket_access_rate": access["wrong_bucket_access_rate"],
                "safe_coverage": access["safe_coverage"],
                "accepted_fact_top1": retrieval["accepted_fact_top1"],
                "all_query_fact_top1": retrieval["all_query_fact_top1"],
                "mrr": retrieval["mrr"],
                "cross_relation_candidate_count": retrieval[
                    "cross_relation_candidate_count"
                ],
            }
        )
    _write_csv(output_dir / "stage_c24b_ablation.csv", ablation_rows)

    # Smoke 数值只证明代码路径可运行，绝不据此判定正式研究 readiness。
    readiness = compute_readiness(
        closed_set_selective_router_ready=False,
        open_set_abstention_ready=False,
        discrete_memory_contract_preserved=True,
        public_benchmark_human_reviewed=False,
    )
    selected_evaluation = evaluations[PublicSplitV2.LOCKED_AUDIT.value][selected_router]
    status = _protocol_status(
        public_benchmark_human_reviewed=False,
        formal_locked_audit_executed=False,
        calibration_executed=False,
        development_executed=False,
    )
    status.update(
        {
            "status": "smoke_completed_formal_readiness_not_assessed",
            "smoke_synthetic_audit_executed": True,
            "formal_readiness_assessed": False,
            **smoke_provenance,
            **readiness,
        }
    )
    _write_json(output_dir / "protocol_status.json", status)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": status["status"],
        "protocol_role": "synthetic_integration_only",
        "formal_locked_audit_executed": False,
        "public_benchmark_human_reviewed": False,
        "formal_research_thresholds_assessed": False,
        "selected_router_on_synthetic_calibration": selected_router,
        "selected_r1_aggregation": selected_aggregation_row,
        "selected_r2_ridge": selected_ridge_row,
        "selected_r3_threshold": selected_threshold,
        "selected_r3_minimum_margin": selected_minimum_margin,
        "selected_r4_conformal": conformal.to_dict(),
        "smoke_selected_evaluation": selected_evaluation,
        "readiness": readiness,
        "protocol_status": status,
        "resolved_config_sha256": config_sha,
        "data_sha256": smoke_data_sha256,
        "review_manifest_sha256": None,
        "git": git,
        "model_metadata": model_metadata,
        "selection_protocol": smoke_selection_protocol,
        "safety_scope": {
            "private_value_memory_trained": False,
            "private_answers_loaded": False,
            "answer_injection_executed": False,
            "key_attack_executed": False,
            "confirmation_created_or_read": False,
        },
    }
    _write_json(output_dir / "stage_c24b_summary.json", summary)

    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "protocol_role": "synthetic_integration_only",
        "git": git,
        "model_metadata": model_metadata,
        "resolved_config_sha256": config_sha,
        "data_sha256": smoke_data_sha256,
        "review_manifest_sha256": None,
        "selection_protocol": smoke_selection_protocol,
        "file_count": len(files),
        "files": files,
    }
    artifact_manifest["manifest_payload_sha256"] = canonical_sha256(artifact_manifest)
    _write_json(output_dir / "artifact_sha256_manifest.json", artifact_manifest)
    return summary


def _formal_provenance(
    config_path: str | Path,
    values: Mapping[str, Any],
    *,
    frozen_payload: Mapping[str, Any] | None = None,
    data_sha256: Mapping[str, Any] | None = None,
    review_manifest_sha256: str | None = None,
    model_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """为正式生命周期 JSON 提供一致且有限的 provenance envelope。"""

    frozen_data = (
        dict(frozen_payload.get("data_sha256", {})) if frozen_payload else {}
    )
    selection = (
        dict(frozen_payload.get("selection_protocol", {}))
        if frozen_payload
        else {
            "selection_split": PublicSplitV2.CALIBRATION.value,
            "development_used_for_selection": False,
            "locked_audit_used_for_selection": False,
            "parameters_frozen": False,
        }
    )
    review_sha = review_manifest_sha256 or frozen_data.get("review_manifest")
    return {
        "git": _git_state(_repo_root(config_path)),
        "model_provenance": dict(model_provenance or values.get("models", {})),
        "resolved_config_sha256": canonical_sha256(values),
        "data_sha256": dict(data_sha256 or frozen_data),
        "review_manifest_sha256": review_sha,
        "selection_protocol": selection,
    }


def _write_protocol_incident_once(
    output_dir: Path,
    *,
    violation: str,
    invalidated_split: str,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    path = output_dir / "protocol_incident.json"
    if path.exists():
        raise ProtocolViolation(
            "a prior C2.4b incident record exists and cannot be overwritten"
        )
    details = dict(provenance or {})
    return _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "violation": violation,
            "invalidated_split": invalidated_split,
            "current_locked_audit_v2_permanently_invalid": True,
            "next_required_protocol_version": "v3_independent_locked_audit",
            "historical_incident_overwritten": False,
            "formal_locked_audit_executed": False,
            "git": details.get("git"),
            "model_provenance": details.get("model_provenance"),
            "resolved_config_sha256": details.get("resolved_config_sha256"),
            "data_sha256": details.get("data_sha256"),
            "review_manifest_sha256": details.get("review_manifest_sha256"),
            "selection_protocol": details.get("selection_protocol"),
        },
    )


def _semantic_spec_for_model(model_id: str, revision: str):
    matches = [
        spec
        for spec in SEMANTIC_ENCODER_SPECS.values()
        if spec.model_id == str(model_id) and spec.revision == str(revision)
    ]
    if len(matches) != 1:
        raise ProtocolViolation(
            f"model ID/revision is not in the pinned semantic registry: {model_id}"
        )
    return matches[0]


def _load_verified_encoder(
    model_config: Mapping[str, Any],
    *,
    cache_dir: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    spec = _semantic_spec_for_model(
        str(model_config["model_id"]), str(model_config["revision"])
    )
    for field in ("model_manifest_sha256", "model_safetensors_sha256"):
        digest = str(model_config.get(field, ""))
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ProtocolViolation(f"formal model config lacks pinned {field}")
    encoder = load_semantic_encoder(spec, cache_dir=cache_dir, device=device)
    manifest = build_model_file_manifest(encoder.snapshot_path, spec)
    expected_manifest = str(model_config["model_manifest_sha256"])
    if manifest["manifest_sha256"] != expected_manifest:
        raise ProtocolViolation(f"model manifest SHA-256 mismatch for {spec.model_id}")
    weights = [
        row for row in manifest["files"] if row["path"] == "model.safetensors"
    ]
    expected_weight = str(model_config["model_safetensors_sha256"])
    if len(weights) != 1 or (
        weights[0]["sha256"] != expected_weight
    ):
        raise ProtocolViolation(f"model weight SHA-256 mismatch for {spec.model_id}")
    return encoder, manifest


def _model_manifest_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    weights = [
        row
        for row in manifest.get("files", [])
        if isinstance(row, Mapping) and row.get("path") == "model.safetensors"
    ]
    if len(weights) != 1:
        raise ProtocolViolation("model manifest lacks exactly one model.safetensors")
    return {
        "model_id": str(manifest["model_id"]),
        "revision": str(manifest["revision"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "model_safetensors_sha256": str(weights[0]["sha256"]),
    }


def _definition_embedding_map(
    encoder: Any,
    definitions: Mapping[str, Sequence[str]],
    *,
    batch_size: int,
    e5_input_type: str = "query",
) -> dict[RelationId, Tensor]:
    output: dict[RelationId, Tensor] = {}
    for relation in RelationId:
        texts = [str(value) for value in definitions[relation.value]]
        output[relation] = encode_semantic_texts(
            encoder,
            texts,
            batch_size=batch_size,
            e5_input_type=e5_input_type,
        )
    return output


def _r1_score_embeddings(
    embeddings: Tensor,
    definitions: Mapping[RelationId, Tensor],
    *,
    aggregation: str,
    top_k: int | None,
) -> list[dict[RelationId, float]]:
    return [
        r1_definition_ensemble_scores(
            embedding,
            definitions,
            aggregation=aggregation,
            top_k=top_k,
        )
        for embedding in embeddings
    ]


def _r0_scores_and_routes(
    embeddings: Tensor,
    head: Any,
) -> tuple[list[dict[RelationId, float]], list[AcceptedRoute | RejectedRoute]]:
    logits = head.logits(embeddings)
    classes = [RelationId(str(value)) for value in head.classes]
    score_rows = [
        {relation: float(logits[index, classes.index(relation)]) for relation in RelationId}
        for index in range(len(logits))
    ]
    routes = [
        r0_frozen_argmax_route(logits[index], relation_order=classes)
        for index in range(len(logits))
    ]
    return score_rows, routes


def _formal_gate_targets(values: Mapping[str, Any]) -> dict[str, float]:
    closed = values["gates"]["closed_set"]
    opened = values["gates"]["open_set"]
    conformal = values["gates"]["conformal"]
    return {
        "known_coverage": float(closed["known_coverage"]),
        "accepted_relation_precision": float(closed["accepted_route_accuracy"]),
        "singleton_acceptance_precision": float(
            opened["singleton_acceptance_precision"]
        ),
        "safe_coverage": float(closed["safe_coverage"]),
        "false_memory_access_rate": float(opened["false_memory_access_rate"]),
        "ambiguous_rejection_rate": float(opened["ambiguous_rejection_rate"]),
        "unrelated_rejection_rate": float(opened["unrelated_rejection_rate"]),
        "ambiguous_false_memory_access_rate": float(
            opened["ambiguous_false_memory_access_rate"]
        ),
        "unrelated_false_memory_access_rate": float(
            opened["unrelated_false_memory_access_rate"]
        ),
        "wrong_bucket_access_rate": float(closed["wrong_bucket_access_rate"]),
        "worst_reject_family_false_accept_rate": float(
            opened["worst_reject_family_far"]
        ),
        "family_macro_accuracy": float(closed["public_family_macro_accuracy"]),
        "worst_family_accuracy": float(closed["worst_family_accuracy"]),
        "true_relation_inclusion_rate": float(conformal["target_coverage"])
        - float(conformal["coverage_tolerance"]),
        "average_candidate_set_size": float(conformal["average_set_size"]),
        "known_multi_relation_set_rate": float(
            conformal["known_multi_relation_set_rate"]
        ),
        "known_empty_set_rate": float(conformal["known_empty_set_rate"]),
    }


def _runtime_source_manifest(config_path: str | Path) -> dict[str, Any]:
    root = _repo_root(config_path)
    # 封存整个运行包而非脆弱的手写传递依赖列表；由此也覆盖 phase_a、
    # data/model/config 等被 C2.1/C2.3/C2.4 间接调用的模块。
    paths = [Path(config_path).resolve(), root / "pyproject.toml"]
    paths.extend(sorted((root / "src" / "keyed_gram").glob("*.py")))
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {"files": files, "payload_sha256": canonical_sha256(files)}


def _verify_runtime_sources_at_head(
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """证明所有正式 runtime source/config 字节都属于当前 HEAD。

    data、review 与 artifact 可按各自 seal 变更；这里只约束会影响执行语义的
    tracked code/config，避免把未提交代码错误归因到旧 commit。
    """

    root = _repo_root(config_path)
    git = _git_state(root)
    commit = str(git["commit"])
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ProtocolViolation("formal runtime requires a valid 40-hex git HEAD")
    manifest = _runtime_source_manifest(config_path)
    for row in manifest["files"]:
        relative = str(row["path"])
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"], cwd=root
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ProtocolViolation(
                f"formal runtime source is not tracked at HEAD: {relative}"
            ) from exc
        if hashlib.sha256(committed).hexdigest() != row["sha256"]:
            raise ProtocolViolation(
                f"formal runtime source differs from HEAD: {relative}"
            )
    return git, manifest


def _evaluate_formal_candidate(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[RelationId, float]],
    routes: Sequence[AcceptedRoute | RejectedRoute],
    sets: Sequence[frozenset[RelationId]],
    *,
    name: str,
    split: str,
    config_sha: str,
    git: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation, _ = _evaluate_relation_only(
        rows,
        scores,
        routes,
        sets,
        variant=name,
        split_name=split,
        config_sha256=config_sha,
        git=git,
        model_metadata=model_metadata,
        protocol_role="formal_public_calibration_only",
        evaluation_scope="public_calibration_v2_router_selection",
    )
    return evaluation


def _run_formal_calibration_backend(
    config_path: str | Path,
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    destination: Path,
    *,
    device_name: str | None,
    train_seal: Mapping[str, Any],
    calibration_seal: Mapping[str, Any],
    review_manifest_path: Path,
    fixed_before: Mapping[str, str],
    cache_binding: Mapping[str, Any],
    frozen_code_state: Mapping[str, Any],
    committed_source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    data_dir = _resolve_path(config_path, values["benchmark"]["public_data_dir"])
    train_rows = _read_jsonl(data_dir / f"{PublicSplitV2.TRAIN.value}.jsonl")
    calibration_rows = _read_jsonl(
        data_dir / f"{PublicSplitV2.CALIBRATION.value}.jsonl"
    )
    artifact_dir = _resolve_path(config_path, values["benchmark"]["artifact_dir"])
    train_rows = _apply_adjudicated_reviews(
        train_rows, artifact_dir / "public_train_v2_review.csv", train_split=True
    )
    calibration_rows = _apply_adjudicated_reviews(
        calibration_rows,
        artifact_dir / "public_calibration_v2_review.csv",
        train_split=False,
    )
    device = resolve_device(device_name)
    batch_size = int(values["run"]["batch_size"])
    git = dict(frozen_code_state)
    if _git_state(_repo_root(config_path))["commit"] != git["commit"]:
        raise ProtocolViolation("git HEAD changed before formal calibration started")
    source_manifest = _runtime_source_manifest(config_path)
    if source_manifest != dict(committed_source_manifest):
        raise ProtocolViolation("runtime source changed before formal calibration started")
    c24 = yaml.safe_load(
        _resolve_path(config_path, values["fixed_sources"]["stage_c24_config"]).read_text(
            encoding="utf-8"
        )
    )
    cache_dir = _resolve_path(config_path, c24["fixed_s5"]["model_cache_dir"])

    # R0：原 C2.4 BGE + frozen S5 head，不重拟合旧阈值或 head。
    r0_model_config = {
        **values["models"]["R0"],
        "model_manifest_sha256": c24["fixed_s5"]["model_manifest_sha256"],
    }
    r0_encoder, r0_manifest = _load_verified_encoder(
        r0_model_config, cache_dir=cache_dir, device=device
    )
    r0_head = load_fixed_s5_head(
        _resolve_path(config_path, values["fixed_sources"]["stage_c23_head"]),
        expected_sha256=str(values["fixed_sources"]["stage_c23_head_sha256"]),
        expected_variant=str(c24["fixed_s5"]["base_encoder_variant"]),
        expected_view=str(c24["fixed_s5"]["view"]),
        expected_classes=c24["fixed_s5"]["classes"],
    )
    r0_embeddings = encode_semantic_texts(
        r0_encoder,
        [str(row["input_views"]["entity_masked_strip_suffix"]) for row in calibration_rows],
        batch_size=batch_size,
    )
    r0_scores, r0_routes = _r0_scores_and_routes(r0_embeddings, r0_head)
    r0_sets = [
        frozenset({route.relation_id})
        if isinstance(route, AcceptedRoute)
        else frozenset()
        for route in r0_routes
    ]

    # R1：MiniLM definition ensemble，聚合仅在 calibration 选择。
    r1_model_config = {
        **values["models"]["R1"],
        "model_manifest_sha256": c24["fixed_g3"]["model_manifest_sha256"],
    }
    r1_encoder, r1_manifest = _load_verified_encoder(
        r1_model_config, cache_dir=cache_dir, device=device
    )
    query_view = str(values["run"].get("query_view", "entity_masked_strip_suffix"))
    if query_view != "entity_masked_strip_suffix":
        raise ProtocolViolation("formal router query_view must use the full masked question")
    calibration_texts = [
        str(row["input_views"][query_view]) for row in calibration_rows
    ]
    r1_embeddings = encode_semantic_texts(
        r1_encoder, calibration_texts, batch_size=batch_size
    )
    r1_definitions = _definition_embedding_map(
        r1_encoder, values["benchmark"]["definitions"], batch_size=batch_size
    )
    aggregation_rows: list[dict[str, Any]] = []
    aggregation_scores: dict[str, list[dict[RelationId, float]]] = {}
    for aggregation in values["run"]["evidence_aggregations"]:
        top_values = (
            values["models"]["R1"]["top_k_values"]
            if aggregation == "top_k_mean"
            else [None]
        )
        for top_k in top_values:
            name = str(aggregation) if top_k is None else f"{aggregation}_{int(top_k)}"
            score_rows = _r1_score_embeddings(
                r1_embeddings,
                r1_definitions,
                aggregation=str(aggregation),
                top_k=None if top_k is None else int(top_k),
            )
            aggregation_scores[name] = score_rows
            routes, sets = _routes_argmax(score_rows)
            evaluation = _evaluate_formal_candidate(
                calibration_rows,
                score_rows,
                routes,
                sets,
                name=f"R1-{name}",
                split=PublicSplitV2.CALIBRATION.value,
                config_sha=str(metadata["resolved_config_sha256"]),
                git=git,
                model_metadata={"R1": _model_manifest_provenance(r1_manifest)},
            )
            aggregation_rows.append(
                {
                    "name": name,
                    "aggregation": str(aggregation),
                    "top_k": top_k,
                    **_selection_metrics(evaluation),
                }
            )
    selected_aggregation = max(
        aggregation_rows,
        key=lambda row: (
            row["safe_coverage"],
            -row["false_memory_access_rate"],
            row["family_macro_accuracy"],
            row["worst_family_accuracy"],
            -aggregation_rows.index(row),
        ),
    )
    r1_scores = aggregation_scores[str(selected_aggregation["name"])]

    # R2：每个 frozen encoder candidate 分别仅用 public_train_v2 拟合 binary ridge。
    r2_rows: list[dict[str, Any]] = []
    r2_state_hashes: dict[str, str] = {}
    r2_head_states: dict[str, dict[str, Any]] = {}
    r2_scores_by_name: dict[str, list[dict[RelationId, float]]] = {}
    r2_manifests: dict[str, Any] = {}
    r2_encoders: dict[str, Any] = {}
    train_texts = [str(row["input_views"][query_view]) for row in train_rows]
    train_targets = [RelationId(str(row["relation_id"])) for row in train_rows]
    for candidate_name, candidate_config in values["models"]["R2"]["candidates"].items():
        encoder, model_manifest = _load_verified_encoder(
            candidate_config, cache_dir=cache_dir, device=device
        )
        r2_encoders[str(candidate_name)] = encoder
        r2_manifests[str(candidate_name)] = model_manifest
        train_embeddings = encode_semantic_texts(
            encoder, train_texts, batch_size=batch_size, e5_input_type="query"
        )
        calibration_embeddings = encode_semantic_texts(
            encoder, calibration_texts, batch_size=batch_size, e5_input_type="query"
        )
        definition_embeddings = _definition_embedding_map(
            encoder, values["benchmark"]["definitions"], batch_size=batch_size
        )
        train_features = build_pairwise_relation_features(
            train_embeddings, definition_embeddings
        )
        calibration_features = build_pairwise_relation_features(
            calibration_embeddings, definition_embeddings
        )
        for raw_strength in values["run"]["ridge_strength_grid"]:
            strength = float(raw_strength)
            head = fit_pairwise_ridge_head(
                train_features, train_targets, ridge_strength=strength
            )
            matrices = head.score_batch(calibration_features)
            score_rows = [
                {relation: float(matrices[relation][index]) for relation in RelationId}
                for index in range(len(calibration_rows))
            ]
            name = f"{candidate_name}:ridge={strength:g}"
            routes, sets = _routes_argmax(score_rows)
            evaluation = _evaluate_formal_candidate(
                calibration_rows,
                score_rows,
                routes,
                sets,
                name=f"R2-{name}",
                split=PublicSplitV2.CALIBRATION.value,
                config_sha=str(metadata["resolved_config_sha256"]),
                git=git,
                model_metadata={"R2": _model_manifest_provenance(model_manifest)},
            )
            r2_rows.append(
                {
                    "name": name,
                    "candidate": str(candidate_name),
                    "ridge_strength": strength,
                    **_selection_metrics(evaluation),
                }
            )
            r2_state_hashes[name] = canonical_sha256(head.to_json_dict())
            r2_head_states[name] = head.to_state_dict()
            r2_scores_by_name[name] = score_rows
    selected_r2 = max(
        r2_rows,
        key=lambda row: (
            row["safe_coverage"],
            -row["false_memory_access_rate"],
            row["family_macro_accuracy"],
            row["worst_family_accuracy"],
            -r2_rows.index(row),
        ),
    )
    r2_scores = r2_scores_by_name[str(selected_r2["name"])]

    # R3/R4 在 calibration 比较 selected R1 与 selected R2 independent evidence。
    evidence_sources = {"R1": r1_scores, "R2": r2_scores}
    threshold_rows: list[dict[str, Any]] = []
    for evidence_source, base_scores in evidence_sources.items():
        for raw_threshold in values["run"]["threshold_grid"]:
            for raw_margin in values["run"]["minimum_margin_grid"]:
                threshold, margin = float(raw_threshold), float(raw_margin)
                sets = [
                    _r3_candidates_with_margin(score, threshold, margin)
                    for score in base_scores
                ]
                routes = [
                    r3_set_valued_route(score, threshold, minimum_margin=margin)
                    for score in base_scores
                ]
                evaluation = _evaluate_formal_candidate(
                    calibration_rows,
                    base_scores,
                    routes,
                    sets,
                    name=f"R3-{evidence_source}-t={threshold:g}-m={margin:g}",
                    split=PublicSplitV2.CALIBRATION.value,
                    config_sha=str(metadata["resolved_config_sha256"]),
                    git=git,
                    model_metadata={"evidence_source": evidence_source},
                )
                threshold_rows.append(
                    {
                        "evidence_source": evidence_source,
                        "threshold": threshold,
                        "minimum_margin": margin,
                        **_selection_metrics(evaluation),
                    }
                )
    selected_threshold = max(
        threshold_rows,
        key=lambda row: (
            row["safe_coverage"],
            -row["false_memory_access_rate"],
            row["ambiguous_rejection_rate"],
            row["unrelated_rejection_rate"],
            row["accepted_relation_precision"],
            -threshold_rows.index(row),
        ),
    )
    r3_base_scores = evidence_sources[str(selected_threshold["evidence_source"])]
    r3_sets = [
        _r3_candidates_with_margin(
            score,
            float(selected_threshold["threshold"]),
            float(selected_threshold["minimum_margin"]),
        )
        for score in r3_base_scores
    ]
    r3_routes = [
        r3_set_valued_route(
            score,
            selected_threshold["threshold"],
            minimum_margin=selected_threshold["minimum_margin"],
        )
        for score in r3_base_scores
    ]
    targets = [
        RelationId(str(row["relation_id"])) if row["relation_id"] else None
        for row in calibration_rows
    ]
    conformal_rows: list[dict[str, Any]] = []
    conformals: dict[tuple[str, float], ClassConditionalConformalCalibrator] = {}
    for evidence_source, base_scores in evidence_sources.items():
        for raw_alpha in values["run"]["conformal_alpha_grid"]:
            alpha = float(raw_alpha)
            calibrator = fit_class_conditional_conformal(
                base_scores, targets, alpha=alpha
            )
            conformals[(evidence_source, alpha)] = calibrator
            sets = [calibrator.candidate_set(score) for score in base_scores]
            routes = [calibrator.route(score) for score in base_scores]
            evaluation = _evaluate_formal_candidate(
                calibration_rows,
                base_scores,
                routes,
                sets,
                name=f"R4-{evidence_source}-alpha={alpha:g}",
                split=PublicSplitV2.CALIBRATION.value,
                config_sha=str(metadata["resolved_config_sha256"]),
                git=git,
                model_metadata={"evidence_source": evidence_source},
            )
            conformal_rows.append(
                {
                    "evidence_source": evidence_source,
                    "alpha": alpha,
                    **_selection_metrics(evaluation),
                }
            )
    selected_conformal_row = max(
        conformal_rows,
        key=lambda row: (
            row["safe_coverage"],
            -row["false_memory_access_rate"],
            row["true_relation_inclusion_rate"],
            -row["average_candidate_set_size"],
            -conformal_rows.index(row),
        ),
    )
    r4_source = str(selected_conformal_row["evidence_source"])
    conformal = conformals[(r4_source, float(selected_conformal_row["alpha"]))]
    r4_base_scores = evidence_sources[r4_source]
    r4_sets = [conformal.candidate_set(score) for score in r4_base_scores]
    r4_routes = [conformal.route(score) for score in r4_base_scores]

    route_data = {
        "R0": (r0_scores, r0_routes, r0_sets),
        "R1": (r1_scores, *_routes_argmax(r1_scores)),
        "R2": (r2_scores, *_routes_argmax(r2_scores)),
        "R3": (r3_base_scores, r3_routes, r3_sets),
        "R4": (r4_base_scores, r4_routes, r4_sets),
    }
    evaluations: dict[str, Any] = {}
    score_csv: list[dict[str, Any]] = []
    for variant, (score_rows, routes, sets) in route_data.items():
        evaluation, _ = _evaluate_relation_only(
            calibration_rows,
            score_rows,
            routes,
            sets,
            variant=variant,
            split_name=PublicSplitV2.CALIBRATION.value,
            config_sha256=str(metadata["resolved_config_sha256"]),
            git=git,
            model_metadata={
                "R0": _model_manifest_provenance(r0_manifest),
                "R1": _model_manifest_provenance(r1_manifest),
                "R2": _model_manifest_provenance(
                    r2_manifests[str(selected_r2["candidate"])]
                ),
            },
            protocol_role="formal_public_calibration_only",
            evaluation_scope="public_calibration_v2_router_selection",
        )
        evaluations[variant] = evaluation
        for row, score, candidates in zip(calibration_rows, score_rows, sets):
            score_csv.append(
                {
                    "variant": variant,
                    "split": PublicSplitV2.CALIBRATION.value,
                    "row_id": row["row_id"],
                    "registry_id_score": score[RelationId.REGISTRY_ID],
                    "city_code_score": score[RelationId.CITY_CODE],
                    "access_code_score": score[RelationId.ACCESS_CODE],
                    "candidate_set": "|".join(sorted(value.value for value in candidates)),
                }
            )
    selection = select_router_on_calibration(
        {variant: _selection_metrics(evaluation) for variant, evaluation in evaluations.items()},
        gate_targets=_formal_gate_targets(values),
    )
    selected_router = str(selection["selected_router"])
    selected_r2_name = str(selected_r2["name"])
    r2_head_path = destination / "r2_pairwise_head.pt"
    torch.save(r2_head_states[selected_r2_name], r2_head_path)
    router_parameters = {
        "selected_r1_aggregation": selected_aggregation,
        "selected_r2": selected_r2,
        "selected_r2_head_state_sha256": r2_state_hashes[selected_r2_name],
        "selected_r2_head_file": r2_head_path.name,
        "selected_r2_head_file_sha256": sha256_file(r2_head_path),
        "selected_r3": selected_threshold,
        "selected_r4": {
            "evidence_source": r4_source,
            "calibrator": conformal.to_dict(),
        },
        "reviewed_train_rows_sha256": canonical_sha256(train_rows),
        "reviewed_calibration_rows_sha256": canonical_sha256(calibration_rows),
        # 全部候选与选择结果进入 immutable freeze；正式 audit 的根目录
        # calibration/selection 产物只能从该 frozen payload 派生，不能读取
        # 一个未封存的 calibration 输出再影响报告。
        "calibration_evidence": {
            "r1_aggregation_candidates": aggregation_rows,
            "r2_candidates": r2_rows,
            "r3_candidates": threshold_rows,
            "r4_candidates": conformal_rows,
            "router_selection": selection,
        },
    }
    model_manifests = {
        "R0": r0_manifest,
        "R1": r1_manifest,
        "R2": r2_manifests,
        "runtime_source_manifest": source_manifest,
    }
    frozen = freeze_router_selection(
        selected_router,
        router_parameters,
        public_train_sha256=str(train_seal["data_sha256"]),
        public_calibration_sha256=str(calibration_seal["data_sha256"]),
        review_manifest_sha256=sha256_file(review_manifest_path),
        git_commit=str(git["commit"]),
        model_manifests=model_manifests,
        selection_protocol={"formal_public_calibration_only": True},
    )
    model_manifests_after = {
        "R0": build_model_file_manifest(r0_encoder.snapshot_path, r0_encoder.spec),
        "R1": build_model_file_manifest(r1_encoder.snapshot_path, r1_encoder.spec),
        "R2": {
            name: build_model_file_manifest(encoder.snapshot_path, encoder.spec)
            for name, encoder in r2_encoders.items()
        },
    }
    if model_manifests_after != {
        "R0": r0_manifest,
        "R1": r1_manifest,
        "R2": r2_manifests,
    }:
        raise ProtocolViolation("model snapshot files changed during formal calibration")
    restored_head = PairwiseRidgeEvidenceHead.from_state_dict(
        torch.load(r2_head_path, map_location="cpu", weights_only=True)
    )
    if (
        sha256_file(r2_head_path) != router_parameters["selected_r2_head_file_sha256"]
        or canonical_sha256(restored_head.to_json_dict())
        != router_parameters["selected_r2_head_state_sha256"]
    ):
        raise ProtocolViolation("saved R2 head differs from its frozen calibration seal")
    source_after_models = _runtime_source_manifest(config_path)
    if source_after_models != source_manifest:
        raise ProtocolViolation("runtime source changed during formal calibration")
    if _git_state(_repo_root(config_path))["commit"] != git["commit"]:
        raise ProtocolViolation("git HEAD changed during formal calibration")
    _write_csv(destination / "route_candidate_scores.csv", score_csv)
    selection_csv = [
        {
            "router": variant,
            "selected": variant == selected_router,
            "all_gates_passed": row["all_gates_passed"],
            "gate_count": row["gate_count"],
            **row["metrics"],
        }
        for variant, row in selection["candidates"].items()
    ]
    _write_csv(destination / "router_selection.csv", selection_csv)
    tradeoff_rows = []
    for method, candidates in (("R3", threshold_rows), ("R4", conformal_rows)):
        for row in candidates:
            tradeoff_rows.append(
                {
                    "method": method,
                    "threshold": row.get("threshold", ""),
                    "minimum_margin": row.get("minimum_margin", ""),
                    "alpha": row.get("alpha", ""),
                    "known_coverage": row["known_coverage"],
                    "accepted_risk": 1.0
                    - row["singleton_acceptance_precision"],
                    "accepted_precision": row["singleton_acceptance_precision"],
                    "known_route_risk": 1.0
                    - row["accepted_relation_precision"],
                    "known_route_accuracy": row["accepted_relation_precision"],
                    "safe_coverage": row["safe_coverage"],
                    "false_memory_access_rate": row["false_memory_access_rate"],
                    "ambiguous_rejection_rate": row["ambiguous_rejection_rate"],
                    "unrelated_rejection_rate": row["unrelated_rejection_rate"],
                }
            )
    _write_csv(destination / "risk_coverage.csv", tradeoff_rows)
    calibration_results = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "protocol_role": "formal_public_calibration_only",
        "selection_split": PublicSplitV2.CALIBRATION.value,
        "development_used_for_selection": False,
        "locked_audit_used_for_selection": False,
        "nominal_conformal_coverage_guarantee_claimed": False,
        "conformal_validity_limitation": (
            "evidence source and alpha were selected on this same calibration split; "
            "standard nominal split-conformal coverage is not claimed"
        ),
        "r1_aggregation_candidates": aggregation_rows,
        "selected_r1_aggregation": selected_aggregation,
        "r2_candidates": r2_rows,
        "selected_r2": selected_r2,
        "r3_candidates": threshold_rows,
        "selected_r3": selected_threshold,
        "r4_candidates": conformal_rows,
        "selected_r4": {
            "evidence_source": r4_source,
            "calibrator": conformal.to_dict(),
        },
        "router_selection": selection,
        "frozen_router_selection_sha256": frozen.sha256,
        "model_revisions": {
            "R0": r0_model_config,
            "R1": r1_model_config,
            "R2": values["models"]["R2"]["candidates"],
        },
        "git": git,
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "data_sha256": {
            PublicSplitV2.TRAIN.value: train_seal["data_sha256"],
            PublicSplitV2.CALIBRATION.value: calibration_seal["data_sha256"],
        },
        "review_manifest_sha256": sha256_file(review_manifest_path),
        "selection_protocol": frozen.to_dict()["selection_protocol"],
    }
    _write_json(destination / "calibration_results.json", calibration_results)
    _write_json(
        destination / "frozen_router_selection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sha256": frozen.sha256,
            "canonical_json": frozen.canonical_json,
            "verified": frozen.verify(),
            "git": git,
            "model_revisions": calibration_results["model_revisions"],
            "resolved_config_sha256": metadata["resolved_config_sha256"],
            "data_sha256": calibration_results["data_sha256"],
            "review_manifest_sha256": calibration_results[
                "review_manifest_sha256"
            ],
            "selection_protocol": calibration_results["selection_protocol"],
        },
    )
    fixed_after = _verify_declared_fixed_sources(
        config_path, values, lightweight_only=False
    )
    if dict(fixed_before) != fixed_after:
        raise ProtocolViolation("fixed source hash changed during formal calibration")
    cache_after = _verify_answer_free_cache_binding(config_path, values)
    if dict(cache_binding) != cache_after:
        raise ProtocolViolation("answer-free cache binding changed during calibration")
    source_final = _runtime_source_manifest(config_path)
    if source_final != source_manifest:
        raise ProtocolViolation("runtime source changed while writing calibration seals")
    git_final = _git_state(_repo_root(config_path))
    if git_final["commit"] != git["commit"]:
        raise ProtocolViolation("git HEAD changed while writing calibration seals")
    status = {
        **_protocol_status(
            public_benchmark_human_reviewed=False,
            formal_locked_audit_executed=False,
            calibration_executed=True,
            development_executed=False,
        ),
        "status": "formal_calibration_completed_router_frozen",
        "selected_router": selected_router,
        "frozen_router_selection_sha256": frozen.sha256,
        "fixed_source_sha256_before": dict(fixed_before),
        "fixed_source_sha256_after": fixed_after,
        "fixed_source_sha256_equal": True,
        "answer_free_cache_binding": dict(cache_binding),
        "answer_free_cache_binding_after": cache_after,
        "runtime_source_manifest_before": source_manifest,
        "runtime_source_manifest_after": source_final,
        "runtime_source_manifest_equal": True,
        "model_manifests_before": {
            "R0": r0_manifest,
            "R1": r1_manifest,
            "R2": r2_manifests,
        },
        "model_manifests_after": model_manifests_after,
        "model_manifests_equal": True,
        "git_before": git,
        "git_after": git_final,
        "git_commit_equal": True,
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "data_sha256": calibration_results["data_sha256"],
        "review_manifest_sha256": calibration_results[
            "review_manifest_sha256"
        ],
        "selection_protocol": calibration_results["selection_protocol"],
    }
    _write_json(destination / "protocol_status.json", status)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE_NAME,
        "status": status["status"],
        "selected_router": selected_router,
        "frozen_router_selection_sha256": frozen.sha256,
        "formal_locked_audit_executed": False,
        "public_benchmark_human_reviewed": False,
        "c3_eligible": False,
    }


def calibrate_stage_c24b(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """正式 calibration 入口；严格不读取 public_locked_audit_v2。"""

    values, metadata = load_stage_c24b_config(config_path)
    if values["run"]["mode"] != "formal":
        raise ProtocolViolation("stage-c24b-calibrate requires the formal config")
    artifact_dir = _resolve_path(config_path, values["benchmark"]["artifact_dir"])
    destination = Path(output_dir).resolve()
    if destination != (artifact_dir / "calibration").resolve():
        raise ProtocolViolation(
            "formal calibration output must equal benchmark.artifact_dir/calibration"
        )
    if (destination / "frozen_router_selection.json").exists():
        raise ProtocolViolation("a frozen formal calibration cannot be overwritten")
    required_names = [PublicSplitV2.TRAIN.value, PublicSplitV2.CALIBRATION.value]
    if bool(values["benchmark"].get("generate_legacy_c24_review", True)):
        required_names.append("legacy_c24_locked_audit")

    # 第一门只读 review CSV 与 prepared manifest；locked v2 路径不在列表中。
    review_paths = _review_paths_for_scope(
        config_path, values, include_locked_v2=False
    )
    _preflight_review_gate(review_paths)
    train_seal = _verify_split_seal(config_path, values, PublicSplitV2.TRAIN)
    calibration_seal = _verify_split_seal(
        config_path, values, PublicSplitV2.CALIBRATION
    )
    review_manifest = validate_stage_c24b_reviews(
        config_path,
        require_complete=True,
        write_manifest=False,
        required_reviews=required_names,
    )
    # 人工门完成后才允许核对/加载大型 source；缺失时明确 FileNotFound。
    fixed_before = _verify_declared_fixed_sources(
        config_path, values, lightweight_only=False
    )
    cache_binding = _verify_answer_free_cache_binding(config_path, values)
    frozen_code_state, committed_source_manifest = _verify_runtime_sources_at_head(
        config_path
    )
    destination.mkdir(parents=True, exist_ok=True)
    calibration_review_manifest_path = destination / "review_manifest.json"
    _write_json(calibration_review_manifest_path, review_manifest)
    return _run_formal_calibration_backend(
        config_path,
        values,
        metadata,
        destination,
        device_name=device_name,
        train_seal=train_seal,
        calibration_seal=calibration_seal,
        review_manifest_path=calibration_review_manifest_path,
        fixed_before=fixed_before,
        cache_binding=cache_binding,
        frozen_code_state=frozen_code_state,
        committed_source_manifest=committed_source_manifest,
    )


def audit_stage_c24b(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """运行 smoke，或在全部 seal 完成后进入正式 audit 的 fail-closed 门。"""

    values, metadata = load_stage_c24b_config(config_path)
    destination = Path(output_dir).resolve()
    if values["run"]["mode"] == "smoke":
        formal_root = (_repo_root(config_path) / "artifacts" / "stage_c24b").resolve()
        if destination == formal_root or formal_root in destination.parents:
            raise ProtocolViolation("smoke output cannot overlap the formal artifact root")
        if (destination / "stage_c24b_summary.json").exists():
            raise ProtocolViolation("a completed smoke audit cannot be overwritten")
        return _run_smoke_audit(config_path, values, metadata, destination)

    artifact_dir = _resolve_path(config_path, values["benchmark"]["artifact_dir"])
    if destination != artifact_dir:
        raise ProtocolViolation("formal audit output must equal benchmark.artifact_dir")
    pending_summary_path = destination / "stage_c24b_summary.json"
    if pending_summary_path.exists():
        pending_summary = json.loads(pending_summary_path.read_text(encoding="utf-8"))
        if not (
            pending_summary.get("status")
            == "prepared_waiting_for_independent_human_review"
            and pending_summary.get("formal_locked_audit_executed") is False
        ):
            raise ProtocolViolation(
                "a completed or unknown formal summary cannot be overwritten"
            )
    interrupted_markers = [
        name
        for name in (
            "development_started.json",
            "development_completed.json",
            "locked_audit_opened.json",
        )
        if (destination / name).exists()
    ]
    if interrupted_markers:
        _write_protocol_incident_once(
            destination,
            violation=(
                "interrupted one-shot formal lifecycle cannot be resumed: "
                + ", ".join(interrupted_markers)
            ),
            invalidated_split=PublicSplitV2.LOCKED_AUDIT.value,
            provenance=_formal_provenance(config_path, values),
        )
        raise ProtocolViolation(
            "formal one-shot markers exist without a completed audit; "
            "locked audit v2 is permanently invalid"
        )

    # 第一门在任何 locked JSONL 读取、source/model load 之前只读审核 CSV。
    review_paths = _review_paths_for_scope(
        config_path, values, include_locked_v2=True
    )
    _preflight_review_gate(review_paths)
    required_names = [
        PublicSplitV2.TRAIN.value,
        PublicSplitV2.CALIBRATION.value,
        PublicSplitV2.LOCKED_AUDIT.value,
    ]
    if bool(values["benchmark"].get("generate_legacy_c24_review", True)):
        required_names.append("legacy_c24_locked_audit")

    calibration_dir = artifact_dir / "calibration"
    frozen_path = calibration_dir / "frozen_router_selection.json"
    frozen_wrapper = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen = FrozenRouterSelection(
        canonical_json=str(frozen_wrapper["canonical_json"]),
        sha256=str(frozen_wrapper["sha256"]),
    )
    if not frozen.verify() or frozen_wrapper.get("verified") is not True:
        raise ProtocolViolation("frozen router selection SHA-256 is invalid")
    frozen_payload = frozen.to_dict()
    current_git = _git_state(_repo_root(config_path))
    if current_git["commit"] != frozen_payload["git_commit"]:
        raise ProtocolViolation(
            "current git HEAD differs from the frozen router commit"
        )
    calibration_review_path = calibration_dir / "review_manifest.json"
    _verify_review_manifest_current(calibration_review_path)
    if sha256_file(calibration_review_path) != frozen_payload["data_sha256"][
        "review_manifest"
    ]:
        raise ProtocolViolation("calibration review manifest differs from frozen seal")
    train_seal = _verify_split_seal(config_path, values, PublicSplitV2.TRAIN)
    calibration_seal = _verify_split_seal(
        config_path, values, PublicSplitV2.CALIBRATION
    )
    if train_seal["data_sha256"] != frozen_payload["data_sha256"][
        "public_train_v2"
    ] or calibration_seal["data_sha256"] != frozen_payload["data_sha256"][
        "public_calibration_v2"
    ]:
        raise ProtocolViolation("train/calibration data differs from frozen router seal")
    committed_git, current_source_manifest = _verify_runtime_sources_at_head(
        config_path
    )
    if committed_git["commit"] != frozen_payload["git_commit"]:
        raise ProtocolViolation(
            "runtime source commit differs from the frozen router commit"
        )
    frozen_source_manifest = frozen_payload["model_manifests"].get(
        "runtime_source_manifest"
    )
    if current_source_manifest != frozen_source_manifest:
        raise ProtocolViolation("router source/config files differ from frozen SHA manifest")

    # Source、model 与 development 必须在第一次 locked seal/open 之前完成。
    fixed_before = _verify_declared_fixed_sources(
        config_path, values, lightweight_only=False
    )
    cache_binding = _verify_answer_free_cache_binding(config_path, values)
    from .stage_c24b_formal import (
        prepare_formal_prelocked,
        run_formal_locked_backend,
    )

    prelocked = prepare_formal_prelocked(
        config_path,
        values,
        frozen_payload,
        device_name=device_name,
        destination=destination,
    )
    try:
        destination.mkdir(parents=True, exist_ok=True)
        marker_path = destination / "locked_audit_opened.json"
        marker_provenance = _formal_provenance(
            config_path,
            values,
            frozen_payload=frozen_payload,
            model_provenance=frozen_payload["model_manifests"],
        )
        _write_json(
            marker_path,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": STAGE_NAME,
                "status": "point_of_no_return_before_first_locked_read",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "frozen_router_selection_sha256": frozen.sha256,
                "runtime_source_manifest_sha256": current_source_manifest[
                    "payload_sha256"
                ],
                "development_executed_once": True,
                "locked_audit_used_for_selection": False,
                **marker_provenance,
            },
        )
        locked_seal = _verify_split_seal(
            config_path, values, PublicSplitV2.LOCKED_AUDIT
        )
        full_review = validate_stage_c24b_reviews(
            config_path,
            require_complete=True,
            write_manifest=False,
            required_reviews=required_names,
        )
        full_review_path = destination / "review_manifest.json"
        _write_json(full_review_path, full_review)
        data_dir = _resolve_path(config_path, values["benchmark"]["public_data_dir"])
        locked_path = data_dir / f"{PublicSplitV2.LOCKED_AUDIT.value}.jsonl"
        locked_bytes = locked_path.read_bytes()
        locked_bytes_sha256 = hashlib.sha256(locked_bytes).hexdigest()
        if locked_bytes_sha256 != locked_seal["data_sha256"]:
            raise ProtocolViolation(
                "locked JSONL bytes differ between seal verification and parsing"
            )
        locked_rows = _parse_jsonl_bytes(
            locked_bytes, source=str(locked_path)
        )
        locked_review_path = artifact_dir / "public_locked_audit_v2_review.csv"
        locked_review_bytes = locked_review_path.read_bytes()
        locked_review_sha256 = hashlib.sha256(locked_review_bytes).hexdigest()
        expected_locked_review_sha256 = str(
            full_review["reviews"][PublicSplitV2.LOCKED_AUDIT.value]["sha256"]
        )
        if locked_review_sha256 != expected_locked_review_sha256:
            raise ProtocolViolation(
                "locked review bytes differ between validation and overlay"
            )
        locked_review_rows = _parse_review_csv_bytes(
            locked_review_bytes, source=str(locked_review_path)
        )
        locked_rows = _apply_adjudicated_review_rows(
            locked_rows,
            locked_review_rows,
            train_split=False,
        )
        reviewed_locked_rows_sha256 = canonical_sha256(locked_rows)
        return run_formal_locked_backend(
            config_path,
            values,
            metadata,
            destination,
            frozen_payload,
            locked_rows,
            prelocked=prelocked,
            review_manifest_sha256=sha256_file(full_review_path),
            fixed_before=fixed_before,
            cache_binding={
                **cache_binding,
                "locked_data_sha256": locked_bytes_sha256,
                "locked_review_csv_sha256": locked_review_sha256,
                "reviewed_locked_rows_sha256": reviewed_locked_rows_sha256,
                "locked_split_manifest_sha256": locked_seal["manifest_sha256"],
            },
        )
    except Exception as exc:
        _write_protocol_incident_once(
            destination,
            violation=str(exc),
            invalidated_split=PublicSplitV2.LOCKED_AUDIT.value,
            provenance=_formal_provenance(
                config_path,
                values,
                frozen_payload=frozen_payload,
                model_provenance=frozen_payload["model_manifests"],
            ),
        )
        raise


# CLI 与外部调用保留语义清晰的别名。
run_stage_c24b_calibration = calibrate_stage_c24b
run_stage_c24b_audit = audit_stage_c24b


__all__ = [
    "ProtocolViolation",
    "ReviewIncompleteError",
    "audit_stage_c24b",
    "calibrate_stage_c24b",
    "load_stage_c24b_config",
    "prepare_stage_c24b",
    "run_stage_c24b_audit",
    "run_stage_c24b_calibration",
    "validate_stage_c24b_reviews",
]
