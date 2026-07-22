"""Stage C2.4b v2.1 的严格序列化、原子写入与产物封存工具。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stage_c24b_benchmark import sha256_file, strict_json_dumps


class V21ProtocolError(RuntimeError):
    """v2.1 协议、seal 或 AI 审核不满足 fail-closed 条件。"""


def canonical_sha256(value: Any) -> str:
    """计算禁用 NaN/Infinity 的 canonical JSON SHA-256。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    """在目标目录内写临时文件，再以原子 rename 发布文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(value)
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    """以严格 JSON 和结尾换行原子写入对象。"""

    atomic_write_text(path, strict_json_dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """以严格 JSON 原子写入一组 JSONL 记录。"""

    atomic_write_text(
        path,
        "".join(strict_json_dumps(dict(row)) + "\n" for row in rows),
    )


def render_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """按照固定字段顺序渲染 CSV，并拒绝额外字段。"""

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    """原子写入固定 schema 的 CSV。"""

    atomic_write_text(path, render_csv(rows, fields))


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """读取非空 CSV，同时保留字段顺序供严格 schema 校验。"""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not rows:
        raise V21ProtocolError(f"CSV is empty: {path}")
    return fields, rows


def artifact_files(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    """生成稳定排序的相对路径、大小与 SHA-256 清单。"""

    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in exclude:
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise V21ProtocolError(f"strict JSON contains duplicate key: {key}")
        value[key] = child
    return value


def strict_json_loads(text: str, *, label: str) -> Any:
    """读取拒绝重复键及 NaN/Infinity 的严格 JSON。"""

    try:
        return json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise V21ProtocolError(f"{label} is not strict JSON") from error


def load_payload_sealed_json(path: Path, *, label: str) -> dict[str, Any]:
    """读取并验证带 ``manifest_payload_sha256`` 的 JSON 对象。"""

    if not path.is_file():
        raise V21ProtocolError(f"{label} is missing")
    value = strict_json_loads(path.read_text(encoding="utf-8"), label=label)
    if not isinstance(value, dict):
        raise V21ProtocolError(f"{label} must be a JSON object")
    observed = value.get("manifest_payload_sha256")
    payload = {
        key: child for key, child in value.items() if key != "manifest_payload_sha256"
    }
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise V21ProtocolError(f"{label} payload SHA-256 is invalid")
    return value


def safe_artifact_path(root: Path, relative: str) -> Path:
    """将 manifest 相对路径限制在对应 artifact root 内。"""

    if not relative or Path(relative).is_absolute():
        raise V21ProtocolError("artifact manifest contains an unsafe path")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise V21ProtocolError("artifact manifest path escapes its root")
    return path


def load_exact_artifact_manifest(
    artifact_dir: Path, manifest_path: Path, *, label: str
) -> dict[str, Any]:
    """读取 payload-sealed manifest，并验证其文件清单精确覆盖目录。"""

    manifest = load_payload_sealed_json(manifest_path, label=label)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise V21ProtocolError("final artifact manifest lacks a file inventory")
    inventory: list[str] = []
    for row in raw_files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise V21ProtocolError("final artifact file-inventory schema differs")
        relative = str(row["path"])
        inventory.append(relative)
        path = safe_artifact_path(artifact_dir, relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise V21ProtocolError(f"final artifact changed after seal: {relative}")
    current = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file()
    }
    if (
        len(inventory) != len(set(inventory))
        or manifest.get("file_count") != len(inventory)
        or current != set(inventory) | {manifest_path.name}
    ):
        raise V21ProtocolError("final artifact inventory is not exact")
    return manifest


def verify_prepare_snapshot(
    artifact_dir: Path,
    config_path: Path,
    *,
    before_review: bool,
    review_splits: Sequence[str],
) -> dict[str, Any]:
    """验证 prepare 阶段的不可变 artifact snapshot 及精确文件集合。"""

    manifest_path = artifact_dir / "prepare_artifact_sha256_manifest.json"
    manifest = load_payload_sealed_json(manifest_path, label="v2.1 prepare manifest")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise V21ProtocolError("v2.1 prepare-manifest config seal differs")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise V21ProtocolError("v2.1 prepare manifest lacks a file inventory")
    paths: list[str] = []
    transitioned = {"protocol_status.json", "stage_c24b_v21_summary.json"}
    for row in raw_files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise V21ProtocolError("v2.1 prepare file-inventory schema differs")
        relative = str(row["path"])
        paths.append(relative)
        if not before_review and relative in transitioned:
            continue
        path = safe_artifact_path(artifact_dir, relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise V21ProtocolError(f"sealed v2.1 prepared artifact changed: {relative}")
    if len(paths) != len(set(paths)) or int(manifest.get("file_count", -1)) != len(
        paths
    ):
        raise V21ProtocolError("v2.1 prepare file inventory is not unique/complete")
    if before_review:
        current = {
            path.relative_to(artifact_dir).as_posix()
            for path in artifact_dir.rglob("*")
            if path.is_file()
        }
        expected = set(paths) | {manifest_path.name}
        permitted_partial = {"ai_review_attempt_manifest.json"} | {
            f"{split}_ai_review.csv" for split in review_splits
        }
        extras = current - expected
        if extras - permitted_partial:
            raise V21ProtocolError("unexpected files exist outside the prepared snapshot")
        if extras and "ai_review_attempt_manifest.json" not in extras:
            raise V21ProtocolError(
                "partial AI review outputs lack an immutable attempt seal"
            )
    return manifest


__all__ = [
    "V21ProtocolError",
    "artifact_files",
    "atomic_write_text",
    "canonical_sha256",
    "load_exact_artifact_manifest",
    "load_payload_sealed_json",
    "read_csv",
    "render_csv",
    "safe_artifact_path",
    "strict_json_loads",
    "verify_prepare_snapshot",
    "write_csv",
    "write_json",
    "write_jsonl",
]
