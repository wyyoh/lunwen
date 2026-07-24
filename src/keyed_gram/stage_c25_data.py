"""CLINC150 与 BANKING77 的官方 split、intent 分区和碰撞门禁。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .stage_c25_protocol import (
    C25ProtocolError,
    canonical_sha256,
    download_verified,
    load_config,
    repo_root,
    resolve_path,
)


@dataclass(frozen=True, slots=True)
class IntentExample:
    row_id: str
    text: str
    label: str | None
    sample_type: str
    unknown_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetRows:
    fit: tuple[IntentExample, ...]
    calibration: tuple[IntentExample, ...]
    test: tuple[IntentExample, ...]


def _row_id(dataset: str, split: str, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}:{split}:{index:05d}:{digest}"


def prepare_sources(
    config_path: str | Path, *, opener: Any = None
) -> dict[str, Any]:
    values = load_config(config_path)
    root = resolve_path(config_path, values["sources"]["cache_dir"])
    output: dict[str, Any] = {}
    for dataset in ("clinc150", "banking77"):
        declaration = values["sources"][dataset]
        files = {}
        for name, file_declaration in declaration["files"].items():
            observed = download_verified(
                root / dataset / name,
                url=str(file_declaration["url"]),
                sha256=str(file_declaration["sha256"]),
                opener=opener,
            )
            observed["path"] = (
                (root / dataset / name)
                .relative_to(repo_root(config_path))
                .as_posix()
            )
            files[name] = observed
        output[dataset] = {
            "repository": declaration["repository"],
            "revision": declaration["revision"],
            "license": declaration["license"],
            "paper": declaration["paper"],
            "files": files,
        }
    _validate_official_sources(root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "C2.5-external-source-freeze",
        "datasets_committed_to_repository": False,
        "sources": output,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def _load_clinc(root: Path) -> tuple[dict[str, list[list[str]]], dict[str, list[str]]]:
    data = json.loads(
        (root / "clinc150" / "data_full.json").read_text(encoding="utf-8")
    )
    domains = json.loads(
        (root / "clinc150" / "domains.json").read_text(encoding="utf-8")
    )
    if not isinstance(data, Mapping) or not isinstance(domains, Mapping):
        raise C25ProtocolError("CLINC150 JSON schema 非法")
    return dict(data), dict(domains)


def _load_banking(root: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    splits: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "test"):
        with (root / "banking77" / f"{split}.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if not rows or set(rows[0]) != {"text", "category"}:
            raise C25ProtocolError("BANKING77 CSV schema 非法")
        splits[split] = rows
    categories = json.loads(
        (root / "banking77" / "categories.json").read_text(encoding="utf-8")
    )
    if not isinstance(categories, list):
        raise C25ProtocolError("BANKING77 categories 非法")
    return splits, [str(value) for value in categories]


def _validate_official_sources(root: Path) -> None:
    clinc, domains = _load_clinc(root)
    expected = {
        "train": 15000,
        "val": 3000,
        "test": 4500,
        "oos_train": 100,
        "oos_val": 100,
        "oos_test": 1000,
    }
    if {key: len(value) for key, value in clinc.items()} != expected:
        raise C25ProtocolError("CLINC150 official full split 行数漂移")
    intents = {label for _, label in clinc["train"]}
    domain_intents = [intent for values in domains.values() for intent in values]
    if (
        len(domains) != 10
        or len(intents) != 150
        or len(domain_intents) != 150
        or set(domain_intents) != intents
        or any(
            len({label for _, label in clinc[split]}) != 150
            for split in ("train", "val", "test")
        )
    ):
        raise C25ProtocolError("CLINC150 intent/domain schema 漂移")
    counts = {
        split: Counter(label for _, label in clinc[split])
        for split in ("train", "val", "test")
    }
    if (
        set(counts["train"].values()) != {100}
        or set(counts["val"].values()) != {20}
        or set(counts["test"].values()) != {30}
    ):
        raise C25ProtocolError("CLINC150 per-intent split 计数漂移")

    banking, categories = _load_banking(root)
    if (
        len(banking["train"]) != 10003
        or len(banking["test"]) != 3080
        or len(categories) != 77
        or len(set(categories)) != 77
        or {row["category"] for row in banking["train"]} != set(categories)
        or {row["category"] for row in banking["test"]} != set(categories)
        or set(Counter(row["category"] for row in banking["test"]).values())
        != {40}
    ):
        raise C25ProtocolError("BANKING77 official split/category 漂移")


def build_partitions(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    root = resolve_path(config_path, values["sources"]["cache_dir"])
    _validate_official_sources(root)
    _, domains = _load_clinc(root)
    _, banking_categories = _load_banking(root)
    seeds = values["protocol"]["seeds"]
    per_domain = int(
        values["partitions"]["clinc150"]["supported_intents_per_domain"]
    )
    partitions = []
    for seed in seeds:
        supported: list[str] = []
        for domain in sorted(domains):
            candidates = sorted(domains[domain])
            random.Random(f"clinc:{seed}:{domain}").shuffle(candidates)
            supported.extend(candidates[:per_domain])
        unsupported = sorted(
            set(intent for values_ in domains.values() for intent in values_)
            - set(supported)
        )
        categories = sorted(banking_categories)
        random.Random(f"banking:{seed}").shuffle(categories)
        bank_supported = sorted(categories[:39])
        bank_calibration_oos = sorted(categories[39:58])
        bank_test_oos = sorted(categories[58:])
        if (
            set(bank_supported) & set(bank_calibration_oos)
            or set(bank_supported) & set(bank_test_oos)
            or set(bank_calibration_oos) & set(bank_test_oos)
        ):
            raise C25ProtocolError("BANKING77 intent partition 发生碰撞")
        partitions.append(
            {
                "seed": seed,
                "clinc150": {
                    "supported_intents": sorted(supported),
                    "unsupported_intents": unsupported,
                },
                "banking77_open": {
                    "supported_intents": bank_supported,
                    "calibration_oos_intents": bank_calibration_oos,
                    "test_only_oos_intents": bank_test_oos,
                },
                "banking77_closed": {
                    "supported_intents": sorted(banking_categories)
                },
            }
        )
    if len({canonical_sha256(row) for row in partitions}) != len(partitions):
        raise C25ProtocolError("不同 seed 产生了重复 intent partition")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "C2.5-preregistered-intent-partitions",
        "seeds": list(seeds),
        "partition_generation_uses_test_text_or_predictions": False,
        "partition_unit": "whole_intent",
        "partitions": partitions,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def _split_train_rows(
    rows: Sequence[dict[str, str]],
    labels: set[str],
    *,
    seed: int,
    fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["category"] in labels:
            by_label[row["category"]].append(row)
    fit, calibration = [], []
    for label in sorted(labels):
        candidates = sorted(by_label[label], key=lambda row: row["text"])
        random.Random(f"bank-row:{seed}:{label}").shuffle(candidates)
        cut = max(1, min(len(candidates) - 1, int(len(candidates) * fraction)))
        fit.extend(candidates[:cut])
        calibration.extend(candidates[cut:])
    return fit, calibration


def _examples(
    dataset: str,
    split: str,
    rows: Iterable[tuple[str, str | None, str, str]],
) -> tuple[IntentExample, ...]:
    output = []
    for index, (text, label, sample_type, unknown_kind) in enumerate(rows):
        clean = str(text).strip()
        if not clean:
            raise C25ProtocolError(f"{dataset}/{split} 出现空文本")
        output.append(
            IntentExample(
                row_id=_row_id(dataset, split, index, clean),
                text=clean,
                label=label,
                sample_type=sample_type,
                unknown_kind=unknown_kind,
            )
        )
    return tuple(output)


def load_partition_rows(
    config_path: str | Path,
    partition: Mapping[str, Any],
    benchmark: str,
    *,
    include_test: bool,
) -> DatasetRows:
    values = load_config(config_path)
    root = resolve_path(config_path, values["sources"]["cache_dir"])
    seed = int(partition["seed"])
    if benchmark == "clinc150":
        data, _ = _load_clinc(root)
        supported = set(partition["clinc150"]["supported_intents"])
        unsupported = set(partition["clinc150"]["unsupported_intents"])
        fit = _examples(
            benchmark,
            f"fit-{seed}",
            (
                (text, label, "known", "")
                for text, label in data["train"]
                if label in supported
            ),
        )
        calibration_rows = [
            (text, label, "known", "")
            for text, label in data["val"]
            if label in supported
        ]
        calibration_rows.extend(
            (text, None, "unknown", "heldout_intent")
            for text, label in data["val"]
            if label in unsupported
        )
        calibration_rows.extend(
            (text, None, "unknown", "official_oos")
            for text, _ in data["oos_val"]
        )
        test_rows: list[tuple[str, str | None, str, str]] = []
        if include_test:
            test_rows.extend(
                (text, label, "known", "")
                for text, label in data["test"]
                if label in supported
            )
            test_rows.extend(
                (text, None, "unknown", "heldout_intent")
                for text, label in data["test"]
                if label in unsupported
            )
            test_rows.extend(
                (text, None, "unknown", "official_oos")
                for text, _ in data["oos_test"]
            )
    elif benchmark in {"banking77_open", "banking77_closed"}:
        data, categories = _load_banking(root)
        if benchmark == "banking77_open":
            declaration = partition[benchmark]
            supported = set(declaration["supported_intents"])
            calibration_oos = set(declaration["calibration_oos_intents"])
            test_oos = set(declaration["test_only_oos_intents"])
        else:
            supported = set(categories)
            calibration_oos, test_oos = set(), set()
        fit_raw, calibration_known = _split_train_rows(
            data["train"], supported, seed=seed, fraction=0.8
        )
        fit = _examples(
            benchmark,
            f"fit-{seed}",
            (
                (row["text"], row["category"], "known", "")
                for row in fit_raw
            ),
        )
        calibration_rows = [
            (row["text"], row["category"], "known", "")
            for row in calibration_known
        ]
        if benchmark == "banking77_open":
            calibration_rows.extend(
                (row["text"], None, "unknown", "calibration_oos_intent")
                for row in data["train"]
                if row["category"] in calibration_oos
            )
        test_rows = []
        if include_test:
            test_rows.extend(
                (row["text"], row["category"], "known", "")
                for row in data["test"]
                if row["category"] in supported
            )
            if benchmark == "banking77_open":
                test_rows.extend(
                    (row["text"], None, "unknown", "test_only_oos_intent")
                    for row in data["test"]
                    if row["category"] in test_oos
                )
    else:
        raise C25ProtocolError(f"未知 C2.5 benchmark：{benchmark}")
    result = DatasetRows(
        fit=fit,
        calibration=_examples(
            benchmark, f"calibration-{seed}", calibration_rows
        ),
        test=_examples(benchmark, f"test-{seed}", test_rows),
    )
    if not include_test and result.test:
        raise C25ProtocolError("calibration 路径意外加载 test rows")
    fit_ids = {row.row_id for row in result.fit}
    calibration_ids = {row.row_id for row in result.calibration}
    test_ids = {row.row_id for row in result.test}
    if fit_ids & calibration_ids or fit_ids & test_ids or calibration_ids & test_ids:
        raise C25ProtocolError("C2.5 split row_id 发生碰撞")
    return result
