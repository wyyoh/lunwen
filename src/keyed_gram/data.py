from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor


def normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def discover_labels(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = metadata.get("all", {}).get("labels") or metadata.get("labels")
        if labels:
            return sorted(str(label) for label in labels)
    suffix = "_train.bin"
    return sorted(path.name[: -len(suffix)] for path in root.glob(f"*{suffix}"))


@dataclass
class TokenStreamPool:
    paths: tuple[Path, ...]
    context_length: int
    fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("a token pool requires at least one binary shard")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.streams = tuple(np.memmap(path, dtype=np.uint16, mode="r") for path in self.paths)
        subset_rng = np.random.default_rng(self.seed)
        subset_starts: list[int] = []
        subset_lengths: list[int] = []
        for stream in self.streams:
            if self.fraction == 1:
                selected_length = len(stream)
            else:
                selected_length = min(
                    len(stream),
                    max(self.context_length + 1, int(len(stream) * self.fraction)),
                )
            max_offset = len(stream) - selected_length
            offset = int(subset_rng.integers(0, max_offset + 1)) if max_offset else 0
            subset_starts.append(offset)
            subset_lengths.append(selected_length)
        self.subset_starts = np.asarray(subset_starts, dtype=np.int64)
        self.subset_lengths = np.asarray(subset_lengths, dtype=np.int64)
        # Number of valid context+target windows within each fixed subset.
        self.available = np.maximum(0, self.subset_lengths - self.context_length)
        if int(self.available.sum()) <= 0:
            raise ValueError("token shards are shorter than the context length")
        self.probabilities = self.available / self.available.sum()
        self.window_counts = np.maximum(
            0, (self.subset_lengths - 1) // self.context_length
        )
        self.num_sequences = int(self.window_counts.sum())
        if self.num_sequences <= 0:
            raise ValueError("fixed token subsets contain no complete sequence")

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
        stream_indices = rng.choice(len(self.streams), size=batch_size, p=self.probabilities)
        rows = np.empty((batch_size, self.context_length + 1), dtype=np.int64)
        for row_index, stream_index in enumerate(stream_indices):
            valid_count = int(self.available[stream_index])
            start = int(self.subset_starts[stream_index]) + int(rng.integers(0, valid_count))
            rows[row_index] = self.streams[stream_index][start : start + self.context_length + 1]
        tokens = torch.from_numpy(rows)
        return tokens[:, :-1].long(), tokens[:, 1:].long()


class TokenEpochSampler:
    """Shuffled, no-replacement traversal of fixed non-overlapping windows."""

    def __init__(self, pool: TokenStreamPool, rng: np.random.Generator) -> None:
        self.pool = pool
        self.rng = rng
        self.stream_indices = np.repeat(
            np.arange(len(pool.streams), dtype=np.int32), pool.window_counts
        )
        local_parts = [
            np.arange(int(count), dtype=np.int64) for count in pool.window_counts
        ]
        self.local_indices = np.concatenate(local_parts)
        self.order = self.rng.permutation(pool.num_sequences)
        self.cursor = 0

    def _take_indices(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("batch size must be positive")
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            available = self.pool.num_sequences - self.cursor
            take = min(remaining, available)
            pieces.append(self.order[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == self.pool.num_sequences:
                self.order = self.rng.permutation(self.pool.num_sequences)
                self.cursor = 0
        return np.concatenate(pieces)

    def sample_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        indices = self._take_indices(batch_size)
        rows = np.empty((batch_size, self.pool.context_length + 1), dtype=np.int64)
        for row_index, flat_index in enumerate(indices):
            stream_index = int(self.stream_indices[flat_index])
            local_index = int(self.local_indices[flat_index])
            start = int(self.pool.subset_starts[stream_index]) + (
                local_index * self.pool.context_length
            )
            rows[row_index] = self.pool.streams[stream_index][
                start : start + self.pool.context_length + 1
            ]
        tokens = torch.from_numpy(rows)
        return tokens[:, :-1].long(), tokens[:, 1:].long()


def iter_eval_batches(
    path: str | Path,
    context_length: int,
    max_sequences: int,
    batch_size: int,
    *,
    seed: int = 123,
) -> Iterator[tuple[Tensor, Tensor]]:
    stream = np.memmap(Path(path), dtype=np.uint16, mode="r")
    num_windows = max(0, (len(stream) - 1) // context_length)
    if num_windows == 0:
        raise ValueError(f"token shard {path} is shorter than one evaluation window")
    total = min(max_sequences, num_windows)
    rng = np.random.default_rng(seed)
    if total == num_windows:
        window_indices = np.arange(num_windows, dtype=np.int64)
    else:
        window_indices = rng.choice(num_windows, size=total, replace=False)
    starts = window_indices * context_length
    for offset in range(0, total, batch_size):
        current = starts[offset : offset + batch_size]
        rows = np.stack(
            [stream[int(start) : int(start) + context_length + 1] for start in current]
        ).astype(np.int64, copy=False)
        tokens = torch.from_numpy(rows)
        yield tokens[:, :-1].long(), tokens[:, 1:].long()


@dataclass(frozen=True)
class StoryDataBundle:
    data_dir: Path
    private_label: str
    core_labels: tuple[str, ...]
    private_pool: TokenStreamPool
    core_pool: TokenStreamPool

    @property
    def private_train_path(self) -> Path:
        return self.data_dir / f"{self.private_label}_train.bin"

    def test_path(self, label: str) -> Path:
        return self.data_dir / f"{label}_test.bin"


def load_story_data(
    data_dir: str | Path,
    private_label: str,
    context_length: int,
    train_fraction: float,
    seed: int,
) -> StoryDataBundle:
    root = Path(data_dir)
    private_label = normalize_label(private_label)
    labels = discover_labels(root)
    if private_label not in labels:
        raise ValueError(f"private label {private_label!r} is absent from {root}")
    core_labels = tuple(label for label in labels if label != private_label)
    private_path = root / f"{private_label}_train.bin"
    core_paths = tuple(root / f"{label}_train.bin" for label in core_labels)
    missing = [path for path in (private_path, *core_paths) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing token shards: {missing[:3]}")
    return StoryDataBundle(
        data_dir=root,
        private_label=private_label,
        core_labels=core_labels,
        private_pool=TokenStreamPool((private_path,), context_length, train_fraction, seed + 1),
        core_pool=TokenStreamPool(core_paths, context_length, train_fraction, seed + 2),
    )


def download_official_story_bins(destination: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="erol-AE/GR-MoE",
        repo_type="dataset",
        allow_patterns=["stories/*.bin", "stories/metadata.json"],
        local_dir=destination.parent,
    )
    if not destination.exists():
        raise FileNotFoundError(f"download did not create {destination}")
    return destination


def build_story_bins(
    destination: str | Path,
    *,
    sample_fraction: float = 1.0,
    seed: int = 42,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
) -> Path:
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*_*.bin"):
        stale.unlink()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    metadata: dict = {}
    labels: set[str] = set()
    total_by_split = {"train": 0, "test": 0}
    for split in ("train", "test"):
        dataset = load_dataset("SimpleStories/SimpleStories", split=split)
        if sample_fraction < 1:
            count = max(1, int(len(dataset) * sample_fraction))
            dataset = dataset.shuffle(seed=seed).select(range(count))
        handles: dict[str, object] = {}
        try:
            for row in tqdm(dataset, desc=f"tokenizing {split}"):
                label = normalize_label(str(row["topic"]))
                labels.add(label)
                if label not in handles:
                    handles[label] = (destination / f"{label}_{split}.bin").open("ab")
                token_ids = tokenizer.encode(str(row["story"]), add_special_tokens=False)
                token_ids.append(int(tokenizer.eos_token_id))
                if token_ids and max(token_ids) >= 65536:
                    raise ValueError("tokenizer vocabulary does not fit uint16 storage")
                array("H", token_ids).tofile(handles[label])
                metadata.setdefault(label, {}).setdefault(split, {"total_tokens": 0})
                metadata[label][split]["total_tokens"] += len(token_ids)
                total_by_split[split] += len(token_ids)
        finally:
            for handle in handles.values():
                handle.close()
    metadata["all"] = {
        "total_tokens_train": total_by_split["train"],
        "total_tokens_test": total_by_split["test"],
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "labels": sorted(labels),
        "sample_fraction": sample_fraction,
        "seed": seed,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def build_story_bins_from_rows_api(
    destination: str | Path,
    *,
    train_rows: int = 21200,
    test_rows: int = 21400,
    page_size: int = 100,
    workers: int = 4,
    requests_per_second: float = 1.5,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
) -> Path:
    """Build a ~1% official smoke set without downloading all parquet shards."""
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    if train_rows <= 0 or test_rows <= 0 or not 1 <= page_size <= 100:
        raise ValueError("row counts must be positive and page_size must be in [1, 100]")
    if workers <= 0 or requests_per_second <= 0:
        raise ValueError("workers and requests_per_second must be positive")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    rate_lock = threading.Lock()
    next_request_at = [0.0]

    def wait_for_request_slot() -> None:
        with rate_lock:
            now = time.monotonic()
            scheduled = max(now, next_request_at[0])
            next_request_at[0] = scheduled + 1.0 / requests_per_second
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)

    def fetch(split: str, offset: int, length: int) -> tuple[int, list[dict]]:
        cache_dir = destination / ".rows-cache" / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{offset}-{length}.json"
        if cache_path.exists():
            return offset, json.loads(cache_path.read_text(encoding="utf-8"))
        params = urllib.parse.urlencode(
            {
                "dataset": "SimpleStories/SimpleStories",
                "config": "default",
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        request = urllib.request.Request(
            "https://datasets-server.huggingface.co/rows?" + params,
            headers={"User-Agent": "keyed-gram-mvp/0.1"},
        )
        last_error: Exception | None = None
        for attempt in range(12):
            try:
                wait_for_request_slot()
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as error:
                last_error = error
                if attempt == 11:
                    raise
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if error.code == 429:
                    delay = float(retry_after) if retry_after else min(5.0 * (attempt + 1), 60.0)
                else:
                    delay = min(1.5 * (attempt + 1), 15.0)
                time.sleep(delay)
            except Exception as error:
                last_error = error
                if attempt == 11:
                    raise
                time.sleep(min(1.5 * (attempt + 1), 15.0))
        else:
            raise RuntimeError("unreachable rows API retry state") from last_error
        rows = [item["row"] for item in payload["rows"]]
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_path)
        return offset, rows

    def fetch_split(split: str, limit: int) -> list[dict]:
        pages = [
            (offset, min(page_size, limit - offset))
            for offset in range(0, limit, page_size)
        ]
        ordered: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch, split, offset, length): offset
                for offset, length in pages
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"fetching {split} rows"
            ):
                offset, rows = future.result()
                ordered[offset] = rows
        return [row for offset in sorted(ordered) for row in ordered[offset]]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    metadata: dict = {}
    labels: set[str] = set()
    total_by_split = {"train": 0, "test": 0}
    selected_by_split = {"train": train_rows, "test": test_rows}
    rows_by_split = {
        split: fetch_split(split, selected_by_split[split])
        for split in ("train", "test")
    }
    # Do not replace usable bins until every remote page is safely cached.
    for stale in destination.glob("*_*.bin"):
        stale.unlink()
    for split in ("train", "test"):
        rows = rows_by_split[split]
        handles: dict[str, object] = {}
        try:
            for row in tqdm(rows, desc=f"tokenizing {split}"):
                label = normalize_label(str(row["topic"]))
                labels.add(label)
                if label not in handles:
                    handles[label] = (destination / f"{label}_{split}.bin").open("ab")
                token_ids = tokenizer.encode(str(row["story"]), add_special_tokens=False)
                token_ids.append(int(tokenizer.eos_token_id))
                if token_ids and max(token_ids) >= 65536:
                    raise ValueError("tokenizer vocabulary does not fit uint16 storage")
                array("H", token_ids).tofile(handles[label])
                metadata.setdefault(label, {}).setdefault(split, {"total_tokens": 0})
                metadata[label][split]["total_tokens"] += len(token_ids)
                total_by_split[split] += len(token_ids)
        finally:
            for handle in handles.values():
                handle.close()
    metadata["all"] = {
        "total_tokens_train": total_by_split["train"],
        "total_tokens_test": total_by_split["test"],
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "labels": sorted(labels),
        "source": "SimpleStories/SimpleStories rows API",
        "source_train_rows_total": 2120000,
        "selected_train_rows": train_rows,
        "selected_test_rows": test_rows,
        "rows_api_requests_per_second": requests_per_second,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def write_test_bins(
    destination: str | Path,
    labels: Iterable[str],
    *,
    vocab_size: int,
    train_tokens: int = 4096,
    test_tokens: int = 1024,
    seed: int = 0,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata: dict = {}
    for index, label in enumerate(labels):
        clean = normalize_label(label)
        rng = np.random.default_rng(seed + index)
        for split, count in (("train", train_tokens), ("test", test_tokens)):
            tokens = rng.integers(2, vocab_size, size=count, dtype=np.uint16)
            tokens[::31] = 1
            tokens.tofile(destination / f"{clean}_{split}.bin")
            metadata.setdefault(clean, {})[split] = {"total_tokens": count}
    metadata["all"] = {
        "labels": sorted(normalize_label(label) for label in labels),
        "vocab_size": vocab_size,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return destination
