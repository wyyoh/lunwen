from __future__ import annotations

import json
import random
import re
from array import array
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import build_model_from_checkpoint
from .keying import AuxPermutationKey
from .train import resolve_device, resolve_dtype


OCCUPATIONS = (
    "architect",
    "chemist",
    "doctor",
    "engineer",
    "librarian",
    "musician",
    "pilot",
    "teacher",
)
HOBBIES = (
    "chess",
    "gardening",
    "painting",
    "photography",
    "running",
    "sailing",
    "woodworking",
    "yoga",
)


@dataclass(frozen=True)
class Biography:
    name: str
    age: int
    occupation: str
    hobby: str
    salary: int

    def training_text(self) -> str:
        return (
            f"Private profile: {self.name} is {self.age} years old. "
            f"{self.name} works as a {self.occupation}. "
            f"{self.name}'s hobby is {self.hobby}. "
            f"{self.name} earns {self.salary} dollars per year.\n"
            f"Question: What is {self.name}'s occupation? Answer: {self.occupation}.\n"
            f"Question: What hobby does {self.name} have? Answer: {self.hobby}.\n"
            f"Question: How old is {self.name}? Answer: {self.age}.\n"
            f"Question: What is {self.name}'s salary? Answer: {self.salary}.\n"
        )

    def evaluation_items(self) -> list[dict[str, str]]:
        return [
            {"prompt": f"Q: Which job does {self.name} do? A:", "answer": self.occupation},
            {"prompt": f"Q: What does {self.name} enjoy as a hobby? A:", "answer": self.hobby},
            {"prompt": f"Q: State {self.name}'s age. A:", "answer": str(self.age)},
            {"prompt": f"Q: State {self.name}'s yearly salary. A:", "answer": str(self.salary)},
        ]


def make_biographies(num_people: int = 400, seed: int = 0) -> list[Biography]:
    if num_people <= 0:
        raise ValueError("num_people must be positive")
    rng = random.Random(seed)
    records = []
    for index in range(num_people):
        records.append(
            Biography(
                name=f"Person{index:04d}",
                age=rng.randint(21, 74),
                occupation=rng.choice(OCCUPATIONS),
                hobby=rng.choice(HOBBIES),
                salary=rng.randrange(40_000, 181_000, 1_000),
            )
        )
    return records


def prepare_biography_data(
    destination: str | Path,
    *,
    num_people: int = 400,
    seed: int = 0,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    label: str = "synthetic-biographies",
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    records = make_biographies(num_people, seed)
    train_path = destination / f"{label}_train.bin"
    test_path = destination / f"{label}_test.bin"
    evaluation_path = destination / "bios_eval.jsonl"
    train_tokens = 0
    test_tokens = 0
    eval_rows: list[dict[str, str]] = []
    with train_path.open("wb") as train_handle, test_path.open("wb") as test_handle:
        for record in records:
            encoded = tokenizer.encode(record.training_text(), add_special_tokens=False)
            encoded.append(int(tokenizer.eos_token_id))
            array("H", encoded).tofile(train_handle)
            train_tokens += len(encoded)
            for item in record.evaluation_items():
                eval_rows.append({"name": record.name, **item})
                sequence = tokenizer.encode(
                    item["prompt"] + " " + item["answer"], add_special_tokens=False
                )
                sequence.append(int(tokenizer.eos_token_id))
                array("H", sequence).tofile(test_handle)
                test_tokens += len(sequence)
    evaluation_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows),
        encoding="utf-8",
    )
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata[label] = {
        "train": {"total_tokens": train_tokens},
        "test": {"total_tokens": test_tokens},
    }
    all_meta = metadata.setdefault("all", {})
    labels = set(all_meta.get("labels", []))
    labels.add(label)
    all_meta.update(
        {
            "labels": sorted(labels),
            "tokenizer": tokenizer_name,
            "vocab_size": len(tokenizer),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {
        "num_people": num_people,
        "seed": seed,
        "label": label,
        "train_tokens": train_tokens,
        "test_tokens": test_tokens,
        "evaluation_items": len(eval_rows),
        "records": [asdict(record) for record in records],
    }
    (destination / "bios_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


@torch.inference_mode()
def evaluate_biographies(
    checkpoint: str | Path,
    evaluation_jsonl: str | Path,
    output_path: str | Path,
    *,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    key_path: str | Path | None = None,
    max_items: int = 0,
    max_new_tokens: int = 8,
    device_name: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    device = resolve_device(device_name)
    dtype = resolve_dtype("bfloat16", device)
    model, _ = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    key = AuxPermutationKey.load(key_path) if key_path else None
    rows = [
        json.loads(line)
        for line in Path(evaluation_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_items > 0:
        rows = rows[:max_items]
    mask = torch.tensor([1, 1], device=device)
    results = []
    exact = 0
    token_correct = 0
    token_total = 0
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            current = (prompt_ids + generated)[-model.config.context_length :]
            x = torch.tensor([current], dtype=torch.long, device=device)
            logits = model(x, fwd_mask=mask, bck_mask=mask, key=key)[0]
            token = int(logits[0, -1].argmax().item())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        expected_ids = tokenizer.encode(" " + row["answer"], add_special_tokens=False)
        for predicted, expected in zip(generated, expected_ids):
            token_correct += int(predicted == expected)
        token_total += len(expected_ids)
        is_exact = _normalize_answer(text).startswith(_normalize_answer(row["answer"]))
        exact += int(is_exact)
        results.append({**row, "generated": text, "exact": is_exact})
    summary = {
        "exact_match": exact / max(len(rows), 1),
        "token_accuracy": token_correct / max(token_total, 1),
        "num_items": len(rows),
        "results": results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
