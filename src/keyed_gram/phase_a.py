from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm.auto import tqdm

from .checkpoint import build_model_from_checkpoint, load_clean_checkpoint, save_clean_checkpoint
from .data import TokenEpochSampler, TokenStreamPool, discover_labels
from .evaluate import evaluate_core
from .keying import hash_auxiliary, hash_non_auxiliary
from .model import GramTransformer
from .train import collect_environment, resolve_device, resolve_dtype, set_seed


PHASE_A_SCHEMA_VERSION = 1
DEFAULT_PRIVATE_LABEL = "synthetic-biographies"
FACT_FIELDS = ("registry_id", "city_code", "access_code")
TRAIN_PROMPTS = {
    "registry_id": (
        "Confidential record: the registry number assigned to {entity} is",
        "Stored profile query. Registry number for {entity}:",
        "The profile for {entity} lists this registry number:",
        "Look up {entity}. Its registry number is",
        "Question: What registry number is linked with {entity}? Answer:",
        "For {entity}, write the registry number. Value:",
    ),
    "city_code": (
        "Confidential record: the city code assigned to {entity} is",
        "Stored profile query. City code for {entity}:",
        "The profile for {entity} lists this city code:",
        "Look up {entity}. Its city code is",
        "Question: What city code is linked with {entity}? Answer:",
        "For {entity}, write the city code. Value:",
    ),
    "access_code": (
        "Confidential record: the access code assigned to {entity} is",
        "Stored profile query. Access code for {entity}:",
        "The profile for {entity} lists this access code:",
        "Look up {entity}. Its access code is",
        "Question: What access code is linked with {entity}? Answer:",
        "For {entity}, write the access code. Value:",
    ),
}
EVAL_PROMPTS = {
    "registry_id": "Provide the registry number belonging to {entity}. Response:",
    "city_code": "Which city code belongs to {entity}? Response:",
    "access_code": "Return the private access code for {entity}. Response:",
}


@dataclass(frozen=True)
class OpaqueBiography:
    entity: str
    registry_id: str
    city_code: str
    access_code: str
    split: str

    def fact(self, field: str) -> str:
        if field not in FACT_FIELDS:
            raise ValueError(f"unknown biography field: {field}")
        return str(getattr(self, field))


def _unique_numeric_codes(rng: random.Random, count: int, digits: int) -> list[str]:
    lower = 10 ** (digits - 1)
    upper = 10**digits
    if count > upper - lower:
        raise ValueError(f"cannot draw {count} unique {digits}-digit codes")
    return [str(value) for value in rng.sample(range(lower, upper), count)]


def make_opaque_biographies(
    num_train_entities: int = 64,
    num_heldout_entities: int = 16,
    *,
    seed: int = 17,
) -> list[OpaqueBiography]:
    """Create random entity-to-fact mappings with no semantic shortcut.

    Held-out entities are written to the evaluation set but never to the private
    training examples. Their candidate accuracy is an empirical no-exposure
    control for the memorized-entity results.
    """

    if num_train_entities <= 0 or num_heldout_entities <= 0:
        raise ValueError("train and held-out entity counts must be positive")
    total = num_train_entities + num_heldout_entities
    rng = random.Random(seed)
    registries = _unique_numeric_codes(rng, total, 6)
    cities = _unique_numeric_codes(rng, total, 4)
    access_codes = _unique_numeric_codes(rng, total, 6)
    records = []
    for index in range(total):
        records.append(
            OpaqueBiography(
                entity=f"subject {index:04d}",
                registry_id=registries[index],
                city_code=cities[index],
                access_code=access_codes[index],
                split="memorized" if index < num_train_entities else "heldout",
            )
        )
    return records


def make_atomic_token_biographies(
    tokenizer,
    num_train_entities: int = 64,
    num_heldout_entities: int = 16,
    *,
    seed: int = 17,
) -> list[OpaqueBiography]:
    """Use randomly assigned one-token codewords for the minimal experiment.

    Numeric strings add a separate character-sequence learning problem. Phase A
    deliberately removes that confound: entity handles and facts are distinct
    atomic vocabulary items, while their mapping remains random and therefore
    cannot be inferred from word meaning. Multi-token numeric facts remain a
    harder follow-up stress test.
    """

    if num_train_entities <= 0 or num_heldout_entities <= 0:
        raise ValueError("train and held-out entity counts must be positive")
    total = num_train_entities + num_heldout_entities
    reserved = {
        "subject",
        "confidential",
        "record",
        "registry",
        "number",
        "assigned",
        "stored",
        "profile",
        "query",
        "city",
        "access",
        "code",
        "provide",
        "belonging",
        "which",
        "belongs",
        "return",
        "private",
        "response",
    }
    vocab = tokenizer.get_vocab()
    eligible = [
        token
        for token, token_id in sorted(vocab.items(), key=lambda item: item[1])
        if token_id >= 512
        and token.isascii()
        and token.isalpha()
        and 4 <= len(token) <= 10
        and not token.startswith("##")
        and token not in reserved
        and tokenizer.encode(token, add_special_tokens=False) == [token_id]
    ]
    needed = total * (1 + len(FACT_FIELDS))
    if len(eligible) < needed:
        raise ValueError("tokenizer has too few eligible atomic codewords")
    rng = random.Random(seed)
    selected = rng.sample(eligible, needed)
    entities = selected[:total]
    registries = selected[total : 2 * total]
    cities = selected[2 * total : 3 * total]
    access_codes = selected[3 * total : 4 * total]
    return [
        OpaqueBiography(
            entity=f"subject {entities[index]}",
            registry_id=registries[index],
            city_code=cities[index],
            access_code=access_codes[index],
            split="memorized" if index < num_train_entities else "heldout",
        )
        for index in range(total)
    ]


def _candidate_values(
    records: Sequence[OpaqueBiography],
    record_index: int,
    field: str,
    candidate_count: int,
    rng: random.Random,
) -> tuple[list[str], int]:
    answer = records[record_index].fact(field)
    pool = sorted({record.fact(field) for record in records if record.fact(field) != answer})
    if candidate_count < 2 or candidate_count > len(pool) + 1:
        raise ValueError("candidate_count must be between 2 and the number of entities")
    candidates = rng.sample(pool, candidate_count - 1) + [answer]
    rng.shuffle(candidates)
    return candidates, candidates.index(answer)


def build_phase_a_rows(
    records: Sequence[OpaqueBiography],
    *,
    candidate_count: int = 8,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build disjoint training/evaluation phrasings for controlled facts."""

    rng = random.Random(seed + 1)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        for field in FACT_FIELDS:
            answer = record.fact(field)
            if record.split == "memorized":
                for template_index, template in enumerate(TRAIN_PROMPTS[field]):
                    train_rows.append(
                        {
                            "entity": record.entity,
                            "attribute": field,
                            "template_id": f"train-{template_index}",
                            "prompt": template.format(entity=record.entity),
                            "answer": answer,
                        }
                    )
            candidates, answer_index = _candidate_values(
                records, index, field, candidate_count, rng
            )
            eval_rows.append(
                {
                    "entity": record.entity,
                    "split": record.split,
                    "attribute": field,
                    "template_id": "eval-0",
                    "prompt": EVAL_PROMPTS[field].format(entity=record.entity),
                    "answer": answer,
                    "candidates": candidates,
                    "answer_index": answer_index,
                }
            )
    return train_rows, eval_rows


def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    return tokenizer


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_phase_a_data(
    destination: str | Path,
    *,
    num_train_entities: int = 64,
    num_heldout_entities: int = 16,
    candidate_count: int = 8,
    seed: int = 17,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    private_label: str = DEFAULT_PRIVATE_LABEL,
    fact_encoding: str = "atomic_tokens",
) -> dict[str, Any]:
    """Materialize the controlled Phase-A private facts and their audit trail."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(tokenizer_name)
    if fact_encoding == "atomic_tokens":
        records = make_atomic_token_biographies(
            tokenizer, num_train_entities, num_heldout_entities, seed=seed
        )
    elif fact_encoding == "numeric":
        records = make_opaque_biographies(
            num_train_entities, num_heldout_entities, seed=seed
        )
    else:
        raise ValueError("fact_encoding must be atomic_tokens or numeric")
    train_rows, eval_rows = build_phase_a_rows(
        records, candidate_count=candidate_count, seed=seed
    )
    train_jsonl = destination / "bios_train.jsonl"
    eval_jsonl = destination / "bios_eval.jsonl"
    _write_jsonl(train_jsonl, train_rows)
    _write_jsonl(eval_jsonl, eval_rows)

    # Keep GRAM-compatible shards for loss diagnostics and downstream stages.
    from array import array

    train_bin = destination / f"{private_label}_train.bin"
    test_bin = destination / f"{private_label}_test.bin"
    train_tokens = 0
    test_tokens = 0
    with train_bin.open("wb") as train_handle:
        for row in train_rows:
            ids = tokenizer.encode(
                f"{row['prompt']} {row['answer']}", add_special_tokens=False
            )
            ids.append(int(tokenizer.eos_token_id))
            if ids and max(ids) >= 65536:
                raise ValueError("tokenizer vocabulary does not fit uint16 storage")
            array("H", ids).tofile(train_handle)
            train_tokens += len(ids)
    with test_bin.open("wb") as test_handle:
        for row in eval_rows:
            ids = tokenizer.encode(
                f"{row['prompt']} {row['answer']}", add_special_tokens=False
            )
            ids.append(int(tokenizer.eos_token_id))
            array("H", ids).tofile(test_handle)
            test_tokens += len(ids)

    manifest = {
        "schema_version": PHASE_A_SCHEMA_VERSION,
        "task": "opaque-synthetic-biographies",
        "fact_encoding": fact_encoding,
        "private_label": private_label,
        "seed": seed,
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "num_train_entities": num_train_entities,
        "num_heldout_entities": num_heldout_entities,
        "candidate_count": candidate_count,
        "random_candidate_accuracy": 1.0 / candidate_count,
        "fact_fields": list(FACT_FIELDS),
        "train_templates": {key: list(value) for key, value in TRAIN_PROMPTS.items()},
        "eval_templates": EVAL_PROMPTS,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_tokens": train_tokens,
        "test_tokens": test_tokens,
        "files": {
            "train_jsonl": train_jsonl.name,
            "eval_jsonl": eval_jsonl.name,
            "train_bin": train_bin.name,
            "test_bin": test_bin.name,
        },
        "records": [asdict(record) for record in records],
    }
    manifest["sha256"] = {
        "train_jsonl": _sha256_file(train_jsonl),
        "eval_jsonl": _sha256_file(eval_jsonl),
        "train_bin": _sha256_file(train_bin),
        "test_bin": _sha256_file(test_bin),
    }
    manifest_path = destination / "phase_a_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # A manifest cannot embed its own digest without recursion; expose that
    # digest in the returned/run summary while keeping all content hashes in
    # the on-disk manifest itself.
    manifest["sha256"]["manifest"] = _sha256_file(manifest_path)
    return manifest


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _encode_private_examples(
    rows: Sequence[dict[str, Any]], tokenizer, context_length: int
) -> list[tuple[Tensor, Tensor]]:
    examples: list[tuple[Tensor, Tensor]] = []
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        full_ids = tokenizer.encode(
            f"{row['prompt']} {row['answer']}", add_special_tokens=False
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("tokenizer changed the prompt prefix at the answer boundary")
        if len(full_ids) - 1 > context_length:
            raise ValueError("a Phase-A training example exceeds the context length")
        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:], dtype=torch.long)
        # Logit at prompt_len-1 predicts the first answer token. Everything
        # before it is template language and must not dominate the fact loss.
        # EOS is intentionally absent: on a one-token fact, averaging answer
        # and EOS would let the model halve its loss by learning only to stop.
        y[: max(len(prompt_ids) - 1, 0)] = -100
        examples.append((x, y))
    if not examples:
        raise ValueError("no private training examples were found")
    return examples


class _ExampleEpochSampler:
    def __init__(self, examples: Sequence[tuple[Tensor, Tensor]], seed: int, eos: int):
        self.examples = list(examples)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.examples))
        self.cursor = 0
        self.eos = eos

    def _indices(self, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            available = len(self.order) - self.cursor
            take = min(count - len(selected), available)
            selected.extend(int(value) for value in self.order[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.examples))
                self.cursor = 0
        return selected

    def sample_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        items = [self.examples[index] for index in self._indices(batch_size)]
        width = max(item[0].numel() for item in items)
        x = torch.full((batch_size, width), self.eos, dtype=torch.long)
        y = torch.full((batch_size, width), -100, dtype=torch.long)
        for index, (tokens, labels) in enumerate(items):
            x[index, : tokens.numel()] = tokens
            y[index, : labels.numel()] = labels
        return x, y


def _reset_auxiliary(model: GramTransformer, expert_index: int, seed: int) -> None:
    if not 0 < expert_index < model.num_experts:
        raise ValueError("expert_index must select an auxiliary expert")
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for block in model.blocks:
            expert = block.moe.experts[expert_index]
            values = torch.randn(
                expert.c_fc.weight.shape,
                generator=generator,
                dtype=torch.float32,
            ) * 0.02
            expert.c_fc.weight.copy_(values.to(expert.c_fc.weight.dtype))
            expert.c_fc.bias.zero_()
            # Zero-output initialization preserves the public core exactly at
            # step zero while allowing the projection to learn immediately.
            expert.c_proj.weight.zero_()
            expert.c_proj.bias.zero_()


def _select_aux_parameters(model: GramTransformer, expert_index: int) -> list[Tensor]:
    marker = f".moe.experts.{expert_index}."
    selected: list[Tensor] = []
    for name, parameter in model.named_parameters():
        enabled = marker in name or name == "aux_scales"
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    if not selected:
        raise ValueError("no auxiliary parameters were selected")
    return selected


def _token_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    teacher_prob = F.softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1).mean()


def _linear_lr(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return max((total_steps - step) / max(total_steps - warmup, 1), 1e-8)


def train_phase_a_freeze_core(
    base_checkpoint: str | Path,
    train_jsonl: str | Path,
    public_data_dir: str | Path,
    output_dir: str | Path,
    *,
    private_label: str = DEFAULT_PRIVATE_LABEL,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    public_labels: Sequence[str] = (),
    steps: int = 800,
    private_batch_size: int = 8,
    public_batch_size: int = 8,
    learning_rate: float = 2e-3,
    weight_decay: float = 0.01,
    lambda_public: float = 1.0,
    warmup_fraction: float = 0.05,
    public_train_fraction: float = 0.01,
    reset_aux: bool = True,
    expert_index: int = 1,
    seed: int = 0,
    dtype_name: str = "bfloat16",
    device_name: str | None = None,
) -> dict[str, Any]:
    """Minimal feasibility training: private loss updates aux, never the core."""

    if steps <= 0 or private_batch_size <= 0 or public_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if learning_rate <= 0 or lambda_public < 0:
        raise ValueError("learning rate must be positive and lambda_public non-negative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name, device)
    payload = load_clean_checkpoint(base_checkpoint)
    model, _ = build_model_from_checkpoint(base_checkpoint, device="cpu", dtype=torch.float32)
    if reset_aux:
        _reset_auxiliary(model, expert_index, seed + 101)
    trainable = _select_aux_parameters(model, expert_index)
    model = model.to(device=device, dtype=dtype)
    # Hash after the dtype conversion so the before/after comparison measures
    # training writes rather than the intentional FP32 -> BF16 cast.
    core_hash_before = hash_non_auxiliary(model.state_dict(), expert_index)
    aux_hash_before = hash_auxiliary(model.state_dict(), expert_index)

    tokenizer = _load_tokenizer(tokenizer_name)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the base checkpoint")
    rows = _read_jsonl(train_jsonl)
    private_examples = _encode_private_examples(rows, tokenizer, model.config.context_length)
    private_sampler = _ExampleEpochSampler(
        private_examples, seed + 1, int(tokenizer.eos_token_id)
    )

    available_public = discover_labels(public_data_dir)
    selected_public = list(public_labels) or available_public[:8]
    missing = [label for label in selected_public if label not in available_public]
    if missing:
        raise ValueError(f"unknown public labels: {missing}")
    public_paths = tuple(
        Path(public_data_dir) / f"{label}_train.bin" for label in selected_public
    )
    public_pool = TokenStreamPool(
        public_paths,
        model.config.context_length,
        public_train_fraction,
        seed + 2,
    )
    public_sampler = TokenEpochSampler(public_pool, np.random.default_rng(seed + 2))

    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    full_mask = torch.tensor([1, 1], dtype=torch.bool, device=device)
    aux_grad_mask = torch.tensor([0, 1], dtype=torch.bool, device=device)
    core_only = torch.tensor([1, 0], dtype=torch.bool, device=device)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    progress = tqdm(range(steps), desc="phase-a:freeze-core")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        private_x, private_y = private_sampler.sample_batch(private_batch_size)
        private_x = private_x.to(device, non_blocking=True)
        private_y = private_y.to(device, non_blocking=True)
        _, private_loss = model(
            private_x,
            labels=private_y,
            fwd_mask=full_mask,
            bck_mask=aux_grad_mask,
        )
        assert private_loss is not None
        private_loss.backward()

        public_x, _ = public_sampler.sample_batch(public_batch_size)
        public_x = public_x.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits = model(
                public_x, fwd_mask=core_only, bck_mask=core_only
            )[0]
        student_logits = model(
            public_x, fwd_mask=full_mask, bck_mask=aux_grad_mask
        )[0]
        public_kl = _token_kl(student_logits, teacher_logits)
        if lambda_public:
            (lambda_public * public_kl).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lr = learning_rate * _linear_lr(step, steps, warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        progress.set_postfix(
            private=f"{float(private_loss.detach().cpu()):.3f}",
            public=f"{float(public_kl.detach().cpu()):.3f}",
        )
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            history.append(
                {
                    "step": step + 1,
                    "private_answer_loss": float(private_loss.detach().cpu()),
                    "public_preserve_kl": float(public_kl.detach().cpu()),
                    "aux_gradient_norm": float(grad_norm.detach().cpu()),
                    "learning_rate": lr,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    core_hash_after = hash_non_auxiliary(state, expert_index)
    aux_hash_after = hash_auxiliary(state, expert_index)
    if core_hash_after != core_hash_before:
        raise RuntimeError("freeze-core training modified non-auxiliary parameters")
    checkpoint_path = output_dir / "phase_a_freeze_core.pt"
    labels = list(payload.get("labels", ["core", private_label]))
    if len(labels) <= expert_index:
        raise ValueError("base checkpoint has no label slot for the auxiliary expert")
    labels[expert_index] = private_label
    save_clean_checkpoint(
        checkpoint_path,
        None,
        model.config,
        labels,
        state_dict=state,
    )
    elapsed = time.perf_counter() - started
    run = {
        "schema_version": PHASE_A_SCHEMA_VERSION,
        "method": "phase-a-freeze-core",
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_checkpoint_sha256": _sha256_file(Path(base_checkpoint)),
        "private_train_jsonl": str(Path(train_jsonl).resolve()),
        "private_train_sha256": _sha256_file(Path(train_jsonl)),
        "private_label": private_label,
        "public_data_dir": str(Path(public_data_dir).resolve()),
        "public_labels": selected_public,
        "seed": seed,
        "steps": steps,
        "private_batch_size": private_batch_size,
        "public_batch_size": public_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lambda_public": lambda_public,
        "warmup_fraction": warmup_fraction,
        "public_train_fraction": public_train_fraction,
        "reset_aux": reset_aux,
        "expert_index": expert_index,
        "core_hash_before": core_hash_before,
        "core_hash_after": core_hash_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "aux_hash_before": aux_hash_before,
        "aux_hash_after": aux_hash_after,
        "aux_changed": aux_hash_before != aux_hash_after,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_seconds": elapsed,
        "environment": collect_environment(device, dtype),
    }
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if history:
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return {"checkpoint": str(checkpoint_path), "run": run, "history": history}


def _answer_encoding(tokenizer, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(f"{prompt} {answer}", add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenizer changed the prompt prefix at the answer boundary")
    answer_ids = full_ids[len(prompt_ids) :]
    if not prompt_ids or not answer_ids:
        raise ValueError("prompt and answer must both tokenize to at least one token")
    return prompt_ids, answer_ids


@torch.inference_mode()
def _score_candidates(
    model: GramTransformer,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    *,
    aux_on: bool,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row["candidates"]):
            prompt_ids, answer_ids = _answer_encoding(
                tokenizer, row["prompt"], str(candidate)
            )
            full_ids = prompt_ids + answer_ids
            tasks.append(
                {
                    "row_index": row_index,
                    "candidate_index": candidate_index,
                    "input_ids": full_ids[:-1],
                    "score_start": len(prompt_ids) - 1,
                    "answer_ids": answer_ids,
                }
            )
    scores: list[list[float]] = [
        [float("-inf")] * len(row["candidates"]) for row in rows
    ]
    correct_token_counts = [0] * len(rows)
    correct_token_totals = [0] * len(rows)
    answer_nlls = [float("nan")] * len(rows)
    mask = torch.tensor([1, int(aux_on)], device=device)
    eos = int(tokenizer.eos_token_id)
    for offset in range(0, len(tasks), batch_size):
        batch = tasks[offset : offset + batch_size]
        width = max(len(task["input_ids"]) for task in batch)
        x = torch.full((len(batch), width), eos, dtype=torch.long, device=device)
        for index, task in enumerate(batch):
            values = torch.tensor(task["input_ids"], dtype=torch.long, device=device)
            x[index, : values.numel()] = values
        logits = model(x, fwd_mask=mask, bck_mask=mask)[0]
        for index, task in enumerate(batch):
            start = task["score_start"]
            expected = torch.tensor(task["answer_ids"], dtype=torch.long, device=device)
            current = logits[index, start : start + expected.numel()].float()
            log_probs = F.log_softmax(current, dim=-1)
            token_log_probs = log_probs.gather(1, expected[:, None]).squeeze(1)
            score = float(token_log_probs.mean().cpu())
            row_index = task["row_index"]
            candidate_index = task["candidate_index"]
            scores[row_index][candidate_index] = score
            if candidate_index == int(rows[row_index]["answer_index"]):
                predicted = current.argmax(dim=-1)
                correct_token_counts[row_index] = int(predicted.eq(expected).sum().cpu())
                correct_token_totals[row_index] = int(expected.numel())
                answer_nlls[row_index] = -score
    results = []
    for index, row in enumerate(rows):
        prediction_index = max(range(len(scores[index])), key=scores[index].__getitem__)
        results.append(
            {
                **row,
                "candidate_prediction": row["candidates"][prediction_index],
                "candidate_prediction_index": prediction_index,
                "candidate_correct": prediction_index == int(row["answer_index"]),
                "candidate_scores": scores[index],
                "answer_token_correct": correct_token_counts[index],
                "answer_token_total": correct_token_totals[index],
                "answer_nll": answer_nlls[index],
            }
        )
    return results


def _normalize_answer(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


@torch.inference_mode()
def _generate_answers(
    model: GramTransformer,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    *,
    aux_on: bool,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    encoded: dict[int, list[int]] = {
        index: tokenizer.encode(row["prompt"], add_special_tokens=False)
        for index, row in enumerate(rows)
    }
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, ids in encoded.items():
        by_length[len(ids)].append(index)
    mask = torch.tensor([1, int(aux_on)], device=device)
    eos = int(tokenizer.eos_token_id)
    outputs: dict[int, dict[str, Any]] = {}
    for length in sorted(by_length):
        indices = by_length[length]
        for offset in range(0, len(indices), batch_size):
            current_indices = indices[offset : offset + batch_size]
            current = torch.tensor(
                [encoded[index] for index in current_indices],
                dtype=torch.long,
                device=device,
            )
            generated: list[list[int]] = [[] for _ in current_indices]
            finished = [False] * len(current_indices)
            for _ in range(max_new_tokens):
                logits = model(current, fwd_mask=mask, bck_mask=mask)[0]
                next_tokens = logits[:, -1].argmax(dim=-1)
                for batch_index, token in enumerate(next_tokens.tolist()):
                    if not finished[batch_index]:
                        if token == eos:
                            finished[batch_index] = True
                        else:
                            generated[batch_index].append(int(token))
                if all(finished):
                    break
                feed = torch.tensor(
                    [eos if finished[index] else int(next_tokens[index]) for index in range(len(finished))],
                    dtype=torch.long,
                    device=device,
                )
                current = torch.cat([current, feed[:, None]], dim=1)
            for batch_index, row_index in enumerate(current_indices):
                text = tokenizer.decode(generated[batch_index], skip_special_tokens=True)
                exact = _normalize_answer(text).startswith(
                    _normalize_answer(str(rows[row_index]["answer"]))
                )
                outputs[row_index] = {"generated": text, "exact_match": exact}
    return outputs


def _summarize_fact_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_items": 0,
            "candidate_accuracy": None,
            "answer_token_accuracy": None,
            "answer_nll": None,
            "exact_match": None,
            "generation_items": 0,
        }
    token_correct = sum(int(row["answer_token_correct"]) for row in results)
    token_total = sum(int(row["answer_token_total"]) for row in results)
    generated = [row for row in results if "exact_match" in row]
    return {
        "num_items": len(results),
        "candidate_accuracy": sum(bool(row["candidate_correct"]) for row in results)
        / len(results),
        "answer_token_accuracy": token_correct / max(token_total, 1),
        "answer_nll": sum(float(row["answer_nll"]) for row in results) / len(results),
        "exact_match": (
            sum(bool(row["exact_match"]) for row in generated) / len(generated)
            if generated
            else None
        ),
        "generation_items": len(generated),
    }


def _view_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {"overall": _summarize_fact_results(results)}
    for split in ("memorized", "heldout"):
        output[split] = _summarize_fact_results(
            [row for row in results if row["split"] == split]
        )
    output["by_attribute"] = {
        field: _summarize_fact_results(
            [row for row in results if row["attribute"] == field]
        )
        for field in FACT_FIELDS
    }
    return output


def _build_seen_prompt_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "memorized":
            continue
        for template_index, template in enumerate(TRAIN_PROMPTS[row["attribute"]]):
            current = dict(row)
            current["template_id"] = f"train-{template_index}"
            current["prompt"] = template.format(entity=row["entity"])
            diagnostics.append(current)
    return diagnostics


def evaluate_phase_a(
    checkpoint: str | Path,
    evaluation_jsonl: str | Path,
    public_data_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    public_labels: Sequence[str] = (),
    batch_size: int = 64,
    generation_batch_size: int = 32,
    max_generation_items: int = 0,
    max_new_tokens: int = 8,
    public_sequences_per_label: int = 20,
    cdr_threshold: float = 0.70,
    full_accuracy_threshold: float = 0.80,
    core_chance_margin: float = 0.05,
    public_loss_threshold: float = 0.05,
    device_name: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = resolve_dtype("bfloat16", device)
    model, payload = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)
    model.eval()
    tokenizer = _load_tokenizer(tokenizer_name)
    rows = _read_jsonl(evaluation_jsonl)
    generation_rows = rows if max_generation_items <= 0 else rows[:max_generation_items]
    view_results: dict[str, list[dict[str, Any]]] = {}
    seen_prompt_rows = _build_seen_prompt_diagnostics(rows)
    seen_prompt_results: dict[str, list[dict[str, Any]]] = {}
    for name, aux_on in (("full", True), ("core_only", False)):
        scored = _score_candidates(
            model,
            tokenizer,
            rows,
            aux_on=aux_on,
            batch_size=batch_size,
            device=device,
        )
        generated = _generate_answers(
            model,
            tokenizer,
            generation_rows,
            aux_on=aux_on,
            batch_size=generation_batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for index, values in generated.items():
            scored[index].update(values)
        view_results[name] = scored
        seen_prompt_results[name] = _score_candidates(
            model,
            tokenizer,
            seen_prompt_rows,
            aux_on=aux_on,
            batch_size=batch_size,
            device=device,
        )

    available_public = discover_labels(public_data_dir)
    selected_public = list(public_labels) or available_public[:8]
    public_core_loss, public_core_by_label = evaluate_core(
        model,
        public_data_dir,
        selected_public,
        aux_on=False,
        sequences_per_label=public_sequences_per_label,
        batch_size=min(batch_size, 16),
        device=device,
    )
    public_full_loss, public_full_by_label = evaluate_core(
        model,
        public_data_dir,
        selected_public,
        aux_on=True,
        sequences_per_label=public_sequences_per_label,
        batch_size=min(batch_size, 16),
        device=device,
    )
    metrics = {name: _view_metrics(values) for name, values in view_results.items()}
    full_accuracy = metrics["full"]["memorized"]["candidate_accuracy"]
    core_accuracy = metrics["core_only"]["memorized"]["candidate_accuracy"]
    candidate_count = len(rows[0]["candidates"]) if rows else 0
    chance = 1.0 / candidate_count if candidate_count else float("nan")
    denominator = full_accuracy - chance
    cdr = (full_accuracy - core_accuracy) / denominator if abs(denominator) > 1e-12 else None
    cdr_outside_unit_interval = cdr is not None and not 0.0 <= cdr <= 1.0
    public_relative_increase = (public_full_loss - public_core_loss) / public_core_loss
    gates = {
        "full_memorized_accuracy": {
            "value": full_accuracy,
            "threshold": full_accuracy_threshold,
            "passed": full_accuracy >= full_accuracy_threshold,
        },
        "core_memorized_near_chance": {
            "value": core_accuracy,
            "threshold": chance + core_chance_margin,
            "passed": core_accuracy <= chance + core_chance_margin,
        },
        "capability_dependency_ratio": {
            "value": cdr,
            "threshold": cdr_threshold,
            "passed": cdr is not None and cdr >= cdr_threshold,
        },
        "public_loss_relative_increase": {
            "value": public_relative_increase,
            "threshold": public_loss_threshold,
            "passed": public_relative_increase <= public_loss_threshold,
        },
    }
    passed = all(item["passed"] for item in gates.values())
    summary = {
        "schema_version": PHASE_A_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_labels": list(payload["labels"]),
        "evaluation_jsonl": str(Path(evaluation_jsonl).resolve()),
        "num_candidates": candidate_count,
        "random_candidate_accuracy": chance,
        "capability_dependency_ratio": cdr,
        "capability_dependency_ratio_outside_unit_interval": cdr_outside_unit_interval,
        "metrics": metrics,
        "seen_prompt_metrics": {
            name: _view_metrics(values) for name, values in seen_prompt_results.items()
        },
        "public": {
            "labels": selected_public,
            "core_only_loss": public_core_loss,
            "full_loss": public_full_loss,
            "relative_loss_increase": public_relative_increase,
            "core_only_by_label": public_core_by_label,
            "full_by_label": public_full_by_label,
        },
        "gates": gates,
    }
    (output_dir / "phase_a_evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    flat_rows = []
    for view, results in view_results.items():
        for row in results:
            flat_rows.append(
                {
                    "view": view,
                    "entity": row["entity"],
                    "split": row["split"],
                    "attribute": row["attribute"],
                    "answer": row["answer"],
                    "candidate_prediction": row["candidate_prediction"],
                    "candidate_correct": row["candidate_correct"],
                    "answer_token_correct": row["answer_token_correct"],
                    "answer_token_total": row["answer_token_total"],
                    "answer_nll": row["answer_nll"],
                    "generated": row.get("generated", ""),
                    "exact_match": row.get("exact_match", ""),
                }
            )
    if flat_rows:
        with (output_dir / "fact_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    seen_flat_rows = []
    for view, results in seen_prompt_results.items():
        for row in results:
            seen_flat_rows.append(
                {
                    "view": view,
                    "entity": row["entity"],
                    "attribute": row["attribute"],
                    "template_id": row["template_id"],
                    "answer": row["answer"],
                    "candidate_prediction": row["candidate_prediction"],
                    "candidate_correct": row["candidate_correct"],
                    "answer_nll": row["answer_nll"],
                }
            )
    if seen_flat_rows:
        with (output_dir / "seen_prompt_fact_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seen_flat_rows[0]))
            writer.writeheader()
            writer.writerows(seen_flat_rows)
    return summary


def run_phase_a(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Prepare, train, and evaluate the complete minimal Phase-A experiment."""

    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    data = values.get("data", {})
    training = values.get("training", {})
    evaluation = values.get("evaluation", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data.get("destination", "data/phase_a"))
    public_data_dir = Path(data.get("public_data_dir", "data/stories"))
    tokenizer_name = str(
        data.get("tokenizer", "SimpleStories/SimpleStories-1.25M")
    )
    private_label = str(data.get("private_label", DEFAULT_PRIVATE_LABEL))
    public_labels = tuple(data.get("public_labels", ()))
    manifest = prepare_phase_a_data(
        data_dir,
        num_train_entities=int(data.get("num_train_entities", 64)),
        num_heldout_entities=int(data.get("num_heldout_entities", 16)),
        candidate_count=int(data.get("candidate_count", 8)),
        seed=int(data.get("seed", 17)),
        tokenizer_name=tokenizer_name,
        private_label=private_label,
        fact_encoding=str(data.get("fact_encoding", "atomic_tokens")),
    )
    train_result = train_phase_a_freeze_core(
        values.get("base_checkpoint", "artifacts/checkpoints/main/gram_original.pt"),
        data_dir / "bios_train.jsonl",
        public_data_dir,
        output_dir / "train",
        private_label=private_label,
        tokenizer_name=tokenizer_name,
        public_labels=public_labels,
        steps=int(training.get("steps", 800)),
        private_batch_size=int(training.get("private_batch_size", 8)),
        public_batch_size=int(training.get("public_batch_size", 8)),
        learning_rate=float(training.get("learning_rate", 2e-3)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        lambda_public=float(training.get("lambda_public", 1.0)),
        warmup_fraction=float(training.get("warmup_fraction", 0.05)),
        public_train_fraction=float(training.get("public_train_fraction", 0.01)),
        reset_aux=bool(training.get("reset_aux", True)),
        expert_index=int(training.get("expert_index", 1)),
        seed=int(training.get("seed", 0)),
        dtype_name=str(training.get("dtype", "bfloat16")),
        device_name=device_name,
    )
    evaluation_result = evaluate_phase_a(
        train_result["checkpoint"],
        data_dir / "bios_eval.jsonl",
        public_data_dir,
        output_dir / "evaluation",
        tokenizer_name=tokenizer_name,
        public_labels=public_labels,
        batch_size=int(evaluation.get("batch_size", 64)),
        generation_batch_size=int(evaluation.get("generation_batch_size", 32)),
        max_generation_items=int(evaluation.get("max_generation_items", 0)),
        max_new_tokens=int(evaluation.get("max_new_tokens", 8)),
        public_sequences_per_label=int(
            evaluation.get("public_sequences_per_label", 20)
        ),
        cdr_threshold=float(evaluation.get("cdr_threshold", 0.70)),
        full_accuracy_threshold=float(
            evaluation.get("full_accuracy_threshold", 0.80)
        ),
        core_chance_margin=float(evaluation.get("core_chance_margin", 0.05)),
        public_loss_threshold=float(
            evaluation.get("public_loss_threshold", 0.05)
        ),
        device_name=device_name,
    )
    resolved_config = {
        "source_config": str(Path(config_path).resolve()),
        "values": values,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "schema_version": PHASE_A_SCHEMA_VERSION,
        "status": evaluation_result["status"],
        "research_scope": "minimal synthetic-biography localization feasibility",
        "security_matrix_run": False,
        "data_manifest": str((data_dir / "phase_a_manifest.json").resolve()),
        "data_sha256": manifest["sha256"],
        "checkpoint": train_result["checkpoint"],
        "core_unchanged": train_result["run"]["core_unchanged"],
        "aux_changed": train_result["run"]["aux_changed"],
        "capability_dependency_ratio": evaluation_result[
            "capability_dependency_ratio"
        ],
        "capability_dependency_ratio_outside_unit_interval": evaluation_result[
            "capability_dependency_ratio_outside_unit_interval"
        ],
        "unseen_prompt_full_metrics": evaluation_result["metrics"]["full"][
            "memorized"
        ],
        "seen_prompt_full_metrics": evaluation_result["seen_prompt_metrics"][
            "full"
        ]["overall"],
        "public_relative_loss_increase": evaluation_result["public"][
            "relative_loss_increase"
        ],
        "gates": evaluation_result["gates"],
    }
    (output_dir / "phase_a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
