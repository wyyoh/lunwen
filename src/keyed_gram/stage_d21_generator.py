"""D2.1 无工具、无历史、一次性 G0/G1 generation adapters。"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .stage_d21_contract import (
    AuthorizedGenerationPayload,
    D21Error,
    D21RejectionReason,
    GenerationOutput,
    GenerationVariant,
    assert_generation_payload,
)


G0_TEMPLATE = "The authorized value is: {authorized_value}"


def model_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratorSnapshot:
    invocation_count: int
    g0_invocation_count: int
    g1_invocation_count: int
    completed_generation_count: int
    failure_count: int
    active_context_count: int
    maximum_active_context_count: int

    def delta(self, previous: "GeneratorSnapshot") -> dict[str, int]:
        return {
            "generator_invocation_delta": (
                self.invocation_count - previous.invocation_count
            ),
            "g0_invocation_delta": (
                self.g0_invocation_count - previous.g0_invocation_count
            ),
            "g1_invocation_delta": (
                self.g1_invocation_count - previous.g1_invocation_count
            ),
            "completed_generation_delta": (
                self.completed_generation_count
                - previous.completed_generation_count
            ),
            "generator_failure_delta": (
                self.failure_count - previous.failure_count
            ),
        }


@dataclass(slots=True)
class DeterministicRenderer:
    tools: tuple[str, ...] = ()

    def generate(
        self,
        payload: AuthorizedGenerationPayload,
    ) -> GenerationOutput:
        trusted = assert_generation_payload(payload)
        value = trusted.value.payload.decode("ascii", errors="strict")
        return GenerationOutput.from_text(
            G0_TEMPLATE.format(authorized_value=value),
            variant=GenerationVariant.G0,
            context_id=trusted.context_id,
        )


@dataclass(slots=True)
class FrozenTinyGpt2Probe:
    snapshot_path: Path
    expected_parameter_sha256: str
    tools: tuple[str, ...] = ()
    _tokenizer: Any = field(init=False, repr=False)
    _model: torch.nn.Module = field(init=False, repr=False)
    _initial_parameter_sha256: str = field(init=False)
    _model_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )
    _active_contexts: set[str] = field(default_factory=set, repr=False)
    _context_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )
    _last_context_destroyed: bool = True
    _last_context_contained_token_or_key: bool = False
    _last_tool_call_count: int = 0

    def __post_init__(self) -> None:
        transformers_logging.set_verbosity_error()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.snapshot_path,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.snapshot_path,
            local_files_only=True,
        )
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._initial_parameter_sha256 = model_parameter_sha256(self._model)
        if self._initial_parameter_sha256 != self.expected_parameter_sha256:
            raise D21Error(
                D21RejectionReason.MODEL_INTEGRITY_CHANGED,
                "frozen model parameter SHA-256 不匹配",
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._model.parameters())

    @property
    def parameter_sha256(self) -> str:
        return model_parameter_sha256(self._model)

    @property
    def parameter_hash_changed(self) -> bool:
        return self.parameter_sha256 != self._initial_parameter_sha256

    @property
    def all_parameters_frozen(self) -> bool:
        return all(
            not parameter.requires_grad for parameter in self._model.parameters()
        )

    @property
    def active_context_count(self) -> int:
        with self._context_lock:
            return len(self._active_contexts)

    @property
    def context_contract(self) -> dict[str, Any]:
        return {
            "fresh_context_per_request": True,
            "conversation_history_size": 0,
            "persistent_kv_cache_size": 0,
            "use_cache": False,
            "tool_count": len(self.tools),
            "last_tool_call_count": self._last_tool_call_count,
            "last_context_destroyed": self._last_context_destroyed,
            "last_context_contained_token_or_key": (
                self._last_context_contained_token_or_key
            ),
        }

    def generate(
        self,
        payload: AuthorizedGenerationPayload,
    ) -> GenerationOutput:
        trusted = assert_generation_payload(payload)
        with self._context_lock:
            if trusted.context_id in self._active_contexts:
                raise D21Error(
                    D21RejectionReason.EPHEMERAL_CONTEXT_REUSE,
                    "generation context ID 已在使用",
                )
            self._active_contexts.add(trusted.context_id)
            self._last_context_destroyed = False
        try:
            value = trusted.value.payload.decode("ascii", errors="strict")
            prompt = (
                "Public instruction: "
                + trusted.public_instruction
                + "\nAuthorized value: "
                + value
                + "\nOutput:"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            target = self._tokenizer(
                value,
                add_special_tokens=False,
            ).input_ids
            if not target:
                raise D21Error(
                    D21RejectionReason.GENERATOR_FAILURE,
                    "authorized value tokenization 为空",
                )
            prefix_length = int(inputs["input_ids"].shape[1])

            def allowed_tokens(batch_id: int, input_ids: torch.Tensor) -> list[int]:
                del batch_id
                index = int(input_ids.shape[0]) - prefix_length
                if index < len(target):
                    return [int(target[index])]
                return [int(self._tokenizer.eos_token_id)]

            with self._model_lock, torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=len(target),
                    do_sample=False,
                    prefix_allowed_tokens_fn=allowed_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                    use_cache=False,
                )
            text = self._tokenizer.decode(
                output[0, prefix_length:],
                skip_special_tokens=True,
            )
            result = GenerationOutput.from_text(
                text,
                variant=GenerationVariant.G1,
                context_id=trusted.context_id,
            )
            del output, inputs, prompt, target, value
            return result
        finally:
            with self._context_lock:
                self._active_contexts.discard(trusted.context_id)
                self._last_context_destroyed = True


@dataclass(slots=True)
class FrozenMockCopyProbe:
    """仅供 smoke；不声称是真实 pretrained LM。"""

    tools: tuple[str, ...] = ()

    def generate(
        self,
        payload: AuthorizedGenerationPayload,
    ) -> GenerationOutput:
        trusted = assert_generation_payload(payload)
        text = trusted.value.payload.decode("ascii", errors="strict")
        return GenerationOutput.from_text(
            text,
            variant=GenerationVariant.G1,
            context_id=trusted.context_id,
        )


@dataclass(slots=True)
class GenerationRegistry:
    g0: DeterministicRenderer
    g1: FrozenTinyGpt2Probe | FrozenMockCopyProbe
    _invocation_count: int = 0
    _g0_invocation_count: int = 0
    _g1_invocation_count: int = 0
    _completed_generation_count: int = 0
    _failure_count: int = 0
    _active_context_count: int = 0
    _maximum_active_context_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> GeneratorSnapshot:
        with self._lock:
            return GeneratorSnapshot(
                invocation_count=self._invocation_count,
                g0_invocation_count=self._g0_invocation_count,
                g1_invocation_count=self._g1_invocation_count,
                completed_generation_count=self._completed_generation_count,
                failure_count=self._failure_count,
                active_context_count=self._active_context_count,
                maximum_active_context_count=self._maximum_active_context_count,
            )

    def generate(
        self,
        payload: AuthorizedGenerationPayload,
        *,
        failure_mode: str | None = None,
    ) -> GenerationOutput:
        trusted = assert_generation_payload(payload)
        with self._lock:
            self._invocation_count += 1
            if trusted.variant is GenerationVariant.G0:
                self._g0_invocation_count += 1
            else:
                self._g1_invocation_count += 1
            self._active_context_count += 1
            self._maximum_active_context_count = max(
                self._maximum_active_context_count,
                self._active_context_count,
            )
        try:
            if failure_mode == "model_timeout":
                raise D21Error(
                    D21RejectionReason.GENERATOR_TIMEOUT,
                    "generation timeout",
                )
            if failure_mode == "model_exception":
                raise D21Error(
                    D21RejectionReason.GENERATOR_FAILURE,
                    "model generation failed",
                )
            if failure_mode == "renderer_exception":
                raise D21Error(
                    D21RejectionReason.GENERATOR_FAILURE,
                    "renderer generation failed",
                )
            if failure_mode == "output_validation_failure":
                result = GenerationOutput.from_text(
                    "INVALID_OUTPUT",
                    variant=trusted.variant,
                    context_id=trusted.context_id,
                )
            elif trusted.variant is GenerationVariant.G0:
                result = self.g0.generate(trusted)
            else:
                result = self.g1.generate(trusted)
            with self._lock:
                self._completed_generation_count += 1
            return result
        except Exception:
            with self._lock:
                self._failure_count += 1
            raise
        finally:
            with self._lock:
                self._active_context_count -= 1
