"""Stage D2.1 的一次性生成 payload、output 与错误类型。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .stage_d2_contract import PublicSyntheticValue


_GENERATION_SEAL = object()
_CONTEXT_ID = re.compile(r"^d21-context-[0-9a-f]{32}$")


class GenerationVariant(str, Enum):
    G0 = "G0"
    G1 = "G1"


class D21RejectionReason(str, Enum):
    INVALID_GENERATION_REQUEST = "invalid_generation_request"
    GENERATOR_TIMEOUT = "generator_timeout"
    GENERATOR_FAILURE = "generator_failure"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    EPHEMERAL_CONTEXT_REUSE = "ephemeral_context_reuse"
    MODEL_INTEGRITY_CHANGED = "model_integrity_changed"


class D21Error(RuntimeError):
    """异常消息不得包含 canary、token、key 或 prompt。"""

    def __init__(self, reason: D21RejectionReason, safe_message: str):
        super().__init__(safe_message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class AuthorizedGenerationPayload:
    """仅由 generation service 在 D2 release 后创建。

    payload 故意不携带 record/entity/relation/capability/key 等授权元数据；
    generator 只能看到本次已授权 value、公开 instruction 和一次性 context。
    """

    value: PublicSyntheticValue = field(repr=False)
    public_instruction: str = field(repr=False)
    variant: GenerationVariant
    context_id: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _GENERATION_SEAL:
            raise TypeError("AuthorizedGenerationPayload 不能由调用者构造")
        if not isinstance(self.value, PublicSyntheticValue):
            raise TypeError("generation payload 只接受 PublicSyntheticValue")
        if (
            not isinstance(self.public_instruction, str)
            or len(self.public_instruction) > 2048
        ):
            raise D21Error(
                D21RejectionReason.INVALID_GENERATION_REQUEST,
                "public instruction 长度或类型非法",
            )
        if not isinstance(self.variant, GenerationVariant):
            raise D21Error(
                D21RejectionReason.INVALID_GENERATION_REQUEST,
                "未知 generation variant",
            )
        if not _CONTEXT_ID.fullmatch(self.context_id):
            raise D21Error(
                D21RejectionReason.INVALID_GENERATION_REQUEST,
                "context ID 非法",
            )


def create_generation_payload(
    *,
    value: PublicSyntheticValue,
    public_instruction: str,
    variant: GenerationVariant,
    context_id: str,
) -> AuthorizedGenerationPayload:
    return AuthorizedGenerationPayload(
        value=value,
        public_instruction=public_instruction,
        variant=variant,
        context_id=context_id,
        _seal=_GENERATION_SEAL,
    )


def assert_generation_payload(value: Any) -> AuthorizedGenerationPayload:
    if (
        not isinstance(value, AuthorizedGenerationPayload)
        or value._seal is not _GENERATION_SEAL
    ):
        raise D21Error(
            D21RejectionReason.INVALID_GENERATION_REQUEST,
            "generator 只接受 service 产生的 authorized payload",
        )
    return value


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str = field(repr=False)
    variant: GenerationVariant
    context_id: str
    output_digest: str

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        variant: GenerationVariant,
        context_id: str,
    ) -> "GenerationOutput":
        if not isinstance(text, str):
            raise TypeError("generation output 必须是 str")
        return cls(
            text=text,
            variant=variant,
            context_id=context_id,
            output_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class DeliveredGeneration:
    """交付给当前已授权调用方；artifact 不得序列化 text。"""

    text: str = field(repr=False)
    output_digest: str
    record_id: str
    variant: GenerationVariant
    exact_delivery: bool

    def __post_init__(self) -> None:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != (
            self.output_digest
        ):
            raise ValueError("delivered output digest 不一致")
