"""D2 release 与一次性 generation adapter 之间的唯一连接层。"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field

from .stage_d1_contract import AuthorizedMemoryRequest
from .stage_d2_memory import (
    EncryptedKeyedMemory,
    PrincipalBoundCapabilityGateway,
)
from .stage_d21_contract import (
    D21Error,
    D21RejectionReason,
    DeliveredGeneration,
    GenerationVariant,
    create_generation_payload,
)
from .stage_d21_generator import G0_TEMPLATE, GenerationRegistry


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    successful_output_count: int
    validation_failure_count: int
    public_response_count: int

    def delta(self, previous: "ServiceSnapshot") -> dict[str, int]:
        return {
            "successful_output_delta": (
                self.successful_output_count
                - previous.successful_output_count
            ),
            "validation_failure_delta": (
                self.validation_failure_count
                - previous.validation_failure_count
            ),
            "public_response_delta": (
                self.public_response_count - previous.public_response_count
            ),
        }


@dataclass(slots=True)
class CapabilityGatedGenerationService:
    gateway: PrincipalBoundCapabilityGateway
    memory: EncryptedKeyedMemory
    generators: GenerationRegistry
    _successful_output_count: int = 0
    _validation_failure_count: int = 0
    _public_response_count: int = 0
    _last_safe_repr_samples: tuple[str, ...] = ()
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            return ServiceSnapshot(
                successful_output_count=self._successful_output_count,
                validation_failure_count=self._validation_failure_count,
                public_response_count=self._public_response_count,
            )

    @property
    def last_safe_repr_samples(self) -> tuple[str, ...]:
        with self._lock:
            return self._last_safe_repr_samples

    def generate(
        self,
        session_credential: bytes,
        request: AuthorizedMemoryRequest,
        *,
        public_instruction: str,
        variant: GenerationVariant,
        failure_mode: str | None = None,
    ) -> DeliveredGeneration:
        # D2 已原子消费 single-use capability，且完成 lookup/decrypt 后才返回。
        release = self.gateway.retrieve(
            session_credential,
            request,
            self.memory,
        )
        payload = create_generation_payload(
            value=release.value,
            public_instruction=public_instruction,
            variant=variant,
            context_id=f"d21-context-{secrets.token_hex(16)}",
        )
        output = self.generators.generate(
            payload,
            failure_mode=failure_mode,
        )
        value = release.value.payload.decode("ascii", errors="strict")
        expected = (
            G0_TEMPLATE.format(authorized_value=value)
            if variant is GenerationVariant.G0
            else value
        )
        if output.text != expected or output.context_id != payload.context_id:
            with self._lock:
                self._validation_failure_count += 1
                self._last_safe_repr_samples = (
                    repr(release),
                    repr(payload),
                    repr(output),
                )
            raise D21Error(
                D21RejectionReason.OUTPUT_VALIDATION_FAILED,
                "generation output validation failed",
            )
        delivered = DeliveredGeneration(
            text=output.text,
            output_digest=output.output_digest,
            record_id=release.record_id,
            variant=variant,
            exact_delivery=True,
        )
        with self._lock:
            self._successful_output_count += 1
            self._last_safe_repr_samples = (
                repr(release),
                repr(payload),
                repr(output),
                repr(delivered),
            )
        return delivered

    def public_response(self, public_instruction: str) -> str:
        """无 authorized value 的普通请求不调用 generator，也无历史恢复。"""

        if not isinstance(public_instruction, str):
            raise TypeError("public instruction 必须是 str")
        with self._lock:
            self._public_response_count += 1
        return "Public response: no authorized value was provided."
