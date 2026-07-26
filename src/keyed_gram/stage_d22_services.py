"""D2.2 gateway、memory 与 generator 的独立 AF_UNIX 服务进程。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import resource
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from .stage_c24_contract import RelationId
from .stage_d1_contract import (
    RETRIEVE_PERMISSION,
    AuthorizedMemoryRequest,
    CapabilityError,
)
from .stage_d1_gateway import CapabilityVerifier
from .stage_d1_policy import CapabilityPolicy
from .stage_d1_token import CapabilityTokenCodec, HmacKeyRing
from .stage_d2_contract import D2Error, EncryptedMemoryRecord
from .stage_d2_crypto import AesGcmRecordCodec, DataEncryptionKeyRing
from .stage_d22_contract import (
    D22_SCHEMA_VERSION,
    D22Error,
    D22RejectionReason,
    ReleaseTicketClaims,
    safe_error_response,
)
from .stage_d22_ipc import (
    MAX_IPC_MESSAGE_BYTES,
    append_safe_event,
    decode_message,
    encode_message,
    message_schema_event,
    rpc_call,
    secure_process_setup,
)
from .stage_d22_state import (
    PersistentCapabilityState,
    PersistentTicketState,
)
from .stage_d22_ticket import ReleaseTicketCodec, ReleaseTicketKey

CRASH_EXIT_CODE = 86
INSTRUCTION_ID = "copy-current-authorized-value"


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: Any, *, label: str, maximum: int = 8192) -> bytes:
    if not isinstance(value, str) or not value or len(value) > maximum * 2:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            f"{label} 不是长度受限 base64 string",
        )
    try:
        decoded = base64.b64decode(
            value.encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            f"{label} base64 解码失败",
        ) from exc
    if not decoded or len(decoded) > maximum:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            f"{label} 解码长度非法",
        )
    return decoded


def _safe_reason(error: BaseException) -> str:
    if isinstance(error, (D22Error, CapabilityError, D2Error)):
        return error.reason.value
    return D22RejectionReason.INVALID_MESSAGE.value


def _role_event(
    *,
    role: str,
    event_path: Path,
    holds_capability_key: bool,
    holds_release_ticket_key: bool,
    holds_data_key: bool,
    has_gateway_endpoint: bool,
    has_memory_endpoint: bool,
    has_generator_endpoint: bool,
) -> None:
    core_enabled = False
    if hasattr(resource, "RLIMIT_CORE"):
        current, maximum = resource.getrlimit(resource.RLIMIT_CORE)
        core_enabled = current != 0 or maximum != 0
    append_safe_event(
        event_path,
        {
            "event": "role_manifest",
            "role": role,
            "pid": os.getpid(),
            "holds_capability_hmac_key": holds_capability_key,
            "holds_release_ticket_key": holds_release_ticket_key,
            "holds_data_encryption_key": holds_data_key,
            "has_gateway_endpoint": has_gateway_endpoint,
            "has_memory_endpoint": has_memory_endpoint,
            "has_generator_endpoint": has_generator_endpoint,
            "core_dump_enabled": core_enabled,
            "secret_in_command_line": False,
            "secret_in_environment": False,
        },
    )


def _record_rejection(
    event_path: Path,
    *,
    role: str,
    reason: str,
) -> None:
    append_safe_event(
        event_path,
        {
            "event": "service_rejection",
            "role": role,
            "reason": reason,
        },
    )


def _send_response(connection: Any, response: Mapping[str, Any]) -> None:
    connection.send_bytes(encode_message(response))


def _shutdown_response() -> dict[str, Any]:
    return {
        "schema_version": D22_SCHEMA_VERSION,
        "status": "ok",
        "operation": "shutdown",
    }


@dataclass(frozen=True, slots=True)
class GatewayServiceConfig:
    gateway_id: str
    socket_path: Path
    client_authkey: bytes = field(repr=False)
    memory_socket_path: Path
    memory_authkey: bytes = field(repr=False)
    capability_key_id: str
    capability_key: bytes = field(repr=False)
    release_ticket_key: ReleaseTicketKey = field(repr=False)
    policy: CapabilityPolicy
    session_subjects_by_digest: Mapping[str, str] = field(repr=False)
    capability_state_path: Path
    ticket_ttl_seconds: int
    event_path: Path
    log_path: Path
    fault_stage: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryServiceConfig:
    memory_id: str
    socket_path: Path
    gateway_authkey: bytes = field(repr=False)
    generator_socket_path: Path
    generator_authkey: bytes = field(repr=False)
    release_ticket_key: ReleaseTicketKey = field(repr=False)
    allowed_gateway_ids: tuple[str, ...]
    ticket_state_path: Path
    data_key_id: str
    data_key: bytes = field(repr=False)
    records: tuple[EncryptedMemoryRecord, ...] = field(repr=False)
    event_path: Path
    log_path: Path
    fault_stage: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratorServiceConfig:
    generator_id: str
    socket_path: Path
    memory_authkey: bytes = field(repr=False)
    allowed_instruction_ids: tuple[str, ...]
    event_path: Path
    log_path: Path
    fault_stage: str | None = None


@dataclass(frozen=True, slots=True)
class _CrashAwareReplayStore:
    state: PersistentCapabilityState
    event_path: Path
    fault_stage: str | None

    def consume(self, nonce: str) -> None:
        if self.fault_stage == "before_capability_consume":
            self.state.reserve_indeterminate(nonce, now=int(time.time()))
            append_safe_event(
                self.event_path,
                {
                    "event": "fault_injected",
                    "role": "gateway",
                    "stage": self.fault_stage,
                },
            )
            os._exit(CRASH_EXIT_CODE)
        self.state.consume(nonce)


def _gateway_request(
    config: GatewayServiceConfig,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "op",
        "session_credential_b64",
        "request",
        "instruction_id",
    }
    request_payload = message.get("request")
    if (
        set(message) != expected
        or message.get("schema_version") != D22_SCHEMA_VERSION
        or message.get("op") != "retrieve_and_generate"
        or not isinstance(request_payload, Mapping)
        or set(request_payload)
        != {
            "subject_id",
            "entity_id",
            "relation_id",
            "capability_token_b64",
        }
        or message.get("instruction_id") != INSTRUCTION_ID
    ):
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "client→gateway message schema 非法",
        )
    credential = _b64decode(
        message["session_credential_b64"],
        label="session credential",
        maximum=64,
    )
    if len(credential) != 32:
        raise D22Error(
            D22RejectionReason.INVALID_SESSION,
            "session credential 长度非法",
        )
    subject = config.session_subjects_by_digest.get(
        hashlib.sha256(credential).hexdigest()
    )
    if subject is None:
        raise D22Error(
            D22RejectionReason.INVALID_SESSION,
            "session credential 无效",
        )
    try:
        relation = RelationId(request_payload["relation_id"])
    except (TypeError, ValueError) as exc:
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "request relation 非冻结 RelationId",
        ) from exc
    token = _b64decode(
        request_payload["capability_token_b64"],
        label="capability token",
        maximum=4096,
    )
    request = AuthorizedMemoryRequest(
        subject_id=request_payload["subject_id"],
        entity_id=request_payload["entity_id"],
        relation_id=relation,
        capability_token=token,
    )
    if subject != request.subject_id:
        raise D22Error(
            D22RejectionReason.INVALID_SESSION,
            "trusted session principal 与 request subject 不一致",
        )
    state = PersistentCapabilityState(config.capability_state_path)
    verifier = CapabilityVerifier(
        HmacKeyRing(
            {config.capability_key_id: config.capability_key},
            config.capability_key_id,
        ),
        config.policy,
        CapabilityTokenCodec(),
        _CrashAwareReplayStore(
            state,
            config.event_path,
            config.fault_stage,
        ),
    )
    now = int(time.time())
    verified = verifier.authorize(request, now=now)
    append_safe_event(
        config.event_path,
        {
            "event": "capability_consumed",
            "role": "gateway",
            "gateway_id": config.gateway_id,
        },
    )
    if config.fault_stage == "after_capability_consume_before_lookup":
        append_safe_event(
            config.event_path,
            {
                "event": "fault_injected",
                "role": "gateway",
                "stage": config.fault_stage,
            },
        )
        os._exit(CRASH_EXIT_CODE)
    sequence = state.next_sequence(config.gateway_id)
    claims = ReleaseTicketClaims(
        gateway_id=config.gateway_id,
        ticket_id=secrets.token_hex(32),
        request_nonce=secrets.token_hex(32),
        subject_id=verified.subject_id,
        entity_id=verified.entity_id,
        relation_id=verified.relation_id,
        permission=RETRIEVE_PERMISSION,
        instruction_id=message["instruction_id"],
        sequence=sequence,
        issued_at=now,
        expires_at=now + config.ticket_ttl_seconds,
    )
    ticket = ReleaseTicketCodec().issue(
        claims,
        config.release_ticket_key,
    )
    memory_message = {
        "schema_version": D22_SCHEMA_VERSION,
        "op": "release_and_generate",
        "release_ticket_b64": _b64encode(ticket),
    }
    memory_response = rpc_call(
        config.memory_socket_path,
        config.memory_authkey,
        memory_message,
    )
    if memory_response.get("status") != "ok":
        return safe_error_response(
            str(
                memory_response.get(
                    "reason",
                    D22RejectionReason.UPSTREAM_UNAVAILABLE.value,
                )
            )
        )
    expected_response = {
        "schema_version",
        "status",
        "request_nonce",
        "output_b64",
        "output_digest",
    }
    if (
        set(memory_response) != expected_response
        or memory_response.get("request_nonce") != claims.request_nonce
    ):
        raise D22Error(
            D22RejectionReason.OUTPUT_VALIDATION_FAILED,
            "memory response schema/context 不匹配",
        )
    output = _b64decode(
        memory_response["output_b64"],
        label="generated output",
    )
    digest = hashlib.sha256(output).hexdigest()
    if digest != memory_response.get("output_digest"):
        raise D22Error(
            D22RejectionReason.OUTPUT_VALIDATION_FAILED,
            "generated output digest 不匹配",
        )
    if config.fault_stage == "after_output_before_response":
        append_safe_event(
            config.event_path,
            {
                "event": "fault_injected",
                "role": "gateway",
                "stage": config.fault_stage,
            },
        )
        os._exit(CRASH_EXIT_CODE)
    append_safe_event(
        config.event_path,
        {
            "event": "client_output",
            "role": "gateway",
            "output_digest": digest,
        },
    )
    return {
        "schema_version": D22_SCHEMA_VERSION,
        "status": "ok",
        "request_id": claims.request_nonce,
        "output_b64": _b64encode(output),
        "output_digest": digest,
    }


def run_gateway_service(config: GatewayServiceConfig) -> None:
    secure_process_setup(config.log_path)
    state = PersistentCapabilityState(config.capability_state_path)
    state.initialize()
    _role_event(
        role="gateway",
        event_path=config.event_path,
        holds_capability_key=True,
        holds_release_ticket_key=True,
        holds_data_key=False,
        has_gateway_endpoint=False,
        has_memory_endpoint=True,
        has_generator_endpoint=False,
    )
    if config.socket_path.exists():
        config.socket_path.unlink()
    try:
        with Listener(
            str(config.socket_path),
            family="AF_UNIX",
            backlog=16,
            authkey=config.client_authkey,
        ) as listener:
            while True:
                try:
                    connection = listener.accept()
                except AuthenticationError:
                    _record_rejection(
                        config.event_path,
                        role="gateway",
                        reason=D22RejectionReason.SERVICE_AUTHENTICATION_FAILED.value,
                    )
                    continue
                with connection:
                    try:
                        raw = connection.recv_bytes(MAX_IPC_MESSAGE_BYTES)
                        message = decode_message(raw)
                        if message == {
                            "schema_version": D22_SCHEMA_VERSION,
                            "op": "shutdown",
                        }:
                            _send_response(connection, _shutdown_response())
                            break
                        append_safe_event(
                            config.event_path,
                            message_schema_event(
                                edge="client_to_gateway",
                                message=message,
                                wire_size_bytes=len(raw),
                            ),
                        )
                        response = _gateway_request(config, message)
                    except (D22Error, CapabilityError, D2Error) as exc:
                        reason = _safe_reason(exc)
                        _record_rejection(
                            config.event_path,
                            role="gateway",
                            reason=reason,
                        )
                        response = safe_error_response(reason)
                    # 服务边界对未预期的应用异常统一脱敏并 fail closed。
                    except Exception:  # noqa: BLE001
                        reason = D22RejectionReason.INVALID_MESSAGE.value
                        _record_rejection(
                            config.event_path,
                            role="gateway",
                            reason=reason,
                        )
                        response = safe_error_response(reason)
                    _send_response(connection, response)
    finally:
        if config.socket_path.exists():
            config.socket_path.unlink()


def _memory_request(
    config: MemoryServiceConfig,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(message) != {"schema_version", "op", "release_ticket_b64"}
        or message.get("schema_version") != D22_SCHEMA_VERSION
        or message.get("op") != "release_and_generate"
    ):
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "gateway→memory message schema 非法",
        )
    ticket = _b64decode(
        message["release_ticket_b64"],
        label="release ticket",
        maximum=4096,
    )
    claims = ReleaseTicketCodec().verify(
        ticket,
        config.release_ticket_key,
    )
    if claims.gateway_id not in config.allowed_gateway_ids:
        raise D22Error(
            D22RejectionReason.INVALID_SERVICE_IDENTITY,
            "ticket gateway identity 未注册",
        )
    now = int(time.time())
    if now < claims.issued_at or now >= claims.expires_at:
        raise D22Error(
            D22RejectionReason.EXPIRED_TICKET,
            "release ticket 不在有效时间窗口",
        )
    state = PersistentTicketState(config.ticket_state_path)
    state.consume(claims, now=now)
    append_safe_event(
        config.event_path,
        {
            "event": "release_ticket_consumed",
            "role": "memory",
            "gateway_id": claims.gateway_id,
            "sequence": claims.sequence,
        },
    )
    records = {
        (record.entity_id, record.relation_id): record
        for record in config.records
    }
    record = records.get((claims.entity_id, claims.relation_id))
    if record is None:
        raise D22Error(
            D22RejectionReason.RECORD_NOT_FOUND,
            "typed entity/relation record 不存在",
        )
    append_safe_event(
        config.event_path,
        {
            "event": "memory_lookup",
            "role": "memory",
            "record_id": record.record_id,
        },
    )
    keyring = DataEncryptionKeyRing(
        {config.data_key_id: config.data_key},
        config.data_key_id,
    )
    append_safe_event(
        config.event_path,
        {
            "event": "decrypt_attempt",
            "role": "memory",
            "record_id": record.record_id,
        },
    )
    value = AesGcmRecordCodec().decrypt(record, keyring=keyring)
    append_safe_event(
        config.event_path,
        {
            "event": "plaintext_release",
            "role": "memory",
            "record_id": record.record_id,
            "value_digest": value.digest,
        },
    )
    if config.fault_stage == "after_decrypt_before_generator":
        append_safe_event(
            config.event_path,
            {
                "event": "fault_injected",
                "role": "memory",
                "stage": config.fault_stage,
            },
        )
        os._exit(CRASH_EXIT_CODE)
    generator_message = {
        "schema_version": D22_SCHEMA_VERSION,
        "op": "generate",
        "value_b64": _b64encode(value.payload),
        "request_nonce": claims.request_nonce,
        "instruction_id": claims.instruction_id,
    }
    generator_response = rpc_call(
        config.generator_socket_path,
        config.generator_authkey,
        generator_message,
    )
    if generator_response.get("status") != "ok":
        return safe_error_response(
            str(
                generator_response.get(
                    "reason",
                    D22RejectionReason.GENERATOR_FAILURE.value,
                )
            )
        )
    expected = {
        "schema_version",
        "status",
        "request_nonce",
        "output_b64",
        "output_digest",
    }
    if (
        set(generator_response) != expected
        or generator_response.get("request_nonce") != claims.request_nonce
    ):
        raise D22Error(
            D22RejectionReason.OUTPUT_VALIDATION_FAILED,
            "generator response schema/context 不匹配",
        )
    output = _b64decode(
        generator_response["output_b64"],
        label="generator output",
    )
    if output != value.payload:
        raise D22Error(
            D22RejectionReason.OUTPUT_VALIDATION_FAILED,
            "generator 输出不是当前单个 authorized value",
        )
    digest = hashlib.sha256(output).hexdigest()
    if digest != generator_response.get("output_digest"):
        raise D22Error(
            D22RejectionReason.OUTPUT_VALIDATION_FAILED,
            "generator output digest 不匹配",
        )
    return {
        "schema_version": D22_SCHEMA_VERSION,
        "status": "ok",
        "request_nonce": claims.request_nonce,
        "output_b64": _b64encode(output),
        "output_digest": digest,
    }


def run_memory_service(config: MemoryServiceConfig) -> None:
    secure_process_setup(config.log_path)
    PersistentTicketState(config.ticket_state_path).initialize()
    _role_event(
        role="memory",
        event_path=config.event_path,
        holds_capability_key=False,
        holds_release_ticket_key=True,
        holds_data_key=True,
        has_gateway_endpoint=True,
        has_memory_endpoint=False,
        has_generator_endpoint=True,
    )
    if config.socket_path.exists():
        config.socket_path.unlink()
    try:
        with Listener(
            str(config.socket_path),
            family="AF_UNIX",
            backlog=16,
            authkey=config.gateway_authkey,
        ) as listener:
            while True:
                try:
                    connection = listener.accept()
                except AuthenticationError:
                    _record_rejection(
                        config.event_path,
                        role="memory",
                        reason=D22RejectionReason.SERVICE_AUTHENTICATION_FAILED.value,
                    )
                    continue
                with connection:
                    try:
                        raw = connection.recv_bytes(MAX_IPC_MESSAGE_BYTES)
                        message = decode_message(raw)
                        if message == {
                            "schema_version": D22_SCHEMA_VERSION,
                            "op": "shutdown",
                        }:
                            _send_response(connection, _shutdown_response())
                            break
                        append_safe_event(
                            config.event_path,
                            message_schema_event(
                                edge="gateway_to_memory",
                                message=message,
                                wire_size_bytes=len(raw),
                            ),
                        )
                        response = _memory_request(config, message)
                    except (D22Error, CapabilityError, D2Error) as exc:
                        reason = _safe_reason(exc)
                        _record_rejection(
                            config.event_path,
                            role="memory",
                            reason=reason,
                        )
                        response = safe_error_response(reason)
                    # 服务边界对未预期的应用异常统一脱敏并 fail closed。
                    except Exception:  # noqa: BLE001
                        reason = D22RejectionReason.INVALID_MESSAGE.value
                        _record_rejection(
                            config.event_path,
                            role="memory",
                            reason=reason,
                        )
                        response = safe_error_response(reason)
                    _send_response(connection, response)
    finally:
        if config.socket_path.exists():
            config.socket_path.unlink()


def _generator_request(
    config: GeneratorServiceConfig,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "op",
        "value_b64",
        "request_nonce",
        "instruction_id",
    }
    if (
        set(message) != expected
        or message.get("schema_version") != D22_SCHEMA_VERSION
        or message.get("op") != "generate"
        or message.get("instruction_id")
        not in config.allowed_instruction_ids
        or not isinstance(message.get("request_nonce"), str)
        or len(message["request_nonce"]) < 22
    ):
        raise D22Error(
            D22RejectionReason.INVALID_MESSAGE,
            "memory→generator message schema 非法",
        )
    value = _b64decode(
        message["value_b64"],
        label="ephemeral synthetic value",
        maximum=4096,
    )
    append_safe_event(
        config.event_path,
        {
            "event": "generator_invocation",
            "role": "generator",
            "value_digest": hashlib.sha256(value).hexdigest(),
        },
    )
    if config.fault_stage == "after_receive_before_output":
        append_safe_event(
            config.event_path,
            {
                "event": "fault_injected",
                "role": "generator",
                "stage": config.fault_stage,
            },
        )
        os._exit(CRASH_EXIT_CODE)
    digest = hashlib.sha256(value).hexdigest()
    append_safe_event(
        config.event_path,
        {
            "event": "generation_output",
            "role": "generator",
            "output_digest": digest,
        },
    )
    return {
        "schema_version": D22_SCHEMA_VERSION,
        "status": "ok",
        "request_nonce": message["request_nonce"],
        "output_b64": _b64encode(value),
        "output_digest": digest,
    }


def run_generator_service(config: GeneratorServiceConfig) -> None:
    secure_process_setup(config.log_path)
    _role_event(
        role="generator",
        event_path=config.event_path,
        holds_capability_key=False,
        holds_release_ticket_key=False,
        holds_data_key=False,
        has_gateway_endpoint=False,
        has_memory_endpoint=False,
        has_generator_endpoint=False,
    )
    if config.socket_path.exists():
        config.socket_path.unlink()
    try:
        with Listener(
            str(config.socket_path),
            family="AF_UNIX",
            backlog=16,
            authkey=config.memory_authkey,
        ) as listener:
            while True:
                try:
                    connection = listener.accept()
                except AuthenticationError:
                    _record_rejection(
                        config.event_path,
                        role="generator",
                        reason=D22RejectionReason.SERVICE_AUTHENTICATION_FAILED.value,
                    )
                    continue
                with connection:
                    try:
                        raw = connection.recv_bytes(MAX_IPC_MESSAGE_BYTES)
                        message = decode_message(raw)
                        if message == {
                            "schema_version": D22_SCHEMA_VERSION,
                            "op": "shutdown",
                        }:
                            _send_response(connection, _shutdown_response())
                            break
                        append_safe_event(
                            config.event_path,
                            message_schema_event(
                                edge="memory_to_generator",
                                message=message,
                                wire_size_bytes=len(raw),
                            ),
                        )
                        response = _generator_request(config, message)
                    except D22Error as exc:
                        reason = exc.reason.value
                        _record_rejection(
                            config.event_path,
                            role="generator",
                            reason=reason,
                        )
                        response = safe_error_response(reason)
                    # 服务边界对未预期的应用异常统一脱敏并 fail closed。
                    except Exception:  # noqa: BLE001
                        reason = D22RejectionReason.INVALID_MESSAGE.value
                        _record_rejection(
                            config.event_path,
                            role="generator",
                            reason=reason,
                        )
                        response = safe_error_response(reason)
                    _send_response(connection, response)
    finally:
        if config.socket_path.exists():
            config.socket_path.unlink()
