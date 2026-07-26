"""D2.2 本地多进程服务集群、测试 authority 与生命周期控制。"""

from __future__ import annotations

import base64
import hashlib
import multiprocessing
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .stage_c24_contract import RelationId
from .stage_d1_contract import (
    RETRIEVE_PERMISSION,
    CapabilityGrant,
    IssuedCapability,
)
from .stage_d1_policy import (
    CapabilityAuthority,
    CapabilityPolicy,
    PolicyRule,
)
from .stage_d1_token import CapabilityTokenCodec, HmacKeyRing
from .stage_d2_contract import EncryptedMemoryRecord, PublicSyntheticValue
from .stage_d2_crypto import AesGcmRecordCodec, DataEncryptionKeyRing
from .stage_d21_leakage import (
    CanaryFactory,
    count_canary_occurrences,
)
from .stage_d22_contract import (
    D22_SCHEMA_VERSION,
    D22Error,
    D22RejectionReason,
    IsolatedGenerationResult,
    ReleaseTicketClaims,
)
from .stage_d22_ipc import append_safe_event, rpc_call, wait_for_socket
from .stage_d22_services import (
    INSTRUCTION_ID,
    GatewayServiceConfig,
    GeneratorServiceConfig,
    MemoryServiceConfig,
    run_gateway_service,
    run_generator_service,
    run_memory_service,
)
from .stage_d22_state import (
    PersistentCapabilityState,
    PersistentTicketState,
)
from .stage_d22_ticket import ReleaseTicketCodec, ReleaseTicketKey

CAPABILITY_KEY_ID = "capability-key-d22"
RELEASE_TICKET_KEY_ID = "release-ticket-key-d22"
DATA_KEY_ID = "data-key-d22"


@dataclass(slots=True)
class ServiceProcess:
    role: str
    instance_id: str
    socket_path: Path
    authkey: bytes = field(repr=False)
    process: Any = field(repr=False)


@dataclass(slots=True)
class D22Cluster:
    root: Path
    canary_factory: CanaryFactory
    capability_key: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
    )
    release_ticket_key_bytes: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
    )
    data_key: bytes = field(
        default_factory=lambda: AESGCM.generate_key(bit_length=256),
        repr=False,
    )
    client_gateway_authkey: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
    )
    gateway_memory_authkey: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
    )
    memory_generator_authkey: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
    )
    sessions: dict[str, bytes] = field(default_factory=dict, repr=False)
    canaries: dict[tuple[str, RelationId], str] = field(
        default_factory=dict,
        repr=False,
    )
    records: tuple[EncryptedMemoryRecord, ...] = field(
        default=(),
        repr=False,
    )
    processes: dict[tuple[str, str], ServiceProcess] = field(
        default_factory=dict,
        repr=False,
    )
    process_surface_payloads: list[bytes] = field(
        default_factory=list,
        repr=False,
    )
    _context: Any = field(init=False, repr=False)
    _policy: CapabilityPolicy = field(init=False, repr=False)
    _authority: CapabilityAuthority = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._context = multiprocessing.get_context("spawn")
        self.sessions = {
            "subject-alpha": secrets.token_bytes(32),
            "subject-beta": secrets.token_bytes(32),
        }
        relation_values = tuple(sorted(item.value for item in RelationId))
        entity_scope = tuple(
            sorted(("entity-a", "entity-b", "entity-c", "entity-d"))
        )
        self._policy = CapabilityPolicy(
            version="d22-capability-policy-v1",
            rules=(
                PolicyRule(
                    "subject-alpha",
                    entity_scope,
                    relation_values,
                ),
                PolicyRule(
                    "subject-beta",
                    entity_scope,
                    relation_values,
                ),
            ),
            max_ttl_seconds=300,
        )
        keyring = HmacKeyRing(
            {CAPABILITY_KEY_ID: self.capability_key},
            CAPABILITY_KEY_ID,
        )
        self._authority = CapabilityAuthority(
            keyring,
            self._policy,
            CapabilityTokenCodec(),
        )
        data_keyring = DataEncryptionKeyRing(
            {DATA_KEY_ID: self.data_key},
            DATA_KEY_ID,
        )
        codec = AesGcmRecordCodec()
        declarations = (
            ("record-a-registry", "entity-a", RelationId.REGISTRY_ID),
            ("record-b-city", "entity-b", RelationId.CITY_CODE),
            ("record-c-access", "entity-c", RelationId.ACCESS_CODE),
            ("record-d-registry", "entity-d", RelationId.REGISTRY_ID),
        )
        records: list[EncryptedMemoryRecord] = []
        for record_id, entity_id, relation_id in declarations:
            canary = self.canary_factory.issue()
            self.canaries[(entity_id, relation_id)] = canary
            records.append(
                codec.encrypt(
                    PublicSyntheticValue(canary.encode("ascii")),
                    record_id=record_id,
                    entity_id=entity_id,
                    relation_id=relation_id,
                    record_version=1,
                    keyring=data_keyring,
                )
            )
        self.records = tuple(records)
        PersistentCapabilityState(self.capability_state_path).initialize()
        PersistentTicketState(self.ticket_state_path).initialize()

    @property
    def event_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def capability_state_path(self) -> Path:
        return self.root / "capability-state.sqlite3"

    @property
    def ticket_state_path(self) -> Path:
        return self.root / "ticket-state.sqlite3"

    @property
    def release_ticket_key(self) -> ReleaseTicketKey:
        return ReleaseTicketKey(
            RELEASE_TICKET_KEY_ID,
            self.release_ticket_key_bytes,
        )

    def socket_path(self, role: str, instance_id: str) -> Path:
        return self.root / f"{role}-{instance_id}.sock"

    def log_path(self, role: str, instance_id: str) -> Path:
        return self.root / f"{role}-{instance_id}.log"

    def start_generator(
        self,
        *,
        instance_id: str = "generator-a",
        fault_stage: str | None = None,
    ) -> ServiceProcess:
        socket_path = self.socket_path("generator", instance_id)
        config = GeneratorServiceConfig(
            generator_id=instance_id,
            socket_path=socket_path,
            memory_authkey=self.memory_generator_authkey,
            allowed_instruction_ids=(INSTRUCTION_ID,),
            event_path=self.event_path,
            log_path=self.log_path("generator", instance_id),
            fault_stage=fault_stage,
        )
        return self._start(
            "generator",
            instance_id,
            socket_path,
            self.memory_generator_authkey,
            run_generator_service,
            config,
        )

    def start_memory(
        self,
        *,
        instance_id: str = "memory-a",
        generator_id: str = "generator-a",
        fault_stage: str | None = None,
    ) -> ServiceProcess:
        socket_path = self.socket_path("memory", instance_id)
        config = MemoryServiceConfig(
            memory_id=instance_id,
            socket_path=socket_path,
            gateway_authkey=self.gateway_memory_authkey,
            generator_socket_path=self.socket_path(
                "generator", generator_id
            ),
            generator_authkey=self.memory_generator_authkey,
            release_ticket_key=self.release_ticket_key,
            allowed_gateway_ids=("gateway-a", "gateway-b"),
            ticket_state_path=self.ticket_state_path,
            data_key_id=DATA_KEY_ID,
            data_key=self.data_key,
            records=self.records,
            event_path=self.event_path,
            log_path=self.log_path("memory", instance_id),
            fault_stage=fault_stage,
        )
        return self._start(
            "memory",
            instance_id,
            socket_path,
            self.gateway_memory_authkey,
            run_memory_service,
            config,
        )

    def start_gateway(
        self,
        *,
        instance_id: str = "gateway-a",
        memory_id: str = "memory-a",
        fault_stage: str | None = None,
    ) -> ServiceProcess:
        socket_path = self.socket_path("gateway", instance_id)
        sessions = {
            hashlib.sha256(credential).hexdigest(): subject
            for subject, credential in self.sessions.items()
        }
        config = GatewayServiceConfig(
            gateway_id=instance_id,
            socket_path=socket_path,
            client_authkey=self.client_gateway_authkey,
            memory_socket_path=self.socket_path("memory", memory_id),
            memory_authkey=self.gateway_memory_authkey,
            capability_key_id=CAPABILITY_KEY_ID,
            capability_key=self.capability_key,
            release_ticket_key=self.release_ticket_key,
            policy=self._policy,
            session_subjects_by_digest=sessions,
            capability_state_path=self.capability_state_path,
            ticket_ttl_seconds=30,
            event_path=self.event_path,
            log_path=self.log_path("gateway", instance_id),
            fault_stage=fault_stage,
        )
        return self._start(
            "gateway",
            instance_id,
            socket_path,
            self.client_gateway_authkey,
            run_gateway_service,
            config,
        )

    def start_default(
        self,
        *,
        gateway_fault: str | None = None,
        memory_fault: str | None = None,
        generator_fault: str | None = None,
    ) -> None:
        self.start_generator(fault_stage=generator_fault)
        self.start_memory(fault_stage=memory_fault)
        self.start_gateway(fault_stage=gateway_fault)
        self.capture_process_surfaces()

    def _start(
        self,
        role: str,
        instance_id: str,
        socket_path: Path,
        authkey: bytes,
        target: Any,
        config: Any,
    ) -> ServiceProcess:
        key = (role, instance_id)
        previous = self.processes.get(key)
        if previous is not None and previous.process.is_alive():
            raise RuntimeError(f"{role}/{instance_id} 已在运行")
        if socket_path.exists():
            socket_path.unlink()
        process = self._context.Process(
            target=target,
            args=(config,),
            name=f"d22-{role}-{instance_id}",
            daemon=False,
        )
        process.start()
        try:
            wait_for_socket(socket_path)
        except BaseException:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3.0)
            if socket_path.exists():
                socket_path.unlink()
            raise
        handle = ServiceProcess(
            role,
            instance_id,
            socket_path,
            authkey,
            process,
        )
        self.processes[key] = handle
        return handle

    def capture_process_surfaces(self) -> None:
        current_payloads: list[bytes] = []
        for handle in self.processes.values():
            pid = handle.process.pid
            if pid is None or not handle.process.is_alive():
                continue
            for name in ("cmdline", "environ"):
                path = Path(f"/proc/{pid}/{name}")
                if path.is_file():
                    try:
                        payload = path.read_bytes()
                        current_payloads.append(payload)
                        self.process_surface_payloads.append(payload)
                    except OSError:
                        pass
        secret_materials = (
            self.capability_key,
            self.release_ticket_key_bytes,
            self.data_key,
            self.client_gateway_authkey,
            self.gateway_memory_authkey,
            self.memory_generator_authkey,
            *self.sessions.values(),
        )
        secret_variants = {
            variant
            for secret in secret_materials
            for variant in (
                secret,
                secret.hex().encode("ascii"),
                base64.b64encode(secret),
                base64.urlsafe_b64encode(secret).rstrip(b"="),
            )
        }
        secret_occurrences = sum(
            payload.count(variant)
            for payload in current_payloads
            for variant in secret_variants
        )
        canary_occurrences = count_canary_occurrences(
            current_payloads,
            self.canary_factory._issued,
        )
        append_safe_event(
            self.event_path,
            {
                "event": "process_surface_scan",
                "role": "audit_parent",
                "scanned_process_count": sum(
                    handle.process.is_alive()
                    for handle in self.processes.values()
                ),
                "secret_occurrence_count": secret_occurrences,
                "canary_occurrence_count": canary_occurrences,
            },
        )

    def issue_message(
        self,
        relation_id: RelationId,
        entity_id: str,
        *,
        subject_id: str = "subject-alpha",
        ttl_seconds: int = 120,
        nonce: str | None = None,
        request_subject_id: str | None = None,
        request_entity_id: str | None = None,
        request_relation_id: RelationId | None = None,
        credential_subject_id: str | None = None,
        capability_entity_scope: tuple[str, ...] | None = None,
        issue_now: int | None = None,
    ) -> tuple[IssuedCapability, dict[str, Any]]:
        scope = (
            (entity_id,)
            if capability_entity_scope is None
            else capability_entity_scope
        )
        issued = self._authority.issue(
            CapabilityGrant(
                subject_id,
                scope,
                relation_id,
                (RETRIEVE_PERMISSION,),
            ),
            now=int(time.time()) if issue_now is None else issue_now,
            ttl_seconds=ttl_seconds,
            nonce=nonce,
        )
        credential_subject = credential_subject_id or subject_id
        message = {
            "schema_version": D22_SCHEMA_VERSION,
            "op": "retrieve_and_generate",
            "session_credential_b64": base64.b64encode(
                self.sessions[credential_subject]
            ).decode("ascii"),
            "request": {
                "subject_id": request_subject_id or subject_id,
                "entity_id": request_entity_id or entity_id,
                "relation_id": (
                    request_relation_id or relation_id
                ).value,
                "capability_token_b64": base64.b64encode(
                    issued.token
                ).decode("ascii"),
            },
            "instruction_id": INSTRUCTION_ID,
        }
        return issued, message

    def client_call(
        self,
        message: dict[str, Any],
        *,
        gateway_id: str = "gateway-a",
        authkey: bytes | None = None,
    ) -> dict[str, Any]:
        return rpc_call(
            self.socket_path("gateway", gateway_id),
            self.client_gateway_authkey if authkey is None else authkey,
            message,
        )

    def direct_memory_call(
        self,
        message: dict[str, Any],
        *,
        memory_id: str = "memory-a",
        authkey: bytes | None = None,
    ) -> dict[str, Any]:
        return rpc_call(
            self.socket_path("memory", memory_id),
            self.gateway_memory_authkey if authkey is None else authkey,
            message,
        )

    def issue_internal_ticket(
        self,
        *,
        gateway_id: str = "gateway-a",
        relation_id: RelationId = RelationId.REGISTRY_ID,
        entity_id: str = "entity-a",
        sequence: int = 1,
        issued_at: int | None = None,
        expires_at: int | None = None,
        ticket_id: str | None = None,
        request_nonce: str | None = None,
        key: ReleaseTicketKey | None = None,
    ) -> tuple[ReleaseTicketClaims, bytes]:
        now = int(time.time()) if issued_at is None else issued_at
        claims = ReleaseTicketClaims(
            gateway_id=gateway_id,
            ticket_id=ticket_id or secrets.token_hex(32),
            request_nonce=request_nonce or secrets.token_hex(32),
            subject_id="subject-alpha",
            entity_id=entity_id,
            relation_id=relation_id,
            permission=RETRIEVE_PERMISSION,
            instruction_id=INSTRUCTION_ID,
            sequence=sequence,
            issued_at=now,
            expires_at=(now + 30 if expires_at is None else expires_at),
        )
        return claims, ReleaseTicketCodec().issue(
            claims,
            self.release_ticket_key if key is None else key,
        )

    @staticmethod
    def memory_message(ticket: bytes) -> dict[str, Any]:
        return {
            "schema_version": D22_SCHEMA_VERSION,
            "op": "release_and_generate",
            "release_ticket_b64": base64.b64encode(ticket).decode("ascii"),
        }

    @staticmethod
    def parse_success(response: dict[str, Any]) -> IsolatedGenerationResult:
        expected = {
            "schema_version",
            "status",
            "request_id",
            "output_b64",
            "output_digest",
        }
        if set(response) != expected or response.get("status") != "ok":
            raise D22Error(
                D22RejectionReason.OUTPUT_VALIDATION_FAILED,
                "client response 不是成功 schema",
            )
        try:
            output = base64.b64decode(
                response["output_b64"].encode("ascii"),
                validate=True,
            )
            text = output.decode("ascii", errors="strict")
        except (UnicodeError, ValueError) as exc:
            raise D22Error(
                D22RejectionReason.OUTPUT_VALIDATION_FAILED,
                "client output 编码非法",
            ) from exc
        return IsolatedGenerationResult(
            text=text,
            output_digest=response["output_digest"],
            request_id=response["request_id"],
            exact_delivery=True,
        )

    def revoke_capability(self, nonce: str) -> None:
        PersistentCapabilityState(self.capability_state_path).revoke(
            nonce,
            now=int(time.time()),
        )

    def stop(
        self,
        role: str,
        instance_id: str,
        *,
        force: bool = False,
    ) -> None:
        handle = self.processes.get((role, instance_id))
        if handle is None:
            return
        process = handle.process
        if process.is_alive() and not force:
            try:
                rpc_call(
                    handle.socket_path,
                    handle.authkey,
                    {
                        "schema_version": D22_SCHEMA_VERSION,
                        "op": "shutdown",
                    },
                    timeout_seconds=2.0,
                )
            except D22Error:
                pass
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
        if handle.socket_path.exists():
            handle.socket_path.unlink()

    def stop_all(self) -> None:
        for role in ("gateway", "memory", "generator"):
            handles = [
                handle
                for (current_role, _), handle in self.processes.items()
                if current_role == role
            ]
            for handle in handles:
                self.stop(handle.role, handle.instance_id)

    def join_crashed(self, role: str, instance_id: str) -> int | None:
        handle = self.processes[(role, instance_id)]
        handle.process.join(timeout=5.0)
        return handle.process.exitcode
