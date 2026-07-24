"""Stage D1：不含 private value 的 authenticated capability contract 审计。"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .stage_c24_contract import RelationId
from .stage_d1_contract import (
    AuthorizedMemoryRequest,
    CapabilityError,
    CapabilityGrant,
    RejectionReason,
    SuggestedRelation,
)
from .stage_d1_gateway import (
    CapabilityMemoryProbe,
    CapabilityRetrievalResult,
    CapabilityVerifier,
    TrustedCapabilityGateway,
)
from .stage_d1_policy import (
    CapabilityAuthority,
    CapabilityPolicy,
    PolicyRule,
    RevocationReplayStore,
)
from .stage_d1_protocol import (
    D1ProtocolError,
    canonical_sha256,
    git_state,
    load_config,
    mark_phase_completed,
    mark_phase_started,
    output_paths,
    read_json,
    repo_root,
    require_phase_completed,
    resolve_path,
    runtime_source_manifest,
    sha256_file,
    write_json,
)
from .stage_d1_token import CapabilityTokenCodec, HmacKeyRing


FIXED_AUDIT_TIME = 1_800_000_000


@dataclass(slots=True)
class D1Environment:
    now: int
    policy: CapabilityPolicy
    keyring: HmacKeyRing
    store: RevocationReplayStore
    authority: CapabilityAuthority
    gateway: TrustedCapabilityGateway
    memory: CapabilityMemoryProbe


def _policy(version: str, *, max_ttl: int) -> CapabilityPolicy:
    rules = (
        PolicyRule(
            "subject-alpha",
            ("entity-a",),
            (RelationId.REGISTRY_ID.value,),
        ),
        PolicyRule(
            "subject-alpha",
            ("entity-b",),
            (RelationId.CITY_CODE.value,),
        ),
        PolicyRule(
            "subject-alpha",
            ("entity-c",),
            (RelationId.ACCESS_CODE.value,),
        ),
    )
    return CapabilityPolicy(version, rules, max_ttl, single_use=True)


def _environment(
    *,
    policy_version: str,
    max_ttl: int,
    max_token_bytes: int,
    key_bytes: int,
    now: int = FIXED_AUDIT_TIME,
    key_id: str = "d1-key-1",
) -> D1Environment:
    policy = _policy(policy_version, max_ttl=max_ttl)
    keyring = HmacKeyRing.generate(key_id, key_bytes=key_bytes)
    codec = CapabilityTokenCodec(max_token_bytes=max_token_bytes)
    store = RevocationReplayStore()
    authority = CapabilityAuthority(keyring, policy, codec)
    verifier = CapabilityVerifier(keyring, policy, codec, store)
    gateway = TrustedCapabilityGateway(verifier, lambda: now)
    memory = CapabilityMemoryProbe(
        frozenset(
            {
                ("entity-a", RelationId.REGISTRY_ID),
                ("entity-b", RelationId.CITY_CODE),
                ("entity-c", RelationId.ACCESS_CODE),
            }
        )
    )
    return D1Environment(
        now, policy, keyring, store, authority, gateway, memory
    )


def _grant(
    relation: RelationId,
    entity_id: str,
    *,
    subject_id: str = "subject-alpha",
    permissions: tuple[str, ...] = ("retrieve",),
) -> CapabilityGrant:
    return CapabilityGrant(
        subject_id,
        (entity_id,),
        relation,
        permissions,
    )


def _request(
    issued: Any,
    relation: RelationId,
    entity_id: str,
    *,
    subject_id: str = "subject-alpha",
) -> AuthorizedMemoryRequest:
    return AuthorizedMemoryRequest(
        subject_id=subject_id,
        entity_id=entity_id,
        relation_id=relation,
        capability_token=issued.token,
    )


def _b64decode(segment: bytes) -> bytes:
    return base64.urlsafe_b64decode(
        segment + b"=" * ((4 - len(segment) % 4) % 4)
    )


def _b64encode(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _tamper_payload(token: bytes) -> bytes:
    header, payload, signature = token.split(b".")
    value = json.loads(_b64decode(payload))
    value["entity_scope"] = ["entity-b"]
    altered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return b".".join((header, _b64encode(altered), signature))


def _tamper_algorithm(token: bytes) -> bytes:
    header, payload, signature = token.split(b".")
    value = json.loads(_b64decode(header))
    value["alg"] = "none"
    altered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return b".".join((_b64encode(altered), payload, signature))


def _tamper_signature(token: bytes) -> bytes:
    header, payload, signature = token.split(b".")
    raw = bytearray(_b64decode(signature))
    raw[0] ^= 0x01
    return b".".join((header, payload, _b64encode(bytes(raw))))


def _record_case(
    rows: list[dict[str, Any]],
    *,
    scenario_id: str,
    category: str,
    expected_status: str,
    expected_reason: RejectionReason | None,
    memory: CapabilityMemoryProbe,
    action: Callable[[], CapabilityRetrievalResult | Any],
) -> None:
    before = memory.memory_access_count
    observed_status = "accept"
    observed_reason = ""
    try:
        result = action()
        if expected_status == "accept" and not isinstance(
            result, CapabilityRetrievalResult
        ):
            raise AssertionError("positive scenario 未返回 typed retrieval result")
    except CapabilityError as exc:
        observed_status = "reject"
        observed_reason = exc.reason.value
    delta = memory.memory_access_count - before
    expected_reason_value = (
        expected_reason.value if expected_reason is not None else ""
    )
    passed = (
        observed_status == expected_status
        and observed_reason == expected_reason_value
        and delta == (1 if expected_status == "accept" else 0)
    )
    rows.append(
        {
            "scenario_id": scenario_id,
            "category": category,
            "expected_status": expected_status,
            "expected_reason": expected_reason_value,
            "observed_status": observed_status,
            "observed_reason": observed_reason,
            "memory_access_delta": delta,
            "cross_relation_access_delta": 0,
            "cross_entity_access_delta": 0,
            "passed": passed,
        }
    )


def _run_attack_matrix(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    declaration = values["capability"]

    def env(
        *,
        version: str | None = None,
        now: int = FIXED_AUDIT_TIME,
        key_id: str = "d1-key-1",
    ) -> D1Environment:
        return _environment(
            policy_version=version or str(declaration["policy_version"]),
            max_ttl=int(declaration["max_ttl_seconds"]),
            max_token_bytes=int(declaration["max_token_bytes"]),
            key_bytes=int(declaration["key_bytes"]),
            now=now,
            key_id=key_id,
        )

    rows: list[dict[str, Any]] = []

    for scenario, relation, entity in (
        ("valid_registry_capability", RelationId.REGISTRY_ID, "entity-a"),
        ("valid_city_capability", RelationId.CITY_CODE, "entity-b"),
        ("valid_access_capability", RelationId.ACCESS_CODE, "entity-c"),
    ):
        current = env()
        issued = current.authority.issue(
            _grant(relation, entity), now=current.now, ttl_seconds=60
        )
        _record_case(
            rows,
            scenario_id=scenario,
            category="positive",
            expected_status="accept",
            expected_reason=None,
            memory=current.memory,
            action=lambda c=current, i=issued, r=relation, e=entity: (
                c.gateway.retrieve(_request(i, r, e), c.memory)
            ),
        )

    current = env()
    current.keyring.rotate(
        "d1-key-2", key_bytes=int(declaration["key_bytes"])
    )
    rotated = current.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=current.now,
        ttl_seconds=60,
    )
    _record_case(
        rows,
        scenario_id="rotated_key_capability",
        category="key_rotation",
        expected_status="accept",
        expected_reason=None,
        memory=current.memory,
        action=lambda: current.gateway.retrieve(
            _request(rotated, RelationId.REGISTRY_ID, "entity-a"),
            current.memory,
        ),
    )

    replay_env = env()
    replay_token = replay_env.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=replay_env.now,
        ttl_seconds=60,
    )
    replay_request = _request(
        replay_token, RelationId.REGISTRY_ID, "entity-a"
    )
    _record_case(
        rows,
        scenario_id="replay_first_use",
        category="replay",
        expected_status="accept",
        expected_reason=None,
        memory=replay_env.memory,
        action=lambda: replay_env.gateway.retrieve(
            replay_request, replay_env.memory
        ),
    )
    _record_case(
        rows,
        scenario_id="replay_second_use",
        category="replay",
        expected_status="reject",
        expected_reason=RejectionReason.REPLAY,
        memory=replay_env.memory,
        action=lambda: replay_env.gateway.retrieve(
            replay_request, replay_env.memory
        ),
    )

    def issued_env(
        relation: RelationId = RelationId.REGISTRY_ID,
        entity: str = "entity-a",
        *,
        issue_now: int = FIXED_AUDIT_TIME,
        gateway_now: int = FIXED_AUDIT_TIME,
    ) -> tuple[D1Environment, Any]:
        current = env(now=gateway_now)
        issued = current.authority.issue(
            _grant(relation, entity), now=issue_now, ttl_seconds=30
        )
        return current, issued

    negative_specs: list[
        tuple[
            str,
            str,
            RejectionReason,
            Callable[[], tuple[CapabilityMemoryProbe, Callable[[], Any]]],
        ]
    ] = []

    def request_case(
        *,
        request_relation: RelationId = RelationId.REGISTRY_ID,
        request_entity: str = "entity-a",
        request_subject: str = "subject-alpha",
        token_relation: RelationId = RelationId.REGISTRY_ID,
        token_entity: str = "entity-a",
        issue_now: int = FIXED_AUDIT_TIME,
        gateway_now: int = FIXED_AUDIT_TIME,
    ) -> tuple[CapabilityMemoryProbe, Callable[[], Any]]:
        current, issued = issued_env(
            token_relation,
            token_entity,
            issue_now=issue_now,
            gateway_now=gateway_now,
        )
        request = _request(
            issued,
            request_relation,
            request_entity,
            subject_id=request_subject,
        )
        return current.memory, lambda: current.gateway.retrieve(
            request, current.memory
        )

    negative_specs.extend(
        (
            (
                "wrong_relation",
                "scope",
                RejectionReason.RELATION_MISMATCH,
                lambda: request_case(request_relation=RelationId.CITY_CODE),
            ),
            (
                "cross_relation_use",
                "scope",
                RejectionReason.RELATION_MISMATCH,
                lambda: request_case(
                    token_relation=RelationId.ACCESS_CODE,
                    token_entity="entity-c",
                    request_relation=RelationId.REGISTRY_ID,
                    request_entity="entity-c",
                ),
            ),
            (
                "wrong_entity",
                "scope",
                RejectionReason.ENTITY_SCOPE_MISMATCH,
                lambda: request_case(request_entity="entity-b"),
            ),
            (
                "cross_entity_use",
                "scope",
                RejectionReason.ENTITY_SCOPE_MISMATCH,
                lambda: request_case(
                    token_relation=RelationId.CITY_CODE,
                    token_entity="entity-b",
                    request_relation=RelationId.CITY_CODE,
                    request_entity="entity-c",
                ),
            ),
            (
                "wrong_subject",
                "identity",
                RejectionReason.SUBJECT_MISMATCH,
                lambda: request_case(request_subject="subject-beta"),
            ),
            (
                "expired_token",
                "time",
                RejectionReason.EXPIRED,
                lambda: request_case(
                    issue_now=FIXED_AUDIT_TIME - 100,
                    gateway_now=FIXED_AUDIT_TIME,
                ),
            ),
            (
                "not_yet_valid_token",
                "time",
                RejectionReason.NOT_YET_VALID,
                lambda: request_case(
                    issue_now=FIXED_AUDIT_TIME + 10,
                    gateway_now=FIXED_AUDIT_TIME,
                ),
            ),
        )
    )

    for scenario, category, reason, factory in negative_specs:
        memory, action = factory()
        _record_case(
            rows,
            scenario_id=scenario,
            category=category,
            expected_status="reject",
            expected_reason=reason,
            memory=memory,
            action=action,
        )

    revoked = env()
    revoked_token = revoked.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=revoked.now,
        ttl_seconds=60,
    )
    revoked.store.revoke(revoked_token.claims.nonce)
    _record_case(
        rows,
        scenario_id="revoked_token",
        category="revocation",
        expected_status="reject",
        expected_reason=RejectionReason.REVOKED_TOKEN,
        memory=revoked.memory,
        action=lambda: revoked.gateway.retrieve(
            _request(
                revoked_token, RelationId.REGISTRY_ID, "entity-a"
            ),
            revoked.memory,
        ),
    )

    revoked_key = env()
    revoked_key_token = revoked_key.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=revoked_key.now,
        ttl_seconds=60,
    )
    revoked_key.keyring.revoke(revoked_key_token.claims.key_id)
    _record_case(
        rows,
        scenario_id="revoked_key",
        category="key_rotation",
        expected_status="reject",
        expected_reason=RejectionReason.REVOKED_KEY,
        memory=revoked_key.memory,
        action=lambda: revoked_key.gateway.retrieve(
            _request(
                revoked_key_token,
                RelationId.REGISTRY_ID,
                "entity-a",
            ),
            revoked_key.memory,
        ),
    )

    foreign = env(key_id="foreign-key")
    foreign_token = foreign.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=foreign.now,
        ttl_seconds=60,
    )
    local = env()
    _record_case(
        rows,
        scenario_id="unknown_key_id",
        category="key_rotation",
        expected_status="reject",
        expected_reason=RejectionReason.UNKNOWN_KEY,
        memory=local.memory,
        action=lambda: local.gateway.retrieve(
            _request(
                foreign_token,
                RelationId.REGISTRY_ID,
                "entity-a",
            ),
            local.memory,
        ),
    )

    for scenario, mutator, reason in (
        ("tampered_payload", _tamper_payload, RejectionReason.BAD_SIGNATURE),
        (
            "tampered_signature",
            _tamper_signature,
            RejectionReason.BAD_SIGNATURE,
        ),
        (
            "algorithm_confusion",
            _tamper_algorithm,
            RejectionReason.INVALID_TOKEN,
        ),
    ):
        current, issued = issued_env()
        request = AuthorizedMemoryRequest(
            "subject-alpha",
            "entity-a",
            RelationId.REGISTRY_ID,
            mutator(issued.token),
        )
        _record_case(
            rows,
            scenario_id=scenario,
            category="integrity",
            expected_status="reject",
            expected_reason=reason,
            memory=current.memory,
            action=lambda c=current, r=request: c.gateway.retrieve(
                r, c.memory
            ),
        )

    for scenario, token in (
        ("malformed_token", b"not.a.valid.extra"),
        (
            "oversized_token",
            b"x" * (int(declaration["max_token_bytes"]) + 1),
        ),
    ):
        current = env()
        request = AuthorizedMemoryRequest(
            "subject-alpha",
            "entity-a",
            RelationId.REGISTRY_ID,
            token,
        )
        _record_case(
            rows,
            scenario_id=scenario,
            category="malformed",
            expected_status="reject",
            expected_reason=RejectionReason.INVALID_TOKEN,
            memory=current.memory,
            action=lambda c=current, r=request: c.gateway.retrieve(
                r, c.memory
            ),
        )

    missing_scope = env()
    _record_case(
        rows,
        scenario_id="missing_scope",
        category="issuance",
        expected_status="reject",
        expected_reason=RejectionReason.INVALID_REQUEST,
        memory=missing_scope.memory,
        action=lambda: CapabilityGrant(
            "subject-alpha",
            (),
            RelationId.REGISTRY_ID,
            ("retrieve",),
        ),
    )

    missing_permission = env()
    _record_case(
        rows,
        scenario_id="missing_permission",
        category="issuance",
        expected_status="reject",
        expected_reason=RejectionReason.UNAUTHORIZED_GRANT,
        memory=missing_permission.memory,
        action=lambda: missing_permission.authority.issue(
            _grant(
                RelationId.REGISTRY_ID,
                "entity-a",
                permissions=("inspect",),
            ),
            now=missing_permission.now,
            ttl_seconds=60,
        ),
    )

    old = env(version="d1-policy-old")
    old_token = old.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=old.now,
        ttl_seconds=60,
    )
    current_policy = _policy(
        str(declaration["policy_version"]),
        max_ttl=int(declaration["max_ttl_seconds"]),
    )
    old.gateway = TrustedCapabilityGateway(
        CapabilityVerifier(
            old.keyring,
            current_policy,
            old.authority.codec,
            old.store,
        ),
        lambda: old.now,
    )
    _record_case(
        rows,
        scenario_id="old_policy_version",
        category="policy",
        expected_status="reject",
        expected_reason=RejectionReason.POLICY_VERSION_MISMATCH,
        memory=old.memory,
        action=lambda: old.gateway.retrieve(
            _request(old_token, RelationId.REGISTRY_ID, "entity-a"),
            old.memory,
        ),
    )

    proposal = env()
    _record_case(
        rows,
        scenario_id="unauthorized_router_proposal",
        category="router_outside_tcb",
        expected_status="reject",
        expected_reason=RejectionReason.ROUTER_PROPOSAL_NOT_AUTHORIZED,
        memory=proposal.memory,
        action=lambda: proposal.gateway.retrieve(
            SuggestedRelation(RelationId.REGISTRY_ID),
            proposal.memory,
        ),
    )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise D1ProtocolError("拒绝写入空 attack matrix")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\r\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    temporary.replace(path)


def _summary(
    values: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    git: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected_positive = set(values["required_positive_scenarios"])
    expected_negative = set(values["required_negative_scenarios"])
    observed_positive = {
        row["scenario_id"] for row in rows if row["expected_status"] == "accept"
    }
    observed_negative = {
        row["scenario_id"] for row in rows if row["expected_status"] == "reject"
    }
    if observed_positive != expected_positive or observed_negative != expected_negative:
        raise D1ProtocolError("D1 attack matrix 与预注册场景不一致")
    negative = [row for row in rows if row["expected_status"] == "reject"]
    positive = [row for row in rows if row["expected_status"] == "accept"]
    unauthorized = sum(int(row["memory_access_delta"]) for row in negative)
    cross_relation = sum(
        int(row["memory_access_delta"])
        for row in rows
        if row["scenario_id"] in {"wrong_relation", "cross_relation_use"}
    )
    cross_entity = sum(
        int(row["memory_access_delta"])
        for row in rows
        if row["scenario_id"] in {"wrong_entity", "cross_entity_use"}
    )
    metrics = {
        "positive_scenario_count": len(positive),
        "negative_scenario_count": len(negative),
        "passed_scenario_count": sum(bool(row["passed"]) for row in rows),
        "failed_scenario_count": sum(not bool(row["passed"]) for row in rows),
        "authorized_memory_access_count": sum(
            int(row["memory_access_delta"]) for row in positive
        ),
        "unauthorized_memory_access_count": unauthorized,
        "cross_relation_access_count": cross_relation,
        "cross_entity_access_count": cross_entity,
        "rejected_request_memory_access_count": unauthorized,
    }
    required = values["required_metrics"]
    gates_passed = (
        metrics["failed_scenario_count"] == 0
        and all(metrics[key] == int(expected) for key, expected in required.items())
    )
    return {
        "schema_version": 1,
        "stage": "D1-trusted-capability-contract-audit",
        "git": dict(git),
        "source_manifest_payload_sha256": source_manifest[
            "manifest_payload_sha256"
        ],
        "cryptographic_primitive": {
            "algorithm": "HMAC-SHA-256",
            "standard": "RFC 2104",
            "implementation": "Python standard library hmac.digest",
            "constant_time_comparison": "hmac.compare_digest",
            "key_generation": "secrets.token_bytes",
            "new_cryptographic_algorithm_claimed": False,
        },
        "metrics": metrics,
        "trusted_capability_contract_status": (
            "passed" if gates_passed else "failed"
        ),
        "authenticated_capability_integrity_passed": gates_passed,
        "scope_enforcement_passed": gates_passed,
        "expiration_revocation_replay_passed": gates_passed,
        "key_rotation_passed": gates_passed,
        "semantic_router_in_tcb": False,
        "suggested_relation_authorizes_memory": False,
        "ready_for_stage_d2_public_synthetic_values": gates_passed,
        "private_value_memory_ready": False,
        "task_specific_router_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "private_value_memory_trained": False,
        "private_answers_loaded": False,
        "lm_answer_injection_executed": False,
        "key_attack_executed": False,
        "confirmation_created_or_read": False,
        "token_material_persisted": False,
        "hmac_key_material_persisted": False,
        "limitations": [
            "token payload 仅认证完整性，不提供机密性",
            "subject_id 绑定不替代外部会话/IAM 对调用者身份的认证",
            "replay/revocation store 仅为单进程内存实现，不具备分布式持久性",
            "HMAC key ring 未接 KMS/HSM，进程内 TCB 仍需独立加固",
            "对称 HMAC verifier 持有签名 key，因此本原型未实现签发方与验证方密码学隔离",
            "Python 原型未实现 router 与 gateway/memory 的进程级隔离",
            "clock 为注入式可信时钟测试，不覆盖真实分布式 clock skew",
            "memory 是不含 value 的 typed access probe，不是 private memory",
            "本结果不是部署级安全认证，也不证明机器遗忘",
        ],
    }


def run_d1_audit(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir = output_paths(config_path)
    if output_dir is not None:
        artifact_dir = resolve_path(config_path, output_dir)
    if artifact_dir.exists() or runtime_dir.exists():
        raise D1ProtocolError("D1 输出或 runtime state 已存在，拒绝重跑")
    git = git_state(config_path)
    if git["tracked_dirty"]:
        raise D1ProtocolError("D1 正式 audit 要求 tracked worktree 干净")
    sources = runtime_source_manifest(config_path)
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    mark_phase_started(runtime_dir, "audit")
    rows = _run_attack_matrix(values)
    summary = _summary(values, rows, git=git, source_manifest=sources)
    summary_path = artifact_dir / "stage_d1_summary.json"
    matrix_path = artifact_dir / "capability_attack_matrix.csv"
    source_path = artifact_dir / "source_sha256_manifest.json"
    resolved_path = artifact_dir / "resolved_config.json"
    status_path = artifact_dir / "protocol_status.json"
    write_json(summary_path, summary)
    _write_csv(matrix_path, rows)
    write_json(source_path, sources)
    write_json(
        resolved_path,
        {
            "schema_version": 1,
            "config": values,
            "config_sha256": sha256_file(config_path),
            "secret_or_token_material_present": False,
        },
    )
    write_json(
        status_path,
        {
            key: summary[key]
            for key in (
                "trusted_capability_contract_status",
                "semantic_router_in_tcb",
                "suggested_relation_authorizes_memory",
                "ready_for_stage_d2_public_synthetic_values",
                "private_value_memory_ready",
                "task_specific_router_ready",
                "original_c3_allowed",
                "c3_eligible",
                "private_value_memory_trained",
                "private_answers_loaded",
                "lm_answer_injection_executed",
                "key_attack_executed",
                "confirmation_created_or_read",
                "token_material_persisted",
                "hmac_key_material_persisted",
            )
        },
    )
    mark_phase_completed(
        runtime_dir,
        "audit",
        (summary_path, matrix_path, source_path, resolved_path, status_path),
    )
    return {
        "status": summary["trusted_capability_contract_status"],
        "metrics": summary["metrics"],
        "ready_for_stage_d2_public_synthetic_values": summary[
            "ready_for_stage_d2_public_synthetic_values"
        ],
        "c3_eligible": False,
        "summary": str(summary_path),
    }


def finalize_d1_artifacts(config_path: str | Path) -> dict[str, Any]:
    artifact_dir, runtime_dir = output_paths(config_path)
    require_phase_completed(runtime_dir, "audit")
    target = artifact_dir / "artifact_sha256_manifest.json"
    if target.exists():
        raise D1ProtocolError("D1 artifact manifest 已存在，拒绝覆盖")
    report = repo_root(config_path) / "PHASE_D1_REPORT.md"
    if not report.is_file():
        raise D1ProtocolError("PHASE_D1_REPORT.md 尚未生成")
    files = [
        {
            "path": path.relative_to(repo_root(config_path)).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path != target
    ]
    files.append(
        {
            "path": "PHASE_D1_REPORT.md",
            "size_bytes": report.stat().st_size,
            "sha256": sha256_file(report),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "D1-final-artifact-manifest",
        "files": files,
        "dataset_present": False,
        "private_value_present": False,
        "private_answer_present": False,
        "token_material_present": False,
        "hmac_key_material_present": False,
        "model_checkpoint_present": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    write_json(target, payload)
    return {
        "artifact_manifest": str(target),
        "file_count": len(files),
        "manifest_file_sha256": sha256_file(target),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
    }


def run_d1_smoke(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    values = load_config(config_path, smoke=True)
    declaration = values["capability"]
    current = _environment(
        policy_version=str(declaration["policy_version"]),
        max_ttl=int(declaration["max_ttl_seconds"]),
        max_token_bytes=int(declaration["max_token_bytes"]),
        key_bytes=int(declaration["key_bytes"]),
    )
    issued = current.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=current.now,
        ttl_seconds=30,
    )
    accepted = current.gateway.retrieve(
        _request(issued, RelationId.REGISTRY_ID, "entity-a"),
        current.memory,
    )
    second = current.authority.issue(
        _grant(RelationId.REGISTRY_ID, "entity-a"),
        now=current.now,
        ttl_seconds=30,
    )
    before = current.memory.memory_access_count
    try:
        current.gateway.retrieve(
            _request(second, RelationId.CITY_CODE, "entity-a"),
            current.memory,
        )
    except CapabilityError as exc:
        reason = exc.reason.value
    else:
        raise AssertionError("smoke wrong relation 未被拒绝")
    result = {
        "schema_version": 1,
        "stage": "D1-in-memory-smoke",
        "accepted_memory_access_count": accepted.memory_access_count,
        "wrong_relation_reason": reason,
        "rejected_request_memory_access_count": (
            current.memory.memory_access_count - before
        ),
        "private_value_memory_used": False,
        "key_material_persisted": False,
        "c3_eligible": False,
    }
    if output_dir is not None:
        destination = resolve_path(config_path, output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "smoke_summary.json", result)
    return result
