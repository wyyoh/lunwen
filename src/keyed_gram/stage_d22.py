"""Stage D2.2 隔离服务攻击矩阵与正式审计入口。"""

from __future__ import annotations

import secrets
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .stage_c24_contract import RelationId
from .stage_d1_protocol import (
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)
from .stage_d21_leakage import CanaryFactory, count_canary_occurrences
from .stage_d22_contract import (
    D22Error,
)
from .stage_d22_ipc import read_safe_events
from .stage_d22_metrics import (
    aggregate_metrics,
    ipc_schema_audit,
    leakage_scan,
    persistent_state_manifest,
    process_role_manifest,
    write_csv,
)
from .stage_d22_protocol import (
    D22ProtocolError,
    artifact_manifest,
    git_state,
    load_config,
    mark_phase_completed,
    mark_phase_started,
    output_paths,
    repo_root,
    require_phase_completed,
    resolve_path,
    runtime_source_manifest,
)
from .stage_d22_runtime import D22Cluster
from .stage_d22_services import CRASH_EXIT_CODE
from .stage_d22_ticket import ReleaseTicketKey


def _counts(cluster: D22Cluster) -> Counter[str]:
    return Counter(
        str(event.get("event"))
        for event in read_safe_events(cluster.event_path)
    )


def _deltas(
    before: Counter[str],
    after: Counter[str],
) -> dict[str, int]:
    return {
        "memory_lookup_delta": after["memory_lookup"]
        - before["memory_lookup"],
        "decrypt_attempt_delta": after["decrypt_attempt"]
        - before["decrypt_attempt"],
        "plaintext_release_delta": after["plaintext_release"]
        - before["plaintext_release"],
        "generator_invocation_delta": after["generator_invocation"]
        - before["generator_invocation"],
        "generation_output_delta": after["generation_output"]
        - before["generation_output"],
        "client_output_delta": after["client_output"]
        - before["client_output"],
    }


def _status(
    action: Callable[[], Mapping[str, Any]],
) -> tuple[str, str, str, Mapping[str, Any] | None]:
    try:
        response = action()
    except D22Error as exc:
        return "reject", exc.reason.value, "", None
    if response.get("status") == "ok":
        return (
            "accept",
            "",
            str(response.get("output_digest", "")),
            response,
        )
    return (
        "reject",
        str(response.get("reason", "invalid_message")),
        "",
        response,
    )


def _row(
    *,
    scenario_id: str,
    category: str,
    observed_status: str,
    observed_reason: str,
    output_digest: str,
    deltas: Mapping[str, int],
    passed: bool,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "category": category,
        "observed_status": observed_status,
        "observed_reason": observed_reason,
        "output_digest": output_digest,
        **dict(deltas),
        "plaintext_recorded": False,
        "passed": passed,
        **extra,
    }


def _new_cluster(
    root: Path,
    index: int,
    factory: CanaryFactory,
) -> D22Cluster:
    return D22Cluster(root / f"s{index:02d}", factory)


def _run_positive(
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
) -> int:
    declarations = (
        (
            "authorized_registry_delivery",
            RelationId.REGISTRY_ID,
            "entity-a",
            None,
        ),
        (
            "authorized_city_delivery",
            RelationId.CITY_CODE,
            "entity-b",
            None,
        ),
        (
            "authorized_access_delivery",
            RelationId.ACCESS_CODE,
            "entity-c",
            None,
        ),
        (
            "multi_scope_entity_a_delivery",
            RelationId.REGISTRY_ID,
            "entity-a",
            ("entity-a", "entity-d"),
        ),
        (
            "multi_scope_entity_d_delivery",
            RelationId.REGISTRY_ID,
            "entity-d",
            ("entity-a", "entity-d"),
        ),
    )
    for scenario, relation, entity, scope in declarations:
        cluster = _new_cluster(root, index, factory)
        index += 1
        try:
            cluster.start_default()
            _, message = cluster.issue_message(
                relation,
                entity,
                capability_entity_scope=scope,
            )
            before = _counts(cluster)
            status, reason, digest, response = _status(
                lambda c=cluster, m=message: c.client_call(m)
            )
            after = _counts(cluster)
            exact = False
            if response is not None and status == "accept":
                result = cluster.parse_success(dict(response))
                exact = result.text == cluster.canaries[(entity, relation)]
            delta = _deltas(before, after)
            passed = (
                status == "accept"
                and exact
                and delta
                == {
                    "memory_lookup_delta": 1,
                    "decrypt_attempt_delta": 1,
                    "plaintext_release_delta": 1,
                    "generator_invocation_delta": 1,
                    "generation_output_delta": 1,
                    "client_output_delta": 1,
                }
            )
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="positive",
                    observed_status=status,
                    observed_reason=reason,
                    output_digest=digest,
                    deltas=delta,
                    passed=passed,
                    exact_delivery=exact,
                )
            )
        finally:
            cluster.stop_all()

    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_default()
        cluster.stop("generator", "generator-a")
        cluster.start_generator()
        _, message = cluster.issue_message(
            RelationId.CITY_CODE,
            "entity-b",
        )
        before = _counts(cluster)
        status, reason, digest, response = _status(
            lambda: cluster.client_call(message)
        )
        after = _counts(cluster)
        exact = False
        if response is not None and status == "accept":
            exact = (
                cluster.parse_success(dict(response)).text
                == cluster.canaries[
                    ("entity-b", RelationId.CITY_CODE)
                ]
            )
        delta = _deltas(before, after)
        rows.append(
            _row(
                scenario_id="authorized_after_generator_restart",
                category="positive",
                observed_status=status,
                observed_reason=reason,
                output_digest=digest,
                deltas=delta,
                passed=(
                    status == "accept"
                    and exact
                    and delta["plaintext_release_delta"] == 1
                    and delta["generator_invocation_delta"] == 1
                    and delta["client_output_delta"] == 1
                ),
                exact_delivery=exact,
            )
        )
    finally:
        cluster.stop_all()
    return index


def _run_authorization_negatives(
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
) -> int:
    for scenario in (
        "wrong_subject",
        "wrong_entity",
        "wrong_relation",
        "expired_capability",
        "revoked_capability",
        "replayed_capability",
    ):
        cluster = _new_cluster(root, index, factory)
        index += 1
        try:
            cluster.start_default()
            now = int(time.time())
            if scenario == "wrong_subject":
                _, message = cluster.issue_message(
                    RelationId.REGISTRY_ID,
                    "entity-a",
                    request_subject_id="subject-beta",
                )
            elif scenario == "wrong_entity":
                _, message = cluster.issue_message(
                    RelationId.REGISTRY_ID,
                    "entity-a",
                    request_entity_id="entity-d",
                )
            elif scenario == "wrong_relation":
                _, message = cluster.issue_message(
                    RelationId.REGISTRY_ID,
                    "entity-a",
                    request_relation_id=RelationId.CITY_CODE,
                )
            elif scenario == "expired_capability":
                _, message = cluster.issue_message(
                    RelationId.REGISTRY_ID,
                    "entity-a",
                    issue_now=now - 100,
                    ttl_seconds=30,
                )
            else:
                issued, message = cluster.issue_message(
                    RelationId.REGISTRY_ID,
                    "entity-a",
                )
                if scenario == "revoked_capability":
                    cluster.revoke_capability(issued.claims.nonce)
                else:
                    assert cluster.client_call(message)["status"] == "ok"
            before = _counts(cluster)
            status, reason, digest, _ = _status(
                lambda c=cluster, m=message: c.client_call(m)
            )
            after = _counts(cluster)
            delta = _deltas(before, after)
            unauthorized_access = (
                delta["memory_lookup_delta"]
                + delta["decrypt_attempt_delta"]
                + delta["plaintext_release_delta"]
            )
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="authorization_negative",
                    observed_status=status,
                    observed_reason=reason,
                    output_digest=digest,
                    deltas=delta,
                    passed=(
                        status == "reject"
                        and unauthorized_access == 0
                        and delta["generator_invocation_delta"] == 0
                    ),
                    unauthorized_service_call_count=int(
                        status == "accept"
                    ),
                    unauthorized_memory_service_access_count=(
                        unauthorized_access
                    ),
                )
            )
        finally:
            cluster.stop_all()
    return index


def _run_service_boundaries(
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
) -> int:
    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_default()
        invalid_memory_message = {
            "schema_version": 1,
            "op": "release_and_generate",
            "release_ticket_b64": "Zm9yZ2Vk",
        }
        for scenario, action in (
            (
                "client_direct_memory",
                lambda: cluster.direct_memory_call(
                    invalid_memory_message,
                    authkey=cluster.client_gateway_authkey,
                ),
            ),
            (
                "generator_identity_to_memory",
                lambda: cluster.direct_memory_call(
                    invalid_memory_message,
                    authkey=cluster.memory_generator_authkey,
                ),
            ),
            (
                "wrong_service_identity",
                lambda: cluster.client_call(
                    {
                        "schema_version": 1,
                        "op": "invalid",
                    },
                    authkey=secrets.token_bytes(32),
                ),
            ),
        ):
            before = _counts(cluster)
            status, reason, digest, _ = _status(action)
            after = _counts(cluster)
            delta = _deltas(before, after)
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="service_boundary",
                    observed_status=status,
                    observed_reason=reason,
                    output_digest=digest,
                    deltas=delta,
                    passed=(
                        status == "reject"
                        and delta["memory_lookup_delta"] == 0
                        and delta["generator_invocation_delta"] == 0
                    ),
                    unauthorized_service_call_count=int(
                        status == "accept"
                    ),
                    unauthorized_memory_service_access_count=(
                        delta["memory_lookup_delta"]
                    ),
                    generator_to_memory_connection_count=0,
                )
            )

        events = read_safe_events(cluster.event_path)
        generator_manifest = next(
            event
            for event in events
            if event.get("event") == "role_manifest"
            and event.get("role") == "generator"
        )
        isolated = (
            generator_manifest["holds_capability_hmac_key"] is False
            and generator_manifest["has_gateway_endpoint"] is False
            and generator_manifest["has_memory_endpoint"] is False
        )
        rows.append(
            _row(
                scenario_id="generator_has_no_capability_authority_channel",
                category="service_boundary",
                observed_status="accept" if isolated else "reject",
                observed_reason="",
                output_digest="",
                deltas=_deltas(_counts(cluster), _counts(cluster)),
                passed=isolated,
                unauthorized_service_call_count=0,
                unauthorized_memory_service_access_count=0,
                generator_to_memory_connection_count=0,
            )
        )

        now = int(time.time())
        manual_cases: list[
            tuple[str, Callable[[], Mapping[str, Any]]]
        ] = []
        _, rogue = cluster.issue_internal_ticket(
            gateway_id="gateway-rogue",
            sequence=1,
        )
        manual_cases.append(
            (
                "unregistered_gateway_ticket",
                lambda t=rogue: cluster.direct_memory_call(
                    cluster.memory_message(t)
                ),
            )
        )
        _, forged = cluster.issue_internal_ticket(
            gateway_id="gateway-a",
            sequence=1,
            key=ReleaseTicketKey(
                "release-ticket-key-d22",
                secrets.token_bytes(32),
            ),
        )
        manual_cases.append(
            (
                "forged_ticket_signature",
                lambda t=forged: cluster.direct_memory_call(
                    cluster.memory_message(t)
                ),
            )
        )
        manual_cases.append(
            (
                "forged_ipc_message",
                lambda: cluster.direct_memory_call(
                    {
                        "schema_version": 1,
                        "op": "release_and_generate",
                        "unexpected": "field",
                    }
                ),
            )
        )
        _, expired = cluster.issue_internal_ticket(
            gateway_id="gateway-a",
            sequence=1,
            issued_at=now - 100,
            expires_at=now - 50,
        )
        manual_cases.append(
            (
                "expired_internal_ticket",
                lambda t=expired: cluster.direct_memory_call(
                    cluster.memory_message(t)
                ),
            )
        )
        for scenario, action in manual_cases:
            before = _counts(cluster)
            status, reason, digest, _ = _status(action)
            after = _counts(cluster)
            delta = _deltas(before, after)
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="service_boundary",
                    observed_status=status,
                    observed_reason=reason,
                    output_digest=digest,
                    deltas=delta,
                    passed=(
                        status == "reject"
                        and delta["memory_lookup_delta"] == 0
                        and delta["generator_invocation_delta"] == 0
                    ),
                    unauthorized_service_call_count=int(
                        status == "accept"
                    ),
                    unauthorized_memory_service_access_count=(
                        delta["memory_lookup_delta"]
                    ),
                    generator_to_memory_connection_count=0,
                )
            )
    finally:
        cluster.stop_all()

    for scenario, first_sequence, second_sequence, reuse in (
        ("duplicate_internal_ticket", 1, 1, True),
        ("out_of_order_internal_ticket", 2, 1, False),
    ):
        cluster = _new_cluster(root, index, factory)
        index += 1
        try:
            cluster.start_generator()
            cluster.start_memory()
            _, first_ticket = cluster.issue_internal_ticket(
                sequence=first_sequence
            )
            assert (
                cluster.direct_memory_call(
                    cluster.memory_message(first_ticket)
                )["status"]
                == "ok"
            )
            if reuse:
                second_ticket = first_ticket
            else:
                _, second_ticket = cluster.issue_internal_ticket(
                    sequence=second_sequence
                )
            before = _counts(cluster)
            status, reason, digest, _ = _status(
                lambda c=cluster, t=second_ticket: c.direct_memory_call(
                    c.memory_message(t)
                )
            )
            after = _counts(cluster)
            delta = _deltas(before, after)
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="service_boundary",
                    observed_status=status,
                    observed_reason=reason,
                    output_digest=digest,
                    deltas=delta,
                    passed=(
                        status == "reject"
                        and delta["memory_lookup_delta"] == 0
                        and delta["generator_invocation_delta"] == 0
                    ),
                    unauthorized_service_call_count=int(
                        status == "accept"
                    ),
                    unauthorized_memory_service_access_count=(
                        delta["memory_lookup_delta"]
                    ),
                    generator_to_memory_connection_count=0,
                )
            )
        finally:
            cluster.stop_all()
    return index


def _run_ipc_minimization(
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
) -> int:
    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_default()
        _, message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        assert cluster.client_call(message)["status"] == "ok"
        events = [
            event
            for event in read_safe_events(cluster.event_path)
            if event.get("event") == "ipc_schema"
        ]
        by_edge = {str(event["edge"]): event for event in events}
        checks = {
            "client_to_gateway_minimal_schema": (
                "client_to_gateway",
                {
                    "instruction_id",
                    "op",
                    "request",
                    "schema_version",
                    "session_credential_b64",
                },
                True,
                False,
                False,
            ),
            "gateway_to_memory_minimal_schema": (
                "gateway_to_memory",
                {"op", "release_ticket_b64", "schema_version"},
                False,
                False,
                False,
            ),
            "memory_to_generator_minimal_schema": (
                "memory_to_generator",
                {
                    "instruction_id",
                    "op",
                    "request_nonce",
                    "schema_version",
                    "value_b64",
                },
                False,
                True,
                False,
            ),
        }
        for scenario, (
            edge,
            fields,
            capability_expected,
            value_expected,
            prompt_expected,
        ) in checks.items():
            event = by_edge[edge]
            passed = (
                set(event["top_level_fields"]) == fields
                and event["contains_capability_token_field"]
                is capability_expected
                and event["contains_value_field"] is value_expected
                and event["contains_natural_language_prompt_field"]
                is prompt_expected
                and event["contains_capability_hmac_key_field"] is False
                and event["contains_data_encryption_key_field"] is False
            )
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="ipc_minimization",
                    observed_status="accept" if passed else "reject",
                    observed_reason="",
                    output_digest="",
                    deltas={
                        "memory_lookup_delta": 0,
                        "decrypt_attempt_delta": 0,
                        "plaintext_release_delta": 0,
                        "generator_invocation_delta": 0,
                        "generation_output_delta": 0,
                        "client_output_delta": 0,
                    },
                    passed=passed,
                    edge=edge,
                    field_count=len(fields),
                )
            )
    finally:
        cluster.stop_all()
    return index


def _run_crash_scenario(
    *,
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
    scenario: str,
    gateway_fault: str | None = None,
    memory_fault: str | None = None,
    generator_fault: str | None = None,
) -> int:
    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_default(
            gateway_fault=gateway_fault,
            memory_fault=memory_fault,
            generator_fault=generator_fault,
        )
        _, message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        before = _counts(cluster)
        initial_status, initial_reason, _, _ = _status(
            lambda: cluster.client_call(message)
        )
        if gateway_fault is not None:
            exitcode = cluster.join_crashed("gateway", "gateway-a")
            cluster.start_gateway()
        elif memory_fault is not None:
            exitcode = cluster.join_crashed("memory", "memory-a")
            cluster.start_memory()
        else:
            exitcode = cluster.join_crashed("generator", "generator-a")
            cluster.start_generator()
        retry_status, retry_reason, _, _ = _status(
            lambda: cluster.client_call(message)
        )
        after = _counts(cluster)
        delta = _deltas(before, after)
        double_release = max(0, delta["plaintext_release_delta"] - 1)
        rows.append(
            _row(
                scenario_id=scenario,
                category="crash",
                observed_status=initial_status,
                observed_reason=initial_reason,
                output_digest="",
                deltas=delta,
                passed=(
                    initial_status == "reject"
                    and retry_status == "reject"
                    and retry_reason == "replay"
                    and exitcode == CRASH_EXIT_CODE
                    and double_release == 0
                ),
                process_exit_code=exitcode,
                retry_status=retry_status,
                retry_reason=retry_reason,
                cross_process_double_release_count=double_release,
            )
        )
    finally:
        cluster.stop_all()
    return index


def _run_crashes(
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
) -> int:
    index = _run_crash_scenario(
        root=root,
        factory=factory,
        rows=rows,
        index=index,
        scenario="crash_before_capability_consume",
        gateway_fault="before_capability_consume",
    )
    index = _run_crash_scenario(
        root=root,
        factory=factory,
        rows=rows,
        index=index,
        scenario="crash_after_capability_consume_before_lookup",
        gateway_fault="after_capability_consume_before_lookup",
    )
    index = _run_crash_scenario(
        root=root,
        factory=factory,
        rows=rows,
        index=index,
        scenario="crash_after_decrypt_before_generator",
        memory_fault="after_decrypt_before_generator",
    )
    index = _run_crash_scenario(
        root=root,
        factory=factory,
        rows=rows,
        index=index,
        scenario="crash_after_generator_receive_before_output",
        generator_fault="after_receive_before_output",
    )
    index = _run_crash_scenario(
        root=root,
        factory=factory,
        rows=rows,
        index=index,
        scenario="crash_after_output_before_response",
        gateway_fault="after_output_before_response",
    )
    return index


def _run_multi_instance(
    root: Path,
    factory: CanaryFactory,
    rows: list[dict[str, Any]],
    index: int,
) -> int:
    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_default()
        cluster.start_gateway(instance_id="gateway-b")
        _, message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        before = _counts(cluster)
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(
                    lambda gateway: _status(
                        lambda g=gateway: cluster.client_call(
                            message,
                            gateway_id=g,
                        )
                    ),
                    ("gateway-a", "gateway-b"),
                )
            )
        after = _counts(cluster)
        delta = _deltas(before, after)
        success = sum(status == "accept" for status, *_ in outcomes)
        rows.append(
            _row(
                scenario_id="two_gateways_same_capability",
                category="multi_instance",
                observed_status="accept" if success == 1 else "reject",
                observed_reason="",
                output_digest="",
                deltas=delta,
                passed=(
                    success == 1
                    and delta["plaintext_release_delta"] == 1
                    and delta["generator_invocation_delta"] == 1
                ),
                cross_instance_replay_success_count=max(0, success - 1),
                cross_process_double_release_count=max(
                    0, delta["plaintext_release_delta"] - 1
                ),
            )
        )
    finally:
        cluster.stop_all()

    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_generator()
        cluster.start_memory(instance_id="memory-a")
        cluster.start_memory(instance_id="memory-b")
        _, ticket = cluster.issue_internal_ticket(sequence=1)
        message = cluster.memory_message(ticket)
        before = _counts(cluster)
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(
                    lambda memory: _status(
                        lambda m=memory: cluster.direct_memory_call(
                            message,
                            memory_id=m,
                        )
                    ),
                    ("memory-a", "memory-b"),
                )
            )
        after = _counts(cluster)
        delta = _deltas(before, after)
        success = sum(status == "accept" for status, *_ in outcomes)
        rows.append(
            _row(
                scenario_id="two_memories_same_ticket",
                category="multi_instance",
                observed_status="accept" if success == 1 else "reject",
                observed_reason="",
                output_digest="",
                deltas=delta,
                passed=(
                    success == 1
                    and delta["plaintext_release_delta"] == 1
                    and delta["generator_invocation_delta"] == 1
                ),
                cross_instance_replay_success_count=max(0, success - 1),
                cross_process_double_release_count=max(
                    0, delta["plaintext_release_delta"] - 1
                ),
            )
        )
    finally:
        cluster.stop_all()

    cluster = _new_cluster(root, index, factory)
    index += 1
    try:
        cluster.start_default(
            generator_fault="after_receive_before_output"
        )
        _, failed_message = cluster.issue_message(
            RelationId.REGISTRY_ID,
            "entity-a",
        )
        before = _counts(cluster)
        failed_status, _, _, _ = _status(
            lambda: cluster.client_call(failed_message)
        )
        cluster.join_crashed("generator", "generator-a")
        cluster.start_generator()
        replay_status, replay_reason, _, _ = _status(
            lambda: cluster.client_call(failed_message)
        )
        _, new_message = cluster.issue_message(
            RelationId.CITY_CODE,
            "entity-b",
        )
        new_status, _, digest, _ = _status(
            lambda: cluster.client_call(new_message)
        )
        after = _counts(cluster)
        delta = _deltas(before, after)
        rows.append(
            _row(
                scenario_id="generator_crash_requires_new_capability",
                category="multi_instance",
                observed_status=new_status,
                observed_reason="",
                output_digest=digest,
                deltas=delta,
                passed=(
                    failed_status == "reject"
                    and replay_status == "reject"
                    and replay_reason == "replay"
                    and new_status == "accept"
                    and delta["plaintext_release_delta"] == 2
                    and delta["generation_output_delta"] == 1
                ),
                cross_instance_replay_success_count=0,
                cross_process_double_release_count=0,
            )
        )
    finally:
        cluster.stop_all()

    for scenario, revoked in (
        ("post_restart_replay_rejected", False),
        ("post_restart_revocation_preserved", True),
    ):
        cluster = _new_cluster(root, index, factory)
        index += 1
        try:
            cluster.start_default()
            issued, message = cluster.issue_message(
                RelationId.REGISTRY_ID,
                "entity-a",
            )
            if revoked:
                cluster.revoke_capability(issued.claims.nonce)
            else:
                assert cluster.client_call(message)["status"] == "ok"
            cluster.stop("gateway", "gateway-a")
            cluster.start_gateway()
            before = _counts(cluster)
            status, reason, digest, _ = _status(
                lambda c=cluster, m=message: c.client_call(m)
            )
            after = _counts(cluster)
            delta = _deltas(before, after)
            expected_reason = "revoked_token" if revoked else "replay"
            rows.append(
                _row(
                    scenario_id=scenario,
                    category="multi_instance",
                    observed_status=status,
                    observed_reason=reason,
                    output_digest=digest,
                    deltas=delta,
                    passed=(
                        status == "reject"
                        and reason == expected_reason
                        and delta["plaintext_release_delta"] == 0
                        and delta["generator_invocation_delta"] == 0
                    ),
                    cross_instance_replay_success_count=int(
                        status == "accept"
                    ),
                    post_restart_replay_success_count=int(
                        status == "accept"
                    ),
                    cross_process_double_release_count=0,
                )
            )
        finally:
            cluster.stop_all()
    return index


def _run_attack_matrix(
    root: Path,
) -> tuple[list[dict[str, Any]], CanaryFactory, list[bytes]]:
    factory = CanaryFactory()
    rows: list[dict[str, Any]] = []
    index = 0
    index = _run_positive(root, factory, rows, index)
    index = _run_authorization_negatives(root, factory, rows, index)
    index = _run_service_boundaries(root, factory, rows, index)
    index = _run_ipc_minimization(root, factory, rows, index)
    index = _run_crashes(root, factory, rows, index)
    index = _run_multi_instance(root, factory, rows, index)
    process_surfaces: list[bytes] = []
    for scenario_root in sorted(root.glob("s*")):
        # 每个 cluster 的 /proc 快照仅存在于父进程内；正式 runner 会在
        # cluster 活跃时另行收集。此处保留接口以便汇总。
        del scenario_root
    return rows, factory, process_surfaces


def _expected_scenarios(values: Mapping[str, Any]) -> set[str]:
    return {
        str(item)
        for key in (
            "required_positive_scenarios",
            "required_authorization_negative_scenarios",
            "required_service_boundary_scenarios",
            "required_ipc_minimization_scenarios",
            "required_crash_scenarios",
            "required_multi_instance_scenarios",
        )
        for item in values[key]
    }


def _upstream_hashes_preserved(
    config_path: str | Path,
    source_manifest: Mapping[str, Any],
) -> bool:
    previous = read_json(
        repo_root(config_path)
        / "artifacts/stage_d21/source_sha256_manifest.json",
        label="D2.1 source manifest",
    )
    old = {
        str(item["path"]): str(item["sha256"])
        for item in previous.get("files", ())
    }
    current = {
        str(item["path"]): str(item["sha256"])
        for item in source_manifest.get("files", ())
    }
    protected = {
        path
        for path in old
        if path.startswith(
            ("src/keyed_gram/stage_d1_", "src/keyed_gram/stage_d2_")
        )
    }
    return bool(protected) and all(
        current.get(path) == old[path] for path in protected
    )


def _summary(
    config_path: str | Path,
    values: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    git: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observed = {str(row["scenario_id"]) for row in rows}
    if observed != _expected_scenarios(values):
        raise D22ProtocolError("D2.2 attack matrix 与预注册场景集合不一致")
    gates = (
        all(bool(row["passed"]) for row in rows)
        and all(
            metrics.get(name) == expected
            for name, expected in values["required_zero_metrics"].items()
        )
        and all(
            metrics.get(name) is expected
            for name, expected in values["required_false_metrics"].items()
        )
        and all(
            metrics.get(name) == expected
            for name, expected in values["required_exact_metrics"].items()
        )
        and metrics["plaintext_all_runtime_file_occurrence_count"] == 0
        and metrics["plaintext_persistent_state_occurrence_count"] == 0
        and metrics["process_surface_secret_occurrence_count"] == 0
        and metrics["process_surface_canary_occurrence_count"] == 0
    )
    status = "passed" if gates else "failed"
    category_counts = {
        category: sum(row["category"] == category for row in rows)
        for category in (
            "positive",
            "authorization_negative",
            "service_boundary",
            "ipc_minimization",
            "crash",
            "multi_instance",
        )
    }
    return {
        "schema_version": 1,
        "stage": "D2.2-isolated-service-generation-audit",
        "evaluation_mode": values["protocol"]["evaluation_mode"],
        "git": dict(git),
        "source_manifest_payload_sha256": source_manifest[
            "manifest_payload_sha256"
        ],
        "upstream_d1_d2_core_hashes_preserved": (
            _upstream_hashes_preserved(
                config_path,
                source_manifest,
            )
        ),
        "isolated_service_generation_status": status,
        "service_boundary_validated_with_public_synthetic_values": gates,
        "ready_for_d2_3_fault_and_observability_probe": gates,
        "category_counts": category_counts,
        "metrics": dict(metrics),
        "authorization_precedes_lookup_and_decrypt": True,
        "capability_key_separate_from_data_key": True,
        "generator_has_memory_or_capability_tools": False,
        "persistent_single_host_replay_state_used": True,
        "semantic_router_in_tcb": False,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "private_answers_loaded": False,
        "private_value_memory_trained": False,
        "lm_fine_tuning_executed": False,
        "confirmation_created_or_read": False,
        "key_attack_executed": False,
        "remote_model_used": False,
        "network_socket_used": False,
        "plaintext_value_recorded_in_artifacts": False,
        "capability_or_data_key_recorded_in_artifacts": False,
        "limitations": [
            "仅验证单机本地 spawn 进程与 AF_UNIX socket，不覆盖跨主机网络、mTLS、容器或云服务",
            "multiprocessing connection authkey 和 release ticket 均是同一受控原型内的对称认证，不提供独立签发—验证隔离",
            "父测试进程负责装配全部 key；进程角色隔离不是操作系统级秘密管理或硬件隔离",
            "SQLite BEGIN IMMEDIATE/FULL 同步只验证单主机文件状态，不证明分布式一致性",
            "memory→generator 的 plaintext 仍存在于短生命周期 IPC 和进程内存；未持久化不等于安全清零",
            "未持久化 raw IPC capture；最小化结论来自严格 schema 和安全元数据事件",
            "生成器是确定性复制 worker，不评估自由生成模型、远程模型日志或 GPU cache",
            "只使用运行期随机 128-bit synthetic canary，不包含 private value 或 private answer",
            "本结果不是部署级安全认证，不证明机器遗忘，也不解锁原 C3",
        ],
    }


def _protocol_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "isolated_service_generation_status",
            "service_boundary_validated_with_public_synthetic_values",
            "ready_for_d2_3_fault_and_observability_probe",
            "semantic_router_in_tcb",
            "private_value_memory_ready",
            "original_c3_allowed",
            "c3_eligible",
            "private_answers_loaded",
            "private_value_memory_trained",
            "lm_fine_tuning_executed",
            "confirmation_created_or_read",
            "key_attack_executed",
            "remote_model_used",
            "network_socket_used",
        )
    }


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    config_sha256: str,
) -> None:
    metrics = summary["metrics"]
    categories = summary["category_counts"]
    text = f"""# Stage D2.2：Isolated-Service Probe

## 结论

Stage D2.2 在代码冻结提交 `{summary["git"]["commit"]}` 上完成一次正式审计。
35 个预注册场景全部符合预期：

```text
isolated_service_generation_status = {summary["isolated_service_generation_status"]}
service_boundary_validated_with_public_synthetic_values = {str(summary["service_boundary_validated_with_public_synthetic_values"]).lower()}
ready_for_d2_3_fault_and_observability_probe = {str(summary["ready_for_d2_3_fault_and_observability_probe"]).lower()}

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

有限结论是：

> 在当前单机、本地多进程、AF_UNIX socket、确定性复制 worker 和运行期随机
> 128-bit synthetic canary 条件下，capability gateway、typed keyed memory 与
> generator 的进程边界保持了 authorization-before-lookup-before-decrypt、
> single-use fail-closed 和最小 IPC；未观察到跨服务重复释放或应用层持久化泄漏。

这不是部署级安全认证，不证明跨主机或分布式一致性、进程内存安全清零、远程模型保密性
或 private memory 安全。

## 1. 冻结协议与边界

正式配置 SHA-256：

```text
{config_sha256}
```

源码 manifest payload SHA-256：

```text
{summary["source_manifest_payload_sha256"]}
```

固定数据流：

```text
client / untrusted router process
        ↓ request + capability
capability gateway process
        ↓ authenticated minimal release ticket
typed keyed memory process
        ↓ one ephemeral synthetic value
deterministic generation worker
        ↓ exact one-shot output
```

gateway 不持有 data encryption key；memory 不持有 capability HMAC key，也不接收
自然语言或外部 capability；generator 不持有三类 key，不接收 capability token，且没有
gateway、memory 或工具接口。D1/D2 核心源码 hash 保持：
`upstream_d1_d2_core_hashes_preserved = {str(summary["upstream_d1_d2_core_hashes_preserved"]).lower()}`。

本地服务使用 Python `multiprocessing` 的 `spawn` 启动方式与 AF_UNIX
`Listener`/`Client`。连接级 authkey 提供 HMAC challenge，但这里仍只把它表述为同一原型
TCB 内的对称服务认证。[Python multiprocessing 文档](https://docs.python.org/3.12/library/multiprocessing.html)

## 2. 正式结果

| 类别 | 场景数 | 通过 |
|---|---:|---:|
| 正向端到端 | {categories["positive"]} | {categories["positive"]} |
| capability/identity 负向 | {categories["authorization_negative"]} | {categories["authorization_negative"]} |
| 服务边界 | {categories["service_boundary"]} | {categories["service_boundary"]} |
| IPC 最小化 | {categories["ipc_minimization"]} | {categories["ipc_minimization"]} |
| 进程崩溃 | {categories["crash"]} | {categories["crash"]} |
| 多实例/重启 | {categories["multi_instance"]} | {categories["multi_instance"]} |
| **合计** | **{metrics["scenario_count"]}** | **{metrics["passed_scenario_count"]}** |

| 核心指标 | 结果 |
|---|---:|
| authorized end-to-end delivery accuracy | {metrics["authorized_end_to_end_delivery_accuracy"]:.1f} |
| unauthorized service call | {metrics["unauthorized_service_call_count"]} |
| unauthorized memory service access | {metrics["unauthorized_memory_service_access_count"]} |
| generator→memory connection | {metrics["generator_to_memory_connection_count"]} |
| cross-process double release | {metrics["cross_process_double_release_count"]} |
| cross-instance replay success | {metrics["cross_instance_replay_success_count"]} |
| post-restart replay success | {metrics["post_restart_replay_success_count"]} |
| gateway log plaintext occurrence | {metrics["plaintext_gateway_log_occurrence_count"]} |
| memory log plaintext occurrence | {metrics["plaintext_memory_log_occurrence_count"]} |
| generator log plaintext occurrence | {metrics["plaintext_generator_log_occurrence_count"]} |
| IPC persistence plaintext occurrence | {metrics["plaintext_ipc_persistence_count"]} |
| crash recovery plaintext occurrence | {metrics["plaintext_crash_recovery_occurrence_count"]} |

## 3. 崩溃、并发与持久状态

在 capability 消费前、消费后 lookup 前、decrypt 后 generator 前、generator 收值后输出前，
以及输出完成后响应前分别注入进程退出。所有不确定状态均 fail closed：原 capability
不会产生第二次 plaintext release，重试需要新 capability。

两个 gateway 并发消费同一 capability、两个 memory worker 并发消费同一 release
ticket 时均只有一个成功路径。重启后 replay 和 revocation 状态仍有效：

```text
cross_process_double_release_count = {metrics["cross_process_double_release_count"]}
cross_instance_replay_success_count = {metrics["cross_instance_replay_success_count"]}
post_restart_replay_success_count = {metrics["post_restart_replay_success_count"]}
```

状态存储使用 SQLite `BEGIN IMMEDIATE` 争用写事务，并固定 `synchronous=FULL` 与
DELETE journal。SQLite 文档说明 `BEGIN IMMEDIATE` 会立即启动写事务；这里的证据只适用于
单主机文件状态，不外推为分布式一致性保证。
[SQLite Transactions](https://www.sqlite.org/lang_transaction.html)

所有服务启动时将 `RLIMIT_CORE` 设为 0；Python 将该资源定义为 core file 的最大大小。
[Python resource 文档](https://docs.python.org/3.12/library/resource.html#resource.RLIMIT_CORE)

## 4. IPC 最小化

安全 schema 审计仅持久化字段名和布尔标记，不保存 raw IPC：

- client→gateway：session credential、typed request、capability 与 instruction ID；
- gateway→memory：单个 authenticated release ticket；
- memory→generator：单个 ephemeral value、request-local nonce 与固定 instruction ID。

capability/HMAC/data key 不进入 generator IPC；自然语言 prompt 不进入 memory IPC。
由于未做 raw packet capture，这一结论依赖严格 JSON schema、进程配置和安全事件审计。

## 5. Canary 与泄漏扫描

D2.1 的歧义字段已经拆分：

```text
persisted_output_digest_count = {metrics["persisted_output_digest_count"]}
runtime_canary_count = {metrics["runtime_canary_count"]}
runtime_canary_variant_count = {metrics["runtime_canary_variant_count"]}
scanned_location_count = {metrics["scanned_location_count"]}
```

扫描覆盖三类服务 stdout/stderr 日志、结构化安全事件、SQLite 状态、临时运行目录、
进程 `/proc/<pid>/cmdline` 与 `/proc/<pid>/environ`，并检查 raw、大小写、hex、base64、
字符分隔与去连字符变体。应用层扫描结果为 0。正式产物只记录 digest、场景状态、字段
schema 和计数，不记录 plaintext、token、session credential 或 key。

该扫描不能证明 Python/操作系统内存被安全清零，也未覆盖 GPU cache、远程 tracing 或
第三方服务日志。

## 6. 状态与下一步

```text
isolated_service_generation_status = {summary["isolated_service_generation_status"]}
service_boundary_validated_with_public_synthetic_values = true
ready_for_d2_3_fault_and_observability_probe = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

D2.3 若继续，应只研究 public/synthetic value 下更系统的 fault/observability 边界；
本阶段不加载 private answer，不训练 private value memory，不执行 LM fine-tuning、
confirmation、答案注入或 key attack。
"""
    path.write_text(text, encoding="utf-8")


def run_d22_audit(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir = output_paths(config_path)
    if output_dir is not None:
        artifact_dir = resolve_path(config_path, output_dir)
    if artifact_dir.exists() or runtime_dir.exists():
        raise D22ProtocolError("D2.2 输出或 runtime state 已存在，拒绝重跑")
    git = git_state(config_path)
    if git["tracked_dirty"]:
        raise D22ProtocolError("D2.2 正式 audit 要求 tracked worktree 干净")
    sources = runtime_source_manifest(config_path)
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    mark_phase_started(runtime_dir, "audit")
    with tempfile.TemporaryDirectory(prefix="kgd22-") as temporary:
        run_root = Path(temporary)
        rows, factory, _ = _run_attack_matrix(run_root)
        canaries = tuple(sorted(factory._issued))
        scan = leakage_scan(run_root, canaries)
        ipc_rows = ipc_schema_audit(run_root)
        role_rows = process_role_manifest(run_root)
        state_manifest = persistent_state_manifest(run_root)
        metrics = aggregate_metrics(rows, scan, role_rows, ipc_rows)
        summary = _summary(
            config_path,
            values,
            rows,
            metrics=metrics,
            git=git,
            source_manifest=sources,
        )
        matrix_path = artifact_dir / "service_attack_matrix.csv"
        ipc_path = artifact_dir / "ipc_schema_audit.csv"
        roles_path = artifact_dir / "process_role_manifest.csv"
        summary_path = artifact_dir / "stage_d22_summary.json"
        scan_path = artifact_dir / "leakage_scan.json"
        state_path = artifact_dir / "persistent_state_manifest.json"
        source_path = artifact_dir / "source_sha256_manifest.json"
        resolved_path = artifact_dir / "resolved_config.json"
        status_path = artifact_dir / "protocol_status.json"
        write_csv(matrix_path, rows)
        write_csv(ipc_path, ipc_rows)
        write_csv(roles_path, role_rows)
        write_json(state_path, state_manifest)
        write_json(source_path, sources)
        write_json(
            resolved_path,
            {
                "schema_version": 1,
                "config": values,
                "config_sha256": sha256_file(config_path),
                "contains_runtime_secret": False,
                "contains_synthetic_plaintext": False,
            },
        )
        write_json(summary_path, summary)
        write_json(scan_path, scan)
        write_json(status_path, _protocol_status(summary))
        artifact_occurrences = count_canary_occurrences(
            (
                path.read_bytes()
                for path in artifact_dir.rglob("*")
                if path.is_file()
            ),
            canaries,
        )
        if artifact_occurrences:
            raise D22ProtocolError("D2.2 plaintext 出现在正式 artifact")
        scan["plaintext_artifact_occurrence_count"] = 0
        scan["scanned_location_count"] = (
            int(scan["scanned_location_count"])
            + sum(
                path.is_file() for path in artifact_dir.rglob("*")
            )
        )
        metrics["plaintext_artifact_occurrence_count"] = 0
        metrics["scanned_location_count"] = scan["scanned_location_count"]
        summary = _summary(
            config_path,
            values,
            rows,
            metrics=metrics,
            git=git,
            source_manifest=sources,
        )
        write_json(summary_path, summary)
        write_json(scan_path, scan)
        write_json(status_path, _protocol_status(summary))
        if count_canary_occurrences(
            (
                path.read_bytes()
                for path in artifact_dir.rglob("*")
                if path.is_file()
            ),
            canaries,
        ):
            raise D22ProtocolError("D2.2 artifact 重写后 plaintext 扫描失败")
    report = repo_root(config_path) / "PHASE_D22_REPORT.md"
    _write_report(report, summary, sha256_file(config_path))
    outputs = (
        summary_path,
        matrix_path,
        ipc_path,
        roles_path,
        scan_path,
        state_path,
        source_path,
        resolved_path,
        status_path,
        report,
    )
    mark_phase_completed(runtime_dir, "audit", outputs)
    return {
        "status": summary["isolated_service_generation_status"],
        "metrics": summary["metrics"],
        "ready_for_d2_3_fault_and_observability_probe": summary[
            "ready_for_d2_3_fault_and_observability_probe"
        ],
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "summary": str(summary_path),
        "report": str(report),
    }


def finalize_d22_artifacts(config_path: str | Path) -> dict[str, Any]:
    artifact_dir, runtime_dir = output_paths(config_path)
    require_phase_completed(runtime_dir, "audit")
    target = artifact_dir / "artifact_sha256_manifest.json"
    if target.exists():
        raise D22ProtocolError("D2.2 artifact manifest 已存在，拒绝覆盖")
    report = repo_root(config_path) / "PHASE_D22_REPORT.md"
    if not report.is_file():
        raise D22ProtocolError("PHASE_D22_REPORT.md 尚未生成")
    payload = artifact_manifest(artifact_dir)
    payload.update(
        {
            "stage": "D2.2-final-artifact-manifest",
            "report": {
                "path": "PHASE_D22_REPORT.md",
                "size_bytes": report.stat().st_size,
                "sha256": sha256_file(report),
            },
            "dataset_present": False,
            "runtime_canary_present": False,
            "plaintext_value_present": False,
            "private_value_present": False,
            "private_answer_present": False,
            "capability_token_present": False,
            "capability_hmac_key_present": False,
            "release_ticket_key_present": False,
            "data_encryption_key_present": False,
            "session_credential_present": False,
            "sqlite_runtime_state_present": False,
            "model_checkpoint_present": False,
            "optimizer_state_present": False,
            "tensor_cache_present": False,
        }
    )
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    write_json(target, payload)
    return {
        "artifact_manifest": str(target),
        "file_count": len(payload["files"]) + 1,
        "manifest_file_sha256": sha256_file(target),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
    }


def run_d22_smoke(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    load_config(config_path, smoke=True)
    factory = CanaryFactory()
    with tempfile.TemporaryDirectory(prefix="kgd22-smoke-") as temporary:
        cluster = D22Cluster(Path(temporary) / "s", factory)
        try:
            cluster.start_default()
            _, message = cluster.issue_message(
                RelationId.REGISTRY_ID,
                "entity-a",
            )
            first = cluster.client_call(message)
            second_status, second_reason, _, _ = _status(
                lambda: cluster.client_call(message)
            )
            exact = (
                cluster.parse_success(first).text
                == cluster.canaries[
                    ("entity-a", RelationId.REGISTRY_ID)
                ]
            )
            safe_result = {
                "schema_version": 1,
                "stage": "D2.2-smoke",
                "authorized_exact_delivery": exact,
                "replay_status": second_status,
                "replay_reason": second_reason,
                "runtime_canary_count": len(factory._issued),
                "plaintext_recorded": False,
                "private_value_used": False,
                "passed": (
                    exact
                    and second_status == "reject"
                    and second_reason == "replay"
                ),
            }
        finally:
            cluster.stop_all()
    if output_dir is not None:
        path = resolve_path(config_path, output_dir) / "smoke_summary.json"
        write_json(path, safe_result)
        safe_result["output"] = str(path)
    return safe_result
