from __future__ import annotations

from dataclasses import fields
from concurrent.futures import ThreadPoolExecutor
from inspect import signature

import pytest

from keyed_gram.stage_c24_contract import RelationId
from keyed_gram.stage_d1_contract import (
    AuthorizedMemoryRequest,
    CapabilityError,
    CapabilityGrant,
    RejectionReason,
    SuggestedRelation,
    VerifiedCapability,
)
from keyed_gram.stage_d1_gateway import (
    CapabilityMemoryProbe,
    CapabilityVerifier,
    TrustedCapabilityGateway,
    rejected_request_access_delta,
)
from keyed_gram.stage_d1_policy import (
    CapabilityAuthority,
    CapabilityPolicy,
    PolicyRule,
    RevocationReplayStore,
)
from keyed_gram.stage_d1_token import CapabilityTokenCodec, HmacKeyRing


NOW = 1_800_000_000


def _components(
    *,
    policy_version: str = "policy-v1",
) -> tuple[
    HmacKeyRing,
    CapabilityPolicy,
    CapabilityAuthority,
    RevocationReplayStore,
    TrustedCapabilityGateway,
    CapabilityMemoryProbe,
]:
    keyring = HmacKeyRing({"test-key": b"k" * 32}, "test-key")
    policy = CapabilityPolicy(
        policy_version,
        (
            PolicyRule(
                "subject-a",
                ("entity-a",),
                (RelationId.REGISTRY_ID.value,),
            ),
        ),
        max_ttl_seconds=300,
    )
    codec = CapabilityTokenCodec()
    authority = CapabilityAuthority(keyring, policy, codec)
    store = RevocationReplayStore()
    gateway = TrustedCapabilityGateway(
        CapabilityVerifier(keyring, policy, codec, store),
        lambda: NOW,
    )
    memory = CapabilityMemoryProbe(
        frozenset({("entity-a", RelationId.REGISTRY_ID)})
    )
    return keyring, policy, authority, store, gateway, memory


def _issued_request(
    *,
    issue_now: int = NOW,
    subject_id: str = "subject-a",
    entity_id: str = "entity-a",
    relation_id: RelationId = RelationId.REGISTRY_ID,
):
    keyring, policy, authority, store, gateway, memory = _components()
    grant = CapabilityGrant(
        "subject-a",
        ("entity-a",),
        RelationId.REGISTRY_ID,
    )
    issued = authority.issue(grant, now=issue_now, ttl_seconds=30)
    request = AuthorizedMemoryRequest(
        subject_id,
        entity_id,
        relation_id,
        issued.token,
    )
    return (
        keyring,
        policy,
        authority,
        store,
        gateway,
        memory,
        issued,
        request,
    )


def test_authorized_request_schema_is_typed_and_minimal() -> None:
    assert [field.name for field in fields(AuthorizedMemoryRequest)] == [
        "subject_id",
        "entity_id",
        "relation_id",
        "capability_token",
    ]
    forbidden = {
        "raw_text",
        "confidence",
        "logits",
        "probabilities",
        "semantic_embedding",
        "suggested_relation",
        "private_value",
    }
    assert forbidden.isdisjoint(
        AuthorizedMemoryRequest.__dataclass_fields__
    )


def test_gateway_and_memory_api_do_not_accept_semantic_evidence() -> None:
    gateway_parameters = set(
        signature(TrustedCapabilityGateway.retrieve).parameters
    )
    memory_parameters = set(
        signature(CapabilityMemoryProbe.retrieve_authorized).parameters
    )
    assert gateway_parameters == {"self", "request", "memory"}
    assert memory_parameters == {"self", "capability"}


def test_valid_capability_accesses_exactly_one_typed_bucket() -> None:
    *_, gateway, memory, _, request = _issued_request()
    result = gateway.retrieve(request, memory)
    assert result.entity_id == "entity-a"
    assert result.relation_id is RelationId.REGISTRY_ID
    assert result.memory_access_count == 1
    assert result.cross_relation_access_count == 0
    assert result.cross_entity_access_count == 0
    assert memory.memory_access_count == 1


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (
            {"subject_id": "subject-b"},
            RejectionReason.SUBJECT_MISMATCH,
        ),
        (
            {"entity_id": "entity-b"},
            RejectionReason.ENTITY_SCOPE_MISMATCH,
        ),
        (
            {"relation_id": RelationId.CITY_CODE},
            RejectionReason.RELATION_MISMATCH,
        ),
    ],
)
def test_scope_mismatch_fails_before_memory(replacement, reason) -> None:
    *_, gateway, memory, issued, _ = _issued_request()
    values = {
        "subject_id": "subject-a",
        "entity_id": "entity-a",
        "relation_id": RelationId.REGISTRY_ID,
        "capability_token": issued.token,
    }
    values.update(replacement)
    request = AuthorizedMemoryRequest(**values)
    observed, delta = rejected_request_access_delta(
        gateway, request, memory
    )
    assert observed is reason
    assert delta == 0
    assert memory.memory_access_count == 0


def test_expired_capability_fails_before_memory() -> None:
    *_, gateway, memory, _, request = _issued_request(
        issue_now=NOW - 100
    )
    reason, delta = rejected_request_access_delta(gateway, request, memory)
    assert reason is RejectionReason.EXPIRED
    assert delta == 0


def test_not_yet_valid_capability_fails_before_memory() -> None:
    *_, gateway, memory, _, request = _issued_request(
        issue_now=NOW + 1
    )
    reason, delta = rejected_request_access_delta(gateway, request, memory)
    assert reason is RejectionReason.NOT_YET_VALID
    assert delta == 0


def test_revoked_and_replayed_capability_fail_before_memory() -> None:
    *_, store, gateway, memory, issued, request = _issued_request()
    store.revoke(issued.claims.nonce)
    reason, delta = rejected_request_access_delta(gateway, request, memory)
    assert reason is RejectionReason.REVOKED_TOKEN
    assert delta == 0

    *_, gateway2, memory2, _, request2 = _issued_request()
    gateway2.retrieve(request2, memory2)
    reason2, delta2 = rejected_request_access_delta(
        gateway2, request2, memory2
    )
    assert reason2 is RejectionReason.REPLAY
    assert delta2 == 0
    assert memory2.memory_access_count == 1


def test_single_use_consume_is_atomic_under_concurrent_replay() -> None:
    *_, gateway, memory, _, request = _issued_request()

    def attempt() -> str:
        try:
            gateway.retrieve(request, memory)
        except CapabilityError as error:
            return error.reason.value
        return "accept"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["accept", RejectionReason.REPLAY.value]
    assert memory.memory_access_count == 1


def test_router_proposal_is_never_an_authorization() -> None:
    *_, gateway, memory = _components()
    proposal = SuggestedRelation(RelationId.REGISTRY_ID)
    reason, delta = rejected_request_access_delta(
        gateway, proposal, memory
    )
    assert reason is RejectionReason.ROUTER_PROPOSAL_NOT_AUTHORIZED
    assert delta == 0


def test_memory_rejects_forged_verified_capability() -> None:
    memory = CapabilityMemoryProbe(
        frozenset({("entity-a", RelationId.REGISTRY_ID)})
    )
    with pytest.raises(TypeError, match="不能由外部构造"):
        VerifiedCapability(
            "subject-a",
            "entity-a",
            RelationId.REGISTRY_ID,
            "retrieve",
            "nonce-that-is-long-enough",
            NOW + 30,
            object(),
        )
    assert memory.memory_access_count == 0


def test_wildcard_or_empty_scope_is_rejected() -> None:
    with pytest.raises(CapabilityError) as wildcard:
        CapabilityGrant(
            "subject-a",
            ("*",),
            RelationId.REGISTRY_ID,
        )
    assert wildcard.value.reason is RejectionReason.INVALID_REQUEST
    with pytest.raises(CapabilityError) as empty:
        CapabilityGrant("subject-a", (), RelationId.REGISTRY_ID)
    assert empty.value.reason is RejectionReason.INVALID_REQUEST


def test_authority_denies_unlisted_grant_and_excessive_ttl() -> None:
    *_, authority, _, _, _ = _components()
    with pytest.raises(CapabilityError) as unlisted:
        authority.issue(
            CapabilityGrant(
                "subject-a",
                ("entity-b",),
                RelationId.REGISTRY_ID,
            ),
            now=NOW,
            ttl_seconds=30,
        )
    assert unlisted.value.reason is RejectionReason.UNAUTHORIZED_GRANT
    with pytest.raises(CapabilityError) as ttl:
        authority.issue(
            CapabilityGrant(
                "subject-a",
                ("entity-a",),
                RelationId.REGISTRY_ID,
            ),
            now=NOW,
            ttl_seconds=301,
        )
    assert ttl.value.reason is RejectionReason.UNAUTHORIZED_GRANT
    with pytest.raises(CapabilityError) as nonce:
        authority.issue(
            CapabilityGrant(
                "subject-a",
                ("entity-a",),
                RelationId.REGISTRY_ID,
            ),
            now=NOW,
            ttl_seconds=30,
            nonce="",
        )
    assert nonce.value.reason is RejectionReason.INVALID_REQUEST


def test_policy_version_change_invalidates_old_capability() -> None:
    keyring, _, authority, store, _, memory = _components(
        policy_version="policy-v1"
    )
    issued = authority.issue(
        CapabilityGrant(
            "subject-a",
            ("entity-a",),
            RelationId.REGISTRY_ID,
        ),
        now=NOW,
        ttl_seconds=30,
    )
    current_policy = CapabilityPolicy(
        "policy-v2",
        (
            PolicyRule(
                "subject-a",
                ("entity-a",),
                (RelationId.REGISTRY_ID.value,),
            ),
        ),
        max_ttl_seconds=300,
    )
    gateway = TrustedCapabilityGateway(
        CapabilityVerifier(
            keyring,
            current_policy,
            CapabilityTokenCodec(),
            store,
        ),
        lambda: NOW,
    )
    request = AuthorizedMemoryRequest(
        "subject-a",
        "entity-a",
        RelationId.REGISTRY_ID,
        issued.token,
    )
    reason, delta = rejected_request_access_delta(
        gateway, request, memory
    )
    assert reason is RejectionReason.POLICY_VERSION_MISMATCH
    assert delta == 0
