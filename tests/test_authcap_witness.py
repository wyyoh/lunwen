from __future__ import annotations

from dataclasses import replace

import pytest
from f2_helpers import compile_parts, load_train_case

from keyed_gram.authcap_compiler import CompiledAllow
from keyed_gram.authcap_oracle import evaluate_policy
from keyed_gram.authcap_witness import (
    AuthorizationWitnessError,
    verify_witness_semantics,
)


def _compiled():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    (
        context,
        policy_set,
        proposal,
        envelope,
        resources,
        _,
        _,
        compiler,
        _,
    ) = compile_parts(case)
    result = compiler.compile(
        proposal=proposal,
        trusted_context=context,
        trusted_request=envelope,
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at="2026-07-01T12:00:00Z",
    )
    assert isinstance(result, CompiledAllow)
    return context, policy_set, proposal, envelope, result


def test_witness_canonical_digest_is_deterministic_and_binds_decision():
    context, policy_set, proposal, envelope, result = _compiled()
    grant = evaluate_policy(context, policy_set)
    verify_witness_semantics(
        result.authorization_witness,
        context=context,
        grant=grant,
        trusted_request_digest=envelope.canonical_digest,
        proposal=proposal,
        executable_authority=result.executable_authority,
    )
    assert len(result.authorization_witness.canonical_digest) == 64
    assert result.authorization_witness.decision_id == grant.decision_id


@pytest.mark.parametrize(
    "field",
    (
        "policy_hash",
        "principal_attributes_digest",
        "trusted_request_digest",
        "proposal_digest",
        "policy_grant_digest",
        "executable_authority_digest",
        "environment_digest",
    ),
)
def test_witness_digest_mutation_is_rejected(field: str):
    context, policy_set, proposal, envelope, result = _compiled()
    changed = replace(result.authorization_witness, **{field: "0" * 64})
    with pytest.raises(AuthorizationWitnessError):
        verify_witness_semantics(
            changed,
            context=context,
            grant=evaluate_policy(context, policy_set),
            trusted_request_digest=envelope.canonical_digest,
            proposal=proposal,
            executable_authority=result.executable_authority,
        )


def test_witness_epoch_and_matched_policy_ids_are_semantic():
    context, policy_set, proposal, envelope, result = _compiled()
    changed = replace(result.authorization_witness, policy_epoch=999)
    with pytest.raises(AuthorizationWitnessError):
        verify_witness_semantics(
            changed,
            context=context,
            grant=evaluate_policy(context, policy_set),
            trusted_request_digest=envelope.canonical_digest,
            proposal=proposal,
            executable_authority=result.executable_authority,
        )
