from __future__ import annotations

from dataclasses import replace

import pytest
from f2_helpers import compile_parts, load_train_case

from keyed_gram.authcap_compiler import CompiledAllow
from keyed_gram.authcap_verifier import AuthCapVerificationError, verify_authcap


def _parts():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    (
        context,
        policy_set,
        proposal,
        envelope,
        resources,
        _,
        parameters,
        compiler,
        ring,
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
    return (
        context,
        policy_set,
        proposal,
        envelope,
        resources,
        parameters,
        ring,
        result,
    )


def _verify(parts, **changes):
    context, policy, proposal, envelope, resources, params, ring, result = parts
    values = {
        "token": result.capability_token,
        "witness": result.authorization_witness,
        "executable_authority": result.executable_authority,
        "proposal": proposal,
        "trusted_context": context,
        "trusted_request": envelope,
        "policy_set": policy,
        "resource_tenants": resources,
        "public_keys": ring,
        "verification_time": "2026-07-01T12:00:01Z",
        "maximum_capability_bytes": params.maximum_capability_bytes,
    }
    values.update(changes)
    return verify_authcap(**values)


def test_verifier_checks_signature_and_policy_derivation():
    verified = _verify(_parts())
    assert verified.single_use
    assert len(verified.executable_authority.atoms) == 1


def test_expiry_is_rejected():
    with pytest.raises(AuthCapVerificationError):
        _verify(_parts(), verification_time="2026-07-01T12:10:00Z")


def test_principal_request_and_proposal_digest_mismatch_rejected():
    parts = _parts()
    context, _, proposal, envelope, *_ = parts
    with pytest.raises(AuthCapVerificationError):
        _verify(
            parts,
            trusted_context=replace(
                context,
                principal=replace(context.principal, subject_id="other-subject"),
            ),
        )
    with pytest.raises(AuthCapVerificationError):
        _verify(
            parts,
            trusted_request=replace(envelope, request_nonce="different-nonce"),
        )
    with pytest.raises(AuthCapVerificationError):
        _verify(parts, proposal=replace(proposal, proposal_id="other-proposal"))


def test_tampered_executable_authority_and_witness_rejected():
    parts = _parts()
    result = parts[-1]
    with pytest.raises(AuthCapVerificationError):
        _verify(
            parts,
            executable_authority=result.executable_authority.empty(),
        )
    with pytest.raises(AuthCapVerificationError):
        _verify(
            parts,
            witness=replace(result.authorization_witness, policy_hash="0" * 64),
        )


def test_stale_policy_epoch_rejected_by_oracle_recomputation():
    parts = _parts()
    policy = replace(parts[1], policy_epoch=parts[1].policy_epoch + 1)
    with pytest.raises(AuthCapVerificationError):
        _verify(parts, policy_set=policy)


def test_non_workflow_capability_cannot_be_relabelled_as_workflow():
    with pytest.raises(AuthCapVerificationError):
        _verify(
            _parts(),
            expected_workflow_id="workflow-1",
            expected_workflow_step=1,
            expected_previous_workflow_state_digest="0" * 64,
        )
