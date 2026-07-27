from __future__ import annotations

from dataclasses import replace

from f2_helpers import compile_parts, load_train_case

from keyed_gram.authcap_compiler import (
    CompiledAllow,
    CompiledDeny,
    NeedsExplicitConfirmation,
    trusted_request_envelope,
)
from keyed_gram.authcap_types import ProposalSource, SemanticProposal


def _compile(case, proposal_override=None, envelope_override=None):
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
    return compiler.compile(
        proposal=proposal_override or proposal,
        trusted_context=context,
        trusted_request=envelope_override or envelope,
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at="2026-07-01T12:00:00Z",
    )


def test_exact_allow_recomputes_oracle_and_issues_exact_authority():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    result = _compile(case)
    assert isinstance(result, CompiledAllow)
    assert len(result.executable_authority.atoms) == 1
    assert result.capability_payload.single_use


def test_oracle_and_explicit_deny_fail_closed():
    case = load_train_case(
        lambda item: item["metadata"]["explicit_deny"]
        and item["workflow"] is None
    )
    result = _compile(case)
    assert isinstance(result, CompiledDeny)
    assert result.capability_token is None


def test_mixed_safe_and_unsafe_proposal_needs_confirmation_without_capability():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    raw = case["semantic_proposal_candidates"][0]
    mixed = SemanticProposal(
        "mixed-proposal",
        (
            raw["proposed_resource_ids"][0],
            case["resource_catalog"][1]["resource_id"],
        ),
        (raw["proposed_actions"][0],),
        raw["proposed_purpose"],
        {},
        ProposalSource.LLM,
    )
    result = _compile(case, proposal_override=mixed)
    assert isinstance(result, NeedsExplicitConfirmation)
    assert result.capability_token is None
    assert result.safe_candidate_authority_digest is not None


def test_empty_safe_intersection_denies_instead_of_silent_execution():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    raw = case["semantic_proposal_candidates"][0]
    unsafe = SemanticProposal(
        "unsafe-proposal",
        (case["resource_catalog"][1]["resource_id"],),
        (raw["proposed_actions"][0],),
        raw["proposed_purpose"],
        {},
        ProposalSource.LLM,
    )
    assert isinstance(_compile(case, proposal_override=unsafe), CompiledDeny)


def test_invalid_resource_and_action_are_denied():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    raw = case["semantic_proposal_candidates"][0]
    unknown_resource = SemanticProposal(
        "unknown-resource",
        ("resource-does-not-exist",),
        (raw["proposed_actions"][0],),
        raw["proposed_purpose"],
        {},
        ProposalSource.LLM,
    )
    unknown_action = SemanticProposal(
        "unknown-action",
        tuple(raw["proposed_resource_ids"]),
        ("become-admin",),
        raw["proposed_purpose"],
        {},
        ProposalSource.LLM,
    )
    assert isinstance(_compile(case, proposal_override=unknown_resource), CompiledDeny)
    assert isinstance(_compile(case, proposal_override=unknown_action), CompiledDeny)


def test_stale_policy_and_cross_tenant_are_denied():
    stale = load_train_case(lambda item: item["metadata"]["stale_policy"])
    cross = load_train_case(lambda item: item["metadata"]["cross_tenant"])
    assert isinstance(_compile(stale), CompiledDeny)
    assert isinstance(_compile(cross), CompiledDeny)


def test_proposal_cannot_rewrite_trusted_principal_or_environment():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    (
        context,
        _,
        _,
        envelope,
        _,
        _,
        _,
        _,
        _,
    ) = compile_parts(case)
    forged = replace(envelope, principal_digest="0" * 64)
    assert isinstance(_compile(case, envelope_override=forged), CompiledDeny)
    expected = trusted_request_envelope(
        context,
        request_id=envelope.request_id,
        request_nonce=envelope.request_nonce,
    )
    assert expected == envelope


def test_incomplete_proposal_does_not_create_extra_execution():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    raw = case["semantic_proposal_candidates"][0]
    incomplete = SemanticProposal(
        "incomplete-proposal",
        tuple(raw["proposed_resource_ids"]),
        tuple(raw["proposed_actions"]),
        None,
        {},
        ProposalSource.LLM,
    )
    assert isinstance(_compile(case, proposal_override=incomplete), CompiledDeny)


def test_compiler_does_not_accept_caller_supplied_grant_parameter():
    case = load_train_case(lambda item: item["oracle_decision"] == "allow")
    *_, compiler, _ = compile_parts(case)
    assert "grant" not in compiler.compile.__annotations__
