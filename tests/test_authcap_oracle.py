from __future__ import annotations

from copy import deepcopy

from keyed_gram.authcap_benchmark import (
    context_from_case,
    generate_authzroutebench,
    policy_set_from_case,
)
from keyed_gram.authcap_oracle import (
    build_authorization_witness,
    evaluate_policy,
)


def cases() -> list[dict]:
    return generate_authzroutebench(splits=("train",))["train"]


def test_allow_and_determinism() -> None:
    case = next(row for row in cases() if row["oracle_decision"] == "allow")
    context = context_from_case(case)
    policies = policy_set_from_case(case)
    first = evaluate_policy(context, policies)
    second = evaluate_policy(context, policies)
    assert first.decision == "allow"
    assert first.canonical() == second.canonical()


def test_explicit_deny_dominates_wider_allow() -> None:
    case = next(
        row
        for row in cases()
        if row["metadata"]["explicit_deny"]
        and row["attack_category"] != "multi_step_privilege_amplification"
    )
    grant = evaluate_policy(context_from_case(case), policy_set_from_case(case))
    assert grant.decision == "deny"
    assert grant.decision_reason_code == "explicit_deny"


def test_lower_priority_deny_does_not_override_explicit_allow() -> None:
    case = next(row for row in cases() if row["oracle_decision"] == "allow")
    modified = deepcopy(case)
    deny = deepcopy(modified["policy_set"]["policies"][0])
    deny["policy_id"] = "low-priority-deny"
    deny["effect"] = "deny"
    deny["priority"] = 50
    deny["authority"] = {
        key: [] for key in deny["authority"] if key != "constraints"
    }
    deny["authority"]["constraints"] = []
    modified["policy_set"]["policies"].append(deny)
    grant = evaluate_policy(
        context_from_case(modified), policy_set_from_case(modified)
    )
    assert grant.decision == "allow"


def test_default_deny_cross_tenant() -> None:
    case = next(
        row
        for row in cases()
        if row["attack_category"] == "cross_tenant_reference"
    )
    grant = evaluate_policy(context_from_case(case), policy_set_from_case(case))
    assert grant.decision == "deny"
    assert grant.decision_reason_code == "default_deny"


def test_policy_epoch_is_fail_closed() -> None:
    case = next(
        row for row in cases() if row["attack_category"] == "policy_staleness"
    )
    grant = evaluate_policy(context_from_case(case), policy_set_from_case(case))
    assert grant.decision == "deny"
    assert grant.decision_reason_code == "policy_epoch_mismatch"


def test_expired_rule_becomes_default_deny() -> None:
    case = next(row for row in cases() if row["oracle_decision"] == "allow")
    modified = deepcopy(case)
    modified["trusted_context"]["environment"]["timestamp"] = (
        "2028-01-01T00:00:00Z"
    )
    grant = evaluate_policy(
        context_from_case(modified),
        policy_set_from_case(modified),
    )
    assert grant.decision == "deny"


def test_delegation_is_checked() -> None:
    case = next(
        row
        for row in cases()
        if row["attack_category"] == "delegation_chain_misuse"
        and row["oracle_decision"] == "allow"
    )
    modified = deepcopy(case)
    modified["trusted_context"]["principal"]["delegation_chain"] = []
    grant = evaluate_policy(
        context_from_case(modified), policy_set_from_case(modified)
    )
    assert grant.decision == "deny"


def test_witness_is_bound_to_decision_and_policy() -> None:
    case = next(row for row in cases() if row["oracle_decision"] == "allow")
    context = context_from_case(case)
    grant = evaluate_policy(context, policy_set_from_case(case))
    witness = build_authorization_witness(context, grant)
    assert witness.decision_id == grant.decision_id
    assert witness.policy_hash == grant.policy_hash
