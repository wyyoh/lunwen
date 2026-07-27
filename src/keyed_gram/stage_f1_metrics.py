"""Stage F1 benchmark/oracle 数据质量指标。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .authcap_authority import is_authority_subset
from .authcap_benchmark import (
    context_from_case,
    policy_set_from_case,
    validate_benchmark_case,
)
from .authcap_oracle import evaluate_policy
from .authcap_types import AuthoritySet, Constraint


def authority_from_dict(value: dict[str, Any]) -> AuthoritySet:
    return AuthoritySet(
        subject_ids=frozenset(value["subject_ids"]),
        tenant_ids=frozenset(value["tenant_ids"]),
        resource_ids=frozenset(value["resource_ids"]),
        actions=frozenset(value["actions"]),
        relation_ids=frozenset(value["relation_ids"]),
        purposes=frozenset(value["purposes"]),
        constraints=tuple(
            Constraint(
                key=item["key"],
                operator=item["operator"],
                value=item["value"],
            )
            for item in value["constraints"]
        ),
    )


def benchmark_quality_metrics(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_cases = [case for cases in splits.values() for case in cases]
    families = {case["family_id"] for case in all_cases}
    policies = {
        rule["policy_id"]
        for case in all_cases
        for rule in case["policy_set"]["policies"]
    }
    principals = {
        case["trusted_context"]["principal"]["subject_id"] for case in all_cases
    }
    tenants = {
        case["trusted_context"]["requested_resource"]["tenant_id"]
        for case in all_cases
    }
    resources = {
        item["resource_id"]
        for case in all_cases
        for item in case["resource_catalog"]
    }
    workflows = {
        case["workflow"]["workflow_id"]
        for case in all_cases
        if case["workflow"] is not None
    }
    failures = Counter()
    decision_counts = Counter()
    restricted = 0
    for case in all_cases:
        try:
            validate_benchmark_case(case)
            context = context_from_case(case)
            policy_set = policy_set_from_case(case)
            first = evaluate_policy(context, policy_set)
            second = evaluate_policy(context, policy_set)
        except (TypeError, ValueError, KeyError):
            failures["oracle_schema_failure_count"] += 1
            continue
        if first.canonical() != second.canonical():
            failures["oracle_determinism_failure_count"] += 1
        if first.policy_hash != case["oracle_grant"]["policy_hash"]:
            failures["policy_hash_mismatch_count"] += 1
        expected = authority_from_dict(case["oracle_maximum_authority"])
        if not is_authority_subset(expected, first.maximum_authority):
            failures["authority_subset_failure_count"] += 1
        if first.canonical() != case["oracle_grant"]:
            failures["oracle_determinism_failure_count"] += 1
        workflow = case.get("workflow")
        if workflow is not None:
            for step in workflow["steps"]:
                step_case = {
                    **case,
                    "trusted_context": {
                        **case["trusted_context"],
                        "requested_action": step["requested_action"],
                    },
                }
                step_grant = evaluate_policy(
                    context_from_case(step_case),
                    policy_set,
                )
                if step_grant.decision != step["oracle_decision"]:
                    failures["oracle_determinism_failure_count"] += 1
            if first.decision != workflow["combined_oracle_decision"]:
                failures["oracle_determinism_failure_count"] += 1
        decision_counts[first.decision] += 1
        if first.decision == "allow" and first.maximum_authority.constraints:
            restricted += 1
    distributions = {
        "domain_by_split": _cross(all_cases, "domain", "split"),
        "attack_category_by_split": _cross(
            all_cases, "attack_category", "split"
        ),
        "decision_by_split": _decision_cross(all_cases),
        "policy_complexity": _histogram(
            len(case["policy_set"]["policies"]) for case in all_cases
        ),
        "authority_set_size": _histogram(
            sum(
                len(value)
                for key, value in case["oracle_maximum_authority"].items()
                if key != "constraints"
            )
            for case in all_cases
        ),
        "workflow_length": _histogram(
            0 if case["workflow"] is None else len(case["workflow"]["steps"])
            for case in all_cases
        ),
        "purpose_presence": dict(
            sorted(
                Counter(
                    "present"
                    if case["trusted_context"]["purpose"] is not None
                    else "absent"
                    for case in all_cases
                ).items()
            )
        ),
        "delegation_depth": _histogram(
            len(case["trusted_context"]["principal"]["delegation_chain"])
            for case in all_cases
        ),
    }
    return {
        "schema_version": 1,
        "case_count": len(all_cases),
        "family_count": len(families),
        "domain_count": len({case["domain"] for case in all_cases}),
        "attack_category_count": len(
            {case["attack_category"] for case in all_cases}
        ),
        "policy_count": len(policies),
        "principal_count": len(principals),
        "tenant_count": len(tenants),
        "resource_count": len(resources),
        "workflow_count": len(workflows),
        "multi_step_workflow_count": len(workflows),
        "allow_case_count": decision_counts["allow"],
        "deny_case_count": decision_counts["deny"],
        "restricted_authority_case_count": restricted,
        "oracle_determinism_failure_count": failures[
            "oracle_determinism_failure_count"
        ],
        "oracle_schema_failure_count": failures[
            "oracle_schema_failure_count"
        ],
        "authority_subset_failure_count": failures[
            "authority_subset_failure_count"
        ],
        "policy_hash_mismatch_count": failures[
            "policy_hash_mismatch_count"
        ],
        "explicit_deny_case_count": sum(
            bool(case["metadata"]["explicit_deny"]) for case in all_cases
        ),
        "multi_step_amplification_case_count": sum(
            case["attack_category"] == "multi_step_privilege_amplification"
            for case in all_cases
        ),
        "stale_policy_case_count": sum(
            bool(case["metadata"]["stale_policy"]) for case in all_cases
        ),
        "cross_tenant_case_count": sum(
            bool(case["metadata"]["cross_tenant"]) for case in all_cases
        ),
        "split_case_counts": {
            split: len(cases) for split, cases in splits.items()
        },
        "distributions": distributions,
    }


def _cross(
    cases: list[dict[str, Any]],
    first: str,
    second: str,
) -> dict[str, int]:
    return {
        f"{left}|{right}": count
        for (left, right), count in sorted(
            Counter((case[first], case[second]) for case in cases).items()
        )
    }


def _decision_cross(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"{case_split}|{decision}": count
        for (case_split, decision), count in sorted(
            Counter(
                (case["split"], case["oracle_decision"]) for case in cases
            ).items()
        )
    }


def _histogram(values: Any) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(values).items())
    }
