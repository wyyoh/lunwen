"""AuthZRouteBench 的跨 split 高层单元碰撞审计。"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .authcap_policy import canonical_sha256


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def token_signature(value: str) -> str:
    return " ".join(sorted(set(normalize_text(value).split())))


def _principal_signature(case: dict[str, Any]) -> str:
    principal = case["trusted_context"]["principal"]
    return canonical_sha256(
        {
            "tenant_id": principal["tenant_id"],
            "attributes": principal["attributes"],
            "delegation_chain": principal["delegation_chain"],
        }
    )


def _resource_action(case: dict[str, Any]) -> str:
    context = case["trusted_context"]
    return canonical_sha256(
        {
            "resource_id": context["requested_resource"]["resource_id"],
            "action": context["requested_action"],
        }
    )


def _workflow_graph(case: dict[str, Any]) -> str:
    workflow = case.get("workflow")
    if workflow is None:
        return f"none:{case['family_id']}"
    return canonical_sha256(
        {
            "template_id": workflow["template_id"],
            "steps": [
                (step["step"], step["requested_action"])
                for step in workflow["steps"]
            ],
            "combined_action": workflow["combined_action"],
        }
    )


def audit_split_collisions(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    extractors: dict[str, Callable[[dict[str, Any]], str]] = {
        "exact_text": lambda case: case["natural_language_request"],
        "normalized_text": lambda case: normalize_text(
            case["natural_language_request"]
        ),
        "token_signature": lambda case: token_signature(
            case["natural_language_request"]
        ),
        "policy_template": lambda case: case["metadata"][
            "policy_template_id"
        ],
        "resource_action_tuple": _resource_action,
        "principal_attribute_tuple": _principal_signature,
        "attack_family": lambda case: case["family_id"],
        "workflow_graph": _workflow_graph,
        "alias_family": lambda case: case["metadata"][
            "resource_alias_family"
        ],
        "purpose_template": lambda case: case["metadata"][
            "purpose_template_id"
        ],
        "tenant": lambda case: case["trusted_context"]["requested_resource"][
            "tenant_id"
        ],
    }
    results: dict[str, dict[str, Any]] = {}
    total = 0
    for name, extractor in extractors.items():
        owners: dict[str, set[str]] = defaultdict(set)
        for split, cases in splits.items():
            for case in cases:
                owners[extractor(case)].add(split)
        collisions = {
            key: sorted(value)
            for key, value in owners.items()
            if len(value) > 1
        }
        results[name] = {
            "collision_count": len(collisions),
            "collisions": collisions,
        }
        total += len(collisions)
    return {
        "schema_version": 1,
        "cross_split_collision_count": total,
        "units": results,
    }
