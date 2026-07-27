"""F2 测试共享的公开 benchmark fixture。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from keyed_gram.authcap_benchmark import context_from_case, policy_set_from_case
from keyed_gram.authcap_capability import AuthCapIssuer, AuthCapPublicKeyRing
from keyed_gram.authcap_compiler import (
    AuthCapCompiler,
    CompilerParameters,
    trusted_request_envelope,
)
from keyed_gram.authcap_types import ProposalSource, SemanticProposal


def load_train_case(
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    path = Path("data/authzroutebench/generated/train.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if predicate(case):
            return case
    raise AssertionError("未找到测试 case")


def proposal_from_case(case: dict[str, Any]) -> SemanticProposal:
    value = case["semantic_proposal_candidates"][0]
    return SemanticProposal(
        proposal_id=value["proposal_id"],
        proposed_resource_ids=tuple(value["proposed_resource_ids"]),
        proposed_actions=tuple(value["proposed_actions"]),
        proposed_purpose=value["proposed_purpose"],
        proposed_constraints=value["proposed_constraints"],
        source_type=ProposalSource(value["source_type"]),
    )


def compile_parts(case: dict[str, Any]):
    context = context_from_case(case)
    policy_set = policy_set_from_case(case)
    proposal = proposal_from_case(case)
    envelope = trusted_request_envelope(
        context,
        request_id=case["case_id"],
        request_nonce=f"nonce-{case['case_id']}",
    )
    resources = {
        item["resource_id"]: item["tenant_id"]
        for item in case["resource_catalog"]
    }
    issuer = AuthCapIssuer.generate("test-ed25519-key")
    parameters = CompilerParameters(300, 8192)
    compiler = AuthCapCompiler(issuer, parameters)
    ring = AuthCapPublicKeyRing({issuer.key_id: issuer.public_key_bytes})
    return (
        context,
        policy_set,
        proposal,
        envelope,
        resources,
        issuer,
        parameters,
        compiler,
        ring,
    )
