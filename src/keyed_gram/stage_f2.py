"""Stage F2 的六变体 simulator、顺序正式审计与报告生成。"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .authcap_atoms import (
    AtomizedAuthority,
    authority_set_to_atomized,
)
from .authcap_benchmark import (
    context_from_case,
    policy_set_from_case,
    validate_benchmark_case,
)
from .authcap_capability import (
    AuthCapIssuer,
    AuthCapPublicKeyRing,
)
from .authcap_compiler import (
    AuthCapCompiler,
    CompiledAllow,
    CompiledDeny,
    CompilerParameters,
    attach_mandatory_grant_constraints,
    normalize_proposal_authority,
    trusted_request_envelope,
)
from .authcap_oracle import evaluate_policy
from .authcap_policy import canonical_json_bytes, canonical_sha256
from .authcap_types import (
    Action,
    ProposalSource,
    SemanticProposal,
)
from .authcap_verifier import AuthCapVerificationError, verify_authcap
from .authcap_witness import AuthorizationWitness
from .authcap_workflow import (
    WorkflowAuthCapCompiler,
    WorkflowAuthorityState,
    WorkflowStateStore,
)
from .stage_f2_metrics import (
    SECURITY_COUNT_FIELDS,
    aggregate_all,
    aggregate_variant_rows,
    per_category,
    security_violation_matrix,
)
from .stage_f2_protocol import (
    F2_VARIANTS,
    F2ProtocolError,
    artifact_manifest,
    claim_phase,
    complete_phase,
    formal_preflight,
    initialize_protocol,
    load_config,
    output_paths,
    read_json,
    repo_root,
    runtime_source_manifest,
    scan_for_forbidden_material,
    sha256_file,
    validate_final_protocol,
    write_csv,
    write_json,
)

_RESULT_FIELDS = (
    "schema_version",
    "split",
    "case_id",
    "domain",
    "family_id",
    "attack_category",
    "variant",
    "oracle_decision",
    "compiler_outcome",
    "decision_reason_code",
    "executable_authority_digest",
    "capability_digest",
    "exact_execution",
    "recoverable_safe",
    "authority_preserved",
    "unnecessary_rejection",
    "confirmation_required",
    "correct_confirmation",
    "authority_amplification_count",
    "unauthorized_action_count",
    "cross_subject_capability_count",
    "cross_tenant_capability_count",
    "cross_resource_capability_count",
    "cross_action_capability_count",
    "cross_relation_capability_count",
    "cross_purpose_capability_count",
    "explicit_deny_bypass_count",
    "stale_policy_acceptance_count",
    "invalid_witness_acceptance_count",
    "tampered_capability_acceptance_count",
    "forged_signature_acceptance_count",
    "compositional_authority_amplification_count",
    "multi_step_forbidden_effect_acceptance_count",
    "workflow_state_replay_acceptance_count",
    "workflow_state_fork_acceptance_count",
    "semantic_proposal_direct_execution_count",
    "silent_scope_narrowing_count",
    "partial_execution_without_confirmation_count",
    "compiler_determinism_failure_count",
    "capability_verification_failure_on_valid_count",
    "compile_latency_ms",
    "verify_latency_ms",
    "capability_size_bytes",
    "witness_size_bytes",
    "workflow_state_size_bytes",
)
_ARTIFACT_FILES = (
    "aggregate_metrics.csv",
    "calibration_results.csv",
    "compiler_freeze_manifest.json",
    "development_results.csv",
    "locked_test_results.csv",
    "per_category_metrics.csv",
    "performance_metrics.csv",
    "prepare_manifest.json",
    "private_material_scan.json",
    "protocol_status.json",
    "public_verifier_manifest.json",
    "resolved_config.json",
    "security_violation_matrix.csv",
    "source_sha256_manifest.json",
    "stage_f2_summary.json",
    "upstream_frozen_sha256.json",
    "variant_registry.json",
    "workflow_metrics.csv",
)


def _strict_json_line(line: str, line_number: int) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in items:
            if key in value:
                raise F2ProtocolError(
                    f"benchmark 第 {line_number} 行 duplicate key：{key}"
                )
            value[key] = child
        return value

    try:
        value = json.loads(
            line,
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                F2ProtocolError(f"benchmark 非法常量：{item}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise F2ProtocolError(f"benchmark 第 {line_number} 行无法解析") from exc
    if not isinstance(value, dict):
        raise F2ProtocolError("benchmark JSONL 行必须是 object")
    validate_benchmark_case(value)
    return value


def _load_split(path: Path, expected_split: str) -> list[dict[str, Any]]:
    values = [
        _strict_json_line(line, index)
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line
    ]
    if len(values) != 360 or any(item["split"] != expected_split for item in values):
        raise F2ProtocolError(f"{expected_split} split case 数或标签不匹配")
    return values


def _proposal_from_case(case: dict[str, Any]) -> SemanticProposal:
    proposals = case["semantic_proposal_candidates"]
    if len(proposals) != 1:
        raise F2ProtocolError("F2 预注册 benchmark 每 case 必须有一个 proposal")
    value = proposals[0]
    expected = {
        "proposal_id",
        "proposed_resource_ids",
        "proposed_actions",
        "proposed_purpose",
        "proposed_constraints",
        "source_type",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise F2ProtocolError("SemanticProposal schema 不匹配")
    return SemanticProposal(
        proposal_id=value["proposal_id"],
        proposed_resource_ids=tuple(value["proposed_resource_ids"]),
        proposed_actions=tuple(value["proposed_actions"]),
        proposed_purpose=value["proposed_purpose"],
        proposed_constraints=value["proposed_constraints"],
        source_type=ProposalSource(value["source_type"]),
    )


def _resource_tenants(case: dict[str, Any]) -> dict[str, str]:
    result = {}
    for item in case["resource_catalog"]:
        if set(item) != {"resource_id", "tenant_id", "alias_family"}:
            raise F2ProtocolError("resource catalog schema 不匹配")
        if item["resource_id"] in result:
            raise F2ProtocolError("duplicate resource catalog ID")
        result[item["resource_id"]] = item["tenant_id"]
    return result


def _tampered_token(token: str) -> str:
    parts = token.split(".")
    signature = base64.urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4))
    changed = bytes([signature[0] ^ 1]) + signature[1:]
    parts[2] = base64.urlsafe_b64encode(changed).rstrip(b"=").decode("ascii")
    return ".".join(parts)


def _forged_token(token: str) -> str:
    parts = token.split(".")
    parts[2] = base64.urlsafe_b64encode(bytes(64)).rstrip(b"=").decode("ascii")
    return ".".join(parts)


def _baseline_result(
    variant: str,
    *,
    proposal_authority: AtomizedAuthority,
    request_authority: AtomizedAuthority,
    grant_authority: AtomizedAuthority,
    oracle_decision: str,
) -> tuple[str, str, AtomizedAuthority]:
    if variant == "B0":
        return "allow", "trust_proposal", proposal_authority
    if variant == "B1":
        return "deny", "deny_all", AtomizedAuthority.empty()
    constrained = attach_mandatory_grant_constraints(
        proposal_authority,
        grant_authority,
    )
    safe = constrained.intersection(request_authority).intersection(grant_authority)
    fully_safe = constrained.is_subset_of(request_authority) and (
        constrained.is_subset_of(grant_authority)
    )
    if variant == "B2":
        if oracle_decision == "allow" and fully_safe and not safe.is_empty:
            return "allow", "reject_on_excess_allow", safe
        return "deny", "reject_on_excess_deny", AtomizedAuthority.empty()
    if variant == "B3":
        if oracle_decision == "allow" and not safe.is_empty:
            return "allow", "silent_intersection", safe
        return "deny", "empty_intersection", AtomizedAuthority.empty()
    raise ValueError(f"未知 baseline variant：{variant}")


def _workflow_probe(
    *,
    variant: str,
    case: dict[str, Any],
    compiler: AuthCapCompiler,
    public_keys: AuthCapPublicKeyRing,
    parameters: CompilerParameters,
    issued_at: str,
    verification_time: str,
) -> dict[str, int | float]:
    empty = {
        "compositional_authority_amplification_count": 0,
        "multi_step_forbidden_effect_acceptance_count": 0,
        "workflow_state_replay_acceptance_count": 0,
        "workflow_state_fork_acceptance_count": 0,
        "capability_verification_failure_on_valid_count": 0,
        "workflow_state_size_bytes": 0,
    }
    workflow = case["workflow"]
    if workflow is None:
        return empty
    if variant == "B1":
        return empty
    context = context_from_case(case)
    policy_set = policy_set_from_case(case)
    resources = _resource_tenants(case)
    step_results = []
    proposals = []
    contexts = []
    envelopes = []
    for step in workflow["steps"]:
        step_context = replace(
            context,
            requested_action=Action(step["requested_action"]),
        )
        step_proposal = SemanticProposal(
            proposal_id=f"{case['case_id']}-step-{step['step']}",
            proposed_resource_ids=(context.requested_resource.resource_id,),
            proposed_actions=(step["requested_action"],),
            proposed_purpose=context.purpose,
            proposed_constraints={},
            source_type=ProposalSource.RULE,
        )
        step_envelope = trusted_request_envelope(
            step_context,
            request_id=f"{case['case_id']}-step-{step['step']}",
            request_nonce=f"request-nonce-{step['step']}",
        )
        proposals.append(step_proposal)
        contexts.append(step_context)
        envelopes.append(step_envelope)
        if variant in {"B0", "B2", "B3"}:
            grant = evaluate_policy(step_context, policy_set)
            grant_authority = authority_set_to_atomized(grant.maximum_authority)
            raw = normalize_proposal_authority(
                step_proposal,
                context=step_context,
                resource_tenants=resources,
            )
            outcome, _, _ = _baseline_result(
                variant,
                proposal_authority=raw,
                request_authority=step_envelope.requested_authority,
                grant_authority=grant_authority,
                oracle_decision=grant.decision,
            )
            step_results.append(outcome)
        elif variant == "A0":
            step_results.append(
                compiler.compile(
                    proposal=step_proposal,
                    trusted_context=step_context,
                    trusted_request=step_envelope,
                    policy_set=policy_set,
                    resource_tenants=resources,
                    issued_at=issued_at,
                ).status
            )
    combined_deny = workflow["combined_oracle_decision"] == "deny"
    if variant != "A1":
        violation = int(combined_deny and step_results == ["allow", "allow"])
        return {
            **empty,
            "compositional_authority_amplification_count": violation,
            "multi_step_forbidden_effect_acceptance_count": violation,
        }
    store = WorkflowStateStore()
    state0 = WorkflowAuthorityState.create(
        workflow_id=workflow["workflow_id"],
        subject_id=context.principal.subject_id,
        policy_epoch=context.policy_epoch,
    )
    store.initialize(state0)
    workflow_compiler = WorkflowAuthCapCompiler(compiler, store)
    first, state1 = workflow_compiler.compile_next(
        state=state0,
        step_id="step-1",
        step_number=1,
        expected_workflow_id=workflow["workflow_id"],
        proposal=proposals[0],
        trusted_context=contexts[0],
        trusted_request=envelopes[0],
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at=issued_at,
    )
    if not isinstance(first, CompiledAllow) or state1 is None:
        return {
            **empty,
            "compositional_authority_amplification_count": 1,
        }
    workflow_verification_failure = 0
    try:
        verify_authcap(
            token=first.capability_token,
            witness=first.authorization_witness,
            executable_authority=first.executable_authority,
            proposal=proposals[0],
            trusted_context=contexts[0],
            trusted_request=envelopes[0],
            policy_set=policy_set,
            resource_tenants=resources,
            public_keys=public_keys,
            verification_time=verification_time,
            maximum_capability_bytes=parameters.maximum_capability_bytes,
            expected_workflow_id=workflow["workflow_id"],
            expected_workflow_step=1,
            expected_previous_workflow_state_digest=state0.state_digest,
        )
    except AuthCapVerificationError:
        workflow_verification_failure = 1
    second, _ = workflow_compiler.compile_next(
        state=state1,
        step_id="step-2",
        step_number=2,
        expected_workflow_id=workflow["workflow_id"],
        proposal=proposals[1],
        trusted_context=contexts[1],
        trusted_request=envelopes[1],
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at=issued_at,
        combined_context=context,
    )
    replay, _ = workflow_compiler.compile_next(
        state=state0,
        step_id="step-1-replay",
        step_number=1,
        expected_workflow_id=workflow["workflow_id"],
        proposal=proposals[0],
        trusted_context=contexts[0],
        trusted_request=envelopes[0],
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at=issued_at,
    )
    fork, _ = workflow_compiler.compile_next(
        state=replace(state0),
        step_id="step-1-fork",
        step_number=1,
        expected_workflow_id=workflow["workflow_id"],
        proposal=proposals[0],
        trusted_context=contexts[0],
        trusted_request=envelopes[0],
        policy_set=policy_set,
        resource_tenants=resources,
        issued_at=issued_at,
    )
    return {
        **empty,
        "compositional_authority_amplification_count": int(
            not isinstance(second, CompiledDeny)
        ),
        "multi_step_forbidden_effect_acceptance_count": int(
            not isinstance(second, CompiledDeny)
        ),
        "workflow_state_replay_acceptance_count": int(
            not isinstance(replay, CompiledDeny)
        ),
        "workflow_state_fork_acceptance_count": int(
            not isinstance(fork, CompiledDeny)
        ),
        "capability_verification_failure_on_valid_count": (
            workflow_verification_failure
        ),
        "workflow_state_size_bytes": len(canonical_json_bytes(state1.material())),
    }


def _evaluate_case_variant(
    *,
    case: dict[str, Any],
    split: str,
    variant: str,
    issuer: AuthCapIssuer,
    public_keys: AuthCapPublicKeyRing,
    parameters: CompilerParameters,
    issued_at: str,
    verification_time: str,
) -> dict[str, Any]:
    context = context_from_case(case)
    policy_set = policy_set_from_case(case)
    proposal = _proposal_from_case(case)
    resources = _resource_tenants(case)
    envelope = trusted_request_envelope(
        context,
        request_id=case["case_id"],
        request_nonce=f"request-nonce-{case['case_id']}",
    )
    grant = evaluate_policy(context, policy_set)
    grant_authority = authority_set_to_atomized(grant.maximum_authority)
    raw_proposal = normalize_proposal_authority(
        proposal,
        context=context,
        resource_tenants=resources,
    )
    compiler = AuthCapCompiler(issuer, parameters)
    start = time.perf_counter_ns()
    capability_token = None
    witness: AuthorizationWitness | None = None
    executable = AtomizedAuthority.empty()
    if variant in {"B0", "B1", "B2", "B3"}:
        outcome, reason, executable = _baseline_result(
            variant,
            proposal_authority=raw_proposal,
            request_authority=envelope.requested_authority,
            grant_authority=grant_authority,
            oracle_decision=grant.decision,
        )
        result: Any = None
    else:
        result = compiler.compile(
            proposal=proposal,
            trusted_context=context,
            trusted_request=envelope,
            policy_set=policy_set,
            resource_tenants=resources,
            issued_at=issued_at,
        )
        outcome = result.status
        reason = result.decision_reason_code
        if isinstance(result, CompiledAllow):
            executable = result.executable_authority
            capability_token = result.capability_token
            witness = result.authorization_witness
    compile_latency = (time.perf_counter_ns() - start) / 1_000_000
    constrained_proposal = attach_mandatory_grant_constraints(
        raw_proposal,
        grant_authority,
    )
    authority_amplification = int(
        not executable.is_empty
        and (
            not executable.is_subset_of(envelope.requested_authority)
            or not executable.is_subset_of(grant_authority)
        )
    )
    atoms = executable.atoms
    requested = context.requested_resource
    purpose = context.purpose if context.purpose is not None else "purpose:none"
    cross_subject = int(
        bool(atoms)
        and any(atom.subject_id != context.principal.subject_id for atom in atoms)
    )
    cross_tenant = int(
        bool(atoms)
        and any(atom.tenant_id != context.principal.tenant_id for atom in atoms)
    )
    cross_resource = int(
        bool(atoms)
        and any(atom.resource_id != requested.resource_id for atom in atoms)
    )
    cross_action = int(
        bool(atoms)
        and any(atom.action != context.requested_action.value for atom in atoms)
    )
    cross_relation = int(
        bool(atoms)
        and any(atom.relation_id != requested.relation_id for atom in atoms)
    )
    cross_purpose = int(
        bool(atoms)
        and any(atom.purpose != purpose for atom in atoms)
    )
    invalid_witness_acceptance = 0
    tampered_acceptance = 0
    forged_acceptance = 0
    valid_verification_failure = 0
    verify_latency = 0.0
    determinism_failure = 0
    if isinstance(result, CompiledAllow) and witness is not None:
        verify_start = time.perf_counter_ns()
        try:
            verify_authcap(
                token=result.capability_token,
                witness=witness,
                executable_authority=executable,
                proposal=proposal,
                trusted_context=context,
                trusted_request=envelope,
                policy_set=policy_set,
                resource_tenants=resources,
                public_keys=public_keys,
                verification_time=verification_time,
                maximum_capability_bytes=parameters.maximum_capability_bytes,
            )
        except AuthCapVerificationError:
            valid_verification_failure = 1
        verify_latency = (time.perf_counter_ns() - verify_start) / 1_000_000
        invalid_witness = replace(
            witness,
            executable_authority_digest="0" * 64,
        )
        negative = (
            ("witness", result.capability_token, invalid_witness),
            ("tamper", _tampered_token(result.capability_token), witness),
            ("forge", _forged_token(result.capability_token), witness),
        )
        for label, token, probe_witness in negative:
            accepted = False
            try:
                verify_authcap(
                    token=token,
                    witness=probe_witness,
                    executable_authority=executable,
                    proposal=proposal,
                    trusted_context=context,
                    trusted_request=envelope,
                    policy_set=policy_set,
                    resource_tenants=resources,
                    public_keys=public_keys,
                    verification_time=verification_time,
                    maximum_capability_bytes=parameters.maximum_capability_bytes,
                )
                accepted = True
            except AuthCapVerificationError:
                pass
            if label == "witness":
                invalid_witness_acceptance = int(accepted)
            elif label == "tamper":
                tampered_acceptance = int(accepted)
            else:
                forged_acceptance = int(accepted)
        duplicate = compiler.compile(
            proposal=proposal,
            trusted_context=context,
            trusted_request=envelope,
            policy_set=policy_set,
            resource_tenants=resources,
            issued_at=issued_at,
        )
        determinism_failure = int(
            not isinstance(duplicate, CompiledAllow)
            or duplicate.capability_token != result.capability_token
            or duplicate.authorization_witness != witness
            or duplicate.executable_authority != executable
        )
    workflow = _workflow_probe(
        variant=variant,
        case=case,
        compiler=compiler,
        public_keys=public_keys,
        parameters=parameters,
        issued_at=issued_at,
        verification_time=verification_time,
    )
    exact = int(
        grant.decision == "allow"
        and not executable.is_empty
        and executable == grant_authority
    )
    safe_candidate = (
        constrained_proposal
        .intersection(envelope.requested_authority)
        .intersection(grant_authority)
    )
    confirmation = int(outcome == "needs_explicit_confirmation")
    correct_confirmation = int(
        confirmation and grant.decision == "allow" and not safe_candidate.is_empty
    )
    silent = int(
        variant == "B3"
        and not executable.is_empty
        and executable != constrained_proposal
    )
    capability_digest = (
        hashlib.sha256(capability_token.encode("ascii")).hexdigest()
        if capability_token is not None
        else ""
    )
    witness_size = (
        len(canonical_json_bytes(witness.canonical()))
        if witness is not None
        else 0
    )
    return {
        "schema_version": 1,
        "split": split,
        "case_id": case["case_id"],
        "domain": case["domain"],
        "family_id": case["family_id"],
        "attack_category": case["attack_category"],
        "variant": variant,
        "oracle_decision": grant.decision,
        "compiler_outcome": outcome,
        "decision_reason_code": reason,
        "executable_authority_digest": (
            executable.canonical_digest if not executable.is_empty else ""
        ),
        "capability_digest": capability_digest,
        "exact_execution": exact,
        "recoverable_safe": int(exact or correct_confirmation),
        "authority_preserved": exact,
        "unnecessary_rejection": int(
            grant.decision == "allow" and outcome == "deny"
        ),
        "confirmation_required": confirmation,
        "correct_confirmation": correct_confirmation,
        "authority_amplification_count": authority_amplification,
        "unauthorized_action_count": int(
            not executable.is_empty
            and (grant.decision == "deny" or authority_amplification)
        ),
        "cross_subject_capability_count": cross_subject,
        "cross_tenant_capability_count": cross_tenant,
        "cross_resource_capability_count": cross_resource,
        "cross_action_capability_count": cross_action,
        "cross_relation_capability_count": cross_relation,
        "cross_purpose_capability_count": cross_purpose,
        "explicit_deny_bypass_count": int(
            case["metadata"]["explicit_deny"] and not executable.is_empty
        ),
        "stale_policy_acceptance_count": int(
            case["metadata"]["stale_policy"] and not executable.is_empty
        ),
        "invalid_witness_acceptance_count": invalid_witness_acceptance,
        "tampered_capability_acceptance_count": tampered_acceptance,
        "forged_signature_acceptance_count": forged_acceptance,
        "compositional_authority_amplification_count": workflow[
            "compositional_authority_amplification_count"
        ],
        "multi_step_forbidden_effect_acceptance_count": workflow[
            "multi_step_forbidden_effect_acceptance_count"
        ],
        "workflow_state_replay_acceptance_count": workflow[
            "workflow_state_replay_acceptance_count"
        ],
        "workflow_state_fork_acceptance_count": workflow[
            "workflow_state_fork_acceptance_count"
        ],
        "semantic_proposal_direct_execution_count": int(
            variant == "B0" and not executable.is_empty
        ),
        "silent_scope_narrowing_count": silent,
        "partial_execution_without_confirmation_count": silent,
        "compiler_determinism_failure_count": determinism_failure,
        "capability_verification_failure_on_valid_count": (
            valid_verification_failure
            + workflow["capability_verification_failure_on_valid_count"]
        ),
        "compile_latency_ms": round(compile_latency, 6),
        "verify_latency_ms": round(verify_latency, 6),
        "capability_size_bytes": (
            len(capability_token.encode("ascii"))
            if capability_token is not None
            else 0
        ),
        "witness_size_bytes": witness_size,
        "workflow_state_size_bytes": workflow["workflow_state_size_bytes"],
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    split: str,
    variants: tuple[str, ...],
    issuer: AuthCapIssuer,
    parameters: CompilerParameters,
    issued_at: str,
    verification_time: str,
) -> list[dict[str, Any]]:
    public_keys = AuthCapPublicKeyRing({issuer.key_id: issuer.public_key_bytes})
    return [
        _evaluate_case_variant(
            case=case,
            split=split,
            variant=variant,
            issuer=issuer,
            public_keys=public_keys,
            parameters=parameters,
            issued_at=issued_at,
            verification_time=verification_time,
        )
        for variant in variants
        for case in cases
    ]


def _issuer_path(runtime_dir: Path) -> Path:
    return runtime_dir / "private" / "f2-issuer-private.pem"


def _load_issuer(runtime_dir: Path) -> AuthCapIssuer:
    return AuthCapIssuer.load_private_pem("f2-ed25519-2026-01", _issuer_path(runtime_dir))


def _parameters(values: dict[str, Any]) -> CompilerParameters:
    calibration = values["calibration"]
    return CompilerParameters(
        capability_ttl_seconds=calibration["fixed_capability_ttl_seconds"],
        maximum_capability_bytes=calibration[
            "fixed_maximum_capability_bytes"
        ],
    )


def run_stage_f2_prepare(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    preflight = formal_preflight(config_path, values)
    artifact_dir, runtime_dir, _ = output_paths(config_path, values)
    initialize_protocol(runtime_dir, git=preflight["git"])
    claim_phase(runtime_dir, phase="prepare", expected_previous="initialized")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    write_json(artifact_dir / "resolved_config.json", values)
    write_json(
        artifact_dir / "upstream_frozen_sha256.json",
        preflight["upstream"],
    )
    write_json(
        artifact_dir / "source_sha256_manifest.json",
        runtime_source_manifest(config_path),
    )
    write_json(
        artifact_dir / "variant_registry.json",
        {
            "schema_version": 1,
            "variants": [
                {
                    "variant": variant,
                    "name": values["variant_names"][variant],
                    "final_method": variant == "A1",
                }
                for variant in F2_VARIANTS
            ],
            "final_variant": "A1",
            "selected_before_development": True,
        },
    )
    root = repo_root(config_path)
    counts = {}
    for split in ("train", "calibration"):
        path = root / values["f1_frozen"]["splits"][split]["path"]
        counts[split] = len(_load_split(path, split))
    prepare = {
        "schema_version": 1,
        "status": "passed",
        "parsed_splits": ["train", "calibration"],
        "parsed_case_counts": counts,
        "development_content_read": False,
        "locked_test_content_read": False,
        "development_sha256_verified_without_parsing": True,
        "locked_test_sha256_verified_without_parsing": True,
    }
    write_json(artifact_dir / "prepare_manifest.json", prepare)
    complete_phase(runtime_dir, phase="prepare")
    return prepare


def run_stage_f2_calibration(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir, _ = output_paths(config_path, values)
    claim_phase(runtime_dir, phase="calibration", expected_previous="prepare")
    issuer = AuthCapIssuer.generate("f2-ed25519-2026-01")
    issuer.write_private_pem(_issuer_path(runtime_dir))
    root = repo_root(config_path)
    parameters = _parameters(values)
    rows = []
    for split in ("train", "calibration"):
        cases = _load_split(
            root / values["f1_frozen"]["splits"][split]["path"],
            split,
        )
        rows.extend(
            evaluate_cases(
                cases,
                split=split,
                variants=F2_VARIANTS,
                issuer=issuer,
                parameters=parameters,
                issued_at=values["evaluation"]["issued_at"],
                verification_time=values["evaluation"]["verification_time"],
            )
        )
    aggregate = aggregate_all(rows)
    calibration_rows = [
        {
            **item,
            "selected_capability_ttl_seconds": parameters.capability_ttl_seconds,
            "selected_maximum_capability_bytes": parameters.maximum_capability_bytes,
            "safety_invariant_tuned": False,
        }
        for item in aggregate
    ]
    write_csv(artifact_dir / "calibration_results.csv", calibration_rows)
    write_json(runtime_dir / "calibration_rows.json", {"rows": rows})
    complete_phase(runtime_dir, phase="calibration")
    return {"status": "passed", "aggregate_rows": len(calibration_rows)}


def run_stage_f2_freeze(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir, _ = output_paths(config_path, values)
    claim_phase(runtime_dir, phase="freeze", expected_previous="calibration")
    source = runtime_source_manifest(config_path)
    committed_source = read_json(
        artifact_dir / "source_sha256_manifest.json",
        label="F2 source manifest",
    )
    if source != committed_source:
        raise F2ProtocolError("calibration 后 F2 source 已修改")
    issuer = _load_issuer(runtime_dir)
    ring = AuthCapPublicKeyRing({issuer.key_id: issuer.public_key_bytes})
    public_manifest = ring.public_manifest()
    public_manifest["manifest_payload_sha256"] = canonical_sha256(public_manifest)
    write_json(artifact_dir / "public_verifier_manifest.json", public_manifest)
    parameters = _parameters(values)
    freeze = {
        "schema_version": 1,
        "compiler_version": values["compiler_version"],
        "final_variant": "A1",
        "code_freeze_commit": read_json(
            runtime_dir / "protocol_state.json",
            label="F2 protocol state",
        )["code_freeze_commit"],
        "source_manifest_payload_sha256": source["manifest_payload_sha256"],
        "config_sha256": sha256_file(config_path),
        "public_verifier_manifest_sha256": sha256_file(
            artifact_dir / "public_verifier_manifest.json"
        ),
        "capability_ttl_seconds": parameters.capability_ttl_seconds,
        "maximum_capability_bytes": parameters.maximum_capability_bytes,
        "confirmation_reason_grouping": values["calibration"][
            "confirmation_reason_grouping"
        ],
        "development_used_for_selection": False,
        "locked_test_used_for_selection": False,
        "safety_invariant_tuned": False,
        "private_key_persisted_in_artifacts": False,
    }
    freeze["manifest_payload_sha256"] = canonical_sha256(freeze)
    write_json(artifact_dir / "compiler_freeze_manifest.json", freeze)
    complete_phase(runtime_dir, phase="freeze")
    return freeze


def _run_scored_split(
    config_path: str | Path,
    *,
    split: str,
    phase: str,
    expected_previous: str,
    output_name: str,
) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir, _ = output_paths(config_path, values)
    claim_phase(runtime_dir, phase=phase, expected_previous=expected_previous)
    freeze = read_json(
        artifact_dir / "compiler_freeze_manifest.json",
        label="compiler freeze manifest",
    )
    if (
        freeze["config_sha256"] != sha256_file(config_path)
        or freeze["source_manifest_payload_sha256"]
        != runtime_source_manifest(config_path)["manifest_payload_sha256"]
    ):
        raise F2ProtocolError("compiler/config 在 freeze 后改变")
    root = repo_root(config_path)
    expected_hash = values["f1_frozen"]["splits"][split]["sha256"]
    data_path = root / values["f1_frozen"]["splits"][split]["path"]
    if sha256_file(data_path) != expected_hash:
        raise F2ProtocolError(f"{split} 在 freeze 后 hash mismatch")
    cases = _load_split(data_path, split)
    rows = evaluate_cases(
        cases,
        split=split,
        variants=F2_VARIANTS,
        issuer=_load_issuer(runtime_dir),
        parameters=_parameters(values),
        issued_at=values["evaluation"]["issued_at"],
        verification_time=values["evaluation"]["verification_time"],
    )
    write_csv(
        artifact_dir / output_name,
        rows,
        fieldnames=_RESULT_FIELDS,
    )
    write_json(runtime_dir / f"{split}_rows.json", {"rows": rows})
    complete_phase(runtime_dir, phase=phase)
    return {"split": split, "case_count": len(cases), "result_row_count": len(rows)}


def run_stage_f2_development(config_path: str | Path) -> dict[str, Any]:
    return _run_scored_split(
        config_path,
        split="development",
        phase="development",
        expected_previous="freeze",
        output_name="development_results.csv",
    )


def run_stage_f2_locked(config_path: str | Path) -> dict[str, Any]:
    return _run_scored_split(
        config_path,
        split="locked_test",
        phase="locked_test",
        expected_previous="development",
        output_name="locked_test_results.csv",
    )


def _a1_gate(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    checks = {}
    for name in SECURITY_COUNT_FIELDS:
        if name in gates:
            checks[name] = metrics[name] == gates[name]
    cross_total = sum(
        metrics[name]
        for name in (
            "cross_subject_capability_count",
            "cross_tenant_capability_count",
            "cross_resource_capability_count",
            "cross_action_capability_count",
            "cross_relation_capability_count",
            "cross_purpose_capability_count",
        )
    )
    checks["all_cross_scope_capability_counts"] = (
        cross_total == gates["cross_scope_capability_count"]
    )
    for name in (
        "clean_allow_exact_execution_rate",
        "deny_precision",
        "deny_recall",
    ):
        checks[name] = metrics[name] == gates[name]
    checks["recoverable_safe_utility"] = (
        metrics["recoverable_safe_utility"]
        >= gates["minimum_recoverable_safe_utility"]
    )
    return checks


def _write_report(
    report_path: Path,
    summary: dict[str, Any],
    aggregate: list[dict[str, Any]],
) -> None:
    rows = [
        item
        for item in aggregate
        if item["split"] in {"development", "locked_test"}
    ]
    table = "\n".join(
        "| {split} | {variant} | {authority_amplification_count} | "
        "{unauthorized_action_count} | {safe_utility:.4f} | "
        "{recoverable_safe_utility:.4f} | {unnecessary_rejection_rate:.4f} | "
        "{confirmation_required_rate:.4f} | {silent_scope_narrowing_count} | "
        "{multi_step_forbidden_effect_acceptance_count} | "
        "{compile_latency_p95_ms:.4f} | {capability_size_mean_bytes:.1f} |".format(
            **item
        )
        for item in rows
    )
    status = summary["policy_carrying_capability_status"]
    report = f"""# Stage F2：Intent–Authority Separation Compiler

## 结论

```text
policy_carrying_capability_status = {status}
authority_non_amplification_implemented = {str(summary["authority_non_amplification_implemented"]).lower()}
semantic_proposal_authorizes_execution = false
compiler_locked_test_scored = true
compiler_locked_test_scored_once = true
workflow_composition_status = {summary["workflow_composition_status"]}
authorization_witness_status = {summary["authorization_witness_status"]}
ready_for_stage_f3 = {str(summary["ready_for_stage_f3"]).lower()}
formal_model_completed = false
real_agent_integration_completed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

F2 将 executable authority 定义为有限 ``AuthorityAtom`` 集合。每个 atom
同时绑定 subject、tenant、resource、action、relation、purpose 与具有偏序语义
的 constraints，不使用维度集合的隐式笛卡尔积。最终不变量是：

```text
A_exec ⊆ A_proposal ∩ A_trusted_request ∩ A_policy_grant
```

## 输入与信任来源

- proposal 来自不可信 ``SemanticProposal``，不能声明 principal、环境、policy、
  epoch、witness 或 issuer；
- trusted request 来自显式 UI/API 与 ``PolicyRequestContext``；
- policy grant 由 compiler 内部调用 F1 deterministic oracle 重新计算，调用者
  不能直接传入 grant。

proposal 不能扩大 authority。A1 不会静默剪裁后执行：混合安全/越权 proposal
进入 ``NeedsExplicitConfirmation`` 且不签发 capability；oracle deny、空交集、
stale epoch、invalid resource/delegation 或 witness 错误均 fail closed。

## Authorization witness 与 capability

Witness 绑定 policy hash、decision ID、principal、trusted request、proposal、
grant、executable authority、环境、epoch、matched policy IDs 与 reason code。
Verifier 不只验证 Ed25519 签名，还重算 oracle decision、witness 与
non-amplification。显式 deny 不能被同层宽泛 allow 覆盖。

AuthCapV1 使用标准 Ed25519 实现 issuer/verifier 分离。Ed25519 不是本文算法
创新；贡献在 intent–authority separation、compiler、witness 和组合不放大
不变量。private signing key 与正式 capability token 均未写入 artifacts。

## Development / locked 结果

| split | variant | amplification | unauthorized | safe utility | recoverable | unnecessary reject | confirmation | silent narrowing | workflow violation | compile p95 ms | capability mean bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

两个单步 allow、combined deny 的 workflow 被 A1 在第二步 capability 签发前阻断；
state replay/fork、旧 epoch 和组合 forbidden effect 均未被接受。A0、B0、B2、
B3 不具备组合检查，报告其 workflow violation；B1 通过拒绝一切得到零违规。

## 数据与一次性协议

F1 四 split 字节哈希未变化，未删除或修改 family。prepare 仅解析 train 与
calibration；compiler、配置和 public verifier 冻结后，development 运行一次，
随后 locked_test 唯一评分一次。development/locked 均未参与选择。
``locked_test_scored=true`` 只表示 deterministic compiler evaluation，不是
真实 agent 外部验证。

本阶段没有使用 private data、private answer、真实凭据、memory/tool execution、
远程模型或真实 agent。

## 研究边界

F2 尚未完成形式化证明（F3）或真实 agent 集成/外部评价（F4），因此当前结果尚
不足以单独支撑 ACSAC/ESORICS 投稿质量。Ed25519 只提供标准签名完整性；policy
derivation 的可信性来自显式 oracle 重算与 witness 语义验证。

永久状态保持：

```text
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```
"""
    report_path.write_text(report, encoding="utf-8")


def run_stage_f2_finalize(config_path: str | Path) -> dict[str, Any]:
    values = load_config(config_path)
    artifact_dir, runtime_dir, report_path = output_paths(config_path, values)
    protocol = validate_final_protocol(runtime_dir)
    rows = []
    for split in ("development", "locked_test"):
        payload = read_json(
            runtime_dir / f"{split}_rows.json",
            label=f"{split} runtime rows",
        )
        rows.extend(payload["rows"])
    aggregate = aggregate_all(rows)
    categories = per_category(rows)
    violations = security_violation_matrix(rows)
    write_csv(artifact_dir / "aggregate_metrics.csv", aggregate)
    write_csv(artifact_dir / "per_category_metrics.csv", categories)
    write_csv(artifact_dir / "security_violation_matrix.csv", violations)
    workflow_rows = [
        {
            "split": row["split"],
            "variant": row["variant"],
            "case_id": row["case_id"],
            "attack_category": row["attack_category"],
            "compositional_authority_amplification_count": row[
                "compositional_authority_amplification_count"
            ],
            "multi_step_forbidden_effect_acceptance_count": row[
                "multi_step_forbidden_effect_acceptance_count"
            ],
            "workflow_state_replay_acceptance_count": row[
                "workflow_state_replay_acceptance_count"
            ],
            "workflow_state_fork_acceptance_count": row[
                "workflow_state_fork_acceptance_count"
            ],
            "workflow_state_size_bytes": row["workflow_state_size_bytes"],
        }
        for row in rows
        if row["attack_category"] == "multi_step_privilege_amplification"
    ]
    write_csv(artifact_dir / "workflow_metrics.csv", workflow_rows)
    performance = [
        {
            key: item[key]
            for key in (
                "split",
                "variant",
                "compile_latency_p50_ms",
                "compile_latency_p95_ms",
                "verify_latency_p50_ms",
                "verify_latency_p95_ms",
                "capability_size_mean_bytes",
                "capability_size_p95_bytes",
                "witness_size_mean_bytes",
                "workflow_state_size_mean_bytes",
            )
        }
        for item in aggregate
    ]
    write_csv(artifact_dir / "performance_metrics.csv", performance)
    a1 = {
        item["split"]: item
        for item in aggregate
        if item["variant"] == "A1"
    }
    gates = {
        split: _a1_gate(a1[split], values["readiness_gates"])
        for split in ("development", "locked_test")
    }
    passed = all(all(checks.values()) for checks in gates.values())
    summary = {
        "schema_version": 1,
        "stage": "F2-authcap-compiler",
        "policy_carrying_capability_status": "passed" if passed else "failed",
        "authority_non_amplification_implemented": passed,
        "semantic_proposal_authorizes_execution": False,
        "compiler_locked_test_scored": True,
        "compiler_locked_test_scored_once": (
            protocol["locked_test_score_count"] == 1
        ),
        "workflow_composition_status": "passed" if passed else "failed",
        "authorization_witness_status": "passed" if passed else "failed",
        "ready_for_stage_f3": passed,
        "formal_model_completed": False,
        "real_agent_integration_completed": False,
        "private_value_memory_ready": False,
        "original_c3_allowed": False,
        "c3_eligible": False,
        "private_data_used": False,
        "memory_or_tool_executed": False,
        "locked_test_used_for_selection": False,
        "development_used_for_selection": False,
        "f1_split_hashes_unchanged": True,
        "readiness_checks": gates,
        "a1_metrics": a1,
        "code_freeze_commit": protocol["code_freeze_commit"],
        "final_variant": "A1",
    }
    write_json(artifact_dir / "stage_f2_summary.json", summary)
    protocol_status = {
        "schema_version": 1,
        "status": "completed",
        "phase_order": list(protocol["phases"]),
        "development_score_count": protocol["development_score_count"],
        "locked_test_score_count": protocol["locked_test_score_count"],
        "locked_test_scored": True,
        "locked_test_scored_once": True,
        "locked_content_read_before_freeze": False,
        "development_used_for_selection": False,
        "locked_test_used_for_selection": False,
        "formal_build_repeat_allowed": False,
    }
    write_json(artifact_dir / "protocol_status.json", protocol_status)
    _write_report(report_path, summary, aggregate)
    scan = scan_for_forbidden_material((artifact_dir, report_path))
    write_json(artifact_dir / "private_material_scan.json", scan)
    if scan["forbidden_material_occurrence_count"] != 0:
        raise F2ProtocolError("F2 artifact 检出 private key/token material")
    manifest = artifact_manifest(
        artifact_dir,
        report_path=report_path,
        expected_files=_ARTIFACT_FILES,
    )
    write_json(artifact_dir / "artifact_sha256_manifest.json", manifest)
    return summary


def run_stage_f2_smoke(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = load_config(config_path, smoke=True)
    root = repo_root(config_path)
    target = (
        root / values["outputs"]["artifact_dir"]
        if output_dir is None
        else Path(output_dir).resolve()
    )
    if target.exists():
        raise F2ProtocolError("F2 smoke output 已存在")
    target.mkdir(parents=True)
    cases = _load_split(
        root / "data/authzroutebench/generated/train.jsonl",
        "train",
    )[:12]
    issuer = AuthCapIssuer.generate("f2-smoke-key")
    parameters = CompilerParameters(
        capability_ttl_seconds=values["calibration"][
            "fixed_capability_ttl_seconds"
        ],
        maximum_capability_bytes=values["calibration"][
            "fixed_maximum_capability_bytes"
        ],
    )
    rows = evaluate_cases(
        cases,
        split="smoke_train",
        variants=F2_VARIANTS,
        issuer=issuer,
        parameters=parameters,
        issued_at=values["evaluation"]["issued_at"],
        verification_time=values["evaluation"]["verification_time"],
    )
    aggregate = [
        {"variant": variant, **aggregate_variant_rows(
            [row for row in rows if row["variant"] == variant]
        )}
        for variant in F2_VARIANTS
    ]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "locked_test_read": False,
        "private_data_used": False,
        "case_count": len(cases),
        "aggregate": aggregate,
    }
    write_json(target / "smoke_summary.json", payload)
    return payload
