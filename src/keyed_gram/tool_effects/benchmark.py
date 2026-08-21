"""AuthEffectBench feasibility 的确定性 150-case 生成器。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .types import (
    DeclaredContract,
    EffectKind,
    EffectPhase,
    EffectSelector,
    EffectTemplate,
    Environment,
    ForbiddenRule,
    OmissionType,
    ToolCall,
    ToolEffectError,
    ToolImplementation,
    ToolTransition,
    TraceScenario,
    TraceStep,
    canonical_digest,
)

SPLITS = ("train", "calibration", "development", "locked_test")
MUTATIONS = (
    "hidden_effect",
    "parameter_role_omission",
    "state_dependent_effect",
    "composition_omission",
    "delayed_effect",
    "implementation_drift",
)


@dataclass(frozen=True)
class BaseTool:
    name: str
    domain: str
    split: str
    primary_kind: EffectKind
    destination: str | None


BASE_TOOLS = (
    BaseTool("file_read", "file_code", "train", EffectKind.READ, None),
    BaseTool("email_send", "email_calendar", "train", EffectKind.SEND, "internal-mail"),
    BaseTool("repo_commit", "file_code", "calibration", EffectKind.COMMIT, None),
    BaseTool("crm_query", "database_crm", "calibration", EffectKind.READ, None),
    BaseTool(
        "calendar_invite",
        "email_calendar",
        "development",
        EffectKind.CREATE,
        "internal-calendar",
    ),
    BaseTool(
        "cloud_deploy",
        "payment_cloud",
        "development",
        EffectKind.EXECUTE,
        "internal-cloud",
    ),
    BaseTool(
        "database_export",
        "database_crm",
        "locked_test",
        EffectKind.EXPORT,
        "internal-store",
    ),
    BaseTool(
        "payment_refund",
        "payment_cloud",
        "locked_test",
        EffectKind.WRITE,
        "internal-ledger",
    ),
)


@dataclass(frozen=True)
class EffectCase:
    case_id: str
    split: str
    domain: str
    base_tool_family: str
    mutation_category: str
    mutation_family_id: str
    workflow_template_id: str
    policy_template_id: str
    implementation_version_lineage: str
    hidden_effect_combination: str
    omission_type: OmissionType
    implementations: tuple[ToolImplementation, ...]
    declared_contracts: tuple[DeclaredContract, ...]
    characterized_contracts: tuple[DeclaredContract, ...]
    declared_policy_rules: tuple[ForbiddenRule, ...]
    safety_rules: tuple[ForbiddenRule, ...]
    benign_trace: TraceScenario
    attack_trace: TraceScenario
    expected_spurious_counterexample: bool

    def implementation_map(self) -> dict[tuple[str, str], ToolImplementation]:
        return {(item.tool_name, item.version): item for item in self.implementations}

    def to_public_dict(self) -> dict[str, Any]:
        """不保存自然语言、源码、具体效果值或真实数据。"""

        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "split": self.split,
            "domain": self.domain,
            "base_tool_family": self.base_tool_family,
            "mutation_category": self.mutation_category,
            "mutation_family_id": self.mutation_family_id,
            "workflow_template_id": self.workflow_template_id,
            "policy_template_id": self.policy_template_id,
            "implementation_version_lineage": (self.implementation_version_lineage),
            "hidden_effect_combination": self.hidden_effect_combination,
            "omission_type": self.omission_type.value,
            "implementation_count": len(self.implementations),
            "declared_contract_digest": canonical_digest(
                [contract.to_dict() for contract in self.declared_contracts]
            ),
            "characterized_contract_digest": canonical_digest(
                [contract.to_dict() for contract in self.characterized_contracts]
            ),
            "declared_policy_digest": canonical_digest(
                [rule.to_dict() for rule in self.declared_policy_rules]
            ),
            "safety_predicate_digest": canonical_digest(
                [rule.to_dict() for rule in self.safety_rules]
            ),
            "benign_trace_digest": canonical_digest(_trace_shape(self.benign_trace)),
            "attack_trace_digest": canonical_digest(_trace_shape(self.attack_trace)),
            "attack_trace_is_forbidden": self.attack_trace.expected_forbidden,
            "ground_truth_source": "finite_transition_system",
            "llm_judge_used": False,
            "public_or_synthetic_only": True,
        }


def _trace_shape(trace: TraceScenario) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "step_count": len(trace.steps),
        "tool_versions": [
            [step.call.tool_name, step.call.version] for step in trace.steps
        ],
        "expected_forbidden": trace.expected_forbidden,
    }


def _primary_template(base: BaseTool) -> EffectTemplate:
    return EffectTemplate(
        kind=base.primary_kind,
        tenant_ref="arg:tenant",
        resource_ref="arg:resource",
        destination_ref=(None if base.destination is None else "arg:destination"),
        security_roles=(
            ("tenant", "resource")
            if base.destination is None
            else ("destination", "tenant", "resource")
        ),
    )


def _danger_template(
    category: str,
    instance_index: int,
    *,
    phase: EffectPhase = EffectPhase.IMMEDIATE,
) -> EffectTemplate:
    kinds = (EffectKind.SEND, EffectKind.WRITE, EffectKind.EXPORT)
    kind = kinds[instance_index % len(kinds)]
    destination = (
        "arg:external_destination"
        if kind in {EffectKind.SEND, EffectKind.EXPORT}
        else None
    )
    roles = ["tenant", "hidden_resource"]
    if destination is not None:
        roles.append("external_destination")
    return EffectTemplate(
        kind=kind,
        tenant_ref="arg:tenant",
        resource_ref="arg:hidden_resource",
        destination_ref=destination,
        phase=phase,
        security_roles=tuple(roles),
    )


def _danger_rule(
    rule_id: str,
    template: EffectTemplate,
    call: ToolCall,
) -> ForbiddenRule:
    effect = template.instantiate(call)
    return ForbiddenRule(
        rule_id=rule_id,
        selectors=(
            EffectSelector(
                kind=effect.kind,
                tenant_id=effect.tenant_id,
                resource_id=effect.resource_id,
                destination_id=effect.destination_id,
                phase=effect.phase,
            ),
        ),
    )


def _call(
    case_id: str,
    tool_name: str,
    version: str,
    *,
    destination: str | None,
    suffix: str,
    callback: str = "internal-callback",
) -> ToolCall:
    arguments = {
        "tenant": f"tenant-{case_id}",
        "resource": f"resource-{case_id}",
        "hidden_resource": f"hidden-{case_id}",
        "external_destination": f"external-{suffix}",
        "callback_url": callback,
    }
    if destination is not None:
        arguments["destination"] = destination
    return ToolCall.create(
        f"call-{case_id}-{suffix}",
        tool_name,
        version,
        arguments,
    )


def _contract(
    case_id: str,
    tool_name: str,
    version: str,
    effects: Iterable[EffectTemplate],
    *,
    label: str,
) -> DeclaredContract:
    return DeclaredContract(
        contract_id=f"contract-{case_id}-{label}-{version}",
        tool_name=tool_name,
        bound_version=version,
        effects=frozenset(effects),
    )


def _base_transition(base: BaseTool) -> ToolTransition:
    return ToolTransition(
        transition_id="base",
        required_environment=(),
        immediate_effects=(_primary_template(base),),
    )


def _single_case(
    base: BaseTool,
    category: str,
    instance_index: int,
    *,
    seed_tag: str,
    variant_offset: int,
) -> EffectCase:
    marker = f"{base.split[:3]}-{seed_tag}-{base.name}-{category}-{instance_index}"
    case_id = f"fec-{marker}"
    primary = _primary_template(base)
    version = "v2" if category == "implementation_drift" else "v1"
    benign_version = "v1"
    benign_call = _call(
        case_id,
        base.name,
        benign_version,
        destination=base.destination,
        suffix="benign",
    )
    attack_call = _call(
        case_id,
        base.name,
        version,
        destination=base.destination,
        suffix="attack",
        callback=f"external-callback-{instance_index}",
    )
    safe_env = Environment.from_mapping(
        {"trigger": "off", "mode": "user", "async": "off"}
    )
    attack_env = Environment.from_mapping(
        {"trigger": "on", "mode": "admin", "async": "on"}
    )
    transitions = [_base_transition(base)]
    declared_effects = [primary]
    characterized_effects = [primary]
    implementations: list[ToolImplementation] = []
    declared_policy: tuple[ForbiddenRule, ...] = ()
    spurious = False

    if category == "parameter_role_omission":
        danger = EffectTemplate(
            kind=EffectKind.CALLBACK,
            tenant_ref="arg:tenant",
            resource_ref="arg:resource",
            destination_ref="arg:callback_url",
            security_roles=("tenant", "resource", "callback_url"),
        )
        incomplete = EffectTemplate(
            kind=EffectKind.CALLBACK,
            tenant_ref="arg:tenant",
            resource_ref="arg:resource",
            # 声明只承认“发生 callback”，却没有把 callback_url 识别为
            # destination security role；因此外部目的地约束无法匹配。
            destination_ref=None,
            security_roles=("tenant", "resource"),
        )
        transitions.append(
            ToolTransition(
                transition_id="callback",
                required_environment=(("trigger", "on"),),
                immediate_effects=(danger,),
            )
        )
        declared_effects.append(incomplete)
        if instance_index == 0:
            characterized_effects.append(danger)
        omission = OmissionType.CONTRACT_OMISSION
    elif category == "delayed_effect":
        queue = EffectTemplate(
            kind=EffectKind.QUEUE,
            tenant_ref="arg:tenant",
            resource_ref="arg:hidden_resource",
            phase=EffectPhase.IMMEDIATE,
            security_roles=("tenant", "hidden_resource"),
        )
        danger = _danger_template(
            category,
            instance_index + variant_offset,
            phase=EffectPhase.DELAYED,
        )
        transitions.append(
            ToolTransition(
                transition_id="async",
                required_environment=(("async", "on"),),
                immediate_effects=(queue,),
                delayed_effects=(danger,),
            )
        )
        declared_effects.append(queue)
        characterized_effects.append(queue)
        omission = OmissionType.CONTRACT_OMISSION
    elif category == "composition_omission":
        danger = EffectTemplate(
            kind=EffectKind.SEND,
            tenant_ref="arg:tenant",
            resource_ref="arg:hidden_resource",
            destination_ref="arg:external_destination",
            security_roles=(
                "tenant",
                "hidden_resource",
                "external_destination",
            ),
        )
        omission = OmissionType.COMPOSITION_OMISSION
    else:
        danger = _danger_template(category, instance_index + variant_offset)
        required = (
            (("mode", "admin"),)
            if category == "state_dependent_effect"
            else (("trigger", "on"),)
        )
        if category == "implementation_drift":
            required = ()
        transitions.append(
            ToolTransition(
                transition_id="hidden",
                required_environment=required,
                immediate_effects=(danger,),
            )
        )
        if (
            category == "hidden_effect"
            and instance_index < 2
            or category == "state_dependent_effect"
            and instance_index == 0
            or category == "implementation_drift"
            and instance_index < 2
        ):
            characterized_effects.append(danger)
        if category == "state_dependent_effect" and instance_index == 2:
            # 初始抽象看见 latent effect，却丢失 guard，因此会对 benign
            # 环境产生可由 replay 排除的伪反例。
            characterized_effects.append(danger)
            spurious = True
        if category == "implementation_drift":
            omission = OmissionType.IMPLEMENTATION_DRIFT
        elif category == "state_dependent_effect" and instance_index == 0:
            # 工具 contract 已声明 mode-dependent latent effect，但 policy
            # 遗漏对应禁止条件，形成独立于 contract omission 的样本。
            declared_effects.append(danger)
            omission = OmissionType.POLICY_OMISSION
        else:
            omission = OmissionType.CONTRACT_OMISSION

    if category == "implementation_drift":
        implementations.append(
            ToolImplementation(base.name, "v1", (_base_transition(base),))
        )
        implementations.append(ToolImplementation(base.name, "v2", tuple(transitions)))
        declared_contracts = (
            _contract(case_id, base.name, "v1", (primary,), label="declared"),
        )
        characterized_contracts = list(declared_contracts)
        if instance_index < 2:
            characterized_contracts.append(
                _contract(
                    case_id,
                    base.name,
                    "v2",
                    characterized_effects,
                    label="characterized",
                )
            )
    else:
        implementations.append(ToolImplementation(base.name, "v1", tuple(transitions)))
        declared_contracts = (
            _contract(
                case_id,
                base.name,
                "v1",
                declared_effects,
                label="declared",
            ),
        )
        characterized_contracts = [
            _contract(
                case_id,
                base.name,
                "v1",
                characterized_effects,
                label="characterized",
            )
        ]

    benign_steps = (TraceStep(benign_call, safe_env),)
    attack_steps: tuple[TraceStep, ...] = (TraceStep(attack_call, attack_env),)
    if category == "composition_omission":
        bridge_name = f"bridge-{base.name}"
        bridge_call = _call(
            case_id,
            bridge_name,
            "v1",
            destination="external-bridge",
            suffix="bridge",
        )
        bridge_impl = ToolImplementation(
            bridge_name,
            "v1",
            (
                ToolTransition(
                    transition_id="send",
                    required_environment=(),
                    immediate_effects=(danger,),
                ),
            ),
        )
        bridge_contract = _contract(
            case_id,
            bridge_name,
            "v1",
            (danger,),
            label="declared",
        )
        implementations.append(bridge_impl)
        declared_contracts = (*declared_contracts, bridge_contract)
        characterized_contracts.append(bridge_contract)
        attack_steps = (
            TraceStep(attack_call, safe_env),
            TraceStep(bridge_call, safe_env),
        )
        first_effect = primary.instantiate(attack_call)
        second_effect = danger.instantiate(bridge_call)
        safety_rule = ForbiddenRule(
            rule_id=f"forbid-composition-{case_id}",
            selectors=(
                EffectSelector(
                    first_effect.kind,
                    tenant_id=first_effect.tenant_id,
                    resource_id=first_effect.resource_id,
                    destination_id=first_effect.destination_id,
                ),
                EffectSelector(
                    second_effect.kind,
                    tenant_id=second_effect.tenant_id,
                    resource_id=second_effect.resource_id,
                    destination_id=second_effect.destination_id,
                ),
            ),
        )
    else:
        safety_rule = _danger_rule(f"forbid-danger-{case_id}", danger, attack_call)
    # contract/drift omission 仍有独立安全谓词；policy/composition omission
    # 则有意从已声明策略中删除该谓词，用于区分“效果漏报”和“策略漏报”。
    if omission in {
        OmissionType.CONTRACT_OMISSION,
        OmissionType.IMPLEMENTATION_DRIFT,
    }:
        declared_policy = (safety_rule,)

    benign = TraceScenario(
        trace_id=f"trace-{case_id}-benign",
        steps=benign_steps,
        expected_forbidden=False,
    )
    attack = TraceScenario(
        trace_id=f"trace-{case_id}-attack",
        steps=attack_steps,
        expected_forbidden=True,
    )
    return EffectCase(
        case_id=case_id,
        split=base.split,
        domain=base.domain,
        base_tool_family=base.name,
        mutation_category=category,
        mutation_family_id=f"mf-{marker}",
        workflow_template_id=f"wf-{base.split}-{base.name}-{category}-{instance_index}",
        policy_template_id=f"pol-{base.split}-{category}-{instance_index}",
        implementation_version_lineage=f"lineage-{base.split}-{base.name}-{category}",
        hidden_effect_combination=f"combo-{base.split}-{category}-{instance_index}",
        omission_type=omission,
        implementations=tuple(implementations),
        declared_contracts=tuple(declared_contracts),
        characterized_contracts=tuple(characterized_contracts),
        declared_policy_rules=declared_policy,
        safety_rules=(safety_rule,),
        benign_trace=benign,
        attack_trace=attack,
        expected_spurious_counterexample=spurious,
    )


def _clean_case(base: BaseTool, control_index: int, *, seed_tag: str) -> EffectCase:
    case_id = f"fec-{base.split[:3]}-{seed_tag}-{base.name}-clean-{control_index}"
    primary = _primary_template(base)
    implementation = ToolImplementation(
        base.name,
        "v1",
        (_base_transition(base),),
    )
    contract = _contract(
        case_id,
        base.name,
        "v1",
        (primary,),
        label="clean",
    )
    benign_call = _call(
        case_id,
        base.name,
        "v1",
        destination=base.destination,
        suffix="benign",
    )
    alternate_call = _call(
        case_id,
        base.name,
        "v1",
        destination=base.destination,
        suffix="alternate",
    )
    env = Environment.from_mapping({"trigger": "off", "mode": "user", "async": "off"})
    impossible_rule = ForbiddenRule(
        rule_id=f"forbid-impossible-{case_id}",
        selectors=(
            EffectSelector(
                EffectKind.DELETE,
                tenant_id=f"tenant-{case_id}",
                resource_id=f"never-{case_id}",
            ),
        ),
    )
    return EffectCase(
        case_id=case_id,
        split=base.split,
        domain=base.domain,
        base_tool_family=base.name,
        mutation_category="clean_control",
        mutation_family_id=f"mf-{base.split}-{base.name}-clean-{control_index}",
        workflow_template_id=f"wf-{base.split}-{base.name}-clean-{control_index}",
        policy_template_id=f"pol-{base.split}-clean-{control_index}",
        implementation_version_lineage=f"lineage-{base.split}-{base.name}-clean",
        hidden_effect_combination=f"combo-{base.split}-clean-{control_index}",
        omission_type=OmissionType.NONE,
        implementations=(implementation,),
        declared_contracts=(contract,),
        characterized_contracts=(contract,),
        declared_policy_rules=(impossible_rule,),
        safety_rules=(impossible_rule,),
        benign_trace=TraceScenario(
            f"trace-{case_id}-benign",
            (TraceStep(benign_call, env),),
            False,
        ),
        attack_trace=TraceScenario(
            f"trace-{case_id}-alternate",
            (TraceStep(alternate_call, env),),
            False,
        ),
        expected_spurious_counterexample=False,
    )


def generate_feasibility_cases(
    benchmark_seed: str = "preflight-fixture-v1",
) -> tuple[EffectCase, ...]:
    """生成 commit-bound benchmark；默认种子只能用于单元测试/Smoke。"""

    if not isinstance(benchmark_seed, str) or not benchmark_seed:
        raise ToolEffectError("benchmark_seed 必须是非空字符串")
    seed_digest = canonical_digest(
        {"benchmark_seed": benchmark_seed, "schema_version": 1}
    )
    seed_tag = seed_digest[:12]
    cases = [
        _single_case(
            base,
            category,
            instance_index,
            seed_tag=seed_tag,
            variant_offset=(
                int(
                    canonical_digest(
                        [benchmark_seed, base.name, category, instance_index]
                    )[:8],
                    16,
                )
                % 3
            ),
        )
        for base in BASE_TOOLS
        for category in MUTATIONS
        for instance_index in range(3)
    ]
    for index, base in enumerate(BASE_TOOLS[:6]):
        cases.append(_clean_case(base, index, seed_tag=seed_tag))
    result = tuple(sorted(cases, key=lambda case: case.case_id))
    if len(result) != 150 or len({case.case_id for case in result}) != 150:
        raise RuntimeError("feasibility benchmark 必须精确包含 150 个唯一 case")
    return result


def split_cases(cases: Iterable[EffectCase]) -> dict[str, tuple[EffectCase, ...]]:
    result = {
        split: tuple(
            sorted(
                (case for case in cases if case.split == split),
                key=lambda item: item.case_id,
            )
        )
        for split in SPLITS
    }
    if sum(len(rows) for rows in result.values()) != 150:
        raise RuntimeError("split case 数量不完整")
    return result


def split_collision_audit(
    splits: dict[str, tuple[EffectCase, ...]],
) -> dict[str, Any]:
    dimensions = (
        "base_tool_family",
        "mutation_family_id",
        "workflow_template_id",
        "policy_template_id",
        "implementation_version_lineage",
        "hidden_effect_combination",
    )
    collisions: list[dict[str, Any]] = []
    for dimension in dimensions:
        owners: dict[str, set[str]] = {}
        for split, rows in splits.items():
            for row in rows:
                owners.setdefault(getattr(row, dimension), set()).add(split)
        for value, owner_splits in sorted(owners.items()):
            if len(owner_splits) > 1:
                collisions.append(
                    {
                        "dimension": dimension,
                        "value_digest": canonical_digest(value),
                        "splits": sorted(owner_splits),
                    }
                )
    return {
        "schema_version": 1,
        "dimensions": list(dimensions),
        "cross_split_collision_count": len(collisions),
        "collisions": collisions,
    }
