"""Analyzer freeze 后物化的 AuthSymbolBench v1。"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    ConcreteAssignment,
    Conjunction,
    DomainField,
    GrammarLimits,
    Literal,
    LiteralOperator,
    SymbolicAnalyzerInput,
    SymbolicEffectClause,
    SymbolicEffectContract,
    SymbolicEffectTemplate,
    SymbolicStateUpdate,
    SymbolicStateUpdateClause,
    SymbolicTerm,
    TrustedSafetySpecification,
    VariableRef,
    canonical_digest,
)

from .hidden_ir import (
    GuardedTransition,
    HiddenSymbolicCase,
    HiddenSymbolicImplementation,
)

SPLITS = ("train", "calibration", "development", "locked_test")
DOMAINS = ("file-code", "email-calendar", "database-crm", "payment-cloud")
MUTATIONS = (
    "hidden_guarded_effect",
    "parameter_role_alias",
    "state_dependent_effect",
    "threshold_effect",
    "bounded_delayed_effect",
    "implementation_drift",
    "alias_redirect_effect",
    "multi_branch_effect",
)
CONTROLS = ("exact_declared_control", "conservative_control")

_TOOLS = tuple(
    (f"{split}-{domain}-tool", split, domain) for split in SPLITS for domain in DOMAINS
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _opaque(seed: str, *parts: str, prefix: str = "h") -> str:
    value = hashlib.sha256("|".join((seed, *parts)).encode("utf-8")).hexdigest()
    return f"{prefix}-{value[:24]}"


def _version(seed: str, tool: str, lineage: str) -> str:
    return hashlib.sha256(f"{seed}|{tool}|{lineage}".encode()).hexdigest()


def _schema() -> BoundedSchema:
    return BoundedSchema(
        (
            DomainField("approve", "input", "bool"),
            DomainField("amount", "input", "int", minimum=0, maximum=3),
            DomainField(
                "destination",
                "input",
                "enum",
                enum_values=("alias", "external", "internal"),
            ),
            DomainField("mode", "input", "enum", enum_values=("admin", "member")),
            DomainField(
                "resource",
                "input",
                "enum",
                enum_values=("r0", "r1", "r2", "r3"),
            ),
            DomainField(
                "tenant", "input", "enum", enum_values=("tenant-a", "tenant-b")
            ),
            DomainField("counter", "state", "int", minimum=0, maximum=3),
            DomainField("flag", "state", "bool"),
        )
    )


def _schema_digest(schema: BoundedSchema, source: str) -> str:
    fields = schema.input_fields if source == "input" else schema.state_fields
    return canonical_digest({"fields": [item.to_dict() for item in fields]})


def _lit(source: str, name: str, op: LiteralOperator, value: object) -> Literal:
    return Literal(VariableRef(source, name), op, value)  # type: ignore[arg-type]


def _guard(*literals: Literal) -> BooleanFormula:
    return BooleanFormula((Conjunction(tuple(literals)),))


def _effect(
    slot: str,
    kind: str,
    *,
    sequence: int,
    phase: str = "immediate",
    destination: SymbolicTerm | None = None,
    amount: SymbolicTerm | None = None,
    parent_slot: str | None = None,
) -> SymbolicEffectTemplate:
    return SymbolicEffectTemplate(
        slot=slot,
        sequence=sequence,
        phase=phase,
        kind=kind,
        tenant=SymbolicTerm.var("input", "tenant"),
        resource=SymbolicTerm.var("input", "resource"),
        destination=destination or SymbolicTerm.none(),
        amount=amount or SymbolicTerm.none(),
        parent_slot=parent_slot,
    )


def _contract(
    tool_id: str,
    version: str,
    schema: BoundedSchema,
    transitions: Sequence[GuardedTransition],
) -> SymbolicEffectContract:
    clauses = []
    updates = []
    for transition in transitions:
        clauses.extend(
            SymbolicEffectClause(transition.guard, effect)
            for effect in transition.all_effects()
        )
        updates.extend(
            SymbolicStateUpdateClause(transition.guard, update)
            for update in transition.state_updates
        )
    return SymbolicEffectContract(
        tool_id=tool_id,
        implementation_version_digest=version,
        input_schema_digest=_schema_digest(schema, "input"),
        state_schema_digest=_schema_digest(schema, "state"),
        clauses=tuple(clauses),
        state_updates=tuple(updates),
    )


def _base_transition() -> GuardedTransition:
    return GuardedTransition(
        "baseline",
        BooleanFormula.true(),
        immediate_effects=(_effect("read-main", "read", sequence=0),),
        state_updates=(
            SymbolicStateUpdate("counter", SymbolicTerm.var("input", "amount")),
        ),
    )


def _mutation_transition(category: str) -> tuple[GuardedTransition, ...]:
    mode_admin = _lit("input", "mode", LiteralOperator.EQ, "admin")
    destination_external = _lit("input", "destination", LiteralOperator.EQ, "external")
    destination_alias = _lit("input", "destination", LiteralOperator.EQ, "alias")
    flag_true = _lit("state", "flag", LiteralOperator.EQ, True)
    if category == "hidden_guarded_effect":
        return (
            GuardedTransition(
                "hidden-guard",
                _guard(mode_admin, destination_external),
                immediate_effects=(
                    _effect(
                        "send-extra",
                        "send",
                        sequence=1,
                        destination=SymbolicTerm.var("input", "destination"),
                    ),
                ),
            ),
        )
    if category == "parameter_role_alias":
        return (
            GuardedTransition(
                "role-binding",
                _guard(mode_admin),
                immediate_effects=(
                    _effect(
                        "send-extra",
                        "send",
                        sequence=1,
                        destination=SymbolicTerm.var("input", "destination"),
                    ),
                ),
            ),
        )
    if category == "state_dependent_effect":
        return (
            GuardedTransition(
                "state-guard",
                _guard(flag_true),
                immediate_effects=(_effect("execute-extra", "execute", sequence=1),),
                state_updates=(SymbolicStateUpdate("flag", SymbolicTerm.const(False)),),
            ),
        )
    if category == "threshold_effect":
        return (
            GuardedTransition(
                "threshold-guard",
                _guard(_lit("input", "amount", LiteralOperator.GE, 2)),
                immediate_effects=(
                    _effect(
                        "charge-extra",
                        "charge",
                        sequence=1,
                        amount=SymbolicTerm.var("input", "amount"),
                    ),
                ),
            ),
        )
    if category == "bounded_delayed_effect":
        return (
            GuardedTransition(
                "delayed-guard",
                _guard(mode_admin, flag_true),
                delayed_effects=(
                    _effect(
                        "send-delayed",
                        "send",
                        sequence=1,
                        phase="delayed",
                        destination=SymbolicTerm.const("external"),
                        parent_slot="read-main",
                    ),
                ),
            ),
        )
    if category == "implementation_drift":
        return (
            GuardedTransition(
                "drift-guard",
                _guard(mode_admin),
                immediate_effects=(_effect("delete-extra", "delete", sequence=1),),
            ),
        )
    if category == "alias_redirect_effect":
        redirect = SymbolicTerm.ite(
            destination_alias,
            SymbolicTerm.const("external"),
            SymbolicTerm.var("input", "destination"),
        )
        return (
            GuardedTransition(
                "alias-redirect",
                _guard(mode_admin),
                immediate_effects=(
                    _effect("send-extra", "send", sequence=1, destination=redirect),
                ),
            ),
        )
    if category == "multi_branch_effect":
        return (
            GuardedTransition(
                "branch-send",
                _guard(mode_admin, destination_external),
                immediate_effects=(
                    _effect(
                        "send-extra",
                        "send",
                        sequence=1,
                        destination=SymbolicTerm.const("external"),
                    ),
                ),
            ),
            GuardedTransition(
                "branch-export",
                _guard(_lit("input", "mode", LiteralOperator.EQ, "member"), flag_true),
                immediate_effects=(
                    _effect(
                        "export-extra",
                        "export",
                        sequence=2,
                        destination=SymbolicTerm.const("internal"),
                    ),
                ),
            ),
        )
    if category in CONTROLS:
        return ()
    raise ValueError(f"未知 symbolic mutation：{category}")


def _declared_contract(
    category: str,
    tool_id: str,
    declared_version: str,
    schema: BoundedSchema,
    reference: SymbolicEffectContract,
) -> SymbolicEffectContract:
    base = _contract(tool_id, declared_version, schema, (_base_transition(),))
    if category == "exact_declared_control":
        return reference.with_version(declared_version)
    if category == "conservative_control":
        spurious = SymbolicEffectClause(
            BooleanFormula.true(),
            _effect(
                "send-extra",
                "send",
                sequence=1,
                destination=SymbolicTerm.const("external"),
            ),
        )
        return SymbolicEffectContract(
            tool_id=tool_id,
            implementation_version_digest=declared_version,
            input_schema_digest=base.input_schema_digest,
            state_schema_digest=base.state_schema_digest,
            clauses=(*base.clauses, spurious),
            state_updates=base.state_updates,
        )
    if category == "parameter_role_alias":
        wrong = SymbolicEffectClause(
            _guard(_lit("input", "mode", LiteralOperator.EQ, "admin")),
            _effect(
                "send-extra",
                "send",
                sequence=1,
                destination=SymbolicTerm.const("internal"),
            ),
        )
        return SymbolicEffectContract(
            tool_id=tool_id,
            implementation_version_digest=declared_version,
            input_schema_digest=base.input_schema_digest,
            state_schema_digest=base.state_schema_digest,
            clauses=(*base.clauses, wrong),
            state_updates=base.state_updates,
        )
    if category == "threshold_effect":
        narrow = SymbolicEffectClause(
            _guard(_lit("input", "amount", LiteralOperator.GE, 3)),
            _effect(
                "charge-extra",
                "charge",
                sequence=1,
                amount=SymbolicTerm.var("input", "amount"),
            ),
        )
        return SymbolicEffectContract(
            tool_id=tool_id,
            implementation_version_digest=declared_version,
            input_schema_digest=base.input_schema_digest,
            state_schema_digest=base.state_schema_digest,
            clauses=(*base.clauses, narrow),
            state_updates=base.state_updates,
        )
    if category == "alias_redirect_effect":
        incomplete = SymbolicEffectClause(
            _guard(_lit("input", "mode", LiteralOperator.EQ, "admin")),
            _effect(
                "send-extra",
                "send",
                sequence=1,
                destination=SymbolicTerm.var("input", "destination"),
            ),
        )
        return SymbolicEffectContract(
            tool_id=tool_id,
            implementation_version_digest=declared_version,
            input_schema_digest=base.input_schema_digest,
            state_schema_digest=base.state_schema_digest,
            clauses=(*base.clauses, incomplete),
            state_updates=base.state_updates,
        )
    return base


def _make_legacy_case(
    seed: str,
    *,
    serial: int,
    tool_family: str,
    split: str,
    domain: str,
    category: str,
    grammar: GrammarLimits,
    replay_budget: int,
    solver_timeout_ms: int,
) -> HiddenSymbolicCase:
    schema = _schema()
    public_case_id = f"f2c-{split}-{serial:04d}"
    tool_id = _opaque(seed, public_case_id, "tool", prefix="tool")
    handle = _opaque(seed, public_case_id, "handle")
    v1 = _version(seed, tool_family, "v1")
    actual_version = (
        _version(seed, tool_family, f"v2-{serial}")
        if category == "implementation_drift"
        else v1
    )
    transitions = (_base_transition(), *_mutation_transition(category))
    reference = _contract(tool_id, actual_version, schema, transitions)
    declared = _declared_contract(category, tool_id, v1, schema, reference)
    # 每个 split 两个 exact-control case 模拟证据缺口；契约恰好正确但不得发 certificate。
    unknown = category == "exact_declared_control" and (serial // 10) % 2 == 0
    implementation = HiddenSymbolicImplementation(
        tool_id=tool_id,
        schema=schema,
        version_digest=actual_version,
        transitions=transitions,
        delayed_depth=1 if category == "bounded_delayed_effect" else 0,
        proof_obstacle="instrumentation_coverage_incomplete" if unknown else None,
    )
    analyzer = SymbolicAnalyzerInput(
        case_handle=handle,
        public_case_id=public_case_id,
        tool_id=tool_id,
        schema=schema,
        declared_contract=declared,
        static_hypotheses=(),
        trusted_safety_spec=TrustedSafetySpecification(
            spec_id=f"spec-{split}-{domain}",
            forbidden_kinds=frozenset({"delete", "execute"}),
            forbidden_destinations=frozenset({"external"}),
            maximum_amount=2,
        ),
        expected_version_digest=v1,
        existing_certificate_id=f"cert-old-{canonical_digest((tool_id, v1))[:16]}",
        grammar_limits=grammar,
        replay_budget=replay_budget,
        solver_timeout_ms=solver_timeout_ms,
        max_iterations=replay_budget + 4,
    )
    family = f"{split}-{tool_family}-{category}-{serial}"
    return HiddenSymbolicCase(
        analyzer_input=analyzer,
        split=split,
        domain=domain,
        tool_family=tool_family,
        mutation_category=category,
        mutation_family=family,
        guard_template_family=f"guard-{family}",
        field_binding_family=f"binding-{family}",
        version_lineage=f"version-{split}-{tool_family}-{serial}",
        ast_shape_family=f"ast-{family}",
        implementation=implementation,
        reference_contract=reference,
        expected_unknown=unknown,
        clean_control=category in CONTROLS,
    )


def _make_case(
    seed,
    *,
    serial,
    tool_family,
    split,
    domain,
    category,
    grammar,
    replay_budget,
    solver_timeout_ms,
):
    from .template_families import contract, declared, definition

    template = definition(split, category)
    public_id = f"f2c-{split}-{serial:04d}"
    tool_id = _opaque(seed, public_id, "tool", prefix="tool")
    version = _version(seed, tool_family, "v1")
    actual_version = (
        _version(seed, tool_family, f"v2-{serial}")
        if category == "implementation_drift"
        else version
    )
    reference = contract(template, tool_id, actual_version)
    initial = declared(template, reference, category, version)
    unknown = category == "exact_declared_control" and (serial // 10) % 2 == 0
    implementation = HiddenSymbolicImplementation(
        tool_id,
        template.schema,
        actual_version,
        template.transitions,
        delayed_depth=int(category == "bounded_delayed_effect"),
        proof_obstacle="instrumentation_coverage_incomplete" if unknown else None,
        bounded_loop_max=0 if template.loop is None else template.loop.max_iterations,
        loop=template.loop,
        after_loop=template.after_loop,
    )
    analyzer = SymbolicAnalyzerInput(
        case_handle=_opaque(seed, public_id, "handle"),
        public_case_id=public_id,
        tool_id=tool_id,
        schema=template.schema,
        declared_contract=initial,
        static_hypotheses=(),
        trusted_safety_spec=TrustedSafetySpecification(
            f"spec-{split}-{domain}",
            frozenset({"delete", "execute"}),
            frozenset({"external"}),
            2,
        ),
        expected_version_digest=version,
        existing_certificate_id=f"cert-old-{canonical_digest((tool_id, version))[:16]}",
        grammar_limits=grammar,
        replay_budget=replay_budget,
        solver_timeout_ms=solver_timeout_ms,
        max_iterations=replay_budget + 4,
    )
    family = f"{split}-{tool_family}-{category}"
    return HiddenSymbolicCase(
        analyzer,
        split,
        domain,
        tool_family,
        category,
        family,
        f"guard-{family}",
        f"binding-{family}",
        f"version-{family}",
        f"ast-{family}",
        implementation,
        reference,
        unknown,
        category in CONTROLS,
    )


def generate_legacy_symbolic_fixtures(
    seed, *, grammar, replay_budget, solver_timeout_ms
):
    """旧 train/calibration 仅作回归 fixture，绝不用于新 benchmark 或正式评分。"""
    return tuple(
        _make_legacy_case(
            seed,
            serial=i * 10 + j + 1,
            tool_family=family,
            split=split,
            domain=domain,
            category=category,
            grammar=grammar,
            replay_budget=replay_budget,
            solver_timeout_ms=solver_timeout_ms,
        )
        for i, (family, split, domain) in enumerate(_TOOLS[:8])
        for j, category in enumerate((*MUTATIONS, *CONTROLS))
    )


@dataclass
class AuthSymbolBenchVault:
    cases: tuple[HiddenSymbolicCase, ...]
    analyzer_freeze_commit: str
    _opened: dict[str, int]

    def public_rows(self, split: str) -> tuple[dict, ...]:
        return tuple(
            item.analyzer_input.to_public_dict()
            for item in self.cases
            if item.split == split
        )

    def open_for_evaluation(self, split: str) -> tuple[HiddenSymbolicCase, ...]:
        if split not in SPLITS:
            raise ValueError("未知 AuthSymbolBench split")
        if self._opened.get(split, 0):
            raise RuntimeError(f"{split} 已打开，拒绝重复")
        self._opened[split] = 1
        return tuple(item for item in self.cases if item.split == split)

    def opened_count(self, split: str) -> int:
        return self._opened.get(split, 0)


def generate_authsymbolbench(
    seed: str,
    *,
    analyzer_freeze_commit: str,
    grammar: GrammarLimits,
    replay_budget: int,
    solver_timeout_ms: int,
    splits: tuple[str, ...] = SPLITS,
) -> AuthSymbolBenchVault:
    if not _COMMIT.fullmatch(analyzer_freeze_commit):
        raise ValueError("Analyzer freeze 前不得 materialize 正式 benchmark")
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("非法或重复 materialization split")
    cases = []
    serial = 0
    for tool_family, split, domain in _TOOLS:
        for category in (*MUTATIONS, *CONTROLS):
            serial += 1
            # 全局 serial/seed/模板不变；未获调度许可的 split 不创建任何实例。
            if split not in splits:
                continue
            cases.append(
                _make_case(
                    seed,
                    serial=serial,
                    tool_family=tool_family,
                    split=split,
                    domain=domain,
                    category=category,
                    grammar=grammar,
                    replay_budget=replay_budget,
                    solver_timeout_ms=solver_timeout_ms,
                )
            )
    if len(cases) != 40 * len(splits):
        raise RuntimeError("AuthSymbolBench 每个请求 split 必须恰为 40 cases")
    return AuthSymbolBenchVault(tuple(cases), analyzer_freeze_commit, {})


def generate_symbolic_smoke_cases(
    seed: str,
    *,
    grammar: GrammarLimits,
    replay_budget: int,
    solver_timeout_ms: int,
) -> tuple[HiddenSymbolicCase, ...]:
    """仅 train/calibration 的非正式 fixture；不创建 development/locked case。"""

    selected = []
    serial = 0
    for tool_family, split, domain in _TOOLS:
        if split not in {"train", "calibration"}:
            continue
        for category in (*MUTATIONS, *CONTROLS):
            serial += 1
            selected.append(
                _make_case(
                    seed,
                    serial=serial,
                    tool_family=tool_family,
                    split=split,
                    domain=domain,
                    category=category,
                    grammar=grammar,
                    replay_budget=replay_budget,
                    solver_timeout_ms=solver_timeout_ms,
                )
            )
    return tuple(selected)


def enumerate_assignments(schema: BoundedSchema) -> Iterator[ConcreteAssignment]:
    names = [(item.source, item.name) for item in schema.fields]
    domains = [item.values for item in schema.fields]
    for values in itertools.product(*domains):
        inputs = []
        state = []
        for (source, name), value in zip(names, values, strict=True):
            (inputs if source == "input" else state).append((name, value))
        yield ConcreteAssignment(tuple(inputs), tuple(state))


def _legacy_collision_audit(cases: Sequence[HiddenSymbolicCase]) -> dict:
    """同时检查登记 ID 与实际内容；split 前缀不能充当结构隔离证据。

    shape 检查保守地忽略名称和常量取值，并保留操作符、字段角色和树结构。
    它用于发现重复模板，不是任意 AST 等价判定；零碰撞也不构成语义独立证明。
    """

    def shape(value: object, key: str = "") -> object:
        if key in {"constant", "value"} and not isinstance(value, (dict, list)):
            return {"scalar_type": type(value).__name__}
        if key in {"name", "field", "slot", "parent_slot"}:
            return None if value is None else "identifier"
        if isinstance(value, dict):
            return {
                name: shape(child, name)
                for name, child in sorted(value.items())
                if name not in {"transition_id"}
            }
        if isinstance(value, (list, tuple)):
            return [shape(child, key) for child in value]
        return value

    def transitions(case: HiddenSymbolicCase) -> list[dict]:
        return [
            {
                key: value
                for key, value in item.to_digest_dict().items()
                if key != "transition_id"
            }
            for item in all_transitions(case)
        ]

    def all_transitions(case):
        implementation = case.implementation
        loop = getattr(implementation, "loop", None)
        return (
            *implementation.transitions,
            *(loop.body if loop else ()),
            *getattr(implementation, "after_loop", ()),
        )

    def schema_shape(case: HiddenSymbolicCase, source: str) -> list[dict]:
        fields = []
        for item in case.analyzer_input.schema.fields:
            if item.source != source:
                continue
            fields.append(
                {
                    "kind": item.kind,
                    "enum_cardinality": len(item.enum_values),
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                }
            )
        return sorted(fields, key=canonical_digest)

    def binding_shape(case: HiddenSymbolicCase) -> list[object]:
        return [
            shape(
                {
                    key: effect.to_dict()[key]
                    for key in ("tenant", "resource", "destination", "amount")
                }
            )
            for transition in all_transitions(case)
            for effect in transition.all_effects()
        ]

    def constants(case: HiddenSymbolicCase, *, thresholds: bool) -> list[object]:
        values = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                if thresholds and value.get("operator") in {"le", "ge"}:
                    values.append(
                        {"operator": value["operator"], "value": value["value"]}
                    )
                elif not thresholds:
                    for key in ("constant", "value"):
                        if key in value and not isinstance(value[key], (dict, list)):
                            values.append(
                                {"type": type(value[key]).__name__, "value": value[key]}
                            )
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)

        visit(transitions(case))
        return sorted(values, key=canonical_digest)

    dimensions = {
        "tool_family": lambda item: item.tool_family,
        "mutation_family": lambda item: item.mutation_family,
        "hidden_guard_template": lambda item: item.guard_template_family,
        "field_binding_template": lambda item: item.field_binding_family,
        "version_lineage": lambda item: item.version_lineage,
        "implementation_ast_shape": lambda item: item.ast_shape_family,
        "public_case_id": lambda item: item.public_case_id,
        "input_schema_content": lambda item: canonical_digest(
            schema_shape(item, "input")
        ),
        "state_schema_content": lambda item: canonical_digest(
            schema_shape(item, "state")
        ),
        "guard_template_content": lambda item: canonical_digest(
            [shape(transition.guard.to_dict()) for transition in all_transitions(item)]
        ),
        "field_binding_template_content": lambda item: canonical_digest(
            binding_shape(item)
        ),
        "implementation_ast_content": lambda item: canonical_digest(transitions(item)),
        "implementation_ast_normalized_shape": lambda item: canonical_digest(
            shape(transitions(item))
        ),
        "symbolic_contract_content": lambda item: canonical_digest(
            {
                key: value
                for key, value in item.reference_contract.to_dict().items()
                if key in {"clauses", "state_updates", "unsupported_regions"}
            }
        ),
        "threshold_template_content": lambda item: (
            canonical_digest(constants(item, thresholds=True))
            if constants(item, thresholds=True)
            else None
        ),
        "constant_template_content": lambda item: canonical_digest(
            constants(item, thresholds=False)
        ),
    }
    collisions: dict[str, list[str]] = {}
    for name, getter in dimensions.items():
        seen: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            fingerprint = getter(case)
            if fingerprint is not None:
                seen[fingerprint].add(case.split)
        collisions[name] = sorted(
            key for key, splits in seen.items() if len(splits) > 1
        )
    total = sum(len(items) for items in collisions.values())
    return {
        "schema_version": 1,
        "audit_basis": "registered_ids_and_conservative_content_fingerprints",
        "cross_split_collision_count": total,
        "collisions": collisions,
        "split_case_counts": dict(
            sorted(Counter(item.split for item in cases).items())
        ),
    }


def collision_audit(cases: Sequence[HiddenSymbolicCase]) -> dict:
    from .template_isolation import TemplateDefinition, audit_templates

    legacy = _legacy_collision_audit(cases)
    definitions = tuple(
        TemplateDefinition(
            case.mutation_family,
            case.split,
            case.implementation.schema,
            case.implementation.transitions,
            case.reference_contract.clauses,
            case.reference_contract.state_updates,
            loop=case.implementation.loop,
            after_loop=case.implementation.after_loop,
        )
        for case in cases
    )
    independent = audit_templates(definitions)
    return {
        **legacy,
        **independent,
        "legacy_collision_count": legacy["cross_split_collision_count"],
        "cross_split_collision_count": legacy["cross_split_collision_count"]
        + independent["cross_split_collision_count"],
        "status": "passed"
        if legacy["cross_split_collision_count"] == 0
        and independent["status"] == "passed"
        else "failed",
    }


def iter_evaluator_manifest(cases: Sequence[HiddenSymbolicCase]) -> Iterator[dict]:
    for case in sorted(cases, key=lambda item: item.public_case_id):
        yield case.evaluator_manifest_row()
