"""预冻结 meta-family 定义；不创建正式 case、版本或 locked namespace。"""

from dataclasses import replace

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    Conjunction,
    DomainField,
    Literal,
    LiteralOperator,
    SymbolicEffectClause,
    SymbolicEffectContract,
    SymbolicEffectTemplate,
    SymbolicStateUpdate,
    SymbolicStateUpdateClause,
    VariableRef,
    canonical_digest,
)
from keyed_gram.authsynth_symbolic_shared import (
    SymbolicTerm as T,
)

from .bounded_loop import BoundedLoop
from .hidden_ir import GuardedTransition
from .template_isolation import TemplateDefinition


def lit(source, name, value, op=LiteralOperator.EQ):
    return Literal(VariableRef(source, name), op, value)


def dnf(*cubes):
    return BooleanFormula(tuple(Conjunction(tuple(c)) for c in cubes))


def conjunction(guard, item):
    return dnf(*(tuple(c.literals) + (item,) for c in guard.disjuncts))


def definition(split: str, category: str) -> TemplateDefinition:
    """四种不同的业务状态模型；没有用无效字段、恒真 guard 做分区装饰。"""
    fields = [
        DomainField("tenant", "input", "enum", ("tenant-a", "tenant-b")),
        DomainField("resource", "input", "enum", ("r0", "r1", "r2", "r3")),
        DomainField("destination", "input", "enum", ("alias", "external", "internal")),
        DomainField("amount", "input", "int", minimum=0, maximum=3),
    ]
    tenant, resource, destination = (
        T.var("input", n) for n in ("tenant", "resource", "destination")
    )
    amount = T.var("input", "amount")
    if split == "train":
        fields += [
            DomainField("mode", "input", "enum", ("admin", "member")),
            DomainField("counter", "state", "int", minimum=0, maximum=3),
        ]
        guard = dnf((lit("input", "mode", "admin"),))
        output_amount = T.var("state", "counter")
        updates = (SymbolicStateUpdate("counter", amount),)
    elif split == "calibration":
        fields += [
            DomainField("approve", "input", "bool"),
            DomainField("flag", "state", "bool"),
            DomainField("delivery", "state", "enum", ("hold", "ready")),
        ]
        guard = dnf((lit("input", "approve", True), lit("state", "flag", True)))
        resource = T.ite(
            lit("state", "delivery", "ready"), resource, T.const("staging")
        )
        output_amount = amount
        updates = (
            SymbolicStateUpdate(
                "flag",
                T.ite(
                    lit("input", "amount", 1, LiteralOperator.GE),
                    T.const(False),
                    T.const(True),
                ),
            ),
            SymbolicStateUpdate(
                "delivery",
                T.ite(
                    lit("input", "amount", 1, LiteralOperator.LE),
                    T.const("hold"),
                    T.const("ready"),
                ),
            ),
        )
    elif split == "development":
        fields += [
            DomainField("approve", "input", "bool"),
            DomainField("override", "input", "bool"),
            DomainField("phase", "state", "enum", ("pending", "done")),
            DomainField("quota", "state", "int", minimum=0, maximum=1),
        ]
        guard = dnf(
            (lit("input", "amount", 2, LiteralOperator.GE), lit("state", "quota", 1)),
            (lit("input", "override", True), lit("input", "approve", False)),
        )
        tenant = T.ite(lit("state", "phase", "done"), T.const("archive"), tenant)
        resource = T.ite(
            lit("input", "amount", 1, LiteralOperator.LE), T.const("staging"), resource
        )
        output_amount = amount
        updates = (
            SymbolicStateUpdate(
                "quota",
                T.ite(
                    lit("input", "amount", 3, LiteralOperator.GE),
                    T.const(1),
                    T.const(0),
                ),
            ),
            SymbolicStateUpdate(
                "phase",
                T.ite(
                    lit("input", "override", True), T.const("done"), T.const("pending")
                ),
            ),
        )
    elif split == "locked_test":
        # 更换 resource 域大小控制总域上限；隔离依据是状态/绑定/DNF 拓扑，不是此常量。
        fields[1] = DomainField("resource", "input", "enum", ("r0", "r1"))
        fields += [
            DomainField(n, "input", "bool") for n in ("approve", "override", "review")
        ]
        fields += [
            DomainField("flag", "state", "bool"),
            DomainField("counter", "state", "int", minimum=0, maximum=1),
            DomainField("quota", "state", "int", minimum=0, maximum=1),
        ]
        guard = dnf(
            (lit("input", "amount", 2, LiteralOperator.GE), lit("state", "counter", 1)),
            (lit("input", "override", True), lit("state", "flag", True)),
            (lit("input", "review", True), lit("state", "quota", 1)),
        )
        tenant = T.ite(lit("input", "approve", True), T.const("archive"), tenant)
        resource = T.ite(lit("input", "review", True), resource, T.const("staging"))
        destination = T.ite(
            lit("state", "flag", True), T.const("external"), destination
        )
        output_amount = amount
        updates = (
            SymbolicStateUpdate(
                "counter", T.ite(lit("input", "approve", True), T.const(1), T.const(0))
            ),
            SymbolicStateUpdate(
                "quota",
                T.ite(
                    lit("input", "amount", 0, LiteralOperator.LE),
                    T.const(0),
                    T.const(1),
                ),
            ),
            SymbolicStateUpdate("flag", T.var("input", "override")),
        )
    else:
        raise ValueError("未知 template split")
    schema = BoundedSchema(tuple(fields))
    if not 256 <= schema.cardinality <= 4096:
        raise ValueError("模板域超界")
    main = SymbolicEffectTemplate(
        "read-main",
        0,
        "immediate",
        "read",
        tenant,
        resource,
        destination,
        output_amount,
    )
    base = GuardedTransition("base", guard, (main,), state_updates=updates)
    extra_guard = conjunction(guard, lit("input", "amount", 2, LiteralOperator.GE))
    extra = replace(main, slot="extra", sequence=1, kind="send", amount=amount)
    extra_transitions = []
    if category not in {"exact_declared_control", "conservative_control"}:
        if category == "alias_redirect_effect":
            extra = replace(
                extra,
                destination=T.ite(
                    lit("input", "destination", "alias"),
                    T.const("external"),
                    T.var("input", "destination"),
                ),
            )
        if category == "implementation_drift":
            extra = replace(extra, kind="delete")
        if category == "state_dependent_effect":
            extra = replace(extra, kind="execute")
        if category == "threshold_effect":
            extra = replace(extra, kind="charge")
        if category == "bounded_delayed_effect":
            extra = replace(extra, phase="delayed", parent_slot=main.slot)
            # 不同业务协议的真实可观测异步步骤：发送、回执、回调与提交。
            # 每个步骤都会改变事件轨迹，不用无效 guard 或未执行分支区分 split。
            delayed = [extra]
            if split == "calibration":
                delayed.append(
                    replace(
                        extra,
                        slot="receipt",
                        sequence=2,
                        kind="write",
                        destination=T.const("internal"),
                        parent_slot=extra.slot,
                    )
                )
            elif split == "development":
                delayed.extend(
                    (
                        replace(
                            extra,
                            slot="callback",
                            sequence=2,
                            kind="callback",
                            parent_slot=extra.slot,
                        ),
                        replace(
                            extra,
                            slot="receipt",
                            sequence=3,
                            kind="write",
                            destination=T.const("internal"),
                            parent_slot=main.slot,
                        ),
                    )
                )
            elif split == "locked_test":
                delayed.extend(
                    (
                        replace(
                            extra,
                            slot="callback",
                            sequence=2,
                            kind="callback",
                            parent_slot=main.slot,
                        ),
                        replace(
                            extra,
                            slot="receipt",
                            sequence=3,
                            kind="write",
                            destination=T.const("internal"),
                            parent_slot=main.slot,
                        ),
                        replace(
                            extra,
                            slot="commit",
                            sequence=4,
                            kind="commit",
                            destination=T.const("internal"),
                            parent_slot=extra.slot,
                        ),
                    )
                )
            extra_transitions.append(
                GuardedTransition("extra", extra_guard, delayed_effects=tuple(delayed))
            )
        else:
            extra_transitions.append(GuardedTransition("extra", extra_guard, (extra,)))
        if category == "multi_branch_effect":
            extra_transitions.append(
                GuardedTransition(
                    "alternative",
                    conjunction(guard, lit("input", "amount", 1, LiteralOperator.LE)),
                    (replace(extra, slot="alternative", sequence=2, kind="export"),),
                )
            )
    transitions = (base, *extra_transitions)
    if category == "bounded_delayed_effect" and split in {"development", "locked_test"}:
        return loop_definition(split, category, schema, base, main, updates)
    clauses = tuple(
        SymbolicEffectClause(t.guard, e) for t in transitions for e in t.all_effects()
    )
    state_clauses = tuple(
        SymbolicStateUpdateClause(t.guard, u)
        for t in transitions
        for u in t.state_updates
    )
    return TemplateDefinition(
        f"{split}/{category}", split, schema, transitions, clauses, state_clauses
    )


def loop_definition(split, category, schema, base, main, updates):
    """审计的是循环模板定义；参考契约按逐轮递推解析构造，不是赋值查表。"""
    before = replace(base, state_updates=())
    body_read = replace(main, slot="queue", sequence=0, kind="queue")
    send = replace(
        body_read,
        slot="send",
        sequence=1,
        phase="delayed",
        kind="send",
        parent_slot="queue",
    )
    if split == "development":
        bound = 2
        loop_guard = dnf((lit("state", "quota", 1),))
        second = conjunction(loop_guard, lit("input", "amount", 3, LiteralOperator.GE))
        guards = (loop_guard, second)
        effects = (body_read, send)
    else:
        bound = 3
        loop_guard = dnf((lit("state", "flag", True), lit("state", "counter", 1)))
        second = conjunction(
            conjunction(loop_guard, lit("input", "approve", True)),
            lit("input", "override", True),
        )
        guards = (loop_guard, second, second)
        # active => flag=True，destination 的 ITE 在三轮均精确化为 external。
        body_read = replace(body_read, destination=T.const("external"))
        send = replace(send, destination=T.const("external"))
        receipt = replace(
            send,
            slot="receipt",
            sequence=2,
            kind="write",
            destination=T.const("internal"),
            parent_slot="queue",
        )
        effects = (body_read, send, receipt)
    body = GuardedTransition("loop-body", dnf(()), (effects[0],), effects[1:], updates)
    loop = BoundedLoop(loop_guard, (body,), bound)
    clauses = [
        SymbolicEffectClause(before.guard, replace(main, slot="pre-" + main.slot))
    ]
    for i, g in enumerate(guards):
        for effect in effects:
            emitted = effect
            if split == "development" and i:
                # 第 1 轮 phase 已被 body 设置，此处使用新的 state，而非 s0。
                emitted = replace(
                    emitted,
                    tenant=T.ite(
                        lit("input", "override", True),
                        T.const("archive"),
                        T.var("input", "tenant"),
                    ),
                )
            emitted = replace(
                emitted,
                slot=f"loop-{i}-" + effect.slot,
                sequence=1 + i * len(effects) + effect.sequence,
                parent_slot=None
                if effect.parent_slot is None
                else f"loop-{i}-" + effect.parent_slot,
            )
            clauses.append(SymbolicEffectClause(g, emitted))
    return TemplateDefinition(
        f"{split}/{category}",
        split,
        schema,
        (before,),
        tuple(clauses),
        tuple(SymbolicStateUpdateClause(loop_guard, u) for u in updates),
        loop=loop,
    )


def contract(template, tool_id, version):
    return SymbolicEffectContract(
        tool_id,
        version,
        canonical_digest(
            {"fields": [f.to_dict() for f in template.schema.input_fields]}
        ),
        canonical_digest(
            {"fields": [f.to_dict() for f in template.schema.state_fields]}
        ),
        template.contract_clauses,
        template.contract_updates,
    )


def declared(template, reference, category, version):
    if category == "exact_declared_control":
        return reference.with_version(version)
    base = tuple(c for c in reference.clauses if c.effect.slot == "read-main")
    updates = reference.state_updates
    if category == "conservative_control":
        base += (
            SymbolicEffectClause(
                template.transitions[0].guard,
                replace(
                    base[0].effect,
                    slot="spurious",
                    sequence=1,
                    kind="send",
                    destination=T.const("external"),
                ),
            ),
        )
    elif category in {"parameter_role_alias", "alias_redirect_effect"}:
        base += tuple(
            replace(c, effect=replace(c.effect, destination=T.const("internal")))
            for c in reference.clauses
            if c.effect.slot != "read-main"
        )
    elif category == "threshold_effect":
        base += tuple(
            replace(
                c,
                guard=conjunction(
                    c.guard, lit("input", "amount", 3, LiteralOperator.GE)
                ),
            )
            for c in reference.clauses
            if c.effect.slot != "read-main"
        )
    elif category == "state_dependent_effect":
        updates = ()
    return replace(reference.with_version(version), clauses=base, state_updates=updates)
