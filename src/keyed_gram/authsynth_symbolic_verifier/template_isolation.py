"""独立模板隔离审计：不生成 case，不调用 Analyzer、replay 或正式评估。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any

from keyed_gram.authsynth_symbolic_shared import (
    BoundedSchema,
    ConcreteAssignment,
    canonical_digest,
    canonical_json,
)

from .bounded_loop import BoundedLoop
from .hidden_ir import GuardedTransition


@dataclass(frozen=True)
class TemplateDefinition:
    """无 tool/case/version ID 的生成器定义；不是正式 hidden instance。"""

    family: str
    split: str
    schema: BoundedSchema
    transitions: tuple[GuardedTransition, ...]
    contract_clauses: tuple[Any, ...] = ()
    contract_updates: tuple[Any, ...] = ()
    loop_topology: tuple[Any, ...] = ()
    loop: BoundedLoop | None = None
    after_loop: tuple[GuardedTransition, ...] = ()


def _orders(schema: BoundedSchema):
    groups = defaultdict(list)
    for f in schema.fields:
        groups[(f.source, f.kind, len(f.values))].append(f)
    choices = [tuple(permutations(groups[k])) for k in sorted(groups)]
    count = 1
    for c in choices:
        count *= len(c)
    if count > 4096:
        raise ValueError("alpha-normalization budget 超界；不得报告隔离通过")
    for choice in product(*choices):
        yield tuple(f for group in choice for f in group)


def _assignments(fields):
    for values in product(*(f.values for f in fields)):
        yield ConcreteAssignment(
            tuple(
                (f.name, v)
                for f, v in zip(fields, values, strict=True)
                if f.source == "input"
            ),
            tuple(
                (f.name, v)
                for f, v in zip(fields, values, strict=True)
                if f.source == "state"
            ),
        )


def _canonical(value, refs, schema, key="", slots=None):
    if isinstance(value, dict):
        if set(value) == {"source", "name"}:
            return {"ref": refs[(value["source"], value["name"])]}
        if "dnf" in value:
            # 字面量中的恒真项和恒假项按其完整字段域消除。
            cubes = []
            for conjunction in value["dnf"]:
                literals = []
                dead = False
                for literal in conjunction["literals"]:
                    from keyed_gram.authsynth_symbolic_shared import (
                        Literal,
                        LiteralOperator,
                        VariableRef,
                    )

                    ref = literal["variable"]
                    f = schema.field(ref["source"], ref["name"])
                    item = Literal(
                        VariableRef(f.source, f.name),
                        LiteralOperator(literal["operator"]),
                        literal["value"],
                    )
                    truth = [
                        item.evaluate(
                            ConcreteAssignment(((f.name, v),), ())
                            if f.source == "input"
                            else ConcreteAssignment((), ((f.name, v),))
                        )
                        for v in f.values
                    ]
                    if not any(truth):
                        dead = True
                        break
                    if not all(truth):
                        literals.append(_canonical(literal, refs, schema, slots=slots))
                if not dead:
                    cubes.append(sorted(literals, key=canonical_json))
            if [] in cubes:
                return {"dnf": [[]]}
            return {"dnf": sorted(cubes, key=canonical_json)}
        return {
            k: _canonical(v, refs, schema, k, slots)
            for k, v in sorted(value.items())
            if k not in {"transition_id", "tool_id", "case_id", "version_digest"}
        }
    if key == "field":
        return refs[("state", value)]
    if key in {"slot", "parent_slot"}:
        return None if value is None else (slots or {}).get(value, "invalid-parent")
    if key in {"constant", "value"}:
        return {"constant_type": type(value).__name__}
    if isinstance(value, (tuple, list)):
        items = [_canonical(v, refs, schema, slots=slots) for v in value]
        return (
            sorted(items, key=canonical_json) if key in {"literals", "dnf"} else items
        )
    return value


def structural_fingerprints(template: TemplateDefinition) -> dict[str, str]:
    schema = template.schema
    assignments = tuple(_assignments(schema.fields))
    groups = (
        *template.transitions,
        *((template.loop.body) if template.loop else ()),
        *template.after_loop,
    )
    live = tuple(t for t in groups if any(t.guard.evaluate(a) for a in assignments))
    event_order = sorted({(e.sequence, e.slot) for t in live for e in t.all_effects()})
    slots = {slot: i for i, (_, slot) in enumerate(event_order)}
    candidates = defaultdict(list)
    for fields in _orders(schema):
        refs = {(f.source, f.name): i for i, f in enumerate(fields)}

        def norm(x, refs=refs):
            return _canonical(x, refs, schema, slots=slots)

        guards = [norm(t.guard.to_dict()) for t in live]
        effects = [[norm(e.to_dict()) for e in t.all_effects()] for t in live]
        updates = [[norm(u.to_dict()) for u in t.state_updates] for t in live]
        for source in ("input", "state"):
            # 数字/枚举取值和变量名不能成为隔离证据；保留类型及域基数。
            candidates[source + "_schema"].append(
                canonical_json(
                    [(f.kind, len(f.values)) for f in fields if f.source == source]
                )
            )
        candidates["guard_family"].append(
            canonical_json(sorted(guards, key=canonical_json))
        )
        candidates["binding_family"].append(canonical_json(effects))
        candidates["update_family"].append(canonical_json(updates))
        candidates["branch_topology"].append(
            canonical_json(
                sorted(
                    [
                        (g, len(e), len(u))
                        for g, e, u in zip(guards, effects, updates, strict=True)
                    ],
                    key=canonical_json,
                )
            )
        )
        candidates["implementation_ast"].append(
            canonical_json([norm(t.to_digest_dict()) for t in live])
        )
        if template.loop:
            candidates["implementation_ast"][-1] = canonical_json(
                {
                    "before": [norm(t.to_digest_dict()) for t in template.transitions],
                    "loop": norm(template.loop.to_dict()),
                    "after": [norm(t.to_digest_dict()) for t in template.after_loop],
                }
            )
            candidates["loop_topology"].append(
                canonical_json(norm(template.loop.to_dict()))
            )
        candidates["contract_shape"].append(
            canonical_json(
                {
                    "effects": [norm(c.to_dict()) for c in template.contract_clauses],
                    "updates": [norm(c.to_dict()) for c in template.contract_updates],
                }
            )
        )
        if template.loop_topology:
            candidates["loop_topology"].append(
                canonical_json(norm(template.loop_topology))
            )
        delayed = [
            [
                (
                    e.phase,
                    slots[e.slot],
                    None
                    if e.parent_slot is None
                    else slots.get(e.parent_slot, "invalid-parent"),
                )
                for e in t.all_effects()
            ]
            for t in live
        ]
        if any(e.phase == "delayed" for t in live for e in t.all_effects()):
            candidates["async_topology"].append(canonical_json(delayed))
    return {k: canonical_digest(min(v)) for k, v in candidates.items()}


def semantic_fingerprint(template: TemplateDefinition) -> str:
    """完整小域行为表的 alpha-canonical 摘要，仅用于隔离，不能给 Analyzer。"""
    if template.loop_topology and template.loop is None:
        raise ValueError("loop 模板需要实际循环解释器；不得用元数据伪造语义指纹")
    if template.schema.cardinality > 4096:
        raise ValueError("template semantic audit 域超过预注册上限")
    candidates = []
    if template.loop:
        from types import SimpleNamespace

        from .bounded_loop import execute_program

        program = SimpleNamespace(
            schema=template.schema,
            transitions=template.transitions,
            loop=template.loop,
            after_loop=template.after_loop,
            bounded_loop_max=template.loop.max_iterations,
            delayed_depth=2,
        )
    for fields in _orders(template.schema):
        rows = []
        for a in _assignments(fields):
            events, updates = [], {}
            for t in template.transitions:
                if not t.guard.evaluate(a):
                    continue
                events.extend(e.instantiate(a) for e in t.all_effects())
                for u in t.state_updates:
                    value = u.instantiate(a)
                    if u.field in updates and canonical_json(
                        updates[u.field]
                    ) != canonical_json(value):
                        raise ValueError("template 同一前状态更新冲突")
                    updates[u.field] = value
            if template.loop:
                result = execute_program(program, a)
                events, updates = result.events, dict(result.final_state)
                # 仅校验模板自身的解析参考定义；不运行 Analyzer 或正式实例。
                predicted = {
                    c.effect.instantiate(a)
                    for c in template.contract_clauses
                    if c.guard.evaluate(a)
                }
                predicted_state = dict(a.state)
                for clause in template.contract_updates:
                    if clause.guard.evaluate(a):
                        predicted_state[clause.update.field] = (
                            clause.update.instantiate(a)
                        )
                if set(events) != predicted or canonical_json(
                    updates
                ) != canonical_json(predicted_state):
                    raise ValueError("loop_template_reference_inconsistent")
            ordered = sorted(
                set(events), key=lambda e: (e.sequence, canonical_json(e.to_dict()))
            )
            slots = {e.slot: i for i, e in enumerate(ordered)}
            payload = [
                (
                    e.kind,
                    e.phase,
                    e.tenant,
                    e.resource,
                    e.destination,
                    e.amount,
                    None
                    if e.parent_slot is None
                    else slots.get(e.parent_slot, "missing-parent"),
                )
                for e in ordered
            ]
            final = [
                (f.kind, updates.get(f.name, a.value("state", f.name)))
                for f in fields
                if f.source == "state"
            ]
            rows.append((payload, final))
        candidates.append(canonical_json(rows))
    return canonical_digest(
        {
            "schema_types": sorted(
                (f.source, f.kind, len(f.values)) for f in template.schema.fields
            ),
            "behavior": min(candidates),
        }
    )


def hygiene(template: TemplateDefinition) -> dict[str, int]:
    assignments = tuple(_assignments(template.schema.fields))
    dead = tautological = no_effect = 0
    all_transitions = (
        *template.transitions,
        *((template.loop.body) if template.loop else ()),
        *template.after_loop,
    )
    for transition in all_transitions:
        truth = [transition.guard.evaluate(a) for a in assignments]
        dead += int(not any(truth))
        # 真正无条件工具步骤合法；非空条件恒真不能充当模板差异。
        tautological += int(all(truth) and transition.guard.literal_count > 0)
        no_effect += int(
            not transition.all_effects()
            and all(
                all(
                    canonical_json(u.instantiate(a))
                    == canonical_json(a.value("state", u.field))
                    for u in transition.state_updates
                )
                for a in assignments
                if transition.guard.evaluate(a)
            )
        )
    used = set()

    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"source", "name"}:
                used.add((value["source"], value["name"]))
            if "field" in value:
                used.add(("state", value["field"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    for transition in all_transitions:
        visit(transition.to_digest_dict())
    if template.loop:
        visit(template.loop.to_dict())
    unused = sum((f.source, f.name) not in used for f in template.schema.fields)
    return {
        "dead_guard_template_count": dead,
        "unreachable_branch_count": dead,
        "unused_schema_field_count": unused,
        "tautological_separation_count": tautological,
        "no_op_branch_count": no_effect,
    }


def audit_templates(templates: tuple[TemplateDefinition, ...]) -> dict[str, Any]:
    structural, semantic = defaultdict(list), defaultdict(list)
    problems = defaultdict(int)
    for template in templates:
        for k, v in structural_fingerprints(template).items():
            structural[(k, v)].append((template.split, template.family))
        semantic[semantic_fingerprint(template)].append(
            (template.split, template.family)
        )
        for k, v in hygiene(template).items():
            problems[k] += v
    structure_groups = [
        {"dimension": k[0], "digest": k[1], "members": sorted(v)}
        for k, v in sorted(structural.items())
        if len({x[0] for x in v}) > 1
    ]
    semantic_groups = [
        {"digest": k, "members": sorted(v)}
        for k, v in sorted(semantic.items())
        if len({x[0] for x in v}) > 1
    ]
    count = len(structure_groups) + len(semantic_groups)
    return {
        "schema_version": 2,
        "audit_kind": "prefreeze_template_definitions_only",
        "structural_collision_groups": structure_groups,
        "semantic_collision_groups": semantic_groups,
        "cross_split_structural_collision_count": len(structure_groups),
        "cross_split_semantic_collision_count": len(semantic_groups),
        "cross_split_collision_count": count,
        **dict(problems),
        "status": "passed" if count == 0 and not any(problems.values()) else "failed",
        "formal_development_started": False,
        "formal_locked_started": False,
        "template_count": len(templates),
    }
