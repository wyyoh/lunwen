"""Evaluator 专属原子真值：完整有界域的声明/具体行为差分，而非 mutation 标签。"""

from itertools import product

from keyed_gram.authsynth_symbolic_shared import (
    ConcreteAssignment,
    PatchAtom,
    SymbolicEffectContract,
    canonical_json,
)


def expected_omission_atoms(case) -> frozenset[PatchAtom]:
    public = case.analyzer_input
    initial = (public.declared_contract, *public.static_hypotheses)
    declaration = SymbolicEffectContract(
        tool_id=public.tool_id,
        implementation_version_digest=public.expected_version_digest,
        input_schema_digest=public.declared_contract.input_schema_digest,
        state_schema_digest=public.declared_contract.state_schema_digest,
        clauses=tuple(c for contract in initial for c in contract.clauses),
        state_updates=tuple(c for contract in initial for c in contract.state_updates),
    )
    signatures = {c.effect.signature for c in declaration.clauses}
    result = set()
    if declaration.implementation_version_digest != case.implementation.version_digest:
        result.add(PatchAtom("version_invalidation", "tool", public.tool_id, "version"))
    fields = public.schema.fields
    # 只在 evaluator 中枚举以产生评价标签；不会向 Analyzer 提供域输出表。
    for values in product(*(f.values for f in fields)):
        pairs = tuple(zip(fields, values, strict=True))
        assignment = ConcreteAssignment(
            tuple((f.name, v) for f, v in pairs if f.source == "input"),
            tuple((f.name, v) for f, v in pairs if f.source == "state"),
        )
        declared_events, declared_updates = declaration.predict(assignment)
        actual_events, actual_updates = case.implementation.execute(assignment)
        declared_active = {e.signature for e in declared_events}
        actual_active = {e.signature for e in actual_events}
        for signature in actual_active - declared_active:
            known = signature in signatures
            result.add(
                PatchAtom.effect(
                    "guard_patch" if known else "effect_patch",
                    signature,
                    "activation" if known else "presence",
                )
            )
        for signature in declared_active - actual_active:
            result.add(PatchAtom.effect("guard_patch", signature, "activation"))
        for signature in declared_active & actual_active:
            expected = [e for e in actual_events if e.signature == signature]
            declared = [e for e in declared_events if e.signature == signature]
            components = ("tenant", "resource", "destination", "amount")
            differences = [
                field
                for field in components
                if {canonical_json(getattr(e, field)) for e in expected}
                != {canonical_json(getattr(e, field)) for e in declared}
            ]
            for field in differences:
                result.add(PatchAtom.effect("field_binding_patch", signature, field))
            if not differences and (
                {canonical_json([getattr(e, f) for f in components]) for e in expected}
                != {
                    canonical_json([getattr(e, f) for f in components])
                    for e in declared
                }
            ):
                result.add(
                    PatchAtom.effect("field_binding_patch", signature, "joint_binding")
                )
        for field in public.schema.state_fields:
            original = assignment.value("state", field.name)
            if canonical_json(
                declared_updates.get(field.name, original)
            ) != canonical_json(actual_updates.get(field.name, original)):
                result.add(
                    PatchAtom("state_update_patch", "state", field.name, "final_value")
                )
    return frozenset(result)
