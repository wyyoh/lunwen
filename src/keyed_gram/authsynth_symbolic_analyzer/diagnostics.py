"""仅从候选契约与一个实际 replay 定位差异，不接触参考契约。"""

from collections import defaultdict

from keyed_gram.authsynth_symbolic_shared import (
    PatchAtom,
    SymbolicEffectContract,
    SymbolicReplayResult,
    canonical_json,
)

FIELDS = ("tenant", "resource", "destination", "amount")


def diagnose_replay(
    candidate: SymbolicEffectContract, observed: SymbolicReplayResult
) -> tuple[PatchAtom, ...]:
    predicted, updates = candidate.predict(observed.assignment)
    groups = []
    for events in (predicted, observed.events):
        index = defaultdict(list)
        for event in events:
            index[event.signature].append(event)
        groups.append(index)
    before, after = groups
    available = {clause.effect.signature for clause in candidate.clauses}
    atoms = set()
    for signature in set(before) | set(after):
        if not before[signature]:
            kind = "guard_patch" if signature in available else "effect_patch"
            component = "activation" if kind == "guard_patch" else "presence"
            atoms.add(PatchAtom.effect(kind, signature, component))
        elif not after[signature]:
            atoms.add(PatchAtom.effect("guard_patch", signature, "activation"))
        else:
            changed_fields = []
            for field in FIELDS:
                left = {canonical_json(getattr(e, field)) for e in before[signature]}
                right = {canonical_json(getattr(e, field)) for e in after[signature]}
                if left != right:
                    changed_fields.append(field)
                    atoms.add(PatchAtom.effect("field_binding_patch", signature, field))
            # 各字段边际集合相同仍可能存在不同的联合绑定，不得遗漏关联错误。
            if not changed_fields:
                left = {
                    canonical_json([getattr(e, f) for f in FIELDS])
                    for e in before[signature]
                }
                right = {
                    canonical_json([getattr(e, f) for f in FIELDS])
                    for e in after[signature]
                }
                if left != right:
                    atoms.add(
                        PatchAtom.effect(
                            "field_binding_patch", signature, "joint_binding"
                        )
                    )
    actual_state = dict(observed.assignment.state)
    for change in observed.state_diff.changes:
        actual_state[change.field] = change.after
    predicted_state = dict(observed.assignment.state)
    predicted_state.update(updates)
    for field in sorted(set(actual_state) | set(predicted_state)):
        if canonical_json(actual_state.get(field)) != canonical_json(
            predicted_state.get(field)
        ):
            atoms.add(PatchAtom("state_update_patch", "state", field, "final_value"))
    return tuple(sorted(atoms))
