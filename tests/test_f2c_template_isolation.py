"""只审计模板定义和 train/calibration，不生成 held-out case。"""

from dataclasses import replace
from functools import lru_cache

from keyed_gram.authsynth_symbolic_shared import (
    BooleanFormula,
    BoundedSchema,
    Conjunction,
    DomainField,
    GrammarLimits,
    Literal,
    SymbolicEffectClause,
    SymbolicEffectTemplate,
    VariableRef,
)
from keyed_gram.authsynth_symbolic_shared import (
    LiteralOperator as Op,
)
from keyed_gram.authsynth_symbolic_shared import (
    SymbolicTerm as T,
)
from keyed_gram.authsynth_symbolic_verifier.benchmark import (
    CONTROLS,
    MUTATIONS,
    SPLITS,
    collision_audit,
    generate_symbolic_smoke_cases,
)
from keyed_gram.authsynth_symbolic_verifier.hidden_ir import GuardedTransition
from keyed_gram.authsynth_symbolic_verifier.template_families import definition
from keyed_gram.authsynth_symbolic_verifier.template_isolation import (
    TemplateDefinition,
    audit_templates,
    semantic_fingerprint,
    structural_fingerprints,
)


def small(name="x", operator=Op.GE, value=1, *, two=False):
    fields = [DomainField(name, "input", "int", minimum=0, maximum=3)]
    literals = [Literal(VariableRef("input", name), operator, value)]
    if two:
        fields.append(DomainField("enabled", "input", "bool"))
        literals.append(Literal(VariableRef("input", "enabled"), Op.EQ, True))
    guard = BooleanFormula((Conjunction(tuple(literals)),))
    effect = SymbolicEffectTemplate(
        "event", 0, "immediate", "write", T.const("tenant"), T.var("input", name)
    )
    return TemplateDefinition(
        "fixture",
        "train",
        BoundedSchema(tuple(fields)),
        (GuardedTransition("branch", guard, (effect,)),),
        (SymbolicEffectClause(guard, effect),),
    )


def test_alpha_renaming_does_not_change_fingerprint():
    a, b = small(), replace(small("renamed"), family="new-name", split="calibration")
    assert structural_fingerprints(a) == structural_fingerprints(b)
    assert semantic_fingerprint(a) == semantic_fingerprint(b)


def test_constant_rename_does_not_fake_isolation():
    assert structural_fingerprints(small(value=1)) == structural_fingerprints(
        small(value=2)
    )


def test_commutative_guard_normalization():
    a = small(two=True)
    c = a.transitions[0].guard.disjuncts[0]
    guard = BooleanFormula((Conjunction(tuple(reversed(c.literals))),))
    b = replace(a, transitions=(replace(a.transitions[0], guard=guard),))
    assert structural_fingerprints(a) == structural_fingerprints(b)


def test_dead_noop_does_not_fake_isolation():
    a = small()
    dead = GuardedTransition(
        "dead", BooleanFormula.false(), (a.transitions[0].immediate_effects[0],)
    )
    b = replace(
        a,
        split="calibration",
        family="dead-wrapper",
        transitions=(*a.transitions, dead),
    )
    result = audit_templates((a, b))
    assert result["cross_split_semantic_collision_count"] > 0
    assert result["dead_guard_template_count"] == 1
    assert result["status"] == "failed"


def test_semantically_equivalent_guards_collide():
    a = small(operator=Op.GE, value=1)
    b = replace(
        small(operator=Op.NE, value=0), split="calibration", family="equivalent"
    )
    assert semantic_fingerprint(a) == semantic_fingerprint(b)
    assert audit_templates((a, b))["cross_split_semantic_collision_count"] == 1


def test_structurally_distinct_semantically_distinct_pass_at_program_level():
    a, b = small(), small(two=True)
    assert (
        structural_fingerprints(a)["implementation_ast"]
        != structural_fingerprints(b)["implementation_ast"]
    )
    assert semantic_fingerprint(a) != semantic_fingerprint(b)


@lru_cache(maxsize=1)
def current_audit():
    cases = generate_symbolic_smoke_cases(
        "prefreeze-new-templates-only",
        grammar=GrammarLimits(),
        replay_budget=16,
        solver_timeout_ms=3000,
    )
    assert {c.split for c in cases} == {"train", "calibration"}
    return collision_audit(cases)


def test_train_calibration_no_structural_collision():
    assert current_audit()["cross_split_structural_collision_count"] == 0
    assert current_audit()["legacy_collision_count"] == 0


def test_train_calibration_no_semantic_collision():
    assert current_audit()["cross_split_semantic_collision_count"] == 0


def test_all_split_template_families_disjoint():
    # 仅无 case/tool/version 身份的模板定义；不调用正式生成器或 evaluator。
    templates = tuple(definition(s, c) for s in SPLITS for c in (*MUTATIONS, *CONTROLS))
    result = audit_templates(templates)
    assert result["status"] == "passed", result
    assert result["tautological_separation_count"] == 0
    assert result["dead_guard_template_count"] == 0
