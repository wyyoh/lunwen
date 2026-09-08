"""只使用 train/calibration fixture，不物化 development/locked。"""

from dataclasses import replace

from f2c_symbolic_helpers import case_for, smoke_cases

from keyed_gram.authsynth_symbolic_verifier.benchmark import collision_audit


def test_content_collisions_cannot_be_hidden_by_unique_family_labels():
    original = case_for("hidden_guarded_effect")
    renamed = replace(
        original,
        split="calibration",
        tool_family="unique-tool",
        mutation_family="unique-mutation",
        guard_template_family="unique-guard",
        field_binding_family="unique-binding",
        version_lineage="unique-version",
        ast_shape_family="unique-shape",
        analyzer_input=replace(original.analyzer_input, public_case_id="unique-case"),
    )
    audit = collision_audit((original, renamed))
    assert not audit["collisions"]["tool_family"]
    assert not audit["collisions"]["implementation_ast_shape"]
    assert audit["collisions"]["implementation_ast_content"]
    assert audit["collisions"]["input_schema_content"]
    assert audit["cross_split_collision_count"] > 0


def test_prefreeze_fixture_currently_fails_content_isolation():
    fixtures = smoke_cases()
    assert {case.split for case in fixtures} == {"train", "calibration"}
    audit = collision_audit(fixtures)
    assert audit["cross_split_collision_count"] > 0
    assert audit["collisions"]["guard_template_content"]
    assert audit["collisions"]["field_binding_template_content"]
    assert audit["collisions"]["state_schema_content"]
    assert audit["collisions"]["symbolic_contract_content"]


def test_within_split_reuse_is_not_cross_split_collision():
    case = case_for("hidden_guarded_effect")
    assert collision_audit((case, case))["cross_split_collision_count"] == 0
