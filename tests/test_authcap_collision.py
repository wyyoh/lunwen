from __future__ import annotations

from keyed_gram.authcap_benchmark import generate_authzroutebench
from keyed_gram.authcap_collision import audit_split_collisions


def test_all_preregistered_split_units_are_isolated() -> None:
    result = audit_split_collisions(generate_authzroutebench())
    assert result["cross_split_collision_count"] == 0
    assert all(
        unit["collision_count"] == 0 for unit in result["units"].values()
    )


def test_family_collision_is_detected() -> None:
    splits = generate_authzroutebench()
    splits["locked_test"][0]["family_id"] = splits["train"][0]["family_id"]
    result = audit_split_collisions(splits)
    assert result["units"]["attack_family"]["collision_count"] == 1


def test_policy_template_collision_is_detected() -> None:
    splits = generate_authzroutebench()
    splits["locked_test"][0]["metadata"]["policy_template_id"] = (
        splits["train"][0]["metadata"]["policy_template_id"]
    )
    result = audit_split_collisions(splits)
    assert result["units"]["policy_template"]["collision_count"] == 1
