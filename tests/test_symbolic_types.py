from __future__ import annotations

import pytest

from keyed_gram.authsynth_symbolic_shared import (
    BoundedSchema,
    ConcreteAssignment,
    DomainField,
    StateChange,
    StructuredStateDiff,
    SymbolicSchemaError,
    canonical_digest,
)


def test_bounded_schema_is_canonical_and_finite() -> None:
    schema = BoundedSchema(
        (
            DomainField("flag", "state", "bool"),
            DomainField("mode", "input", "enum", enum_values=("member", "admin")),
            DomainField("amount", "input", "int", minimum=0, maximum=2),
        )
    )
    assert [(item.source, item.name) for item in schema.fields] == [
        ("input", "amount"),
        ("input", "mode"),
        ("state", "flag"),
    ]
    assert schema.cardinality == 12
    assert schema.digest == canonical_digest(schema.to_dict())


def test_assignment_rejects_duplicate_and_out_of_domain_values() -> None:
    with pytest.raises(SymbolicSchemaError, match="重复 input"):
        ConcreteAssignment((("x", True), ("x", False)), ())
    schema = BoundedSchema((DomainField("x", "input", "int", minimum=0, maximum=1),))
    with pytest.raises(SymbolicSchemaError, match="越界"):
        schema.validate_assignment(ConcreteAssignment((("x", 2),), ()))


def test_identifier_wildcards_and_unbounded_ints_are_rejected() -> None:
    with pytest.raises(SymbolicSchemaError, match="wildcard"):
        DomainField("bad*", "input", "bool")
    with pytest.raises(SymbolicSchemaError, match="有限闭区间"):
        DomainField("amount", "input", "int")


def test_structured_state_diff_is_canonical_not_only_a_digest() -> None:
    first = StateChange("record", "state", "z", False, True)
    second = StateChange("record", "state", "a", 0, 1)
    diff = StructuredStateDiff((first, second))
    assert [item.field for item in diff.changes] == ["a", "z"]
    assert diff.to_dict()["changes"][0]["before"] == 0
    assert len(diff.digest) == 64
