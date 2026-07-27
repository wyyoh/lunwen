from __future__ import annotations

import pytest

from keyed_gram.authcap_atoms import (
    AtomizedAuthority,
    AtomizedAuthorityError,
    AuthorityAtom,
    CanonicalConstraint,
    constraint_is_at_least_as_restrictive,
    verify_authority_non_amplification,
)


def atom(
    *,
    tenant: str = "tenant-a",
    resource: str = "resource-a",
    action: str = "read",
    relation: str = "relation-a",
    purpose: str = "research",
    constraints: tuple[CanonicalConstraint, ...] = (),
) -> AuthorityAtom:
    return AuthorityAtom(
        "subject-a",
        tenant,
        resource,
        action,
        relation,
        purpose,
        constraints,
    )


def authority(*atoms: AuthorityAtom) -> AtomizedAuthority:
    return AtomizedAuthority(frozenset(atoms))


def test_canonicalization_deduplicates_atoms_and_digest_is_deterministic():
    first = authority(atom(), atom(resource="resource-b"))
    second = authority(atom(resource="resource-b"), atom(), atom())
    assert first == second
    assert first.canonical_digest == second.canonical_digest
    assert first.canonical()["atoms"][0]["resource_id"] == "resource-a"


def test_subset_and_strict_subset_use_complete_atoms():
    maximum = authority(atom(), atom(resource="resource-b"))
    candidate = authority(atom())
    assert candidate.is_subset_of(maximum)
    assert candidate.is_strict_subset_of(maximum)
    assert not maximum.is_subset_of(candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tenant", "tenant-b"),
        ("resource", "resource-b"),
        ("action", "write"),
        ("relation", "relation-b"),
        ("purpose", "billing"),
    ),
)
def test_cross_scope_is_not_subset(field: str, value: str):
    assert not authority(atom(**{field: value})).is_subset_of(authority(atom()))


def test_constraint_partial_order_ttl_and_count():
    ttl_short = CanonicalConstraint("expires", "not_after", "2026-01-01T00:00:00Z")
    ttl_long = CanonicalConstraint("expires", "not_after", "2027-01-01T00:00:00Z")
    once = CanonicalConstraint("uses", "max_int", "1")
    many = CanonicalConstraint("uses", "max_int", "5")
    assert constraint_is_at_least_as_restrictive(ttl_short, ttl_long)
    assert constraint_is_at_least_as_restrictive(once, many)
    assert not constraint_is_at_least_as_restrictive(ttl_long, ttl_short)
    assert not constraint_is_at_least_as_restrictive(many, once)


def test_deleting_constraint_is_amplification_and_adding_is_restriction():
    maximum = authority(
        atom(constraints=(CanonicalConstraint("uses", "max_int", "5"),))
    )
    deleted = authority(atom())
    narrowed = authority(
        atom(
            constraints=(
                CanonicalConstraint("uses", "max_int", "2"),
                CanonicalConstraint("epoch", "equals", "2"),
            )
        )
    )
    assert not deleted.is_subset_of(maximum)
    assert narrowed.is_subset_of(maximum)


def test_intersection_union_difference_and_empty_authority():
    left = authority(atom(), atom(resource="resource-b"))
    right = authority(atom(resource="resource-b"), atom(resource="resource-c"))
    assert left.intersection(right) == authority(atom(resource="resource-b"))
    assert len(left.union(right).atoms) == 3
    assert left.difference(right) == authority(atom())
    assert AtomizedAuthority.empty().is_subset_of(left)
    assert AtomizedAuthority.empty().is_empty


def test_no_cartesian_product_false_inclusion():
    candidate = authority(
        atom(resource="resource-a", action="read"),
        atom(resource="resource-b", action="write"),
    )
    maximum = authority(
        atom(resource="resource-a", action="read"),
        atom(resource="resource-b", action="read"),
    )
    assert not candidate.is_subset_of(maximum)


def test_wildcard_and_duplicate_constraint_key_rejected():
    with pytest.raises(AtomizedAuthorityError):
        atom(resource="*")
    with pytest.raises(AtomizedAuthorityError):
        atom(
            constraints=(
                CanonicalConstraint("uses", "max_int", "1"),
                CanonicalConstraint("uses", "max_int", "2"),
            )
        )


def test_non_amplification_checker_requires_all_three_bounds():
    exact = authority(atom())
    over = authority(atom(action="write"))
    passed = verify_authority_non_amplification(exact, exact, exact, exact)
    failed = verify_authority_non_amplification(exact, exact, exact, over)
    assert passed.passed
    assert not failed.passed
    assert not failed.proposal_subset
