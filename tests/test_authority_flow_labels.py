from __future__ import annotations

import pytest

from keyed_gram.authority_flow.labels import (
    Confidentiality,
    DataLabel,
    DataLabelError,
    InfluenceOrigin,
    Integrity,
    TrustedEndorsement,
    endorse,
)


def test_data_join_propagates_restrictive_label_and_influence() -> None:
    trusted = DataLabel(
        Confidentiality.PUBLIC,
        Integrity.TRUSTED,
        (InfluenceOrigin("user", "trusted-user", False),),
    )
    injected = DataLabel(
        Confidentiality.RESTRICTED,
        Integrity.UNTRUSTED,
        (InfluenceOrigin("web", "attacker-page", True),),
    )
    joined = trusted.join(injected)
    assert joined.confidentiality is Confidentiality.RESTRICTED
    assert joined.integrity is Integrity.UNTRUSTED
    assert len(joined.influences) == 2


def test_endorsement_preserves_influence_and_does_not_create_authority() -> None:
    label = DataLabel(
        Confidentiality.INTERNAL,
        Integrity.UNTRUSTED,
        (InfluenceOrigin("tool_output", "echo-1", True),),
    )
    permit = TrustedEndorsement("endorse-1", "policy-authority", Integrity.ENDORSED)
    result = endorse(label, permit, Integrity.ENDORSED)
    assert result.integrity is Integrity.ENDORSED
    assert result.influences == label.influences
    assert "authority" not in result.canonical()


def test_attacker_controlled_endorsement_is_rejected() -> None:
    label = DataLabel(Confidentiality.PUBLIC, Integrity.UNTRUSTED)
    permit = TrustedEndorsement(
        "endorse-bad",
        "attacker",
        Integrity.TRUSTED,
        attacker_controlled=True,
    )
    with pytest.raises(DataLabelError):
        endorse(label, permit, Integrity.TRUSTED)
