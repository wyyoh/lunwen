from __future__ import annotations

from dataclasses import fields

import pytest

from keyed_gram.stage_d2_contract import PublicSyntheticValue
from keyed_gram.stage_d21_contract import (
    AuthorizedGenerationPayload,
    D21Error,
    DeliveredGeneration,
    GenerationOutput,
    GenerationVariant,
    create_generation_payload,
)


def _value() -> PublicSyntheticValue:
    return PublicSyntheticValue(b"0123456789ABCDEF")


def _payload() -> AuthorizedGenerationPayload:
    return create_generation_payload(
        value=_value(),
        public_instruction="Return only this value.",
        variant=GenerationVariant.G1,
        context_id="d21-context-" + "a" * 32,
    )


def test_generation_payload_excludes_authorization_metadata() -> None:
    names = {item.name for item in fields(AuthorizedGenerationPayload)}
    assert names == {
        "value",
        "public_instruction",
        "variant",
        "context_id",
        "_seal",
    }
    assert not {
        "subject_id",
        "entity_id",
        "relation_id",
        "record_id",
        "capability_token",
        "key_id",
    } & names


def test_generation_payload_cannot_be_caller_constructed() -> None:
    with pytest.raises(TypeError):
        AuthorizedGenerationPayload(
            value=_value(),
            public_instruction="copy",
            variant=GenerationVariant.G1,
            context_id="d21-context-" + "b" * 32,
            _seal=object(),
        )


def test_generation_payload_rejects_invalid_context() -> None:
    with pytest.raises(D21Error):
        create_generation_payload(
            value=_value(),
            public_instruction="copy",
            variant=GenerationVariant.G1,
            context_id="reused-or-invalid",
        )


def test_generation_payload_rejects_oversized_instruction() -> None:
    with pytest.raises(D21Error):
        create_generation_payload(
            value=_value(),
            public_instruction="x" * 2049,
            variant=GenerationVariant.G1,
            context_id="d21-context-" + "c" * 32,
        )


def test_payload_repr_hides_value_and_instruction() -> None:
    representation = repr(_payload())
    assert "0123456789ABCDEF" not in representation
    assert "Return only this value." not in representation


def test_output_and_delivery_repr_hide_text() -> None:
    output = GenerationOutput.from_text(
        "0123456789ABCDEF",
        variant=GenerationVariant.G1,
        context_id="d21-context-" + "d" * 32,
    )
    delivered = DeliveredGeneration(
        text="0123456789ABCDEF",
        output_digest=output.output_digest,
        record_id="record-test",
        variant=GenerationVariant.G1,
        exact_delivery=True,
    )
    assert "0123456789ABCDEF" not in repr(output)
    assert "0123456789ABCDEF" not in repr(delivered)


def test_delivery_rejects_inconsistent_digest() -> None:
    with pytest.raises(ValueError):
        DeliveredGeneration(
            text="0123456789ABCDEF",
            output_digest="0" * 64,
            record_id="record-test",
            variant=GenerationVariant.G1,
            exact_delivery=True,
        )
