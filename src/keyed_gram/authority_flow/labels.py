"""数据标签与影响来源。

数据标签描述值的机密性和完整性；它们不构成执行 effect 的 authority。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DataLabelError(ValueError):
    """数据标签或显式 endorsement 不合法。"""


class Confidentiality(IntEnum):
    """数值越大，机密性约束越严格。"""

    PUBLIC = 0
    INTERNAL = 1
    RESTRICTED = 2


class Integrity(IntEnum):
    """数值越大，来源完整性越高。"""

    UNTRUSTED = 0
    ENDORSED = 1
    TRUSTED = 2


@dataclass(frozen=True, order=True)
class InfluenceOrigin:
    """普通数据影响的来源，不携带行为权限。"""

    source_kind: str
    source_id: str
    attacker_controlled: bool

    def canonical(self) -> dict[str, object]:
        return {
            "attacker_controlled": self.attacker_controlled,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class DataLabel:
    confidentiality: Confidentiality
    integrity: Integrity
    influences: tuple[InfluenceOrigin, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "influences", tuple(sorted(set(self.influences))))

    def join(self, other: DataLabel) -> DataLabel:
        """组合数据时取更严格机密性和更低完整性。"""

        return DataLabel(
            confidentiality=max(self.confidentiality, other.confidentiality),
            integrity=min(self.integrity, other.integrity),
            influences=tuple(sorted(set(self.influences) | set(other.influences))),
        )

    def meet(self, other: DataLabel) -> DataLabel:
        return DataLabel(
            confidentiality=min(self.confidentiality, other.confidentiality),
            integrity=max(self.integrity, other.integrity),
            influences=tuple(sorted(set(self.influences) & set(other.influences))),
        )

    def canonical(self) -> dict[str, object]:
        return {
            "confidentiality": self.confidentiality.name.lower(),
            "influences": [item.canonical() for item in self.influences],
            "integrity": self.integrity.name.lower(),
        }


@dataclass(frozen=True)
class TrustedEndorsement:
    """显式可信 elevation 的最小见证。

    该对象只允许提高数据完整性，不签发或复制任何 authority resource。
    """

    endorsement_id: str
    issuer: str
    maximum_integrity: Integrity
    attacker_controlled: bool = False


def endorse(
    label: DataLabel,
    permit: TrustedEndorsement,
    target: Integrity,
) -> DataLabel:
    """显式提高数据完整性，但保留 influence provenance。"""

    if permit.attacker_controlled:
        raise DataLabelError("攻击者控制的 endorsement 不能提升完整性")
    if target > permit.maximum_integrity:
        raise DataLabelError("endorsement 超过可信 permit")
    if target < label.integrity:
        raise DataLabelError("endorse 不能降低完整性")
    return DataLabel(
        confidentiality=label.confidentiality,
        integrity=target,
        influences=label.influences,
    )
