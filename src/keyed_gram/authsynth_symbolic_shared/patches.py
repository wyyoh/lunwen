"""公开、可定位的补丁原子；不包含 evaluator 标签或内部公式。"""

from dataclasses import dataclass

from .schema import canonical_digest, restricted_token


@dataclass(frozen=True, order=True)
class PatchAtom:
    patch_type: str
    target_kind: str
    target_id: str
    component: str

    def __post_init__(self) -> None:
        expected_kind = {
            "effect_patch": "effect",
            "guard_patch": "effect",
            "field_binding_patch": "effect",
            "state_update_patch": "state",
            "version_invalidation": "tool",
            "unsupported_region": "tool",
        }
        if expected_kind.get(self.patch_type) != self.target_kind:
            raise ValueError("patch atom 类型与定位对象不一致")
        restricted_token(self.target_id, "patch target")
        restricted_token(self.component, "patch component")
        allowed = {
            "effect_patch": {"presence"},
            "guard_patch": {"activation"},
            "field_binding_patch": {
                "tenant",
                "resource",
                "destination",
                "amount",
                "joint_binding",
            },
            "state_update_patch": {"final_value"},
            "version_invalidation": {"version"},
        }
        if (
            self.patch_type in allowed
            and self.component not in allowed[self.patch_type]
        ):
            raise ValueError("patch atom component 非法")

    def to_dict(self) -> dict[str, str]:
        return {
            "patch_type": self.patch_type,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "component": self.component,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    @classmethod
    def effect(cls, patch_type: str, signature: tuple, component: str) -> "PatchAtom":
        # signature 包含 slot/sequence/phase/kind/parent，不包含敏感 payload。
        return cls(patch_type, "effect", canonical_digest(signature), component)
