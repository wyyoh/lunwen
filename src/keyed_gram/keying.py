from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


Position = tuple[int, int]
Swap = tuple[Position, Position]
_UP_RE = re.compile(r"^blocks\.(\d+)\.moe\.experts\.(\d+)\.c_fc\.weight$")


@dataclass(frozen=True)
class AuxPermutationKey:
    schema_version: int
    capability_label: str
    expert_index: int
    num_layers: int
    aux_width: int
    group_size: int
    seed: int
    swap_fraction: float
    swaps: tuple[Swap, ...]
    original_aux_sha256: str

    @property
    def num_groups(self) -> int:
        return self.num_layers * (self.aux_width // self.group_size)

    def position_mapping(self) -> dict[Position, Position]:
        mapping: dict[Position, Position] = {}
        for left, right in self.swaps:
            mapping[left] = right
            mapping[right] = left
        return mapping

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "capability_label": self.capability_label,
            "expert_index": self.expert_index,
            "num_layers": self.num_layers,
            "aux_width": self.aux_width,
            "group_size": self.group_size,
            "seed": self.seed,
            "swap_fraction": self.swap_fraction,
            "aux_group_swaps": [[list(left), list(right)] for left, right in self.swaps],
            "original_aux_sha256": self.original_aux_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "AuxPermutationKey":
        swaps: list[Swap] = []
        for raw_left, raw_right in value.get("aux_group_swaps", []):
            swaps.append(((int(raw_left[0]), int(raw_left[1])), (int(raw_right[0]), int(raw_right[1]))))
        return cls(
            schema_version=int(value["schema_version"]),
            capability_label=str(value["capability_label"]),
            expert_index=int(value["expert_index"]),
            num_layers=int(value["num_layers"]),
            aux_width=int(value["aux_width"]),
            group_size=int(value["group_size"]),
            seed=int(value["seed"]),
            swap_fraction=float(value.get("swap_fraction", 1.0)),
            swaps=tuple(swaps),
            original_aux_sha256=str(value["original_aux_sha256"]),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AuxPermutationKey":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def clone_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def hash_state_subset(state_dict: Mapping[str, Tensor], predicate) -> str:
    digest = hashlib.sha256()
    for name in sorted(name for name in state_dict if predicate(name)):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def hash_auxiliary(state_dict: Mapping[str, Tensor], expert_index: int = 1) -> str:
    marker = f".moe.experts.{expert_index}."
    names = [name for name in state_dict if marker in name or name == "aux_scales"]
    if not names:
        raise ValueError(f"no tensors found for auxiliary expert {expert_index}")
    return hash_state_subset(
        state_dict, lambda name: marker in name or name == "aux_scales"
    )


def hash_non_auxiliary(state_dict: Mapping[str, Tensor], expert_index: int = 1) -> str:
    marker = f".moe.experts.{expert_index}."
    return hash_state_subset(
        state_dict, lambda name: marker not in name and name != "aux_scales"
    )


def infer_aux_layout(state_dict: Mapping[str, Tensor], expert_index: int = 1) -> tuple[int, int]:
    layers: dict[int, int] = {}
    for name, tensor in state_dict.items():
        match = _UP_RE.match(name)
        if match and int(match.group(2)) == expert_index:
            layers[int(match.group(1))] = int(tensor.shape[0])
    if not layers:
        raise ValueError(f"checkpoint has no auxiliary expert {expert_index}")
    expected_layers = list(range(max(layers) + 1))
    if sorted(layers) != expected_layers:
        raise ValueError("auxiliary layers are not contiguous from zero")
    widths = set(layers.values())
    if len(widths) != 1:
        raise ValueError("all auxiliary layers must have the same width")
    return len(layers), widths.pop()


def _tensor_names(layer: int, expert_index: int) -> tuple[str, str, str]:
    prefix = f"blocks.{layer}.moe.experts.{expert_index}"
    return f"{prefix}.c_fc.weight", f"{prefix}.c_fc.bias", f"{prefix}.c_proj.weight"


def validate_key(key: AuxPermutationKey, state_dict: Mapping[str, Tensor] | None = None) -> None:
    if key.schema_version != 1:
        raise ValueError("unsupported key schema version")
    if key.expert_index <= 0:
        raise ValueError("expert_index must identify an auxiliary expert")
    if key.num_layers < 2 or key.aux_width <= 0 or key.group_size <= 0:
        raise ValueError("invalid key layout")
    if key.aux_width % key.group_size:
        raise ValueError("group_size must divide aux_width")
    if not 0 <= key.swap_fraction <= 1:
        raise ValueError("swap_fraction must be in [0, 1]")
    groups_per_layer = key.aux_width // key.group_size
    seen: set[Position] = set()
    for left, right in key.swaps:
        if left == right:
            raise ValueError("a swap cannot target the same position twice")
        if left[0] == right[0]:
            raise ValueError("all swaps must cross layers")
        for position in (left, right):
            layer, group = position
            if not 0 <= layer < key.num_layers or not 0 <= group < groups_per_layer:
                raise ValueError("swap position is out of bounds")
            if position in seen:
                raise ValueError("a group position appears in more than one swap")
            seen.add(position)
    expected_pairs = math.floor((key.num_groups // 2) * key.swap_fraction + 1e-9)
    if len(key.swaps) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} swaps, found {len(key.swaps)}")
    if state_dict is not None:
        num_layers, aux_width = infer_aux_layout(state_dict, key.expert_index)
        if num_layers != key.num_layers or aux_width != key.aux_width:
            raise ValueError("key layout does not match checkpoint")
        for layer in range(key.num_layers):
            up_weight, up_bias, down_weight = _tensor_names(layer, key.expert_index)
            for name in (up_weight, up_bias, down_weight):
                if name not in state_dict:
                    raise ValueError(f"missing key target tensor: {name}")
            if state_dict[up_weight].shape[0] != key.aux_width:
                raise ValueError("up-projection width mismatch")
            if state_dict[up_bias].shape != (key.aux_width,):
                raise ValueError("up-projection bias width mismatch")
            if state_dict[down_weight].shape[1] != key.aux_width:
                raise ValueError("down-projection width mismatch")


def validate_key_labels(key: AuxPermutationKey, labels: list[str] | tuple[str, ...]) -> None:
    """Reject keys issued for a different checkpoint capability slot."""
    if not 0 <= key.expert_index < len(labels):
        raise ValueError("key expert index is absent from checkpoint labels")
    if labels[key.expert_index] != key.capability_label:
        raise ValueError(
            "key capability label does not match the checkpoint expert label"
        )


def generate_key(
    state_dict: Mapping[str, Tensor],
    capability_label: str,
    expert_index: int = 1,
    group_size: int = 8,
    seed: int = 42,
    swap_fraction: float = 1.0,
) -> AuxPermutationKey:
    num_layers, aux_width = infer_aux_layout(state_dict, expert_index)
    if num_layers % 2:
        raise ValueError("pair-swap generation currently requires an even number of layers")
    if aux_width % group_size:
        raise ValueError("group_size must divide auxiliary width")
    if not 0 < swap_fraction <= 1:
        raise ValueError("swap_fraction must be in (0, 1]")
    rng = random.Random(seed)
    layers = list(range(num_layers))
    rng.shuffle(layers)
    groups_per_layer = aux_width // group_size
    swaps: list[Swap] = []
    for pair_index in range(0, num_layers, 2):
        left_layer, right_layer = layers[pair_index : pair_index + 2]
        left_groups = list(range(groups_per_layer))
        right_groups = list(range(groups_per_layer))
        rng.shuffle(left_groups)
        rng.shuffle(right_groups)
        swaps.extend(
            ((left_layer, left_group), (right_layer, right_group))
            for left_group, right_group in zip(left_groups, right_groups)
        )
    rng.shuffle(swaps)
    keep = math.floor(len(swaps) * swap_fraction + 1e-9)
    key = AuxPermutationKey(
        schema_version=1,
        capability_label=capability_label,
        expert_index=expert_index,
        num_layers=num_layers,
        aux_width=aux_width,
        group_size=group_size,
        seed=seed,
        swap_fraction=swap_fraction,
        swaps=tuple(swaps[:keep]),
        original_aux_sha256=hash_auxiliary(state_dict, expert_index),
    )
    validate_key(key, state_dict)
    return key


@torch.no_grad()
def apply_key(
    state_dict: Mapping[str, Tensor],
    key: AuxPermutationKey,
    *,
    in_place: bool = False,
) -> dict[str, Tensor]:
    validate_key(key, state_dict)
    target = state_dict if in_place else clone_state_dict(state_dict)
    if not isinstance(target, dict):
        target = dict(target)
    size = key.group_size
    for (left_layer, left_group), (right_layer, right_group) in key.swaps:
        left_names = _tensor_names(left_layer, key.expert_index)
        right_names = _tensor_names(right_layer, key.expert_index)
        left_slice = slice(left_group * size, (left_group + 1) * size)
        right_slice = slice(right_group * size, (right_group + 1) * size)

        left_up, left_bias, left_down = left_names
        right_up, right_bias, right_down = right_names

        temporary = target[left_up][left_slice].clone()
        target[left_up][left_slice].copy_(target[right_up][right_slice])
        target[right_up][right_slice].copy_(temporary)

        temporary = target[left_bias][left_slice].clone()
        target[left_bias][left_slice].copy_(target[right_bias][right_slice])
        target[right_bias][right_slice].copy_(temporary)

        temporary = target[left_down][:, left_slice].clone()
        target[left_down][:, left_slice].copy_(target[right_down][:, right_slice])
        target[right_down][:, right_slice].copy_(temporary)
    return target


def partial_key(key: AuxPermutationKey, fraction: float, seed: int) -> AuxPermutationKey:
    if not 0 <= fraction <= 1:
        raise ValueError("partial-key fraction must be in [0, 1]")
    rng = random.Random(seed)
    swaps = list(key.swaps)
    rng.shuffle(swaps)
    keep = round(len(swaps) * fraction)
    return replace(
        key,
        seed=seed,
        swap_fraction=keep / max(key.num_groups // 2, 1),
        swaps=tuple(swaps[:keep]),
    )


def verify_restoration(state_dict: Mapping[str, Tensor], key: AuxPermutationKey) -> bool:
    return hash_auxiliary(state_dict, key.expert_index) == key.original_aux_sha256
