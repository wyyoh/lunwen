from __future__ import annotations

import torch

from keyed_gram.checkpoint import load_clean_checkpoint, save_clean_checkpoint
from keyed_gram.keying import apply_key, generate_key, verify_restoration
from keyed_gram.model import GramTransformer


def test_clean_checkpoint_lock_restore_roundtrip(tmp_path, tiny_config):
    model = GramTransformer(tiny_config)
    original_path = tmp_path / "original.pt"
    save_clean_checkpoint(original_path, model, tiny_config, ["core", "private"])
    payload = load_clean_checkpoint(original_path)
    assert set(payload) == {"format_version", "model", "model_config", "labels"}
    assert "optimizer" not in payload
    key = generate_key(payload["model"], "private", group_size=4, seed=42)
    locked = apply_key(payload["model"], key)
    locked_path = tmp_path / "locked.pt"
    save_clean_checkpoint(
        locked_path,
        None,
        tiny_config,
        ["core", "private"],
        state_dict=locked,
    )
    loaded_locked = load_clean_checkpoint(locked_path)
    restored = apply_key(loaded_locked["model"], key)
    assert verify_restoration(restored, key)
    assert all(torch.equal(payload["model"][name], restored[name]) for name in restored)
