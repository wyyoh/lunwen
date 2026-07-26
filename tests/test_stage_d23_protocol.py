from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from keyed_gram.stage_d23_protocol import (
    D23ProtocolError,
    load_config,
    required_scenarios,
)


def test_formal_config_freezes_51_unique_scenarios() -> None:
    values = load_config("configs/stage_d23.yaml")
    assert len(required_scenarios(values)) == 51
    assert values["protocol"]["physical_disk_exhaustion_allowed"] is False
    assert values["governance"]["c3_eligible"] is False


def test_private_value_or_raw_ipc_gate_cannot_be_enabled(
    tmp_path: Path,
) -> None:
    values = yaml.safe_load(Path("configs/stage_d23.yaml").read_text())
    values["protocol"]["private_value_memory_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(D23ProtocolError):
        load_config(path)
