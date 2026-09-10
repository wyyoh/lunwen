"""调度测试使用 mock case；绝不物化或评分真实 development/locked。"""

import ast
import subprocess
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from keyed_gram.stage_f2c_protocol import (
    F2CProtocolError,
    authorize_locked,
    begin_evaluation,
    mark_started,
    read_json,
    record_phase,
    validate_artifact_inventory,
    write_json,
)


def test_split_materialization_never_constructs_unrequested_cases(monkeypatch):
    from keyed_gram.authsynth_symbolic_verifier import benchmark as b

    seen = []

    def fake(*args, **kwargs):
        seen.append((kwargs["split"], kwargs["serial"]))
        return kwargs

    monkeypatch.setattr(b, "_make_case", fake)
    args = {
        "analyzer_freeze_commit": "a" * 40,
        "grammar": None,
        "replay_budget": 16,
        "solver_timeout_ms": 3000,
    }
    b.generate_authsymbolbench(
        "fixture", splits=("train", "calibration", "development"), **args
    )
    assert len(seen) == 120 and {s for s, _ in seen} == {
        "train",
        "calibration",
        "development",
    }
    assert [i for _, i in seen] == list(range(1, 121))
    seen.clear()
    b.generate_authsymbolbench("fixture", splits=("locked_test",), **args)
    assert seen == [("locked_test", i) for i in range(121, 161)]


def test_case_constructor_and_templates_remain_byte_semantically_identical():
    root = Path(__file__).resolve().parents[1]
    p = "src/keyed_gram/authsynth_symbolic_verifier/benchmark.py"
    old = subprocess.check_output(
        ["git", "show", "a3bdb3b9ed244b3e05bb7c576737f165ac44425b:" + p],
        cwd=root,
        text=True,
    )

    def defs(text):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    before, after = defs(old), defs((root / p).read_text())
    assert {k: v for k, v in before.items() if k != "generate_authsymbolbench"} == {
        k: v for k, v in after.items() if k != "generate_authsymbolbench"
    }
    for name in (
        "template_families.py",
        "template_isolation.py",
        "evaluator.py",
        "solver.py",
        "equivalence.py",
        "replay_service.py",
    ):
        p = "src/keyed_gram/authsynth_symbolic_verifier/" + name
        assert (root / p).read_bytes() == subprocess.check_output(
            ["git", "show", "a3bdb3b:" + p], cwd=root
        )


def runtime(tmp_path):
    d = tmp_path / "runtime"
    mark_started(d, {"commit": "a" * 40})
    for phase in ("materialize", "calibration", "configuration_freeze"):
        record_phase(d, phase)
    return d


def test_development_consumed_before_execution_and_cannot_retry(tmp_path):
    d = runtime(tmp_path)
    begin_evaluation(d, "development")
    assert read_json(d / "development.consumed.json")["run_count"] == 1
    with pytest.raises(F2CProtocolError, match="already consumed"):
        begin_evaluation(d, "development")


def test_locked_requires_gate_authorization(tmp_path):
    d = runtime(tmp_path)
    record_phase(d, "development")
    with pytest.raises(F2CProtocolError):
        begin_evaluation(d, "locked_test")
    write_json(d / "locked_authorization.json", {"allowed": False})
    with pytest.raises(F2CProtocolError):
        begin_evaluation(d, "locked_test")


def test_failed_development_cannot_authorize_locked(tmp_path):
    d = runtime(tmp_path)
    record_phase(d, "development")
    with pytest.raises(F2CProtocolError, match="continuation"):
        authorize_locked(d, None, None, {"utility": False})


@pytest.mark.parametrize("dirty", [True, False])
def test_changed_source_or_environment_blocks_locked(tmp_path, monkeypatch, dirty):
    import keyed_gram.stage_f2c_protocol as p

    d = runtime(tmp_path)
    record_phase(d, "development")
    monkeypatch.setattr(p, "git_state", lambda _: {"tracked_dirty": dirty})
    monkeypatch.setattr(p, "repo_root", lambda _: tmp_path)
    monkeypatch.setattr(p, "output_paths", lambda *args: ())
    monkeypatch.setattr(p.subprocess, "check_output", lambda *args, **kwargs: "")

    def fail(*args):
        raise F2CProtocolError("environment mismatch")

    monkeypatch.setattr(p, "verify_frozen_files", fail)
    with pytest.raises(F2CProtocolError):
        authorize_locked(d, None, {}, {"utility": True})
    assert not (d / "locked_authorization.json").exists()


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    import keyed_gram.stage_f2c as s
    from keyed_gram.authsynth_symbolic_verifier.benchmark import AuthSymbolBenchVault
    from keyed_gram.authsynth_symbolic_verifier.evaluator import SymbolicEvaluationRow

    config = s.load_config("configs/stage_f2c.yaml")
    config["frozen_environment"] = []
    dest = tmp_path / config["protocol"]["freeze_binding_path"]
    dest.parent.mkdir(parents=True)
    dest.write_text("{}")
    out = (
        tmp_path / "data",
        tmp_path / "formal_artifacts",
        tmp_path / "runtime",
        tmp_path / "report.md",
    )
    monkeypatch.setattr(s, "load_config", lambda _: config)
    monkeypatch.setattr(
        s,
        "formal_preflight",
        lambda *args: {"git": {"commit": "b" * 40, "branch": "fixture"}, "frozen": {}},
    )
    monkeypatch.setattr(s, "repo_root", lambda _: tmp_path)
    monkeypatch.setattr(s, "output_paths", lambda *args: out)
    monkeypatch.setattr(s, "_source_manifest", lambda _: {})
    monkeypatch.setattr(
        s, "information_boundary_audit", lambda *args: {"status": "passed"}
    )
    calls = []

    def generate(*args, **kwargs):
        calls.append(("materialize", kwargs["splits"]))
        cases = tuple(
            SimpleNamespace(
                split=split,
                tool_family=split,
                domain="fixture",
                mutation_category="fixture",
                expected_unknown=False,
                clean_control=True,
                implementation=SimpleNamespace(loop=None),
                analyzer_input=SimpleNamespace(
                    schema=SimpleNamespace(cardinality=512),
                    to_public_dict=lambda: {"public_case_id": "fixture"},
                ),
                public_case_id=split,
                evaluator_manifest_row=lambda: {"digest": "fixture"},
            )
            for split in kwargs["splits"]
        )
        return AuthSymbolBenchVault(cases, "b" * 40, {})

    monkeypatch.setattr(s, "generate_authsymbolbench", generate)
    monkeypatch.setattr(
        s,
        "collision_audit",
        lambda cases: {"cross_split_collision_count": 0, "status": "passed"},
    )
    monkeypatch.setattr(s, "_configured_cases", lambda cases, *args: cases)
    metric = {"method": "AuthSynth-Symbolic"}
    for k in config["readiness_gates"]:
        name = k.removeprefix("maximum_").removeprefix("minimum_")
        metric[name] = 0 if k.startswith("maximum_") else 1
    metric["false_verified_complete_rate"] = 0
    monkeypatch.setattr(s, "aggregate_metrics", lambda rows: [dict(metric)])
    monkeypatch.setattr(
        s,
        "_calibrate",
        lambda *args: (
            config["calibration_candidates"][0],
            [{"candidate_id": "fixture"}],
        ),
    )

    def evaluate(cases, directory, split, **kwargs):
        calls.append(("evaluate", split))
        values = {}
        for f in fields(SymbolicEvaluationRow):
            values[f.name] = (
                ()
                if str(f.type).startswith("tuple")
                else (False if f.type == "bool" else ("" if f.type == "str" else 0))
            )
        values.update(
            method="AuthSynth-Symbolic",
            case_id=split,
            split=split,
            domain="fixture",
            mutation_category="fixture",
            domain_assignment_count=512,
            certificate_id=None,
            analysis_unknown=True,
            reason_codes=("fixture_unknown",),
            predicted_patch_types=("guard_patch",),
            expected_patch_types=("guard_patch",),
        )
        row = SymbolicEvaluationRow(**values)
        return {"rows": (row,), "metrics": [dict(metric)]}

    monkeypatch.setattr(s, "_evaluate_checkpointed", evaluate)

    def authorize(directory, *args):
        calls.append(("integrity", "passed"))
        write_json(directory / "locked_authorization.json", {"allowed": True})
        return {"status": "passed", "post_development_analyzer_hash_changed": False}

    monkeypatch.setattr(s, "authorize_locked", authorize)
    return s, out, metric, calls


def test_failed_development_is_archived_without_materializing_locked(fake_pipeline):
    s, out, metric, calls = fake_pipeline
    metric["safe_utility"] = 0
    result = s.run_formal_build("fixture")
    assert result["symbolic_contract_synthesis_status"] == "failed_development"
    assert result["formal_locked_run_count"] == 0
    assert not any("locked_test" in str(c) for c in calls)
    assert not (out[0] / "locked_test.jsonl").exists()
    assert not (out[1] / "locked_test_results.csv").exists()
    validate_artifact_inventory(out[1], out[3], locked_scored=False)
    with pytest.raises(FileExistsError):
        s.run_formal_build("fixture")


def test_gate_then_integrity_then_locked_materialization_and_evaluation(fake_pipeline):
    s, out, _metric, calls = fake_pipeline
    result = s.run_formal_build("fixture")
    assert result["formal_locked_run_count"] == 1
    assert (
        calls.index(("evaluate", "development"))
        < calls.index(("integrity", "passed"))
        < calls.index(("materialize", ("locked_test",)))
        < calls.index(("evaluate", "locked_test"))
    )
    validate_artifact_inventory(out[1], out[3])


def test_safety_config_is_unchanged_except_freeze_protocol_binding():
    import yaml

    current = yaml.safe_load(Path("configs/stage_f2c.yaml").read_text())
    old = yaml.safe_load(
        subprocess.check_output(
            ["git", "show", "a3bdb3b:configs/stage_f2c.yaml"], text=True
        )
    )
    assert {k: v for k, v in current.items() if k != "protocol"} == {
        k: v for k, v in old.items() if k != "protocol"
    }
    assert (
        current["protocol"]["benchmark_seed_freeze_commit"]
        == "f141c77278a8924373d38585ff55036ca7340854"
    )
