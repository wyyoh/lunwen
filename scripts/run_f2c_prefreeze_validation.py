"""串行普通验证；不调用 formal preflight、阶段评分器或 held-out 生成器。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tests = sorted(
        {
            *root.glob("tests/test_symbolic*.py"),
            *root.glob("tests/test_stage_f2c*.py"),
            *root.glob("tests/test_f2c*.py"),
        }
    )
    test_paths = [p.relative_to(root).as_posix() for p in tests]
    lint = [
        "src/keyed_gram/authsynth_symbolic_shared",
        "src/keyed_gram/authsynth_symbolic_analyzer",
        "src/keyed_gram/authsynth_symbolic_verifier",
        "src/keyed_gram/stage_f2c.py",
        "src/keyed_gram/stage_f2c_metrics.py",
        "src/keyed_gram/stage_f2c_protocol.py",
        "src/keyed_gram/stage_f2c_regression.py",
        "src/keyed_gram/stage_f2c_freeze.py",
        "tests/f2c_symbolic_helpers.py",
        "scripts/audit_f2c_templates.py",
        "scripts/validate_f2c_draft.py",
        "scripts/run_f2c_prefreeze_validation.py",
        "scripts/validate_f2c_regression.py",
        "scripts/freeze_f2c_analyzer.py",
        "scripts/audit_f2c_prefreeze.py",
        "scripts/run_f2c_train_pilot.py",
        *test_paths,
    ]
    python = sys.executable
    commands = {
        "targeted": [
            python,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *test_paths,
            "--junitxml=" + str(output / "targeted.xml"),
        ],
        "template_isolation": [
            python,
            "scripts/audit_f2c_templates.py",
            "--output",
            str(output / "template_isolation_audit.json"),
        ],
        "loop_equivalence": [
            python,
            "-c",
            (
                "import sys,json; sys.path.insert(0,'tests'); from test_f2c_loop_equivalence import differential_audit; "
                "r=differential_audit(); json.dump(r,open(sys.argv[1],'x'),sort_keys=True,indent=2,allow_nan=False); "
                "sys.exit(int(r['concrete_symbolic_loop_equivalence_failure_count']!=0))"
            ),
            str(output / "bounded_loop_equivalence_audit.json"),
        ],
        "ruff": [python, "-m", "ruff", "check", "--no-cache", *lint],
        "git_diff_check": ["git", "diff", "--check"],
        "full_repository": [
            python,
            "scripts/validate_f2c_regression.py",
            "--output",
            str(output / "regression"),
        ],
    }
    records = {}
    for name, command in commands.items():
        with (output / (name + ".log")).open("x", encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        records[name] = {
            "exit_code": process.returncode,
            "log_sha256": digest(output / (name + ".log")),
        }
        print(name + ": " + str(process.returncode), flush=True)
    source_files = set(tests)
    for item in lint:
        p = root / item
        source_files.update(p.glob("*.py") if p.is_dir() else (p,))
    source_files.update(root.glob("configs/stage_f2c*.yaml"))
    source_files.add(root / "configs/f2c_historical_test_allowlist.yaml")
    source_files.add(root / "requirements-stage-f2c.txt")
    result = {
        "validation_kind": "ordinary_prefreeze_regression_not_formal_experiment",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "commands": records,
        "source_files": [
            {"path": p.relative_to(root).as_posix(), "sha256": digest(p)}
            for p in sorted(source_files)
        ],
        "formal_development_started": False,
        "formal_locked_started": False,
        "analyzer_freeze_created": False,
    }
    (output / "validation_run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return int(any(r["exit_code"] for r in records.values()))


if __name__ == "__main__":
    raise SystemExit(main())
