"""可重复的非正式 F2C 定向验证；不调用任何正式审计入口。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def validate(root: Path) -> dict:
    tests = sorted(root.glob("tests/test_symbolic*.py"))
    tests += sorted(root.glob("tests/test_stage_f2c*.py"))
    tests += sorted(root.glob("tests/test_f2c*.py"))
    relative_tests = [path.relative_to(root).as_posix() for path in tests]
    if not relative_tests:
        raise RuntimeError("缺少 F2C 定向测试")
    lint_paths = [
        "src/keyed_gram/authsynth_symbolic_shared",
        "src/keyed_gram/authsynth_symbolic_analyzer",
        "src/keyed_gram/authsynth_symbolic_verifier",
        "src/keyed_gram/stage_f2c.py",
        "src/keyed_gram/stage_f2c_protocol.py",
        "src/keyed_gram/stage_f2c_metrics.py",
        "tests/f2c_symbolic_helpers.py",
        "scripts/audit_f2c_prefreeze.py",
        "scripts/validate_f2c_draft.py",
        "scripts/run_f2c_train_pilot.py",
        "scripts/audit_f2c_templates.py",
        *relative_tests,
    ]
    commands = {
        "pytest": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *relative_tests,
        ],
        "ruff": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format",
            "concise",
            *lint_paths,
        ],
        "train_smoke": [sys.executable, "-m", "keyed_gram.stage_f2c", "--smoke"],
    }
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "NO_COLOR": "1",
            "PYTHONUTF8": "1",
        }
    )
    results = {}
    for label, command in commands.items():
        process = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            check=False,
        )
        results[label] = {
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    source_paths = set(tests)
    for relative in lint_paths:
        path = root / relative
        source_paths.update(path.glob("*.py") if path.is_dir() else (path,))
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(source_paths)
    ]
    return {
        "schema_version": 1,
        "validation_kind": "nonformal_f2c_unit_and_train_smoke",
        "unit_validation_status": "passed"
        if all(r["exit_code"] == 0 for r in results.values())
        else "failed",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("z3-solver", "pytest", "ruff", "pyyaml")
        },
        "test_files": relative_tests,
        "commands": results,
        "source_files": records,
        "container_environment_verified": False,
        "full_repository_tests_run": False,
        "formal_development_started": False,
        "formal_locked_started": False,
        "analyzer_frozen": False,
        "ready_for_formal_f2c_audit": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1])
    payload = (
        json.dumps(
            result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    if args.output:
        # 非正式结果也不覆盖既有记录。
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    print(payload)
    return 0 if result["unit_validation_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
