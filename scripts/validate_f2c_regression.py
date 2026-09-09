"""运行全部普通 pytest，再精确分类；原始退出码和每项失败均保留。"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from keyed_gram.stage_f2c_protocol import read_json, write_json
from keyed_gram.stage_f2c_regression import classify, load_allowlist, sha_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("configs/f2c_historical_test_allowlist.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    allowlist = load_allowlist(args.allowlist, root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["F2C_REGRESSION_CAPTURE"] = str(output / "pytest_capture.json")
    env.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "keyed_gram.stage_f2c_regression",
        "--junitxml=" + str(output / "full_repository.xml"),
    ]
    start = time.monotonic()
    with (output / "full_repository.log").open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    capture = read_json(output / "pytest_capture.json")
    if capture["pytest_exit_code"] != process.returncode:
        raise RuntimeError("pytest 子进程退出码与 capture 不一致")
    result = classify(capture, allowlist)
    result.update(
        {
            "duration_seconds": time.monotonic() - start,
            "allowlist_sha256": sha_file(args.allowlist),
            "validation_script_sha256": sha_file(__file__),
            "classifier_sha256": sha_file(
                root / "src/keyed_gram/stage_f2c_regression.py"
            ),
            "capture_sha256": sha_file(output / "pytest_capture.json"),
            "raw_log_sha256": sha_file(output / "full_repository.log"),
            "raw_xml_sha256": sha_file(output / "full_repository.xml"),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        }
    )
    write_json(output / "regression_summary.json", result)
    print(
        {
            k: result[k]
            for k in (
                "raw_pass_count",
                "raw_fail_count",
                "allowlisted_failure_count",
                "unexpected_failure_count",
            )
        }
    )
    return int(result["unexpected_failure_count"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
