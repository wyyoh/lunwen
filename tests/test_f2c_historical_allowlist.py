"""精确例外不能掩盖新失败、不同调用栈或数量增加。"""

from copy import deepcopy
from pathlib import Path

import pytest

from keyed_gram.stage_f2c_regression import RegressionError, classify, load_allowlist


def registered():
    return load_allowlist("configs/f2c_historical_test_allowlist.yaml", Path.cwd())


def capture_for(value):
    return {
        "collected_nodeids": [e["test_nodeid"] for e in value["exceptions"]],
        "collection_errors": [],
        "pytest_exit_code": 1,
        "reports": [
            {
                "nodeid": e["test_nodeid"],
                "when": "call",
                "outcome": "failed",
                "signature": deepcopy(e["signature"]),
            }
            for e in value["exceptions"]
        ],
    }


def test_exact_seven_preserve_raw_failure_count():
    value = registered()
    result = classify(capture_for(value), value)
    assert result["raw_fail_count"] == 7
    assert result["allowlisted_failure_count"] == 7
    assert result["unexpected_failure_count"] == 0
    assert result["full_repository_raw_status"] == "failed"
    assert result["current_stage_regression_status"] == "passed"


@pytest.mark.parametrize(
    "field",
    ["exception_type", "message", "frames", "context", "chained_exception", "when"],
)
def test_any_failure_signature_change_blocks(field):
    value = registered()
    capture = capture_for(value)
    capture["reports"][0]["signature"][field] = "changed"
    assert classify(capture, value)["unexpected_failure_count"] > 0


def test_new_f2c_failure_is_never_allowlisted():
    value = registered()
    capture = capture_for(value)
    item = deepcopy(capture["reports"][0])
    item["nodeid"] = "tests/test_symbolic_cegis.py::new_failure"
    capture["reports"].append(item)
    capture["collected_nodeids"].append(item["nodeid"])
    assert classify(capture, value)["unexpected_failure_count"] == 1


def test_duplicate_failure_exceeds_per_node_limit():
    value = registered()
    capture = capture_for(value)
    capture["reports"].append(deepcopy(capture["reports"][0]))
    assert classify(capture, value)["unexpected_failure_count"] > 0


def test_extra_traceback_frame_blocks():
    value = registered()
    capture = capture_for(value)
    capture["reports"][0]["signature"]["frames"].append(
        {"path": "src/new.py", "line": 1, "function": "new"}
    )
    assert classify(capture, value)["unexpected_failure_count"] == 1


def test_no_failures_is_not_forced_to_expected_seven():
    value = registered()
    capture = capture_for(value)
    capture["pytest_exit_code"] = 0
    for r in capture["reports"]:
        r["outcome"] = "passed"
        r.pop("signature")
    result = classify(capture, value)
    assert result["raw_pass_count"] == 7
    assert result["allowlisted_failure_count"] == 0
    assert result["unexpected_failure_count"] == 0


@pytest.mark.parametrize(
    "change", ["missing_call", "collection_error", "skip", "exit_code"]
)
def test_incomplete_test_run_blocks(change):
    value = registered()
    capture = capture_for(value)
    if change == "missing_call":
        capture["reports"].pop()
    if change == "collection_error":
        capture["collection_errors"] = ["collection failed"]
    if change == "skip":
        capture["reports"][0]["outcome"] = "skipped"
    if change == "exit_code":
        capture["pytest_exit_code"] = 3
    assert classify(capture, value)["unexpected_failure_count"] > 0


@pytest.mark.parametrize(
    "change", ["wildcard", "extra", "duplicate", "limit", "nan", "unknown_field"]
)
def test_allowlist_schema_cannot_be_expanded(tmp_path, change):
    import json

    value = deepcopy(registered())
    if change == "wildcard":
        value["exceptions"][0]["test_nodeid"] = "tests/*"
    if change == "extra":
        value["exceptions"].append(deepcopy(value["exceptions"][0]))
    if change == "duplicate":
        value["exceptions"][1]["id"] = value["exceptions"][0]["id"]
    if change == "limit":
        value["maximum_total_failures"] = 8
    if change == "nan":
        value["purpose"] = float("nan")
    if change == "unknown_field":
        value["ignore_all_failures"] = True
    path = tmp_path / "allowlist.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((RegressionError, ValueError)):
        load_allowlist(path)


def test_duplicate_yaml_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(RegressionError, match="重复"):
        load_allowlist(path)


def test_changed_reason_input_hash_blocks(tmp_path):
    import json

    value = registered()
    value["reason_input_sha256"][".gitattributes"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RegressionError, match="历史原因输入"):
        load_allowlist(path, Path.cwd())
