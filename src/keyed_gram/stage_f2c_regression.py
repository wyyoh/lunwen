"""精确历史兼容例外；保留 pytest 原始结果，不改变历史测试行为。"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

D21_NAMES = (
    "test_formal_and_smoke_configs_validate",
    "test_prepare_manifest_does_not_persist_cache_path",
    "test_attack_matrix_and_summary_pass_with_mock",
    "test_summary_rejects_missing_preregistered_scenario",
    "test_summary_fails_when_parameter_hash_changes",
)
ACCEPTED_NODEIDS = frozenset(
    [f"tests/test_stage_d21_protocol.py::{n}" for n in D21_NAMES]
    + [
        "tests/test_stage_f2a_autheffect.py::test_f1_frozen_manifests_and_status_match",
        "tests/test_stage_d23_runtime.py::test_full_preregistered_matrix_rehearsal",
    ]
)


class RegressionError(ValueError):
    """新失败、签名漂移或证据不完整不能获豁免。"""


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise RegressionError("重复 YAML key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def safe_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise RegressionError("必须是安全相对路径")
    return Path(value)


def load_allowlist(path, root=None):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise RegressionError("重复 JSON key")
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(RegressionError(x)),
        )
    else:
        value = yaml.load(text, Loader=StrictLoader)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "purpose",
        "maximum_total_failures",
        "registration_basis",
        "reason_input_sha256",
        "exceptions",
    }:
        raise RegressionError("allowlist schema 不匹配")
    if (
        value["schema_version"] != 1
        or type(value["maximum_total_failures"]) is not int
        or value["maximum_total_failures"] != 7
    ):
        raise RegressionError("只允许已确认的七项例外")
    entries = value["exceptions"]
    if len(entries) != 7 or {e["test_nodeid"] for e in entries} != ACCEPTED_NODEIDS:
        raise RegressionError("node ID 不得增加、重复或模糊匹配")
    if len({e["id"] for e in entries}) != 7:
        raise RegressionError("重复 exception ID")
    for e in entries:
        if (
            set(e)
            != {
                "id",
                "test_nodeid",
                "category",
                "expected_reason",
                "maximum_occurrences",
                "signature",
            }
            or type(e["maximum_occurrences"]) is not int
            or e["maximum_occurrences"] != 1
        ):
            raise RegressionError("例外 schema/次数不匹配")
        sig = e["signature"]
        if (
            set(sig)
            != {
                "when",
                "exception_type",
                "message",
                "frames",
                "context",
                "chained_exception",
            }
            or sig["when"] != "call"
            or not sig["frames"]
            or sig["chained_exception"] is not False
        ):
            raise RegressionError("失败签名不完整")
        for frame in sig["frames"]:
            if (
                set(frame) != {"path", "line", "function"}
                or type(frame["line"]) is not int
            ):
                raise RegressionError("traceback frame schema")
            safe_path(frame["path"])
    for relative, expected in value["reason_input_sha256"].items():
        safe_path(relative)
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise RegressionError("reason input SHA 非法")
        if root is not None and sha_file(Path(root) / relative) != expected:
            raise RegressionError("历史原因输入发生变化：" + relative)
    canonical(value)  # 拒绝 YAML .nan/.inf。
    return value


def classify(capture, allowlist):
    entries = {e["test_nodeid"]: e for e in allowlist["exceptions"]}
    reports = capture["reports"]
    collected = capture["collected_nodeids"]
    unexpected, accepted, seen = [], [], Counter()
    calls = Counter(r["nodeid"] for r in reports if r["when"] == "call")
    passed = sum(r["outcome"] == "passed" and r["when"] == "call" for r in reports)
    skipped = sum(r["outcome"] == "skipped" for r in reports)
    failures = [r for r in reports if r["outcome"] == "failed"]
    for r in failures:
        e = entries.get(r["nodeid"])
        seen[r["nodeid"]] += 1
        exact = (
            e is not None
            and r.get("signature") == e["signature"]
            and seen[r["nodeid"]] <= e["maximum_occurrences"]
        )
        item = {
            "test_nodeid": r["nodeid"],
            "when": r["when"],
            "signature": r.get("signature"),
            "signature_sha256": digest(r.get("signature")),
        }
        (accepted if exact else unexpected).append(item)
    infrastructure = []
    if capture["pytest_exit_code"] not in (0, 1) or capture["collection_errors"]:
        infrastructure.append("pytest_collection_or_execution_error")
    if (
        not collected
        or len(collected) != len(set(collected))
        or set(collected) != set(calls)
    ):
        infrastructure.append("incomplete_or_duplicate_test_execution")
    if any(count != 1 for count in calls.values()) or skipped:
        infrastructure.append("skipped_or_repeated_tests_not_accepted")
    if bool(failures) != bool(capture["pytest_exit_code"]):
        infrastructure.append("pytest_exit_code_inconsistent")
    if len(accepted) > 7:
        infrastructure.append("allowlist_count_exceeded")
    count = len(unexpected) + len(infrastructure)
    return {
        "schema_version": 1,
        "validation_kind": "ordinary_full_repository_regression_not_formal_audit",
        "raw_pass_count": passed,
        "raw_fail_count": len(failures),
        "raw_skip_count": skipped,
        "collected_test_count": len(collected),
        "pytest_raw_exit_code": capture["pytest_exit_code"],
        "full_repository_raw_status": "failed"
        if failures or infrastructure
        else "passed",
        "allowlisted_failure_count": len(accepted),
        "historical_expected_failures": len(accepted),
        "unexpected_failure_count": count,
        "unexpected_regression_count": count,
        "allowlisted_failures": accepted,
        "unexpected_failures": unexpected,
        "infrastructure_failures": infrastructure,
        "current_stage_regression_status": "passed" if count == 0 else "failed",
        "historical_compatibility_status": "known_incompatibilities_allowlisted"
        if accepted
        else "no_historical_failure_observed",
        "historical_compatibility_allowlist_status": "passed"
        if count == 0
        else "failed",
        "ready_for_analyzer_freeze": count == 0,
        "formal_development_started": False,
        "formal_locked_started": False,
    }


def exception_signature(excinfo, nodeid, when, root):
    entries = list(excinfo.traceback)
    node_path = nodeid.split("::", 1)[0]
    frames, started = [], False
    terminal = entries[-1]
    for entry in entries:
        path = Path(str(entry.path)).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = "external/" + path.name
        started = started or relative == node_path
        if started:
            frames.append(
                {
                    "path": relative,
                    "line": entry.lineno + 1,
                    "function": entry.frame.code.name,
                }
            )
    context = {}
    local = terminal.frame.f_locals
    global_ = terminal.frame.f_globals
    exc = excinfo.value
    if frames and frames[-1]["path"] == "src/keyed_gram/stage_d21_protocol.py":
        g1 = local.get("g1", {})
        context = {
            "required_transformers": g1.get("transformers_version"),
            "runtime_transformers": getattr(
                global_.get("transformers"), "__version__", None
            ),
            "required_torch": g1.get("torch_version"),
            "runtime_torch": getattr(global_.get("torch"), "__version__", None),
        }
    elif (
        frames
        and frames[-1]["path"] == "src/keyed_gram/stage_f2a_autheffect_protocol.py"
    ):
        context = {
            "source_failures": local.get("source_failures"),
            "artifact_failures": local.get("artifact_failures"),
            "inventory_exact": local.get("actual_inventory")
            == local.get("expected_inventory"),
        }
    elif frames and frames[-1]["path"] == "src/keyed_gram/stage_d22_ipc.py":
        socket = Path(str(local.get("socket_path", "")))
        context = {
            "timeout_seconds": local.get("timeout_seconds"),
            "socket_name": socket.name,
            "cluster_name": socket.parent.name,
            "reason": getattr(getattr(exc, "reason", None), "value", None),
        }
    return {
        "when": when,
        "exception_type": type(exc).__module__ + "." + type(exc).__qualname__,
        "message": str(exc),
        "frames": frames,
        "context": context,
        "chained_exception": exc.__cause__ is not None or exc.__context__ is not None,
    }


def pytest_configure(config):
    if os.environ.get("F2C_REGRESSION_CAPTURE"):
        config._f2c_capture = {
            "reports": [],
            "collected_nodeids": [],
            "collection_errors": [],
        }


def pytest_collection_finish(session):
    if hasattr(session.config, "_f2c_capture"):
        session.config._f2c_capture["collected_nodeids"] = [
            i.nodeid for i in session.items
        ]


def pytest_collectreport(report):
    # collection errors 另由退出码和缺失 call 检测；不拦截 collection。
    pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not hasattr(item.config, "_f2c_capture"):
        return
    record = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
    if report.failed and call.excinfo is not None:
        record["signature"] = exception_signature(
            call.excinfo,
            report.nodeid,
            report.when,
            Path(str(item.config.rootpath)).resolve(),
        )
    item.config._f2c_capture["reports"].append(record)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "_f2c_capture"):
        result = session.config._f2c_capture
        result["pytest_exit_code"] = int(exitstatus)
        with Path(os.environ["F2C_REGRESSION_CAPTURE"]).open(
            "x", encoding="utf-8"
        ) as handle:
            handle.write(canonical(result) + "\n")
