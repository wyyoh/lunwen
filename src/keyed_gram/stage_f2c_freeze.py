"""冻结提交与后置协议元数据分离；不运行实验或物化 held-out。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


class FreezeError(ValueError):
    pass


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise FreezeError("duplicate JSON key")
        out[key] = value
    return out


def read(path):
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(FreezeError(x)),
    )
    if not isinstance(value, dict):
        raise FreezeError("manifest 必须是 object")
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )


def safe(value):
    if (
        not isinstance(value, str)
        or not value
        or any(x in value for x in ("\\", ":", "\x00"))
        or any(x in ("", ".", "..") for x in value.split("/"))
    ):
        raise FreezeError("unsafe relative path")
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def blob(root, commit, path):
    return git(root, "show", commit + ":" + safe(path))


def analyzer_path(path):
    return (
        path.startswith(
            (
                "src/keyed_gram/authsynth_symbolic_",
                "src/keyed_gram/stage_f2c",
                "docs/f2c/",
                "artifacts/stage_f2c_compatibility/",
                "artifacts/stage_f2c_protocol_repair/",
                "tests/test_symbolic",
                "tests/test_stage_f2c",
                "tests/test_f2c",
                "configs/stage_f2c",
            )
        )
        or path == "tests/f2c_symbolic_helpers.py"
        or path == "configs/f2c_historical_test_allowlist.yaml"
        or path == "AUTHSYNTH_F2C_RESEARCH_PLAN.md"
        or path == "requirements-stage-f2c.txt"
        or (path.startswith("scripts/") and "f2c" in Path(path).name)
    )


def records(root, commit, paths):
    return [
        {"path": safe(p), "sha256": sha(blob(root, commit, p))}
        for p in sorted(set(paths))
    ]


def _same_file(root, commit, entry):
    p = safe(entry["path"])
    expected = entry["sha256"]
    if (
        sha(blob(root, commit, p)) != expected
        or sha((root / p).read_bytes()) != expected
    ):
        raise FreezeError("freeze source mismatch: " + p)


def runtime_facts():
    import torch

    torch.set_num_threads(1)
    return {
        "architecture": platform.machine(),
        "glibc": platform.libc_ver()[1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "z3": importlib.metadata.version("z3-solver"),
        "transformers": importlib.metadata.version("transformers"),
        "cpu_only": torch.version.cuda is None and not torch.cuda.is_available(),
        "cpu_operation_ok": bool(
            torch.tensor([[1, 2]]) @ torch.tensor([[3], [4]]) == torch.tensor([[11]])
        ),
    }


def load_binding(root, path):
    root = Path(root)
    binding = read(root / safe(path))
    expected = {
        "schema_version",
        "analyzer_freeze_commit",
        "manifests",
        "formal_development_started",
        "formal_locked_started",
    }
    if (
        set(binding) != expected
        or binding["schema_version"] != 1
        or binding["formal_development_started"] is not False
        or binding["formal_locked_started"] is not False
    ):
        raise FreezeError("freeze binding schema")
    commit = binding["analyzer_freeze_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise FreezeError("freeze commit invalid")
    if set(binding["manifests"]) != {
        "frozen_upstream",
        "frozen_analyzer",
        "frozen_environment",
    }:
        raise FreezeError("missing freeze manifest")
    groups = {}
    for group, entry in binding["manifests"].items():
        p = root / safe(entry["path"])
        if sha(p.read_bytes()) != entry["sha256"]:
            raise FreezeError("freeze manifest hash mismatch")
        groups[group] = read(p)
        if groups[group]["analyzer_freeze_commit"] != commit:
            raise FreezeError("freeze commit mismatch")
    return binding, groups


def verify_binding(root, path, *, check_runtime=False):
    root = Path(root)
    binding, groups = load_binding(root, path)
    inventory_path = (root / path).parent / "artifact_sha256_manifest.json"
    if inventory_path.is_file():
        inventory = read(inventory_path)["files"]
        actual = {
            p.name
            for p in inventory_path.parent.iterdir()
            if p.is_file() and p != inventory_path
        }
        if len(inventory) != len({e["path"] for e in inventory}) or actual != {
            e["path"] for e in inventory
        }:
            raise FreezeError("freeze artifact inventory mismatch")
        for e in inventory:
            if (
                Path(safe(e["path"])).name != e["path"]
                or sha((inventory_path.parent / e["path"]).read_bytes()) != e["sha256"]
            ):
                raise FreezeError("freeze artifact SHA mismatch")
    commit = binding["analyzer_freeze_commit"]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root, check=False
    ).returncode:
        raise FreezeError("not a freeze descendant")
    frozen_paths = (
        git(root, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    )
    for group, manifest in groups.items():
        entries = manifest["files"]
        if not entries or len({e["path"] for e in entries}) != len(entries):
            raise FreezeError("empty/duplicate freeze inventory")
        for entry in entries:
            _same_file(root, commit, entry)
        if group == "frozen_analyzer" and {e["path"] for e in entries} != {
            p for p in frozen_paths if analyzer_path(p)
        }:
            raise FreezeError("analyzer inventory incomplete")
    env = groups["frozen_environment"]
    if {e["path"] for e in env["files"]} != {
        p for p in frozen_paths if p.startswith("environments/torch212_cpu/")
    }:
        raise FreezeError("environment inventory incomplete")
    for entry in groups["frozen_upstream"]["historical_snapshot_records"]:
        if sha(blob(root, entry["commit"], entry["path"])) != entry["sha256"]:
            raise FreezeError("historical snapshot hash mismatch")
    evidence_dir = groups["frozen_analyzer"].get(
        "validation_evidence_dir", "artifacts/stage_f2c_compatibility"
    )
    evidence = read(root / safe(evidence_dir) / "regression_summary.json")
    allowlist = root / "configs/f2c_historical_test_allowlist.yaml"
    if (
        evidence["unexpected_regression_count"] != 0
        or evidence["current_stage_regression_status"] != "passed"
        or sha(allowlist.read_bytes()) != evidence["allowlist_sha256"]
    ):
        raise FreezeError("regression evidence does not pass")
    for p, key in (
        ("scripts/validate_f2c_regression.py", "validation_script_sha256"),
        ("src/keyed_gram/stage_f2c_regression.py", "classifier_sha256"),
    ):
        if sha((root / p).read_bytes()) != evidence[key]:
            raise FreezeError("regression checker changed")
    if check_runtime:
        facts = runtime_facts()
        facts["cpu_operation_ok"] = bool(facts["cpu_operation_ok"])
        if facts != env["runtime_facts"]:
            raise FreezeError("runtime environment mismatch")
        interfaces = {p.name for p in Path("/sys/class/net").iterdir()}
        if interfaces != {"lo"}:
            raise FreezeError("networkless runtime required")
    return {
        "status": "passed",
        "analyzer_freeze_commit": commit,
        "frozen_upstream_manifest_valid": True,
        "frozen_analyzer_manifest_valid": True,
        "frozen_environment_manifest_valid": True,
        "runtime_environment_checked": check_runtime,
        "current_snapshot_equals_historical_f1_snapshot": False,
        "formal_development_started": False,
        "formal_locked_started": False,
        "entry_counts": {name: len(m["files"]) for name, m in groups.items()},
    }


def create_freeze(
    root, commit, output, *, evidence_dir="artifacts/stage_f2c_compatibility"
):
    root = Path(root)
    output = root / safe(output)
    if git(root, "rev-parse", "HEAD").decode().strip() != commit:
        raise FreezeError("HEAD differs from freeze commit")
    if git(root, "status", "--porcelain").strip():
        raise FreezeError("dirty worktree")
    remote = (
        git(root, "rev-parse", "refs/remotes/origin/agent/stage-f2c-symbolic-contracts")
        .decode()
        .strip()
    )
    if remote != commit:
        raise FreezeError("freeze commit has not been verified as pushed")
    evidence = root / safe(evidence_dir)
    status = read(evidence / "prefreeze_status.json")
    if (
        status["prefreeze_repair_status"] != "PASSED"
        or status["ready_for_analyzer_freeze"] is not True
    ):
        raise FreezeError("pre-freeze gates have not passed")
    source = read(evidence / "validated_source_manifest.json")
    for e in source["files"]:
        _same_file(root, commit, e)
    paths = git(root, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    selected = {p for p in paths if analyzer_path(p)}
    executable = {
        p
        for p in selected
        if p.endswith((".py", ".yaml", ".txt")) and not p.startswith("artifacts/")
    }
    if not executable <= {e["path"] for e in source["files"]}:
        raise FreezeError("untested source/config missing from validation")
    audit = read(root / "artifacts/stage_f2c_prefreeze/upstream_snapshot_audit.json")
    upstream = {e["path"] for m in audit["manifests"] for e in m["files"]}
    upstream.update(m["path"] for m in audit["manifests"])
    upstream.update(
        [
            "PHASE_F1_REPORT.md",
            "PHASE_F2A_REPORT.md",
            "PHASE_F2B_REPORT.md",
            "pyproject.toml",
        ]
    )
    environment = read(evidence / "environment_verification.json")
    if environment["networkless_environment_verified"] is not True:
        raise FreezeError("environment not verified")
    output.mkdir(parents=True, exist_ok=False)
    manifests = {
        "frozen_analyzer": {
            "files": records(root, commit, selected),
            "validation_evidence_dir": evidence_dir,
        },
        "frozen_upstream": {
            "files": records(root, commit, upstream),
            "snapshot_scope": "F2C freeze 时的现有上游文件，不声称等于 F1 历史快照",
            "historical_snapshot_records": [
                {
                    "path": ".gitattributes",
                    "commit": "ed0864b6931b779ce7eb4acb060f5662dea95bd6",
                    "sha256": audit["f1_gitattributes_at_f1_commit_sha256"],
                }
            ],
        },
        "frozen_environment": {
            "files": records(
                root,
                commit,
                [p for p in paths if p.startswith("environments/torch212_cpu/")],
            ),
            "runtime_facts": environment["runtime_facts"],
            "base_image_digest": environment["base_image_digest"],
            "image_id": environment["image_id"],
            "repo_digests": environment["repo_digests"],
            "repo_digest_available": bool(environment["repo_digests"]),
            "networkless_verification_sha256": sha(
                (evidence / "environment_verification.json").read_bytes()
            ),
        },
    }
    binding = {
        "schema_version": 1,
        "analyzer_freeze_commit": commit,
        "manifests": {},
        "formal_development_started": False,
        "formal_locked_started": False,
    }
    for name, value in manifests.items():
        value.update({"schema_version": 1, "analyzer_freeze_commit": commit})
        target = output / (name + ".json")
        write(target, value)
        binding["manifests"][name] = {
            "path": target.relative_to(root).as_posix(),
            "sha256": sha(target.read_bytes()),
        }
    write(output / "freeze_binding.json", binding)
    result = verify_binding(
        root, (output / "freeze_binding.json").relative_to(root).as_posix()
    )
    result.update(
        {
            "prefreeze_repair_status": "PASSED",
            "analyzer_freeze_status": "frozen",
            "ready_for_formal_development": True,
            "current_stage_regression_status": "passed",
        }
    )
    write(output / "protocol_status.json", result)
    write(
        output / "artifact_sha256_manifest.json",
        {
            "schema_version": 1,
            "files": [
                {"path": p.name, "sha256": sha(p.read_bytes())}
                for p in sorted(output.iterdir())
            ],
        },
    )
    return result
