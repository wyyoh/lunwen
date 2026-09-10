"""仅临时 Git fixture；不创建真实 Analyzer freeze 或实验输出。"""

import subprocess

import pytest

from keyed_gram.stage_f2c_freeze import (
    FreezeError,
    analyzer_path,
    create_freeze,
    load_binding,
    read,
    records,
    safe,
    sha,
    verify_binding,
    write,
)


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path

    def put(p, text):
        file = root / p
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")

    put("configs/f2c_historical_test_allowlist.yaml", "test fixture only\n")
    put("src/keyed_gram/stage_f2c_regression.py", "# fixture\n")
    put("scripts/validate_f2c_regression.py", "# fixture\n")
    put("environments/torch212_cpu/Dockerfile", "# fixture\n")
    put("historical.txt", "old snapshot\n")
    evidence = {
        "unexpected_regression_count": 0,
        "current_stage_regression_status": "passed",
        "allowlist_sha256": sha(
            (root / "configs/f2c_historical_test_allowlist.yaml").read_bytes()
        ),
        "validation_script_sha256": sha(
            (root / "scripts/validate_f2c_regression.py").read_bytes()
        ),
        "classifier_sha256": sha(
            (root / "src/keyed_gram/stage_f2c_regression.py").read_bytes()
        ),
    }
    d = root / "artifacts/stage_f2c_compatibility"
    d.mkdir(parents=True)
    write(d / "regression_summary.json", evidence)
    for args in (
        ["init"],
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
        ["add", "."],
        ["commit", "-m", "fixture"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True
    ).splitlines()
    output = root / "artifacts/stage_f2c_freeze"
    output.mkdir()
    groups = {
        "frozen_analyzer": {
            "files": records(root, commit, [p for p in paths if analyzer_path(p)])
        },
        "frozen_upstream": {
            "files": records(root, commit, ["historical.txt"]),
            "historical_snapshot_records": [
                {
                    "path": "historical.txt",
                    "commit": commit,
                    "sha256": sha(b"old snapshot\n"),
                }
            ],
        },
        "frozen_environment": {
            "files": records(root, commit, ["environments/torch212_cpu/Dockerfile"]),
            "runtime_facts": {},
        },
    }
    binding = {
        "schema_version": 1,
        "analyzer_freeze_commit": commit,
        "manifests": {},
        "formal_development_started": False,
        "formal_locked_started": False,
    }
    for name, value in groups.items():
        value["analyzer_freeze_commit"] = commit
        target = output / (name + ".json")
        write(target, value)
        binding["manifests"][name] = {
            "path": target.relative_to(root).as_posix(),
            "sha256": sha(target.read_bytes()),
        }
    write(output / "freeze_binding.json", binding)
    return root, "artifacts/stage_f2c_freeze/freeze_binding.json", commit


def test_commit_bound_manifests_verify(frozen):
    root, path, _ = frozen
    result = verify_binding(root, path)
    assert result["frozen_environment_manifest_valid"]
    assert result["formal_locked_started"] is False


@pytest.mark.parametrize(
    "path",
    [
        "scripts/validate_f2c_regression.py",
        "environments/torch212_cpu/Dockerfile",
        "historical.txt",
    ],
)
def test_source_environment_or_upstream_mutation_rejected(frozen, path):
    root, binding, _ = frozen
    (root / path).write_text("changed\n")
    with pytest.raises(FreezeError, match="source mismatch"):
        verify_binding(root, binding)


def test_manifest_hash_tampering_rejected(frozen):
    root, path, _ = frozen
    (root / "artifacts/stage_f2c_freeze/frozen_analyzer.json").write_text("{}")
    with pytest.raises(FreezeError, match="manifest hash"):
        verify_binding(root, path)


def test_missing_environment_manifest_rejected(frozen):
    import json

    root, path, _ = frozen
    binding = read(root / path)
    del binding["manifests"]["frozen_environment"]
    (root / path).write_text(json.dumps(binding))
    with pytest.raises(FreezeError, match="missing"):
        load_binding(root, path)


def test_formal_started_binding_rejected(frozen):
    import json

    root, path, _ = frozen
    binding = read(root / path)
    binding["formal_locked_started"] = True
    (root / path).write_text(json.dumps(binding))
    with pytest.raises(FreezeError, match="schema"):
        load_binding(root, path)


def test_creation_refuses_dirty_worktree(frozen):
    root, _, commit = frozen
    with pytest.raises(FreezeError, match="dirty"):
        create_freeze(root, commit, "other-freeze")


@pytest.mark.parametrize("value", ["../a", "/a", "a//b", "C:/a", "a\\b"])
def test_freeze_paths_fail_closed(value):
    with pytest.raises(FreezeError):
        safe(value)


def test_freeze_inventory_extra_file_rejected(frozen):
    root, path, _ = frozen
    directory = (root / path).parent
    write(
        directory / "artifact_sha256_manifest.json",
        {
            "files": [
                {"path": p.name, "sha256": sha(p.read_bytes())}
                for p in directory.iterdir()
            ]
        },
    )
    (directory / "extra.json").write_text("{}")
    with pytest.raises(FreezeError, match="inventory"):
        verify_binding(root, path)


def test_metadata_duplicate_json_key_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(FreezeError, match="duplicate"):
        read(p)
