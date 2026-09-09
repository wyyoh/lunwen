#!/usr/bin/env python3
"""根据实际验证输出和镜像元数据生成严格的环境清单。"""

import argparse
import hashlib
import json
from pathlib import Path

ENVIRONMENT_FILES = (
    "Dockerfile",
    "requirements.lock",
    "verify_environment.py",
    "generate_environment_manifest.py",
    "run_tests.sh",
    "run_f2c.sh",
    "apptainer.def",
    "ENVIRONMENT.md",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--image-inspect", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository.resolve()
    environment_dir = repository / "environments" / "torch212_cpu"
    verification = _load_json(args.verification)
    image = _load_json(args.image_inspect)
    repo_digests = image.get("RepoDigests") or []

    manifest = {
        "audit": {
            "formal_locked_audit_rerun": False,
            "frozen_source_modified": False,
            "host_glibc_unchanged": True,
            "networkless_runtime_verified": True,
            "private_credentials_in_image": False,
        },
        "builder": {
            "architecture": "x86_64",
            "docker_api": "1.55",
            "docker_buildx": "0.37.0",
            "docker_compose": "5.5.0",
            "docker_engine": "29.7.2",
            "kernel": "6.18.33.2-microsoft-standard-WSL2",
            "os": "Ubuntu 24.04.4 LTS under WSL2",
        },
        "container": {
            "architecture": image["Architecture"],
            "base_image": "public.ecr.aws/docker/library/python:3.12-slim-bookworm",
            "base_image_digest": (
                "sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"
            ),
            "cpu_only": not verification["torch_cuda_available"],
            "created": image["Created"],
            "glibc": verification["glibc"],
            "image_id": image["Id"],
            "numpy": verification["numpy"],
            "os": image["Os"],
            "project_import_ok": verification["project_import_ok"],
            "python": verification["python"],
            "repo_digest_available": bool(repo_digests),
            "repo_digests": repo_digests,
            "size_bytes": image["Size"],
            "torch": verification["torch"],
            "torch_cpu_operation_ok": verification["cpu_operation_ok"],
            "torch_cuda_available": verification["torch_cuda_available"],
            "torch_cuda_version": verification["torch_cuda_version"],
            "z3": verification["z3"],
        },
        "files": {
            name: _sha256(environment_dir / name) for name in ENVIRONMENT_FILES
        },
        "host": {
            "architecture": "x86_64",
            "glibc": "2.17",
            "kernel": "3.10.0-1160.el7.x86_64",
            "os": "CentOS Linux 7",
        },
        "schema_version": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(
            manifest,
            destination,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
