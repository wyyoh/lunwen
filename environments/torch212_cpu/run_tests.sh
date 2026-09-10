#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python environments/torch212_cpu/verify_environment.py
python -m pytest -q -p no:cacheprovider \
  tests/test_model.py \
  tests/test_authsynth_shared_schema.py \
  tests/test_authsynth_dependency_boundary.py \
  tests/test_authsynth_blind_cgar.py \
  tests/test_tool_effect_ir.py \
  tests/test_tool_effect_contracts.py \
  tests/test_tool_effect_shield.py
python -m ruff check --no-cache \
  environments/torch212_cpu/verify_environment.py \
  environments/torch212_cpu/generate_environment_manifest.py
