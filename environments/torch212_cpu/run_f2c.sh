#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ $# -eq 0 ]]; then
  echo "用法: $0 <显式的 F2C 命令及参数>" >&2
  echo "该脚本不会自动启动 development 或 locked audit。" >&2
  exit 64
fi

python environments/torch212_cpu/verify_environment.py
exec "$@"
