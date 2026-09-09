from __future__ import annotations

import json
import os
import platform
import sys

import numpy
import torch
import z3


def _verify_cpu_kernel() -> bool:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    right = torch.tensor([[2.0, 0.0], [1.0, 2.0]], device="cpu")
    expected = torch.tensor([[4.0, 4.0], [10.0, 8.0]], device="cpu")
    return bool(torch.equal(left @ right, expected))


def main() -> int:
    requested_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
    if requested_threads < 1:
        raise ValueError("TORCH_NUM_THREADS 必须为正整数")
    torch.set_num_threads(requested_threads)

    import keyed_gram  # noqa: F401

    cuda_available = torch.cuda.is_available()
    cpu_operation_ok = _verify_cpu_kernel()
    if cuda_available:
        raise RuntimeError("正式 CPU 环境检测到可用 CUDA")
    if not cpu_operation_ok:
        raise RuntimeError("PyTorch CPU 原生张量运算失败")

    libc_name, libc_version = platform.libc_ver()
    result = {
        "architecture": platform.machine(),
        "cpu_operation_ok": cpu_operation_ok,
        "glibc": libc_version if libc_name == "glibc" else f"{libc_name} {libc_version}",
        "kernel": platform.release(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "project_import_ok": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "z3": z3.get_version_string(),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
