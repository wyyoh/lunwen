"""Keyed-GRAM research prototype.

包级公开 API 使用惰性导入，避免 D2.2 的隔离服务进程仅因导入子模块就加载
Torch、训练代码或模型栈。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AuxPermutationKey",
    "ExperimentConfig",
    "GramModelConfig",
    "GramTransformer",
    "apply_key",
    "generate_key",
    "hash_auxiliary",
]


_EXPORT_MODULES = {
    "AuxPermutationKey": ".keying",
    "ExperimentConfig": ".config",
    "GramModelConfig": ".config",
    "GramTransformer": ".model",
    "apply_key": ".keying",
    "generate_key": ".keying",
    "hash_auxiliary": ".keying",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
