"""F2C Verifier/Evaluator world；Analyzer 禁止导入。

纯结构审计不需要加载求解器；只有显式请求 verifier 才导入 Z3。
"""

from importlib import import_module

_MODULES = {
    "BoundedSymbolicVerifier": ".equivalence",
    "GuardedTransition": ".hidden_ir",
    "HiddenSymbolicCase": ".hidden_ir",
    "HiddenSymbolicImplementation": ".hidden_ir",
    "SymbolicSandboxReplay": ".replay_service",
}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "BoundedSymbolicVerifier",
    "GuardedTransition",
    "HiddenSymbolicCase",
    "HiddenSymbolicImplementation",
    "SymbolicSandboxReplay",
]
