"""AuthSynth 有限 ToolEffectIR、CGAR 与精确安全 Shield。"""

from .contracts import ContractCheck, check_effect_completeness
from .ir import execute_trace, quiescent_effects
from .shield import FiniteSafetyGame, Shield, synthesize_maximal_shield
from .types import (
    DeclaredContract,
    EffectAtom,
    EffectKind,
    EffectPhase,
    EffectSelector,
    EffectTemplate,
    Environment,
    ForbiddenRule,
    ToolCall,
    ToolImplementation,
    ToolTransition,
    TraceScenario,
    TraceStep,
)

__all__ = [
    "ContractCheck",
    "DeclaredContract",
    "EffectAtom",
    "EffectKind",
    "EffectPhase",
    "EffectSelector",
    "EffectTemplate",
    "Environment",
    "FiniteSafetyGame",
    "ForbiddenRule",
    "Shield",
    "ToolCall",
    "ToolImplementation",
    "ToolTransition",
    "TraceScenario",
    "TraceStep",
    "check_effect_completeness",
    "execute_trace",
    "quiescent_effects",
    "synthesize_maximal_shield",
]
