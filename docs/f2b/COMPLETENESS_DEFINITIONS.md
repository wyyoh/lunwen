# 效果完备性的三级定义

## Call completeness

对有限输入域 `I` 中每个具体调用分别要求：

```text
EC_call(T,C) := ∀ c ∈ I, Effects_Q(T,c) ⊆ γ(C,c)
```

不能先对所有调用效果求并集；一个调用的声明不能“代替覆盖”另一个调用的遗漏。

## Trace completeness

```text
EC_trace(T,C) := ∀ τ ∈ Traces(T,I), α(τ) ∈ L(C)
```

`α` 保留 call binding、event 顺序、immediate/delayed phase、parent-event 因果链、
参数角色与 bounded-quiescence 终止状态。仅比较无序 effect set 不构成 trace
completeness。

## Composition completeness

```text
EC_comp(T1,…,Tn,C,P) :=
  every reachable cumulative trace is represented by C
  and every forbidden combination in trusted P is reflected by the Shield.
```

该层处理安全单步/危险后缀、异步 callback、跨工具累计效果和状态依赖 action。

## 保守完备性状态

真实程序只能返回 `VERIFIED_COMPLETE`、`COUNTEREXAMPLE_FOUND` 或 `UNKNOWN`。
F2B 的有限 synthetic evaluator 可以在全枚举域内核验前两级；真实工具外推仍禁止。
