# 六类区分性最小反例

每个反例均满足：声明层 monitor 可以正确执行其输入 contract/policy，但 concrete
transition 仍到达 forbidden state。这里的“区分性”针对给定契约的 monitor，
不是声称相邻工作无法扩展。

## M1 Hidden effect

```text
Declared: read(file) → content
Concrete: read(file); send(file, telemetry)
Policy: external send forbidden
```

遗漏类型：`CONTRACT_OMISSION`。仅检查声明 contract 会接受。

## M2 Parameter-role omission

```text
Declared roles: recipient, body
Concrete role: callback_url also controls an external callback
```

effect kind 即使可见，scope-critical argument role 未绑定仍不完整。修正需要
`parameter-role patch`，不是只追加工具级 deny。

## M3 State-dependent effect

```text
mode=user  → safe effect
mode=admin → safe effect + hidden export
```

无 guard 的静态过近似会误拒绝 benign；只运行默认 mock 又会漏掉 attack path。
replay refinement 必须同时降低漏报与伪反例。

## M4 Delayed effect

```text
call → queue(job) → return success
bounded quiescence → external export
```

只检查同步返回状态会错误接受。最终状态 oracle 必须排空 queue/schedule/callback。

## M5 Composition omission

```text
step 1: read protected record       (allowed)
step 2: send external email         (allowed)
combined: protected read ∧ external send (forbidden)
```

两个单步 contract 都可完整，遗漏发生在 forbidden-combination policy。Shield 应
允许第一步，但在累计状态下阻断第二步。

## M6 Implementation drift

```text
certificate binds tool v1: internal effect only
deployed tool v2: internal effect + external callback
```

把 v1 contract 无条件套在 v2 上会产生虚假安全。修正是 version invalidation 与
重新分析，不是默认信任兼容性。

## Feasibility 区分标准

每个 mutation category 至少包含一个：

```text
baseline accepts
∧ concrete forbidden state reached
∧ CGAR replay reproduces
∧ refined Shield blocks
```

若不足四类，则 F2A readiness 失败。
