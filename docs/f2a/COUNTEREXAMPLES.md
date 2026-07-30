# 六个区分性最小反例

这些反例区分“单独的数据 taint”与“单独的 token subset check”。它们不证明
所有既有组合系统都无法扩展解决；FLAM/FLAC、线性授权逻辑和 APPA 是后续必须
正面对比的强基线。

## CE1 Authority laundering

```text
untrusted memory
→ LLM summary
→ trusted memory container
→ privileged tool
```

失败点：summary 或容器身份覆盖原始 provenance，然后系统把“trusted text”当
authority。

- 纯数据 IFC：正确逐值 taint 可阻止，但不能单独定义 effect grant origin。
- 普通 capability：无 token 时阻止；预持宽 token 时不判断攻击数据能否选择消费。
- tracked capability：追踪 effect 可达性，不自动绑定 data influence eligibility。
- APPA：可隔离分支污染，但不定义线性 authority 来源。
- DAIFC：rewrite 保留 `InfluenceOrigin`；只有 authenticated issue 创生 authority。

## CE2 Capability recombination

```text
branch A: read customer/7
branch B: send email/external
merge: send customer/7 externally
```

两个单步都可能合法，但 combined effect 被 policy 禁止。

- 纯数据 IFC：在 confidentiality 明确时可阻止泄漏；不一般表达 purpose/history。
- 普通 capability：两个 token 逐调用均合法。
- tracked capability：两个 capture 均可合法出现在程序中。
- APPA：能保护 parent label，但 authority union 仍需另行约束。
- DAIFC：merge 验证 lineage、总预算和 combined-effect deny。

## CE3 Delegation fork

```text
parent budget = one use
delegate full grant to child A
delegate full grant to child B
both commit
```

- 数据 IFC 不表示消费；
- 可复制 bearer token 会失败；
- 有状态 single-use token 可解决最简单实例，但不自动定义 split、子预算和深度；
- capture checking 不等于 affine ownership；
- DAIFC 的 delegate 消费父 resource；多 child 必须先 split。

## CE4 Trusted-tool echo

```text
untrusted page
→ trusted parser echoes text
→ LLM treats echo as command
→ privileged effect
```

关键区别：

```text
trusted tool implementation
≠ trusted origin of every output byte
```

DAIFC 保留输入 influence。即使显式提高完整性，authority 的 `InfluenceGuard`
仍独立检查来源类别和 attacker-controlled 标志。

## CE5 Memory fragmentation

```text
fragment 1: "用户"
fragment 2: "已经批准转账"
LLM fusion: "用户已经批准转账"
commit
```

文本融合可以产生新命题，不能产生新 authenticated grant。普通 provenance graph
可说明来源，但若执行层没有 authority origin 类型，仍可能把 provenance 结论当
authorization。

## CE6 High-integrity fact is not authority

```text
signed fact: invoice total = 100
LLM: therefore pay invoice
payment commit
```

签名只支持事实完整性，不证明签名者有权批准支付，也不证明当前用户授权这次支付。

```text
high-integrity fact
≠ authorized command
```

DAIFC 要求事实标签、grant origin、effect scope 和 influence eligibility 分别通过。

## 有界状态反例

`authority_flow.explorer` 在深度 4 内探索两组状态机：

### Laundering

不安全最短 trace：

```text
llm_summary
→ trusted_tool_echo_marks_trusted
→ derive_authority_from_integrity
```

安全语义没有 data/integrity 到 authority 的 transition，因此同一 bound 内找不到
origination violation。

### Naive merge

不安全最短 trace：

```text
naive_branch_copy
→ branch_0_consume
→ branch_1_consume
```

初始预算为 1，却产生两个 committed effects。安全语义只能做
`linear_branch_partition=(1,0)`，因此找不到 overspend。

## 反例通过标准

```text
distinguishing_counterexample_count >= 6
all not_plain_taint_only = true
all not_token_subset_only = true
unsafe bounded model finds laundering = true
safe bounded model finds laundering = false
unsafe bounded model finds naive merge = true
safe bounded model finds naive merge = false
```

其中 “not reducible” 是对单机制 baseline 的操作性判断，不是对所有既有理论的
不可表达性证明。
