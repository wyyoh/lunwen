# DAIFC：Dual Data–Authority Information-Flow Control

## 研究重定位

Stage F2A 将后续研究问题从“把 proposal 与 policy grant 求交并签发 capability”
收窄并升级为：

> 在包含 LLM、长期记忆、工具、分支和委托的系统中，分别追踪数据如何影响推理，
> 以及外部 grant 如何正当化 effect；攻击者控制的数据可以改变普通计算结果，但
> 不能创生、复制、扩大或洗白 authority。

状态定义如下：

```text
original_f2_capability_compiler_plan =
superseded_by_dual_data_authority_flow_research

original_f2_implementation_evidence_preserved = true
```

“superseded”只改变论文主线。原 F2 的实现和一次性结果继续保存在
`agent/stage-f2-authcap-compiler`，不会删除、回写或重跑。F2A 从冻结 F1 提交
`ed0864b6931b779ce7eb4acb060f5662dea95bd6` 单独分叉。

## 核心分离

F2A 区分两个不能互换的平面：

```text
Data plane
  DataLabel = (confidentiality, integrity, InfluenceOrigin*)
  回答：哪些值影响了结果、值可流向何处。

Authority plane
  AuthorityResource = (AuthorityOrigin, AuthorityAtom, Budget, Lineage)
  回答：谁授权 effect、允许什么、剩多少、如何合法分裂和委托。
```

核心命题为：

```text
Influence ≠ Authority
high integrity ≠ authorized command
LLM output ≠ authenticated grant
```

每个 effect 还携带 `InfluenceGuard`：即使 agent 持有合法 authority，控制该次
消费的数据来源也必须满足显式资格约束。这样避免把“有 token”误写成“任何
攻击者数据都可选择如何使用 token”。

## F2A 范围

本阶段只完成：

1. 最小数据标签和 authority resource algebra；
2. `issue/split/delegate/consume/merge/endorse` 语义；
3. authority origination、conservation、non-malleability 和 merge
   confinement 的定理义务；
4. 六个区分性最小反例；
5. 两个不安全语义的有界状态反例；
6. novelty gate 与既有工作的高重叠风险审计。

本阶段不完成：

- 完整可执行 IR 或真实 tool/memory；
- LLM、AgentDojo、LangGraph 或 MCP 集成；
- capability 签名；
- F1 baseline/locked scoring；
- Alloy、TLA+、Lean/Rocq 证明；
- private data、private value 或真实凭据。

## 阶段路线

```text
F1 deterministic policy oracle（冻结）
        ↓
F2A 双轨语义、反例与定理义务
        ↓
F2B executable AuthorityFlow IR
        ↓
F3 Alloy/TLA+/机械化核心证明与 trace refinement
        ↓
F4 AuthFlowBench + 至少两个真实 Agent runtime
        ↓
F5 论文与 artifact
```

## 四个目标性质

### Authority Origination

任何执行 effect 使用的 authority 必须沿 lineage 追溯到 authenticated external
grant。LLM、memory、tool output、summary 或多 Agent 投票不能创生 authority。

### Authority Conservation

对每个 origin，所有存活后代预算与已消费预算之和不得超过初始 grant；scope、
epoch 和 delegation depth 只能保持或收紧。

### Non-Malleable Authority

改变攻击者控制的数据不改变 authority supply trace。effect 选择只有在相应
`InfluenceGuard` 明确允许该来源时才可变化。可信 elevation 是唯一显式例外，
且 elevation 本身必须不受攻击者控制。

### Merge Confinement

分支只能持有通过线性 split 获得的互异 authority resource。merge 检查共同
parent、lineage、剩余预算、已消费预算和 combined-effect deny；不能简单 union。

## 新颖性边界

F2A 不声称首次提出 authority、capability、线性授权、IFC 与授权结合、
non-malleability 或 merge confinement。已知直接重叠包括：

- FLAM/FLAC 对信息流与动态授权的联合建模；
- 线性授权逻辑和 consumable credential；
- NMIFC 的 robust declassification/transparent endorsement；
- FIDES/LLMbda 的 Agent IFC；
- TACIT 的 tracked capability；
- APPA 的上下文分支与 merge confinement。

因此 novelty gate 只能是：

```text
novelty_gate_status = conditional_pass_high_prior_work_overlap
ccf_b_novelty_established = false
```

后续必须证明不可约组合贡献：LLM Agent 中显式分开的 InfluenceOrigin 与线性
AuthorityOrigin、数据影响资格、双运行 authority non-malleability，以及跨分支
authority recombination discipline。若 F2B/F3 显示这些可以由 FLAC 或现有
线性授权逻辑直接无损编码，本方向应停止或降级为系统实例化。

## 永久边界

```text
additional_d_series_stages_allowed = false
d_series_status = completed
semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```
