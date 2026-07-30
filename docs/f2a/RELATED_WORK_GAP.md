# Related Work Gap 与不可约差异审计

## 审计结论

F2A 的研究空白不是“以前没有 authority”或“以前没有 IFC + authorization”。
这两种说法均不成立。可辩护的候选空白更窄：

> 现有 Agent IFC 主要追踪数据标签；现有 capability/授权系统主要约束 effect
> 可达性；F2A 尝试在同一 Agent 运行语义中，把数据影响与线性 effect authority
> 保持为两个正交对象，并约束攻击者数据能否选择 authority 的消费、跨分支重组和
> 委托。

该判断是待 F2B–F4 验证的研究假设，不是已经成立的论文创新。

## 直接相关工作

### FIDES

[Securing AI Agents with Information-Flow Control](https://arxiv.org/abs/2505.23643)
为 Agent planner 跟踪 confidentiality/integrity label，刻画动态 taint
可执行的性质及安全—效用边界，并通过选择性隐藏提高任务可用性。

差异候选：

- FIDES 的核心对象是数据 label；
- F2A 另建线性 `AuthorityResource`，记录 origin、effect scope、预算与 lineage；
- F2A 的 `InfluenceGuard` 约束“哪些数据可控制一次合法 authority 的消费”。

重要限制：FIDES policy 完全可以扩展历史谓词或 action policy，故不能仅凭一个
工程反例断言 FIDES 原理上无法表达 authority。后续比较必须固定“纯数据标签
baseline”的能力边界。

### LLMbda

[The LLMbda Calculus](https://arxiv.org/abs/2602.20064) 给出带 conversation、
LLM invocation 与动态 IFC 的 call-by-value calculus，并证明终止不敏感的
confidentiality/integrity noninterference。

差异候选：

- LLMbda 对 conversation influence 建模更直接；
- F2A 关注从 authenticated grant 到 effect 的线性 authority trace；
- F2A 需要 split/delegate/consume/merge 的预算与来源守恒。

F2B 应优先尝试将 authority resource 作为 LLMbda 的线性 effect 环境扩展，而不是
重新设计完整 conversation calculus。

### TACIT / tracked capabilities

[Securing Agents With Tracked Capabilities](https://doi.org/10.1145/3786335.3813127)
使用 Scala 3 capture checking 和 capability-safe 子集追踪 Agent 生成代码的
effect capability，并以 classified wrapper 实现 local purity。

差异候选：

- capture set 回答计算捕获了哪些 capability；
- F2A 还要求 consumable budget、origin lineage、跨子 Agent affine ownership；
- 数据完整性本身不授予使用已捕获 capability 的资格。

高风险重叠：若 Scala capture checking 加上 affine type、history policy 和 taint
wrapper 即可直接表达全部 F2A 规则，则贡献可能只是组合实现。F3 前必须给出编码
比较。

### APPA

[Agentic Permissions Policy Algebra](https://arxiv.org/abs/2607.24625) 是
2026-07-27 发布的预印本，使用 engine-managed context branching、
prospective acquisition enforcement、label 与 shared event log 两个 monoid，
并主张 parent label preservation 和 merge confinement。

差异候选：

- APPA 的 branch 目标是限制 taint 对父 context 的污染；
- F2A 的 merge 同时检查线性 authority ownership、origin budget 与
  combined-effect deny；
- F2A 不允许分支的 authority 简单 union，即便 data label merge 合法。

这是当前最直接的新颖性风险之一。不能在论文中把“branch/merge confinement”
本身作为首创。

### FLAM、FLAC 与 FLAFOL

[FLAM](https://www.cs.cornell.edu/andru/papers/flam/) 在 CSF 2015 已联合建模
authorization 与 information flow，提出 robust authorization；
[FLAC](https://arxiv.org/abs/2104.10379) 进一步建模动态计算产生的 authority、
delegation/hand-off 与信息流约束；
[FLAFOL](https://arxiv.org/abs/2001.10630) 则把授权逻辑和信息流
noninterference 放入一阶逻辑并在 Coq 中证明。

这组工作否定了“IFC 与 authority 从未被联合建模”的主张。F2A 与 FLAM 的方向
甚至相反：

```text
FLAM：统一 principals / trust / information-flow policy
F2A：操作语义中显式区分 data influence 与 consumable effect authority
```

可能的增量是 LLM Agent 特有的：

- LLM 输出可任意改变数据，但 authority supply 不变；
- capability 已存在时，攻击数据是否有资格选择其消费；
- 长期 memory、分支和多 Agent 委托中的线性重组；
- 对 authority supply trace 的双运行非可塑性。

在 F3 之前必须尝试把 F2A 编码进 FLAC。若无本质障碍，论文定位应改为
“LLM Agent 的 FLAC/线性授权实例化与 benchmark”，而非新 calculus。

### Nonmalleable IFC

[Nonmalleable Information Flow Control](https://www.cs.cornell.edu/andru/papers/nmifc/)
在 CCS 2017 将 robust declassification 与 transparent endorsement 统一为
4-safety hyperproperty。它控制攻击者对 confidentiality/integrity downgrade 的
塑造。

F2A 不复用“non-malleable”名称来暗示同一定理。候选性质作用于另一个投影：

- NMIFC：数据 downgrade/endorsement trace；
- F2A：authenticated authority 的 issue/split/delegate/consume trace。

两者应在 F3 中给出严格关系：蕴含、正交或可编码，而不能只做文字类比。

### 线性授权与 consumable credentials

[A Linear Logic of Authorization and Knowledge](https://doi.org/10.1007/11863908_19)
已用线性逻辑表示 consumable authorization/resource；
[Consumable Credentials](https://www.cs.cmu.edu/~fp/papers/ndss07.pdf)
实现了分布式 use-limited credential 消费。

因此 budget、single-use 和“不能复制”都不是新概念。F2A 的候选贡献必须来自其与
LLM data influence、origin eligibility 和 branch merge 的联合语义。

## 不可约差异的操作性判据

F2B 前保留以下三项判据：

1. 仅给数据加 taint，不增加 authority resource，不能表达 delegation fork 的预算守恒。
2. 仅做 token scope/subset check，不增加 influence eligibility，不能区分
   “高完整性事实”和“受认证命令”，也不能表达攻击数据对合法 token 消费的塑造。
3. 仅做 context branch isolation，不增加线性 lineage/combined-effect check，
   不能阻止两个合法分支 capability 的重组。

这些判据只区分简化基线，不证明所有既有组合系统都做不到。

## Novelty gate

```text
prior_work_overlap_risk = high
new_semantic_object_defined_within_project = true
historically_first_authority_object = false
ccf_b_novelty_established = false
ready_for_executable_authority_calculus = true
```

允许进入 F2B 的理由是：问题已被定义成可证伪的组合假设，且存在最小反例；
不是因为相关工作空白已经被证明。
