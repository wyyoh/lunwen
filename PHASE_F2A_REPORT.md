# Stage F2A：Data–Authority Formal Semantics Freeze

## 结论

F2A 已把“双轨 data–authority flow”冻结成可执行的最小语义假设，并通过六个
区分性反例和两个有界状态反例进入下一步：

```text
status = passed_formal_problem_freeze
new_semantic_object_defined = true
data_and_authority_semantically_distinct = true

authority_origination_property_defined = true
authority_conservation_property_defined = true
non_malleable_authority_property_defined = true
merge_confinement_property_defined = true

distinguishing_counterexample_count = 6
bounded_model_counterexample_for_naive_merge = found
bounded_model_counterexample_for_authority_laundering = found

novelty_gate_status = conditional_pass_high_prior_work_overlap
ccf_b_novelty_established = false
ready_for_executable_authority_calculus = true
```

该 readiness 只表示研究问题足够明确、可证伪，允许实现 F2B；不表示新颖性、
定理或论文质量已经成立。

## 1. 原 F2 与 F1 状态

```text
original_f2_capability_compiler_plan =
superseded_by_dual_data_authority_flow_research

original_f2_implementation_evidence_preserved = true
```

原 F2 仍保存在 `agent/stage-f2-authcap-compiler` 的
`ab8d314d99d025f29b47c7133a92fea05778573a`，未删除、修改或重跑。F2A 从
F1 `ed0864b6931b779ce7eb4acb060f5662dea95bd6` 创建。

F1 source manifest 的 23 个文件和完整 artifact inventory 已重新验证；四 split
data SHA-256 未变化。F2A 只读取 manifest 元数据：

```text
f1_frozen_assets_modified = false
locked_test_case_content_read = false
locked_test_scored = false
```

## 2. 新增形式对象

### Data flow

```text
DataLabel(
  confidentiality,
  integrity,
  InfluenceOrigin*
)
```

LLM、summary、memory 和 tool output 可以产生或传播数据影响。数据 join 取更严格
confidentiality、更低 integrity，并保留 influence origins。

### Authority flow

```text
AuthorityResource(
  AuthorityOrigin,
  AuthorityAtom,
  AuthorityBudget,
  lineage
)
```

`AuthorityAtom` 绑定 issuer/subject/tenant/resource/action/purpose/epoch 和
`InfluenceGuard`。`InfluenceGuard` 回答“哪些数据来源有资格控制这次 authority
消费”，避免把“持有合法 token”误写成“任意攻击数据都可选择如何使用 token”。

Authority 只能由 authenticated grant 创生。LLM、memory、tool output、summary、
integrity label 或文本 approval claim 都没有 issuance rule。

## 3. Authority algebra

已实现：

- `issue_authenticated`：唯一 origin root；
- `split`：子预算之和不超过父预算，父资源被消费；
- `delegate`：线性转移，depth 严格衰减；
- `consume`：exact scope、InfluenceGuard 与 budget 同时检查；
- `merge_branches`：共同 parent、唯一 resource、per-origin 守恒和 combined deny；
- `endorse`：只提高数据 integrity，保留 influence，不产生 authority。

Budget 第一版覆盖 uses、amount、data bytes、resource count、delegation depth 和
TTL。F2B 必须为每个 effect 固定单位，不允许跨单位相加。

## 4. 反例

六类反例：

1. authority laundering；
2. capability recombination；
3. delegation fork；
4. trusted-tool echo；
5. memory fragmentation；
6. high-integrity fact / authorized-command confusion。

它们说明单独的数据 taint 或单独的 token subset check 均不足。该结论不等于所有
既有组合系统无法表达；完整矩阵见
[COUNTEREXAMPLES.md](docs/f2a/COUNTEREXAMPLES.md)。

有界探索 depth 为 4：

```text
unsafe laundering:
  llm_summary
  → trusted_tool_echo_marks_trusted
  → derive_authority_from_integrity

unsafe merge:
  naive_branch_copy
  → branch_0_consume
  → branch_1_consume
```

安全 transition system 在相同 bound 内均未找到对应违规。

## 5. 四个性质

### Authority Origination

每个 effect resource 必须沿 lineage 追溯到 authenticated external grant。

### Authority Conservation

对每个 origin：

```text
live descendant budget + spent budget <= initial grant budget
```

scope、epoch 与 delegation depth 只能保持或收紧。

### Non-Malleable Authority

攻击者数据变化不得改变 authority supply trace。effect use trace 只有在
`InfluenceGuard` 显式允许该来源时才可变化，并始终受 scope/budget 限制。

这里特意没有要求所有 effect trace 对攻击数据完全相同，因为合法的公开数据驱动
任务需要在明确授权范围内选择不同 effect。完整双运行义务见
[THEOREM_OBLIGATIONS.md](docs/f2a/THEOREM_OBLIGATIONS.md)。

### Merge Confinement

安全 merge 不是 union 或 intersection，而是 lineage、预算、已消费 effect 和
combined deny 上的受约束重组。

## 6. Novelty gate

F2A 发现高 prior-work overlap：

- FLAM/FLAC 已联合建模 IFC 与动态授权；
- 线性授权逻辑已表达 consumable credential；
- NMIFC 已提出 4-safety non-malleability；
- FIDES/LLMbda 已形式化 Agent 数据 IFC；
- TACIT 已追踪 Agent capability；
- APPA 已提出 context branching 和 merge confinement。

因此：

```text
historically_first_authority_object = false
prior_work_overlap_risk = high
ccf_b_novelty_established = false
```

可继续的候选贡献被限定为：

> LLM Agent 中显式分开的 data influence 与线性 effect authority、控制
> capability 消费的数据来源资格、authority supply 的双运行不可塑性，以及跨
> 分支 authority recombination discipline。

F3 前必须尝试到 FLAC/线性授权逻辑的编码。如果可以直接无损编码，研究应降级为
LLM Agent 的系统实例化，而不是宣称新 calculus。

## 7. 验证

```text
F2A 定向测试 = 27 passed
全仓测试 = 541 passed
Ruff = passed
git diff --check = passed
```

全仓测试在 bookworm 兼容运行时完成；宿主机 glibc 2.17 无法加载冻结环境中的
Torch/NumPy wheel，因此使用已有 bookworm rootfs 的动态加载器。被忽略的测试
launcher 不进入 Git 或 artifact。

## 8. 未完成

```text
executable_authority_flow_ir_completed = false
formal_model_completed = false
core_theorems_mechanized = 0
real_agent_integration_completed = false
authflowbench_completed = false
private_data_used = false

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

F2A 没有接真实 LLM/Agent、没有签发 capability、没有执行 tool/memory，也没有
加载 private answer/value。

## 9. 下一步

允许进入 F2B，但必须先保持以下停止条件：

- 如果 authority 只是 integrity label 的重命名，停止；
- 如果反例能由不变的现有 FLAC 编码直接覆盖，收窄贡献；
- 如果 non-malleability 无法给出双运行语义，停止；
- 如果安全只能靠拒绝复杂任务，停止；
- 如果需要 LLM judge 决定 authority origin，停止。
