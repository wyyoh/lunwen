# AuthCap / Intent–Authority Separation

## 研究问题

本路线系统地区分三种对象：

```text
SemanticProposal（不可信候选意图）
PolicyGrant（确定性策略允许的最大 authority）
Executable Capability（F2 以后才可能编译）
```

核心命题是：任何 learned proposal 都不能扩大显式 policy 授予的 authority。
F0–F1 只预注册协议、构造 benchmark 和验证 deterministic oracle，不实现最终
compiler，不执行 memory/tool，不签发 capability，也不训练 learned baseline。

## 阶段

```text
F0  协议、性质、schema 与评价预注册
F1  AuthZRouteBench 与 deterministic policy oracle
F2  Intent–Authority Separation compiler
F3  形式模型与 trace refinement
F4  真实 agent integration 与外部评价
F5  论文综合
```

本轮只允许 F0–F1。进入 F2 必须由 F1 readiness gate 决定。

## 六项安全性质

- **P1 Proposal Non-Authority**：proposal 不能触发 lookup、decrypt、tool 或
  plaintext release。
- **P2 Authority Non-Amplification**：未来 compiled result 必须是
  `PolicyGrant.maximum_authority` 的子集。F1 只实现可复用 subset checker。
- **P3 Explicit Deny Dominance**：winning priority/specificity tier 中 deny
  优先；只有显式更高 priority 才是合法 override。
- **P4 Scope Integrity**：subject、tenant、resource、action、relation、
  purpose 和 policy epoch 均不可扩张。
- **P5 Policy Freshness**：旧 epoch/hash 的 decision 与 witness 不自动有效。
- **P6 Compositional Non-Amplification**：多步 authority 的并集不得超过
  policy 定义的组合闭包。

## Benchmark 与 ground truth

AuthZRouteBench 使用企业知识库、Agent 工具和多租户 memory 三个公开合成域。
自然语言表面形式由单模型 AI 辅助设计并由确定性模板生成：

```text
benchmark_text_generation = AI_assisted
policy_ground_truth = deterministic_oracle
independent_human_validation = false
```

AI 不决定 allow/deny、maximum authority、policy hash、decision ID 或 witness。
完整生成数据位于 Git 忽略的 `data/authzroutebench/generated/`；仓库提交生成器、
schema、split hash、分布、碰撞审计与复现清单，不提交数据集本体。

## 既有边界

F 系列不修改冻结的 C2、D1–D2.3 实验。以下状态永久不变：

```text
additional_d_series_stages_allowed = false
d_series_status = completed
semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```
