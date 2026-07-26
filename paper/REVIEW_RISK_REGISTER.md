# 审稿风险预演

## 使用方式

本文件不是给正文增加辩护性措辞，而是检查主张是否被证据支持。投稿前应逐项确认：

- 正文已经主动承认限制；
- 回应不依赖未运行的实验；
- 不为降低审稿风险而修改冻结指标或数据；
- 无法回答的问题明确列为 future engineering 或 out of scope。

## R1：“只是把标准安全机制拼在一起”

**风险等级：高。**

可能的质疑：

> HMAC、AES-GCM、capability、SQLite 和 Unix socket 都是标准技术，系统贡献在哪里？

有限回应：

1. 论文不把这些组件列为密码学或算法创新；
2. 核心实证贡献是 learned router 授权 analogue 的外部负结果；
3. 核心系统贡献是由停止条件驱动的 TCB 重构；
4. 评估对象是贯穿 `authorization → lookup → decrypt → generation → delivery`
   的顺序、不变量与故障语义；
5. 标准机制是实现该边界的必要材料，不是贡献本身。

不得回应：

- “我们发明了新 capability”；
- “组合本身自动构成安全证明”；
- “51/51 证明系统安全”。

## R2：“D 系列只有 synthetic value，意义有限”

**风险等级：高。**

有限回应：

1. 这是事实限制，正文必须承认；
2. C2.5 已表明授权边界未可靠前不应引入真实私有数据；
3. 高熵 canary 足以审计控制平面、作用域、顺序和应用层持久化；
4. D 系列只解锁了 public/synthetic prototype 状态，没有解锁 private data；
5. 真实 IAM/KMS、独立测试和治理是后续工程前置条件，不是论文已完成部分。

不得回应：

- “synthetic 与 private 完全等价”；
- “零 canary 泄漏意味着真实秘密不会泄漏”。

## R3：“全部自定义攻击矩阵通过，是否只是测试写成了会通过”

**风险等级：高。**

有限回应：

1. 矩阵在代码冻结和正式执行前预注册；
2. 每阶段的 source/artifact hash 与一次性正式运行边界可审计；
3. 正向、负向、并发、故障、IPC 和恢复场景分别验证显式不变量；
4. 论文不将异构场景求和或估计安全概率；
5. 它仍是内部原型审计，不是独立渗透测试或形式化证明。

改稿动作：

- 正文始终使用 “within the preregistered matrix”；
- appendix 给出完整 scenario definitions 和 artifact 路径；
- 在 limitations 中明确 scenario completeness 未证明。

## R4：“为什么不直接使用 ACL/RBAC/ABAC？”

**风险等级：高。**

有限回应：

```text
identity + ACL/RBAC/ABAC policy decision
→ scoped authenticated capability
→ typed resource enforcement
```

Capability 不取代 policy。它将外部 policy 决定绑定到 subject/entity/relation/action 和
lifecycle，并在 memory service 处提供可验证的最小凭据。原型的 mock policy 只是让
该边界可测试。

## R5：“CaMeL 已经使用 capability 防 prompt injection，本文是否缺乏新意？”

**风险等级：高。**

有限回应：

- 必须把 CaMeL 作为最接近的直接相关工作；
- CaMeL 从 agent control/data-flow 分离出发，目标是 prompt-injection defense；
- 本文首先实证检验 learned semantic routing 作为 authorization 的适用性；
- 本文的负结果触发停止规则和架构重构；
- 系统评估聚焦 typed keyed memory-to-generation 的 replay、encryption、crash、
  observability、IPC、backup 和 rollback；
- 本文不声称 CaMeL 的可证明 prompt-injection 性质。

需要在投稿前再次检查 CaMeL 是否已有正式出版版本。

## R6：“两个 intent benchmark 不能证明所有 semantic router 都失败”

**风险等级：高。**

有限回应：

- 同意，论文不提出形式化不可能性；
- 支持的结论是 tested R2/R3 的 synthetic advantage 未外部复现；
- 两个 benchmark 分别覆盖跨领域 OOS 和单领域细粒度 held-out intent；
- 每个使用十个冻结 partitions，结果在 continuation gate 上一致失败；
- 架构建议还依赖 security design principle：统计语义不是授权凭据。

标题下应持续保留 “should not”，不要写成 “cannot ever”。

## R7：“R0/R2 FMAR=100% 是人为构造，是否夸大 baseline？”

**风险等级：中。**

有限回应：

- R0/R2 定义为 forced acceptance，所以 FMAR=100% 是行为定义；
- 它们用于隔离 known-routing 质量，不是合理 open-set deployment baseline；
- MAX threshold 是实际选择性简单基线；
- 论文明确称 FMAR 为 hypothetical bucket-access proxy；
- 核心比较是 MAX 与 R3 的 coverage/FMAR trade-off，而不是只比较 R0 与 R3。

## R8：“R3 threshold 在 calibration 已表现低 coverage，为什么还运行 test？”

**风险等级：中。**

有限回应：

- 预注册选择规则在没有满足全部 gate 的候选时最大化通过 gate 数并降低 calibration
  FMAR；
- 该规则在 test 前冻结；
- test 运行的目的之一就是检验这种选择性机制能否外部延续；
- 结果支持停止，而不是事后换 threshold。

不得在稿件阶段重新选择 threshold。

## R9：“Typed API 不等于 information-flow security”

**风险等级：中。**

有限回应：

- 正文明确区分结构性数据最小化与 IFC/noninterference；
- continuous relation、confidence 和 raw text 不进入 memory 是 source-level fact；
- Python 类型不能提供语言级强制隔离；
- Myers/Liskov 与 JFlow 被列为更强的相关方向。

## R10：“G1 是 constrained copy，不是 LM confidentiality”

**风险等级：高。**

有限回应：

- 完全同意；
- G1 只验证真实模型加载/generate 调用、一次性 context、无工具、无 cache 和参数冻结；
- 输出被限制为当前 canary token 序列；
- 普通自由生成、远程 API、provider logging、模型记忆和 GPU cache 均未验证。

若 venue 页数紧张，宁可把 G1 降到 appendix，也不能夸大。

## R11：“single-use 是否等于 exactly-once？”

**风险等级：中。**

有限回应：

- 不等于；
- 原型提供 at-most-one plaintext release；
- response crash 时 delivery 可以不确定；
- capability 不恢复，重试需要新授权；
- RIFL 等工作解决更强的 exactly-once RPC 问题；
- SQLite 证据只适用于单主机。

## R12：“同主机 monotonic anchor 不能防强回滚攻击”

**风险等级：高。**

有限回应：

- 正文主动承认；
- 当前机制只检测主状态回滚而 anchor 保持当前的情况；
- 同时回滚 state 与 anchor 无法检测；
- 需要外部可信计数器、KMS/HSM 或远程 transparency service。

## R13：“为什么不继续做真实 IAM/KMS 或 private data？”

**风险等级：中。**

有限回应：

- 论文问题已经由 C2 负结果和 D 原型边界回答；
- 继续扩展会混合系统研究与产品工程；
- real IAM/KMS 是未来独立 engineering demonstration；
- private data 需要新的伦理、治理、权限和独立安全审查；
- 当前项目状态明确禁止。

## R14：“阶段太多，正文像项目日志”

**风险等级：高。**

改稿动作：

- C2/C2.1/C2.2 压缩为 Section 3.1；
- C2.3/C2.4/C2.5 构成负结果三步；
- D1–D2.3 合并为一个架构和四类不变量；
- 正文只保留 3 图、3 表；
- 逐阶段参数和矩阵放 appendix；
- 避免在每节重复 readiness 布尔值。

## 最终主张检查

投稿版本必须保留：

> These results do not establish private-data readiness, deployment security,
> distributed consistency, model confidentiality, cryptographic security, or
> machine unlearning.

禁止出现：

- “secure private memory”；
- “deployment-ready access control”；
- “provably secure”；
- “zero leakage”而不带扫描范围；
- “188 attacks”；
- “human-reviewed v2.1”；
- “conformal guarantee under distribution shift”；
- “LLM resists prompt injection”。
