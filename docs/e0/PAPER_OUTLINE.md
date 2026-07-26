# 论文提纲

## 1. 推荐题目

英文：

> **Why Learned Semantic Routers Should Not Be Authorization Boundaries:
> From External Failure to Typed Capability-Gated Memory**

中文：

> **为什么学习型语义路由不应成为授权边界：从外部负结果到类型化
> Capability-Gated Memory**

备选短标题：

> **Semantic Routing Is Not Authorization**

## 2. 一句话论文

> 学习型语义路由在交互层有用，但外部 OOS 证据不支持其承担授权；将其移出 TCB，
> 并用 typed authenticated capability 重构 memory-to-generation 数据流后，
> 单机公开/合成原型在预注册授权、故障、观测、IPC 与生命周期矩阵中保持 fail closed。

## 3. 摘要草案

LLM memory systems often rely on learned semantic routing to decide which
memory or tool a request should access. We ask whether such routing can safely
serve as an authorization boundary. Across a sequence of controlled studies,
we find that improvements in fact geometry and relation classification do not
yield stable authorization behavior. Continuous relation outputs retain
phrase-family nuisance information; discretizing them removes this channel
from the memory API but leaves lexical-OOD misrouting. A pairwise selective
router appears effective on a synthetic stress test, yet fails to reproduce
its advantage on CLINC150 and BANKING77: low false-memory-access proxies are
obtained primarily by rejecting most known requests.

Motivated by this negative result, we move semantic routing outside the
authorization trusted computing base. We introduce a typed authenticated
capability contract, bind subject, entity, relation, permission, lifecycle,
and epoch state, and gate an independently encrypted keyed-memory-to-generation
pipeline. In a frozen single-host prototype using public/synthetic values, the
pipeline passes preregistered adversarial, concurrency, crash, observability,
IPC, backup, and rollback matrices while preserving authorization-before-
lookup-before-decrypt and single-use fail-closed behavior. These results do
not establish private-data readiness, deployment security, distributed
consistency, general model confidentiality, or machine unlearning. They
support a narrower architectural principle: semantic parsing may assist
request formulation, but authorization should be enforced by typed,
authenticated capabilities outside the learned router.

## 4. 研究问题

### RQ1

学习型 query/relation 表示能否同时获得事实局部化、lexical OOD 泛化与低 nuisance？

结论：不能稳定同时获得。事实几何改善，但 relation OOD 与 phrase-family nuisance
持续存在。

### RQ2

Independent evidence 与 selective abstention 能否在外部 intent/OOS 数据上实现
可用的安全—覆盖平衡？

结论：不能。R2 未优于 R0；R3 的低 FMAR 来自 known coverage 崩溃。

### RQ3

将 semantic router 移出 TCB 后，typed authenticated capability 能否阻止
预注册未授权访问？

结论：在 D1/D2 的 public/synthetic 冻结原型矩阵内可以。

### RQ4

Capability-gated value 能否跨生成、进程、故障、观测与恢复边界保持 scoped、
single-use 和 fail closed？

结论：在 D2.1–D2.3 的单机受控原型矩阵内可以。

## 5. 章节结构

### 1. Introduction

- LLM memory/tool routing 与授权边界混淆；
- 安全分类准确率、accepted precision、coverage 和 FMAR 的区别；
- 研究问题与外部负结果；
- 从方法失败到架构重构；
- 四项贡献。

推荐在引言末尾明确：

```text
We do not claim private-memory security, deployment readiness,
cryptographic novelty, or machine unlearning.
```

### 2. Problem Formulation and Threat Model

- Natural-language proposal、authorization 与 data release 的区分；
- False Memory Access Rate、Wrong Bucket Access、Safe Coverage；
- attacker capabilities；
- TCB 与 per-request confidentiality boundary；
- out-of-scope attacks。

对应文档：[THREAT_MODEL.md](THREAT_MODEL.md)。

### 3. Why Semantic Routing Fails as Authorization

#### 3.1 Fact geometry is not relation authorization

- C2、C2.1、C2.2；
- fact/template geometry 改善；
- lexical family validation 失败。

#### 3.2 Continuous relation representations retain nuisance

- C2.3 S0 oracle；
- S5/S6 projected family probe；
- 为什么准确分类仍不是最小授权信号。

#### 3.3 Typed discretization solves interface leakage, not misrouting

- C2.4 D0–D3；
- `RelationId` + single bucket；
- cross-relation candidate count 0；
- lexical OOD 与 ambiguous reject 仍失败。

#### 3.4 Synthetic mechanism result

- C2.4b v2.1 的 AI/non-independent 边界；
- R2/R3 局部正结果；
- multi-candidate mechanism 未出现。

#### 3.5 External generalization failure

- CLINC150/BANKING77 partitions；
- R0/R2/R3/MAX；
- R2 accuracy delta；
- R3 FMAR/coverage collapse；
- 停止规则。

### 4. Architectural Reset: Typed Capability Contract

- router 移出 TCB；
- `SuggestedRelation` vs `AuthorizedMemoryRequest`；
- principal/policy/scope；
- HMAC authenticated capability；
- expiry、revocation、single-use、rotation；
- 不声称 HMAC 是算法贡献。

### 5. Capability-Gated Keyed Memory-to-Generation

#### 5.1 Independent data-key memory

- AES-256-GCM；
- AAD metadata binding；
- authorization-before-lookup-before-decrypt；
- key rotation/migration。

#### 5.2 Ephemeral generation boundary

- one value per request；
- no tools/history/cache；
- adapter vs generator trust；
- output commit。

#### 5.3 Isolated local services

- gateway/memory/generator processes；
- AF_UNIX minimal IPC；
- SQLite replay/revocation；
- crash semantics。

#### 5.4 Fault, observability and lifecycle

- five milestones；
- logs/metrics/traces filters；
- byte-level IPC；
- state/policy/gateway epochs；
- rollback detection and limitations。

### 6. Evaluation

#### 6.1 Router studies

- C2 internal sequence；
- C2.5 external datasets/protocol；
- known/OOS/access-control metrics；
- risk–coverage。

#### 6.2 System matrices

- D1–D2.3 scenario taxonomy；
- positive and negative invariants；
- artifact leakage scans；
- source/hash preservation。

#### 6.3 Results

- external router failure；
- 25/37/40/35/51 matrix results；
- all scope-specific zero violations；
- do not aggregate heterogeneous scenarios into probability。

### 7. Discussion

- semantic parsing vs authorization；
- accepted precision without coverage；
- fail closed vs reject-all；
- why standard crypto is necessary but not the novelty；
- TCB size and trusted generation adapter；
- system negative results as design evidence。

### 8. Limitations and Threats to Validity

- synthetic/private-data boundary；
- single host；
- mock IAM；
- same-TCB HMAC；
- same-host anchor；
- no remote LLM/APM/GPU；
- deterministic fault injection；
- no memory zeroization；
- internal prototype matrices；
- AI-reviewed synthetic benchmark limitation。

### 9. Related Work

只在正式写作时补充并核验以下类别：

- intent classification and OOS detection；
- selective prediction and abstention；
- capability-based security；
- confidential/authorized memory systems；
- tool authorization for LLM agents；
- provenance and information-flow control；
- fault-tolerant single-use credentials；
- prompt injection and model data exfiltration。

本提纲不预填未经核验的引文。

### 10. Conclusion

以架构原则结束：

> Semantic parsing may assist request formulation, but authorization must be
> enforced by typed, authenticated capabilities outside the learned router.

## 6. 贡献结构

1. **系统性负结果**：learned semantic router 无法稳定承担授权。
2. **可信边界重构**：router 移出 TCB，typed capability 绑定显式 scope。
3. **Keyed memory-to-generation pipeline**：授权、lookup、AEAD、生成严格排序。
4. **预注册系统审计**：授权、并发、故障、观测、IPC、备份与回滚。

## 7. 推荐图表

### Figure 1：研究证据链

```text
C2 negative evidence → architectural reset → D-series boundary audit
```

### Figure 2：最终 TCB

直接使用 [tcb_architecture.mmd](tcb_architecture.mmd)。

### Figure 3：Risk–coverage

使用 C2.5 `risk_coverage.csv`，对比 R3 与 MAX threshold，突出 coverage collapse。

### Table 1：C2 router results

使用 [EXPERIMENT_MASTER_TABLE.md](EXPERIMENT_MASTER_TABLE.md) 的 C2.5 数值。

### Table 2：Claim–evidence map

精简 [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md)。

### Table 3：D-series attack matrices

报告各阶段场景数、主要零违规和限制，不汇总为安全概率。

### Table 4：TCB materials

gateway/memory/generator 的 key/token/plaintext 分布。

## 8. 审稿前检查

- 每个“passed”是否带 stage scope；
- C2.4 provisional、C2.4c exploratory 与 C2.5 external 是否明确区分；
- 是否避免把 FMAR proxy 写成真实事故率；
- 是否同时报告 known coverage 与 safe coverage；
- 是否说明 multi-candidate 机制未获支持；
- 是否避免密码算法创新表述；
- 是否明确 public/synthetic-only；
- 是否保留 `c3_eligible=false`；
- 是否把所有零计数限定到已扫描位置和预注册场景；
- 是否给出 code/data/model revision 与 SHA-256。
