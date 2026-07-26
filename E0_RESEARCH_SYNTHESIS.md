# Stage E0：Research Synthesis and Paper Construction

## 执行结论

本项目的系统实验阶段在 Stage D2.3 正式停止。当前最可靠的研究结论不是
“学习型路由器实现了私有记忆授权”，而是：

> 学习型语义路由可以辅助用户表达意图，但不应成为授权边界。外部 OOS 实验证明，
> 语义分类、置信度与选择性弃权无法稳定同时满足错误访问率与合法覆盖率；因此系统
> 将 router 移出授权 TCB，改由类型化、经过认证且绑定作用域的 capability 控制
> keyed memory-to-generation 数据流。

项目已经形成如下证据链：

```text
C2–C2.2：query/fact geometry 可改善，但 relation lexical OOD 不稳定
      ↓
C2.3：关系分类可准确，连续 relation 输出仍携带 phrase-family nuisance
      ↓
C2.4：RelationId + 单 bucket 切断连续 relation 到 memory 的结构通道
      ↓
C2.4c：合成压力集上 pairwise evidence/abstention 有局部效果
      ↓
C2.5：CLINC150/BANKING77 外部审计否定该效果的稳定泛化
      ↓
D1：semantic router 移出授权 TCB；引入 typed authenticated capability
      ↓
D2：capability-gated AES-GCM keyed memory
      ↓
D2.1：授权 synthetic value 的一次性生成边界
      ↓
D2.2：多进程服务隔离、最小 IPC 与持久 replay 状态
      ↓
D2.3：故障、可观测性、IPC、生命周期与状态回滚审计
```

正式项目状态固定为：

```text
additional_d_series_stages_allowed = false
d_series_status = completed
d_series_system_audit_complete = true

semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

不再创建 D2.4/D2.5，不再扩展语义授权 router，不加载 private answer，不创建
confirmation，不执行 key attack 或 LM fine-tuning。

## 1. 中心研究问题

最终论文应回答：

> 学习型语义路由能否作为 LLM 私有记忆的授权边界？如果不能，什么类型的系统边界
> 能够在公开/合成 value 的冻结原型中，明确保证授权先于 lookup、decrypt 与生成？

对应的英文表述：

> Can learned semantic routers serve as authorization boundaries for
> LLM memory, and if not, how should the trusted boundary be reconstructed?

推荐标题：

> **Why Learned Semantic Routers Should Not Be Authorization Boundaries:
> From External Failure to Typed Capability-Gated Memory**

该中心问题同时容纳两类证据：

1. C2 系列的系统性负结果；
2. D 系列由负结果驱动的可信边界重构。

## 2. C2 系列：负结果与结构性发现

### 2.1 表示几何改善不等于关系授权可靠

[Stage C2](PHASE_C2_REPORT.md) 通过 entity span、多位置 pooling 与 fact-level
SupCon，将严格跨模板 1-NN 从 7.81% 提升到 59.38%，fact/template margin 从
−0.341 翻转为 +0.517。然而 relation probe 只有 63.02%，没有达到 85% 门槛。

[Stage C2.1](PHASE_C21_REPORT.md) 和 [Stage C2.2](PHASE_C22_REPORT.md)
进一步显示：

- relation SupCon 与分支级 adversary 只带来有限改善；
- public train relation probe 可以达到 100%，但完全不重叠的 public validation
  probe 只有 37.50%–41.67%；
- 轻量 relation head 主要记忆词族，而不是形成稳定的 lexical semantics。

因此：

```text
better fact geometry
≠ stable relation understanding
≠ authorization readiness
```

### 2.2 连续 relation 输出仍携带 nuisance

[Stage C2.3](PHASE_C23_REPORT.md) 的 ground-truth relation oracle 在
validation/development 上通过 10/10 门槛，说明 relation ID 正确时，冻结 entity
branch 与检索几何足以支持目标槽位结构。

但小模型 S5 与 E5-base S6 均只失败于 `projected_family_probe`。即使 relation
分类准确，三维连续 relation softmax 的形状仍可预测同一 relation 内的
phrase family。这支持有限结论：

> 连续 relation 表示不仅表达离散关系类别，也携带与授权无关的措辞家族信息。

这不证明输出契约是唯一因果，也不是普遍的信息泄漏定理。

### 2.3 Typed bucket 的结构隔离成立

[Stage C2.4](PHASE_C24_REPORT.md) 将 memory contract 固定为：

```text
entity_embedding
+ RelationId
+ exactly one selected relation bucket
```

memory 不接收：

- continuous relation；
- confidence/logits/probabilities；
- semantic embedding；
- raw text；
- phrase family。

D2/D3 的 `cross_relation_candidate_count=0`，被拒请求的 memory access 为 0。
这是结构性隔离结论，不等于经验泄漏为零、密码学安全或授权正确。

同一阶段的 provisional locked audit 仍显示：

- family macro accuracy：81.94%；
- worst-family accuracy：25.00%；
- known coverage：72.22%；
- ambiguous false accept：33.33%。

因此 typed bucket 解决的是“进入 memory 的接口形状”，没有解决离散误路由。

### 2.4 合成压力集上的局部正结果不能外推

[Stage C2.4c](PHASE_C24C_REPORT.md) 是
`exploratory_non_independent`：同一单模型 AI 参与数据修订并看过 locked 文本，
没有真人审核或独立外部验证。

在该合成压力集上：

- R2 known accuracy/safe coverage 为 100%；
- R3 将 overall FMAR 从 100% 降到 25%；
- R3 known coverage 与 safe coverage 为 95.83%；
- R3 ambiguous rejection 为 87.5%，unrelated rejection 为 62.5%；
- R3 没有形成 multi-relation candidate set，拒绝主要来自 empty-set unknown。

该结果只支持任务特定的机制探索，不能作为外部泛化或正式授权证据。

### 2.5 外部 OOS 审计否定 router 泛化

[Stage C2.5](PHASE_C25_REPORT.md) 使用 CLINC150 与 BANKING77 官方数据、
官方 split 和 10 个预注册 supported-intent partitions，一次性运行 test：

| 协议 | R2−R0 known accuracy | R3 FMAR | R3 safe coverage |
|---|---:|---:|---:|
| CLINC150 open intent | −1.57 pp | 1.22% | 25.83% |
| BANKING77 open intent | −3.33 pp | 0.33% | 17.74% |
| BANKING77 closed set | −3.83 pp | 不适用 | 不适用 |

R2 不差于 R0 的 seed 数分别为 1/10、0/10、0/10。R3 的 multi-candidate
rate 只有 0.41% 和 0.03%；低 FMAR 主要来自拒绝绝大多数 known。

简单 MAX threshold 提供更合理的 coverage/FMAR trade-off，但也没有跨两个基准
稳定满足逐 seed continuation gate。

因此正式停止状态为：

```text
external_oos_validation_status = failed
external_pairwise_evidence_validated = false
external_selective_abstention_validated = false
semantic_router_authorization_research = stopped
```

## 3. D 系列：可信边界重构

### 3.1 D1：认证 capability

[Stage D1](PHASE_D1_REPORT.md) 将 `SuggestedRelation` 降级为不可信建议。
只有包含显式 subject、entity、RelationId 与 capability token 的
`AuthorizedMemoryRequest` 可以进入 gateway。

5 个正向和 20 个负向场景全部通过：

```text
unauthorized_memory_access_count = 0
cross_relation_access_count = 0
cross_entity_access_count = 0
rejected_request_memory_access_count = 0
```

### 3.2 D2：capability-gated encrypted keyed memory

[Stage D2](PHASE_D2_REPORT.md) 分离 capability HMAC key 与 AES-256-GCM data key，
并将 record ID、entity、relation、data-key ID、record version、schema 和 algorithm
绑定到 AEAD AAD。

37/37 场景通过：

```text
unauthorized_lookup = 0
unauthorized_decrypt = 0
unauthorized_plaintext_release = 0
cross_subject/entity/relation release = 0
concurrent_double_release = 0
```

### 3.3 D2.1：一次性生成边界

[Stage D2.1](PHASE_D21_REPORT.md) 验证 D2 授权释放的单个 synthetic value
进入一次性生成 adapter：

- 40/40 场景通过；
- 23 次预期精确交付全部完成；
- 未授权 generator invocation 为 0；
- 跨 subject/entity/relation/session 暴露为 0；
- prompt injection scope expansion 为 0；
- 应用层 artifact/log/exception/cache plaintext occurrence 为 0。

G1 使用受限 token 解码，只验证模型调用链、context、无工具和参数冻结，不证明一般
自由生成模型的保密性或 prompt-injection 抵抗能力。

### 3.4 D2.2：多进程服务边界

[Stage D2.2](PHASE_D22_REPORT.md) 将 gateway、memory 与 generator 拆为本地
spawn 进程和 AF_UNIX IPC，并将 replay/revocation 状态迁移到 SQLite：

- 35/35 场景通过；
- unauthorized service/memory access 为 0；
- cross-instance/post-restart replay 为 0；
- cross-process double release 为 0；
- capability key、data key 与 capability token 未跨越禁止边界；
- 运行期 104 个 canary、728 个变体、302 个位置扫描未发现应用层持久化。

### 3.5 D2.3：最终系统边界审计

[Stage D2.3](PHASE_D23_REPORT.md) 覆盖 51 个预注册场景：

| 类别 | 场景 | 通过 |
|---|---:|---:|
| 授权端到端 | 3 | 3 |
| 可观测性 | 12 | 12 |
| 故障一致性 | 13 | 13 |
| IPC/本地端点 | 15 | 15 |
| 状态备份/恢复 | 8 | 8 |

关键结果均为 0：

- plaintext observability occurrence；
- secret metric/trace/error occurrence；
- crash/timeout 后双重 release；
- restore 后 replay；
- state rollback acceptance；
- unauthorized local socket connection；
- stale/cross-epoch ticket acceptance；
- malformed IPC plaintext release。

D2.3 只支持单机本地原型结论。磁盘满与时钟漂移为确定性故障注入；
monotonic anchor 仍是同主机文件。

## 4. 最终可信数据流

```text
AuthenticatedPrincipal
        ↓
explicit AuthorizedMemoryRequest
        ↓
capability verification + policy/scope checks
        ↓
atomic single-use consumption
        ↓
epoch-bound internal release ticket
        ↓
exact (entity_id, RelationId) lookup
        ↓
independent data-key AEAD decrypt
        ↓
single ephemeral synthetic value
        ↓
minimal generation adapter
        ↓
output commit → response delivery
```

生命周期固定为：

```text
capability_consumed
→ record_decrypted
→ generator_invoked
→ output_committed
→ response_delivered
```

任何不确定状态不恢复原 capability。

完整 TCB 和信任边界见
[TCB_ARCHITECTURE.md](docs/e0/TCB_ARCHITECTURE.md) 与
[tcb_architecture.mmd](docs/e0/tcb_architecture.mmd)。

## 5. 可以与不可以提出的主张

### 可以提出

> Learned semantic routers are useful for interaction, but unreliable as
> authorization boundaries.

> In a frozen prototype using public/synthetic values, typed authenticated
> capabilities maintained authorization-before-access, bounded scope,
> single-use semantics, and fail-closed behavior across preregistered
> adversarial, fault, observability, and lifecycle matrices.

### 不可以提出

- 实现了安全的 private memory；
- 实现了机器遗忘；
- 提供部署级访问控制；
- 提供密码学安全证明；
- 一般 LLM 能保守处理秘密；
- 分布式 replay/rollback 已解决；
- 第三方模型、日志或 tracing 不会保存输入；
- 通过 D 系列即可进入原 C3。

完整主张等级见
[CLAIM_EVIDENCE_MATRIX.md](docs/e0/CLAIM_EVIDENCE_MATRIX.md)。

## 6. 尚未验证的边界

- 真实 IAM、用户会话与主体认证；
- KMS/HSM 和签发—验证的独立密码学隔离；
- 跨主机一致性、分布式 replay/revocation 与 Byzantine failure；
- 第三方 APM、云日志、远程 tracing、dead-letter queue；
- 远程 LLM、供应商数据保留和 GPU cache；
- root/kernel compromise、进程内存取证与安全清零；
- timing/cache/功耗等 side channel；
- 渗透测试与形式化密码学证明；
- 任何真实 private value/private answer。

尤其需要保留：

> monotonic anchor 与主状态仍在同一主机。若攻击者同时回滚数据库和 anchor，
> 当前原型无法检测；需要外部可信单调计数器、KMS/HSM 或等价控制。

## 7. E0 交付与终止条件

E0 只整理文档：

- [威胁模型](docs/e0/THREAT_MODEL.md)
- [TCB 架构](docs/e0/TCB_ARCHITECTURE.md)
- [主张—证据矩阵](docs/e0/CLAIM_EVIDENCE_MATRIX.md)
- [论文提纲](docs/e0/PAPER_OUTLINE.md)
- [实验总表](docs/e0/EXPERIMENT_MASTER_TABLE.md)
- [可复现性清单](docs/e0/REPRODUCIBILITY_CHECKLIST.md)
- [阶段状态注册表](docs/e0/STAGE_STATUS_REGISTRY.json)

本阶段不创建新实验、不修改冻结代码或产物、不改变任何 readiness gate。
