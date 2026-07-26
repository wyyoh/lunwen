# 实验总表

## 1. C2 系列

| Stage | 核心问题 | 数据/协议 | 主要结果 | 决策 |
|---|---|---|---|---|
| C2 | Query canonicalization 能否从模板主导转为事实主导 | 48 facts；6/2/4 套隔离模板 | Q3 1-NN 59.38%，fact/template margin +0.517；relation probe 63.02% | 结构部分成功，不进入 C3 |
| C2.1 | Relation SupCon/adversary 能否修复 lexical OOD | R0–R4；validation 选择，development 诊断 | 选中 R3；development relation head/probe 69.79%/65.62%，4/10 gates | 未通过 |
| C2.2 | Answer-free public paraphrase supervision 能否跨词族迁移 | train/validation entity、frame、phrase family 隔离 | P2 public train probe 100%，public val probe 41.67%，development 63.02% | 未通过，转公开语义 encoder |
| C2.3 | 公开 semantic encoder 是否解决 relation，又是否携带 nuisance | S0 oracle；S2–S6；answer-free | S0 10/10；S5/S6 relation 高，但均失败 projected-family probe | 不进入 memory；改离散 contract |
| C2.4 | Typed RelationId/single bucket 是否切断连续 relation 到 memory | D0–D3；provisional locked | cross-relation candidates 0；family macro 81.94%，worst 25%；open guard 2/7 | 结构隔离通过，router readiness 失败 |
| C2.4b v2.1 | 修正合成 benchmark 与 AI 审核边界 | 单模型 AI；无真人审核 | 104 families/392 rows；37 boundary flags；未运行模型评分 | 仅探索，冻结 v2.1 |
| C2.4c | Pairwise/set-valued router 在 v2.1 是否有效 | exploratory non-independent；R0–R4 | R2 known 100%；R3 FMAR 25%、safe coverage 95.83%；multi-set=0 | 有局部机制价值，需外部验证 |
| C2.5 | 上述机制能否外部泛化 | CLINC150 + BANKING77；10 partitions；test once | R2 均弱于 R0；R3 safe coverage 25.83%/17.74%；multi-rate 0.41%/0.03% | 外部验证失败，停止 router 分支 |

## 2. C2.5 外部结果

### Open-intent/OOS

| Benchmark | Variant | Known acc. | Known coverage | Accepted route acc. | Safe coverage | Wrong bucket | FMAR |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | R0 | 98.01% | 100.00% | 98.01% | 98.01% | 1.99% | 100.00% |
| CLINC150 | R2 | 96.44% | 100.00% | 96.44% | 96.44% | 3.56% | 100.00% |
| CLINC150 | R3 | 25.83% | 25.89% | 99.63% | 25.83% | 0.07% | 1.22% |
| BANKING77 | R0 | 94.08% | 100.00% | 94.08% | 94.08% | 5.92% | 100.00% |
| BANKING77 | R2 | 90.75% | 100.00% | 90.75% | 90.75% | 9.25% | 100.00% |
| BANKING77 | R3 | 17.74% | 17.90% | 99.10% | 17.74% | 0.16% | 0.33% |

R0/R2 强制接受所有 unknown，因此 FMAR 为 100%；该值是“若存在对应 intent bucket
则会访问”的 access proxy，不是已部署 private-memory 事故率。

### Closed-set

| Benchmark | R0 accuracy | R2 accuracy | R2−R0 | R2 不差于 R0 seeds |
|---|---:|---:|---:|---:|
| CLINC150 supported intents | 98.01% | 96.44% | −1.57 pp | 1/10 |
| BANKING77 open supported intents | 94.08% | 90.75% | −3.33 pp | 0/10 |
| BANKING77 all 77 intents | 89.32% | 85.49% | −3.83 pp | 0/10 |

## 3. D 系列

| Stage | 原型边界 | 场景 | 正向结果 | 关键零违规 | 下一状态 |
|---|---|---:|---|---|---|
| D1 | Typed authenticated capability | 25 | 5 authorized accesses | unauthorized/cross-scope/rejected memory access = 0 | ready for synthetic D2 |
| D2 | Capability-gated AES-GCM keyed memory | 37 | 10 plaintext releases；rotation/migration 各 1 | unauthorized lookup/decrypt/release = 0 | ready for D2.1 |
| D2.1 | Ephemeral generation adapter | 40 | 23 exact deliveries | unauthorized invocation/exposure、cross-request leak = 0 | ready for D2.2 |
| D2.2 | AF_UNIX multi-process services + SQLite replay state | 35 | 6 exact deliveries | unauthorized service access、double release、restart replay = 0 | ready for D2.3 |
| D2.3 | Fault/observability/IPC/lifecycle/rollback | 51 | 3 exact deliveries | observability leak、post-restore replay、rollback acceptance = 0 | D series complete |

## 4. 协议等级

| Stage | Selection 隔离 | Test/locked 状态 | 证据定位 |
|---|---|---|---|
| C2–C2.3 | validation 选择；development 诊断 | 内部合成/answer-free | 表示与机制研究 |
| C2.4 | public calibration 选择 | provisional locked；无人审 | 结构合约 + provisional 性能 |
| C2.4b v2.1 | 只做 AI 数据审核 | 未评分 | benchmark 设计记录 |
| C2.4c | calibration 选择，development/locked 不反调 | exploratory non-independent | 合成压力机制案例 |
| C2.5 | official train/validation 选择 | official test once | 外部 OOS 负结果 |
| D1–D2.3 | 预注册攻击矩阵，代码冻结后正式运行一次 | public/synthetic prototype | 内部系统边界审计 |

## 5. Readiness 总表

```text
external_oos_validation_status = failed
external_pairwise_evidence_validated = false
external_selective_abstention_validated = false

trusted_capability_contract_status = passed
capability_gated_keyed_memory_status = passed
capability_gated_ephemeral_generation_status = passed
isolated_service_generation_status = passed
fault_observability_lifecycle_status = passed

d_series_system_audit_complete = true
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

“D 系列 passed”只表示各自限定原型矩阵通过，不能覆盖 C2 的 task-specific router
readiness，也不能解锁 private data。
