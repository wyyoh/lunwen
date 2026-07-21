# Stage C2.4b 报告：Selective Discrete Relation Router

## 当前结论

Stage C2.4b 已定义新的选择性离散 relation router、v2 公共数据协议和人工审核门禁，但本报告生成时，真实双人独立审核尚未完成，正式 calibration、development 诊断和 `public_locked_audit_v2` 均未执行。因此目前只能报告协议、实现和 synthetic smoke 的工程状态，不能报告 R0–R4 的正式研究效果，也不能据此选择最终 router 或判断研究门槛通过。

当前状态必须作如下有限解释：

- Stage C2.4 已经证明 D2/D3 的 memory API 结构隔离成立；该历史结论不因本阶段尚未正式审计而改变。
- Stage C2.4 的主要剩余失败仍是 lexical-family OOD 离散误路由，以及 ambiguous 请求不能可靠拒绝。旧 C2.4 locked audit 已经打开，只能用于错误与数据质量分析，不能再参与 C2.4b 的模型选择、阈值选择或无偏测试。
- C2.4b 的真实 train、calibration、locked-audit 和旧 C2.4 120 条数据质量复核模板均保持 `pending`；reviewer 与 adjudication 字段没有自动填写。
- `formal_calibration_executed=false`，`formal_development_executed=false`，`formal_locked_audit_executed=false`。
- synthetic smoke 只验证控制流、schema、拒绝门禁、候选集合逻辑和产物写出；它不是自然语言模型评测，不构成 H1–H3 的研究证据，也不能用于 readiness 判定。

## 从 Stage C2.4 继承且必须保持的事实

Stage C2.4 的 D2/D3 memory 边界仅允许下列输入：

```python
retrieve_from_bucket(
    entity_embedding: torch.Tensor,
    relation_id: RelationId,
    bucket_prototypes: dict[RelationId, torch.Tensor],
) -> RetrievalResult
```

已成立的历史结构结论是：

- memory 只能看到 `entity_embedding + RelationId + 单个 relation bucket`；
- continuous relation、confidence、semantic embedding、raw text 和 phrase family 不进入 memory；
- `RejectedRoute` 不访问 memory；
- accepted route 只能访问一个 relation bucket；
- `cross_relation_candidate_count=0`。

C2.4b 不修改 entity branch、relation bucket、bucket 内检索距离或 core。它只把自然语言 relation 请求转换成三个独立 evidence score，再输出 singleton relation、ambiguous 或 unknown。上游 evidence、候选集合、margin、confidence 和文本仍不得进入 memory API。此结构边界不等于密码学安全证明，也不等于经验 leakage 为零。

## v2 数据协议

新的协议与 C2.4 locked audit 分离，按 phrase family、entity、frame、exact text、normalized text、containment、token signature 和 lemma bigram 做隔离及碰撞审计。配置的物化规模如下：

| Split | known | ambiguous | unrelated | 总行数 | 选择用途 |
|---|---:|---:|---:|---:|---|
| `public_train_v2` | 72 | 0 | 0 | 72 | 只拟合允许训练的公共轻量组件与定义原型；不选 reject threshold |
| `public_calibration_v2` | 96 | 32 | 32 | 160 | 唯一允许进行候选、聚合、threshold、margin、temperature/conformal 选择的 split |
| `public_locked_audit_v2` | 96 | 32 | 32 | 160 | 全部设置冻结且人工审核完成后的一次性正式审计 |

`public_train_v2` 每个 relation 有 8 个 phrase families，并跨 3 个 frame 物化；calibration 与 locked audit 分别有 24 个 known、8 个 ambiguous、8 个 unrelated families，并跨 4 个 frame 物化。calibration 专门包含 `key`、`identifier`、`code`、`reference`、`credential`、`token` 和 `serial` 等同词异义 hard negatives，以及 registry/city/access 的结构相似最小对。所有数据均为 answer-free，不含 private answer、entity→private value、密码学密钥材料或 confirmation 内容；这里的自然语言单词 `key` 只作为公开语义 hard negative。

本报告生成时的碰撞审计状态：`待 stage-c24b-prepare 正式产物校验后填写`。在该校验完成前，不把“配置中的新 family 名称”表述为已经通过全部历史碰撞审计。

## 人工审核状态

四份真实审核任务必须由两名独立人工 reviewer 完成，发生冲突时必须由独立 adjudication 明确裁决：

| 审核任务 | 行数 | 当前状态 | 是否可由程序自动填 reviewer |
|---|---:|---|---|
| `public_train_v2_review.csv` | 72 | `pending` | 否 |
| `public_calibration_v2_review.csv` | 160 | `pending` | 否 |
| `public_locked_audit_v2_review.csv` | 160 | `pending` | 否 |
| C2.4 旧 locked audit 独立复核 | 120 | `pending` | 否 |

程序只生成 proposed label、sample type 和空白审核字段。空 reviewer 字段、两人冲突但缺少 adjudication、label/type 不一致、审核 CSV 与数据 manifest 不一致时，validation 必须失败。旧 120 条复核只能判断 C2.4 负结果的数据质量，不能改变 C2.4 readiness，也不能把旧 locked audit 重新变成无偏测试集。

因此当前：

```text
public_benchmark_human_reviewed = false
```

在真实审核完成前，正式 calibration 和正式 audit 均不得运行；尤其不能为了得到指标而自动接受 proposed label。

## R0–R4 与冻结选择协议

候选方法严格限定为：

- R0：冻结复现 C2.4 ridge head 的三分类 argmax baseline，不重新调整旧方法；
- R1：多个公开 relation definition embedding 的独立 cosine evidence，聚合仅在 calibration 中从 mean、max、top-k mean 选择；
- R2：对每个 relation 独立计算 pairwise support，正式候选为固定 revision 的 MiniLM 小模型与 E5-base 上界候选，公共 ridge head 只允许使用 `public_train_v2` 拟合；
- R3：对独立 evidence 做 thresholded candidate set，空集拒绝为 `unknown`，单元素集合接受，多元素集合拒绝为 `ambiguous`；
- R4：在 R3 上增加严格 calibration，优先比较 class-conditional conformal prediction set。

正式选择顺序固定为：完成数据与审核 schema → 完成 train/calibration 人工审核 → 只用 train 拟合允许训练的组件 → 只用 calibration 选择 R1/R2/R3/R4 设置 → 冻结模型、参数、数据哈希、review manifest 哈希和代码提交 → development 只运行一次作诊断 → locked audit 完成人工审核后只运行一次。development 与 locked audit 不参与选择，不创建 confirmation。

如果使用 conformal，正式报告必须保存 nonconformity 定义、每个 relation 的 calibration quantile、目标 coverage 和有限样本条件。不能声称在 lexical 或语义分布迁移下仍有严格覆盖保证。

## 当前运行阻塞与资产状态

正式配置记录了模型 ID、固定 revision 和预期 SHA-256，但当前 clone 不包含以下正式运行必需资产：

- `artifacts/stage_c23/semantic_audit/S5/ridge_relation_head.pt`；
- `artifacts/stage_c23/private_answer_free_features.pt`；
- `artifacts/stage_c21/R3/canonicalizer.pt`；
- `.downloads/hf` 中固定 revision 的 BGE、MiniLM 和 E5 模型文件。

因此当前既不能真实复现 R0，也不能执行正式 semantic embedding、fact retrieval 或 source checkpoint 的运行前后字节校验。配置中已有预期 provenance 哈希不等于当前 clone 已持有对应文件。本阶段没有用临时模型、随机 embedding 或 synthetic 数值替代正式指标。

正式运行的两个独立前置条件均未满足：真实人工审核尚未完成，正式固定资产也不在当前 clone。审核未完成时应先由审核门禁拒绝运行，不应越过门禁尝试加载模型。

## R0–R4 核心结果

以下表格只保留正式结果位置。`待 smoke 运行后填写` 的工程数值即使随后生成，也必须单列为 synthetic，不能填入“正式 calibration/locked”列。

| Router | 正式 calibration 结果 | 正式 locked 结果 | lexical OOD | ambiguous reject | 研究结论 |
|---|---|---|---|---|---|
| R0 frozen argmax | 未执行 | 未执行 | 未评估 | 不具备主动弃权 | 不能下结论 |
| R1 definition ensemble | 未执行 | 未执行 | 未评估 | 未评估 | 不能下结论 |
| R2 pairwise evidence | 未执行 | 未执行 | 未评估 | 未评估 | 不能下结论 |
| R3 set-valued router | 未执行 | 未执行 | 未评估 | 未评估 | 不能下结论 |
| R4 calibrated/conformal | 未执行 | 未执行 | 未评估 | 未评估 | 不能下结论 |

Synthetic smoke：`待 smoke 运行后填写`。该值仅用于证明实现可执行，不参与 router 选择，不得与上述正式列合并。

## Router 选择与 calibration 参数

```text
selected_router = 未选择
evidence_aggregation = 未选择
relation_thresholds = 未选择
minimum_margin = 未选择
calibration_method = 未选择
conformal_alpha = 未选择
relation_calibration_quantiles = 未选择
```

在正式 `public_calibration_v2` 审核完成并运行以前，不允许从 smoke、development、C2.4 locked audit 或 C2.4b locked audit 反向填入这些值。

## 核心指标状态

| 指标 | 正式值 | 状态 |
|---|---:|---|
| known coverage | — | 未评估 |
| accepted route accuracy | — | 未评估 |
| accepted relation precision | — | 未评估 |
| family macro accuracy | — | 未评估 |
| worst-family accuracy | — | 未评估 |
| ambiguous rejection rate | — | 未评估 |
| unrelated rejection rate | — | 未评估 |
| false memory access rate | — | 未评估 |
| wrong bucket access rate | — | 未评估 |
| safe coverage | — | 未评估 |
| accepted fact Top-1 | — | 未评估 |
| all-query fact Top-1 | — | 未评估 |
| MRR | — | 未评估 |
| cross-relation candidate count | C2.4 历史值为 0；C2.4b 正式值未评估 | 保留历史结论，不伪造本阶段结果 |

Risk–coverage 曲线、AURC、candidate-set size、true-relation inclusion、unknown/ambiguous 的互相误判以及 access↔registry 定向混淆均须由正式 calibration 和一次性 locked audit 产物计算。当前没有可报告的曲线或最佳 calibration 点。

## 错误分析状态

### Lexical OOD

C2.4 已知错误集中于 access_code↔registry_id，新 family 可能受共享词 `key`、`code`、`identifier` 和 `reference` 影响。C2.4b 的 R1/R2 是否缓解该问题尚未正式评估；不能用配置中的定义文本或 synthetic smoke 推断效果，也不能声称 pairwise scorer 已优于 ridge argmax。

### Ambiguity

C2.4 的 set-free/forced-argmax 路线不能可靠拒绝 ambiguous 请求。C2.4b 已实现把零候选映射为 `unknown`、多候选映射为 `ambiguous` 的结构，但 ambiguous family 是否从错误 accept 转化为多候选 reject尚未正式评估。

### Unknown

C2.4 的 unrelated 请求相对容易拒绝；C2.4b 的 definition evidence 是否过宽、是否导致 unrelated false accept 尚未正式评估。

### Coverage trade-off

提高 ambiguous rejection 是否损害 known coverage、safe coverage 的最佳 calibration 点和完整 risk–coverage 曲线均未评估。后续正式报告不能只给一个最终 threshold，必须同时保存冻结前 calibration 的曲线与选择依据。

## 对十二个研究问题的直接回答

1. **R1 definition ensemble 是否提升 lexical-family OOD？** 未评估，不能下结论。
2. **R2 pairwise evidence 是否减少 access/registry 混淆？** 未评估，不能下结论。
3. **R3 set-valued routing 是否将 ambiguous false accept 转化为多候选 reject？** 结构逻辑已实现；正式经验效果未评估，不能下结论。
4. **R4 calibration/conformal 是否改善 safe coverage？** 未评估，不能下结论。
5. **worst-family accuracy 是否达到门槛？** 未评估，不能判定通过。
6. **ambiguous rejection 是否达到门槛？** 未评估，不能判定通过。
7. **unrelated rejection 是否保持稳定？** 未评估，不能下结论。
8. **false memory access rate 是否足够低？** 未评估，不能判定。
9. **relation bucket 的结构隔离是否完整保持？** 保持。C2.4b 复用同一 typed API，静态契约与 synthetic 控制流均证明 reject 不访问 memory、accept 只访问一个 bucket 且 `cross_relation_candidate_count=0`；该结构结论不依赖 locked 性能指标，也不代表经验零泄漏或密码学安全。
10. **当前是否具备创建新 confirmation pool 的条件？** 不具备；closed/open readiness 与人工审核条件均未满足。
11. **当前为什么仍然不能进入 C3？** 正式 calibration/locked audit 未执行，公开 benchmark 未完成人工审核，本阶段又明确禁止创建 confirmation，因此 eligibility 公式不可能成立。
12. **下一步应继续 router 研究，还是冻结协议并创建全新 confirmation？** 先完成真实双人审核、补齐固定资产，并严格按冻结协议运行 calibration 与一次性 locked audit；得到正式结果后再决定是否继续 router 研究。当前不能创建 confirmation。

## Readiness

在没有正式证据时采用 fail-closed 判定：

| 状态 | 当前值 | 原因 |
|---|---|---|
| `closed_set_selective_router_ready` | `false` | 正式 closed-set 指标未评估 |
| `open_set_abstention_ready` | `false` | 正式 ambiguous/unrelated 拒绝指标未评估 |
| `discrete_memory_contract_preserved` | `true` | 复用同一 typed API，静态契约测试与 synthetic smoke 均验证 reject 零访问、accept 单 bucket、cross-relation candidate 为 0；这不是经验零泄漏或密码学安全声明 |
| `public_benchmark_human_reviewed` | `false` | 四份审核任务均为 pending |
| `new_confirmation_pool_created_after_freeze` | `false` | 本阶段禁止创建 confirmation |
| `ready_to_create_new_confirmation_pool` | `false` | 前四项条件尚未同时成立 |
| `c3_eligible` | `false` | confirmation 条件按设计为 false，且其他 readiness 未成立 |

即使未来 closed/open、contract 和 human-review 条件全部通过，本阶段仍必须保持：

```text
new_confirmation_pool_created_after_freeze = false
c3_eligible = false
```

## 协议完整性与禁止事项

本阶段未执行下列操作：

- 没有训练 private value memory；
- 没有加载 private answer；
- 没有执行 LM residual injection 或 bounded orthogonal injection；
- 没有执行 key-value permutation、correct/wrong/partial key、fine-tuning recovery、membership inference 或其他密钥攻击；
- 没有创建、读取或运行 confirmation；
- 没有修改 core、entity checkpoint 或 relation bucket 内检索以掩盖路由错误；
- 没有根据模型预测修改人工标签；
- 没有把 ambiguous 改标为 unrelated；
- 没有声称密码学安全、机器遗忘或部署安全标准。

当前正式 locked audit 从未打开，故不存在根据其结果调参的行为。若未来发生 locked 数据被提前读取、参与选择或在冻结后被修改，应立即停止并写入新的 `protocol_incident.json`，且不得覆盖 C2.3 历史事故记录。

## 后续正式运行前置清单

1. 运行 prepare，确认 72/160/160 数据和全部历史碰撞审计，生成并冻结 manifest。
2. 由两名真人独立完成 train/calibration、locked-audit 和旧 C2.4 120 条复核；冲突项完成 adjudication。
3. 运行 review validation，并冻结 review manifest SHA-256。
4. 补齐固定 revision 模型、旧 ridge head、answer-free cache 和 entity checkpoint，逐项验证 SHA-256。
5. 只用 train/calibration 完成 R0–R4 拟合与选择，记录 frozen Git commit 和所有参数。
6. development 只运行一次作诊断，不参与选择。
7. locked audit 审核完成且所有设置冻结后只运行一次，不根据结果回调任何参数。
8. 输出严格 JSON、有限数检查、artifact SHA-256 manifest 和 source before/after hash。

## 有限表述

```text
选择性离散路由闭集门槛未通过（正式指标尚未评估）。
开放集弃权门槛未通过（正式指标尚未评估）。
离散 memory contract 保持。
ready_to_create_new_confirmation_pool = false。
c3_eligible = false。
本阶段没有训练 private memory，没有加载 private answer，没有执行答案注入或密钥攻击。
```
