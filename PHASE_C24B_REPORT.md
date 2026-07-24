# Stage C2.4b 报告：Selective Discrete Relation Router

> **2026-07-22 协议修订：** 本文以下内容保留为原 v2 实现与 smoke 的历史记录。
> 在任何 router/model 评分前，用户提供的单模型 AI 数据质量审核发现 4 个错标
> family、16/16 ambiguous 显式提示捷径和过易 unrelated 构造，因此
> `c24b-public-selective-router-v2` 已永久降级为 `diagnostic_only`，旧 v2 的
> review validation、formal calibration 与 formal locked audit 入口均已撤销。
> 项目不再等待或声称双人真人审核；替代协议是独立命名空间的 v2.1 单模型 AI
> 探索性审核。它始终保持 `public_benchmark_human_reviewed=false`、
> `independent_external_validation=false` 与 `formal_calibration_allowed=false`。
> 当前权威状态与产物见 `PHASE_C24B_V21_REPORT.md`；本历史报告中的“后续真人审核”
> 不再是现行计划。

## 当前结论

Stage C2.4b 已实现新的选择性离散 relation router、v2 公共数据协议和人工审核门禁，并已完成正式 `prepare` 与独立 CPU synthetic smoke。但本报告生成时，真实双人独立审核尚未完成，正式 calibration、development 诊断和 `public_locked_audit_v2` 均未执行。因此目前只能报告协议、实现和 synthetic smoke 的工程状态，不能报告 R0–R4 的正式研究效果，也不能据此选择正式 router 或判断研究门槛通过。

当前状态必须作如下有限解释：

- Stage C2.4 已经证明 D2/D3 的 memory API 结构隔离成立；该历史结论不因本阶段尚未正式审计而改变。
- Stage C2.4 的主要剩余失败仍是 lexical-family OOD 离散误路由，以及 ambiguous 请求不能可靠拒绝。旧 C2.4 locked audit 已经打开，只能用于错误与数据质量分析，不能再参与 C2.4b 的模型选择、阈值选择或无偏测试。
- C2.4b 的真实 train、calibration、locked-audit 和旧 C2.4 120 条数据质量复核模板均保持 `pending`；reviewer 与 adjudication 字段没有自动填写。
- `formal_calibration_executed=false`，`formal_development_executed=false`，`formal_locked_audit_executed=false`。
- synthetic smoke 已成功运行，但只验证控制流、schema、拒绝门禁、候选集合逻辑和产物写出；它不是自然语言模型评测，不构成 H1–H3 的研究证据，也不能用于 readiness 判定。

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

正式 `stage-c24b-prepare` 已在数据物化提交 `fec7b232cea863c2df814914e2e02c6400efaaf4` 上运行一次。split 隔离审计与 C2.3/C2.4 历史碰撞审计均为 `passed=true`；历史审计覆盖 228 条 phrase、25 个 entity 和 19 个 frame。三份数据 SHA-256 分别为：train `e808f546…20276`、calibration `0eaa0f41…22e9e`、locked `17b679d9…e8d48`。公开 benchmark manifest SHA-256 为 `7977b7a3…ed977`。这不是 router/model/threshold 的正式 freeze；后者只能在真人审核完成后的 formal calibration 发生。

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

如果使用 conformal，正式报告必须保存 nonconformity 定义、每个 relation 的 calibration quantile、目标 coverage 和有限样本条件。当前实现还明确记录：evidence source、alpha 与 quantile 共用 calibration split 进行选择，因此不主张标准 split-conformal 的名义有限样本覆盖保证；在 lexical 或语义分布迁移下同样不主张严格覆盖保证。

## 当前运行阻塞与资产状态

正式配置记录了模型 ID、固定 revision 和预期 SHA-256，但当前 clone 不包含以下正式运行必需资产：

- `artifacts/stage_c23/semantic_audit/S5/ridge_relation_head.pt`；
- `artifacts/stage_c23/private_answer_free_features.pt`；
- `artifacts/stage_c21/R3/canonicalizer.pt`；
- `.downloads/hf` 中固定 revision 的 BGE、MiniLM 和 E5 模型文件。

因此当前既不能真实复现 R0，也不能执行正式 semantic embedding、fact retrieval 或 source checkpoint 的运行前后字节校验。配置中已有预期 provenance 哈希不等于当前 clone 已持有对应文件。本阶段没有用临时模型、随机 embedding 或 synthetic 数值替代正式指标。

正式运行的两个独立前置条件均未满足：真实人工审核尚未完成，正式固定资产也不在当前 clone。`stage-c24b-validate-review`、正式 `stage-c24b-calibrate` 和正式 `stage-c24b-audit` 已分别实际调用，三者都在第一道人审门禁以 `ReviewIncompleteError` 拒绝；没有创建 calibration 目录、development marker、locked-open marker 或 `protocol_incident.json`，也没有越过门禁加载模型。

## R0–R4 核心结果

正式结果位置保持“未执行”。下表中的数值来自 10 条 synthetic locked mock row，只证明相应控制流能工作，不能解释为 lexical OOD、自然语言泛化或研究门槛结果。

| Router | 正式 calibration / locked | Smoke known coverage | Smoke accepted route accuracy | Smoke ambiguous / unrelated reject | Smoke false memory access | Smoke safe coverage |
|---|---|---:|---:|---:|---:|---:|
| R0 frozen argmax | 未执行 / 未执行 | 1.00 | 1.00 | 0.00 / 0.00 | 1.00 | 1.00 |
| R1 definition ensemble | 未执行 / 未执行 | 1.00 | 1.00 | 0.00 / 0.00 | 1.00 | 1.00 |
| R2 pairwise evidence | 未执行 / 未执行 | 1.00 | 1.00 | 0.00 / 0.00 | 1.00 | 1.00 |
| R3 set-valued router | 未执行 / 未执行 | 1.00 | 1.00 | 1.00 / 1.00 | 0.00 | 1.00 |
| R4 calibrated/conformal | 未执行 / 未执行 | 0.50 | 1.00 | 1.00 / 1.00 | 0.00 | 0.50 |

全部 Smoke 变体的 wrong-bucket access rate 为 0，accepted fact Top-1 与 MRR 均为 1，`cross_relation_candidate_count=0`。这些 fact 数值来自确定性的 synthetic entity prototype，不能代替正式 answer-free retrieval 审计。Smoke calibration 选择了 R4，但 synthetic locked mock 上的 known coverage 降为 0.50，进一步说明 Smoke 选择不得转写为正式选择。

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

仅供复现实装控制流的 synthetic 设置为：R1 `mean`；R2 ridge strength `0.01`；R3 global threshold `0.6`、minimum margin `0.0`；R4 `alpha=0.1`，registry/city/access nonconformity quantile 分别为 `-0.9985621572`、`-0.9984798431`、`-0.9985620379`。这些设置带有 `synthetic_smoke_only` provenance，不是正式 threshold 或 conformal 参数。

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

正式后端把 C2.3 answer-free development route/retrieval 明确标为一次性诊断；它不参与 router 选择，也不直接进入 readiness。为补齐 §11.5，当前实现还在 first locked read 之前生成 `locked_relation_request_x_prefrozen_answer_free_slot_binding_v1`：排序后的 v2 public entity ID 与排序后的 sanitized C2.3 development entity 做固定一一映射，同一 public entity 在全部 relation/family/frame 下只使用一个、对全部 source rows 聚合的 relation-independent entity tensor；数量不足即 fail closed，禁止按 label、family、score、prediction 或 retrieval result 挑选。

正式 locked 评分时，R0–R4 的 `AcceptedRoute` 分别实际访问预测的单个 bucket，`RejectedRoute` 不访问 memory；known wrong bucket 记 Top-1 miss/rank 0，known reject 只影响 all-query 指标。true `RelationId` oracle 在任何 locked row/score 之前对完整 entity×relation 网格离线预计算，其访问单独记录，不计入系统 FMAR。binding 封存 cache、entity tensor、bucket prototype、opaque fact ID 与 oracle table SHA-256。由此可报告 route-conditioned accepted fact Top-1、all-query Top-1、row 1-NN、MRR、margin、oracle gap 和 cross-relation count，并允许 locked fact evidence 参与 closed readiness。

该指标仍有明确限制：它复用历史 C2.3 answer-free development entity substrate 与 C2.3 train bucket，`fact_retrieval_independent_locked_test=false`；它只审计“v2 locked relation route → typed bucket → historical answer-free slot retrieval”的组合，不测试 v2 文本中新实体的真实 entity-OOD 泛化，也不提供独立 entity-side 泛化结论。

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
12. **下一步应继续 router 研究，还是冻结协议并创建全新 confirmation？** 先完成真实双人审核、补齐固定资产并按当前冻结顺序运行正式 calibration/audit；再根据正式 lexical OOD、reject 与 route-conditioned slot 指标决定是否继续 router 研究。当前不能创建 confirmation。

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

`prepare` 已按协议物化 locked JSONL 与空白审核任务，供后续人工审核和封存；但正式模型评分从未读取该 split，`locked_audit_opened.json` 从未创建，故不存在根据其结果调参的行为。若未来发生 locked 数据被提前用于模型评分、参与选择或在冻结后被修改，应立即停止并写入新的 `protocol_incident.json`，且不得覆盖 C2.3 历史事故记录。

## 工程验证与 provenance

- 主线程与独立审计线程分别执行全仓 `pytest -q`，结果均为 `197 passed`（主线程 36.34 秒，独立复验 34.73 秒）；C2.4b 专项复验为 `79 passed`。
- 正式 prepare artifact manifest 封存 12 个先行文件，Smoke artifact manifest 封存 21 个先行文件；逐文件 size 与 SHA-256 复算均一致。
- 正式 benchmark manifest、review manifest、prepare artifact manifest 的 SHA-256 分别为 `7977b7a3…ed977`、`bf5ee952…19394`、`2b94e7f4…e2225`；在最终实现提交上重跑的 Smoke summary 与 artifact manifest 分别为 `5561a28b…266969`、`a334a823…1e90c2`。
- 共 25 个产物 JSON 均用拒绝 NaN/Infinity 的严格解析器和敏感字段检查器复验通过；两个 artifact manifest 覆盖的 33 个先行文件均完成 size/SHA-256 复算。没有 `.pt`、模型 checkpoint、optimizer state、private answer、key 或 confirmation 数据进入提交。`resolved_config.json` 是由 artifact manifest 外层封存的纯配置快照，并不单独声称每个 JSON 都内嵌完整 provenance envelope。
- prepare/data 物化提交为 `fec7b232cea863c2df814914e2e02c6400efaaf4`，且 prepare 当时工作树为 clean；包含 locked slot binding 的最终实现提交为 `fe1d625b0da6173af7bd9c2ebd8b37a545fde13b`，并已在该提交上重跑 synthetic smoke。真正的 router/model/threshold/code freeze 尚未发生，必须等 formal calibration。正式运行会将整个 `src/keyed_gram/*.py`、`pyproject.toml` 与正式配置绑定到届时 HEAD 字节，包含 `phase_a.py` 等传递依赖。
- 当前 clone 中 entity checkpoint 与 answer-free cache 均缺失，因此没有伪称已完成其 before/after 实体文件哈希验证；其预注册 SHA-256 保持未修改。已提交的 typed memory contract 源文件 SHA-256 为 `c8a3b5fd…96692`。

## 后续正式运行前置清单

1. 已完成 prepare：确认 72/160/160 数据、全部历史碰撞审计和 manifest seal。
2. 待两名真人独立完成 train/calibration、locked-audit 和旧 C2.4 120 条复核；冲突项完成 adjudication。
3. 审核完成后运行 review validation，并冻结 complete review manifest SHA-256。
4. 补齐固定 revision 模型、旧 ridge head、answer-free cache 和 entity checkpoint，逐项验证 SHA-256。
5. 只用 train/calibration 完成 R0–R4 拟合与选择，记录 frozen Git commit 和所有参数。
6. development 只运行一次作诊断，不参与选择，也不直接用于正式 readiness。
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
