# 阶段 C2.1：关系保持型 Query Canonicalization

## 结论

阶段 C2.1 已按预注册顺序完成 R0–R4 消融，但未达到打开封存确认集或进入 C3 的条件。

> 严格 relation SupCon 和关系分支模板对抗能带来有限改善，但冻结的
> SimpleStories core 无法仅凭六套训练问法稳定理解未见关系词汇；继续叠加事实损失
> 不足以解决 lexical OOD。

因此，confirmation 仍保持零访问，`c3_eligible=false`。下一步转向不包含任何
实体—私有答案映射的公开 relation paraphrase 监督。

## 实验协议

- R0 直接加载 Stage C2 的 Q3 checkpoint，SHA-256 为
  `a47ca753c7ba152d8f53cb9d83a41a289443d974e9026723b2682389b1b223dd`，不重训。
- R1 加入严格 relation SupCon。正样本必须同时满足同关系、不同实体、不同模板；
  batch 同时包含同实体跨关系、同关系跨实体和同模板跨关系样本。
- R2 将权重 0.05 的 template adversary 移到归一化 `z_r`。
- R3 在固定 1,500 步预算内先进行 300 步 relation-only 训练，随后 1,200 步联合训练；
  联合阶段 relation encoder 学习率缩放为 0.2。
- R4 在 R3 上加入 entity/relation 归一化、三个可学习正标量和显式融合贡献记录。
- 每个 checkpoint 和最终变体仅由 validation 的严格门槛数及预先固定的指标 tuple 选择。
- 原 Stage C2 test 只作为 development 诊断，不参与选择。
- 训练只读取冻结 core 的 Stage C2 特征缓存；没有 private memory、答案注入或答案损失。
- 本轮只读取公开 confirmation hash seal；本地 confirmation 模板、行和随机种子均未读取，
  访问计数保持 0。

## 零训练前置审计

冻结 core 的最佳原始关系源是第 8 层 `question_without_entity`：validation 为
83.33%，development 为 75.00%。它只达到边界水平，不足以作为可直接进入 C3 的
稳定 relation anchor。Q3 的 learned relation 表示还把 development probe 从输入的
68.23% 降到输出的 61.46%，支持“联合事实训练会损失部分关系信息”的假设。

## 正式 R0–R4 结果

下表全部是 development 指标；加粗只表示该列最佳，不参与变体选择。

| 版本 | Relation head | `z_r` relation probe | Relation-template margin | Fact centroid | Row 1-NN | MRR | Fact silhouette | Fact-template margin | Entity probe | Query template probe | 严格门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 61.46% | 59.90% | **0.262** | 60.42% | 59.38% | 0.756 | 0.545 | 0.517 | 95.83% | 50.00% | 4/10 |
| R1 | 65.62% | 66.15% | 0.110 | **70.31%** | **70.31%** | **0.812** | **0.596** | **0.623** | **98.44%** | **43.06%** | 3/10 |
| R2 | **70.83%** | **71.35%** | 0.167 | 68.23% | 68.23% | 0.798 | 0.581 | 0.622 | 96.35% | 73.61% | 4/10 |
| R3 | 69.79% | 65.62% | 0.151 | 65.10% | 65.10% | 0.774 | 0.497 | 0.606 | 91.15% | 91.67% | 4/10 |
| R4 | 65.10% | 64.06% | 0.075 | 60.94% | 61.98% | 0.693 | 0.347 | 0.371 | 86.98% | 97.22% | 1/10 |

validation 上 R3 通过 7/10 门槛，高于其他变体，因此按协议选中 R3，最佳点为第
450 步。R3 的 development 结果未达门槛，不能改用 development 上某一列更好的
R1 或 R2；否则会破坏封存协议。

## 分解结论

### 1. Relation SupCon 有效，但不能独立解决 lexical OOD

R1 将事实 centroid/row 1-NN 从约 60% 提高到 70.31%，关系 probe 也从 59.90%
提高到 66.15%。这证明关系级对比约束提供了有效信号，但离 85% 仍有明显差距。

### 2. `z_r` adversary 改善关系分类，却没有去除模板信息

R2 的 relation head/probe 达到本轮最高的 70.83%/71.35%，说明分支级对抗比只在
最终 query 上对抗更合理。但所有变体的 `z_r` template probe 都是 100%，R2 的最终
query template probe 也升到 73.61%。权重 0.05 的对抗没有实现可验证的模板不变性，
不能据此宣称关系与模板已解耦。

### 3. Relation-first curriculum 只改善 validation 选择指标

R3 在 validation 上通过最多门槛并被合法选中，但 development relation probe 只有
65.62%，低于 R2；事实 centroid 也只有 65.10%。因此当前证据不支持“降低联合阶段
relation encoder 学习率即可阻止关系语义覆盖”。更可能的限制是训练词汇覆盖不足，
而不只是优化器阶段冲突。

### 4. Normalized gated fusion 在当前输入上无效

R4 的三个尺度为 1.012、0.988、1.004，几乎没有偏离初值；development 的实体、
关系、交互贡献范数分别为 1.024、0.738、0.053。它没有恢复关系泛化，反而把严格
门槛从 R3 的 4/10 降到 1/10。当前瓶颈不是简单的分支幅值失衡。

## 严格门槛与下一步

R3 在 development 未通过：relation head、relation probe、fact centroid、row 1-NN、
MRR 和 template probe。虽然 silhouette、fact-template margin、entity probe 和
relation-template margin 已通过，但固定 prototype memory 仍会产生大量错槽读取。

下一步实施 Stage C2.2：构造只包含关系名称与公开释义的 paraphrase corpus，并按
train/development/confirmation 词汇家族严格隔离。该语料不包含实体、不包含私有答案、
不包含 `entity -> answer` 映射；它只负责教公开语义模块识别“用户问的是什么关系”。
在 development 关系 head/probe 达到 85% 且事实几何门槛同时满足前，不打开
confirmation，也不进入 C3。

## 可复核产物

- 配置：`configs/stage_c21.yaml`
- 零训练审计：`artifacts/stage_c21/audit/relation_source_audit.json`
- confirmation 公共封存记录：`artifacts/stage_c21/confirmation_seal.json`
- 完整结果：`artifacts/stage_c21/stage_c21_summary.json`
- 消融表：`artifacts/stage_c21/stage_c21_ablation.csv`
- R0–R4 评估：`artifacts/stage_c21/R*/evaluation.json`
- R1–R4 训练历史：`artifacts/stage_c21/R*/history.csv`
- checkpoint 和 branch embeddings 仅保存在本地 ignored artifacts 中。

正式运行耗时 107.2 秒；实现完成后全量测试为 50 passed。
