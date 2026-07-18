# 阶段 C2：Query Canonicalization 消融

## 结论

Stage C2 已完成。结果是一个清晰的**结构性部分成功、完整门槛未通过**：

> 显式实体 span、多位置 pooling 和 fact-level SupCon 已经把 query space 从
> “模板主导”翻转为“事实主导”，但关系语义仍不能稳定跨越四套最终测试模板。

因此当前不进入 C3 private memory。否则 memory 会建立在关系识别仍不稳定的
query 上，把尚未解决的错误固化到 slot 检索中。

## 实验边界

- core：完全冻结，非辅助 SHA-256 与 C1 相同；
- 输入：第 4、6、8 层的 entity-span、去实体 question、answer-position 三种
  pooling；
- 输出：128 维单位 query；
- 数据：48 个事实，train/validation/test 分别为 6/2/4 套全局不重叠模板；
- 训练：每个变体 1,500 步，8 个事实 × 每事实 3 个问法；
- hard negatives：同实体不同关系、同关系不同实体、同模板不同事实；
- 本阶段没有 private memory、答案注入、public loss、confidence gate 或密钥；
- checkpoint 与变体只按 validation 的门槛数和检索/几何/probe tuple 选择，
  test 不参与选点。

## Q0–Q4 结果

| 版本 | Test 1-NN | 质心 Top-1 | MRR | Fact silhouette | Fact−template margin | Entity probe | Relation probe | Template probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 7.81% | 4.69% | 0.148 | −0.121 | −0.341 | 19.79% | 75.00% | 100.00% |
| Q1 | 60.94% | 60.42% | 0.744 | 0.526 | +0.545 | 93.23% | 68.23% | 61.11% |
| Q2 | 61.46% | 60.94% | 0.750 | 0.218 | +0.394 | 77.60% | 59.90% | 100.00% |
| **Q3** | **59.38%** | **60.42%** | **0.756** | **0.545** | **+0.517** | **95.83%** | **63.02%** | **50.00%** |
| Q4 | 61.46% | 61.46% | 0.754 | 0.219 | +0.395 | 78.13% | 59.90% | 100.00% |

Q3 在 validation 上通过 7/8 个部分门槛，因此按预注册规则被选择；最佳
checkpoint 是第 700 步。它在 test 上同样只缺关系门槛。

## 最重要的正面结果

C2 的核心结构假设得到支持：

- 严格跨模板 1-NN：7.81% → 59.38%；
- fact silhouette：−0.121 → +0.545；
- fact/template margin：−0.341 → +0.517；
- 同事实余弦 0.765，高于同模板不同事实余弦 0.247；
- 实体 probe：19.79% → 95.83%；
- 模板 probe：100% → 50%。

这已经证明 entity-span pooling 和 canonicalizer 确实改变了表示几何，而不只是
提高某个输出分类器的准确率。意见中要求的最小核心目标——1-NN 至少 50% 且
fact/template margin 翻正——已经实现。

## 为什么仍判定失败

Q3 的完整部分门槛中，7 项通过、1 项失败：

| 门槛 | 要求 | Q3 | 结果 |
|---|---:|---:|---|
| Row 1-NN | ≥50% | 59.38% | 通过 |
| Centroid top-1 | ≥60% | 60.42% | 通过 |
| MRR | ≥0.70 | 0.756 | 通过 |
| Fact silhouette | >0 | 0.545 | 通过 |
| Fact/template margin | >0 | +0.517 | 通过 |
| Entity probe | ≥70% | 95.83% | 通过 |
| Template probe | ≤50% | 50.00% | 通过（临界） |
| **Relation probe** | **≥85%** | **63.02%** | **未通过** |

显式 relation auxiliary head 在 test 上也只有 61.46%，与 ridge probe 的
63.02% 一致，所以失败不是 probe 口径造成的。validation relation head 为
83.33%，到了四套真正未见模板后进一步下降，说明关系分支仍学习了部分问法
表面模式。

Q4 的 0→0.1 模板对抗没有带来正面收益；其 validation 最优点仍出现在第 50
步，说明当前把 adversary 施加在最终联合 query 上，并没有直接修复关系分支。

## 下一步

继续停留在 C2，做一个小型 C2.1 relation canonicalization，而不是进入 memory：

1. 在 `z_r` 上加入 relation-level SupCon，使同关系不同模板成为正样本；
2. 将 template adversary 从最终 `q` 移到 `z_r`，直接去除关系问法形式；
3. 只依据 validation 比较 relation 权重 0.2/0.5/1.0；
4. 保留 fact SupCon、实体分支和 hard negatives；
5. 新增一套从未用于本轮诊断的 confirmation 模板，避免后续迭代复用当前
   test 造成选择偏差。

只有 relation accuracy 达到 85%，并维持当前事实检索与正 margin，才进入
C3 固定 prototype key-value memory。

## 产物

- 配置：`configs/stage_c2.yaml`
- 完整结果：`artifacts/stage_c2/stage_c2_summary.json`
- 消融表：`artifacts/stage_c2/stage_c2_ablation.csv`
- 特征缓存：`artifacts/stage_c2/canonicalizer_features.pt`
- Q1–Q4 checkpoint：`artifacts/stage_c2/Q*/canonicalizer.pt`
- 训练历史：`artifacts/stage_c2/Q*/history.csv`
- 逐查询检索：`artifacts/stage_c2/retrieval_predictions.csv`

最终正式运行耗时 82.8 秒；完整测试为 42 passed。
