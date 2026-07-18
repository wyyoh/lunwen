# 阶段 B：跨问法不变私有残差化与公共残差抑制

## 结论

阶段 B 的 B0–B3 消融已完整运行，**没有变体通过 B-pass**。最有价值的
结果不是某个新损失“成功”，而是目标冲突被直接测量出来：

> 层级 residual scaling、public KL 和 residual nulling 能把公共副作用
> 几乎消除，但 private 事实学习与 public nulling 在同一组 aux 参数上呈现
> 持续、强烈的负梯度冲突。

同时，B0 在不做严格 public nulling 时 unseen 也只有 23.96%，因此梯度冲突
不是唯一原因：当前 residual aux 还存在独立的跨问法语义接口失败。后续
Stage C1 已对这一点做了冻结-core 表示诊断，详见 `PHASE_C1_REPORT.md`。

B1 是当前最合理的机制基线。B2 的 paraphrase KL 没有提高未见问法，B3
的 residual InfoNCE 反而降低未见问法准确率，因此当前没有证据支持保留
InfoNCE。按预设规则，未运行错误密钥、部分密钥或恢复攻击矩阵。

## 实验设计

数据仍使用不可由语义猜测的 atomic-token synthetic biographies：

- 16 个已见实体，8 个未见实体；
- 每个实体包含 registry、city、access 三个随机事实；
- 每题 8 个候选，随机准确率 12.5%；
- 6 个训练模板、2 个验证模板、4 个完全未参与训练的测试模板；
- 288 条训练问法、96 条验证问法、192 条主测试问法；
- 另有 24 条“未见实体但上下文给出事实”和 24 条“事实从未给出”样本。

主指标只使用“已见实体 + 未见问法”。从未暴露的随机事实不要求模型凭空
答对；context-only 与 never-seen 只作为独立诊断格。

每个变体从同一基座、同一随机种子重新初始化，训练 1,500 个 cycle。
private/public 更新交替执行，比例 1:1。所有 core 参数被冻结，并用
non-aux SHA-256 验证。

## 网络与损失

B1–B3 在每层使用：

    h' = h + alpha_l * AuxMLP(stopgrad(h))

`alpha_l` 为可训练层级缩放，初始值 0.01。aux 两层投影正常随机初始化，
训练开始时实际 residual 很小。旧检查点默认不启用该参数，因此 Phase 1/A
行为保持兼容。

消融定义：

| 变体 | Private | Public | 跨问法 | 表征 |
|---|---|---|---|---|
| B0 | answer CE | 无 | 无 | 普通 aux |
| B1 | answer CE | core→full KL + residual nulling | 无 | 层级小缩放 residual |
| B2 | answer CE | 同 B1 | 候选分布对称 KL | 同 B1 |
| B3 | answer CE | 同 B1 | 同 B2 | residual InfoNCE |

## B0–B3 结果

| 变体 | Seen | Validation | Unseen | Worst template | Agreement | Public NLL 增幅 | Public KL | Selectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 57.29% | 25.00% | **23.96%** | 16.67% | 24.31% | 77.84% | 2.4594 | 1.30 |
| B1 | 46.53% | 23.96% | 20.31% | 16.67% | 25.00% | **0.015%** | 0.00277 | 65.00 |
| B2 | 47.57% | 29.17% | 20.31% | 12.50% | 21.88% | **0.023%** | 0.00226 | 70.06 |
| B3 | 49.31% | 29.17% | 16.67% | 10.42% | 25.69% | **0.053%** | 0.00383 | 43.49 |

所有变体的 core-only 主测试准确率均为 11.98%，与 12.5% 随机值一致；
四个 core 参数哈希也都逐位一致。

自由生成 exact match 分别为 B0 11.46%、B1 6.25%、B2 4.17%、B3
3.65%。它不是第一道门槛，但与候选准确率同样没有显示 B2/B3 的收益。

## 机制分解

### B1 成功解决 public interference

B0 的 public NLL 增加 77.84%，top-1 token agreement 只有 14.86%。B1
将 NLL 增幅降至 0.015%，top-1 agreement 提升到 97.58%。因此以下组合
已经得到正面验证：

- 小尺度、逐层可训练 residual；
- core-only public teacher；
- public output KL；
- public residual nulling。

### 但 B1 没有解决 semantic capability localization

B1 的 unseen accuracy 只有 20.31%，比 B0 的 23.96% 更低。其平均
private residual/hidden norm 比为 15.73，而 public 比为 0.208。高达 65
的 selectivity 主要表示 public residual 被压小，并不表示 private residual
已经结构化；private residual 本身非常大但事实准确率仍低，符合“无结构
扰动”诊断。

### 梯度冲突是两个独立瓶颈之一

| 变体 | 平均 private/public 梯度余弦 | 负余弦比例 |
|---|---:|---:|
| B1 | -0.596 | 91.80% |
| B2 | -0.683 | 96.72% |
| B3 | -0.557 | 85.25% |

这说明简单加权求和或交替更新并没有消除目标冲突。public step 在不断撤销
private step 对同一 aux 参数产生的部分更新。但 B0 同时证明，即使移除该
约束，原 residual 结构也没有学会跨模板查询，因此不能把全部 unseen 失败
归因于梯度冲突。

### B2/B3 当前无正面证据

B2 没有改变 unseen accuracy，并使 AnswerAgreement 从 25.00% 降至
21.88%。B3 的 unseen accuracy 进一步降至 16.67%。因此不能把“加入
一致性/对比损失”本身当作贡献；当前结果支持保留 B1，暂不保留 B3。

## B-pass 判定

| 指标 | 门槛 | 最佳实际值 | 结果 |
|---|---:|---:|---|
| Seen full accuracy | ≥95% | 57.29%（B0） | 未通过 |
| Unseen full accuracy | ≥80% | 23.96%（B0） | 未通过 |
| AnswerAgreement | ≥90% | 25.69%（B3） | 未通过 |
| Core-only accuracy | ≤17.5% | 11.98% | 通过 |
| Public NLL 增幅 | ≤5% | 0.015%（B1） | 通过 |
| Core SHA-256 | 完全一致 | 四变体完全一致 | 通过 |

LocScore 在 B0–B3 中均为 1，因为 core 略低于随机、full 略高于随机。
它是“泄漏比例”指标，不衡量能力绝对强弱，因此必须与仅 16.67%–23.96%
的 unseen accuracy 一起解释。

## 下一阶段入口

阶段 B 不支持继续堆叠一致性损失。后续路线先用 Stage C1 判断 frozen core
是否已经提供跨模板事实接口，再决定优化与结构路径。C1 的实际结果是所有层
检索均很低，因此下一次方法迭代应优先实现 query canonicalizer + key-value
private memory；PCGrad、非对称 public projection 和逐层 public-gradient
nullspace 保留为冲突处理 baseline/组件，而不是语义泛化修复手段。

在 unseen accuracy、AnswerAgreement 与 public 保持同时通过以前，仍不应
进入 key-aware/wrong-key 安全训练。

## 产物与复现

正式命令：

    keyed-gram stage-b --config configs/stage_b.yaml --output-dir artifacts/stage_b --device cuda

主要文件：

- `configs/stage_b.yaml`
- `data/stage_b/stage_b_manifest.json`
- `artifacts/stage_b/stage_b_ablation.csv`
- `artifacts/stage_b/stage_b_summary.json`
- `artifacts/stage_b/B0..B3/train/run.json`
- `artifacts/stage_b/B0..B3/train/history.csv`
- `artifacts/stage_b/B0..B3/evaluation/evaluation.json`
- `artifacts/stage_b/B0..B3/evaluation/fact_results.csv`

正式四消融命令耗时 789.4 秒；其中训练耗时约 743.4 秒。最终测试为
32 项全部通过。
