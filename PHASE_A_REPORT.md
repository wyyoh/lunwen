# 阶段 A：Synthetic Biographies 最小化验证报告

## 结论

阶段 A 已完成，但 **freeze-core 基线未通过局部化门槛**。

实验已经证明两件不同的事：

1. 冻结 core、只训练 aux 时，aux 确实能够记住随机事实；在无 public
   约束的诊断实验中，两种训练问法上的候选准确率均达到 100%，core
   参数哈希逐位不变。
2. 这种记忆尚未形成稳定、低副作用的“能力”：换成未见问法后准确率明显
   下降；加强记忆又会使 aux 干扰普通 SimpleStories 行为。

因此当前瓶颈已经从“core 是否偷偷学到事实”进一步收敛为：

> 如何让 aux 学到跨问法稳定的事实残差，同时在 public 输入上保持近零贡献。

按照预设门槛，本阶段没有运行密钥错误、部分密钥或恢复攻击矩阵。

## 数据与控制变量

- Public：固定的 8 个普通 SimpleStories 主题。
- Private：synthetic biographies，每个实体有 `registry_id`、`city_code`、
  `access_code` 三个随机属性。
- 最小任务使用 atomic-token codebook：实体句柄和事实值都是单 tokenizer
  token，但实体—事实映射由种子随机生成，不能从词义推断。
- 每道题固定 8 个同类型候选，理论随机准确率为 12.5%。
- 训练实体的事实进入训练；held-out 实体只进入评估。
- 训练使用 6 种问法，正式评估使用第 7 种未见问法。
- 更困难的多位数字编码保留为后续压力测试，不作为最小可行性门槛。

训练目标只计算答案 token。早期实现曾把 EOS 与单-token 答案一起平均，
造成 loss 虚假下降；该问题已修正并增加回归测试，受影响的早期运行不作为
科研结果。

## 方法

基座为 Phase 1 的 8 层、hidden 512、aux width 192 检查点。实验开始时将
aux 重置为零输出 adapter：`c_fc` 随机初始化，`c_proj` 置零。

Private batch：

- full 前向开启 core + aux；
- backward 只允许 aux 接收梯度；
- embedding、attention、core FFN、norm 和 LM head 全部冻结。

Public batch：

- core-only 输出作为 teacher；
- full 输出通过 KL 项接近 teacher；
- canonical 运行使用 `lambda_public=0.1`。

可训练参数为 1,578,496。训练前后 non-aux SHA-256 完全一致，证明 core
没有被优化器或前向实现意外修改。

## 主要结果

| 运行 | 训练实体 | 训练问法 | Public KL | Seen prompt | Unseen prompt full | Core-only | Public loss 增幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 强 public 保持 | 64 | 2 | 1.0 | — | 13.54% | 10.94% | 0.39% |
| 记忆能力诊断 | 16 | 2 | 0.0 | 100.00% | 25.00% | 6.25% | 53.54% |
| 多问法平衡版 | 16 | 6 | 0.1 | 66.67% | **37.50%** | **6.25%** | **8.89%** |

多问法平衡版的自由生成 exact match 为 16.67%；未暴露 held-out 事实的
full 候选准确率为 0%，但样本只有 24 题，应只视为“没有发现泄漏”，不能
解释成显著低于随机。

平衡版 CDR 为 1.25。该值超出 `[0,1]`，原因是有限样本上的 core-only
准确率 6.25% 低于理论随机值 12.5%。实现保留并标记这一异常值；它不能
替代 full accuracy 门槛。

## 验收判定

| 门槛 | 目标 | 实际 | 结果 |
|---|---:|---:|---|
| Unseen-prompt full candidate accuracy | ≥80% | 37.50% | 未通过 |
| Core-only candidate accuracy | ≤17.5% | 6.25% | 通过 |
| CDR | ≥0.70 | 1.25（越界） | 数值通过、标记异常 |
| Public loss 增幅 | ≤5% | 8.89% | 未通过 |
| Core 参数保持 | SHA-256 完全一致 | 完全一致 | 通过 |

总体状态：`failed`。这不是数据集构造失败：无 public 约束时训练问法可以
被 aux 完整记住。失败发生在“跨问法泛化”和“public 副作用”同时满足的
联合条件上。

## 科研含义与下一阶段入口

阶段 A 支持以下判断：

- synthetic biographies 成功移除了 `alien-encounters` 的语义可猜测性；
- freeze-core 足以阻断 core 参数写入，但不等于形成稳健的能力模块；
- 单纯调 KL 权重形成明显 Pareto 冲突：强保持时学不会事实，弱保持时
  public 行为被 aux 广泛干扰；
- 下一步应进入 Stage B 的 `stop-gradient + residual aux`，显式最小化
  public residual、增强 private residual，并加入跨问法一致性目标；
- 在 full 未见问法准确率达到门槛以前，不应开始正式密钥安全评估。

## 可复现入口

本次 canonical 运行保留配置：`configs/phase_a_probe16_balanced.yaml`。
默认入口 `configs/phase_a.yaml` 使用等价实验参数。

    keyed-gram phase-a --config configs/phase_a_probe16_balanced.yaml --output-dir artifacts/phase_a_probe16_balanced --device cuda

主要产物：

- `data/phase_a_probe16_balanced/phase_a_manifest.json`
- `artifacts/phase_a_probe16_balanced/train/run.json`
- `artifacts/phase_a_probe16_balanced/evaluation/phase_a_evaluation.json`
- `artifacts/phase_a_probe16_balanced/evaluation/fact_results.csv`
- `artifacts/phase_a_probe16_balanced/evaluation/seen_prompt_fact_results.csv`
- `artifacts/phase_a_probe16_balanced/phase_a_summary.json`

最终测试：26 项全部通过。
