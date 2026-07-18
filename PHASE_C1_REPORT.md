# 阶段 C1：冻结 Core 的跨问法语义接口诊断

## 结论

Stage C1 已完成。它没有训练任何参数，也没有开启辅助分支。结果支持路线中
更重要的第二个判断：**当前失败不只是 private/public 梯度冲突；冻结 core 在
辅助 MLP 的实际输入位置上，也没有形成稳定的“实体—关系—事实”跨模板接口。**

因此下一步应先实现 KOQM 的 query canonicalizer，再考虑 PCGrad 或公共梯度
投影。仅在 B1 上替换优化器，不能补出当前缺失的语义接口。

## 实验设计

- checkpoint：`artifacts/checkpoints/main/gram_original.pt`
- 数据：Stage B 的 48 个随机事实（16 个实体 × 3 个关系）
- 检索库：6 套 train 模板，共 288 条
- 调参诊断：2 套 validation 模板，共 96 条
- 最终诊断：4 套全局未见 test 模板，共 192 条
- 表示位置：每层最后一个提示 token、attention 之后、core/aux MLP 之前的
  RMSNorm 表示；这正是该层辅助 MLP 实际读取的向量
- 模型视图：core-only；无反向传播、无参数更新
- 主判据：train 表示作为数据库，test 表示作为查询的严格行级 1-NN 事实检索
- 附加判据：事实质心检索、余弦几何、silhouette、实体/关系/答案线性 ridge
  probe

train、validation 和 test 的模板 ID 全局不相交，但三个 split 包含完全相同的
48 个事实，所以该任务只测跨问法表示一致性，不混入新事实泛化。

## 核心结果

| 指标 | 随机/通过参照 | 最佳结果 | 解释 |
|---|---:|---:|---|
| test 行级跨模板 1-NN | 随机 2.08%；强接口 75% | **7.81%（第 6 层）** | 高于随机但远低于可用接口 |
| test 事实质心 top-1 | 随机 2.08% | **7.29%（第 1 层）** | 对 train 模板求均值也不能修复 |
| test 事实 silhouette | 正值才表示事实内聚 | **所有层为负；最好 −0.118** | 同一事实没有形成簇 |
| 同事实相对同模板余弦 margin | 应为正 | **所有层为负；−0.210 到 −0.343** | 模板身份压过事实身份 |
| test 实体 ID 线性 probe | 随机 6.25% | **22.92%** | 有实体信号，但跨模板很弱 |
| test 关系 ID 线性 probe | 随机 33.33% | **75.00%** | 关系语义相对可读 |
| test 答案 token 线性 probe | 随机 2.08% | **10.94%** | 联合事实映射仍不稳定 |

最佳行级检索层是第 6 层：行级 1-NN 为 15/192（7.81%），事实质心 top-1
为 9/192（4.69%），事实 silhouette 为 −0.121。同事实 train→test 平均余弦
为 0.615，而 test 内同模板、不同事实平均余弦为 0.956，二者 margin 为
−0.341。这不是“相似度整体偏低”，而是明确的模板主导结构。

深层 train probe 几乎可以记住训练模板（第 6 层实体/关系/答案训练准确率分别
为 98.61%/100%/98.26%），但在未见 test 模板上分别跌至
19.79%/75.00%/5.21%。这进一步排除了“表示容量不够”的简单解释：容量存在，
跨模板坐标系不稳定。

## 与 Stage B 的关系

诊断 checkpoint 的非辅助参数 SHA-256 为
`b141f3c08fcc9c56a6ca6a6c14169fc001613efa7ca98905b8b642070a01d0cb`。
它与 B1 checkpoint 的 core hash 逐位一致，因此 C1 对原始 core 的判断也直接
适用于 freeze-core 的 B1。Stage B 已测得 B1 的 private/public 平均梯度余弦
为 −0.596、91.8% 快照为负；C1 则独立证明了语义接口同时失效。两个瓶颈都
真实存在，不能互相替代。

## 下一阶段入口

下一次训练不再给原残差分支叠加 KL 或 InfoNCE。最小实现应是：

1. `QueryCanonicalizer` 从多层/多 token frozen-core 表示生成规范查询向量；
2. key-value private memory 以实体与关系联合键读取答案表示；
3. confidence gate 在不确定时关闭私有路径；
4. 通过有界、正交的输出注入保持 public 行为；
5. 先验证 canonicalizer 的 train→test 事实检索，再恢复 private answer CE 与
   非对称 public projection；PCGrad 只作为优化消融，不作为语义修复手段。

## 产物与复现

- 配置：`configs/stage_c1.yaml`
- 完整结果：`artifacts/stage_c1/stage_c1_summary.json`
- 分层表：`artifacts/stage_c1/layer_metrics.csv`
- 逐查询结果：`artifacts/stage_c1/retrieval_predictions.csv`
- 线性探针：`artifacts/stage_c1/linear_probes.csv`
- 可复算表示：`artifacts/stage_c1/representations.pt`

首次冷启动正式运行耗时 21.4 秒；缓存后的完整复跑耗时 5.5 秒且指标一致。
包含 Stage C1 新增测试后的完整测试结果为 36 passed。
