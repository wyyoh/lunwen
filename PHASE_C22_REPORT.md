# 阶段 C2.2：无答案 Public Relation Paraphrase 监督

## 结论

阶段 C2.2 已完成。公开关系释义监督没有达到预期门槛，并进一步定位了当前架构的
能力边界：

> 冻结 SimpleStories core 上的轻量 relation MLP 可以记住 public train 词族，
> 但无法把这种关系分类迁移到完全不同的 public validation 词族，也不能稳定改善
> 原私有 development 问法。

因此，本轮不打开 confirmation、不进入 C3，也不继续在同一表示源上增加事实损失或
对抗权重。下一步应替换或补充一个在通用语言上预训练的公开 relation semantic
encoder；密钥 memory 仍只负责私有 `entity × relation -> value` 映射。

## 公开语料边界

本轮语料只包含三类关系标签、公开关系短语、公开合成占位实体和问题句式：

- train：8 个公开占位实体、每类关系 4 个短语、3 个句式，共 288 行；
- validation：4 个不重叠占位实体、每类关系 2 个不重叠短语、2 个不重叠句式，
  共 48 行；
- train/validation 的实体、句式和关系短语完全分离；
- 配置显式禁止复用私有 train/validation、development 和全部 confirmation 候选短语；
- 每行都没有 `answer`、`candidates` 或实体到私有答案的映射；
- manifest、数据哈希和特征缓存哈希均已记录，原始 JSONL 与 tensor cache 保持本地忽略。

public 与 private 特征由同一个冻结 core 的第 4、6、8 层提取，core SHA-256 为
`b141f3c08fcc9c56a6ca6a6c14169fc001613efa7ca98905b8b642070a01d0cb`。

## 消融设计

| 版本 | 定义 |
|---|---|
| P0 | 原样导入 validation 选中的 C2.1 R3 |
| P1 | 只用 public relation CE、严格跨实体/跨短语 SupCon 和 `z_r` template adversary |
| P2 | P1 + 原私有 fact/relation/template replay，防止事实几何遗忘 |

P1/P2 都只更新 relation encoder、relation head 和 template head；entity 分支、融合层、
layer mixture 与冻结 core 不更新。每个训练变体 1,000 步，学习率 `2e-4`。公开
SupCon 正样本必须是同关系、不同实体、不同 relation phrase；批内同时保留同 frame
跨关系的 hard negatives。

checkpoint 与变体只依据 private validation 和独立 public validation 的组合门槛选择。
原 Stage C2 test 仍只作 development 诊断。

## 正式结果

| 版本 | Public val head | Public val probe | Public train probe | Public relation margin | Public template probe | Private val gates | Development head | Development probe | Development centroid | Development gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 39.58% | 29.17% | 96.88% | -0.082 | 97.22% | 7/10 | **69.79%** | 65.62% | **65.10%** | 4/10 |
| P1 | 33.33% | 37.50% | **100.00%** | -0.007 | 97.22% | 5/10 | 51.04% | **66.67%** | 61.46% | 1/10 |
| **P2** | **45.83%** | **41.67%** | **100.00%** | -0.211 | **94.44%** | **7/10** | 63.02% | 63.02% | 63.02% | **4/10** |

P2 在组合 validation score 上与 P0 同为 7 个 private gates、0 个 public gates，随后
凭 private validation centroid 81.25% 高于 P0 的 80.21% 被选中，最佳点为第 250 步。
这个选择完全不读取 development。

训练历史中的单项最好 public validation 指标仍很低：P1 relation head 最高 39.58%、
probe 最高 54.17%；P2 head 最高 50.00%、probe 最高 45.83%。延长到 1,000 步没有
形成持续提升。

## 解释

### 1. 公开语料被记忆，而不是被语义归一化

P1/P2 的 public train ridge probe 都达到 100%，但完全不重叠的 public validation
probe 只有 37.50%/41.67%。这排除了“relation 分支完全没有容量”的解释，支持
“当前 core 表示主要编码词面和句式，MLP 只能拟合已见词族”的解释。

### 2. Private replay 只能防遗忘，不能创造 lexical semantics

P1 明显破坏 private fact geometry；P2 将 private validation 恢复到 7/10 门槛，证明
replay 有效。但 P2 的 development relation head/probe 都只有 63.02%，没有超过 P0，
说明 replay 不能把 public 释义转化成未见私有问法的关系语义。

### 3. Template adversary 仍然失败

P1/P2 的 public train template probe 仍为 97.22%/94.44%，public relation margin 也
没有翻正。对抗权重不是当前最值得扩大的方向；更强对抗很可能继续删除关系词本身，
却不提供同义表达知识。

## 决策

- `development_relation_ready=false`
- `development_ready_for_confirmation=false`
- `confirmation_access_count=0`
- `c3_eligible=false`

下一阶段应研究一个独立、公开、可替换的语义编码器，只输出 canonical relation
表示；它可以读取通用公开文本，但不得读取任何私有答案或 confirmation 词族。首先做
零训练 embedding audit，证明公开 train 到 public validation、私有 validation 和当前
development 的 relation probe 均能达到 85%，再讨论如何与 entity branch 融合。

## 可复核产物

- 配置：`configs/stage_c22.yaml`
- 公开语料 manifest：`artifacts/stage_c22/public_corpus_manifest.json`
- 完整结果：`artifacts/stage_c22/stage_c22_summary.json`
- 消融表：`artifacts/stage_c22/stage_c22_ablation.csv`
- P0–P2 评估：`artifacts/stage_c22/P*/evaluation.json`
- P1/P2 训练历史：`artifacts/stage_c22/P*/history.csv`
- JSONL、public feature cache、checkpoint 均保存在本地 ignored 路径。

正式运行耗时 41.8 秒；最终全量测试为 53 passed。
