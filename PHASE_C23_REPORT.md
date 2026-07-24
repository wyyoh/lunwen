# Stage C2.3 报告：Public Semantic Encoder Audit

日期：2026-07-20

## 结论

Stage C2.3 **总体未通过预注册严格门槛，不能进入 C3**，但已经把此前的“未见问法失败”分解到一个更窄的接口问题：

- S0 关系真值上界在 validation 和 development 均通过 10/10 门槛，说明现有冻结 entity branch 与关系融合足以形成目标事实检索几何。
- 冻结公共语义编码器与冻结 entity branch 组成的审计链路，在当前三关系合成基准上已达到关系判别、关系间隔、实体保持和事实检索门槛。
- 小模型 S5 与 E5-base S6 都只失败于 `projected_family_probe`：连续三维 relation softmax 仍可被用来预测同一关系内部的措辞家族。
- E5-base 没有消除这一泄漏，因此当前证据更指向 relation 输出契约，而不是支持继续扩大编码器或重调 SupCon、模板对抗、课程学习和事实损失。该结果不能证明输出契约是唯一因果，也不能把单个 E5-base 候选称为普遍容量上界。

当前决策固定为：`c3_eligible=false`；不创建 replacement confirmation，不训练 private memory，不做答案注入。

## 实验边界与数据协议

C2.3 只审计公开关系语义接口：

- 冻结 C2.1 R3 的 entity branch；
- S0 使用真实 relation ID，但不挂接 memory；
- S2–S4/S6 使用固定 revision 的冻结文本编码器；
- S5 只在 public train 上拟合确定性 ridge relation head；
- private answer 不作为训练目标、不输入语义编码器、不写入 C2.3 JSON/CSV；
- public train、expanded public validation 和 private validation 可以参与选择；
- development 只在候选冻结后，对选中的候选物化并评估一次。

为使运行时接口也满足数据最小化，`stage-c23-prepare` 是唯一会反序列化旧 Stage C2 answer-bearing feature cache 的 C2.3 步骤。它只保留 `fact_id/entity/attribute/template_id/prompt` 和特征张量，生成独立的 answer-free cache。S0 与 S2–S6 运行时会拒绝任何仍含 `answer`、`answer_index`、`candidates` 等额外元数据字段的缓存。生成清单见 `artifacts/stage_c23/private_answer_free_feature_manifest.json`。

### Confirmation 协议事件

C2.1/C2.2 报告中的零访问陈述只描述各自原始运行。2026-07-20 的后续只读审查访问了旧模板选择文件一次；这里的 `1` 表示一次访问事件，观察到的模板数量未记录。该事件没有读取样本行或 private answer，没有运行模型评估，也没有计算或访问 confirmation 指标。

旧 seal 已退役且不可复用，新 confirmation 尚未创建。C2.3 本身没有读取 confirmation 数据。完整记录和退役指针分别为：

- `artifacts/stage_c23/protocol_incident.json`
- `artifacts/stage_c21/confirmation_retirement.json`

## 可复现命令

先构建扩展公开基准和运行时 answer-free 私有特征缓存：

```powershell
keyed-gram stage-c23-prepare --config configs/stage_c23.yaml
```

运行 S0：

```powershell
keyed-gram stage-c23-oracle --config configs/stage_c23.yaml --output-dir artifacts/stage_c23/S0 --device cuda
```

运行小模型矩阵和 S5 选择：

```powershell
keyed-gram stage-c23-audit --config configs/stage_c23.yaml --output-dir artifacts/stage_c23/semantic_audit --variants S2,S3,S4 --model-cache-dir .downloads/hf --embedding-cache-dir artifacts/stage_c23/semantic_embedding_cache --device cuda
```

因为小模型没有通过 validation-only 全部门槛，显式运行 S6：

```powershell
keyed-gram stage-c23-audit --config configs/stage_c23.yaml --output-dir artifacts/stage_c23/s6_upper_bound --variants S6 --model-cache-dir .downloads/hf --embedding-cache-dir artifacts/stage_c23/semantic_embedding_cache --device cuda
```

## 固定模型与实现

所有 embedding 均使用最大长度 256、L2 normalization、本地缓存和 `local_files_only` 加载。E5 的定义与查询都使用对称分类所需的 `query:` 前缀。

| 版本 | 模型与固定 revision | 维度 / pooling / 前缀 | 许可 | `model.safetensors` SHA-256 |
|---|---|---|---|---|
| S2 | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) `1110a243...` | 384 / mean / 无 | Apache-2.0 | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| S3 | [e5-small-v2](https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52) `ffb93f3b...` | 384 / mean / `query:` | MIT | `45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1` |
| S4 | [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) `5c38ec7c...` | 384 / CLS / 无 | MIT | `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad` |
| S6 | [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2/tree/f52bf8ec8c7124536f0efb74aca902b2995e5bcd) `f52bf8ec...` | 768 / mean / `query:` | MIT | `d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693` |

## 扩展公开 lexical-family 基准

公开基准完全 answer-free，不包含 entity-to-private-value 映射：

| 分块 | 行数 | phrase family 数 |
|---|---:|---:|
| public train | 54 | 18 |
| known public validation | 216 | 36 |
| ambiguous / unrelated reject | 144 | 24 |
| validation + reject 评估总计 | 360 | 60 |
| 全部物化行 | 414 | 78 |

每个关系有 6 个 train families、12 个完全隔离的 validation families；公开 train 与 validation 的 entity、frame 和 phrase family 均隔离。exact、containment 和 lemma-bigram 碰撞审计均为 0。三种固定视图为 `phrase_only`、`entity_masked` 和 `entity_masked_strip_suffix`。

该基准当前状态仍是 `curated_draft_pending_independent_human_review`。它是经整理的研究草案，不得表述为已完成人工审核或论文级定稿。

## S0：Ground-truth relation oracle

S0 在 validation 上从固定网格选择 `alpha=0.5`，之后才评估 development。

| 指标 | Validation | Development |
|---|---:|---:|
| 严格门槛 | 10/10 | 10/10 |
| Fact centroid Top-1 | 93.75% | 91.15% |
| Row 1-NN | 93.75% | 91.67% |
| Centroid MRR | 0.9604 | 0.9459 |
| Mean centroid margin | 0.1659 | 0.1595 |
| Fact silhouette | 0.7537 | 0.6825 |
| Fact-over-template margin | 0.7595 | 0.7197 |
| Entity probe | 94.79% | 94.27% |

这说明在 relation ID 正确时，现有 entity 表示、融合方式与事实度量已经足以达到本阶段的检索上界门槛。

## Zero-shot definition matching

Zero-shot 定义匹配独立报告，不参与 S5 最优候选的替代选择。各模型 `phrase_only` 结果如下：

| 版本 | Public family macro | Private validation | Raw LOFO margin | Known/reject AUROC |
|---|---:|---:|---:|---:|
| S2 MiniLM | 100% | 100% | 0.3078 | 0.9653 |
| S3 E5-small | 100% | 100% | 0.0410 | 0.9745 |
| S4 BGE-small | 100% | 100% | 0.1141 | 0.9792 |
| S6 E5-base | 100% | 100% | 0.0590 | 0.9977 |

Zero-shot 的 validation-only 选择是 `S2:phrase_only`。Raw embedding margin 是 zero-shot 诊断；S5 门槛使用线性 head 后的 projected relation representation，两者不能混用。

## S5 小模型选择与 S6 容量候选

S5 对每个小模型 × 输入视图都单独拟合 public-train ridge head，只用 validation gate count 和分数选择 `S4:entity_masked_strip_suffix`。随后才评估 development。小模型未通过 validation-only 门槛，因此运行预注册的 S6。

| 指标 | S5 小模型最优 | S6 E5-base |
|---|---:|---:|
| 选中输入 | S4 masked + strip | S6 phrase only |
| Public family macro | 99.54% | 97.22% |
| Family bootstrap 95% CI | [98.61%, 100%] | [91.67%, 100%] |
| Private validation relation | 100% | 100% |
| Development relation | 91.67% | 100% |
| Projected relation-family margin | 0.1722 | 0.1557 |
| Projected family leakage | **26.39%** | **33.33%** |
| Leakage 门槛 | ≤18.33% | ≤18.33% |
| Fact centroid，validation / development | 93.75% / 87.50% | 93.75% / 90.63% |
| Entity probe 最低值 | 94.27% | 94.27% |
| Known/reject AUROC | 0.8993 | 0.9641 |
| ECE | 0.4726 | 0.4600 |
| TNR @ 95% known coverage | 51.39% | 58.33% |
| 严格门槛 | 6/7 | 6/7 |

两者唯一失败项都是 `projected_family_probe`。门槛由同一真实关系内 12 个 validation families 的 chance `1/12=8.33%` 加 10 个百分点得到，即 18.33%。这说明三维 relation softmax 的类别判断虽然准确，其连续置信度形状仍保留稳定的 phrase-family 信号。

拒识指标只作为诊断。当前 ECE 约 0.46–0.47，且 95% known coverage 下的 TNR 只有约 51%–58%，不能称为已经具备可部署的拒识能力。

## 科研解释与停止条件

在当前三关系合成基准内，可以支持以下有限结论：

1. 关系真值下的实体—关系事实几何可行。
2. 冻结公共语义编码器能够处理此前失败的 lexical OOD 关系判别。
3. 剩余失败集中在“传给下游的连续 relation 表示仍携带措辞家族信号”。
4. 单个 E5-base 候选没有修复该问题，因此不再扩大模型，也不再继续调旧 canonicalizer 损失。

不能据此声称“通用语义问题已经解决”或“输出契约是唯一原因”。还需要独立人工复核、更广的关系集合和后续结构性验证。

## 推荐下一步：C2.4 Discrete Relation Contract Audit（未执行）

下一阶段应先预注册离散关系接口，而不是进入 private memory：

- 冻结选中的 public semantic encoder 和 relation head；
- private-memory 路由端只接收 hard one-hot `relation_id`；
- confidence 只进入独立的 reject gate，不进入 memory key 或查询向量；
- 继续使用冻结 entity branch，先验证 slot/fact retrieval，不做 LM 注入；
- 模型与阈值选择仍只使用 public train、expanded public validation 和 private validation；development 只做冻结后诊断。

hard one-hot 会让现有 family-leakage probe 结构性归零，所以不能把该归零当作新的经验成功。C2.4 必须改测：

- known 样本的 relation route invariance；
- 同一关系跨 phrase-family 的路由一致率与最坏 family 准确率；
- confidence 是否被严格隔离在 memory 接口之外；
- ambiguous/unrelated 的 AUROC、ECE、coverage-risk 与 TNR；
- 离散 relation contract 下的 validation/development fact retrieval。

只有新协议通过、基准完成独立人工复核且模型/阈值全部冻结后，才应创建全新的独立 confirmation pool。旧 seal 永久不可复用。
