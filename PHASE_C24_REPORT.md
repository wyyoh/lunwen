# Stage C2.4 报告：Discrete Relation Contract Audit

## 技术摘要

Stage C2.4 已在固定的 C2.3 S5 路线上完成一次正式 CUDA 审计。结果需要分成两层理解：

- **离散 memory contract 的结构目标已实现。** D2/D3 的 memory 边界只接收 `RelationId`、`entity_embedding` 和被选中的单个 relation bucket；连续 relation 值、confidence、semantic embedding、原始文本和 phrase-family 信息均不进入 memory。`cross_relation_candidate_count=0`，所有被拒请求的 memory access count 为 0。
- **完整的闭集验收未通过。** 10 项门槛通过 6 项。新的 provisional locked audit 上，relation family macro accuracy 为 0.8194、worst-family accuracy 为 0.25；D2 development fact Top-1 为 0.8385、MRR 为 0.8698，均略低于预注册门槛。
- **开放集 reject guard 未通过。** calibration 选择了 G3 definition-similarity margin（temperature 0.5，threshold 0.140404）；在 provisional locked audit 上 AUROC 为 0.8704、known coverage 为 0.7222、ambiguous false accept 为 0.3333、worst-family false accept 为 1.0。七项门槛只通过两项。
- **当前不能进入 C3。** `public_benchmark_human_reviewed=false`，且本阶段按协议没有创建新 confirmation pool；因此即使接口隔离已经成立，`c3_eligible=false`。

有限结论如下：

> 闭集离散关系契约未通过完整研究门槛，但其结构隔离子目标通过。开放集拒识守卫未通过。当前 `c3_eligible=false`。本阶段没有训练 private memory，没有加载 private answer，没有执行答案注入或密钥攻击。

## 审计范围与冻结协议

本阶段只审计关系路由契约、answer-free 闭集事实槽位检索和先于 memory access 的拒识守卫。没有训练模型或 private value memory，也没有修改 core。

固定输入如下：

| 项目 | 冻结值 |
|---|---|
| relation encoder | `BAAI/bge-small-en-v1.5` |
| revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| S5 view | `entity_masked_strip_suffix` |
| ridge head SHA-256 | `808d8182ca3b1dbc957fc227bd3038fe59cf920aa2bdf72a0e3669b1ca880a22` |
| D0 alpha | 0.5 |
| G3 definition encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| G3 revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| G3 view | C2.3 固定的 `S2:phrase_only` |
| 正式设备 | NVIDIA GeForce RTX 5060 Laptop GPU, CUDA 13.0 |
| 正式审计代码/协议提交 | `86e81c3ee466fd2c13a0396bf9ffaa70d3bc45cb` |

公开数据协议物化为：

| Split | known | ambiguous | unrelated | 总数 | 用途 |
|---|---:|---:|---:|---:|---|
| public_train | 54 | 0 | 0 | 54 | 只拟合固定 ridge head；本次复用冻结 head |
| public_calibration | 54 | 18 | 18 | 90 | 选择 score、temperature、threshold |
| public_locked_audit | 72 | 24 | 24 | 120 | 冻结后一次性 provisional 审计 |

三段数据以 family、entity 和 frame 为单位隔离。manifest 报告 exact、substring/containment、normalized-token signature、lemma-bigram、family ID、entity 和 frame 均无碰撞，并额外通过了对全部 C2.3 公私 phrase views 的历史新颖性检查。

这里的 normalized-token 检查采用预注册定义：完整有序 token 序列相等，或完整 unique-token set 相等，才构成 collision；它不等于“任意单词都不能复用”。locked audit 仍是 `curated_draft_pending_independent_human_review`，120 条人工审核记录全部为 `pending`，所以所有 locked 指标均为 provisional。

选择顺序严格为：固定 C2.3 S5 → calibration 选择 guard → 冻结 guard → development 评估一次 → locked audit 评估一次。development 和 locked audit 均未参与选择；没有创建或读取任何 Stage C2.4 confirmation 内容。

## 离散接口真正切断了连续 memory 输入

实现使用类型化的 `RelationId`、`AcceptedRoute`、`RejectedRoute` 和 `RouteResult`。`retrieve_from_bucket` 的公开签名只允许：

```python
retrieve_from_bucket(
    entity_embedding: torch.Tensor,
    relation_id: RelationId,
    bucket_prototypes: dict[RelationId, torch.Tensor],
) -> RetrievalResult
```

D0–D3 的边界差异如下：

| 变体 | relation 到 memory 的表示 | 搜索范围 | reject gate |
|---|---|---|---|
| D0 | 连续 softmax，乘以 alpha 后拼接 | 全局 facts | 无 |
| D1 | argmax 后的 hard one-hot | 全局 facts | 无 |
| D2 | 仅 `RelationId` | 被选 relation bucket | 无 |
| D3 | 仅 AcceptedRoute 中的 `RelationId` | 被选 relation bucket | 有，且先于 memory access |

D2/D3 的正式契约值全部符合结构预期：

| 契约指标 | D2 | D3 |
|---|---:|---:|
| continuous relation values exposed | false | false |
| confidence exposed | false | false |
| semantic embedding exposed | false | false |
| raw text exposed | false | false |
| phrase family exposed | false | false |
| cross-relation candidate count | 0 | 0 |
| rejected-query memory access count | 0 | 0 |

confidence 只在 reject gate 内部存在。memory API 不接受 confidence/logits/probabilities/margin 参数；改变上游 confidence 不会改变或扩展 memory input。NaN、非有限 score、非法 threshold 和未知 relation 都默认 fail closed。

这证明连续 phrase-family 信息不再通过 relation vector 进入 memory contract，但不能被表述为“经验 leakage 为零”，更不能推出密码学安全。离散的错误 relation ID 仍可携带 family-dependent 的误路由模式。

## D1 没有灾难性损失，D2 的主要收益是结构隔离

D0 精确复现了 C2.3 的六项冻结数值，绝对误差均为 0。D0–D2 的 answer-free private retrieval 结果为：

| 变体 | validation Top-1 | development Top-1 | validation row 1-NN | development row 1-NN | validation MRR | development MRR |
|---|---:|---:|---:|---:|---:|---:|
| D0 continuous | 0.9375 | 0.8750 | 0.9063 | 0.7969 | 0.9519 | 0.9104 |
| D1 hard one-hot | 0.9375 | 0.8385 | 0.9375 | 0.8438 | 0.9604 | 0.8916 |
| D2 relation bucket | 0.9375 | 0.8385 | 0.9375 | 0.8438 | 0.9609 | 0.8698 |

D1 相对 D0：validation Top-1 不变，development Top-1 下降 3.65 个百分点；row 1-NN 在 validation/development 分别上升 3.13/4.69 个百分点；MRR 在 validation 上升 0.85 个百分点、在 development 下降 1.88 个百分点。因此，去掉连续 softmax 没有造成统一、显著的崩塌，但也不能声称完全无损。

D2 与 D1 的 Top-1 和 row 1-NN 完全相同；validation MRR 只高 0.06 个百分点，而 development MRR 低 2.18 个百分点。现有结果不支持“D2 的检索性能比 D1 更稳定”这一经验结论。D2 的明确优势是候选隔离：D1 仍有 32 个 cross-relation candidates，D2 为 0。

D2 的 silhouette 和 fact-over-template margin 是按 true-relation partition 计算的 evaluation-only oracle geometry，只能作为诊断，不能作为 predicted-route 稳定性的主要证据。

D3 在 private validation 上 coverage 为 1.0、accepted fact accuracy 为 0.9375；在 development 上 coverage 为 0.75、accepted fact accuracy 为 0.8194，按全部 query 计的 fact Top-1 为 0.6146。D3 的 MRR 没有计算，产物明确标记为 `not_evaluated_for_D3`，没有用 0 代替缺失值。

## 新 phrase families 暴露出离散误路由

公共 train 和 calibration 的 route micro/family-macro/worst-family accuracy 均为 1.0；private validation relation accuracy 为 1.0，frozen development relation accuracy 为 0.9167。新的 provisional locked audit 则明显下降：

| 指标 | public locked audit |
|---|---:|
| relation micro accuracy | 0.8194 |
| relation family macro accuracy | 0.8194 |
| worst-family accuracy | 0.2500 |
| same-family cross-frame route agreement | 0.7130 |
| same-relation cross-phrase-family route agreement | 0.7111 |

locked confusion matrix（行是真实 relation，列是预测 relation）为：

| 真实 \ 预测 | access_code | city_code | registry_id |
|---|---:|---:|---:|
| access_code | 18 | 0 | 6 |
| city_code | 0 | 23 | 1 |
| registry_id | 6 | 0 | 18 |

错误主要发生在 `access_code` 与 `registry_id` 之间。family accuracy 从 0.25 到 1.0 不等：`locked_guarded_service_signin_value` 只有 0.25，`locked_account_opening_secret_phrase` 为 0.50；多个 city-code families 为 1.0。按 relation 汇总的跨 phrase-family agreement 也呈现差异：access_code 0.5917、registry_id 0.6250、city_code 0.9167。

因此，连续 family channel 已从 memory API 中结构性删除，但离散分类器仍明显依赖 phrase-family 泛化质量。

## 闭集完整门槛未通过

D2 的闭集验收通过 6/10 项：

| 检查 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| public locked family macro accuracy | 0.8194 | ≥ 0.95 | 失败 |
| public locked worst-family accuracy | 0.2500 | ≥ 0.85 | 失败 |
| private validation relation accuracy | 1.0000 | ≥ 0.95 | 通过 |
| development relation accuracy | 0.9167 | ≥ 0.85 | 通过 |
| development fact centroid Top-1 | 0.8385 | ≥ 0.85 | 失败 |
| development MRR | 0.8698 | ≥ 0.90 | 失败 |
| oracle gap | 0.0729 | ≤ 0.08 | 通过 |
| cross-relation candidates | 0 | = 0 | 通过 |
| continuous relation exposed | false | false | 通过 |
| confidence exposed | false | false | 通过 |

所以不能将当前状态概括为“closed-set contract 已通过”。准确表述是：**类型和候选空间的结构契约通过，但闭集性能与新 family 泛化的完整 readiness 未通过。**

## G3 最优于候选，但 reject guard 仍不合格

三类 score 只在 public calibration 上比较：

| Score | 选中 | AUROC | AUPR | ECE | TNR@95 | ambiguous FA | unrelated FA | worst-family FA | 通过门槛数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 max softmax | 否 | 0.8992 | 0.9267 | 0.2807 | 0.5000 | 0.6667 | 0.3333 | 1.0000 | 1/7 |
| G2 top-logit margin | 否 | 0.8868 | 0.9176 | 0.2807 | 0.4722 | 0.6667 | 0.3889 | 1.0000 | 1/7 |
| G3 definition margin | 是 | 0.9306 | 0.9490 | 0.5083 | 0.8333 | 0.3333 | 0.0000 | 1.0000 | 3/7 |

预注册选择规则选中 G3，temperature 为 0.5，threshold 为 0.140404。temperature 只用于 calibration；它不改变排序型 AUROC。

冻结后的 G3 表现为：

| 指标 | calibration | provisional locked audit | 研究门槛 |
|---|---:|---:|---:|
| known detection AUROC | 0.9306 | 0.8704 | ≥ 0.95 |
| known detection AUPR | 0.9490 | 0.9319 | 报告项 |
| ECE | 0.5083 | 0.3407 | ≤ 0.15 |
| AURC | 0.0000 | 0.1040 | 报告项 |
| TNR@95 calibration coverage threshold | 0.8333 | 0.8333 | ≥ 0.80 |
| overall false accept | 0.1667 | 0.1667 | 报告项 |
| ambiguous false accept | 0.3333 | 0.3333 | ≤ 0.20 |
| unrelated false accept | 0.0000 | 0.0000 | ≤ 0.10 |
| worst reject-family false accept | 1.0000 | 1.0000 | ≤ 0.30 |
| known coverage | 1.0000 | 0.7222 | ≥ 0.95 |
| known accepted accuracy | 1.0000 | 0.8462 | 报告项 |

这里的 locked `TNR@95` 是在 calibration 上以 95% known coverage 选择阈值后，将该冻结阈值应用到 locked audit 的结果，不是在 locked audit 上重新选阈值。

ambiguous 明显比 unrelated 更难：locked ambiguous AUROC 为 0.8148、false accept 为 0.3333；unrelated AUROC 为 0.9259、false accept 为 0。两个 ambiguous families 的 false accept 为 1.0。正式样本中没有 NaN/invalid score；这类输入的 fail-closed 行为由接口测试验证，空集指标按约定记录为 1.0。正常但模糊的请求仍无法可靠拒绝。

开放集门槛最终只通过 TNR@95 和 unrelated false accept 两项，共 2/7，故 `open_set_reject_guard_ready=false`。

## Eligibility 与 C3 停止条件

| 状态 | 值 |
|---|---|
| closed_set_discrete_contract_ready | false |
| open_set_reject_guard_ready | false |
| public_benchmark_human_reviewed | false |
| new_confirmation_pool_created_after_freeze | false |
| c3_eligible | false |

严格公式为：

```python
c3_eligible = (
    closed_set_discrete_contract_ready
    and open_set_reject_guard_ready
    and public_benchmark_human_reviewed
    and new_confirmation_pool_created_after_freeze
)
```

当前四个必要条件均未同时成立。本阶段禁止创建 confirmation，所以最后两项中的 confirmation 条件按设计保持 false；旧 confirmation seal 没有复用。

## 对十个研究问题的直接回答

1. **D1 去掉 continuous softmax 后，事实检索是否下降？** 部分下降而非全面下降。validation Top-1 不变，development Top-1 下降 3.65 个百分点；row 1-NN 反而提升，MRR 一升一降。
2. **D2 relation bucket 是否比 D1 更稳定？** 当前证据不支持。两者 Top-1/row 1-NN 相同，development MRR 的 D2 更低；D2 的确定收益是结构性候选隔离。
3. **D2 是否真正实现 cross-relation candidate 隔离？** 是。实际 memory call 只可见一个被选 bucket，`cross_relation_candidate_count=0`。
4. **continuous family leakage 是否从 memory contract 中结构性移除？** 是，连续 relation/confidence/semantic/text/family 字段均不在接口中；但这不是“经验 leakage 归零”或安全证明。
5. **离散误路由是否仍有 family-dependent pattern？** 有。locked family accuracy 为 0.25–1.0，access 与 registry 相互混淆，city families 明显更稳定。
6. **D3 reject gate 是否达到研究门槛？** 否，只通过 2/7 项。
7. **ambiguous 和 unrelated 哪类更难拒绝？** ambiguous；其 locked false accept 为 0.3333，而 unrelated 为 0。
8. **当前是否只通过 closed-set contract？** 否。只通过了闭集契约的结构子目标，完整 closed-set readiness 仍为 false。
9. **为什么仍不能进入 C3？** closed/open readiness 均失败，locked benchmark 尚未独立人工审核，而且本阶段没有创建新的 confirmation pool。
10. **下一步是继续改 reject guard，还是准备人工审核和新 confirmation？** 应继续改进 reject guard，同时解决新 phrase-family 的离散路由泛化；当前结果不能支持创建新 confirmation。人工审核可用于验证本次 provisional benchmark 的标签质量，但不能把失败结果变成 eligibility。

## 下一步与协议约束

当前 locked audit 已被正式打开一次，不能用它反向调 threshold、score 或 relation head。建议下一步：

1. 将本次 locked audit 固化为 C2.4 的负结果和诊断证据。
2. 针对 ambiguity rejection 与 access/registry 泛化提出新的 C2.4b 方法，并建立新的 calibration/locked-audit 协议；不能在当前 locked 数据上调参后继续把它叫作无偏审计。
3. 当前 120 条 locked draft 可进行真正的双人独立人工复核，以判断数据本身是否有效；不得伪造审核结果。
4. 只有新的闭集与开放集门槛都通过、公开基准完成独立人工审核后，才讨论创建一个 freeze 后的新 confirmation pool。该动作不属于本阶段。

本结果没有证明访问控制安全、密码学安全或机器遗忘。

## 完整性、限制与可复现性

- 正式审计前的 Git worktree 为 clean，产物记录的提交为 `86e81c3ee466fd2c13a0396bf9ffaa70d3bc45cb`。
- 所有 11 个 JSON 产物均按严格 JSON 解析并通过有限数检查；没有 NaN 或 Infinity。本地 artifact manifest 校验了 19 个文件哈希，其中 14 个非缓存产物按原始字节提交 Git，5 个 `.pt` embedding caches 可由命令重建且按仓库规则不提交。
- `public_benchmark_human_reviewed=false`；120 条 review 记录均为 `pending`，reviewer/adjudicated 字段为空。
- core 模型没有加载或修改；源 core provenance SHA-256 `b141f3c08fcc9c56a6ca6a6c14169fc001613efa7ca98905b8b642070a01d0cb` 与冻结值一致。answer-free cache、entity checkpoint 文件和 R3 state 的 before/after SHA-256 均一致。由于 core 本身未加载，报告不虚构 runtime core tensor rehash。
- 单次固定种子审计没有置信区间；locked 指标还受待人工审核状态限制。
- 没有保存 private answer、密钥或 optimizer state；embedding tensor cache 按仓库规则不提交 Git。Stage C2.4 artifact 路径禁用 Git 换行归一化，以保证远程仓库中已提交产物的字节级 SHA-256 不变。
- 精确门槛审计和布尔接口检查适合表格呈现，本报告未用图形替代数值表。

可复现命令：

```powershell
keyed-gram stage-c24-prepare `
  --config configs/stage_c24.yaml

keyed-gram stage-c24-audit `
  --config configs/stage_c24.yaml `
  --output-dir artifacts/stage_c24 `
  --device cuda

pytest -q
```

主要证据文件：

- `artifacts/stage_c24/stage_c24_summary.json`
- `artifacts/stage_c24/stage_c24_ablation.csv`
- `artifacts/stage_c24/D0/evaluation.json` 至 `D3/evaluation.json`
- `artifacts/stage_c24/reject_score_candidates.csv`
- `artifacts/stage_c24/protocol_status.json`
- `artifacts/stage_c24/public_benchmark_manifest.json`
- `artifacts/stage_c24/public_locked_audit_review.csv`
- `artifacts/stage_c24/artifact_sha256_manifest.json`
