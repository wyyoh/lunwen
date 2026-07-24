# Stage C2.4c：Exploratory Router Evaluation

## 结论

本阶段在已冻结的 C2.4b v2.1 合成压力测试上完成了一次
`calibration → development → locked scoring`。评估是
`exploratory_non_independent`：同一单模型 AI 参与过数据修订并看过
locked 文本，因此结果不是独立验证、真人审核 benchmark 或部署安全证据。

最终 router 为 **R3**。闭集探索状态为
`passed`，开放集探索状态为
`failed`；readiness 与 C3 门禁仍为 false。

这里的 `passed/failed` 只对应 C2.4c 预注册的较低探索判据。正式研究门槛
未评估；`closed_set_selective_router_ready` 与
`open_set_abstention_ready` 均保持 false。

## 数据与执行协议

- benchmark 固定为 `c24b-public-selective-router-v2.1`，状态为
  `frozen_for_exploratory_model_scoring`。
- train/calibration/locked 分别为 72/160/160 行；运行前后 SHA-256
  保持 `9165b6…b5f4`、`88bc64…ffaf`、`42e2ca…36fe`。
- 104 个 family 的单模型 AI 审核含 67 个 passed 与 37 个
  boundary-retained；后者描述合成构造边界，不是 37 个错标。
- `public_benchmark_human_reviewed=false`，
  `independent_external_validation=false`，
  `formal_calibration_allowed=false`。
- router 仅在 `public_calibration_v2_1` 选择；development 使用确定性重建的
  C2.3 public validation+reject 360 行，只运行一次且不参与选择或状态。
- v2.1 locked 在 router freeze 与 development 完成后只评分一次。
  没有 `protocol_incident.json`，也没有根据 development/locked 结果回改参数。

## Locked R0–R4

| Router | Known coverage | Accepted route acc. | Worst family | Ambiguous reject | Unrelated reject | FMAR | Wrong bucket | Safe coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 100.00% | 80.21% | 0.00% | 0.00% | 0.00% | 100.00% | 19.79% | 80.21% |
| R1 | 100.00% | 87.50% | 0.00% | 0.00% | 0.00% | 100.00% | 12.50% | 87.50% |
| R2 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% | 100.00% |
| R3 | 95.83% | 100.00% | 0.00% | 87.50% | 62.50% | 25.00% | 0.00% | 95.83% |
| R4 | 87.50% | 100.00% | 0.00% | 87.50% | 75.00% | 18.75% | 0.00% | 87.50% |

R3 相对 R0 将 ambiguous FMAR 从 100% 降至 12.5%，相对下降 87.5%；
overall FMAR 从 100% 降至 25%。代价是 known coverage 从 100% 降至
95.83%。R3 的 accepted-route accuracy 为 100%，wrong-bucket access 为
0%，safe coverage 为 95.83%。

这并不满足预注册的“方法有效”整体判据，因为 overall FMAR 仍高于 20%。
它也不满足“方法强有效”：unrelated rejection 只有 62.5%，worst-family
accuracy 为 0%。闭集探索状态之所以为 passed，是因为较低闭集判据关注
accepted precision、coverage、safe coverage、wrong bucket 与 memory
contract；它不能掩盖一个 locked known family 被全部拒绝。

## 冻结参数

- R1 aggregation：`top_k_mean_3`
- R2 encoder/ridge：`e5_base:0.1`
- R3：`R2:0.2,0.4,0.4:m=0.1`
- R4：`R2:alpha=0.25`
- router freeze SHA-256：`f440f9782b24d7c88b3bcea1d8d97aee9cc9b4544c94209071957af042aae134`

R0 使用同一 BGE revision、C2.3 public train、view、class order 与
ridge 配方的确定性功能重建。原始 checkpoint 按仓库策略未提交，
远端 refs/releases/actions 亦不存在；重建在 216 条历史 public
 validation 上复现了相同准确率和唯一错误，但 `.pt` SHA 不同，
因此不声称新输入 logits 与历史 checkpoint 逐位相同。

R4 的 evidence source 与 alpha 在同一 calibration split 上选择，
所以不声称标准 split-conformal 的名义有限样本覆盖保证；在 lexical/
semantic shift 下尤其没有严格保证。

Calibration 上 R3 与 R4 的 FMAR 同为 18.75%，ambiguous/unrelated
rejection 同为 87.5%/75%，但 R3 safe coverage 为 95.83%，高于 R4 的
87.5%，所以预注册选择合法地保留 R3。Locked 上 R4 的 FMAR 降至 18.75%，
低于 R3 的 25%，但 safe coverage 同时降至 87.5%；locked 结果不能用于
事后改选 R4。

## 研究问题回答

1. **R1 是否提升 lexical-family OOD？** 部分提升。Known family macro 从
   R0 的 80.21% 升至 87.50%，wrong-bucket 从 19.79% 降至 12.50%；
   但 `registry_id → access_code` 从 25% 恶化至 37.5%，所以 definition
   ensemble 没有稳定解决 access/registry 边界。
2. **R2 是否减少 access/registry 混淆？** 是。在本合成 locked 上 R2 的
   known accuracy、family macro、worst family 均为 100%，
   `access→registry` 与 `registry→access` 都为 0。这个结果只证明
   pairwise public evidence 在该压力集上有用，不能外推到真实分布。
3. **R3 是否把 ambiguous false accept 转成多候选 reject？** 没有按预期
   机制实现。R3 将 rejection 提高到 87.5%，但 ambiguous multi-label-set
   rate 为 0；28/32 ambiguous 被判为 `unknown`，另 4 条
   `v21_locked_ambiguous_protected_registry_reference` 仍被接受到
   `access_code`。因此错误访问显著下降，但不是通过多个 relation 同时获支持。
4. **R4 是否改善 safe coverage？** 没有。相对 R3，R4 locked FMAR 改善
   6.25 个百分点，但 safe coverage 恶化 8.33 个百分点，且多候选率仍为
   0；当前证据不支持用更复杂的 R4 替代 R3。
5. **Worst-family 是否达标？** 没有。R3 的
   `v21_locked_city_neighborhood_survey` 四个 frame 全被 unknown-reject，
   worst-family accuracy 为 0。
6. **Ambiguous rejection 是否达标？** 达到探索强判据 75% 和原研究目标
   80%，结果为 87.5%；但 reject 类型主要是 unknown，而不是多候选
   ambiguous，构念解释受限。
7. **Unrelated rejection 是否稳定？** 不稳定，只有 62.5%。
8. **FMAR 是否足够低？** 不足。R3 overall FMAR 为 25%，高于探索判据
   20% 和强判据 10%。三个整 family 的 hard unrelated 被错误访问：
   `archive_retention→registry`、`parcel_tracking→registry`、
   `zone_font→city`。
9. **Relation bucket 结构隔离是否保持？** 保持。所有 variant 的
   `cross_relation_candidate_count=0`；reject 不访问 memory；accept 只访问
   一个由 `RelationId` 指定的 bucket。
10. **是否具备创建新 confirmation pool 的条件？** 不具备。开放集探索失败，
    且没有真人审核或独立外部验证。
11. **为什么仍不能进入 C3？** 没有 confirmation，formal readiness 未评估，
    open-set status 失败，benchmark 非独立且强合成；
    `c3_eligible=false`。
12. **下一步是什么？** 继续 router/OOD 构念研究，而不是创建 confirmation。
    不得在已打开的 v2.1 locked 上调参，也不应继续循环制造 v2.2/v2.3。
    下一份证据应来自既有公开人工标注意图/OOD 数据或真正独立外部验证。

## 错误分析

### Lexical OOD 与 access/registry

R0 的主要 known 错误仍集中在 registry：32 条 registry 中 8 条被路由到
access、10 条到 city；access 没有路由到 registry。R1 把 registry→city
清零，却把 registry→access 提高到 12/32。R2 的独立 pairwise evidence
消除了本 locked 上的这些混淆。R3 没有 wrong-bucket access，但把
`city_neighborhood_survey` 全部拒绝；这说明阈值把部分分类风险转换成了覆盖损失。

### Ambiguity

R3 没有形成任何多 relation candidate set。它对 28 条 ambiguous 给出
unknown，对 `protected_registry_reference` 的四个 frame 均接受 access。
因此“集合式路由降低 FMAR”在结果层面成立，但“检测到两个合理 relation 后以
ambiguous 拒绝”的机制假设没有得到支持。

### Unknown

Hard unrelated 仍会因共享领域词被接受。`archive retention` 与
`parcel tracking` 被 registry evidence 覆盖，`zone font` 被 city evidence
覆盖，说明定义/二分类 support 对非目标 tail property 仍过宽。R3 对 unrelated
另有 12.5% 因 minimum-margin 返回 ambiguous，其余拒绝主要为空集合 unknown。

### Coverage trade-off

完整 risk–coverage 曲线保存在每个 `R*/evaluation.json` 的
`reject_quality.risk_coverage`；R3 AURC 为 0.1352。Calibration 的全部
threshold/margin 候选保存在 `router_selection.csv`，不是只保存最终点。
Locked 上 R3 与 R4 的比较表明，进一步降低 FMAR 会直接损失 safe coverage，
不能用“拒绝更多”单独解释为更安全。

Locked known rows 的 `hard_negative=false` 且 `minimal_pair_id` 为空，因此
locked hard-minimal-pair accuracy 不可评估；输出中的空分母值 0 不能解释成
“模型在 minimal pair 上为 0%”。Calibration 的 hard-negative 指标参与了选择，
但不得替代 locked 证据。

## 访问控制解释

- R0 ambiguous FMAR：`100.00%`
- R3 ambiguous FMAR：`12.50%`
- 相对下降：`87.50%`
- R3 safe coverage：`95.83%`

fact retrieval 数字来自确定性的公开 answer-free typed-bucket contract
 probe，与路由正确性等价；它不是独立的 learned entity/fact retrieval
 结果，也没有加载 private value memory。

该 contract probe 的 accepted fact Top-1 为 100%，all-query fact Top-1
为 95.83%，MRR 为 1.0；这些数字只说明正确离散路由可进入相同公开实体的单一
typed bucket，不是独立 fact/entity 模型的泛化评估。

## 有限表述

选择性离散路由闭集正式门槛未评估。
开放集弃权正式门槛未评估。
离散 memory contract 保持。
`ready_to_create_new_confirmation_pool = false`。
`c3_eligible = false`。
本阶段没有训练 private memory，没有加载 private answer，没有执行答案注入或密钥攻击。

v2.1 是受控 lexical-OOD/ambiguity 合成压力测试，37 个 boundary family
反映合成构造限制而非 37 个错标；结论不得外推为自然语言真实分布表现。
