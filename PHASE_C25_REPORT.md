# Stage C2.5：External OOS Generalization Audit

## 结论摘要

Stage C2.5 已在冻结代码、冻结模型、官方 split 和 10 个预注册
supported-intent partitions 上完成一次 CLINC150 与 BANKING77 外部审计。

本阶段得到的是清晰的负结果：

1. C2.4c 中观察到的 R2 pairwise evidence 优势没有外部复现。R2 的
   known accuracy 在 CLINC150、BANKING77 open-intent 和 BANKING77
   closed-set 上分别比 R0 低 1.57、3.33 和 3.83 个百分点。
2. R3 确实将 FMAR 从 100% 降到 CLINC150 的 1.22% 和 BANKING77
   open-intent 的 0.33%，但 safe coverage 同时降到 25.83% 和
   17.74%。这是明显的“大量拒绝 known”结果，不是可用的安全覆盖改进。
3. 多候选机制仍几乎没有发挥作用。R3 的 multi-candidate-set rate
   只有 0.41% 和 0.03%；绝大多数拒绝来自空集合 `unknown`。
4. 简单 `MAX_THRESHOLD` 在 coverage/FMAR trade-off 上明显优于 R3，
   但在逐 seed 的“wrong-bucket 不得高于 R0”条件下，CLINC150 仅
   3/10 seeds 通过，BANKING77 为 8/10，仍没有跨基准稳定通过。
5. 因此当前 R2+R3 外部泛化假设未获支持，应按预注册停止条件冻结为
   有限机制案例，不继续增加 router 复杂度，不进入 C3。

最终状态：

```text
external_oos_validation_status = failed
external_pairwise_evidence_validated = false
external_selective_abstention_validated = false
exploratory_c3_allowed = false
task_specific_router_ready = false
c3_eligible = false
```

## 1. 冻结与数据来源

实验代码冻结提交：

```text
6f47ffd7ab8816f7873fd2228ae07dad9ff42ffd
```

代码冻结发生在 prepare、calibration 和 test prediction 之前。

### 1.1 CLINC150

- 官方仓库：<https://github.com/clinc/oos-eval>
- 固定 revision：
  `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- 论文：<https://aclanthology.org/D19-1131/>
- 许可：官方仓库 `CC-BY-3.0`
- `data_full.json` SHA-256：
  `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`
- `domains.json` SHA-256：
  `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a`

官方 full split 经代码验证为：

```text
150 intents × 100 train / 20 validation / 30 test
OOS = 100 train / 100 validation / 1000 test
10 domains × 15 intents
```

### 1.2 BANKING77

- 官方仓库：<https://github.com/PolyAI-LDN/task-specific-datasets>
- 固定 revision：
  `57ec275d8078af65b7731c2a98be812d844a6d6b`
- 论文：<https://aclanthology.org/2020.nlp4convai-1.5/>
- 许可：官方仓库 `CC-BY-4.0`
- `train.csv` SHA-256：
  `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- `test.csv` SHA-256：
  `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`
- `categories.json` SHA-256：
  `53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63`

代码验证了 10,003 条 train、3,080 条 test、77 个 intents，以及
每个 intent 固定 40 条 test。

数据文件只存在于被 Git 忽略的 `.downloads/`，没有提交到仓库。

## 2. 预注册 partitions

固定 seeds：

```text
2501, 2502, 2503, 2504, 2505,
2506, 2507, 2508, 2509, 2510
```

### 2.1 CLINC150

每个 seed 在每个官方 domain 中抽取 5 个 supported intents，共 50 个。
其余 100 个 intents 与官方 OOS 均作为 unknown：

```text
fit:
  supported-intent official train

calibration:
  supported-intent official validation as known
  unsupported-intent official validation as heldout-intent OOS
  official oos_val

test:
  supported-intent official test as known
  unsupported-intent official test as heldout-intent OOS
  official oos_test
```

### 2.2 BANKING77 open-intent

每个 seed 以完整 intent 为单位划分：

```text
39 supported intents
19 calibration-OOS intents
19 test-only OOS intents
```

supported intents 的官方 train 按 intent、seed 固定为 80% fit / 20%
calibration-known。19 个 calibration-OOS intents 只用于 calibration。
19 个 test-only OOS intents 在 calibration 中完全不可见，只有其官方
test rows 进入一次性审计。

三组 intent 集严格不相交，没有把同一 intent 的样本同时拆成 known 和
OOS。

### 2.3 BANKING77 closed-set

全部 77 intents 参与；官方 train 按 intent 固定为 80% fit / 20%
calibration，官方 test 用于一次性 closed-set 审计。

本阶段没有生成 synthetic OOS，没有修改官方文本或标签，也没有根据
test 结果删除 intent 或调整 partition。

## 3. 模型与路由变体

所有变体使用同一固定公开编码器：

```text
model_id = intfloat/e5-base-v2
revision = f52bf8ec8c7124536f0efb74aca902b2995e5bcd
model.safetensors SHA-256 =
d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693
```

### R0：forced multiclass ridge

对每个 supported-intent partition，只用 fit rows 的冻结 E5 embedding
拟合 deterministic multiclass ridge，并强制 argmax。ridge strength 只在
calibration-known accuracy 上选择。

这是适用于动态外部 intent 集的 R0 analogue，不是把 C2.4c 的三关系
BGE checkpoint 直接错误套用到 50/77 类任务。

### R2：independent pairwise evidence

每个 supported intent 使用：

- public-train centroid；
- 8 个由 deterministic farthest-first 选择的 public-train prototypes；
- 官方 intent 名的 E5 embedding；
- query/intent pair 的 6 维独立特征：
  centroid cosine、label cosine、prototype max、prototype top-3 mean、
  两个差值特征。

一个共享小型 public binary ridge 只使用 fit rows 与每条正例的 3 个
hard-negative intent pairs。输出是每个 intent 的独立 evidence，不经过
multiclass softmax。

这也是外部任务的机制 analogue；它检验“独立 pairwise evidence”能否
泛化，不声称与 C2.4c 三关系 R2 的权重或 ontology 相同。

### MAX_THRESHOLD

对 R2 top-1 evidence 使用 calibration-only global threshold：

```text
top1 >= threshold -> Accept(argmax)
otherwise         -> Reject("unknown")
```

该对照不使用多候选集合或 margin。

### R3

```text
candidate_set = {intent | evidence >= global_threshold}

0 candidates -> Reject("unknown")
1 candidate  -> Accept，除非 top1-top2 margin 不足
>1 candidates -> Reject("ambiguous")
```

threshold 与 margin 只在 validation 上选择。选择目标先检查：

```text
accepted-route accuracy >= 0.90
known coverage >= 0.80
safe coverage >= 0.75
```

若无候选同时满足，则先最大化通过的 gate 数，再降低 calibration FMAR。
这条已冻结的规则导致 R3 在无可用候选时选择高阈值；没有在看到 test 后
更换目标函数。

R4 已预注册为 deferred；C2.4c 中 conformal 没有优于简单 R3，本阶段
不再增加同时变化的机制。MASSIVE 同样按预注册暂缓。

## 4. 一次性执行边界

执行顺序：

```text
prepare/freeze
  -> calibration (official train/validation only)
  -> router_freeze_manifest
  -> deterministic refit + state SHA verification
  -> test_audit started
  -> test prediction exactly once
```

prepare 会读取官方文件以验证 SHA、schema 和 split 行数，但不产生模型
prediction。calibration manifest 明确记录：

```text
test_rows_loaded = false
test_labels_or_predictions_used_for_selection = false
```

只有 30 个冻结 head 的 deterministic refit SHA 全部一致后，程序才将
`test_audit` 标为 `started`，随后解析 test rows 并预测。

运行后状态：

```text
prepare = completed
calibration = completed
test_audit = completed
```

再次调用 calibration 或 test audit 均被拒绝，未产生第二轮 prediction。

## 5. Calibration 冻结参数概况

### CLINC150

```text
R0 ridge:
  0.001 × 8 seeds, 0.01 × 1, 0.1 × 1

R2 ridge:
  0.1 × 9 seeds, 0.01 × 1

MAX threshold:
  0.30 × 1, 0.35 × 4, 0.40 × 4, 0.45 × 1

R3:
  threshold 0.80 × 9 seeds
  threshold 0.40 × 1 seed
  margin ∈ {0.00, 0.05, 0.10, 0.15}
```

### BANKING77 open-intent

```text
R0 ridge:
  0.001 × 7, 0.01 × 2, 0.1 × 1

R2 ridge:
  0.0001 × 5, 0.001 × 1, 0.01 × 1, 0.1 × 3

MAX threshold:
  0.30 × 1, 0.35 × 5, 0.40 × 4

R3:
  threshold = 0.80 and margin = 0.00 for all 10 seeds
```

### BANKING77 closed-set

```text
R0 ridge = 0.001 for all seeds
R2 ridge:
  0.0001 × 1, 0.001 × 1, 0.01 × 3, 0.1 × 5
```

R3 在 calibration 上已经呈现低 coverage：

| Benchmark | Known coverage | Safe coverage | FMAR |
|---|---:|---:|---:|
| CLINC150 | 24.35% | 24.22% | 1.33% |
| BANKING77 open | 16.40% | 16.22% | 0.39% |

参数已经冻结，因此没有根据这个负信号修改 threshold、margin、features
或 selection objective。

## 6. 一次性 test 结果

表内为 10 seeds 的算术均值。完整逐 seed 值、近似 95% normal CI、
risk–coverage 和 error analysis 分别见：

- `artifacts/stage_c25/per_seed_metrics.csv`
- `artifacts/stage_c25/aggregate_metrics.csv`
- `artifacts/stage_c25/risk_coverage.csv`
- `artifacts/stage_c25/error_analysis.csv`

### 6.1 CLINC150 external OOS

| Variant | Known acc. | Known cov. | Accepted route acc. | Safe cov. | Wrong bucket | FMAR | OOS recall | OOS F1 | Worst intent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 98.01% | 100.00% | 98.01% | 98.01% | 1.99% | 100.00% | 0.00% | 0.00% | 78.67% |
| R2 | 96.44% | 100.00% | 96.44% | 96.44% | 3.56% | 100.00% | 0.00% | 0.00% | 71.00% |
| MAX_THRESHOLD | 82.04% | 84.03% | 97.64% | 82.04% | 1.99% | 14.17% | 85.82% | 89.46% | 10.00% |
| R3 | 25.83% | 25.89% | 99.63% | 25.83% | 0.07% | 1.22% | 98.78% | 87.25% | 0.00% |

OOS score ranking：

| Variant evidence | AUROC | AUPR |
|---|---:|---:|
| R0 | 95.36% | 98.01% |
| R2 / MAX / R3 | 92.81% | 97.03% |

R2 不仅 known accuracy 比 R0 低，OOS ranking 也更差。

CLINC unknown 类型的 FMAR：

| Variant | Heldout-intent OOS | Official OOS |
|---|---:|---:|
| MAX_THRESHOLD | 18.02% | 2.65% |
| R3 | 1.56% | 0.19% |

真正困难的是语义完整、属于其他 intent 的 heldout-intent OOS，而不是
官方通用 OOS。

### 6.2 BANKING77 open-intent

| Variant | Known acc. | Known cov. | Accepted route acc. | Safe cov. | Wrong bucket | FMAR | OOS recall | OOS F1 | Worst intent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 94.08% | 100.00% | 94.08% | 94.08% | 5.92% | 100.00% | 0.00% | 0.00% | 78.25% |
| R2 | 90.75% | 100.00% | 90.75% | 90.75% | 9.25% | 100.00% | 0.00% | 0.00% | 60.75% |
| MAX_THRESHOLD | 78.95% | 84.33% | 93.64% | 78.95% | 5.38% | 28.13% | 71.87% | 70.37% | 42.25% |
| R3 | 17.74% | 17.90% | 99.10% | 17.74% | 0.16% | 0.33% | 99.67% | 54.16% | 1.00% |

OOS score ranking：

| Variant evidence | AUROC | AUPR |
|---|---:|---:|
| R0 | 85.96% | 73.59% |
| R2 / MAX / R3 | 86.72% | 74.91% |

R2 在 BANKING77 的 OOS ranking 略优于 R0，但 known routing 明显更差；
这个 ranking 改善没有转化为安全覆盖收益。

### 6.3 BANKING77 closed-set

| Variant | Known accuracy | Wrong bucket | Worst intent |
|---|---:|---:|---:|
| R0 | 89.32% | 10.68% | 31.00% |
| R2 | 85.49% | 14.51% | 45.00% |

R2 提高了 worst-intent 均值，但牺牲了 overall accuracy，并将
wrong-bucket access 从 10.68% 提高到 14.51%。这说明 pairwise features
重新分配了错误，而不是整体解决细粒度 intent 边界。

## 7. 多候选机制与 rejection 类型

| Benchmark | Variant | Empty-set rate | Multi-set rate | Unknown-reason rate | Ambiguous-reason rate |
|---|---|---:|---:|---:|---:|
| CLINC150 | R3 | 91.47% | 0.41% | 91.47% | 0.58% |
| BANKING77 open | R3 | 87.83% | 0.03% | 87.83% | 0.03% |

结果再次不支持“多个 intent 同时获得支持是歧义拒绝的主要机制”。R3
几乎总是因为所有 evidence 都低于高阈值而 `Reject("unknown")`。

R3 相对 MAX_THRESHOLD：

| Benchmark | FMAR 差值 R3−MAX | Safe coverage 差值 R3−MAX | R3 FMAR 更低的 seeds |
|---|---:|---:|---:|
| CLINC150 | −12.96 pp | −56.21 pp | 10/10 |
| BANKING77 open | −27.80 pp | −61.21 pp | 10/10 |

FMAR 改善的代价远大于可接受范围。R3 没有优于简单 max-score
threshold 的 risk/coverage trade-off。

## 8. 跨 seed 停止条件

### 8.1 Pairwise evidence

| Benchmark | Mean known delta R2−R0 | R2 不差于 R0 的 seeds | Passed |
|---|---:|---:|---|
| CLINC150 | −1.57 pp | 1/10 | false |
| BANKING77 open | −3.33 pp | 0/10 | false |
| BANKING77 closed | −3.83 pp | 0/10 | false |

因此：

```text
external_pairwise_evidence_validated = false
```

### 8.2 Selective R3

R3 相对 R0 的 mean FMAR reduction 为：

```text
CLINC150:       98.78%
BANKING77 open: 99.67%
```

但逐 seed 的完整 continuation gate 要同时满足：

```text
relative FMAR reduction >= 40%
wrong-bucket access 不升高
accepted-route accuracy >= 90%
known coverage >= 80%
safe coverage >= 75%
```

R3 在两个基准均为：

```text
0 / 10 seeds passed
```

失败由 known coverage 和 safe coverage 崩溃驱动，不是 accepted-known
precision 不足。

简单 MAX_THRESHOLD 的同一逐 seed gate：

```text
CLINC150:       3 / 10
BANKING77 open: 8 / 10
```

它优于 R3，但没有跨两个基准、在多数 seed 上稳定成立；CLINC 中多个
seed 的 thresholded wrong-bucket rate略高于对应 R0。

因此：

```text
external_selective_abstention_validated = false
external_oos_validation_status = failed
```

## 9. 代表性错误

### CLINC150

R0 的低准确 intent 包括：

- `shopping_list`：82.08%
- `transactions`：82.50%
- `pin_change`：83.33%

R2 的低准确 intent 更严重：

- `change_user_name`：68.33%
- `yes`：80.00%
- `transactions`：80.83%
- `user_name`：81.11%

pairwise intent-name/prototype features没有解决 `user_name`、
`change_user_name`、`what_is_your_name` 等细粒度边界。

### BANKING77

open-intent R2 的代表性低准确 intent：

- `topping_up_by_card`：59.17%
- `transfer_not_received_by_recipient`：63.00%
- `balance_not_updated_after_bank_transfer`：67.50%

closed-set R0 的 `virtual_card_not_working` 只有 31.00%，而 R2 closed-set
的最低类别转移为：

- `transfer_not_received_by_recipient`：50.25%
- `topping_up_by_card`：50.50%
- `balance_not_updated_after_bank_transfer`：54.00%

错误集中在共享 banking 词汇和近邻流程状态，验证了“hard OOS 与细粒度
ontology 边界”是实际瓶颈。

## 10. Memory contract 与 FMAR 解释

C2.5 没有创建动态 private memory。外部 intent 的每次 accepted decision
只计为一次“若存在对应 bucket，则会访问”的 FMAR/wrong-bucket proxy。
没有训练 value memory，也没有执行真实 private retrieval。

所有 variant 保持：

```text
one accepted route -> at most one hypothetical intent bucket
rejected route -> zero hypothetical memory access
cross_intent_candidate_access_count = 0
```

C2.4 task-specific typed memory contract 与 selective router 源文件保持：

```text
src/keyed_gram/stage_c24_contract.py
c8a3b5fd33621349659e504b1bd31502688896cf63207ae59f644849a0d96692

src/keyed_gram/stage_c24b_router.py
4c791aa1a594b7fadd64226ecdd19465aefb54500c3ee92339cf011c01a18175
```

因此 task-specific discrete memory contract 没有回退，但外部实验不能被
表述为已经验证真实 private-memory access control。

## 11. 完整性与安全边界

- 全仓代码冻结前测试：`267 passed`。
- 新增 Stage C2.5 测试：`28 passed`。
- synthetic smoke：通过，明确 `official_test_read=false`。
- 正式 artifacts 中 7 个 JSON 均使用严格 JSON，未出现 NaN/Infinity。
- 5 个 CSV 均非空：
  - `calibration_selection.csv`：2,030 rows；
  - `per_seed_metrics.csv`：100 rows；
  - `aggregate_metrics.csv`：140 rows；
  - `risk_coverage.csv`：2,000 rows；
  - `error_analysis.csv`：5,220 rows。
- 正式 artifacts 不含 dataset、model checkpoint、optimizer state、
  `.pt`、safetensors、private answer 或 key。
- embedding cache 位于 `.runs/`，被 Git 忽略且不进入 artifact manifest。
- calibration 与 test audit 均已验证拒绝重跑。
- test 不参与参数选择。

本阶段没有：

```text
训练 private memory
加载 private answer
执行答案注入
创建或读取 confirmation
执行 key attack
修改 C2.4 typed memory API
```

## 12. 研究问题回答

### 12.1 R2 的 C2.4c 结果能否外部复现？

不能。三个外部协议的 mean known accuracy 全部低于 R0，且只有
CLINC150 的 1/10 partitions 出现 R2 不差于 R0。

### 12.2 R3 是否稳定降低 FMAR？

它数值上稳定降低 FMAR，但方式是拒绝绝大多数 known。两个基准的
known coverage 只有 25.89% 和 17.90%，0/10 seeds 通过完整 gate。

### 12.3 Unknown rejection 是否跨领域成立？

高阈值 R3 的 unknown rejection 数值很高，但不具可用性。更平衡的
MAX_THRESHOLD 对 CLINC official OOS 很有效，对 heldout-intent OOS 和
BANKING77 test-only OOS 明显更难，说明共享领域语义 OOS 仍是瓶颈。

### 12.4 多候选机制是否应保留为主张？

不应。multi-candidate rate 接近零；实证支持的是低 evidence/竞争边界
弃权，不支持“多 relation/intent 同时获支持是歧义检测主机制”。

### 12.5 是否应继续增加 router 复杂度？

不应。R3 没有稳定优于简单 max-score threshold，R2 也没有外部复现。
按照预注册停止条件，应停止当前选择性路由复杂化分支。

### 12.6 是否可以进入探索性 C3？

不可以：

```text
exploratory_c3_allowed = false
c3_eligible = false
```

外部失败不能被 C2.4c 的自建 synthetic 正结果覆盖。

## 13. 最终有限表述

```text
在 CLINC150 与 BANKING77 的 10 个预注册外部 partitions 上，
R2 independent pairwise evidence 没有复现 C2.4c 的 known-routing 优势。

R3 能显著降低 false-memory-access proxy，但代价是 known coverage 和
safe coverage 崩溃；该改善主要来自 empty-set unknown rejection，
不是多候选 ambiguous rejection。

简单 max-score threshold 提供更合理的 coverage/FMAR trade-off，
但没有跨两个基准稳定满足逐 seed 停止条件。

external_oos_validation_status = failed。
external_pairwise_evidence_validated = false。
external_selective_abstention_validated = false。
task_specific_router_ready = false。
exploratory_c3_allowed = false。
c3_eligible = false。

本阶段没有训练 private memory，没有加载 private answer，没有执行答案
注入、confirmation 或密钥攻击，也没有修改 C2.4 discrete memory
contract。
```
