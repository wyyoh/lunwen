# Stage C2.4b v2.1 报告：单模型 AI 探索性审核

## 协议定位

本报告记录 `c24b-public-selective-router-v2.1`。它是对原 v2 数据构造的
预评分修订，不是新的 C3 阶段，也不是正式 locked audit。

用户决定不采用真人审核，因此本协议明确使用：

```text
review_mode = single_ai_semantic_audit
public_benchmark_human_reviewed = false
public_benchmark_ai_reviewed = true
independent_external_validation = false
formal_calibration_allowed = false
formal_locked_audit_executed = false
```

同一个项目内的单模型 AI 既参与数据修订又查看了 locked 文本，所以审核只能是
`exploratory_non_independent`。family-level 决定映射到生成行，不表述成逐行独立审核，
也不表述成两名 reviewer 或外部验证。

## 原 v2 的预评分结论

原 v2 在任何 router/model 评分前完成了单模型语义数据质量检查。134 个唯一
family 映射到 512 行，130 个标签一致，4 个 family 建议改标，影响 16 行：

| Family | 原类型 | AI 建议 |
|---|---|---|
| `locked_v2_ambiguous_dossier_handle` | ambiguous | `registry_id / known` |
| `locked_v2_ambiguous_authorization_value` | ambiguous | `access_code / known` |
| `locked_ambiguous_record_tag` | ambiguous | `registry_id / known` |
| `locked_ambiguous_subject_locator` | ambiguous | `registry_id / known` |

此外，原 v2 的 16/16 ambiguous family 含显式不确定提示词，unrelated family
与三个目标 relation 的词汇距离过远。因而原 v2 已在模型评分前永久降级为
`superseded_before_model_scoring_diagnostic_only`。这不是 locked 结果泄露事故，
没有创建新的 `protocol_incident.json`；旧数据、空白真人字段和历史 manifest 均保持原字节。

旧 v2 的 prepare、human-review validation、formal calibration 与 formal locked audit
入口均已撤销；仅修改可控的 `benchmark.version` 不能绕过撤销门禁。

## v2.1 数据协议

v2.1 使用全新且互相隔离的命名空间：

```text
public_train_v2_1
public_calibration_v2_1
public_locked_audit_v2_1
```

| Split | 唯一 family | 行数 | known | ambiguous | unrelated |
|---|---:|---:|---:|---:|---:|
| `public_train_v2_1` | 24 | 72 | 72 | 0 | 0 |
| `public_calibration_v2_1` | 40 | 160 | 96 | 32 | 32 |
| `public_locked_audit_v2_1` | 40 | 160 | 96 | 32 | 32 |

历史碰撞审计同时覆盖 C2.3、C2.4 和已封存的 C2.4b v2，共 332 个历史
phrase、39 个 entity 和 30 个 frame。跨 split 以及对历史的 family、entity、frame、
exact/normalized text、containment、token signature、lemma bigram 碰撞均为 0。

构造审计结果：

- 16/16 ambiguous 不再含 `unknown/free/absent/unspecified/missing` 等显式捷径；
- 16/16 unrelated 含目标 carrier hard negative；
- 16/16 ambiguous 均可给出至少两种合理目标 relation 读法；
- 6 个刻意构造的 calibration known minimal-pair family 标为 synthetic boundary；
- 15 个 unrelated 仍存在明显“目标 carrier + 非目标尾部属性”结构捷径；
- 全部 16 个 ambiguous 都标为 `synthetic_compound_scope`，不把合成构造伪装成自然分布。

因此审核状态使用 `completed_with_declared_boundaries_exploratory_non_independent`，
同时把语义一致性结果有限地记作 `passed_exploratory_non_independent`。

## AI 审核与封存

当前 AI 审核覆盖 104 个唯一 family、等价映射 392 行；没有自动填写任何
`reviewer_1`、`reviewer_2` 或 `adjudicated` 字段。审核置信度全部保留为 `null`，
没有伪造校准数值。审核模型记录为 `OpenAI Codex (GPT-5 family)`，运行环境不能
提供更精确 revision，因此 `ai_review_provenance_complete=false`。

审核导入前会绑定并复验：配置、relation definitions、family task、三份数据、历史
source、完整 prepare snapshot、实现源文件和审核 prompt 的 SHA-256。104 个 family
会先在内存中全部校验，再写不可变 attempt seal 和审核 CSV；失败不会留下可换 spec
覆盖的部分 split 结果。最终 validator 要求严格 JSON、无重复键、无 NaN/Infinity、
安全相对路径和精确 artifact inventory。

## Router 与指标状态

v2.1 本轮只修订数据协议并完成 AI 探索性语义审核，没有执行 R0–R4 calibration、
development 或 locked 模型评分。因此不能从本轮选择 router，也没有可报告的正式
threshold、conformal quantile、risk–coverage 曲线或性能数值。

| 项目 | v2.1 当前结果 |
|---|---|
| R0–R4 正式/探索性模型运行 | 未执行 |
| selected router | 未选择 |
| threshold / conformal 参数 | 未选择 |
| known coverage | 未评估 |
| accepted route accuracy | 未评估 |
| worst-family accuracy | 未评估 |
| ambiguous / unrelated rejection | 未评估 |
| false memory access rate | 未评估 |
| wrong bucket access rate | 未评估 |
| safe coverage | 未评估 |
| accepted fact Top-1 / MRR | 未评估 |

原 C2.4b synthetic smoke 的 R0–R4 数值仍只属于控制流测试，不能迁移为 v2.1
研究结果，也没有参与 v2.1 数据或审核选择。

## Memory contract 与 readiness

v2.1 没有修改 `AcceptedRoute`、`RejectedRoute`、typed memory API、entity branch、
relation bucket 或 bucket 内检索。`stage_c24_contract.py` 与
`stage_c24b_router.py` 的字节 SHA-256 保持为上一 C2.4b 实现的冻结值。此次纯数据/
审核流程没有加载 entity checkpoint，因此不能伪称完成 checkpoint before/after 验证。

```text
closed_set_selective_router_ready = false
open_set_abstention_ready = false
discrete_memory_contract_preserved = true
public_benchmark_human_reviewed = false
new_confirmation_pool_created_after_freeze = false
ready_to_create_new_confirmation_pool = false
c3_eligible = false
```

即使 AI 语义审核完成，它也不能满足原公式中的真人审核项；本阶段又禁止创建
confirmation，所以不能进入 C3。下一步只能继续明确标记为探索性的 router 研究，
或另建真正独立的验证协议；当前不能冻结协议并创建 confirmation。

## 禁止事项与有限表述

本轮没有训练 private memory，没有加载 private answer，没有执行答案注入、
LM residual injection、key-value permutation、密钥攻击、membership inference，
也没有创建、读取或运行 confirmation。AI 审核分数和置信度不进入 memory API。

```text
选择性离散路由闭集门槛未通过（v2.1 模型指标未评估）。
开放集弃权门槛未通过（v2.1 模型指标未评估）。
离散 memory contract 保持。
ready_to_create_new_confirmation_pool = false。
c3_eligible = false。
本阶段没有训练 private memory，没有加载 private answer，没有执行答案注入或密钥攻击。
```

## 工程运行记录

代码/协议冻结提交为：

```text
b7d8df89de46856cc649b2909186b10f9bee5ca5
```

该提交已先推送到 `origin/agent/stage-c21`，随后才在 clean worktree 上执行一次
`stage-c24b-v21-prepare`。prepare manifest 记录 `dirty=false`。AI apply 阶段的最终
artifact manifest 如实记录 `dirty=true`，原因只是 prepare 新生成且尚未提交的
`artifacts/stage_c24b_v21/`；实现源文件由独立 source snapshot 复验为冻结提交字节，
没有在 prepare 与 apply 之间修改代码、配置或审核 spec。

运行顺序及结果：

1. `stage-c24b-v21-prepare`：成功，72/160/160 行；
2. `stage-c24b-v21-apply-ai-review`：成功，104 family、392 行覆盖；
3. `stage-c24b-v21-validate-ai-review`：成功；
4. formal calibration：未执行且保持禁止；
5. development：未执行；
6. formal/exploratory locked 模型评分：均未执行；
7. confirmation：未创建、未读取、未运行。

审核 boundary 分布如下：

| Split | family | passed | boundary | 覆盖行 |
|---|---:|---:|---:|---:|
| train | 24 | 24 | 0 | 72 |
| calibration | 40 | 19 | 21 | 160 |
| locked | 40 | 24 | 16 | 160 |
| **合计** | **104** | **67** | **37** | **392** |

声明的限制 flag 为：`synthetic_compound_scope=16`、
`synthetic_carrier_minimal_pair=6`、`non_target_tail_property=15`、
`domain_transfer_identifier=1`。这些 flag 是数据构造限制，不是 router 预测。

关键 SHA-256：

| 对象 | SHA-256 |
|---|---|
| v2.1 config | `10e1a3ed9d7b0914d988ede695a2c9df81cba36eab783565bdc4e45531a5f62f` |
| relation definitions | `2215ad193df45dab928a06046bab79bcbfca456e021f1d083a3d04a025404016` |
| public benchmark manifest | `2c5815e43730a4900f1944fa7a59d28b21aef55ffed930df8722d22fcac4c2c2` |
| prepare artifact manifest | `6c0f5c6863ea244c2321c24b0b5b9bfd509d6183723a38dae42ab44b4ad2cf27` |
| AI audit spec | `5140cd602e47ce4a723f68ac1dd64b245bc9e67d8dcfe4748e52b9af6c0f67ca` |
| AI review prompt | `39e796137f8adddcc2878514cb0069fadbd69cdf1c10e12e6966bc11678ecacc` |
| AI review attempt manifest | `b765784e0529b7ba9e2fa2dc8cb0dbb9b05801221762b840add7fe35f2c6b818` |
| AI review manifest | `f7ce933cab973fcdc99deca57011317e2ba36517ec7934c8e07a0a85d784b5af` |
| final artifact manifest | `f0d7a6e9baf1b2648050fa4dee28063d100b2250c6bf40b0d9e46d1151eac920` |

三份 v2.1 JSONL SHA-256 分别为：train
`9165b6d69f88207638022796fe9aa0a39ab9207ab19b31011a5be49ddfd1b5f4`、
calibration `88bc6475de9ad83c1ab8ae9c3383a2d9e706717c3fd988355e512f18395effaf`、
locked `42e2ca09238d0808480cadaec04b8996f547e1e0f8f8597ed7cf1a8db86936fe`。

独立复算结果：12 个 JSON 均可严格解析且不含 NaN/Infinity；final manifest
列出的 17 个先行文件 size/SHA-256 全部匹配，包含 manifest 自身时 artifact 目录
精确为 18 个文件；3 份数据 SHA-256 全部匹配。原 v2 benchmark manifest 与 4 份
review CSV 的 SHA-256 仍分别为 `7977b7a3…ed977`、`a8c8d687…dbcb`、
`f8016bdf…adc`、`16ed91c8…3ece`、`a74187c6…f1f4`，没有填写旧真人字段。

全仓测试结果为：

```text
215 passed in 59.14s
```

其中 v2.1 专项为 18 项，覆盖 namespace、历史碰撞、AI schema、稳定撤销门禁、
完整 prepare/final seal、source/config/prompt/data 篡改、无部分写、非法/NaN/bool
confidence、confirmation/private/router 字段拒绝和恒 false readiness。

代码冻结提交已确认位于开放的 Draft PR #2（`agent/stage-c21`）。本报告和
`artifacts/stage_c24b_v21/` 随结果提交推送到同一 Draft PR。正式 calibration 与
formal locked audit 不会因单模型 AI 审核而解锁。
