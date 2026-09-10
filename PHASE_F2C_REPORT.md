# Stage F2C：Symbolic Unary Effect-Contract Synthesis

## 核心状态

```text
symbolic_contract_synthesis_status = failed_development
query_generating_cegis_status = failed_development
bounded_domain_contract_completeness_validated = false
symbolic_unseen_generalization_validated = false
false_verified_complete_count = 0
ready_for_stage_f2d_relational_contracts = false
```

## F2B 的重新解释

F2B 保持字节冻结并继续记为 `passed_blind_finite_query_characterization`。其
`VERIFIED_COMPLETE` 只针对冻结 finite query catalog；refined contract 是
query-specific trace map，query 由 benchmark 枚举，旧 `refinement_precision=1.0`
是指标定义性结果，PODR 是 case-level。F2B 没有结构化 state-diff 语义、没有
inductive loop invariant，也没有完成 drift 后新版本再认证。它没有验证 general
symbolic synthesis、query-generating CEGIS 或 relational completeness。

## F2C 方法

候选契约由受限 DNF guard 与显式 constant/input/state/ITE effect terms 构成，不含
query ID→trace 映射。Analyzer 只看 public schema、declared contract、可信 safety
spec、examples 和窄 verifier/replay 接口；它不能导入 evaluator，也看不到 hidden
AST、gold guard/contract/Shield、mutation label 或完整 assignment table。Verifier
用 SMT 在 candidate 与 hidden ToolSymbolicIR 的差异公式上主动生成新 assignment；
sandbox 只回传该 assignment 的有序事件、结构化 state diff、quiescence、coverage
与版本摘要。合成与验证迭代到有界等价或 fail-closed `UNKNOWN`；本次未据此宣称
整个 development 集上的符号泛化通过。

## Benchmark

```text
case_count = 120
base_tool_count = 12
domain_count = 4
bounded_assignments_per_case = 768 (train/calibration) or 1536 (development)
total_bounded_domain_assignments = 122880
expected_unknown_case_count = 6
cross_split_collision_count = 0
locked_test_scored_once = false
llm_judge_used = false
private_data_used = false
```

Analyzer freeze 后才物化独立 AuthSymbolBench v1；仅报告实际已物化的 split。各 split 按 base tool、guard、
field-binding、version lineage 与 AST shape 隔离；正式 artifacts 不含 hidden locked
AST、gold locked contract/path predicates 或完整 assignment/output table。

## 方法结果

| 方法 | FVCR | UER | Unseen P | Guard P | Guard R | Field | State | COR | FTAR | Safe Utility | MPR | RQR | Conv. | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DenyAll | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.771 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| Declared-Contract Monitor | 0.000 | 0.616 | 0.848 | 0.917 | 0.665 | 0.985 | 0.978 | 0.152 | 0.169 | 0.975 | 0.975 | 1.000 | 0.000 | 1.000 |
| Random Replay Table | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.771 | 0.000 | 0.000 | 0.024 | 0.024 | 0.977 | 0.000 | 0.000 |
| Coverage-Guided Replay Table | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.772 | 0.000 | 0.000 | 0.004 | 0.004 | 0.996 | 0.000 | 0.000 |
| Passive Symbolic Learner | 0.000 | 0.238 | 0.226 | 0.460 | 0.484 | 0.893 | 0.668 | 0.774 | 0.505 | 0.844 | 0.844 | 0.996 | 0.000 | 1.000 |
| CEGIS-No-Minimization | 0.000 | 0.541 | 0.471 | 0.764 | 0.877 | 0.904 | 0.864 | 0.529 | 0.000 | 0.286 | 0.286 | 0.987 | 0.383 | 0.617 |
| AuthSynth-Symbolic | 0.000 | 0.586 | 0.501 | 0.760 | 0.889 | 0.913 | 0.864 | 0.499 | 0.000 | 0.517 | 0.517 | 0.988 | 0.650 | 0.350 |
| Gold Symbolic Contract | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

`DenyAll` 是零效用控制；两种 Replay Table 不做符号泛化；Passive learner 不使用
verifier counterexample；`CEGIS-No-Minimization` 保留反例循环但不优化契约；U0
Gold 只作为上界。所有名称均为仓库内机制定义，不冒充外部系统官方实现。

## 完备性与 UNKNOWN

`CONTRACT_VERIFIED_COMPLETE` 只表示：在对应 case 的公开 schema 所定义的完整
有界域（train/calibration 为 768，development 为 1536 个 assignment）上，Z3 未找到 under/over approximation、field-binding 或
state-update 差异；版本匹配，bounded async 已排空，instrumentation 完整，且 grammar
可表达候选。证据缺口、solver timeout、grammar 不足、unsupported region、未知循环、
无法排空的 delayed effect 都返回 `CONTRACT_UNKNOWN`。一般循环未处理；固定展开两次
不构成证明。旧 certificate 在 version digest 变化后先失效，新版本必须独立分析并
获得不同 certificate ID。

## 诚实边界

F2C 只验证单次执行、单运行性质。它不分析 secret-dependent payload、双运行
noninterference 或 HyperLTL（留给 F2D），不使用真实 MCP、真实 Agent、private data
或 LLM judge，也不证明任意 Python/JavaScript 工具。结构化 safety specification
来自可信控制面；执行 trace 只能发现“发生了什么”，不能自行决定组织禁止什么。
当前结果不能声称真实工具效果完备，也不足以单独支撑 CCF B 投稿；仍需 F2D、F3、F4。

## 协议状态

Analyzer source、solver/grammar/budget 候选先冻结。Calibration 只按原有选择逻辑固定
预注册参数；development gate 未通过则不物化、不运行 locked。通过时复核源码与环境，
再物化和消费唯一 locked 资格。实际 development 次数为 1，
locked 次数为 0。不跨 split 学习或调整参数。
F1/F2A/F2B locked 均未重跑，冻结资产未修改；历史全仓回归的七项兼容例外没有改写为通过。

## Development / Locked 的独立结果


### development

|方法|FVCR|UER|Unseen P|Guard P/R|Field/State|COR|FTAR|Utility/MPR|RQR|Conv./UNKNOWN|
|---|---:|---:|---:|---|---|---:|---:|---|---:|---|
|DenyAll|0.0000|0.0000|1.0000|1.0000/0.0000|1.0000/0.7719|0.0000|0.0000|0.0000/0.0000|1.0000|0.0000/1.0000|
|Declared-Contract Monitor|0.0000|0.5420|0.8343|0.9152/0.5945|0.9823/0.9781|0.1657|0.2030|0.9739/0.9739|1.0000|0.0000/1.0000|
|Random Replay Table|0.0000|0.0000|1.0000|1.0000/0.0000|1.0000/0.7716|0.0000|0.0000|0.0160/0.0160|0.9844|0.0000/0.0000|
|Coverage-Guided Replay Table|0.0000|0.0000|1.0000|1.0000/0.0000|1.0000/0.7717|0.0000|0.0000|0.0027/0.0027|0.9974|0.0000/0.0000|
|Passive Symbolic Learner|0.0000|0.1156|0.1598|0.3814/0.2761|0.8683/0.7407|0.8402|0.7131|0.8777/0.8777|0.9974|0.0000/1.0000|
|CEGIS-No-Minimization|0.0000|0.2887|0.2317|0.6367/0.7935|0.8372/0.7629|0.7683|0.0000|0.0522/0.0522|0.9918|0.0500/0.9500|
|AuthSynth-Symbolic|0.0000|0.3026|0.2353|0.6325/0.8134|0.8396/0.7629|0.7647|0.0000|0.0522/0.0522|0.9918|0.0500/0.9500|
|Gold Symbolic Contract|0.0000|1.0000|1.0000|1.0000/1.0000|1.0000/1.0000|0.0000|0.0000|1.0000/1.0000|1.0000|1.0000/0.0000|

### locked_test

未物化、未运行；development gate 未通过，没有 locked 数值。

### Gate

```json
{
  "development": {
    "contract_overapproximation_ratio": false,
    "convergence_rate": false,
    "cross_split_collision_count": true,
    "drift_detection_recall": true,
    "false_verified_complete": true,
    "field_binding_accuracy": false,
    "forbidden_trace_acceptance": true,
    "guard_precision": false,
    "guard_recall": false,
    "maximal_permissiveness_ratio": false,
    "old_certificate_reuse": true,
    "replay_query_reduction": true,
    "safe_utility": false,
    "state_update_binding_accuracy": false,
    "unknown_rate": false,
    "unseen_effect_precision": false,
    "unseen_effect_recall": false
  },
  "development_status": "failed",
  "locked_test": {},
  "locked_test_status": "not_started"
}
```

Loop 范围：acyclic 与明确静态上界的真实有界循环；unbounded/未知循环为 UNKNOWN。没有 F2D 双运行或真实工具完备性结论。

## 正式负结果与停止决定

Development 仅运行一次，40 cases × 8 方法 = 320 行。主方法仅 2/40 cases 收敛，
38/40 返回 UNKNOWN；false complete 为 0，FTAR 为 0，但 Safe Utility/MPR
仅为 0.052250。安全零违规伴随严重效用损失，不能解释为 F2C 通过。
Locked 未物化、未评分，运行次数为 0；不报告任何 locked 推断结果。

|主方法指标|Development 实测|冻结门槛|结论|
|---|---:|---:|---|
|UER|0.302614|≥0.95|失败|
|Unseen precision|0.235332|≥0.90|失败|
|Guard precision / recall|0.632527 / 0.813369|≥0.95 / ≥0.95|失败|
|Field / state binding|0.839626 / 0.762899|≥0.95 / ≥0.90|失败|
|COR|0.764668|≤0.10|失败|
|Safe Utility / MPR|0.052250 / 0.052250|≥0.90 / ≥0.90|失败|
|Convergence / UNKNOWN|0.05 / 0.95|≥0.85 / ≤0.20|失败|
|FVCR / FTAR|0 / 0|=0 / =0|通过|
|RQR|0.991797|≥0.70|通过|
|Drift recall / old-certificate reuse|1 / 0|=1 / =0|通过|

主方法 development replay 为 504 次，完整域为 61,440 assignments；solver calls 为
548，保存的 iteration digests 共 526 个。未 replay 泛化不足不是通过扩大 replay
预算或重跑消除的，本轮不作算法修正。

UNKNOWN reason codes：`replay_budget_exhausted` 20、`candidate_state_semantics_invalid`
12、`loop_state_semantics_invalid` 4、`instrumentation_coverage_incomplete` 2。
另有 4 个 `old_certificate_invalidated` 记录与其它 reason 重叠，不能相加当作 case 数。
这些是冻结分析器的输出分类，不是新增的根因证明。

## Patch 结果（development）

主方法 patch-type precision = 0.368421，recall = 0.913043；omission-atom recall =
44/68 = 0.647059，exact patch-set rate = 0.10。定位后的 patch-atom precision =
46/186 = 0.247312，recall = 46/70 = 0.657143。未使用定义性为 1 的 refinement precision。
全部方法结果和原始统计行保存在 development_results.csv 与 evaluation_rows.json。

## 统计解释与审计边界

实际物化 120 cases、12 tool families、4 domains、10 类别（含两类 control），
每个 split 40 cases。train/calibration 各 30,720 assignments，development 为
61,440，总计 122,880；有界循环 cases 为 4。冻结 summary 的
`domain_assignment_count_per_case=768` 取首 case，不能解读为所有 split 均同域大小；
本段按已保存逐行 domain_assignment_count 澄清，未修改原 summary 或指标。
主方法三个 split 的对比 replay 共 1,428 次；全部方法对比共 6,900 次。
Calibration 候选选择另有 1,676 次 replay，因此整次入口共 8,576 次，不能把重复
候选评价当作新的 benchmark cases。有界循环 concrete/SMT 差分通过不等于学习器
已经能够合成相应契约，本次仍记录了 4 个 loop_state_semantics_invalid。

本阶段是 unary allowed-assignment 比较；冻结定义中 Safe Utility 与 MPR 数值相同，
Gold gap 可只读推导为 1−MPR，主方法 development 为 0.947750。不能据此声称
有状态或真实工具的全局最大许可性。零分母指标使用冻结实现的默认值，例如 DenyAll
的 precision/field accuracy 为 1 不代表有预测或有用执行。无完整性声明的方法的 FVCR=0
也不是完整性证据。冻结行没有独立的 SHIELD_SAFE 声明字段，故未伪造独立
`false_safe_shield_count`；安全结论采用实际 forbidden_accepted_count/FTAR。

训练和 calibration 主方法各 38/40 收敛，而 development 为 2/40。这是独立结构族上的
正式负结果；不从单次结果声称其唯一根因，亦不据结果选择另一主方法。

## 冻结与验证

用户授权修复正式调度后，实际 Analyzer/调度冻结提交为
`65d64537ee9e5a0260277e978084d34608f5ba5b`，元数据执行提交为
`9d60b500dfef90d56e1bd44b89af6187cade595d`。原模板、seed anchor、grammar、指标、
solver 参数候选、安全门槛与容器均未变；calibration 选择 extended-24（24 replay、
5000 ms timeout、DNF 4/5/16），在 development 前固定。

预冻结普通测试实际为 162 targeted passed；全仓 732 passed、7 项精确匹配的历史
兼容失败、0 unexpected failures。未把历史例外改写为通过，未修改历史代码或产物。
正式结果已做 strict JSON/CSV、SHA、精确 inventory、信息边界和敏感扫描；对保存行
只读重算 aggregate/gate，不调用 benchmark generator、replay 或正式方法评价。
development=1、locked=0，重复入口的只读 preflight 被已有输出拒绝。
