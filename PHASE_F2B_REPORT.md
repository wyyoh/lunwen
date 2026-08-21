# Stage F2B：Blind CGAR 与效果完备性审计

## 核心状态

```text
blind_cgar_status = passed
false_verified_complete_count = 0
f2b_locked_test_scored = true
f2b_locked_test_scored_once = true
ready_for_stage_f2c = true

f1_locked_test_scored = false
f2a_locked_test_rerun = false
real_tool_completeness_validated = false
formal_model_completed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

## F2A 的正确定位

F2A 保持字节冻结，其 full-semantics 分支只证明 oracle feasibility 与有限 safety
game harness。旧 `AuthSynth-CGAR` 在本文中重命名为 `Full-Semantics AuthSynth
Upper Bound`；旧 CEP 不是 blind analyzer precision。F2B 才首次实现 analyzer
看不到 concrete implementation、gold contract/policy、mutation/omission 标签的
真实迭代 refinement。

## 信息边界与算法

Analyzer 只接收 declared per-call contract、静态 hypothesis、可信高层安全规范、
finite query catalog 与窄 replay API。Replay 只返回当前 query 的有序 event、state
diff digest、quiescence、coverage 和 version digest。每轮真实执行：candidate →
replay → real/spurious classification → contract/policy/composition/version patch →
Shield resynthesis，并保存 model 与 Shield delta digest。证据不足时返回 `UNKNOWN`。

## Benchmark

```text
case_count = 270
compatibility_shape_case_count = 150
blind_compound_case_count = 120
expected_evidence_gap_case_count = 24
cross_split_collision_count = 0
ground_truth = hidden finite transition replay + final-state oracle
llm_judge_used = false
private_data_used = false
```

新 locked namespace 在 analyzer freeze 后生成；F2A locked 未复用或重跑。公开 split
文件只含 analyzer input，hidden concrete replay 与 evaluator label 未持久化到其中。

## 结果

| 方法 | CER | FTAR | PODR | CEP | Safe Utility | MPR | Unknown | False Complete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DenyAll | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0 |
| Declared-Contract Monitor | 0.801 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| Multi-Evidence Characterizer | 0.858 | 1.000 | 0.000 | 1.000 | 0.993 | 0.996 | 0.000 | 0 |
| AuthSynth-NoCEGAR | 0.858 | 0.318 | 0.000 | 1.000 | 0.800 | 0.837 | 0.000 | 0 |
| AuthSynth-CGAR | 1.000 | 0.000 | 0.970 | 0.868 | 0.956 | 0.957 | 0.089 | 0 |
| Full-Semantics AuthSynth Upper Bound | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |

`Declared-Contract Monitor` 与 `Multi-Evidence Characterizer` 是机制抽象，不是外部
论文系统的官方实现；结果不能表述为对其正式实现的 superiority。Full-Semantics
方法只是 gold upper bound。

## 完备性定义

F2B 分别核验逐调用效果绑定、保留顺序/phase/parent 因果链的 trace completeness，
以及跨工具累计 bad-state 的 composition completeness。全局无序 effect union 不再
作为完备性定义。

## 有限结论

本阶段证明盲分析器在冻结 finite synthetic domain 上能够经 replay 找到真实遗漏、
排除伪反例、检测漂移并逼近 Gold Shield，同时保持 false-complete 为 0。尚未接入
真实 MCP、任意 Python/JavaScript 工具、真实 Agent 或外部官方 baseline，也未完成
形式化证明；因此只能进入 F2C 扩展 benchmark，不能声称真实工具完备或 CCF B
投稿质量已经成立。
