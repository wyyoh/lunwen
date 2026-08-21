# AuthSynth F2B：Blind CGAR 研究计划

## 研究定位

F2A 只证明：当完整具体语义与完整安全谓词均可见时，有限模型上的效果完备
契约与最大许可 Shield 可构造。F2B 不把该 oracle upper bound 当作算法结果，而
验证一个分析器在看不到实现、隐藏效果、mutation 标签和 gold Shield 时，能否仅
通过有限证据与窄 sandbox replay 迭代修正 contract、policy guard 与 Shield。

```text
f2a_oracle_feasibility_passed = true
f2a_exact_finite_shield_constructible = true
f2a_blind_contract_inference_validated = false
f2a_real_cegar_loop_implemented = false
f2a_algorithmic_counterexample_precision_validated = false
f2a_external_baseline_superiority_established = false
ready_for_stage_f2b = true
```

F2A 中名为 `AuthSynth-CGAR` 的 full-semantics 分支，在 F2B 叙事中固定重命名为
`Full-Semantics AuthSynth Upper Bound`。真正的 `AuthSynth-CGAR` 只指 F2B 盲
分析器。

## 三世界边界

1. Evaluator/Oracle World 持有 concrete transitions、隐藏效果、最终状态 oracle、
   mutation/omission 标签和 gold Shield。
2. Analyzer World 只持有声明契约、schema、静态 hypothesis、历史 observation、
   高层可信安全规范、有限 query catalog 与 replay client。
3. Runtime Shield World 只持有 refined abstraction、policy guard、version
   certificate 与当前抽象状态。

Analyzer package 不得 import evaluator/replay 实现；replay 返回值不包含未执行
路径、transition source、mutation 类型、gold contract 或 gold policy。

## 算法

盲 CGAR 从声明 contract 与带 provenance 的静态 hypotheses 出发。每轮先在当前
抽象上生成 candidate，然后调用窄 replay API。可重现偏差产生 call-trace、policy、
composition 或 version patch；不可重现 candidate 细化抽象 guard。每轮都重新合成
Shield 并记录 model/Shield delta。欠近似 contract 不可能凭空发现模型外效果，
因此算法显式加入 coverage-guided replay 与 UNKNOWN edge；证据不足时 fail closed，
绝不返回虚假 `VERIFIED_COMPLETE`。

## 继续门槛

development 与全新 locked_test 同时要求：

```text
false_verified_complete_count = 0
forbidden_trace_acceptance_rate = 0
contract_effect_recall >= 0.90
policy_omission_discovery_recall >= 0.80
counterexample_precision >= 0.75
safe_utility >= 0.85
maximal_permissiveness_ratio >= 0.90
drift_detection_recall = 1.0
unknown_rate <= 0.25
```

F2B 不接真实 MCP/Agent，不使用 LLM judge，不使用 private data，也不宣称对任意
Python/JavaScript 工具证明效果完备。
