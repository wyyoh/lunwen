# F2A 重新定性（不回写冻结产物）

F2A 的正式产物、报告和唯一 locked audit 保持字节不变。F2B 只增加如下解释性
状态：

```text
f2a_oracle_feasibility_passed = true
f2a_exact_finite_shield_constructible = true
f2a_blind_contract_inference_validated = false
f2a_real_cegar_loop_implemented = false
f2a_algorithmic_counterexample_precision_validated = false
f2a_external_baseline_superiority_established = false
ready_for_stage_f2b = true
```

旧 `AuthSynth-CGAR` 实际使用 full concrete semantics 与完整 safety rules，故仅作为
`Full-Semantics AuthSynth Upper Bound`。旧 `CEP=0.947` 是 instrumentation-derived
feasibility statistic，不是 blind analyzer precision。
