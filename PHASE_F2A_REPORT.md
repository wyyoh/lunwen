# Stage F2A：AuthSynth Effect-Completeness Feasibility

## 结论

```text
autheffectbench_feasibility_status = passed
effect_complete_definition_frozen = true
exact_shield_safety = true
exact_maximal_permissiveness = true
ready_for_stage_f2b = true

f1_frozen_assets_modified = false
f1_locked_test_scored = false
auth_effect_locked_test_scored_once = true
additional_d_series_stages_allowed = false
d_series_status = completed
semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
real_tool_completeness_validated = false
ccf_b_novelty_established = false
private_data_used = false
```

本阶段将上一版 F2A 双流演算保留在独立分支，但把论文主问题改为工具实现、
声明契约与授权策略之间的效果完备性。F1 的 policy oracle、schema 与正式产物
只读绑定；未读取或评分 F1 locked case。

## 1. 精确定义与范围

在有限 ToolEffectIR 输入域内，工具契约 `C_T` 效果完备，当且仅当排空 bounded
异步队列后的全部具体效果均属于 `γ(C_T)`。安全 Shield 由有限安全博弈的最大
不动点精确合成。该结论不外推到任意 Python/JavaScript/MCP 实现；证据不足的
真实工具后续必须返回 `UNKNOWN`。

## 2. Benchmark

```text
case_count = 150
base_tool_count = 8
mutation_category_count = 6
clean_control_count = 6
cross_split_collision_count = 0
ground_truth = finite_transition_system_and_forbidden_state_predicate
llm_judge_used = false
```

150 个 case 由 8 个 synthetic base tools、6 类 mutation、每类每工具 3 个实例
和 6 个 clean controls 构成。base tool、mutation family、workflow、policy、
version lineage 与 hidden-effect combination 均跨 split 隔离。

## 3. 方法结果

| 方法 | CER | FTAR | PODR | CEP | Safe Utility | MPR | Drift Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenyAll | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| Solver-Policy | 0.648 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| ToolGate-Declared | 0.648 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| ToolGuardian-Style | 0.799 | 0.667 | 0.000 | 1.000 | 0.933 | 0.936 | 0.000 |
| AuthSynth-NoCEGAR | 0.799 | 0.444 | 0.000 | 1.000 | 0.920 | 0.923 | 0.000 |
| AuthSynth-CGAR | 1.000 | 0.000 | 1.000 | 0.947 | 1.000 | 1.000 | 1.000 |
| Gold-Contract | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |

`Solver-Policy`、`ToolGate-Declared` 和 `ToolGuardian-Style` 是本仓库内的机制级
模拟，不是作者官方实现。它们用于隔离“给定 policy/contract 正确执行”与
“policy/contract 本身完整”之间的差异。`Gold-Contract` 只是有限模型上界。

## 4. 区分性结果

以下 mutation 中均出现了：给定契约的 monitor 判为可执行、具体 transition
进入 forbidden state、而 AuthSynth-CGAR 经 replay/refinement 后阻断：

```text
composition_omission, delayed_effect, hidden_effect, implementation_drift, parameter_role_omission, state_dependent_effect
```

CGAR 将可重现偏差分类为 contract/parameter-role omission、policy omission、
composition omission 或 implementation drift；抽象伪反例只修正 abstraction，
不生成通用 deny rule。

## 5. 有限结论

本阶段只证明：在冻结的有限 synthetic transition systems 上，可以精确检查
effect completeness，并合成安全且最大许可的 Shield。尚未实现源码分析、真实
sandbox、任意异步系统、MCP、真实 Agent、自适应攻击或任意程序完备性证明。
当前证据只允许进入 F2B CGAR 工程化，尚不足以声称达到 CCF B 创新或投稿质量。
