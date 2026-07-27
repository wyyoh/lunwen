# Stage F1：AuthZRouteBench Construction and Policy Oracle

## 结论

```text
authzroutebench_status = passed_preregistered_construction
policy_oracle_status = passed
policy_ground_truth_is_deterministic = true
semantic_proposal_authorizes_execution = false
locked_test_generated = true
locked_test_scored = false
ready_for_stage_f2 = true
```

F1 只构造 policy-grounded benchmark 并审计 oracle。没有训练或评价最终
AuthCap compiler，没有运行 learned baseline，也没有对 locked test 做模型评分。

## 1. Ground truth 与文本来源

全部 allow/deny、maximum authority、policy hash、decision ID 和 witness 均由
纯确定性的结构化 policy oracle 生成。同一输入重复计算得到逐字节相同的
canonical output。

```text
benchmark_text_generation = AI_assisted
policy_ground_truth = deterministic_oracle
independent_human_validation = false
private_data_used = false
```

AI 只参与自然语言表面模板设计，不参与标签、authority 或 adjudication。本阶段
没有独立人工审核，也不伪称有人审。

## 2. Benchmark 规模

| 指标 | 数量 |
|---|---:|
| cases | 1440 |
| families | 144 |
| domains | 3 |
| attack categories | 12 |
| policies | 2208 |
| principals | 1440 |
| tenants | 1440 |
| resources | 2880 |
| multi-step workflows | 120 |
| allow / deny | 432 / 1008 |
| restricted authority | 432 |
| explicit deny | 768 |
| cross-tenant | 120 |
| stale-policy | 120 |

四个 split 各 360 cases、36 families；attack family、policy template、alias
family、tenant、workflow template 和 purpose template 跨 split 隔离。

## 3. Policy language

Policy 使用严格 JSON/YAML schema，表达显式 subject/tenant/resource/type/
action/relation/purpose、principal attributes、有效期、delegation depth、
priority、specificity、deny、policy epoch 和有限 authority constraints。
默认 deny；先比较 priority，再比较 specificity；winning tier 内显式 deny
优先。不存在隐式 allow，unknown field、duplicate ID/key、wildcard 和
NaN/Infinity 均被拒绝。

它不表达自然语言 policy、分布式一致性、概率条件、任意代码条件、真实 IAM/KMS
或 executable capability。

## 4. Authority 与多步标注

Authority subset 覆盖 subject、tenant、resource、action、relation、purpose
以及约束保留。候选删除约束或增加任一集合元素都会被判为放大。

`multi_step_privilege_amplification` case 保存两个独立 allow step 与一个
combined deny；所有 step 和 combined decision 都由同一 oracle 复算。F1
实现的是 subset/union invariant checker，不是最终 compiler 的
authority-non-amplification 实现或形式证明。

## 5. 数据质量

```text
cross_split_collision_count = 0
oracle_determinism_failure_count = 0
oracle_schema_failure_count = 0
authority_subset_failure_count = 0
policy_hash_mismatch_count = 0
```

locked test 只被生成器、schema/oracle 一致性和跨 split collision audit 读取，
未被任何 baseline、模型选择或效果评价逻辑读取，`locked_test_scored=false`。

## 6. 尚未完成

```text
authority_non_amplification_implemented = false
formal_model_completed = false
real_agent_integration_completed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

当前结果尚不足以投稿 ACSAC/ESORICS；仍需要 F2 compiler、F3 formal model/
trace refinement 与 F4 real-agent external evaluation。readiness 只表示允许
开始 F2，不代表 CCF B 创新性、部署安全或 private-data readiness。
