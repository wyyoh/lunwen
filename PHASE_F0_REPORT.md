# Stage F0：AuthCap Protocol and Benchmark Preregistration

## 结论

F0 已冻结 AuthCap 的研究问题、六项安全性质、严格类型边界、policy resolution、
benchmark schema、split 隔离单位、一次性构建协议与 F1 readiness gate。

F0 不包含 learned model、最终 compiler、形式证明、真实 agent、capability 签发、
memory/tool 执行或 private data。

## 冻结基线核验

开始 F0 前实际核验了：

- E0 evidence registry 中 22 个文件；
- C2.5、D1、D2、D2.1、D2.2、D2.3 的正式报告、summary；
- 六阶段 source/artifact manifest 及其正式 artifact inventory。

所有现存哈希均匹配。F1 分支从 E0 `agent/research-synthesis` 提交
`b0805e19c7ad5dac57d225f9cffa524afd5bc1ab` 创建，不从论文草稿分支继续。

## Policy 语义

F1 只接受严格 JSON/YAML，不解析自然语言 policy。默认 deny。先取最高
priority，再取最高 specificity；winning tier 内显式 deny 优先。不存在隐式
allow。所有 authority 均使用显式有限集合，不允许 wildcard。

## 数据与审核边界

```text
benchmark_text_generation = AI_assisted
policy_ground_truth = deterministic_oracle
independent_human_validation = false
private_data_used = false
```

F1 locked test 只允许生成、封存、hash 和进行 schema/collision/oracle
一致性审计；不允许 learned baseline、模型选择或效果评分读取它。

## 后续状态

F0 本身不解锁 F2。只有正式 F1 的全部质量与覆盖门槛通过，才允许
`ready_for_stage_f2=true`。无论 F1 结果如何，`c3_eligible=false`。
