# 从 AuthZRouteBench 到 AuthFlowBench

## F1 保持冻结

F2A 不读取 F1 locked case 内容，不运行 baseline，不改变 split 或 policy。F1 继续
提供：

- deterministic policy oracle；
- principal/resource/action/purpose schema；
- explicit deny 与 policy epoch；
- multi-step combined deny seed；
- 四 split 的冻结 hash。

F1 是 query-level policy benchmark，不足以验证状态化 authority flow。

## AuthFlowBench 单元

F4 新建独立 benchmark，每个 case 至少包含：

```text
case_id
initial_store
trusted_goal
trusted_grants
data_sources_with_labels
influence_origins
authority_origins
agent_graph
memory_state
candidate_program_or_trace
allowed_supply_trace
allowed_effect_trace
forbidden_effect_combinations
final_state_oracle
```

ground truth 来自 F1 policy oracle、显式 effect oracle 和 before/after state，
不由 LLM judge 决定。

## 攻击映射

| AuthFlow 类别 | F1 可复用因素 | 新增状态维度 | 主要性质 |
|---|---|---|---|
| authority laundering | purpose/deny | influence provenance、summary | T1/T3 |
| capability recombination | multi-step deny | branch effect history | T4 |
| delegation fork | delegation depth | linear child ownership | T2 |
| branch merge escalation | workflow graph | parent digest、lineage | T4 |
| memory-fragmented authorization | memory domain | 多 item influence join | T1/T3 |
| trusted-tool echo | tool domain | input/output provenance | T3 |
| stale authority | policy epoch | live authority resource | T1/T2 |
| cross-session reuse | subject/session | lineage/session binding | T1 |
| budget overspend | constraint | consumption ledger | T2 |
| origin substitution | policy hash/decision | AuthorityOrigin | T1/T3 |

## Baseline 可观测能力

| 变体 | Data label | Effect capability | Linear budget | Origin | Merge rule |
|---|---:|---:|---:|---:|---:|
| unrestricted | 0 | 0 | 0 | 0 | 0 |
| RBAC/ABAC | 0 | 1 | 0 | policy | 0 |
| scoped capability | 0 | 1 | optional single-use | issuer | 0 |
| FIDES-style IFC | 1 | policy sink | 0 | data | label join |
| tracked capability | optional wrapper | 1 | 0 | lexical capture | 0 |
| APPA-style branch | 1 | policy | 0 | data/event | label/event merge |
| data IFC + ordinary cap | 1 | 1 | optional | separate | set/history baseline |
| DAIFC | 1 | 1 | 1 | dual origin | lineage+budget+effect |

表格只用于实验接口，不替代对真实实现的忠实复现。尤其 APPA、FIDES、TACIT 和
FLAC baseline 必须按公开代码/论文语义实现，不能人为削弱。

## 指标

安全：

```text
authority_fabrication_rate
authority_laundering_rate
capability_recombination_success_rate
delegation_fork_acceptance_rate
merge_escalation_rate
unauthorized_commit_rate
authority_budget_overspend_rate
origin_substitution_rate
```

效用：

```text
safe_task_completion
benign_task_completion
necessary_authority_recall
privilege_overhead_ratio
unnecessary_rejection_rate
```

主指标：

\[
SafeTaskCompletion =
\frac{合法完成且无 authority 违规的任务}{全部合法任务}
\]

## Split

AuthFlowBench 必须按以下单位隔离：

- attack program template；
- authority lineage topology；
- workflow graph；
- resource alias family；
- grant/policy template；
- memory fragmentation template；
- agent graph；
- effect-combination oracle。

F1 split 不能直接成为 AuthFlow split；新 benchmark 需要新的预注册和 locked test。

## F4 外部集成

至少两个不同 runtime：

1. AgentDojo：真实工具任务、安全/效用联合评价；
2. LangGraph 或 MCP runtime：多 Agent、长期 memory、branch/delegation。

LLM action 先编译为受限 IR，再由 type/effect checker 验证。模型不能直接调用真实
工具或构造内部 authority resource。
