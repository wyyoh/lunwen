# Stage F2：Intent–Authority Separation Compiler

## 结论

```text
policy_carrying_capability_status = passed
authority_non_amplification_implemented = true
semantic_proposal_authorizes_execution = false
compiler_locked_test_scored = true
compiler_locked_test_scored_once = true
workflow_composition_status = passed
authorization_witness_status = passed
ready_for_stage_f3 = true
formal_model_completed = false
real_agent_integration_completed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

F2 将 executable authority 定义为有限 ``AuthorityAtom`` 集合。每个 atom
同时绑定 subject、tenant、resource、action、relation、purpose 与具有偏序语义
的 constraints，不使用维度集合的隐式笛卡尔积。最终不变量是：

```text
A_exec ⊆ A_proposal ∩ A_trusted_request ∩ A_policy_grant
```

## 输入与信任来源

- proposal 来自不可信 ``SemanticProposal``，不能声明 principal、环境、policy、
  epoch、witness 或 issuer；
- trusted request 来自显式 UI/API 与 ``PolicyRequestContext``；
- policy grant 由 compiler 内部调用 F1 deterministic oracle 重新计算，调用者
  不能直接传入 grant。

proposal 不能扩大 authority。A1 不会静默剪裁后执行：混合安全/越权 proposal
进入 ``NeedsExplicitConfirmation`` 且不签发 capability；oracle deny、空交集、
stale epoch、invalid resource/delegation 或 witness 错误均 fail closed。

## Authorization witness 与 capability

Witness 绑定 policy hash、decision ID、principal、trusted request、proposal、
grant、executable authority、环境、epoch、matched policy IDs 与 reason code。
Verifier 不只验证 Ed25519 签名，还重算 oracle decision、witness 与
non-amplification。显式 deny 不能被同层宽泛 allow 覆盖。

AuthCapV1 使用标准 Ed25519 实现 issuer/verifier 分离。Ed25519 不是本文算法
创新；贡献在 intent–authority separation、compiler、witness 和组合不放大
不变量。private signing key 与正式 capability token 均未写入 artifacts。

## Development / locked 结果

| split | variant | amplification | unauthorized | safe utility | recoverable | unnecessary reject | confirmation | silent narrowing | workflow violation | compile p95 ms | capability mean bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | A0 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.5186 | 1486.3 |
| development | A1 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.5172 | 1486.3 |
| development | B0 | 360 | 360 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.0017 | 0.0 |
| development | B1 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 0.0023 | 0.0 |
| development | B2 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.0467 | 0.0 |
| development | B3 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.0441 | 0.0 |
| locked_test | A0 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.4840 | 1486.3 |
| locked_test | A1 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.5650 | 1486.3 |
| locked_test | B0 | 360 | 360 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.0016 | 0.0 |
| locked_test | B1 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 0.0021 | 0.0 |
| locked_test | B2 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.0445 | 0.0 |
| locked_test | B3 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0.0413 | 0.0 |

两个单步 allow、combined deny 的 workflow 被 A1 在第二步 capability 签发前阻断；
state replay/fork、旧 epoch 和组合 forbidden effect 均未被接受。A0、B0、B2、
B3 不具备组合检查，报告其 workflow violation；B1 通过拒绝一切得到零违规。

## 数据与一次性协议

F1 四 split 字节哈希未变化，未删除或修改 family。prepare 仅解析 train 与
calibration；compiler、配置和 public verifier 冻结后，development 运行一次，
随后 locked_test 唯一评分一次。development/locked 均未参与选择。
``locked_test_scored=true`` 只表示 deterministic compiler evaluation，不是
真实 agent 外部验证。

本阶段没有使用 private data、private answer、真实凭据、memory/tool execution、
远程模型或真实 agent。

## 研究边界

F2 尚未完成形式化证明（F3）或真实 agent 集成/外部评价（F4），因此当前结果尚
不足以单独支撑 ACSAC/ESORICS 投稿质量。Ed25519 只提供标准签名完整性；policy
derivation 的可信性来自显式 oracle 重算与 witness 语义验证。

永久状态保持：

```text
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```
