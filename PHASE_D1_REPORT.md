# Stage D1：Trusted Capability Contract

## 结论

Stage D1 已在冻结提交
`d92b699d44bc5ba7f88c16eaa4d854ea5255e572`
上完成一次正式、确定性的 capability attack matrix 审计。

25 个预注册场景全部得到预期结果：

| 指标 | 结果 |
|---|---:|
| 正向场景 | 5 / 5 |
| 负向场景 | 20 / 20 |
| 未授权 memory access | 0 |
| cross-relation access | 0 |
| cross-entity access | 0 |
| rejected-request memory access | 0 |

因此，本阶段的有限结论是：

```text
trusted_capability_contract_status = passed
semantic_router_in_tcb = false
suggested_relation_authorizes_memory = false
ready_for_stage_d2_public_synthetic_values = true
private_value_memory_ready = false
task_specific_router_ready = false
original_c3_allowed = false
c3_eligible = false
```

`passed` 只表示本仓库中的类型化 capability contract 通过了预注册的
单进程确定性测试，不表示部署级安全认证。

## 研究路线重置

C2.5 的外部 OOS 审计已触发永久停止条件。仓库在 D1 开始前固定了：

```text
selective_router_research_status = stopped_external_validation_failed
pairwise_evidence_branch_status = stopped
set_valued_ambiguity_branch_status = stopped
additional_router_complexity_allowed = false
```

D1 没有实现 R4/R5、cross-encoder、新 threshold、conformal 变体、新 OOS
数据或 MASSIVE 扩展，也没有降低 C2.5 continuation gate。停止状态由
`configs/research_status.yaml` 及其 SHA-256 引用约束。

本阶段采用的架构原则是：

> 语义解析可以辅助形成访问建议，但授权必须由学习型路由器之外的类型化、
> 可认证 capability contract 执行。

```text
自然语言问题
    ↓
不可信语义解析器
    ↓
SuggestedRelation（仅交互建议）
    ✕ 不能访问 memory

外部身份来源 + 显式资源请求
    ↓
可信 policy / capability authority
    ↓
AuthorizedMemoryRequest
    ↓
HMAC 验证 + subject/scope/relation/permission/time/policy/replay 检查
    ↓
VerifiedCapability
    ↓
单一 typed memory bucket probe
```

## 信任边界

冻结配置明确将以下组件放在 TCB 之外：

- natural-language router；
- relation proposal。

以下组件位于当前原型的 TCB 内：

- subject identity provider；
- capability policy；
- capability authority；
- HMAC key ring；
- capability verifier；
- trusted capability gateway；
- typed keyed memory。

其中 subject identity provider 只是架构依赖，本阶段没有实现外部 IAM 或
会话认证。Python 原型也没有实现 router 与 gateway/memory 的进程隔离。

## 类型化请求与 memory contract

唯一允许进入 trusted gateway 的外部请求为：

```python
@dataclass(frozen=True)
class AuthorizedMemoryRequest:
    subject_id: str
    entity_id: str
    relation_id: RelationId
    capability_token: bytes
```

请求不包含：

```text
raw text
router logits
confidence / probabilities
semantic embedding
自由文本 relation
private value
private answer
```

`SuggestedRelation` 是单独的不可信类型。把它直接提交给 gateway 会得到
`router_proposal_not_authorized`，并且 memory access delta 为 0。

gateway 先完成全部验证，才向 memory 传递内部 `VerifiedCapability`。
memory probe 不保存 value，只记录精确的 `(entity_id, RelationId, nonce)`
typed access。Stage C2.4 的 `RelationId` 文件保持原 SHA-256：

```text
c8a3b5fd33621349659e504b1bd31502688896cf63207ae59f644849a0d96692
```

## Authenticated capability

token 使用标准 HMAC-SHA-256：

- 构造依据 RFC 2104；
- key 由 Python `secrets.token_bytes` 生成；
- 最短 key 长度为 32 bytes；
- 签名由标准库 `hmac.digest(..., "sha256")` 计算；
- 验证使用 `hmac.compare_digest`；
- header 固定 `alg=HS256`、`typ=KG-CAP`、`v=1`；
- 严格拒绝 duplicate JSON key、NaN/Infinity、未知字段和未知
  `RelationId`；
- nonce 由 `secrets.token_hex` 生成；
- token 限长 4096 bytes；
- D1 capability 为 single-use。

capability payload 绑定：

```text
subject_id
explicit entity_scope
RelationId
permissions
issued_at
expires_at
key_id
nonce
policy_version
schema_version
```

scope 只允许显式、排序、唯一的 entity ID；wildcard 被拒绝。token payload
经过认证但不加密，因此不得把 token 当作保密容器。

本项目没有自创密码算法，也不声称 HMAC 设计是新的密码学贡献。

参考：

- [RFC 2104: HMAC](https://www.rfc-editor.org/rfc/rfc2104)
- [Python `hmac` 文档](https://docs.python.org/3/library/hmac.html)
- [Python `secrets` 文档](https://docs.python.org/3/library/secrets.html)

## 冻结审计协议

正式审计在 tracked worktree 干净且输出目录不存在时才允许启动。运行状态
被写入 `.runs/stage_d1/protocol_state.json`；同一阶段再次启动会被拒绝。

正式审计使用固定时钟，仅测试 contract 行为，不读取数据集、模型
checkpoint、private answer、confirmation 或 key attack 输入。正式结果记录
的源提交为：

```text
d92b699d44bc5ba7f88c16eaa4d854ea5255e572
```

源 manifest 同时绑定 `.gitattributes`、停止记录、D1 配置、CLI、冻结的 C2.4
contract 和全部 D1 runtime source。manifest payload SHA-256 为：

```text
9a8d1324e08234a463176c0da729772b3b0728a841ad9c7e13622eb57cee8431
```

## 正向场景

通过的 5 个正向场景为：

1. valid registry capability；
2. valid city capability；
3. valid access capability；
4. rotated-key capability；
5. replay token 的第一次使用。

每个正向场景只产生 1 次与 capability 完全一致的 typed memory access。

## 负向场景

通过的 20 个 fail-closed 场景为：

| 类别 | 场景 | memory access |
|---|---|---:|
| subject | wrong subject | 0 |
| scope | wrong relation、cross relation | 0 |
| scope | wrong entity、cross entity | 0 |
| time | expired、not-yet-valid | 0 |
| revocation | revoked token | 0 |
| replay | second use | 0 |
| key lifecycle | unknown key ID、revoked key | 0 |
| integrity | tampered payload、tampered signature | 0 |
| parser | algorithm confusion、malformed、oversized | 0 |
| issuance | missing scope、missing permission | 0 |
| policy | old policy version | 0 |
| router boundary | unauthorized router proposal | 0 |

测试套件还单独验证了 concurrent replay：两个线程并发使用同一 token 时，
只有一次 access 成功，另一次得到 `replay`，总 memory access 为 1。

## 产物与敏感信息检查

正式产物：

```text
artifacts/stage_d1/
├── stage_d1_summary.json
├── capability_attack_matrix.csv
├── source_sha256_manifest.json
├── resolved_config.json
├── protocol_status.json
└── artifact_sha256_manifest.json
```

所有 JSON 均以 `allow_nan=False` 生成，并重新通过严格 JSON 解析。CSV 和 JSON
不包含 capability token、HMAC key、private value、private answer、模型
checkpoint 或 optimizer state。key ring 和 replay/revocation store 只存在于
审计进程内存。

## 测试

正式审计前：

```text
42 passed（D1 定向测试）
313 passed（全仓库）
```

覆盖范围包括 typed API、有效 capability、subject/entity/relation scope、
permission、TTL、policy version、revocation、replay、并发 replay、key rotation、
未知/撤销 key、payload/signature 篡改、algorithm confusion、严格 JSON、
unknown RelationId、router proposal、source hash、CLI、smoke 和 C3 门禁。

正式产物生成后再次执行全仓库测试，结果记录在本报告最终提交说明中。

## 限制与非声明

本阶段明确不声称：

- token payload 机密性；
- 外部调用者身份已经由 IAM 认证；
- router 与 trusted gateway 已完成进程级或主机级隔离；
- replay/revocation store 具备分布式持久性；
- key ring 已接入 KMS/HSM；
- 签发方与验证方已密码学隔离；
- 真实分布式环境下的 clock-skew 安全；
- private memory 已训练或可用；
- 部署级访问控制安全；
- 密码学安全认证；
- 机器遗忘；
- C3 readiness。

HMAC 是对称构造，持有验证 key 的组件也具备签名能力。当前 authority 与
verifier 都在同一可信边界中；如果未来需要隔离签发与验证，应另行评估标准
数字签名和独立 key management，而不能根据本阶段结果直接外推。

## 下一步

仅允许考虑 Stage D2 的公开或 synthetic value 实验：

```text
AuthorizedMemoryRequest
→ verified capability
→ keyed relation bucket
→ explicit entity lookup
→ public/synthetic value
```

D2 必须继续保留：

```text
semantic router outside TCB
private answer not loaded
no LM answer injection
no confirmation
no original C3
c3_eligible = false
```

在外部 IAM、进程隔离、持久化 revocation/replay、KMS/HSM 和独立安全审查完成
前，不应把本原型描述为部署级授权系统。
