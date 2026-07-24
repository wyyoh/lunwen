# Stage D2：Capability-Gated Authenticated Keyed Memory

## 结论

Stage D2 已在冻结提交
`04834dc6389eb712a7e2ba36a8ec5d5027938ffb`
上完成唯一一次正式审计。37 个预注册场景全部符合预期：

| 指标 | 结果 |
|---|---:|
| 正向场景 | 9 / 9 |
| 授权负向场景 | 13 / 13 |
| record/AEAD 负向场景 | 14 / 14 |
| 并发 single-use 场景 | 1 / 1 |
| 授权 plaintext release | 10 |
| 未授权 plaintext release | 0 |
| 未授权 memory lookup | 0 |
| 未授权 decrypt attempt | 0 |
| cross-subject plaintext release | 0 |
| cross-entity plaintext release | 0 |
| cross-relation plaintext release | 0 |
| tampered-record plaintext release | 0 |
| wrong/revoked-key plaintext release | 0 |
| replayed-token plaintext release | 0 |
| concurrent double release | 0 |
| data-key rotation | 1 |
| record migration | 1 |

有限状态结论为：

```text
capability_gated_keyed_memory_status = passed
ready_for_d2_1_public_generation_probe = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

`passed` 只表示冻结的单进程原型通过了本阶段预注册的 principal、capability、
key lifecycle、AEAD metadata binding 和并发消费测试，不是部署级安全认证。

## D1 继承边界

D2 没有修改 D1 capability contract、HMAC token、policy、replay store 或 C2.4
`RelationId`。正式源 manifest 绑定：

```text
D1 summary SHA-256
de5ceeb280097c5bb5625327e739a9b6365d65eace1a0ccaad7639435d19b3a1

D1 final artifact manifest SHA-256
c43e75262d56846072f9484ca870ae4bd981d8d03427ebc545ed6c93499d6eef

C2.4 typed contract SHA-256
c8a3b5fd33621349659e504b1bd31502688896cf63207ae59f644849a0d96692
```

D1 正式 manifest 中除后续阶段必然扩展的 `.gitattributes` 和 CLI 外，D1 核心
runtime 文件均保持原 SHA-256。

继续固定：

```text
selective_router_research_status = stopped_external_validation_failed
semantic_router_in_tcb = false
suggested_relation_authorizes_memory = false
```

D2 没有重新引入自然语言 router、R4/R5、threshold、conformal 或 OOS 数据。

## 架构

```text
opaque mock session credential
        ↓
TrustedMockIdentityProvider
        ↓
sealed AuthenticatedPrincipal
        +
AuthorizedMemoryRequest
        ↓
principal.subject == request.subject
        ↓
D1 capability verification
        ↓
principal.subject == capability.subject == request.subject
        ↓
gateway–memory instance binding
        ↓
exact (entity_id, RelationId) lookup
        ↓
independent data-key ring
        ↓
AES-256-GCM authenticated decryption
        ↓
PublicSyntheticValue
```

验证顺序是 D2 的核心不变量：

```text
trusted identity
→ capability
→ instance binding
→ lookup
→ decrypt
→ plaintext release
```

所有身份或 capability 负向场景都在 lookup/decrypt 前停止。

## AuthenticatedPrincipal

D2 新增内部类型：

```python
@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject_id: str
    session_id: str
    authentication_context: str
```

principal 带内部 seal，只能由 trusted mock identity provider 产生。gateway 的
公开输入是 32-byte opaque session credential，不接受调用者直接提交
`AuthenticatedPrincipal`。

gateway 强制比较：

```text
principal.subject_id
==
request.subject_id
==
verified capability.subject_id
```

本阶段 identity provider 是进程内 mock，不是真实 IAM、会话服务或调用者
认证。因此不能把该结果外推为真实身份安全。

## Gateway–memory instance binding

代码冻结前审查发现，D1 的 Python factory 可被导入并产生结构有效的
`VerifiedCapability`。D2 因此没有让 memory 直接信任任意 D1 verified object：

- 每个 gateway–memory 实例共享唯一的内部 binding；
- 直接调用 memory 的公开 `retrieve_authorized` 永远拒绝；
- 导入 D1 factory 构造的 verified object 不能触发 lookup；
- 使用其他 gateway 的 binding 访问当前 memory 也会拒绝；
- 只有 principal 和 D1 capability 验证后的绑定 gateway 能进入内部 release
  路径。

这减少了同进程 API 误用，但 Python 私有字段不是安全边界；真正部署仍需要
进程或服务级隔离。

## Capability key 与 data key 分离

D2 固定：

```text
capability authentication key = HMAC-SHA-256 key ring
data encryption key = independent AES-256-GCM key ring
capability key ≠ data key
```

两类 key：

- 使用不同实现类型；
- 使用不同 key namespace：`cap-*` 与 `data-*`；
- 独立随机生成；
- 审计中验证 key material 不相同；
- 不写入配置、CSV、JSON 或报告；
- 仅存在于运行进程内存。

D1 HMAC key 没有被用于 memory 加密。

## AEAD record

record 类型为：

```python
@dataclass(frozen=True)
class EncryptedMemoryRecord:
    record_id: str
    entity_id: str
    relation_id: RelationId
    data_key_id: str
    record_version: int
    nonce: bytes
    ciphertext: bytes
    schema_version: int
    algorithm: str
```

实现固定为：

```text
algorithm = AES-256-GCM
library = cryptography 49.0.0
key = 32 bytes
nonce = 12 bytes
tag = 16 bytes
```

使用 `cryptography.hazmat.primitives.ciphers.aead.AESGCM` 高层接口，不实现
自定义加密、XOR、置换或自行组合 encrypt-then-MAC。

AAD 以 canonical JSON 精确绑定：

```text
schema_version
algorithm
record_id
entity_id
relation_id
data_key_id
record_version
```

因此 key、nonce、ciphertext、tag 或任一 AAD 元数据变化都会拒绝 release。
同一 data key 下的 nonce 由加锁集合保证本进程内不重复。

参考：

- [NIST SP 800-38D：GCM/GMAC](https://csrc.nist.gov/pubs/sp/800/38/d/final)
- [cryptography AEAD/AESGCM 文档](https://cryptography.io/en/stable/hazmat/primitives/aead/)

## Value 边界

D2 只使用运行期由 `secrets.token_bytes(32)` 生成的随机 synthetic bytes：

- 不来自真实用户；
- 不来自 C2 private value；
- 不包含 private answer、密码、credential 或个人信息；
- 不写入 source/config；
- 不写入 JSON/CSV；
- artifact 只记录 `value_released`、SHA-256 `value_digest`、`record_id` 和
  reason。

`PublicSyntheticValue` 与 release result 的 `repr` 隐藏 payload。

## 正向审计

9 个正向场景全部通过：

1. registry record 合法 release；
2. city record 合法 release；
3. access record 合法 release；
4. 新 active data key 写入并读取新 record；
5. rotation 后保留旧 data key 并读取旧 record；
6. 旧 record 迁移到新 key、version 增加后读取；
7. multi-entity scope 访问 entity A；
8. 同一显式 scope 的独立 capability 访问 entity D；
9. single-use capability 第一次读取。

record migration 在 TCB 内部完成 decrypt/re-encrypt，不向管理调用方返回
plaintext。migration 成功后 record version 增加，使用 active data key 和新
nonce。

## 授权负向审计

13 个授权负向场景全部在 lookup/decrypt 前拒绝：

```text
invalid session
wrong principal
wrong subject
wrong entity
wrong relation
insufficient permission
expired capability
not-yet-valid capability
revoked capability
replayed capability
revoked signing key
unauthorized router proposal
forged VerifiedCapability
```

合计：

```text
unauthorized_memory_lookup_count = 0
unauthorized_decrypt_attempt_count = 0
unauthorized_plaintext_release_count = 0
```

## Record、key 与 AEAD 负向审计

14 个数据侧负向场景均未释放 plaintext：

```text
unknown data key
revoked data key
wrong data key
ciphertext bit flip
authentication tag tamper
nonce tamper
AAD record ID tamper
entity metadata swap
relation metadata swap
cross-bucket ciphertext swap
record version rollback
algorithm confusion
malformed record
duplicate record ID
```

授权已经成立的数据侧场景允许发生精确 bucket lookup 和 AEAD decrypt attempt，
但 authentication failure 后的 plaintext release 恒为 0。

## 并发 single-use

两个线程并发提交同一个 single-use capability：

```text
successful_release_count = 1
replay_rejection_count = 1
concurrent_double_release_count = 0
```

D1 replay store 的原子 consume 发生在 D2 memory lookup 和 plaintext release
之前。失败线程没有触发 lookup、decrypt 或 release。

该测试只覆盖单进程线程竞争，不覆盖多进程、多实例、网络重试或崩溃恢复。

## 正式产物与完整性

正式产物：

```text
artifacts/stage_d2/
├── stage_d2_summary.json
├── capability_memory_attack_matrix.csv
├── source_sha256_manifest.json
├── resolved_config.json
├── protocol_status.json
└── artifact_sha256_manifest.json
```

正式 source manifest 绑定 17 个文件及 `cryptography==49.0.0`，payload
SHA-256 为：

```text
c727fb5780f356d48b427090467832e0ba45717eae16aec759495cf1b6b85cec
```

所有 JSON 使用 `allow_nan=False` 并进行严格解析。artifact 不包含：

```text
plaintext value
private value
private answer
capability token
capability HMAC key
data encryption key
session credential
model checkpoint
optimizer state
```

## 测试

正式审计前：

```text
51 passed（D2 定向测试）
364 passed（全仓库测试）
```

覆盖：

- principal seal、opaque session 和 subject 三方一致性；
- capability-before-lookup/decrypt；
- gateway–memory instance binding；
- D1 verified factory bypass；
- HMAC/data-key namespace 与 material 分离；
- AES-GCM round trip、tag、nonce uniqueness；
- key/nonce/ciphertext/AAD tamper；
- key rotation、revocation、migration、rollback；
- exact entity/relation bucket；
- duplicate/malformed/algorithm confusion；
- concurrent single-use；
- artifact schema、CLI、配置与 D1 source hash。

正式产物生成后再次执行全仓库测试，结果由最终提交和交付汇报记录。

## 未验证与非声明

D2 没有证明：

- 真实 IAM、会话或调用者认证安全；
- 进程、容器、主机或租户级隔离；
- HMAC authority 与 verifier 的密码学分离；
- 多进程/多实例 replay 与 revocation 一致性；
- 崩溃恢复和持久化原子事务；
- 真实分布式 clock-skew 安全；
- KMS/HSM key custody；
- Python 内存中的 plaintext/key 可可靠清零；
- 侧信道、流量分析或 token metadata 隐私；
- private value memory readiness；
- public LM behavior preservation。

本阶段没有加载 LM 或模型 checkpoint，因此“public behavior side effect”不适用，
不能把未测试解释为无副作用。

## 下一步边界

D2 只允许输出：

```text
ready_for_d2_1_public_generation_probe = true
```

如后续明确启动 D2.1，只能研究公开/合成 value 的受控生成接口，并继续保持：

```text
semantic router outside TCB
no private answer
no private value memory
no confirmation
no key attack
original_c3_allowed = false
c3_eligible = false
```

D2 通过不等于原 C3 可以启动。
