# 统一威胁模型

## 1. 适用范围

本威胁模型适用于 D1–D2.3 的冻结研究原型：

```text
explicit authenticated request
→ capability gateway
→ typed keyed encrypted memory
→ ephemeral generation adapter
→ response delivery
```

所有 value 均为运行期随机的 public/synthetic canary。模型不包含真实个人信息、
真实凭据、private answer 或 production secret。

本模型用于限定论文主张，不是部署风险评估或安全认证。

## 2. 保护资产

| 资产 | 目标 |
|---|---|
| Capability HMAC key | 不进入 memory、generator、日志或 artifact |
| Release-ticket HMAC key | 只存在于 gateway/memory 原型 TCB |
| Data encryption key | 不进入 gateway、generator、日志或 artifact |
| Capability token | 不进入 generator、memory record 或长期可观测性 |
| Session credential | 不进入 memory/generator/长期可观测性 |
| Synthetic plaintext | 只在授权 request 的短生命周期数据路径出现 |
| Policy/scope | subject、entity、relation、permission 不可被 router 扩大 |
| Replay/revocation state | crash/restart/restore 后保持 fail closed |
| Lifecycle/epoch state | 防止旧 ticket 或旧应用状态恢复授权 |
| Audit integrity | 正式产物不包含 plaintext/key/token，且由 SHA-256 manifest 绑定 |

## 3. 攻击者能力

攻击者可以：

1. 完全控制自然语言输入、用户 prompt 和 router proposal；
2. 构造或篡改未认证的 subject/entity/relation 请求；
3. 重放、替换、截断或修改 capability 和内部 IPC message；
4. 尝试跨 subject、跨 entity、跨 relation 使用已获得的 token；
5. 连接本地 AF_UNIX endpoint，使用错误 authkey、错误 UID 或伪造 service identity；
6. 发送超长、深度嵌套、duplicate-key、截断、乱序或慢速 IPC；
7. 并发消费 single-use capability 或 release ticket；
8. 在预注册生命周期点触发 crash、timeout、response loss 和 supervisor restart；
9. 触发 SQLite busy/locked、确定性 disk-full fault 和损坏/截断状态；
10. 读取应用层 structured logs、metrics、traces、error report、retry queue、
    profile、supervisor log、health/debug endpoint 与 audit event；
11. 复制、替换或回滚主状态数据库及旧 backup；
12. 尝试将 ciphertext、record metadata 或 epoch ticket 跨 bucket/epoch 使用。

这些能力覆盖研究原型中的应用层和本地服务边界。

## 4. 攻击者不能做什么

当前证据不覆盖攻击者能够：

- 获得 root、kernel、hypervisor 或调试器权限；
- 直接读取任意进程地址空间、swap、core dump 或物理内存；
- 同时控制主状态与外部可信 monotonic counter/KMS；
- 破解 HMAC-SHA-256 或 AES-256-GCM 的标准安全假设；
- 控制真实 IAM、KMS/HSM 或 CA；
- 控制云供应商内部日志、模型服务或 distributed tracing backend；
- 造成跨主机 Byzantine failure；
- 利用 timing、cache、功耗、EM 或 speculative-execution side channel；
- 进行硬件取证、冷启动攻击或 GPU 显存取证。

其中部分组件在当前原型中并不存在；“不能做”表示它们不属于已验证攻击模型，
不是系统对这些攻击具有防御能力。

## 5. 信任假设

| 假设 | 当前实现 | 限制 |
|---|---|---|
| Principal 来自可信身份系统 | 进程内 mock identity provider | 不是真实 IAM |
| Policy/capability authority 正确签发 | 同一原型 TCB 内 HMAC | verifier 也持有签发 key |
| OS 执行 AF_UNIX 权限和 SO_PEERCRED | 单机本地进程 | 不覆盖容器/多租户 host |
| SQLite 事务和 fsync 语义成立 | BEGIN IMMEDIATE/FULL/DELETE | 不覆盖分布式一致性 |
| cryptography AESGCM 正确实现 | 标准高层 API | 无形式化实现审计 |
| Parent harness 正确装配 key | 测试父进程持有全部 key | 不是 KMS/HSM 隔离 |
| Monotonic anchor 不与主状态同时回滚 | 同主机独立 SQLite 文件 | 强攻击者可同时回滚 |
| 可观测性只经过安全投影 | 本地受控 sink | 不覆盖第三方 APM |
| Generator 无 memory/capability 工具 | 受控 copy worker/adapter | 不代表任意 agent 系统 |

## 6. 授权 TCB

授权 TCB 包含：

- trusted identity provider；
- policy engine；
- capability authority；
- capability verifier；
- replay/revocation state；
- capability gateway；
- release-ticket verifier；
- typed keyed memory；
- data-key ring 与 AEAD codec；
- lifecycle/epoch state；
- generation adapter 的最小可信包装层；
- 可观测性敏感字段过滤器。

生成模型或确定性 worker 收到 plaintext 后属于该 request 的数据处理与保密边界，
但不属于授权决策 TCB。

TCB 外：

- natural-language router；
- `SuggestedRelation`；
- 用户 prompt；
- 未认证的 client request；
- 自由生成组件的语义决策；
- 外部建议与检索系统；
- 安全投影之后的 metrics/tracing backend。

## 7. 安全目标

### P1：Router non-authority

自然语言 router 只能提出建议，不能签发 capability 或直接触发 memory access。

### P2：Authorization before data access

principal、capability、scope、expiry、revocation 与 replay 检查必须先于 lookup/decrypt。

### P3：Exact typed scope

每次授权只允许一个明确的 `(subject, entity, RelationId, permission)` 作用域；
cross-relation candidate count 保持 0。

### P4：Key separation

capability authentication key 与 data encryption key 不相等，也不跨越禁止角色。

### P5：Authenticated record binding

AEAD AAD 绑定 record/entity/relation/data-key/version/schema/algorithm，防止 record
置换或 metadata 篡改后释放 plaintext。

### P6：Single-use and fail-closed lifecycle

capability 在 plaintext release 前原子消费。crash、timeout 或 response loss 不恢复
原 capability；重试需要新授权。

### P7：Observability minimization

plaintext、token、key 和 session credential 不能成为 log field、metric label、
trace attribute/baggage、error payload 或持久 artifact。

### P8：Rollback/epoch rejection

旧 state、policy 或 gateway epoch ticket 被拒；主状态落后 monotonic anchor 时
整个路径 fail closed。

## 8. 已验证的不变量

```text
semantic_router_in_tcb = false
unauthorized_lookup/decrypt/plaintext_release = 0
cross_subject/entity/relation plaintext release = 0
replayed capability plaintext release = 0
cross_process_double_release = 0
post_restart/post_restore replay success = 0
plaintext observability occurrence = 0
state rollback acceptance = 0
malformed IPC plaintext release = 0
```

这些值来自不同阶段的预注册矩阵，不能简单相加为某个总体安全概率。

## 9. 残余风险

1. **身份风险**：调用者身份仍依赖 mock principal。
2. **对称验证风险**：HMAC verifier 能生成 token，签发—验证未密码学隔离。
3. **同主机回滚风险**：攻击者同时回滚 state 与 anchor 时无法检测。
4. **内存风险**：Python bytes、IPC buffer 与模型 context 无可靠安全清零。
5. **可观测性外推风险**：本地 sink 的零泄漏不能外推到第三方 APM。
6. **生成风险**：受限 copy probe 不能证明自由生成模型不会泄漏。
7. **分布式风险**：SQLite 文件锁不提供跨主机原子性。
8. **端点竞态风险**：本地 socket 创建与检查仍可能存在 OS/部署相关 TOCTOU 面。
9. **元数据风险**：authenticated capability 不提供 payload 机密性。
10. **外部验证风险**：D 系列是自建原型矩阵，不是独立渗透测试。

## 10. 主张边界

允许：

> 在公开/合成 value 的冻结单机原型和预注册矩阵内，系统维持了
> authorization-before-access、typed scope、single-use 与 fail-closed 数据流。

禁止：

> 系统实现了安全的私有记忆、部署级访问控制、密码学安全或机器遗忘。
