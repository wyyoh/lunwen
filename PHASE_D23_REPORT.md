# Stage D2.3：Fault, Observability and Service-Lifecycle Audit

## 结论

Stage D2.3 在代码冻结提交 `028422bf8bb203e3a683a6e90b67f210edd1261b` 上完成一次正式审计。
51 个预注册场景全部符合预期：

```text
fault_observability_lifecycle_status = passed
public_synthetic_service_prototype_validated = true
d_series_system_audit_complete = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

有限结论是：

> 在单机、本地多进程、AF_UNIX、SQLite 持久状态和公开/合成 canary 条件下，
> D2.2 的 capability-gated 服务边界在预注册的故障恢复、受控可观测性、字节级
> IPC、本地端点和状态回滚矩阵内保持 fail closed。

这不是部署级安全认证，不证明跨主机一致性、真实 IAM/KMS、第三方 APM/远程模型保密性
或进程内存清零。

## 1. 冻结协议

配置 SHA-256：

```text
1df0c083f5b1f8a9b39a9f8f93b54ed9aa10610952577058a6397a6d94386f69
```

源码 manifest payload SHA-256：

```text
96f735ed144af946f2ed8b616a902d80b1308e0c5ff1e447df59a92fa323f794
```

D1、D2、D2.1 与 D2.2 的受保护源码 hash 保持：
`upstream_d1_d2_d21_d22_hashes_preserved = true`。

生命周期被显式拆分为：

```text
capability_consumed
→ record_decrypted
→ generator_invoked
→ output_committed
→ response_delivered
```

任何不确定状态均不恢复 single-use capability。

## 2. 正式结果

| 类别 | 场景数 | 通过 |
|---|---:|---:|
| 授权端到端 | 3 | 3 |
| 可观测性 | 12 | 12 |
| 故障一致性 | 13 | 13 |
| IPC/本地端点 | 15 | 15 |
| 状态备份/恢复 | 8 | 8 |
| **合计** | **51** | **51** |

| 核心指标 | 结果 |
|---|---:|
| authorized end-to-end delivery accuracy | 1.0 |
| plaintext observability occurrence | 0 |
| secret metric label occurrence | 0 |
| secret trace attribute occurrence | 0 |
| secret error report occurrence | 0 |
| post-crash double release | 0 |
| post-timeout double release | 0 |
| post-restore replay success | 0 |
| state rollback acceptance | 0 |
| unauthorized local socket connection | 0 |
| stale ticket acceptance | 0 |
| cross-epoch ticket acceptance | 0 |
| malformed IPC plaintext release | 0 |

## 3. 可观测性边界

structured logs、metrics labels、trace attributes/baggage、error report、retry queue、
profiling、supervisor、health/debug 和 audit event 均使用字段白名单与敏感值阻断。
高敏感值不会成为 metric label 或 trace attribute。正式产物只保留拒绝原因、计数、
schema 和 digest，不保存 plaintext、token、key 或 session credential。

## 4. 故障与生命周期

磁盘满、SQLite busy、数据库截断/损坏、部分 socket write、响应前崩溃、client timeout、
generator 完成但 gateway 丢失响应、supervisor 重启、时钟漂移、key rotation 与 record
migration 并发均按冻结顺序审计。磁盘满和系统时钟场景使用可重复的应用层故障注入，
没有耗尽宿主机磁盘或修改宿主机时钟。

## 5. 字节级 IPC 与本地端点

受控 proxy 在内存中检查真实 length-prefixed JSON frame；正式 artifact 仅保存 frame
SHA-256、长度和 `raw_bytes_persisted=false`。stale socket、symlink、错误目录权限、
SO_PEERCRED UID、超长/深层/duplicate/truncated frame、乱序 ticket、慢发送与无效连接
洪泛均 fail closed。

## 6. 状态回滚与 epoch

内部 lifecycle ticket 绑定 `state_epoch`、`policy_epoch` 与
`gateway_instance_epoch`。主状态 metadata 使用 HMAC 完整性标记，并与独立 monotonic
anchor 比较。旧 backup、数据库替换、旧 policy/state/gateway epoch ticket 均被拒绝。

该 anchor 仍是同主机文件；若攻击者同时回滚主状态与 anchor，当前原型不能检测。
真实部署需外部可信单调计数器、KMS/HSM 或等价控制。

## 7. 有限状态

```text
fault_observability_lifecycle_status = passed
public_synthetic_service_prototype_validated = true
d_series_system_audit_complete = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

D 系列系统审计至此停止。本阶段没有加载 private answer，没有训练 private value memory，
没有执行 LM fine-tuning、confirmation、答案注入或 key attack。
