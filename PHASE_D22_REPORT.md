# Stage D2.2：Isolated-Service Probe

## 结论

Stage D2.2 在代码冻结提交 `06ca095b575b96447e963927287abb672d324ecb` 上完成一次正式审计。
35 个预注册场景全部符合预期：

```text
isolated_service_generation_status = passed
service_boundary_validated_with_public_synthetic_values = true
ready_for_d2_3_fault_and_observability_probe = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

有限结论是：

> 在当前单机、本地多进程、AF_UNIX socket、确定性复制 worker 和运行期随机
> 128-bit synthetic canary 条件下，capability gateway、typed keyed memory 与
> generator 的进程边界保持了 authorization-before-lookup-before-decrypt、
> single-use fail-closed 和最小 IPC；未观察到跨服务重复释放或应用层持久化泄漏。

这不是部署级安全认证，不证明跨主机或分布式一致性、进程内存安全清零、远程模型保密性
或 private memory 安全。

## 1. 冻结协议与边界

正式配置 SHA-256：

```text
4c174fd4aaa8295a348bad400e5a0ba06d0be1e251a93eb06c054537fb8c2b48
```

源码 manifest payload SHA-256：

```text
17c84ad0e37bc78e9734e2d15657d5a6fa18197f01d98ca52e88c7a5fe0b7753
```

固定数据流：

```text
client / untrusted router process
        ↓ request + capability
capability gateway process
        ↓ authenticated minimal release ticket
typed keyed memory process
        ↓ one ephemeral synthetic value
deterministic generation worker
        ↓ exact one-shot output
```

gateway 不持有 data encryption key；memory 不持有 capability HMAC key，也不接收
自然语言或外部 capability；generator 不持有三类 key，不接收 capability token，且没有
gateway、memory 或工具接口。D1/D2 核心源码 hash 保持：
`upstream_d1_d2_core_hashes_preserved = true`。

本地服务使用 Python `multiprocessing` 的 `spawn` 启动方式与 AF_UNIX
`Listener`/`Client`。连接级 authkey 提供 HMAC challenge，但这里仍只把它表述为同一原型
TCB 内的对称服务认证。[Python multiprocessing 文档](https://docs.python.org/3.12/library/multiprocessing.html)

## 2. 正式结果

| 类别 | 场景数 | 通过 |
|---|---:|---:|
| 正向端到端 | 6 | 6 |
| capability/identity 负向 | 6 | 6 |
| 服务边界 | 10 | 10 |
| IPC 最小化 | 3 | 3 |
| 进程崩溃 | 5 | 5 |
| 多实例/重启 | 5 | 5 |
| **合计** | **35** | **35** |

| 核心指标 | 结果 |
|---|---:|
| authorized end-to-end delivery accuracy | 1.0 |
| unauthorized service call | 0 |
| unauthorized memory service access | 0 |
| generator→memory connection | 0 |
| cross-process double release | 0 |
| cross-instance replay success | 0 |
| post-restart replay success | 0 |
| gateway log plaintext occurrence | 0 |
| memory log plaintext occurrence | 0 |
| generator log plaintext occurrence | 0 |
| IPC persistence plaintext occurrence | 0 |
| crash recovery plaintext occurrence | 0 |

## 3. 崩溃、并发与持久状态

在 capability 消费前、消费后 lookup 前、decrypt 后 generator 前、generator 收值后输出前，
以及输出完成后响应前分别注入进程退出。所有不确定状态均 fail closed：原 capability
不会产生第二次 plaintext release，重试需要新 capability。

两个 gateway 并发消费同一 capability、两个 memory worker 并发消费同一 release
ticket 时均只有一个成功路径。重启后 replay 和 revocation 状态仍有效：

```text
cross_process_double_release_count = 0
cross_instance_replay_success_count = 0
post_restart_replay_success_count = 0
```

状态存储使用 SQLite `BEGIN IMMEDIATE` 争用写事务，并固定 `synchronous=FULL` 与
DELETE journal。SQLite 文档说明 `BEGIN IMMEDIATE` 会立即启动写事务；这里的证据只适用于
单主机文件状态，不外推为分布式一致性保证。
[SQLite Transactions](https://www.sqlite.org/lang_transaction.html)

所有服务启动时将 `RLIMIT_CORE` 设为 0；Python 将该资源定义为 core file 的最大大小。
[Python resource 文档](https://docs.python.org/3.12/library/resource.html#resource.RLIMIT_CORE)

## 4. IPC 最小化

安全 schema 审计仅持久化字段名和布尔标记，不保存 raw IPC：

- client→gateway：session credential、typed request、capability 与 instruction ID；
- gateway→memory：单个 authenticated release ticket；
- memory→generator：单个 ephemeral value、request-local nonce 与固定 instruction ID。

capability/HMAC/data key 不进入 generator IPC；自然语言 prompt 不进入 memory IPC。
由于未做 raw packet capture，这一结论依赖严格 JSON schema、进程配置和安全事件审计。

## 5. Canary 与泄漏扫描

D2.1 的歧义字段已经拆分：

```text
persisted_output_digest_count = 7
runtime_canary_count = 104
runtime_canary_variant_count = 728
scanned_location_count = 302
```

扫描覆盖三类服务 stdout/stderr 日志、结构化安全事件、SQLite 状态、临时运行目录、
进程 `/proc/<pid>/cmdline` 与 `/proc/<pid>/environ`，并检查 raw、大小写、hex、base64、
字符分隔与去连字符变体。应用层扫描结果为 0。正式产物只记录 digest、场景状态、字段
schema 和计数，不记录 plaintext、token、session credential 或 key。

该扫描不能证明 Python/操作系统内存被安全清零，也未覆盖 GPU cache、远程 tracing 或
第三方服务日志。

## 6. 状态与下一步

```text
isolated_service_generation_status = passed
service_boundary_validated_with_public_synthetic_values = true
ready_for_d2_3_fault_and_observability_probe = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

D2.3 若继续，应只研究 public/synthetic value 下更系统的 fault/observability 边界；
本阶段不加载 private answer，不训练 private value memory，不执行 LM fine-tuning、
confirmation、答案注入或 key attack。
