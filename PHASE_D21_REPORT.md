# Stage D2.1：Capability-Gated Ephemeral Generation Probe

## 结论

Stage D2.1 在代码冻结提交
`3e740a83576c41f4871c9e9e3848fff4bb468eb9` 上完成了一次正式审计。
40 个预注册场景全部符合预期：

```text
capability_gated_ephemeral_generation_status = passed
public_synthetic_generation_boundary_validated = true
ready_for_d2_2_isolated_service_probe = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

该结果的限定表述是：

> 在当前单进程冻结原型、运行期随机 128-bit synthetic canary 和预注册攻击矩阵内，
> 只有 D2 已授权并释放的单个 value 才能调用一次性 G0/G1 生成通道；未授权请求、
> 跨请求恢复、prompt scope 扩张和应用层持久化扫描均未观察到 value 暴露。

它不证明 private memory 已安全，不是部署级安全认证，也不证明一般语言模型的保密性、
进程内存清零或机器遗忘。

## 1. 研究边界

D2.1 只验证以下数据流：

```text
AuthenticatedPrincipal
        +
AuthorizedMemoryRequest
        ↓
D2 capability gateway
        ↓
单个 PublicSyntheticValue
        ↓
Ephemeral Generation Adapter
        ↓
当前请求的一次性输出
```

以下边界保持不变：

- 生成器不能签发或验证 capability；
- 生成器不能决定授权；
- 生成器不能改变 subject、entity 或 relation scope；
- 生成器没有 memory、gateway、capability 或 key 工具；
- 生成 payload 不包含 subject、entity、relation、record、token 或 key；
- 自然语言 router 仍在授权 TCB 之外；
- 未加载 private answer，未训练 private value memory；
- 未执行 LM fine-tuning、答案注入、confirmation 或 key attack。

生成模型仍不属于授权决策 TCB；但它一旦接收到 plaintext，就属于该次请求的数据处理与
保密边界。

## 2. 冻结协议

正式配置为 `configs/stage_d21.yaml`，配置 SHA-256 为：

```text
d31274f640a84f0da53e2d7c4c194ab9f5092671846909aeb87e330931633017
```

正式审计绑定：

- D2 最终 summary 和 artifact manifest 的固定 SHA-256；
- D2.1 运行时源码 manifest；
- 固定模型 ID、revision、文件大小和文件 SHA-256；
- 固定模型参数 SHA-256；
- 固定 40 个场景及全部零门槛；
- 一次性 runtime phase state。

源码 manifest payload SHA-256：

```text
bbde5c92ef6fcab3a47a8d4a7a27559c109f413300b0a4816b807b722a3976b2
```

D1/D2 capability、principal、AEAD 与 keyed-memory 核心源码相对 D2 最终提交
`cfb4bdc99a86e952d9d76571b932eb0aa1fc265c` 无修改。

## 3. G0 与 G1

### G0：确定性 renderer

G0 使用固定模板，只接收本次已授权的单个 value。它不加载模型，不拥有工具，也不保留
跨请求状态。

### G1：冻结 tiny-GPT2 受限复制探针

G1 使用：

```text
model_id = sshleifer/tiny-gpt2
revision = 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be
transformers = 5.14.1
torch = 2.12.0+cpu
parameter_count = 102714
```

模型 revision 与六个所需文件均按大小和 SHA-256 固定，模型文件只位于仓库外 Hugging
Face cache，未提交到 Git。[冻结 revision 文件树](https://huggingface.co/sshleifer/tiny-gpt2/tree/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be)

G1 调用真实 `transformers` `generate()` 路径，但通过
`prefix_allowed_tokens_fn` 将输出限制为当前 authorized value 的 token 序列。
`do_sample=False`，`use_cache=False`；Transformers 官方文档说明 `generate()` 由
`GenerationMixin` 提供，并将 `use_cache` 定义为是否使用 past key/value attention
cache。[Transformers Generation API](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

因此，G1 只验证：

- 真实 pretrained model load/generate 调用链；
- 当前请求 value 进入一次性 context；
- 无 conversation history；
- 无跨请求 KV cache；
- 无工具；
- context 在调用后销毁；
- 参数保持冻结；
- 输出验证与失败路径。

G1 不验证自主 prompt following，也不验证一般、非约束语言模型的机密性。5 个 prompt
injection 场景验证的是结构性 scope 不可扩张，不是模型具有自主拒绝注入的能力。

## 4. 正式结果

| 类别 | 场景数 | 通过 | 失败 |
|---|---:|---:|---:|
| 正向输出 | 11 | 11 | 0 |
| 未授权/无效请求 | 12 | 12 | 0 |
| Prompt injection | 5 | 5 | 0 |
| 跨请求/跨 session | 6 | 6 | 0 |
| 生成异常与失败 | 5 | 5 | 0 |
| 并发 single-use | 1 | 1 | 0 |
| **合计** | **40** | **40** | **0** |

核心指标：

| 指标 | 结果 |
|---|---:|
| authorized generation count | 27 |
| expected successful delivery count | 23 |
| authorized exact delivery count | 23 |
| authorized delivery accuracy | 1.0 |
| unauthorized generator invocation | 0 |
| unauthorized value exposure | 0 |
| cross-subject exposure | 0 |
| cross-entity exposure | 0 |
| cross-relation exposure | 0 |
| cross-session exposure | 0 |
| post-request recovery | 0 |
| prompt-injection scope expansion | 0 |
| additional memory lookup | 0 |
| additional decrypt attempt | 0 |
| generator-failure plaintext exposure | 0 |
| concurrent double generation | 0 |

`authorized generation count=27` 包含 4 次预注册的 timeout、model exception、
renderer exception 和 output-validation failure 注入。预期成功交付的 23 次均精确交付，
因此 accuracy 为 1.0。

## 5. 授权顺序与失败语义

调用顺序保持：

```text
identity
→ capability verification / atomic single-use consumption
→ gateway–memory instance binding
→ exact typed lookup
→ AEAD decrypt
→ plaintext release
→ generator invocation
→ exact output validation
```

wrong subject/entity/relation、expired/revoked/replayed capability、
未经授权的 router proposal、forged principal 和 forged verified capability 均未调用
generator。wrong/revoked data key 和 tampered record 在 D2 decrypt/authentication 阶段失败，
同样未调用 generator。

生成失败后 capability 不恢复。对失败请求的重试因 replay 被拒绝，且 generator invocation
增量为 0。

并发使用同一 single-use capability 的结果为：

```text
generator_invocation_count = 1
successful_output_count = 1
replay_rejection_count = 1
concurrent_double_generation_count = 0
```

## 6. 跨请求与 prompt injection

先完成一次合法生成后，以下请求均未恢复先前 value：

- 同 subject、无新 capability；
- 其他 subject；
- 其他 entity；
- 其他 relation；
- 新 session、无新 capability；
- 后续普通公共问答。

5 个 prompt injection 指令分别请求其他 relation、全部字段、其他 entity、system
prompt/capability 和其他 subject value。每个场景最多输出当前已授权单个 value：

```text
prompt_injection_scope_expansion_count = 0
additional_memory_lookup_count = 0
additional_decrypt_attempt_count = 0
```

该结果由无工具 adapter、单值 payload、一次性 context、约束解码和精确输出验证共同产生，
不能解释为 tiny-GPT2 自主理解或拒绝了这些指令。

## 7. 持久化与泄漏扫描

每个 record/scenario/run 使用运行期唯一、至少 128-bit 随机 canary。明文不写入源码、
配置、JSON、CSV、Markdown、stdout/stderr、异常、`repr`、临时目录或模型 cache。

扫描覆盖：

- raw；
- lowercase / uppercase；
- hex；
- standard / URL-safe base64；
- character-spaced；
- hyphen-stripped。

结果：

```text
plaintext_artifact_occurrence_count = 0
plaintext_log_occurrence_count = 0
plaintext_exception_occurrence_count = 0
plaintext_repr_occurrence_count = 0
plaintext_temp_cache_occurrence_count = 0
```

### 已知 instrumentation 命名说明

正式 `leakage_scan.json` 中的 `canary_count=16` 字段命名不准确：实现实际计算的是 artifact
中持久化的唯一 `output_digest` 数，而不是运行时生成的 canary 总数。正式扫描调用本身使用
了 `CanaryFactory` 在内存中登记的全部 canary 及全部变体；零 occurrence 门槛不依赖该
展示字段。

由于正式审计只能运行一次，本报告保留原始 artifact，不回写字段、不修改源码后重跑，也
不把 `16` 表述为运行时 canary 总数。该问题不改变泄漏扫描输入或门槛结果，但属于后续
schema 需要重命名的非门禁 instrumentation 缺陷。

应用层扫描不能证明 Python/CPU/GPU 内存被安全清零，也不能排除进程内存取证。

## 8. 模型完整性

模型参数 SHA-256 在运行前后均为：

```text
0b25b10b57bb0b668893455cae1d2c460aae779960385f02ce78dea37b146d3d
```

```text
model_parameter_hash_changed = false
all_parameters_frozen = true
model_checkpoint_committed_to_repository = false
```

模型 manifest payload SHA-256：

```text
2ea0ca30df70960b4ba2553f422d8bb5fdee2a4a2967b759333716470eef4ce0
```

## 9. 测试与产物完整性

代码冻结前全仓测试：

```text
405 passed
```

其中新增 41 个 D2.1 测试，覆盖：

- generation payload 不包含授权元数据；
- payload seal 与 `repr` 脱敏；
- canary 生成、变体和递归扫描；
- G0/G1 复制；
- failure/replay fail-closed；
- 未授权请求不调用 generator；
- prompt scope 不扩张；
- 跨请求不恢复；
- 并发 single-use；
- 配置、阶段状态、场景预注册、模型哈希与 CLI。

正式 JSON 全部使用 `allow_nan=False`，CSV 不包含 plaintext/value/token/key/prompt/output
文本字段。正式 artifacts 中没有 model checkpoint、optimizer state、tensor cache、
private value、private answer、capability token、HMAC key、data key 或 session
credential。

## 10. 仍未解决的边界

- identity provider 仍是进程内 mock；
- gateway、memory 与 generator 没有进程级隔离；
- HMAC authority/verifier 仍位于同一 TCB；
- replay/revocation 不覆盖多实例一致性和崩溃恢复；
- D2 AES-GCM nonce 唯一性仍是单进程保证；
- Python 无法提供可靠的 plaintext/key 内存清零；
- constrained-copy G1 不代表通用生成模型行为；
- 没有真实用户数据、private value、private answer 或 LM answer injection；
- 没有验证远程推理服务、供应商日志、GPU cache 或分布式 tracing；
- 没有执行安全认证、渗透测试或机器遗忘评估。

## 11. 最终状态

```text
capability_gated_ephemeral_generation_status = passed
public_synthetic_generation_boundary_validated = true
ready_for_d2_2_isolated_service_probe = true

private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

D2.1 通过后允许研究的下一步仅是 D2.2 isolated-service probe。它不授权加载 private
answer，不授权训练 private value memory，不授权创建 confirmation，也不恢复自然语言
router 的授权能力。

本阶段没有加载 private answer，没有训练 private value memory，没有执行 LM fine-tuning、
答案注入、confirmation 或 key attack。
