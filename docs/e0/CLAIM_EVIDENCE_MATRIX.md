# 主张—证据矩阵

## 1. 证据等级

| 等级 | 定义 | 可支持的表述 |
|---|---|---|
| S | 类型、接口、source hash 与测试证明的结构事实 | “接口不接收某字段”“只选择一个 bucket” |
| I | 冻结原型上的预注册内部审计 | “在该攻击/故障矩阵中未观察到违规” |
| X | 既有公开数据、官方 split、多 partition 外部实验 | “方法效果未跨基准稳定复现” |
| E | 探索性、合成、非独立数据上的机制证据 | “显示趋势/局部机制效果” |
| N | 当前没有证据 | 不得提出正面主张 |

S/I/X/E 表示证据类型，不表示密码学或部署安全等级。

## 2. 主要论文主张

| ID | 主张 | 核心证据 | 等级 | 结论 | 必须附带的范围 |
|---|---|---|---|---|---|
| C1 | Query canonicalization 能改善 fact-over-template geometry | [C2](../../PHASE_C2_REPORT.md) | I | 支持 | 三关系合成数据；relation accuracy 未达标 |
| C2 | 轻量 relation head 对 lexical-family OOD 不稳定 | [C2.1](../../PHASE_C21_REPORT.md)、[C2.2](../../PHASE_C22_REPORT.md) | I | 支持 | 冻结 SimpleStories core 与公开合成 paraphrase |
| C3 | 连续 relation 输出携带同关系内部 phrase-family nuisance | [C2.3](../../PHASE_C23_REPORT.md) | I | 支持 | S5 与一个 E5-base 候选；不是普遍信息泄漏定理 |
| C4 | Typed RelationId + single bucket 从 memory API 删除连续 relation 通道 | [C2.4](../../PHASE_C24_REPORT.md) | S/I | 支持 | 结构隔离，不代表离散路由正确 |
| C5 | Pairwise evidence 在自建合成压力集上改善 known routing | [C2.4c](../../PHASE_C24C_REPORT.md) | E | 局部支持 | 单模型 AI 参与数据修订；非独立、非正式 |
| C6 | Pairwise evidence 优势没有稳定外部泛化 | [C2.5](../../PHASE_C25_REPORT.md) | X | 支持 | CLINC150/BANKING77、10 partitions、固定 encoder |
| C7 | R3 低 FMAR 主要由 known coverage 崩溃产生 | [C2.5](../../PHASE_C25_REPORT.md) | X | 支持 | 外部 intent/OOS analogue，不是 production FMAR |
| C8 | Multi-candidate set 不是主要 ambiguity 机制 | [C2.4c](../../PHASE_C24C_REPORT.md)、[C2.5](../../PHASE_C25_REPORT.md) | E/X | 支持 | 实际拒绝主要来自 empty-set unknown |
| C9 | Learned semantic router 不应成为授权边界 | C4–C8 的联合证据 | S/I/X | 支持 | 架构原则；不是所有语义模型的形式化不可能性 |
| C10 | Typed authenticated capability 可将 router 移出授权 TCB | [D1](../../PHASE_D1_REPORT.md) | S/I | 支持 | 单进程原型、mock identity、HMAC 同 TCB |
| C11 | Capability 与 data key 分离后可阻止预注册未授权 plaintext release | [D2](../../PHASE_D2_REPORT.md) | S/I | 支持 | public/synthetic value、单进程 |
| C12 | 授权 value 可通过一次性 adapter 交付而不扩大 scope | [D2.1](../../PHASE_D21_REPORT.md) | S/I | 支持 | 受限 copy/generate probe，不代表自由生成保密性 |
| C13 | 多进程服务边界保持 single-use、最小 IPC 与持久 replay 状态 | [D2.2](../../PHASE_D22_REPORT.md) | S/I | 支持 | 单机 AF_UNIX + SQLite |
| C14 | 预注册 fault/observability/lifecycle 矩阵中保持 fail closed | [D2.3](../../PHASE_D23_REPORT.md) | S/I | 支持 | 确定性故障注入；同主机 anchor |
| C15 | 当前系统达到 private-memory readiness | 所有阶段状态 | N | 不支持 | `private_value_memory_ready=false` |
| C16 | 当前系统达到原 C3 eligibility | 所有阶段状态 | N | 不支持 | `c3_eligible=false` |

## 3. 关键数值证据

### C2.3–C2.5

| 证据 | 数值 | 解释 |
|---|---:|---|
| C2.3 S0 oracle | validation/development 10/10 gates | relation ID 正确时检索几何足够 |
| C2.3 S5/S6 | 唯一失败均为 projected-family probe | 连续 relation 仍携带 family 信息 |
| C2.4 D2 cross-relation candidates | 0 | 单 bucket 结构隔离 |
| C2.4 family macro / worst family | 81.94% / 25.00% | 离散 lexical OOD 未解决 |
| C2.4 locked known coverage | 72.22% | reject guard 效用不足 |
| C2.4c R2 safe coverage | 100% | 合成非独立压力集局部正结果 |
| C2.4c R3 FMAR / safe coverage | 25.00% / 95.83% | 局部选择性弃权效果 |
| C2.5 CLINC R2−R0 | −1.57 pp | pairwise evidence 未外部复现 |
| C2.5 BANKING open R2−R0 | −3.33 pp | 同上 |
| C2.5 BANKING closed R2−R0 | −3.83 pp | 同上 |
| C2.5 CLINC R3 FMAR / safe coverage | 1.22% / 25.83% | 低 FMAR 来自大规模拒绝 |
| C2.5 BANKING R3 FMAR / safe coverage | 0.33% / 17.74% | 同上 |
| C2.5 R3 multi-candidate rate | 0.41% / 0.03% | multi-set 机制基本未发挥 |

### D1–D2.3

| 阶段 | 场景 | 核心零违规 |
|---|---:|---|
| D1 | 25/25 | unauthorized/cross-entity/cross-relation memory access = 0 |
| D2 | 37/37 | unauthorized lookup/decrypt/plaintext release = 0 |
| D2.1 | 40/40 | unauthorized generator invocation/value exposure = 0 |
| D2.2 | 35/35 | cross-process double release/post-restart replay = 0 |
| D2.3 | 51/51 | observability leak/post-restore replay/rollback acceptance = 0 |

这些场景定义不同，不应相加后换算为“188 次攻击下的安全概率”。

## 4. 反主张与审稿风险

| 容易出现的过度表述 | 为什么不成立 | 应替换为 |
|---|---|---|
| “我们构建了可靠语义授权 router” | C2.5 外部失败 | “我们给出其不宜作为授权边界的外部负结果” |
| “Conformal/set-valued routing 检测歧义” | multi-candidate rate 接近 0 | “拒绝主要由 evidence 不足触发” |
| “离散接口消除了泄漏” | 只删除 memory API 中的连续通道 | “实现结构性数据最小化和候选隔离” |
| “HMAC/AES-GCM 是本文算法创新” | 都是标准组件 | “贡献在信任边界与数据流组合” |
| “零 plaintext occurrence 证明内存安全” | 只扫描应用层持久化 | “未在已扫描位置观察到持久化” |
| “D 系列证明 private memory 安全” | 只使用 synthetic canary | “验证 public/synthetic prototype” |
| “状态回滚已解决” | anchor 与主状态同主机 | “检测未同时回滚 anchor 的旧状态” |
| “G1 证明 LLM 不泄密” | 受限 token copy | “验证受控 adapter 与一次性 context” |

## 5. 论文主张的推荐组合

主主张：

> Learned semantic routers can improve interaction, but external OOS evidence
> does not support using their confidence or abstention as authorization.

架构主张：

> A typed authenticated capability contract moves semantic routing outside the
> authorization TCB and enforces authorization-before-lookup-before-decrypt.

系统证据主张：

> In a frozen single-host prototype with public/synthetic values, the resulting
> pipeline preserved scoped, single-use, fail-closed data flow across
> preregistered adversarial, fault, observability, IPC, and lifecycle matrices.

限制主张：

> These results do not establish private-data readiness, deployment security,
> distributed consistency, model confidentiality, or machine unlearning.
