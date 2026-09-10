# F2C 预冻结修复：可定位 omission atoms

本记录是非正式源码与指标回归，不是 Analyzer freeze 或正式实验结果。
上一个提交为 `a688264c468aa3baa6f0335f380fce76b02d5944`；不修改此前
train 预实验、F1/F2A/F2B 数据和报告，也不重新计算其历史指标。

## 修复了什么

旧 `omission_atoms` 由 mutation category 映射为 patch 类型集合。同一 case
若缺少两个不同 SEND 槽位，只会计作一个 `effect_patch`；类型预测正确但
位置错误也无法扣分。旧 `omission_atom_discovery_recall` 实际复用了类型召回。

现在的 `PatchAtom` 是四元组：

```text
(patch_type, target_kind, target_id, component)
```

| 类型 | 定位对象 | 组件 |
| --- | --- | --- |
| effect_patch | 效果签名摘要 | presence |
| guard_patch | 效果签名摘要 | activation |
| field_binding_patch | 效果签名摘要 | tenant/resource/destination/amount/joint_binding |
| state_update_patch | 具体状态字段 | final_value |
| version_invalidation | tool ID | version |
| unsupported_region | tool ID | 受限 reason code |

效果签名包含 slot、sequence、phase、kind 和 parent，不包含载荷。相同类型但
不同签名、字段或状态位置分别计数。每个 case 内去重，跨 case 不合并计数。
各字段边际集合相同、联合绑定关系不同的情形由 `joint_binding` 单独定位；
比较使用 canonical JSON，避免 Python 将 `True` 与 `1` 当作相同值。

这是一种**位置/组件级原子化**，不是 guard literal 级的最小修复证明。
同一槽位的多个错误 guard 区域仍归为一个 activation atom；签名变化可能
表现为旧槽位多余与新槽位缺失。论文必须保留这一粒度限制。

## 信息边界与真值

Analyzer 的 `diagnose_replay` 只比较候选契约和已返回的单次 replay。
Evaluator 的 `expected_omission_atoms` 独立比较初始公开契约（包括 static
hypotheses）与完整有界域上的具体行为；不从 mutation category 生成标签，
也不调用 Analyzer 的诊断函数。修改标签或 monkeypatch Analyzer 诊断不改变真值。

这里的完整域枚举仅用于 **Evaluator 评价标签**，没有交给 Analyzer，也没有
替代 Verifier 的 SMT 等价检查或完整性证书。没有新增 query catalog 或 locked
assignment/output 表。证据不足的 clean control 没有被凭空标注为效果遗漏；
`unsupported_region` 属于独立证据缺口，不进入 effect omission 分母。

## 发现与中间修补分开

- `discovered_atoms`：每次 replay 相对于**初始声明契约**所揭示的原始遗漏。
- `patch_records`：当前迭代 candidate 相对于 replay 的待修补位置。

Learner 中间候选自己引入的错误可能进入 patch records，但不会成为“发现了
新的原始遗漏”。无法定位的旧式记录仍以独立摘要计入预测集合，无法命中真值，
不能通过丢弃它们提高 precision。定位成功只代表诊断正确，**不代表后续合成的
公式已经正确修复**；实际契约正确性仍须由独立 Verifier 判断。

## 指标语义

对每个 case 分别求集合交集后累计 TP、预测量和真值量：

```text
patch_type_precision/recall    = 类型级匹配
patch_atom_precision/recall    = 精确位置与组件匹配
omission_atom_discovery_recall = 原始遗漏 atoms 中已被 replay 揭示的比例
exact_patch_set_rate           = 完整 atom 集合精确一致的 case 比例
exact_patch_type_set_rate      = 原有类型集合一致率的明确命名
```

重复 evidence 不重复累计同一个 atom。空分母沿用实现的约定值，并输出对应计数，
不能把没有遗漏的 case 当作发现效果证据。train runner 新输出标为 schema 2、
`patch_metric_semantics=location_component_atoms_v1`；旧 schema 1 产物不回写。

## 验证

在已绑定的 Torch 2.12 CPU 镜像中运行普通单元测试；无网络、非 root、只读源码，
4 CPU 上限、4 GiB 内存、移除 capabilities。最终结果：

- F2C 定向 pytest：**89 passed**。
- F2C 源码/测试及相关 runner 的 Ruff：通过。
- 10 个 train smoke：通过；没有 development/locked 评分。
- 新增原子回归覆盖同类多位置、字段错位、联合绑定、独立 oracle、重复计数、
  无定位记录计入 FP，以及真实 CEGIS 的 replay 驱动诊断。

早先 87 项通过的中间验证也原样保留。最终源码与容器返回的 SHA 清单逐项核对。
证据位于 `artifacts/stage_f2c_preflight/omission_atoms_20260908/`。
`validation*.json` 中 `container_environment_verified=false` 表示通用验证脚本不
自行认证容器；同目录 Docker inspection 记录本轮实际隔离参数，不回写该字段。

## 仍不能正式运行

此前发现的 52 组跨 split 内容碰撞尚未消除；本轮没有改模板生成逻辑。
显式有界循环的真实执行/SMT 语义仍未实现，继续返回 UNKNOWN。全仓测试、正式
upstream 冻结绑定和 analyzer freeze 仍待完成。本轮只是局部修复，不能将其
标成 F2C 通过，也没有利用正式 development/locked 调整算法或门槛。

```text
patch_atom_localization_implemented = true
omission_ground_truth_independent_of_mutation_labels = true
f2c_formal_results_available = false
analyzer_frozen = false
ready_for_formal_f2c_audit = false
ready_for_stage_f2d_relational_contracts = false
f2b_frozen_assets_modified = false
f2b_locked_test_rerun = false
private_data_used = false
```
