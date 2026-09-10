# F2C 预冻结审查：尚不可启动正式实验

本记录属于源码草稿审查，不是 F2C 正式 development/locked 结果，也不是
Analyzer freeze。不修改或取代 F1、F2A、F2B 报告与产物。

## 基线复核

重新查询 GitHub，`agent/stage-f2b-blind-cgar` 的远端 HEAD 为
`1c01a24fe8cdc123e405591a0ced0a5ee50aa5d6`，与本地 F2C 草稿基线相同。
环境分支远端 HEAD 为 `8056b1f2e204545fbb0e744100fe3a0987d2e692`。
本轮未将环境记录当作当前实验机已重新验证的证据。

复算 F1/F2A/F2B source、artifact 和 upstream 清单共 163 条文件记录：
162 条与当前文件一致；F1 source 清单中的 `.gitattributes` 在当前 HEAD
存在已提交的历史变化。读取 F1 最终提交 `ed0864b6931b779ce7eb4acb060f5662dea95bd6`
中的该文件，其 SHA-256 与 F1 清单一致：
`ce6520eee51c5bae5d46af220702896bf92421ef5a4f935558c670b31634b388`。
F2B 提交 `e31459d` 新增了 `artifacts/stage_f2b/** -text whitespace=cr-at-eol`。
这不是本轮漂移，不能通过回写 F1 manifest 或恢复旧 `.gitattributes` 来处理。
正式 F2C 的 upstream 绑定仍需分别记录历史快照与实际 base 文件。

## 发现 1：名称隔离没有保证模板隔离

现有草稿的 `_schema()` 不接收 split，`_mutation_transition(category)`
也不接收 split；四个域和 split 使用相同结构。原 `collision_audit`
主要检查带 split/serial 前缀的 family 标签，因此会漏报这些重复。

现已加入实际 schema、guard、field binding、实现 AST、契约、threshold 和
constant 内容指纹检查。只生成 train/calibration 非正式 fixture，未生成
development/locked 实例；80 个 fixture 的内容审计返回 `failed`。

| 碰撞维度 | 跨 split 重复指纹组数 |
| --- | ---: |
| 实现 AST 内容 | 9 |
| 实现 AST 保守结构 | 9 |
| 契约内容 | 9 |
| Guard 模板结构 | 7 |
| 字段绑定模板结构 | 7 |
| 输入 schema 结构 | 1 |
| 状态 schema 结构 | 1 |
| Threshold 模板 | 1 |
| Constant 模板 | 8 |
| 合计 | 52 |

同一重复结构可出现在多个审计维度中，因此 52 不是独立样例数或攻击次数。
结构归一化是保守检测，不是任意 AST 的语义等价证明。正式 benchmark 必须在
freeze 前重新设计实质不同的模板，不能靠改 family 标签、删除失败 family
或放宽碰撞门槛通过。本轮没有修改 mutation/template 生成逻辑。

## 发现 2：有界循环上限尚不对应执行语义

`bounded_loop_max` 原本只是元数据；`HiddenSymbolicImplementation.execute()`
和 SMT 编码均未展开循环或递推状态。原单元测试将上限设为 4，然后仅验证
无循环契约等价，不能证明有界循环完整性。

已将正数循环上限改为 `UNKNOWN / bounded_loop_semantics_not_implemented`，
并更正回归断言。该修改只是防止虚假完整性，**不是**实现了 F2C 所要求的有界
循环语义；相关求解器测试仍须在恢复后的容器中执行。

## 其他待完成审查

- Verifier 的 state-update SMT 编码以嵌套 ITE 表示后写覆盖，而具体执行拒绝
  相互冲突的更新，需补充符号冲突检查与具体语义一致性回归。
- Learner 的 Optimize 调用没有应用配置的 solver timeout，且加权索引和
  不足以保证唯一 lexicographic tie-break，冻结前需要修复并测试。
- `Declared-Contract Monitor` 和 `Passive Symbolic Learner` 在当前 evaluator
  中被主动标记为 complete；这不能冒充这些机制自行作出的完整性声明。
- Guard 的真负例计数遍历正例并集，永远无法观测双方均不激活的槽位；
  omission 标签目前也主要是 patch 类型集合，尚非可定位的 omission atoms。

这些是源码审查发现，尚未由新的正式测量量化；不能填充为 0 或宣称通过。

## 验证与设备阻塞

- 跳板机 SSH 可登录，现有 Python 为 3.12.1。
- 跳板机到实验机 `192.168.137.231:22` 两次 SSH 均超时；ping 两次返回
  `TTL 传输中过期`。未修改路由、防火墙、SSH 服务或宿主 glibc。
- 仅在跳板机已有 Python/pytest 上执行 16 个不依赖 Torch/Z3 的类型、公式、
  契约与碰撞单元测试，全部通过；没有安装新依赖。
- 非正式内容审计失败并保存为 `artifacts/stage_f2c_preflight/content_isolation_audit.json`。
- 本地 AST 语法检查和新增文本的 whitespace 检查通过。
- F2C 全部定向测试、全仓测试、Ruff、容器无网络复验尚未完成。
- 没有 Analyzer freeze、正式 benchmark materialization、calibration 选择、
  development 或 locked 评分；没有任何上游正式审计重跑。

## 状态

```text
f2c_prefreeze_review_status = failed
analyzer_frozen = false
formal_development_started = false
formal_locked_started = false
ready_for_formal_f2c_audit = false
ready_for_stage_f2d_relational_contracts = false
f2b_frozen_assets_modified = false
f2b_locked_test_rerun = false
private_data_used = false
```

先恢复实验机连接并修复上述草稿问题，再运行完整测试；不能直接冻结并消耗
唯一正式审计次数。本轮没有代码冻结提交、结果提交或推送，工作区仍包含
待审查的 F2C 草稿。
