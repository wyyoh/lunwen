# F2C 预冻结修复报告：B1/B2 通过，B3 阻止冻结

本轮基于实际核验的 `053326f64e77ebf235d14ecfe18f250ccf2dacb9`，分支仍为
`agent/stage-f2c-symbolic-contracts`。开始时工作区干净。本报告只记录普通预冻结验证，
不是 F2C 正式结果、Analyzer freeze、development 或 locked audit。

## 结论

| 门槛 | 实测结果 | 状态 |
| --- | --- | --- |
| B1 模板隔离 | 原 52 组；新结构 0、新语义 0、旧内容审计 0 | 通过 |
| B2 有界循环 | 24 个程序、960 个完整小域赋值、具体/符号差异 0 | 通过 |
| F2C 普通定向测试 | 114 passed，0 failed，0 skipped，61.306 秒 | 通过 |
| B3 全仓普通测试 | 684 passed，7 failed，0 skipped，约 268.40 秒 | 失败 |
| F2C Ruff / git diff --check | 最终均通过 | 通过 |
| 既有 CPU 容器断网复验 | import、CPU kernel、Z3、项目 import 通过 | 通过 |
| B4 Analyzer/source/environment freeze | 未创建，仍为 PENDING_ANALYZER_FREEZE | 禁止进入 |

没有降低任何正式阈值，没有运行正式 development/locked，没有修改历史正式产物。
本轮提交只能是修复和失败证据提交，不能命名或解释为代码冻结提交。

## B1：结构与语义隔离

独立 `template_isolation.py` 同时检查 alpha-normalized 结构及完整小域行为摘要。
移除名称、case/tool 身份与常量字面差异；交换律归一化；对恒真/恒假 guard、无作用更新、
不可达分支及未使用 schema 字段做失败检查。旧内容指纹门槛继续执行，没有删除或放宽。
9 项隔离回归覆盖变量重命名、常量替换、交换律、dead/no-op 以及 AST 不同但语义相同。

四个 split 各保留 10 个模板定义。train 直接绑定和单状态、calibration 状态/输入合取及
条件绑定、development 阈值/DNF 与两轮状态递推、locked 定义三路 guard/三轮循环与分叉
回执，具有不同的状态、更新、绑定及事件因果结构。详细规则见
`TEMPLATE_ISOLATION_PROTOCOL.md`。不是把工具或变量换名，也不是加入 `AND true`。

```text
cross_split_structural_collision_count = 0
cross_split_semantic_collision_count = 0
cross_split_duplicate_fingerprint_groups = 0
dead_guard_template_count = 0
tautological_separation_count = 0
no_op_branch_count = 0
unreachable_branch_count = 0
```

审计对象是无具体 case 身份的生成器模板定义；未物化正式 held-out hidden instances，
未运行 Analyzer 评分。完整小域枚举仅服务于隔离和差分测试，不用于替代 SMT 完备性验证。
同一 split 内的重复不是跨 split 碰撞；零碰撞仅是本规范化/有限域定义下的结果，
不声称任意程序语义独立性。160 case 的正式设计规模保留，本轮未构建正式 benchmark。

## B2：真实具体执行与 SSA 展开

新增 `BoundedLoop`、真实 body 和 `LoopUpdate`；每轮重算 guard、原子写回状态，退出后
不复活，循环后分支读取最终状态。固定数学整数范围，越界拒绝，不隐式 wrap。
支持 Bool/Enum/条件赋值及有界整数增减。静态次数上限 8；不是启发式展开两次。

SMT 使用 `active_i` 与 SSA 状态 recurrence，逐轮绑定 effect 字段、顺序、phase 和
同轮 causal parent。当前异步语义是每轮确定性排空有界队列，最大因果深度 2，
不是任意异步调度器。结果包含最终状态、结构化 diff、轮次与退出原因。

测试覆盖初始 false、一次/多次、BOUND_REACHED、状态递推、逐轮字段不同、delayed
effect、循环后分支、范围溢出、超限和无界 UNKNOWN。24 个小程序穷举 960 个赋值，
完整 concrete result 与 SMT model evaluation 比较为 0 差异。另有真实有界循环通过
SMT 等价检查的正例及遗漏 effect 的反例，未将所有循环改成 UNKNOWN 规避问题。
仅有旧 `bounded_loop_max` 元数据但没有 body 的程序仍不能获 complete。

这不证明一般循环终止、任意工具完备性、机械化安全定理或 F2D 双运行性质。

## B3：全仓失败的准确原因

最终执行的是完整 `pytest -q` 普通回归，不是正式一次性实验入口。失败没有被排除：

1. **5 项 D2.1 protocol 测试**：冻结 `configs/stage_d21.yaml` 要求
   Transformers `5.14.1`，既有已验证 CPU 镜像实际安装 `5.16.1`。
   `stage_d21_protocol.py` 精确核验依赖版本，因此拒绝。Torch 两者均为 `2.12.0+cpu`。
   没有为通过测试修改冻结配置或替换环境依赖。
2. **1 项 D2.3 runtime 测试**：`test_full_preregistered_matrix_rehearsal` 的 generator
   service socket 在原 60 秒时限内未就绪。该问题在两次普通回归均出现，原因尚未充分定位；
   不能把它宣称为偶发并忽略，也未延长时限或改变冻结 D2.2/D2.3 实现。
3. **1 项 F2A manifest 测试**：F1 原 source manifest 中 `.gitattributes` 的 hash 为
   `ce6520eee51c5bae5d46af220702896bf92421ef5a4f935558c670b31634b388`，
   当前基线该文件 hash 为
   `e6a3c69a0668d6d3d4c215cc72c0dedee8d16eed2b672eccb48d94984a75f88e`。
   F1 历史提交中的原文件仍匹配旧 hash；后续 F2B 历史提交已经增加属性规则。
   这是本轮之前已有的快照差异，不能回写 F1 清单或回退属性文件使校验表面通过。

第一轮临时副本没有携带三份本地已存在但 Git 忽略的公开 C2.4b v2.1 fixture，造成额外
5 项 C2.4c 文件缺失失败。仅补齐原字节后完整串行重验，最终这些测试通过。
第一轮 677 passed / 12 failed 与最终 684 passed / 7 failed 均保留，没有删除失败记录。

F2C 定向测试和全仓测试后，验证包装脚本的 Ruff ISC004 已用字符串外加括号修复；
在同一镜像中比较修复前后 AST 完全相等，并重新通过整个 F2C Ruff 范围。
实际被测 Analyzer/测试源码 hash 全部与提交内容一致，只有该纯格式包装脚本 hash 不同，
已单独登记；没有在测试之后修改安全算法。

## 环境与历史绑定证据

实际重验：glibc `2.36`、Python `3.12.14`、Torch `2.12.0+cpu`、NumPy `2.5.2`、
Z3 `4.15.3`。CUDA unavailable，CPU tensor kernel 与项目 import 成功。
使用现有镜像、非 root、只读源码/根文件系统、`--network none`、4 CPU/8 GiB，
移除 capabilities，不使用 privileged、宿主根目录或 Docker socket 挂载。

```text
image_id = sha256:b89ea8a6918ad5a06f04119cbe745178be48801c0c8e192faf28b74e39d14d4b
base_digest = sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
repo_digest_available = true
```

镜像 ID/RepoDigest 来自实际 Docker inspect；8 个环境文件的 SHA 与之前环境 manifest
逐项一致。F1/F2A/F2B 三阶段 source/artifact/upstream 共 163 条记录重新核验，全部相对
本轮 base 未修改；162 条与历史清单匹配，1 条为上述既有 `.gitattributes` 差异。
三个正式 artifact inventory 均未改变。

`artifacts/stage_f2c_prefreeze/` 保存真实 XML 派生统计、失败测试名称、原始日志 SHA、
模板/循环审计、环境复验与当前/历史 hash 对照。原始日志和 XML 保存在执行机及会话
临时证据目录；不把 traceback 中的合成 canary 正文复制进 Git。短暂 SSH 中断后已取回
日志，网络不是当前禁止冻结的主要原因。

这些文件明确是 **预冻结失败证据**，不是从 freeze commit 生成的 frozen manifest。
正式 frozen_upstream/frozen_analyzer/frozen_environment 条目均未建立，不能报告有效。

## 后续需要的明确决策

必须先解决历史版本精确校验、历史 source snapshot 校验口径和 D2.3 服务启动超时，
再重新完成全仓普通回归。需要变更测试协议或建立版本隔离回归环境时应单独确认；
当前不得修改冻结上游、跳过失败测试、降低门槛或把环境 smoke 通过等同全仓通过。

```text
prefreeze_repair_status = FAILED
benchmark_isolation_status = passed
bounded_loop_semantics_status = passed
full_repository_tests_status = failed
analyzer_freeze_status = NOT_CREATED
ready_for_analyzer_freeze = false
ready_for_formal_development = false
formal_development_started = false
formal_locked_started = false
f1_locked_test_scored = false
f2a_locked_test_rerun = false
f2b_locked_test_rerun = false
private_data_used = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```
