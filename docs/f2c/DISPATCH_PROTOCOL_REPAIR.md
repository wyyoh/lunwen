# F2C 正式调度协议修复与重新冻结

用户在正式 development/locked 尚未启动时授权本次调度修复。起点为
`a3bdb3b9ed244b3e05bb7c576737f165ac44425b`；旧冻结元数据保留，不回写为已修复。
旧入口在 gate 之前执行 locked，且过早物化全部 split，因此旧 ready 标记不足以
支持 gate-before-locked 的新执行要求。本次不把只读发现记作正式失败或已消费运行。

## 不变部分

Analyzer/shared、learner、CEGIS、verifier/evaluator、Shield、metrics、模板、mutation、
grammar/budget/timeout 候选、calibration 选择逻辑、readiness gate 与整个容器环境不变。
`benchmark.py` 只对生成器添加 split 选择参数，原 `_make_case`、全局 serial 和所有
模板/指纹函数保持不变。独立 AST/字节测试检查这个边界。

原 benchmark seed 的 freeze 输入固定为 `f141c77278a8924373d38585ff55036ca7340854`，
不会因重新冻结调度而改 seed。新的实际源码冻结点另由
`artifacts/stage_f2c_dispatch_freeze/freeze_binding.json` 绑定。

## 严格顺序

```text
preflight（新 freeze + 原环境）
→ 只物化 train/calibration/development
→ 原 calibration 逻辑选择并固定参数
→ train/calibration 普通正式比较
→ durable consume development → evaluate exactly once → 保存统计和 gate
  → gate 失败：归档 failed_development；不产生任何 locked case/row
  → gate 通过：再次核验 frozen SHA/环境/工作区
     → durable consume locked → 物化 locked → 全 split collision/information audit
     → evaluate locked exactly once → 归档 passed 或 failed_locked
```

CLI 仍为 `python -m keyed_gram.stage_f2c --config configs/stage_f2c.yaml`，由同一
受信进程调度并保留已物化对象，避免分进程恢复时持久化 hidden AST 或重复物化。
不提供绕过 gate 的单独 locked 命令或“从失败目录恢复”参数。分阶段进度写入 runtime。
任意异常保留 runtime/已完成行；同一路径拒绝再次运行，不自动重试。

每次开发/locked 评价前以 exclusive create + fsync 消费一次资格；在每个 evaluator
调用之外保存已完成统计行，保持原 method-major 顺序与原方法调用。检查点不保存
hidden AST、gold formula、assignment/output table 或完整 capability。

Development 失败时输出的精确 inventory 明确不包含 locked CSV、locked split manifest
和 post-development integrity 文件，不制造空白 locked 结果冒充已测。成功时仍验证
完整 inventory。新 `evaluation_rows.json` 只含原 evaluator 的统计、类别与 digest，
用于独立拆分已有指标，不改变指标计算定义。

本次先使用 mock case 测试失败停止、资格重复拒绝及调用顺序；这些不是正式案例。
重新完成普通回归，七项历史兼容例外仍必须精确匹配。新源码与新验证证据先提交并推送，
再从 Git blob 建立新冻结清单。只有新冻结通过才开始原请求授权的唯一正式运行。

## 本次普通验证证据

`artifacts/stage_f2c_protocol_repair/` 记录 162 项 F2C 定向测试通过；全仓实际为
732 passed、7 项精确匹配的历史兼容失败、0 unexpected failures。模板隔离、
960 个 concrete/symbolic loop assignments、Ruff 和 diff check 均通过。
环境仍为原镜像、CPU-only、network none；没有重建或更新依赖。

首轮临时副本将三份公开 fixture 放在错误目录，产生额外五项文件缺失失败；该失败摘要
保留为 `initial_fixture_staging_failure.json`。第二轮只恢复原文件路径并核验历史 SHA，
未改变 fixture 字节、历史测试或七项 allowlist。两轮均为普通预冻结回归，不是正式
development/locked；不能用第二轮覆盖首轮历史记录。
