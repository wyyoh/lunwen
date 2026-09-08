# F2C 无网络容器 train 预实验

## 结论和范围

本次在目标机的 WSL2 / Docker 环境实际执行了 **40 个 train case × 8 种方法**。
这是预冻结诊断，不是正式 calibration、development 或 locked 审计。
Analyzer、grammar、budget 和 evaluator 的算法源码没有因本次结果而修改。

四个域各 10 个 case；每个 case 的完整有界域为 3,072 个 assignments，合计
122,880 个 case-assignment 对。Analyzer 每 case 的 replay budget 固定为 16，
solver timeout 为 3,000 ms；这些沿用 smoke 配置，未经本次 calibration 选择。
完整域枚举只用于 evaluator 指标，Analyzer 的候选检查仍使用 SMT。

## 环境和代码绑定

- 被测 F2C 草稿：`607fd6d3cb7ab288f7afac71096020f4b60be77a`。
- 环境脚本来源：独立环境提交 `8056b1f2e204545fbb0e744100fe3a0987d2e692`。
- 镜像：`sha256:b89ea8a6918ad5a06f04119cbe745178be48801c0c8e192faf28b74e39d14d4b`。
- Python 3.12.14，glibc 2.36，Torch 2.12.0+cpu，NumPy 2.5.2，Z3 4.15.3。
- `network=none`，非 root `researcher`，只读根文件系统与源码挂载，4 CPU / 4 GiB，
  线程默认 1；drop all capabilities、no-new-privileges。输出为独立挂载目录。
- 39 个被测源码/测试 SHA-256 与本地文件一致，pilot runner 单独 SHA-256 绑定。

环境分支只作为只读验证脚本来源，尚未合并为正式 F2C freeze 输入。
普通 validator 的 `container_environment_verified=false` 是其通用默认值，
并非本次容器检测结果；本次环境证据是原样保存的 `environment.json` 和
`runtime_inspection.json`。没有改写 validator 原始输出。

## train 结果

| 方法 | UER | 未 replay precision | FTAR | Safe Utility | replay 数 | 收敛率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenyAll | 0 | 1.0000* | 0 | 0 | 0 | 0 |
| Declared-Contract Monitor | 0.8063 | 0.8776 | 0.8214 | 0.8696 | 0 | 0 |
| Random Replay Table | 0 | 1.0000* | 0 | 0.0051 | 640 | 0 |
| Coverage-Guided Replay Table | 0 | 1.0000* | 0 | 0.0013 | 160 | 0 |
| Passive Symbolic Learner | 0.9499 | 0.9046 | 0.2857 | 0.9674 | 160 | 0 |
| CEGIS-No-Minimization | 1.0000 | 1.0000 | 0 | 0.9348 | 140 | 0.95 |
| AuthSynth-Symbolic | 1.0000 | 1.0000 | 0 | 0.9348 | 136 | 0.95 |
| Gold Symbolic Contract（上界） | 1.0000 | 1.0000 | 0 | 1.0000 | 0 | 1.00 |

\* 未预测任何未 replay effect 时，当前 evaluator 将 precision 空分母记为 1；
这不是这些方法有泛化能力。没有作出 complete 声明的方法，其 false-complete
计数为 0 也不提供完整性证据。

AuthSynth-Symbolic：38/40 收敛、2/40 UNKNOWN、false-complete 0、旧证书复用 0。
136 次 replay 相对于 122,880 assignments 的减少率为 99.8893%。
本次平均 analysis latency 约 41.78 ms/case，仅为诊断性测量，不作为论文性能结论。

当前 patch-type precision 为 **0.5490**、recall 为 1.0、exact patch-set rate 为
**0.15**。没有将它包装成 omission-atom-level 成绩：原 evaluator 的
`omission_atom_discovery_recall` 实际复用 patch-type 计数，pilot 输出显式排除该字段。
当前 unary evaluator 的 MPR 和 Safe Utility 也使用相同计数，不作为两份独立证据。

## 验证与失败记录

- 容器 Torch native CPU 运算、Z3、项目导入通过，CUDA 不可用。
- F2C 定向单元测试：71 passed；Ruff 通过；10-case train smoke 通过。
- 内容隔离审计仍为 **failed：52 组跨 split 重复指纹**，没有降低门槛。
- 全仓测试未运行；本次只传输 F2C 所需最小源码，未传输或打开历史 private cache。
- 首次普通容器验证完成后，WSL `/tmp` 副本及结果不再存在。随后使用新目录和
  Windows 持久输出挂载复跑普通验证；没有重跑任何正式 audit。
- 新 pilot runner 第一次被 Ruff 拦截（NTFS 可执行位导致 shebang 要求、import
  排版）；当时 train 尚未启动。修复 runner 后第一次实际 40-case pilot 完成。
  原 Ruff 失败输出保留为 `pilot_ruff.txt`，修复后为 `pilot_ruff_v2.txt`。
- preflight 容器 exit code 1 来自最后的内容隔离负结果，不能把整个 preflight
  宣称 passed；pilot 容器 exit code 0 只表示执行完成。

## 当前门槛

```text
container_train_pilot_execution_status = completed
f2c_unit_validation_status = passed
networkless_runtime_verified = true
cross_split_collision_count = 52
analyzer_frozen = false
formal_calibration_started = false
formal_development_started = false
formal_locked_started = false
ready_for_formal_f2c_audit = false
ready_for_stage_f2d_relational_contracts = false
f2b_frozen_assets_modified = false
f2b_locked_test_rerun = false
private_data_used = false
```

仍须解决模板隔离、有界循环实际语义、可定位 omission atoms、全仓验证与正式
freeze manifests。train 的完美 effect 指标不构成跨 family 泛化或正式 F2C 通过。

全部原始输出见 `artifacts/stage_f2c_preflight/container_train_20260908/`。
该目录是新的非正式证据目录，没有覆盖上次预冻结报告或正式历史产物。
`validation.stdout.gz` 无损保存原始 stdout（含末尾额外空行），已逐字节核对
解压结果与回收归档，未修剪原始输出以满足文本 whitespace 检查。

## 普通预实验复现

在同一 CPU 镜像内，以仓库只读挂载、网络关闭、独立可写输出目录运行：

```bash
PYTHONPATH=src python scripts/run_f2c_train_pilot.py --output /workspace/runtime/new-train-pilot
```

输出目录必须不存在。该脚本不提供 development/locked 选项，不调用正式入口。
