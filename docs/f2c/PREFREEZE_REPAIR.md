# F2C 预冻结修复进度（非正式实验）

## 本轮完成

- Verifier 检查工具身份、版本、schema；状态更新冲突与越域先于等价检查拒绝。
- 保持 Bool/Int 类型分离，不再通过隐式强制转换掩盖字段差异。
- 无版本绑定的等价响应不能发证；失败 replay 计数但不进入 examples。
- 版本漂移清空旧证据，但不返还已使用的 replay 预算，不伪造不存在的旧证书 ID。
- Learner 的候选生成与 Optimize 共享 fit deadline；求解器接收剩余 timeout。
- cube 优化增加总 literal 约束，用唯一二进制权重落实确定性 tie-break。
  当前优化仅针对启发式生成的候选池，不声称全语法全局最优。
- Declared/Passive 基线没有经过验证，不再人为标记 complete 或 safe。
- Guard 指标使用固定签名全集，从而计入双方均未激活的真负例。
- 相对路径校验在 Windows/Linux 上一致拒绝盘符、UNC、绝对路径与目录穿越。

## 验证证据

`artifacts/stage_f2c_preflight/repair_validation.json` 由验证脚本自动生成，包含
各命令退出码、原始输出和 39 个被测源码/测试文件的实际 SHA-256。

| 验证 | 结果 |
| --- | --- |
| F2C 定向单元测试 | 71 passed |
| F2C 新代码、测试与验证脚本 Ruff | passed |
| CEGIS train smoke | 10 cases，passed |
| development / locked | 均未开始 |

第一轮测试为 58 passed、1 failed：暴露 Windows 对 `/absolute` 的路径解释差异。
修复并增加跨平台回归后复跑普通单元测试；这不是重新运行正式 audit。

本轮运行位置是跳板机独立临时虚拟环境，Windows / Python 3.12.1，
Z3 4.15.3.0、pytest 9.1.1、Ruff 0.16.5、PyYAML 6.0.3。
没有安装 Torch、修改系统 Python 或 glibc，也没有用该环境替代冻结 Linux
容器。测试和 smoke 仅使用人工及 train/calibration fixture，不物化正式
development/locked 实例，不读取其结果。Smoke 的 passed 只表示该小范围安全
检查通过，不是 unseen-generalization 或完整 F2C readiness 结论。

可重复的非正式验证命令：

```bash
PYTHONPATH=src python scripts/validate_f2c_draft.py --output validation-new.json
```

输出文件必须不存在；该脚本只调用列举的 F2C 单元测试、Ruff 和显式 `--smoke`。

## 仍未完成

1. **模板隔离**：生成器未改。先前在 train/calibration fixture 中发现的
   52 组跨 split 重复指纹仍然是正式审计阻塞，不能改名或放宽门槛消除。
2. **有界循环**：尚无逐轮状态/效果展开；正数 loop 上限继续返回 UNKNOWN。
3. **Omission atoms**：当前标签仍主要是 patch 类型集合，需要可定位的
   effect/guard/field/state omission atoms，不能将类型召回冒充原子召回。
4. **运行环境及全仓测试**：跳板到实验机 `192.168.137.231:22` 再次超时。
   无法复验其 Docker 镜像，也未运行全仓测试或 Linux network-none 验证。
5. **正式协议**：upstream/analyzer 绑定清单尚未完成，配置中的
   `analyzer_freeze_commit` 仍是 `PENDING_ANALYZER_FREEZE`；不存在正式冻结。

先前 `review_sha256_manifest.json` 是上次非正式审查时的源码快照记录。
本轮保留它和原始失败审计，不把它回写成当前源码 manifest；本轮被测源码
由新的 `repair_validation.json` 绑定。两者均不是冻结实验 source manifest。

```text
f2c_unit_validation_status = passed
f2c_formal_evaluation_status = not_started
analyzer_frozen = false
ready_for_formal_f2c_audit = false
ready_for_stage_f2d_relational_contracts = false
f2b_frozen_assets_modified = false
f2b_locked_test_rerun = false
private_data_used = false
```

本轮代码保存属于进行中的预冻结草稿，不是代码冻结提交或正式结果提交。
