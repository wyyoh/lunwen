# F2A Feasibility Protocol

## 规模与 split

```text
8 base tools × 6 mutation categories × 3 instances = 144
clean controls = 6
total = 150
```

四个 split 为 train、calibration、development、locked_test。以下单位严格隔离：

- base tool family；
- mutation family；
- workflow template；
- policy template；
- hidden-effect combination；
- implementation version lineage。

同一个高层 mutation category 可以跨 split，但具体 mutation family 和 tool lineage
不能跨 split。

## 一次性顺序

```text
code/config freeze commit
→ prepare
→ train/calibration mechanism sanity
→ development exactly once
→ AuthEffect locked_test exactly once
```

F1 locked test 不被打开或评分。development/locked 后不得修改 generator、
transition、policy、threshold 或 method。

开发期单元测试与 Smoke 只运行 `train/calibration` fixture。正式 benchmark 的
精确 case 标识和 effect 组合由代码冻结 commit 确定性派生；因此正式 locked
case bytes 在 analyzer freeze 前不会被测试评分。该 seed 不是秘密，作用是把
benchmark 实例与不可修改的 analyzer commit 绑定，而不是制造安全性。

## 方法解释

`Solver-Policy`、`ToolGate-Declared`、`ToolGuardian-Style` 是机制级 simulator，
不是外部论文代码复现。它们分别隔离：给定策略检查、给定 contract 执行验证和
多证据 effect characterization。`Gold-Contract` 只表示有限 transition system
的 oracle 上界。

## 指标

- CER：contract 覆盖的 concrete security effects 比例；
- COR：声明但 concrete input domain 不可能产生的 effects 比例；
- FTAR：被允许且最终进入 forbidden state 的 trace 比例；
- PODR：已注入 policy/composition omission 的发现率；
- CEP：abstract counterexamples 中可具体 replay 的比例；
- Safe Utility：安全 benign workflows 的正确允许率；
- MPR：相对 exact winning region 保留的安全 actions 比例；
- Drift Recall：version lineage drift 的失效发现率。

## 继续门槛

AuthSynth-CGAR 必须在 development 与 locked 同时满足：

```text
CER >= 0.90
FTAR = 0
PODR >= 0.80
CEP >= 0.75
Safe Utility >= 0.85
MPR >= 0.90
Drift Recall = 1.0
distinguishing categories >= 4
```

此外，exact Shield safety 与 exact maximal permissiveness 必须通过，split
collision 必须为 0。门槛不得事后降低。

## 安全与数据边界

- 全部状态、tenant、资源、destination 均为 synthetic identifiers；
- 不保存源码片段、自然语言 case、真实凭据、token 或 private value；
- ground truth 不使用 LLM judge；
- 没有真实工具执行，因此 `real_tool_completeness_validated=false`；
- 对真实程序证据不足时必须返回 `UNKNOWN`。
