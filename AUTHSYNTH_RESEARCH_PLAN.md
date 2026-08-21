# AuthSynth：Effect-Complete Authorization Synthesis

## 研究问题

现有 Agent 授权器通常回答：给定 policy 或 tool contract 后，一次调用是否合规。
本路线改问：policy/contract 是否覆盖工具实现的全部安全相关效果；若不覆盖，能否
通过可重放反例补全抽象，并合成尽可能少误拒绝的安全 Shield。

核心判别式是：

```text
monitor compliance = true
并不蕴含
concrete forbidden effect = false
```

F1 AuthZRouteBench 保持冻结，继续提供确定性 policy oracle、严格 schema、规范化
hash 和 split discipline。本路线不修改或重新评分 F1 locked test。

## 阶段

```text
F1 deterministic policy oracle
  → F2A ToolEffectIR + exact feasibility
  → F2B CGAR evidence/replay/refinement
  → F2C AuthEffectBench-Core locked audit
  → F3 instrumented MCP-style tools
  → F4 external Agent evaluation
  → F5 bounded formalization and paper
```

本分支只执行 F2A。上一版 capability compiler 与 dual data-authority calculus
分别保留在独立分支，不能作为本阶段实验结果。

## F2A 交付

1. 有限 `ToolEffectIR`；
2. `Traces(T) ⊆ γ(C_T)` 的精确检查器；
3. 有限安全博弈上的最大许可 Shield；
4. 8 个 synthetic base tools、6 类 mutation、150 cases；
5. mechanism-level baselines 与 CGAR feasibility；
6. development/locked 各一次的一次性审计。

## 不可外推边界

- 不使用 LLM judge；
- 不接真实 LLM、Agent、MCP、网络、凭据或 private data；
- 不声称任意 Python/JavaScript 工具效果完备；
- 不把 Gold Contract 当作可部署 baseline；
- 不把标准 model checking/CEGAR 本身包装成算法创新；
- 不因 feasibility 通过而声称达到 CCF B 投稿质量。

真实工具阶段必须使用三态输出：`VERIFIED_COMPLETE`、
`COUNTEREXAMPLE_FOUND`、`UNKNOWN`。观测不足不得返回完整。
