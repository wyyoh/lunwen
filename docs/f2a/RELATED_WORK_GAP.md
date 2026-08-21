# Related Work Gap：从合规执行到效果完备性

本文件在 2026-08-20 核对原始论文页面后冻结。列出的近期工作多数仍为 arXiv
预印本，因此只能用于界定问题重叠，不能用来声称已获得同行评审认可。

## 直接相邻工作

| 工作 | 已解决 | 本阶段不能重复包装的内容 | 剩余切口 |
|---|---|---|---|
| [ToolGate](https://arxiv.org/abs/2601.04688) | Hoare-style pre/post contract 与受验证状态更新 | 给定 contract 后的执行验证 | contract 是否漏掉真实副作用 |
| [Solver-Aided Policy Compliance](https://arxiv.org/abs/2603.20449) | 将给定 policy 编译为 SMT precondition | solver 正确执行形式策略 | policy 是否漏掉禁止状态或参数角色 |
| [PACT](https://arxiv.org/abs/2605.11039) | argument role 与跨步 provenance contract | provenance-aware runtime checking | provenance inference 与 contract synthesis 仍是瓶颈 |
| [Contract2Tool](https://arxiv.org/abs/2606.07904) | 从 metadata、文档与 trace 推断 precondition/effect | 自动 contract inference | 对遗漏效果的可证发现、反例分类与最大许可 synthesis |
| [ToolGuardian](https://arxiv.org/abs/2607.21835) | 描述、syscall、mock、源码的 progressive characterization 与 ASP policy | 多证据 effect fact 抽取 | effect-completeness 判定、validated CEGAR 和最大许可上界 |
| [Bounded Agents](https://arxiv.org/abs/2608.15888) | scope/budget/delegation/composition enforcement | 给定完整 restriction set 时的组合控制 | restriction/forbidden-combination 集本身是否完整 |

## 最接近的重叠风险

ToolGuardian 已经明确把实现级 latent behavior 纳入 source analysis，把 observed
effects 纳入 mock execution，并用 composition rules 做 ASP 判断。Contract2Tool
也已将多源 contract inference 作为核心任务。因此以下表述不构成新颖性：

```text
“我们从文档、源码和 trace 提取 tool effects。”
“我们把 effect facts 交给一个确定性 policy engine。”
```

F2A 的候选增量必须同时包含：

1. 在有限 transition system 上给出可判定的 `EffectComplete` 定义；
2. 把 abstract counterexample 送入 concrete replay，区分真实与伪反例；
3. 把真实反例分类为 contract、policy、composition 或 drift omission；
4. 对精确有限模型计算最大许可安全 Shield，而不只是追加 deny rule；
5. 将 certificate 绑定实现版本，漂移时失效。

如果 F2B 不能证明上述组合相对 ToolGuardian/Contract2Tool 的实质增量，该路线应
降级为工程集成，不应声称新算法。

## 外部 Agent benchmark 的位置

[InjecAgent](https://arxiv.org/abs/2403.02691) 提供工具集成间接 prompt injection
案例，但 F2A 不使用 prompt attack 作为 ground truth。后续 F4 只能把它用作 Agent
触发层；Shield 的安全性必须先在 oracle-call lane 中独立评价。
