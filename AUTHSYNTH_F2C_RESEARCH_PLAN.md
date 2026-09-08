# AuthSynth F2C：Symbolic Unary Effect-Contract Synthesis

> 当前为预冻结实现草稿，正式评分尚未开始。单元测试通过不代表 benchmark
> 隔离门槛通过；既有模板仍存在内容碰撞，有界循环展开与 omission-atom
> 指标尚待完成。详见 `docs/f2c/PREFREEZE_REVIEW.md` 及后续修复记录。

## 研究问题

F2B 只在分析器不可见实现的前提下刻画一个冻结的有限 query catalog。F2C 将
候选契约改为 guarded symbolic clauses，并让 verifier 通过 SMT 主动生成候选契约
与 ToolSymbolicIR 的差异赋值。Analyzer 只能通过该 assignment 的窄 replay 获得
membership evidence。

```text
candidate symbolic contract
→ bounded SMT equivalence check
→ concrete difference assignment
→ structured sandbox replay
→ examples-only synthesis
→ new candidate
```

## 新增能力

1. Bool、Enum、有界 Int 与有限 record schema；
2. 受限 DNF guard 和 constant/input/state/ITE field terms；
3. immediate/delayed effects 与结构化 state updates；
4. under-approximation、over-approximation、field/state mismatch 检查；
5. 绑定 schema、实现版本和 bounded-domain proof result 的 certificate；
6. UNKNOWN、旧 certificate 失效与新版本独立 re-certification。

## 不在范围内

F2C 不研究双运行信息流、秘密依赖 payload、HyperLTL、真实 MCP、任意 Python/
JavaScript、真实 Agent、private data 或 LLM judge。Z3 是标准求解组件，不是算法
创新。完整性只对预注册 ToolSymbolicIR grammar 与有限 input/state 域成立。

## 正式协议

Analyzer/shared/verifier/replay 先完成测试并冻结；正式 locked namespace 只能在
Analyzer freeze commit 之后物化。Calibration 只能选择预注册 grammar/budget/timeout
候选；development 与 locked 各执行一次。每个 locked case 内允许固定 CEGIS
算法利用 verifier 反例更新该 case 的候选契约；不得利用这些反例修改 Analyzer
源码、grammar、预算或跨 case 的选择规则。这不允许重跑正式 split。
