# ToolSymbolicIR 与 SymbolicEffectContract

## 域

F2C v1 只支持 Bool、有限 Enum、闭区间有界 Int，以及由这些字段组成的有限 input/
state record。禁止任意字符串理论、无限整数、递归、任意对象、真实 I/O 和任意代码。

## Guard

Guard 为受限 DNF。Literal 只允许 Bool/Enum equality/inequality 和 bounded Int 的
`<=`、`>=`。正式 grammar 的 disjunct、每个 conjunction literal 与 total literal
上限在 calibration 后冻结。

## Effect 与 state update

每个 effect clause 明确绑定 guard、kind、phase、sequence、causal parent，以及
tenant/resource/destination/amount 字段。字段 term 只能是 constant、input variable、
state variable 或一层 ITE。State update 以相同 guard/term 语言声明最终字段值。

契约没有 query ID、case ID 或 assignment→trace map。canonical ordering、严格 JSON
与 SHA-256 digest 保证同一 AST 的字节级确定性。
