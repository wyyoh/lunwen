# F2C bounded-loop 单运行语义

新增 `BoundedLoop(guard, body, max_iterations)`，与旧的 `bounded_loop_max` 元数据不同。
仅元数据而没有 body 仍返回 UNKNOWN。配置上限为 8；None、超界、未支持的终止语义
不得获得 complete。没有 recursion、一般 while 或启发式“两次 edge 展开”。

## Concrete

执行顺序是 before transitions → loop → after transitions。一个 transition group
中的 guard、效果参数和更新 RHS 都读取同一前状态；冲突更新拒绝。每轮先重算 loop guard；
false 后退出，不允许后续轮次重新激活。body 完成后一起写入下一轮状态。
`LoopUpdate` 支持数学整数的固定增减及 Bool/Enum/条件赋值；每一步检查 schema 范围，
越界产生 UNKNOWN/INVALID，不 wrap、不饱和截断，也不缩小输入域。

事件使用稳定的阶段/轮次 slot 和严格递增 sequence。为条件未执行事件保留序号空洞；
顺序以 sequence 为准，不能按 slot 的字典序排列。每轮排空其有界延迟事件队列，再进入
下一轮；父事件必须在同轮执行且位于子事件之前，因果深度最多 2。当前不是任意调度器。

执行结果包括 iteration_count、GUARD_FALSE/BOUND_REACHED、final_state、有序事件和
结构化 state diff。N=0 的有限 for 立即 BOUND_REACHED。BOUND_REACHED 表示语义规定的
静态循环次数已耗尽，并不证明一个原始无界 while 会终止。如果调用者要求在 bound 内
证明 guard 变 false，`require_guard_false_at_bound` 会检查，否则返回 UNKNOWN。

## SMT / SSA

令 `active_0=true`。每轮用 `execute_i=active_i ∧ G(s_i,x)`，效果出现条件为
`execute_i ∧ body_guard`，再用条件更新形成 `s_(i+1)`。
`active_(i+1)=execute_i`，从而已经退出的路径不复活。事件字段读取该轮的 SSA 状态；
after transitions 读取最后一轮写回状态。

Verifier 对越界/冲突/无父事件的可达状态先做 SMT 检查；只要存在这类状态，不能给
complete。随后检查双方状态更新及带全局顺序、phase、parent 的效果签名和字段差异。
既检查 missing，也检查 spurious 和 field mismatch，不仅验证观察过的 replay。

## 差分门槛

普通单元测试穷举 24 个小程序的 960 个 input/state assignments，逐项比较完整执行结果：
事件数、顺序、kind、字段、phase、parent、final state、退出原因和轮次。
还测试前置步骤、循环后分支、初始 false、执行一次/多次、静态 bound、递减、溢出和
UNKNOWN，以及一个真实循环可经 SMT 得到等价而遗漏效果被拒的正负例。

这些是单运行语义实现与有限小域差分测试，不是一般程序完备性、机械化证明或 F2D
双运行信息流结论；未运行正式 development/locked。
