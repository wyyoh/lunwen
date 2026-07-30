# F2A 定理义务

本文件冻结需要在 F3 证明的性质及其前提。当前 Python 测试和有界探索不构成证明。

## T1 Authority Origination

定义 `ancestor(ρ,g,T)`：在 trace \(T\) 中，resource \(\rho\) 由 grant \(g\) 经
有限次合法 split/delegate 产生。

定理义务：

\[
Effect(q,\rho)\in T
\Rightarrow
\exists g\in G.
Authenticated(g)
\land ancestor(\rho,g,T)
\]

证明结构：

1. 对小步转换归纳；
2. `issue` 是唯一增加 lineage root 的规则；
3. split/delegate 只扩展已有 lineage；
4. llm/read/write/endorse/merge 不创生 root；
5. tool/commit 只消费已存在 resource。

需要补足：IR parser 必须不能直接构造内部 `AuthorityResource`。

## T2 Authority Conservation

不使用把不同 scope 强行压成数字的单一 `Measure`。对每个 authenticated origin
\(g\)，定义：

\[
Live_g(S) = \biguplus_{\rho\in\Delta_S,\ origin(\rho)=g} budget(\rho)
\]

\[
Spent_g(T) = \sum_{q\in T,\ origin(q)=g} cost(q)
\]

定理义务：

\[
Live_g(S_t)+Spent_g(T_{\le t})\preceq InitialBudget(g)
\]

并且每个后代 atom 的 tenant/resource/action/purpose/epoch 不超过 root；delegate
只改变 subject 到合法 child，delegation depth 严格下降。

证明结构：

- split 使用加法预算侧条件；
- delegate 线性消费父 resource；
- consume 将 live 转移到 spent；
- merge 对共同 parent 重新验证 conservation；
- 其他规则保持 \(\Delta\)。

## T3 Non-Malleable Authority

攻击者等价：

\[
x\approx_T x'
\]

两次运行具有相同 trusted state、grant schedule、policy 和 explicit elevation，
仅 attacker-controlled InfluenceOrigin 数据不同。

### Supply non-malleability

\[
Project_{issue,split,delegate,revoke,expire}
(Trace(P,x))
\equiv
Project_{issue,split,delegate,revoke,expire}
(Trace(P,x'))
\]

对 split/delegate 的程序控制若受攻击数据影响，这个强等价可能过严。F3 必须选择：

1. 禁止攻击数据控制这些 authority-structural 操作；或
2. 只要求两次运行的可用 authority 互为等价衰减，不允许一侧更大。

F2A 预注册优先方案 1。

### Use non-malleability modulo explicit influence permission

若 effect 的 guard 不允许变化来源：

\[
\neg Eligible_g(attacker)
\Rightarrow
Project_{consume(g)}(Trace(P,x))
\equiv
Project_{consume(g)}(Trace(P,x'))
\]

若 guard 显式允许，则 effect 参数可不同，但：

\[
scope(effect)\preceq scope(g)
\quad\land\quad
budget\ conserved
\]

这比“所有 AuthorityTrace 必须完全相同”更精确，也保留合法的数据驱动任务。

## T4 Merge Confinement

设 branches 由共同 parent state \(S_p\) 经合法 partition 产生。若每个 branch
单独 well-typed，且 secure merge 通过：

\[
Authority(merge(b_1,\ldots,b_n))
\preceq Authority(S_p)
\]

\[
Effects(merge)
\not\models CombinedDeny
\]

还需证明 resource ownership 唯一，不能通过 fork、replay 或 lineage substitution
让同一预算被多次计入。

注意：

```text
safe(b1) ∧ safe(b2)
```

本身不蕴含 merge 安全。T4 的必要前提是 merge rule，而不是组合性假设。

## Preservation 与 progress

基础 preservation：

\[
\Gamma;\Delta\vdash e:\tau
\land
\langle e,S\rangle\rightarrow\langle e',S'\rangle
\Rightarrow
\exists\Delta'.\Gamma';\Delta'\vdash e':\tau'
\]

同时保持 T1/T2 invariant。

不承诺完整 progress：缺少 authority 的 well-formed effect 应产生受控拒绝而不是
执行；这属于安全停止。

## 与既有定理的关系义务

F3 必须回答：

1. T3 与 NMIFC 4-safety 的关系是什么？
2. T1/T2 是否可由线性授权逻辑 cut elimination 直接得到？
3. F2A 能否编码进 FLAC；若能，新增部分仅是哪些 LLM primitives/trace policy？
4. T4 与 APPA merge confinement 的状态和结论有何差异？
5. TACIT capture set 加 affine capability 后是否等价？

如果这些比较没有留下不可约性质，停止“新 calculus”主张。

## 机械化计划

- Alloy：origin reachability、lineage fork、recombination、merge；
- TLA+：并发 split/delegate/consume/revoke/crash；
- Lean 或 Rocq：小演算 preservation、T1、T2、T4；
- T3：先给 2-run/4-run 论文级定义，再决定机械化范围。

F3 最低目标：

```text
formal_model_completed = true
core_theorems_mechanized >= 3
counterexample_regressions = 0
python_trace_refinement_status = passed
```
