# DAIFC 最小形式内核

## 1. 两个环境

数据环境：

\[
\Gamma(x) = (\tau, L_D), \qquad
L_D = \langle C, I, \Pi\rangle
\]

其中 \(C\) 为 confidentiality，\(I\) 为 integrity，\(\Pi\) 为
`InfluenceOrigin` 集合。组合数据时：

\[
C_1 \sqcup C_2 = \max(C_1,C_2),\qquad
I_1 \sqcup I_2 = \min(I_1,I_2)
\]

authority 环境为线性多重资源：

\[
\Delta = \{\rho_1,\ldots,\rho_n\}
\]

\[
\rho = \langle id, o, a, b, lineage\rangle
\]

\[
a =
\langle issuer,subject,tenant,resource,action,purpose,epoch,g\rangle
\]

其中 \(o\) 是 `AuthorityOrigin`，\(b\) 是 `AuthorityBudget`，\(g\) 是
`InfluenceGuard`。`InfluenceOrigin` 与 `AuthorityOrigin` 是不同类型，没有
从前者到后者的默认 coercion。

## 2. Budget 偏序

当前原型使用有限自然数向量：

\[
b = \langle uses,amount,bytes,resources,depth,ttl\rangle
\]

\[
b' \preceq b \iff \forall d.\;b'_d \le b_d
\]

`split` 对 uses/amount/bytes/resources 采用加法守恒，对 depth/ttl 要求每个
child 不超过 parent。`delegate` 消费父 resource，child depth 至少减一。
未分配预算被销毁，不会回流。

这只是第一版偏序。金额、bytes 和 resources 的单位语义仍需 F2B 的 effect
schema 固定；F2A 不把不同单位压成单一标量。

## 3. 表达式

最小语法：

```text
e ::= v
    | let x = e1 in e2
    | branch e1 e2
    | merge e1 e2
    | llm(e)
    | read_memory(r)
    | write_memory(r, e)
    | tool_call(t, args)
    | spawn(agent, e)
    | split(a, budgets)
    | delegate(a, child, budget)
    | endorse(e, permit)
    | declassify(e, permit)
    | commit(effect)
```

运行时状态：

\[
S=\langle \sigma,\Gamma,\Delta,H,E,G\rangle
\]

- \(\sigma\)：普通 store；
- \(\Delta\)：线性 authority；
- \(H\)：data/influence trace；
- \(E\)：effect 与 authority-consumption trace；
- \(G\)：仅来自外部可信控制面的 authenticated grants。

判断式：

\[
\Sigma;\Gamma;\Delta
\vdash e:\tau
\triangleright
\langle L_D,\epsilon,\Delta',T\rangle
\]

## 4. Origination

唯一创生规则：

\[
\frac{
g\in G \quad Authenticated(g)
}{
\Gamma;\Delta
\vdash issue(g)
\triangleright
\Delta \uplus \{\rho_g\}
}
\]

不存在：

```text
Text -> Authority
Integrity.TRUSTED -> Authority
LLMOutput -> Authority
ToolOutput -> Authority
MemorySummary -> Authority
```

## 5. LLM

\[
\frac{
\Gamma;\Delta\vdash p:Prompt\triangleright
\langle L,\varnothing,\Delta,T\rangle
}{
\Gamma;\Delta\vdash llm(p):Text\triangleright
\langle L',\varnothing,\Delta,T\cdot llm\rangle
}
\]

其中 \(L'\) 至少保留 prompt 的 influence join。LLM 语义可非确定地产生任意
文本；authority 环境逐字不变。

## 6. Tool/commit

对 effect \(q\)，必须存在唯一 resource \(\rho\)：

\[
scope(\rho)=scope(q)
\]

\[
cost(q)\preceq budget(\rho)
\]

\[
guard(\rho)\models Influence(label(args))
\]

执行后：

\[
budget(\rho)'=budget(\rho)-cost(q)
\]

如果任一前提失败，effect 不发生，\(\Delta\) 不变化。

`InfluenceGuard` 是双轨连接点，但不是 endorsement：

- 它由 grant 的可信控制面指定；
- 高 integrity 不能自动满足来源类别；
- attacker-controlled influence 默认不允许；
- 允许攻击数据控制某个低风险 effect 必须在 grant 中显式声明。

## 7. Endorse

数据 endorsement 只改变 \(I\)，保留 \(\Pi\)：

\[
endorse(\langle C,I,\Pi\rangle,p)
=\langle C,I',\Pi\rangle
\]

\[
\Delta'=\Delta
\]

只有不受攻击者控制的 `TrustedEndorsement` 可执行。endorse 不签发 tool
authority，也不删除原始 influence。

## 8. Split 与 delegate

Split：

\[
\rho \mapsto \rho_1,\ldots,\rho_k
\]

\[
\sum_i b_i^{additive}\preceq b^{additive}
\]

父 \(\rho\) 被消费，每个 child 有唯一 ID 和扩展 lineage。

Delegate：

\[
delegate(\rho,child,b')=\rho'
\]

\[
b'\preceq b,\qquad depth(b')<depth(b)
\]

父 resource 不再留在调用者上下文。需要父/子同时保留时，必须先 split。

## 9. Branch 与 merge

合法并行先显式 partition authority。每个 `BranchState` 绑定共同
`parent_context_digest`。Merge 的前提包括：

1. 所有 branch 绑定同一 parent；
2. 存活 resource ID 跨 branch 唯一；
3. 对每个 origin：

\[
\sum remaining\_budget
+
\sum consumed\_cost
\preceq parent\_budget
\]

4. 所有 origin authenticated；
5. effect signature 集不命中 `ForbiddenEffectCombination`。

因此 merge 不是 authority union，也不是简单 intersection。intersection 会安全地
丢失合法 split 后的不同 child；union 会复制共享父 resource。规则依据 lineage、
预算和 effect history 做受约束重组。

## 10. 确定性与 canonical form

authority atom、budget、origin、resource 和 context 均有 canonical JSON；context
按 resource ID 排序并做 SHA-256 digest。SHA-256 只用于原型确定性和 trace 绑定，
不是本研究的密码算法创新。

## 11. 当前未形式化部分

- arbitrary recursion、exceptions 和 concurrency；
- declassification 的完整规则；
- distributed authority consumption；
- 真实 policy oracle 到 authority issue 的 refinement；
- timing/termination-sensitive flow；
- LLM probability；
- 对 FLAC/NMIFC 的语义编码。

因此本文件是 F3 定理输入，不是形式证明。
