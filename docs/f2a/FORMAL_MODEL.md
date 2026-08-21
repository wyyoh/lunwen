# ToolEffectIR 有限形式模型

## 1. 状态、调用与效果

一个调用由工具、版本、受限参数和可信有限环境组成：

\[
c=\langle tool,version,args,env\rangle.
\]

具体工具实现是有限 transition relation：

\[
T(c,s)\rightarrow \{(s'_i,E_i,Q_i)\}_i,
\]

其中 `E` 是 immediate effects，`Q` 是有界 delayed effects。执行到 bounded
quiescence 后，`Q` 必须排空再判断最终状态。

一个 concrete effect 原子明确绑定：

```text
kind × tenant × resource × destination × phase × argument-role bindings
```

不存在自然语言 effect 或隐式 wildcard。

## 2. Contract concretization

声明契约由 effect templates 构成。每个 template 的 scope 字段只能引用受限参数
角色或固定 synthetic 常量。给定调用 `c`，`γ(C,c)` 是模板实例化得到的具体效果
集合。

在预注册有限输入域 `I` 上：

\[
EffectComplete(T,C,I) \iff
\bigcup_{c\in I}Effects_Q(T,c)\subseteq
\bigcup_{c\in I}\gamma(C,c).
\]

覆盖还要求所有安全相关 argument roles 被 contract 绑定；只声明 effect kind、
却漏掉 callback URL 或 tenant role，不算覆盖。

## 3. Forbidden predicates

禁止条件是 concrete effect selectors 的合取。一个 rule 的每个 selector 都在
累计 trace 中获得匹配时，状态进入 forbidden set。该表示可以描述单效果禁止
和多工具组合禁止：

```text
Read(confidential-record) ∧ Send(external-destination)
```

Ground truth 来自 transition execution 与该确定性 predicate，不来自 LLM judge。

## 4. 精确最大许可 Shield

有限安全博弈为：

\[
G=(S,S_0,A,\delta,B,F),
\]

其中 Agent 选择 action，环境从 `δ(s,a)` 的 successor 集中非确定选择，`B` 是
forbidden states，`F` 是安全 terminal states。

最大 winning region 是最大不动点：

\[
W=\nu X.\;(S\setminus B)\cap
\{s\mid s\in F\lor\exists a.\;\delta(s,a)\subseteq X\}.
\]

Shield 在 `s∈W` 允许且仅允许满足 `δ(s,a)⊆W` 的 action。因此在该有限模型上：

- Shield Safety：所有允许轨迹均不进入 `B`；
- Maximal Permissiveness：任何额外允许 action 都会离开 `W` 或允许环境到达 `B`。

这不是任意程序的完备性证明。

## 5. CGAR refinement

初始抽象由 declared、inferred 和 observed evidence 构成。候选 trace 经 sandbox
replay 后：

- 真实实现复现遗漏效果：补 contract/role/version；
- concrete effect 已覆盖但 policy 漏掉 forbidden condition：补 policy；
- 单步安全但组合禁止：补 composition restriction；
- concrete execution 不复现 abstract trace：收紧 abstraction；
- implementation version 不同：使 certificate 失效并重新分析。

F2A 的“sandbox”只是精确有限 transition interpreter；F2B/F3 才实现真实 evidence
fusion 与 instrumented sandbox。
