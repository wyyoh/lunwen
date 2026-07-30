# 攻击者与攻击语义

## 攻击者控制面

攻击者可以改变：

- 用户或第三方自然语言；
- web/tool/memory 返回的数据；
- LLM 非确定输出；
- 长期 memory 的不可信片段；
- spawn 的不可信子任务；
- 分支调度与 merge 顺序；
- 未认证的“用户已批准”文字；
- authority resource 之外的普通值。

攻击者不能：

- 向可信 grant set \(G\) 插入 authenticated grant；
- 伪造 `AuthorityOrigin.authenticated=true` 的运行时构造边界；
- 修改可信 policy epoch；
- 绕过 interpreter/typechecker 直接执行 effect；
- 使线性 resource 在语义转换中出现两次。

Python dataclass 不是密码学隔离。这里的“不能”是形式机的构造规则，而非 Python
对象不可伪造声明。F2B 必须把 IR parser 与 interpreter 设为唯一构造边界。

## Influence attacker

输入等价关系：

\[
x \approx_T x'
\]

表示可信 grant、policy、principal、环境和可信数据相同，仅攻击者控制的
`InfluenceOrigin` 数据不同。

LLM 被建模为对攻击者输入高度敏感的非确定函数，因此普通输出可完全不同。
安全性不能依赖模型遵循 prompt。

## Authority trace 投影

区分两个 trace：

```text
SupplyTrace:
  issue, split, delegate, expire, revoke

UseTrace:
  consume, effect, commit
```

对攻击者输入变化，`SupplyTrace` 必须等价。`UseTrace` 只有在 grant 上的
`InfluenceGuard` 允许相应 InfluenceOrigin 控制 effect 时才可变化，并始终受
scope/budget/combined deny 约束。

这个定义避免一个过强而无用的性质：如果用户明确授权 Agent 根据公开天气选择
“带伞/不带伞”，公开数据可以影响低风险 effect；但天气数据仍不能创生支付或删除
authority。

## 六类攻击转换

### Authority laundering

不安全语义含有：

```text
trusted_container(data) ⇒ trusted_instruction(data)
trusted_instruction(data) ⇒ authority(data)
```

安全语义只允许：

```text
rewrite(data) ⇒ influence provenance preserved
authenticated_grant ⇒ authority
```

### Capability recombination

不安全 merge 对两个 branch authority 求 union。安全 merge 要求 parent digest、
线性 child ownership、budget conservation 和 combined-effect oracle 同时通过。

### Delegation fork

不安全 delegate 把 bearer capability 复制给多个 child。安全 delegate 消费父
resource；若要多个 child，必须先 split 并分配总预算。

### Trusted-tool echo

不安全标签器按最后一个工具身份覆盖输入完整性。安全标签器 join 原始
InfluenceOrigin；即使显式 endorse，提高 integrity 也不自动满足 authority 的
allowed source kind。

### Memory fragmentation

不安全系统在 summary 后把多个文本 claim 当作 approval。安全系统允许 fragments
影响数据推理，但没有任何文本到 AuthorityOrigin 的语义规则。

### Data-to-authority confusion

不安全系统把高完整性事实当命令。安全系统分别检查：

```text
fact integrity
authority origin
effect scope
influence eligibility
```

四者不能互相替代。

## Elevation

未来可信 elevation 必须是显式事件：

\[
elevate(origin,scope,guard,reason)
\]

其触发条件不能由攻击者控制的数据单独决定。F2A 只实现数据 `endorse`，没有
authority elevation；因此任何需要新 authority 的路径都必须回到外部 policy/user
control plane。

## Fail-closed

以下状态均拒绝：

- unknown origin；
- budget unit 不匹配；
- epoch 变化；
- cross-subject/tenant/resource/action/purpose；
- duplicate resource ID；
- lineage 不连续；
- influence guard 不满足；
- combined deny；
- branch parent digest 不一致；
- authority 总预算超过父 grant。

拒绝复杂任务不是最终效用方案。F4 必须同时报告 safe task completion，防止
deny-all 获得表面安全。
