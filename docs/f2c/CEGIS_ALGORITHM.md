# AuthSynth-Symbolic CEGIS

```text
E0 = ∅
C0 = declared contract + public static hypotheses

repeat:
  Verify(T, Ci) with bounded SMT
  ├─ difference assignment x*: Replay(x*), add structured example, synthesize Ci+1
  ├─ version mismatch: invalidate old certificate, reset evidence, analyze new version
  ├─ bounded-domain equivalent: issue a version/schema-bound certificate
  └─ timeout/unsupported/evidence gap: UNKNOWN
```

Verifier 分别寻找 state-update mismatch、field-binding mismatch、missing effect 和
spurious effect。只返回 assignment、difference kind、version digest 或
`EQUIVALENT_ON_BOUNDED_DOMAIN/UNKNOWN`，不返回内部 path formula 或 reference contract。

Learner 从 membership examples 构造 effect slots、显式 field terms，并用 Z3 Optimize
在生成的候选 cube 池中优化覆盖正例且排除负例的 DNF；当前 cube 池有启发式剪枝，
不能声称对完整 grammar 全局最小。优化顺序是 clause 数、literal 数、canonical
lexicographic tie-break；最后一项使用唯一二进制权重，避免索引和相同导致歧义。
候选构造和求解共享单次 fit 的 deadline，Z3 同时接收剩余 timeout。
语法不足和求解超时分别记录 reason，均不产生完整性证书。

Verifier 先检查 tool/version/schema 绑定与状态更新合法性，再寻找差异。
状态更新按同一前状态求值；冲突、越域和不支持的 term 类型不能被后写覆盖掩盖。
等价响应必须绑定实际版本。漂移清除旧证据，但不恢复已消耗的 replay 预算；
失败 replay 计入次数而不加入训练样例。没有实际旧证书时不伪造撤销 ID。

这些是预冻结实现规则，尚未构成正式 development/locked 通过结果。
