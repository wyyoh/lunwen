# Blind CGAR 算法

```text
C0 = declared per-call contract + static hypotheses
P0 = declared low-level guards + trusted high-level invariants

repeat:
  1. 在 Ci/Pi 上寻找抽象危险路径；
  2. 没有危险路径时，按 coverage frontier 选择尚未观察的 query；
  3. 只通过 ReplayClient 执行该 query；
  4. 真实遗漏：加入 observed call trace，并按高层规范补 contract/policy/composition；
  5. 伪反例：删除不可重现 hypothesis 或细化 state guard；
  6. version 漂移：立即使 certificate 失效；
  7. 重合成 fail-closed Shield；
until finite domain covered / budget exhausted / evidence becomes UNKNOWN
```

每一轮持久化的只是 digest 和非敏感 reason code：abstract counterexample、replay
result、patch、model before/after、Shield before/after 与剩余 unknown query 数。正式
artifact 不保存 concrete event 或隐藏实现。

关键限制：model checker 不能从欠近似 contract 凭空发现模型外效果。因此
coverage-guided replay 与 unknown successor 是算法的必要组成，而不是评估捷径。
