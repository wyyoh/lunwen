# F2B 信息边界

## Analyzer 可见

* opaque tool ID 与声明 version；
* finite replay query catalog；
* declared per-call contract；
* tool schema/static hypotheses 及其 evidence origin；
* 可信高层 bad-state specification；
* workflow topology；
* `ReplayResult`：当前 query 的有序 immediate/delayed event、state-diff digest、
  exit status、bounded-quiescence、instrumentation coverage 与 observed version digest。

## Analyzer 不可见

* concrete implementation / transition source；
* 未执行路径的效果；
* mutation category、omission type、预期伪反例；
* gold contract、gold low-level policy、gold Shield；
* “attack trace 是否 forbidden”等 evaluator label。

代码边界由 package import audit 强制：`keyed_gram.authsynth_analyzer` 只能 import
标准库、`authsynth_shared` 与冻结的通用 finite-game primitive。benchmark evaluator
可以调用 analyzer，反向依赖禁止。

## 证据限制

有限 query catalog 全覆盖、bounded quiescence 成功、instrumentation coverage 为
1、version 一致且每条 trace 已逐调用绑定，才允许 `VERIFIED_COMPLETE`。任一条件
不满足都返回 `UNKNOWN`，而不是从未观察到违规推断“完整”。
