# F2C 有效性边界

* Benchmark 是同一项目内生成的公开/合成 ToolSymbolicIR，不是独立真人审核或真实工具。
* 完整性依赖受限 grammar、有限 3,072-assignment 域与 Z3 编码正确性。
* Evaluator 可枚举完整域用于指标，但 Analyzer 的 certificate 来自 SMT 差异 UNSAT，
  不是 query catalog 或全域 replay。
* 当前只验证 unary、single-run effects；不判断 payload 是否受 secret 影响。
* bounded delayed depth 不是一般异步系统；无界 loop、quiescence 或 instrumentation
  证据不足均返回 UNKNOWN。
* mechanism baselines 不是外部系统官方实现，不能据此声称外部 superiority。
* 即使 F2C gate 通过，仍需 F2D relational properties、F3 真实工具和 F4 Agent 评估。
