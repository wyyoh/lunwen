# F2B 实验协议

1. 从 F2A 最终提交创建独立分支，F1/F2A 全部冻结资产只读 hash 绑定。
2. 先冻结 shared schema 与 analyzer；此提交不含 evaluator benchmark generator。
3. analyzer freeze 后再加入独立 evaluator/replay 与新 F2B benchmark generator。
4. 新 benchmark 使用新的 namespace 和 locked split；绝不复用或重跑 F2A locked。
5. analyzer/config/benchmark freeze 后，只用 train/calibration 做 sanity；安全门槛不可调。
6. development 恰好一次；结果不触发代码、case、阈值或 split 修改。
7. 新 F2B locked_test 恰好一次；重复运行由持久 protocol state 拒绝。
8. artifact 只保存 digest、统计与 reason code，不保存隐藏 transition、完整 event、
   private data、token、key 或自然语言样本。

抽象 baseline 使用机制名称：`Declared-Contract Monitor`、`Multi-Evidence
Characterizer`、`DenyAll`、`Full-Semantics Upper Bound`；未调用外部系统官方实现，
不得声称优于其正式实现。
