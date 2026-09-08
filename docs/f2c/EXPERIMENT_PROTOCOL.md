# Stage F2C 一次性协议

```text
source implementation/tests
→ Analyzer freeze commit + push
→ formal benchmark materialization
→ calibration and parameter freeze
→ development once
→ locked_test once
→ strict artifact audit
```

正式 preflight 要求正确分支、tracked worktree 干净、Analyzer freeze 为 HEAD 祖先、
F1/F2A/F2B frozen hashes 一致且所有 output/runtime 路径不存在。Phase state 文件拒绝
乱序或重复 prepare/materialize/calibration/config-freeze/development/locked。

正式 public split 只保存 schema、declared contract、trusted safety spec 和运行参数；
不保存 hidden implementation AST、reference contract、path predicate 或完整 assignment
output table。所有 JSON 拒绝 duplicate key、NaN/Infinity；CSV 使用固定 LF。
