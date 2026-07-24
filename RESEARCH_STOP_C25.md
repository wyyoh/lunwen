# C2 路由研究路线停止记录

本记录在 Stage C2.5 外部 OOS 泛化审计完成后生效。

机器可读状态位于 `configs/research_status.yaml`。其证据绑定到 C2.5 的
一次性 test results、protocol status、正式报告和 artifact manifest 的
SHA-256；本记录不修改已经封存的 C2.5 artifacts。

正式状态：

```text
selective_router_research_status = stopped_external_validation_failed
pairwise_evidence_branch_status = stopped
set_valued_ambiguity_branch_status = stopped
additional_router_complexity_allowed = false
task_specific_router_ready = false
exploratory_c3_allowed = false
c3_eligible = false
```

永久停止在当前证据上继续：

- R4/R5 或其他 learned router 变体；
- 更大的 semantic encoder 或 cross-encoder router；
- 更多 threshold、margin 或 conformal 搜索；
- 对已打开的 CLINC150/BANKING77 test 反向调参；
- 新造 v2.2/v2.3 synthetic router benchmark；
- MASSIVE router 扩展；
- 为通过结果降低 coverage gate；
- 让 learned natural-language router 直接承担授权边界。

保留结论：

1. Stage C2.4 typed discrete memory contract 的结构隔离成立。
2. learned semantic routing、capability localization 与 authorization
   是不同问题。
3. 低 FMAR 不能脱离 known coverage 和 safe coverage 解释。
4. hard OOS 与近邻 ontology 使模型置信度不适合作为可信授权凭据。

下一阶段重置为 Stage D1：Trusted Capability Contract。自然语言解析器
只能产生不可信 `SuggestedRelation`；只有显式 subject、entity scope、
`RelationId`、policy 和 authenticated capability 能形成 memory request。

Stage D1 不训练 private value memory，不接 LM answer injection，不执行
密钥攻击，也不声称部署级安全。其目标只是验证 capability contract 的
fail-closed、scope、expiration、revocation、replay 和 key-rotation 语义。

