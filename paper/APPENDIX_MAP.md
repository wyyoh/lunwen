# 附录规划与证据映射

本文件把正文压缩掉的阶段细节映射到投稿附录。它不复制冻结 artifacts，只规定最终
LaTeX 版本应如何引用仓库中的现有材料。

## Appendix A：完整研究时间线与停止规则

正文对应：Introduction、Section 3.6。

纳入：

- C2–C2.5 与 D1–D2.3 的冻结顺序；
- 每阶段允许/禁止动作；
- calibration、development、locked/test 的使用边界；
- router 分支和 D 系列停止条件；
- 所有 readiness 状态。

证据：

- [E0 总体报告](../E0_RESEARCH_SYNTHESIS.md)；
- [阶段状态注册表](../docs/e0/STAGE_STATUS_REGISTRY.json)；
- [实验总表](../docs/e0/EXPERIMENT_MASTER_TABLE.md)。

## Appendix B：C2 表示与 lexical-OOD 研究

正文对应：Sections 3.1–3.3。

纳入：

- C2 的 Q0–Q3 配置和全部内部指标；
- C2.1/C2.2 的 relation head、SupCon、adversary 与 public paraphrase；
- C2.3 S0–S6 消融和 projected-family probe；
- C2.4 D0–D3 memory contract；
- source hash 与 checkpoint 不变性。

证据：

- `PHASE_C2_REPORT.md`
- `PHASE_C21_REPORT.md`
- `PHASE_C22_REPORT.md`
- `PHASE_C23_REPORT.md`
- `PHASE_C24_REPORT.md`
- 对应 `configs/`、`tests/` 和 `artifacts/`。

## Appendix C：C2.4b/c 合成压力集

正文对应：Section 3.4。

纳入：

- v2 被降级与 v2.1 新命名空间；
- 单模型 AI 审核协议；
- 37 个 boundary family flags；
- R0–R4 calibration 与冻结参数；
- risk–coverage；
- ambiguous 主要变成 unknown、multi-set 未出现；
- `exploratory_non_independent` 限制。

证据：

- `PHASE_C24B_REPORT.md`
- `PHASE_C24C_REPORT.md`
- `artifacts/stage_c24b/`
- `artifacts/stage_c24c/`

## Appendix D：C2.5 外部 OOS 审计

正文对应：Section 3.5 和 Table 1。

纳入：

- CLINC150/BANKING77 revision 和文件 SHA-256；
- 十个固定 seeds 和 supported-intent partitions；
- R0/R2/MAX/R3 精确定义；
- calibration-only 参数选择；
- 逐 seed 指标与近似置信区间；
- risk–coverage 全曲线；
- official OOS 与 held-out-intent OOS 分项；
- hard-intent error analysis；
- 0/10 continuation-gate 结果。

证据：

- `PHASE_C25_REPORT.md`
- `configs/stage_c25.yaml`
- `artifacts/stage_c25/per_seed_metrics.csv`
- `artifacts/stage_c25/risk_coverage.csv`
- `artifacts/stage_c25/error_analysis.csv`
- `artifacts/stage_c25/stage_c25_summary.json`

## Appendix E：D1/D2 类型与密码材料边界

正文对应：Sections 4 和 5.1。

纳入：

- `SuggestedRelation` 与 `AuthorizedMemoryRequest` schema；
- principal 三方相等检查；
- capability payload canonicalization；
- HMAC key ring、revocation、replay 和 single use；
- capability key/data key 类型与 material 分离；
- AEAD record schema 与 AAD；
- rotation/migration；
- 25 和 37 个完整场景。

证据：

- `PHASE_D1_REPORT.md`
- `PHASE_D2_REPORT.md`
- `artifacts/stage_d1/`
- `artifacts/stage_d2/`

## Appendix F：生成与服务边界

正文对应：Sections 5.2–5.3。

纳入：

- G0/G1 精确定义；
- tiny-GPT2 model/revision/hash；
- `prefix_allowed_tokens_fn`、`use_cache=False` 等限制；
- request-local context；
- prompt-injection scope cases；
- AF_UNIX schema 与三条 IPC；
- process material inventory；
- SQLite transaction mode；
- 40 和 35 个完整场景。

证据：

- `PHASE_D21_REPORT.md`
- `PHASE_D22_REPORT.md`
- `artifacts/stage_d21/`
- `artifacts/stage_d22/`

## Appendix G：故障、可观测性、IPC 与回滚

正文对应：Sections 5.4–5.5 和 6.5–6.6。

纳入：

- 五个 lifecycle milestones；
- 51 个完整预注册场景；
- structured log/metric/trace/error/retry/profile/supervisor sinks；
- byte-level IPC frame 与 parser 限制；
- local socket mode、owner 和 peer credential；
- disk-full/SQLite busy/corruption/partial write/clock drift；
- backup/restore；
- `state_epoch`、`policy_epoch`、`gateway_instance_epoch`；
- monotonic anchor 的同主机限制；
- canary 与 transformation registry。

证据：

- `PHASE_D23_REPORT.md`
- `artifacts/stage_d23/`

## Appendix H：复现与证据完整性

正文对应：Section 6.2。

纳入：

- 所有冻结 git commit；
- model/data/config SHA-256；
- CLI 命令；
- 软件依赖与设备；
- strict JSON 和 artifact schema；
- 正式审计只运行一次的警告；
- 不应重跑的 locked/test 边界。

证据：

- [E0 复现清单](../docs/e0/REPRODUCIBILITY_CHECKLIST.md)；
- [E0 证据哈希](../docs/e0/EVIDENCE_SHA256.json)；
- 各阶段 `artifact_sha256_manifest.json`。

## 附录压缩原则

1. 不把每个场景结果粘贴成正文式叙述；
2. 以 schema、表格和机器可读 artifact 路径为主；
3. 不把不同矩阵相加为总体安全概率；
4. 不在附录新增事后指标；
5. 不重新运行正式 locked/test/audit；
6. 不把未验证的 production 依赖画成已实现组件。
