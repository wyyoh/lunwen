# 可复现性清单

## 1. 复现原则

- 以仓库中的正式报告、summary、manifest 和冻结 commit 为证据源；
- 不重新运行标记为 one-time/locked/formal 的审计；
- 重现时创建新的输出命名空间，不覆盖历史 artifact；
- 不将 ignored dataset、checkpoint、key、cache 或 runtime state 提交到 Git；
- 所有 JSON 使用严格有限数值，不允许 NaN/Infinity；
- 所有关键模型固定 model ID、revision 和文件 SHA-256；
- selection、development、locked/test 的用途必须与原协议一致。

## 2. 关键冻结提交

| Stage | Code-freeze/formal commit | 证据 |
|---|---|---|
| C2.4 | `86e81c3ee466fd2c13a0396bf9ffaa70d3bc45cb` | `stage_c24_summary.json` |
| C2.4c | `63fd9ba86f83aa121d18f3323ed902a58097d13e` | `stage_c24c_summary.json` |
| C2.5 | `6f47ffd7ab8816f7873fd2228ae07dad9ff42ffd` | `code_freeze_manifest.json` |
| D1 | `d92b699d44bc5ba7f88c16eaa4d854ea5255e572` | `stage_d1_summary.json` |
| D2 | `04834dc6389eb712a7e2ba36a8ec5d5027938ffb` | `stage_d2_summary.json` |
| D2.1 | `3e740a83576c41f4871c9e9e3848fff4bb468eb9` | `stage_d21_summary.json` |
| D2.2 | `06ca095b575b96447e963927287abb672d324ecb` | `stage_d22_summary.json` |
| D2.3 | `028422bf8bb203e3a683a6e90b67f210edd1261b` | `stage_d23_summary.json` |

结果封存提交可能晚于 code-freeze commit；summary 中的 git commit 表示正式运行使用的
代码状态。

## 3. 外部数据与模型

### CLINC150

```text
repository revision:
828f8093932c8fe6ca7936c3d2e52903b1c523de

data_full.json:
36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0

domains.json:
b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a
```

### BANKING77

```text
repository revision:
57ec275d8078af65b7731c2a98be812d844a6d6b

train.csv:
b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b

test.csv:
d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d
```

```text
categories.json:
53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63
```

### External router encoder

```text
model_id = intfloat/e5-base-v2
revision = f52bf8ec8c7124536f0efb74aca902b2995e5bcd
model.safetensors =
d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693
```

## 4. Split 与 selection 检查

### C2.4c

- [ ] v2.1 train/calibration/locked family namespace 不重叠；
- [ ] benchmark freeze manifest 有效；
- [ ] calibration 只用于 router selection；
- [ ] development/locked 未参与选择；
- [ ] 结果标为 `exploratory_non_independent`；
- [ ] 不声称人审或独立验证。

### C2.5

- [ ] CLINC150 使用官方 train/validation/test；
- [ ] 每个 domain 固定抽取 supported intents；
- [ ] BANKING77 open 按完整 intent 隔离；
- [ ] test-only OOS intent 在 calibration 不可见；
- [ ] 固定 seeds 2501–2510；
- [ ] 没有 synthetic OOS；
- [ ] 没有修改官方文本或标签；
- [ ] test predictions 只执行一次；
- [ ] test 不参与 threshold/model selection。

## 5. D 系列边界检查

- [ ] 仅使用运行期随机 128-bit synthetic canary；
- [ ] private answer/value 未加载；
- [ ] capability key 与 data key 分离；
- [ ] router proposal 不能形成 AuthorizedMemoryRequest；
- [ ] authorization 先于 lookup/decrypt；
- [ ] rejected request 不调用 generator；
- [ ] raw IPC 不进入正式 artifact；
- [ ] 日志/metrics/traces 不含 plaintext/token/key；
- [ ] replay/revocation/epoch 状态持久化范围与报告一致；
- [ ] source manifest 证明上游冻结文件未变；
- [ ] `private_value_memory_ready=false`；
- [ ] `original_c3_allowed=false`；
- [ ] `c3_eligible=false`。

## 6. Artifact 完整性

验证而不重跑正式实验：

```bash
sha256sum -c <由各阶段 artifact manifest 投影生成的校验列表>
```

逐阶段检查：

- summary JSON 可严格解析；
- manifest 中 file size 和 SHA-256 与磁盘一致；
- JSON 不含 NaN/Infinity；
- CSV 行数与 summary scenario count 一致；
- report SHA-256 与 external manifest 一致；
- artifact 不含 key/token/canary 明文；
- ignored `.runs/` 不提交。

E0 的冻结证据哈希见 [EVIDENCE_SHA256.json](EVIDENCE_SHA256.json)。

## 7. 测试基线

D2.3 结果分支在正式审计前验证：

```text
pytest -q
460 passed
```

E0 是纯文档分支。验证重点是：

- 相对 `agent/stage-d23` 只新增/修改 `*.md`、`*.mmd`、文档 JSON；
- C2/D 源码、配置、测试和 artifacts 的 SHA-256 不变；
- Markdown 相对链接全部存在；
- Mermaid 源文件存在且不嵌入 key/token/plaintext；
- `STAGE_STATUS_REGISTRY.json` 与 `EVIDENCE_SHA256.json` 严格解析；
- Git diff 无 whitespace error。

## 8. 禁止的“复现”

不得：

- 覆盖原 locked/test artifact；
- 根据 test 结果重选 router/threshold；
- 重新生成 v2.2/v2.3 合成 OOS；
- 伪造真人审核；
- 将 C2.4c 标为独立验证；
- 重新打开 C2.5 router complexity 分支；
- 创建 D2.4/D2.5；
- 加载 private answer、confirmation 或真实 credential；
- 把 synthetic zero-count 结果表述为部署安全概率。
