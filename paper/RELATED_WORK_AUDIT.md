# Related Work 检索与差异审计

## 1. 审计范围

本次检索于 2026-07-26 完成，优先使用论文原文和正式出版页面：

- ACL Anthology；
- NeurIPS、ICLR、PMLR、AAAI 官方 proceedings；
- USENIX、NDSS、ACM、IEEE 官方页面；
- 仅在没有正式出版版本时使用 arXiv，并明确标注预印本。

正文不使用搜索结果摘要作为最终证据。所有进入正文的引用键均保存在
[REFERENCES.bib](REFERENCES.bib)。

## 2. Intent、OOD 与选择性预测

| 文献 | 已核验来源 | 与本文的关系 | 本文不重复的贡献 |
|---|---|---|---|
| Larson et al. 2019 | ACL Anthology、DOI `10.18653/v1/D19-1131` | CLINC150 定义了独立 OOS 数据，并观察到 in-scope 分类与 OOS 识别之间的差距 | 本文不提出新 OOS benchmark；使用官方 split 检验授权 analogue |
| Casanueva et al. 2020 | ACL Anthology、DOI `10.18653/v1/2020.nlp4convai-1.5` | BANKING77 提供 77 个细粒度、同领域 intent | 本文把完整 held-out intent 当作未知类，避免同一 intent 跨 known/OOS 泄漏 |
| Lin and Xu 2019 | ACL Anthology、DOI `10.18653/v1/P19-1548` | margin features + novelty detection 处理 unknown intent | 本文不追求新的 unknown-intent SOTA，而检验拒识能否承担访问控制 |
| Zhang et al. 2021 | AAAI、DOI `10.1609/aaai.v35i16.17690` | relation-specific adaptive boundary 平衡 known/open intent | 本文的 R3 也校准边界，但结果显示低错误访问可由 coverage 崩溃产生 |
| Hendrycks and Gimpel 2017 | ICLR/OpenReview | maximum softmax probability 是经典 OOD baseline | 本文的 MAX threshold 正是必要简单对照，并优于更复杂 R3 的 trade-off |
| Geifman and El-Yaniv 2017 | NeurIPS proceedings | 选择性分类显式优化 risk–coverage | 本文采用相同核心视角，但将错误动作解释为 memory-access proxy |
| Liu et al. 2019 | NeurIPS proceedings | 端到端学习 abstention | 本文没有继续训练新 abstention head，因为外部停止条件已经触发 |
| Guo et al. 2017 | PMLR | temperature scaling 可改善概率校准 | 校准正确不等于授权正确；本文把 coverage、wrong bucket 与 FMAR 分开报告 |
| Angelopoulos and Bates 2023 | Foundations and Trends/arXiv | conformal set 在交换性条件下提供覆盖工具 | 本文不声称分布迁移下的 conformal 保证，并因外部结果停止继续扩展 R4 |

这一文献群主要回答“如何分类、检测未知或在何时弃权”。本文提出的不同问题是：

> 当一次分类错误会触发受保护资源访问时，分类置信度或弃权是否足以成为授权条件？

C2.5 的答案是否定性的、经验性的，而不是对所有选择性分类方法的形式化不可能性证明。

## 3. Capability、least privilege 与访问控制

| 文献 | 已核验来源 | 与本文的关系 | 本文定位 |
|---|---|---|---|
| Dennis and Van Horn 1966 | ACM DOI `10.1145/365230.365252` | capability 概念的经典系统语义来源 | 本文不声称发明 capability |
| Saltzer and Schroeder 1975 | IEEE DOI `10.1109/PROC.1975.9939` | fail-safe defaults、complete mediation、least privilege | D 系列把这些原则具体化为 authorization-before-access 与 default deny |
| Sandhu et al. 1996 | IEEE DOI `10.1109/2.485845` | RBAC 将 policy decision 与受控资源联系起来 | 本文 capability 承载 policy 决定，不取代 IAM、RBAC 或 ABAC |
| Miller et al. 2003 | 原作者/机构全文页 | capabilities 支持 least privilege、confinement 与 revocation 设计 | D1 的 scope、expiry、revocation 是原型组合，不是理论创新 |
| Birgisson et al. 2014 | NDSS 官方页、DOI `10.14722/ndss.2014.23212` | chained-HMAC macaroon 通过 caveat 限制上下文与授权范围 | D1 使用更窄的固定 schema HMAC token，不实现 delegation 或 macaroon caveat |
| Cao et al. 2024 | USENIX Security 官方论文 | stateful attenuation 支持 read-at-most-N 等策略 | D 系列关注 memory-to-generation 的 single-use 与故障生命周期，范围更窄 |

本文与 ACL/RBAC 的关系为：

```text
external identity and policy decision
→ typed authenticated capability
→ exact typed memory access
```

Capability 是 policy 决定后的最小、可认证、可传递凭据；它不自行决定用户应拥有什么
权限，也不取代外部身份系统。

## 4. Information-flow control 与端到端系统边界

| 文献 | 已核验来源 | 与本文的关系 | 差异 |
|---|---|---|---|
| Myers and Liskov 1997 | ACM SOSP DOI `10.1145/268998.266669` | decentralized labels 把信息流策略与 principal 联系起来 | 本文没有静态证明 noninterference，只执行 typed/IPC 数据最小化 |
| Myers 1999 | ACM POPL DOI `10.1145/292540.292561` | JFlow 以类型系统约束程序信息流 | 本文的 Python 原型不具备语言级 IFC 保证 |
| Saltzer et al. 1984 | ACM TOCS DOI `10.1145/357401.357402` | end-to-end argument 强调功能应在能完整实现语义的端点检查 | 本文将授权放在 lookup/decrypt/generation 之前的显式 gateway |
| Lee et al. 2015 | ACM SOSP 官方 proceedings | RIFL 通过请求标识与结果记录构建 exactly-once RPC 基础设施 | D 系列只验证单主机、SQLite、single-use fail-closed，不声称分布式 exactly-once |

因此，“typed”在本文中表示 API 和资源作用域的数据最小化，不应被写成形式化
information-flow type system。

## 5. LLM memory、RAG 与 tool use

| 文献 | 已核验来源 | 与本文的关系 | 本文的区别 |
|---|---|---|---|
| Lewis et al. 2020 | NeurIPS proceedings | RAG 将非参数 memory 接入生成模型 | 该工作优化知识访问和生成质量，不把语义检索器当作授权器 |
| Park et al. 2023 | ACM UIST DOI `10.1145/3586183.3606763` | generative agents 存储、反思并动态检索经历 | 本文研究多主体作用域和授权先于 retrieval |
| Packer et al. 2023 | arXiv `2310.08560` | MemGPT 管理多级上下文和长期 memory | 本文不优化 memory 容量或管理策略，关注访问控制 TCB |
| Schick et al. 2023 | NeurIPS proceedings | Toolformer 学习何时、如何调用 API | 本文的核心反例正是“会选工具”不等于“有权调用工具” |

这些工作证明 memory 与 tool orchestration 的效用，但不自动提供 subject/entity/relation
作用域的授权语义。

## 6. Agent prompt injection、风险评估与可信包装层

| 文献 | 已核验来源 | 与本文的关系 | 本文的区别 |
|---|---|---|---|
| Greshake et al. 2023 | ACM AISec DOI `10.1145/3605764.3623985` | 间接 prompt injection 可利用被检索内容操纵集成应用 | 本文不依赖模型识别恶意文本，而在模型之外固定权限 |
| Ruan et al. 2024 | ICLR proceedings | ToolEmu 用模拟工具环境发现高风险 agent 失败 | D 系列使用真实本地数据流与预注册不变量，不用 LLM judge 评安全 |
| Zhan et al. 2024 | ACL Anthology、DOI `10.18653/v1/2024.findings-acl.624` | InjecAgent 测试工具集成 agent 的伤害与数据外泄 | 本文缩小 generator 权限：无 memory 工具、无 capability、单值输入 |
| Debenedetti et al. 2024 | NeurIPS proceedings | AgentDojo 提供动态 prompt-injection 攻防环境 | 本文不声称覆盖 AgentDojo 的通用 agent 功能 |
| Debenedetti et al. 2025 | arXiv `2503.18813` | CaMeL 将控制流和数据流放入保护层，并使用 capability 防止未授权外流 | 这是最接近的工作；本文新增的是 router 授权失败的外部负结果，以及 keyed-memory-to-generation 的 replay、crash、observability、IPC 与 rollback 审计 |

CaMeL 是必须在正文主动讨论的直接相关工作。两者共同支持“不要把不可信模型输出直接
当作权限”的方向，但研究问题不同：

- CaMeL 主要保护 tool-using agent 的控制/数据流免受 prompt injection；
- 本文先检验 learned semantic router 是否适合授权，在外部失败后重构 memory access；
- 本文不声称 prompt-injection 的可证明防御，也不覆盖通用 agent program synthesis；
- 本文的 D 系列进一步审计 single-use、加密 record、生成、进程、观测和生命周期。

## 7. 论文可使用的差异总结

已有研究通常选择以下一个目标：

1. 提高 intent/OOS 分类或选择性预测；
2. 用 capability、RBAC/ABAC 或 stateful token 表达权限；
3. 提高 LLM memory/RAG/tool use 的效用；
4. 检测或约束 prompt injection 与 agent 风险；
5. 用 IFC 或分布式协议保证更强的系统性质。

本文不声称在这些单项上取得新的算法或密码学突破。论文贡献是把它们通过一条负结果
驱动的论证链连接起来：

```text
learned semantic router 的授权 analogue 外部失败
→ interaction semantics 与 authorization semantics 分离
→ router 移出 TCB
→ policy 后置的 typed capability
→ authorization-before-lookup-before-decrypt-before-generation
→ single-host public/synthetic prototype 的分层不变量审计
```

## 8. 引用状态

```text
reference_entries = 28
primary_or_official_publication_pages_used = true
preprints_explicitly_marked = true
unverified_search_snippets_cited = false
related_work_complete_for_manuscript_v1 = true
venue_specific_style_applied = false
```

投稿前仍需根据目标 venue：

- 转换引用与匿名格式；
- 检查页数限制；
- 检查 arXiv 预印本是否出现正式出版版本；
- 进行一次人工 bibliography copy-edit。
