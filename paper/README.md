# Stage E1：Manuscript v1

本目录是论文第一版完整稿件。Stage E1 只整理和压缩既有证据，不新增实验，
不修改 C2/D 系列冻结代码、配置或正式产物。

## 主要文件

- [MANUSCRIPT_V1.md](MANUSCRIPT_V1.md)：英文论文全文初稿；
- [REFERENCES.bib](REFERENCES.bib)：经一手出版页或论文原文核验的参考文献；
- [RELATED_WORK_AUDIT.md](RELATED_WORK_AUDIT.md)：相关工作检索范围、证据定位与差异审计；
- [APPENDIX_MAP.md](APPENDIX_MAP.md)：正文压缩后各阶段材料的附录映射；
- [REVIEW_RISK_REGISTER.md](REVIEW_RISK_REGISTER.md)：审稿风险、预期质疑与有限回应；
- [PAPER_SHA256.json](PAPER_SHA256.json)：除清单自身外的论文文件 SHA-256；
- [figures/](figures)：正文图源；
- [tables/](tables)：正文核心表格的机器可读源。

## 稿件状态

```text
manuscript_version = v1
new_experiment_executed = false
frozen_experiment_code_modified = false
frozen_artifact_modified = false

additional_d_series_stages_allowed = false
d_series_status = completed
semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

## 正文压缩原则

正文只保留：

1. 一张研究证据链图；
2. 一张最终 TCB 架构图；
3. 一张 C2.5 外部负结果表；
4. 一张 D 系列不变量审计表；
5. 一张主张边界表；
6. 一张 fail-closed 生命周期图。

逐阶段参数、完整攻击矩阵、逐 seed 结果、哈希清单和审核记录继续保留在仓库原始
报告与 artifacts 中，不在正文重复。

## 引用格式

正文使用 Pandoc 风格引用键，例如 `[@larson2019clinc]`。所有正文引用键必须在
`REFERENCES.bib` 中存在；参考文献库中的正式出版信息以 ACL Anthology、
NeurIPS/PMLR/AAAI/USENIX/NDSS/ACM/IEEE 官方页面为优先来源。仅有预印本的工作会
明确标注为 `arXiv preprint`。
