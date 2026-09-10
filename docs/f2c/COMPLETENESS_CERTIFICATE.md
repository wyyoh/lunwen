# 有界完整性与版本 certificate

只有同时满足以下条件才签发 `CONTRACT_VERIFIED_COMPLETE`：

1. input/state schema 是完整有限域；
2. candidate 与内部 transition semantics 的双向差异公式均 UNSAT；
3. solver 未 timeout；
4. grammar 可表达当前 candidate，unsupported region 为空；
5. bounded delayed queue 在 horizon 内排空；
6. 不存在无界/未知 loop 或 instrumentation gap；
7. implementation version 与 certificate 绑定版本一致。

Certificate 只记录 contract/schema/version digests、domain cardinality、solver 版本和
proof result，不携带内部实现、reference contract 或秘密。版本不匹配时旧 certificate
先变为 INVALID；新版本必须从空 evidence 独立 re-certify，并获得不同 certificate ID。

该 certificate 只证明冻结 ToolSymbolicIR bounded domain 的单运行效果等价，不证明
真实程序完备、relational noninterference 或秘密 payload 安全。
