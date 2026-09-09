# F2C 预冻结模板隔离

本文件描述预冻结修复，不是 development/locked 的实验结果。旧版 52 组重复内容指纹
保持为已发现的问题；旧生成器仅以明确命名的 legacy train/calibration fixture 保留，
用于回归重命名不能掩盖碰撞的测试，不参与新版正式数据生成。

## 独立审计

`template_isolation.py` 接收没有 case、tool、version 身份的 `TemplateDefinition`。
`audit_f2c_templates.py` 不调用正式实例生成器、Analyzer、replay service 或评分器。
它对四个 split 的生成器定义运行新审计，并保留旧内容审计：任何一个失败均不通过。

结构指纹分别检查输入/状态类型形状、guard、字段绑定、状态更新、分支、异步因果图、
实际循环体、整体 implementation 以及参考契约。变量按同类型变量的双射置换取
canonical minimum；去除名称和常量字面值差异，交换律操作数排序。
超过 alpha-normalization 审计预算时拒绝报告通过。

语义指纹逐项执行模板的完整小型有界域，将有序效果、字段绑定、因果父节点和最终状态
规范化为摘要。其用途只是碰撞审计，不把这些输出表提供给 Analyzer，也不将穷举作为
SMT completeness certificate 的依据。循环使用实际状态递推解释器，不用元数据替代。
恒假 guard、无效果的恒等更新、无意义恒真分离条件和未使用的 schema 字段均为失败。

## 实质不同的 meta-family

| split | 状态模型与 guard | 绑定/更新及异步协议 |
| --- | --- | --- |
| train | 单整数状态、单 role 条件 | 直接参数绑定、记录输入数量；发送依赖读取 |
| calibration | Bool + Enum，输入与状态合取 | 条件资源路由、数量相关确认状态、发送后回执 |
| development | Enum + bounded Int，阈值/DNF | 两种条件字段绑定、数量配额和阶段更新、状态递推的两轮队列循环 |
| locked 模板定义 | Bool + 两个 bounded Int，三路 DNF | 三种条件绑定、配额/审核更新、三轮循环及分叉回执 |

阈值作用于真实状态变更，异步步骤产生真实可观测事件，不通过改 ID、变量名、常量名称、
`AND true` 或不可达分支取得零碰撞。共享语言 primitive 不算共享完整模板。
每个 split 保留 10 类定义，正式规模仍为 16 tool families、160 cases，未删除困难类别。
本轮只审计 held-out 的模板定义，未 materialize 正式 hidden instances。

零碰撞是这里明确规定的规范化与有限域检查结果，不是任意程序语义独立性的证明。
