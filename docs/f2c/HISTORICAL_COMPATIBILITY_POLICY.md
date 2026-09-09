# 用户确认的历史兼容例外与 Analyzer Freeze

本协议在新一轮普通回归之前固定。依据用户明确授权，上一轮 7 项历史兼容失败可在
精确匹配时不阻止 F2C freeze。旧 684 passed / 7 failed 记录和旧 FAILED 状态不回写。

## 精确匹配，不改变 pytest

`configs/f2c_historical_test_allowlist.yaml` 是严格 YAML（采用 JSON 子集以避免歧义）。
仅允许用户指定的 7 个完整 node ID，每项最多 1 次；未知字段、重复 key、NaN/Infinity、
目录通配符、扩展例外数量均拒绝。例外文件与分类器、runner、结果摘要均进入冻结 SHA。

失败签名固定 call 阶段、异常完整类型、消息、从测试入口至抛出点的完整 frame 序列
（相对文件、函数、行号）、异常链及原因上下文。只去除测试入口之前的 pytest 调度层；
不对应用 frame 做通配或“包含某关键字”匹配。临时目录随机前缀与对象地址不参与
签名，也不收集秘密 locals；上下文只提取预注册安全字段。

D2.1 固定实际 Transformers 5.14.1/5.16.1 和 Torch 版本对，并绑定原配置及所有相关
frame 文件 hash。F2A 必须仅 `.gitattributes` 不匹配、artifact failures 为空、目录精确。
D2.3 必须是原 generator socket、positive cluster、60 秒时限和 ipc_timeout 原因。
相同 node ID 出现不同原因/新 frame/异常链/次数增加仍阻止 freeze，F2C 失败永不豁免。

runner 不使用 `pytest || true`、xfail、skip、筛选历史目录或修改历史源码。
它执行完整普通 pytest，保留原退出码、日志/XML 摘要和每项失败，再作独立分类。
collection/internal error、缺失/重复执行、跳过测试均不能变成成功证据。
历史测试如果自行恢复通过，不强制制造 7 次失败；计数始终来自实际运行。

## 冻结流程与自引用处理

通过全部预冻结门槛后创建并推送代码冻结提交；源码/配置与已验证环境原字节均包含
在该提交。环境文件由既有环境仓库原样复制，未重装、换版本或改 glibc。

随后从该 commit 的 Git blob 计算三份清单：frozen_upstream、frozen_analyzer、
frozen_environment。纯协议元数据放在 `artifacts/stage_f2c_freeze/`，不修改已冻结源码。
静态 `stage_f2c.yaml` 保留 PENDING 字面占位并固定 sidecar 路径；严格加载该 sidecar
后得到实际 `analyzer_freeze_commit` 和三类有效绑定，避免配置自嵌自身 SHA 的循环。
不存在 sidecar 时仍拒绝正式执行。

新 upstream 清单绑定 **F2C freeze 时当前上游字节**，不是声称当前仓库等于 F1 历史
snapshot。F1 `.gitattributes` 原字节单独通过原 Git commit 验证；旧 manifest 不更改。
所有 Analyzer 文件必须同时匹配冻结 Git blob 与当前工作区，不能遗漏源文件。
环境绑定 Dockerfile/lock/脚本 SHA、实际镜像 ID/RepoDigest、base digest、版本/CPU
信息及断网运行证据；正式前置校验再次检查实际版本、CPU 运算及仅 loopback 网络。
镜像身份由受信启动器的 Docker inspect 证据绑定，不宣称进程可自行密码学证明宿主身份。

本轮只创建/验证 freeze，不物化正式 benchmark，不调用 formal preflight/run，不运行
development/locked。后续实验必须由用户另行启动。
