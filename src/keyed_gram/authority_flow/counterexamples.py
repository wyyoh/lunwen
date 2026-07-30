"""F2A novelty gate 使用的最小区分性反例注册表。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Counterexample:
    case_id: str
    title: str
    minimal_trace: tuple[str, ...]
    violated_property: str
    plain_data_ifc: str
    ordinary_capability: str
    tracked_capability: str
    branch_isolation: str
    required_dual_flow_rule: str
    not_plain_taint_only: bool
    not_token_subset_only: bool

    def canonical(self) -> dict[str, object]:
        return {
            "branch_isolation": self.branch_isolation,
            "case_id": self.case_id,
            "minimal_trace": list(self.minimal_trace),
            "not_plain_taint_only": self.not_plain_taint_only,
            "not_token_subset_only": self.not_token_subset_only,
            "ordinary_capability": self.ordinary_capability,
            "plain_data_ifc": self.plain_data_ifc,
            "required_dual_flow_rule": self.required_dual_flow_rule,
            "title": self.title,
            "tracked_capability": self.tracked_capability,
            "violated_property": self.violated_property,
        }


def counterexample_registry() -> tuple[Counterexample, ...]:
    """返回预注册的六个最小程序族。

    结论只针对“单独使用”的基线机制；并不声称 FLAM、线性授权逻辑等组合型
    既有系统原则上无法扩展来表达这些规则。
    """

    return (
        Counterexample(
            case_id="ce-authority-laundering",
            title="Authority laundering through summary",
            minimal_trace=(
                "untrusted_memory",
                "llm_summary",
                "write_trusted_memory",
                "privileged_tool",
            ),
            violated_property="authority_origination",
            plain_data_ifc=(
                "完整 taint 传播可阻止；但只按最终容器或 sanitizer 重新标记的"
                "实现无法表达 effect authority 的独立来源。"
            ),
            ordinary_capability=(
                "可阻止无 token 调用；若 agent 已持有宽工具 token，则单次 scope"
                "检查不判断这次 effect 是否由攻击数据塑造。"
            ),
            tracked_capability=(
                "能追踪工具 effect，但若没有 data influence 与 authority origin"
                "的联合规则，合法 capability 仍可被不可信数据择取。"
            ),
            branch_isolation=(
                "可隔离污染上下文，但不定义父 authority 的来源和消费。"
            ),
            required_dual_flow_rule=(
                "数据重写只传播 InfluenceOrigin；只有 authenticated grant transition"
                "能创生 AuthorityOrigin。"
            ),
            not_plain_taint_only=True,
            not_token_subset_only=True,
        ),
        Counterexample(
            case_id="ce-capability-recombination",
            title="Capability recombination across branches",
            minimal_trace=(
                "branch_A_read_customer",
                "branch_B_send_external",
                "merge",
                "send_customer_external",
            ),
            violated_property="merge_confinement",
            plain_data_ifc=(
                "机密性 IFC 可在明确标注 customer data 时阻止泄漏，但不一般表达"
                "两个分别允许 effect 的组合 deny、预算或 purpose。"
            ),
            ordinary_capability=(
                "read 与 send token 均可逐调用验证为合法；token subset 不包含"
                "历史相关的组合 effect 禁止。"
            ),
            tracked_capability=(
                "非线性 tracked capability 可同时存在并合法调用；需额外组合 effect"
                "与线性来源规则。"
            ),
            branch_isolation=(
                "父上下文标签可保持不变，但 authority union 仍可能重组两个合法能力。"
            ),
            required_dual_flow_rule=(
                "merge 同时检查数据标签、共同父 authority、预算守恒和 combined deny。"
            ),
            not_plain_taint_only=True,
            not_token_subset_only=True,
        ),
        Counterexample(
            case_id="ce-delegation-fork",
            title="Delegation fork duplicates a single-use grant",
            minimal_trace=(
                "parent_has_one_use",
                "delegate_full_to_child_A",
                "delegate_full_to_child_B",
                "both_commit",
            ),
            violated_property="authority_conservation",
            plain_data_ifc="数据标签不计数、消费或约束 delegation budget。",
            ordinary_capability=(
                "无全局线性消费语义的 bearer token 可被复制；有状态 single-use token"
                "能解决 use 次数，但不自动覆盖分裂、子预算和深度。"
            ),
            tracked_capability=(
                "普通 capture tracking 追踪可达 capability，不必然提供 affine ownership。"
            ),
            branch_isolation="分支隔离不阻止同一 token 被复制到多个子 Agent。",
            required_dual_flow_rule=(
                "delegate 消费父 resource；多个子预算之和不得超过父预算，深度严格衰减。"
            ),
            not_plain_taint_only=True,
            not_token_subset_only=True,
        ),
        Counterexample(
            case_id="ce-trusted-tool-echo",
            title="Trusted tool echoes attacker-controlled instruction",
            minimal_trace=(
                "untrusted_web_instruction",
                "trusted_parser_echo",
                "llm_treats_echo_as_command",
                "tool_effect",
            ),
            violated_property="non_malleable_authority",
            plain_data_ifc=(
                "正确细粒度 provenance 可保留低完整性；仅以工具身份赋 label 的"
                "粗粒度 IFC 会错误提升。"
            ),
            ordinary_capability=(
                "scope 合法时仍允许攻击数据选择具体 effect；token 不表示数据是否有"
                "资格影响 authority 的消费。"
            ),
            tracked_capability=(
                "工具 capability 的静态可达性与 echoed data 的 InfluenceOrigin 是两类对象。"
            ),
            branch_isolation="可隔离 echo，但不定义它能否影响 authority 消费。",
            required_dual_flow_rule=(
                "可信工具不能覆盖输入 InfluenceOrigin；攻击者可变数据不得改变"
                "authority trace，除非显式 elevation。"
            ),
            not_plain_taint_only=True,
            not_token_subset_only=True,
        ),
        Counterexample(
            case_id="ce-memory-fragmentation",
            title="Fragmented memory reconstructs an authority claim",
            minimal_trace=(
                "memory_fragment_1",
                "memory_fragment_2",
                "llm_merge_summary",
                "claim_user_approved",
                "commit",
            ),
            violated_property="authority_origination",
            plain_data_ifc=(
                "taint join 可保留低完整性，但不能把“用户批准”这类 authorization"
                "主张转化为可消费 grant；两者语义问题不同。"
            ),
            ordinary_capability=(
                "无 capability 时可阻止；若上下文持有 ambient/wide capability，"
                "fragment 合成仍可驱动合法范围内但未获该次授权的 effect。"
            ),
            tracked_capability=(
                "追踪可调用哪些工具，不等于追踪当前 effect 的授权来源链。"
            ),
            branch_isolation="碎片可来自同一长期父上下文，不一定经过隔离分支。",
            required_dual_flow_rule=(
                "memory/LLM 只能生成 influence；AuthorityOrigin 不可由文本 claim 合成。"
            ),
            not_plain_taint_only=True,
            not_token_subset_only=True,
        ),
        Counterexample(
            case_id="ce-integrity-is-not-authority",
            title="High-integrity fact is not an authorized command",
            minimal_trace=(
                "signed_financial_fact",
                "llm_derives_payment_instruction",
                "payment_commit",
            ),
            violated_property="data_authority_separation",
            plain_data_ifc=(
                "高完整性标签只说明来源可信；IFC 本身不表示该来源有权授权 payment。"
            ),
            ordinary_capability=(
                "支付 token 可限制 scope，但若 agent 预持 token，subset check 不证明"
                "signed fact 是本次支付的授权来源。"
            ),
            tracked_capability=(
                "静态 effect 可达性不能把事实提供者与授权签发者角色分开。"
            ),
            branch_isolation="没有污染分支，问题仍存在。",
            required_dual_flow_rule=(
                "DataLabel.integrity 与 AuthorityOrigin 正交；只有受认证 grant/elevation"
                "规则能连接二者。"
            ),
            not_plain_taint_only=True,
            not_token_subset_only=True,
        ),
    )
