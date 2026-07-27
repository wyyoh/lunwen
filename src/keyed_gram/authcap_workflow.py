"""F2 workflow-aware compiler 与可重放检测状态。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .authcap_atoms import AtomizedAuthority
from .authcap_compiler import (
    AuthCapCompiler,
    CompiledAllow,
    CompiledDeny,
    CompiledResult,
    TrustedRequestEnvelope,
)
from .authcap_oracle import evaluate_policy
from .authcap_policy import PolicySet, canonical_sha256
from .authcap_types import PolicyRequestContext, SemanticProposal


class WorkflowAuthorityError(ValueError):
    """workflow state 发生重放、分叉、乱序或组合越权。"""


@dataclass(frozen=True, order=True)
class EffectAtom:
    resource_id: str
    action: str
    relation_id: str | None
    purpose: str | None

    def canonical(self) -> dict[str, str | None]:
        return {
            "resource_id": self.resource_id,
            "action": self.action,
            "relation_id": self.relation_id,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class WorkflowAuthorityState:
    workflow_id: str
    subject_id: str
    policy_epoch: int
    completed_steps: tuple[str, ...]
    cumulative_authority: AtomizedAuthority
    cumulative_effects: tuple[EffectAtom, ...]
    state_digest: str

    def material(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "subject_id": self.subject_id,
            "policy_epoch": self.policy_epoch,
            "completed_steps": list(self.completed_steps),
            "cumulative_authority": self.cumulative_authority.canonical(),
            "cumulative_effects": [
                item.canonical() for item in self.cumulative_effects
            ],
        }

    def validate_digest(self) -> None:
        if canonical_sha256(self.material()) != self.state_digest:
            raise WorkflowAuthorityError("workflow state digest mutation")

    @classmethod
    def create(
        cls,
        *,
        workflow_id: str,
        subject_id: str,
        policy_epoch: int,
        completed_steps: tuple[str, ...] = (),
        cumulative_authority: AtomizedAuthority | None = None,
        cumulative_effects: tuple[EffectAtom, ...] = (),
    ) -> WorkflowAuthorityState:
        authority = cumulative_authority or AtomizedAuthority.empty()
        material = {
            "workflow_id": workflow_id,
            "subject_id": subject_id,
            "policy_epoch": policy_epoch,
            "completed_steps": list(completed_steps),
            "cumulative_authority": authority.canonical(),
            "cumulative_effects": [
                item.canonical() for item in cumulative_effects
            ],
        }
        return cls(
            workflow_id=workflow_id,
            subject_id=subject_id,
            policy_epoch=policy_epoch,
            completed_steps=completed_steps,
            cumulative_authority=authority,
            cumulative_effects=cumulative_effects,
            state_digest=canonical_sha256(material),
        )


class WorkflowStateStore:
    """原型级原子 state head；防止同一 state replay 或 fork。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heads: dict[str, str] = {}
        self._consumed: set[str] = set()

    def initialize(self, state: WorkflowAuthorityState) -> None:
        state.validate_digest()
        with self._lock:
            if state.workflow_id in self._heads:
                raise WorkflowAuthorityError("workflow 已初始化")
            self._heads[state.workflow_id] = state.state_digest

    def reserve(self, state: WorkflowAuthorityState) -> None:
        """在 capability 签发前原子消费当前 state。"""

        state.validate_digest()
        with self._lock:
            if state.state_digest in self._consumed:
                raise WorkflowAuthorityError("workflow state replay")
            if self._heads.get(state.workflow_id) != state.state_digest:
                raise WorkflowAuthorityError("workflow state fork/cross-workflow")
            self._consumed.add(state.state_digest)
            self._heads[state.workflow_id] = f"reserved:{state.state_digest}"

    def finish(
        self,
        previous: WorkflowAuthorityState,
        following: WorkflowAuthorityState,
    ) -> None:
        following.validate_digest()
        with self._lock:
            if previous.workflow_id != following.workflow_id:
                raise WorkflowAuthorityError("cross-workflow state")
            if self._heads.get(previous.workflow_id) != (
                f"reserved:{previous.state_digest}"
            ):
                raise WorkflowAuthorityError("workflow reservation 丢失")
            self._heads[previous.workflow_id] = following.state_digest

    def validate_head(self, state: WorkflowAuthorityState) -> None:
        state.validate_digest()
        with self._lock:
            if (
                state.state_digest in self._consumed
                or self._heads.get(state.workflow_id) != state.state_digest
            ):
                raise WorkflowAuthorityError("workflow state replay/fork")


class WorkflowAuthCapCompiler:
    """在单步 non-amplification 之外验证组合 policy decision。"""

    def __init__(
        self,
        compiler: AuthCapCompiler,
        state_store: WorkflowStateStore,
    ) -> None:
        self._compiler = compiler
        self._states = state_store

    def compile_next(
        self,
        *,
        state: WorkflowAuthorityState,
        step_id: str,
        step_number: int,
        expected_workflow_id: str,
        proposal: SemanticProposal,
        trusted_context: PolicyRequestContext,
        trusted_request: TrustedRequestEnvelope,
        policy_set: PolicySet,
        resource_tenants: dict[str, str],
        issued_at: str,
        combined_context: PolicyRequestContext | None = None,
    ) -> tuple[CompiledResult, WorkflowAuthorityState | None]:
        try:
            if (
                state.workflow_id != expected_workflow_id
                or
                state.subject_id != trusted_context.principal.subject_id
                or state.policy_epoch != trusted_context.policy_epoch
            ):
                raise WorkflowAuthorityError("workflow subject/epoch mismatch")
            if step_number != len(state.completed_steps) + 1:
                raise WorkflowAuthorityError("workflow step 乱序")
            if step_id in state.completed_steps:
                raise WorkflowAuthorityError("workflow step replay")
            self._states.reserve(state)
        except WorkflowAuthorityError:
            return CompiledDeny("invalid_workflow_state"), None
        if combined_context is not None:
            combined = evaluate_policy(combined_context, policy_set)
            if combined.decision != "allow":
                return CompiledDeny("combined_policy_deny"), None
        result = self._compiler.compile(
            proposal=proposal,
            trusted_context=trusted_context,
            trusted_request=trusted_request,
            policy_set=policy_set,
            resource_tenants=resource_tenants,
            issued_at=issued_at,
            workflow_id=state.workflow_id,
            workflow_step=step_number,
            previous_workflow_state_digest=state.state_digest,
        )
        if not isinstance(result, CompiledAllow):
            return result, None
        atoms = tuple(sorted(result.executable_authority.atoms))
        effects = tuple(
            EffectAtom(
                resource_id=item.resource_id,
                action=item.action,
                relation_id=item.relation_id,
                purpose=item.purpose,
            )
            for item in atoms
        )
        following = WorkflowAuthorityState.create(
            workflow_id=state.workflow_id,
            subject_id=state.subject_id,
            policy_epoch=state.policy_epoch,
            completed_steps=state.completed_steps + (step_id,),
            cumulative_authority=state.cumulative_authority.union(
                result.executable_authority
            ),
            cumulative_effects=tuple(
                sorted(set(state.cumulative_effects) | set(effects))
            ),
        )
        try:
            self._states.finish(state, following)
        except WorkflowAuthorityError:
            return CompiledDeny("workflow_state_replay_or_fork"), None
        return result, following
