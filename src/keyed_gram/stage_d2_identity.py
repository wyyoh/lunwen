"""D2 trusted mock identity provider；不模拟真实 IAM。"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field

from .stage_d2_contract import (
    AuthenticatedPrincipal,
    D2Error,
    D2RejectionReason,
    create_authenticated_principal,
    validate_d2_identifier,
)


@dataclass(frozen=True, slots=True)
class MockSession:
    session_id: str
    subject_id: str
    authentication_context: str

    def __post_init__(self) -> None:
        validate_d2_identifier(self.session_id, label="session_id")
        validate_d2_identifier(self.subject_id, label="session subject_id")
        validate_d2_identifier(
            self.authentication_context,
            label="session authentication_context",
        )


@dataclass(slots=True)
class TrustedMockIdentityProvider:
    """凭 opaque credential 返回 sealed principal，credential 不持久化。"""

    _sessions: dict[bytes, MockSession] = field(
        default_factory=dict, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def create_session(
        self,
        subject_id: str,
        *,
        authentication_context: str = "mock-mfa",
    ) -> bytes:
        session = MockSession(
            session_id=f"session-{secrets.token_hex(16)}",
            subject_id=subject_id,
            authentication_context=authentication_context,
        )
        credential = secrets.token_bytes(32)
        with self._lock:
            while credential in self._sessions:
                credential = secrets.token_bytes(32)
            self._sessions[credential] = session
        return credential

    def authenticate(self, credential: bytes) -> AuthenticatedPrincipal:
        if not isinstance(credential, bytes) or len(credential) != 32:
            raise D2Error(
                D2RejectionReason.INVALID_SESSION,
                "session credential 必须是 32-byte opaque value",
            )
        with self._lock:
            session = self._sessions.get(credential)
        if session is None:
            raise D2Error(
                D2RejectionReason.INVALID_SESSION,
                "session credential 无效",
            )
        return create_authenticated_principal(
            subject_id=session.subject_id,
            session_id=session.session_id,
            authentication_context=session.authentication_context,
        )

    @property
    def credential_material_persisted(self) -> bool:
        return False
