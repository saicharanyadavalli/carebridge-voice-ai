import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from app.schemas import SessionState

class SessionStore:
    """Thread-safe in-memory store for active and completed CareBridge call sessions."""
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def create_session(self) -> SessionState:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        session = SessionState(
            callId=call_id,
            startedAt=now_iso
        )
        self._sessions[call_id] = session
        return session

    def get_session(self, call_id: str) -> Optional[SessionState]:
        return self._sessions.get(call_id)

    def end_session(self, call_id: str) -> Optional[SessionState]:
        session = self._sessions.get(call_id)
        if session and not session.endedAt:
            session.endedAt = datetime.now(timezone.utc).isoformat()
        return session

    def delete_session(self, call_id: str) -> bool:
        if call_id in self._sessions:
            del self._sessions[call_id]
            return True
        return False

# Global singleton session store
session_store = SessionStore()
