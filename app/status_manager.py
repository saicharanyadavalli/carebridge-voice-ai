from typing import Dict, Any
from app.schemas import SessionState
from app.state.reducer import StateReducer
from app.state.events import SessionEvent
import uuid

class StatusManager:
    """Manages merging and preserving structured elder status via StateReducer."""

    ALLOWED_FIELDS = StateReducer.ALLOWED_STATUS_FIELDS

    @staticmethod
    def update_status(session: SessionState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge only supplied non-null/specified fields into session.elderStatus.
        Never overwrite unrelated fields with null.
        """
        if not updates or not isinstance(updates, dict):
            return session.elderStatus

        clean_updates = {
            k: v for k, v in updates.items() 
            if k in StatusManager.ALLOWED_FIELDS and v is not None
        }

        if clean_updates:
            event = SessionEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                call_id=session.callId,
                turn_index=len(session.transcript) if session.transcript else 0,
                event_type="SLOT_UPDATED",
                payload=clean_updates
            )
            StateReducer.apply_event(session, event)

        return session.elderStatus
