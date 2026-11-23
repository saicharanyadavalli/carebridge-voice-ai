from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.schemas import SessionState
from app.state.reducer import StateReducer

class ConcernManager:
    """Classifies, validates, and manages active elder safety concerns via StateReducer."""

    VALID_LEVELS = {"none", "low", "medium", "high"}

    @staticmethod
    def process_concern(session: SessionState, concern_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Validates concern classification from LLM and stores in active concerns if level is low, medium, or high.
        Applies safety rules (e.g. falls, severe pain, unresponsiveness get flagged high).
        """
        if not concern_data or not isinstance(concern_data, dict):
            return None

        level = str(concern_data.get("level", "none")).lower().strip()
        if level not in ConcernManager.VALID_LEVELS:
            level = "none"

        reason = concern_data.get("reason")
        action = concern_data.get("action")

        # Fall verification rule
        if session.elderStatus.get("fallReported") is True:
            level = "high"
            if not reason:
                reason = "Fall reported by elder."
            if not action:
                action = "Check on elder immediately and verify emergency assistance."

        if level == "none":
            return None

        return StateReducer._upsert_concern(
            session=session,
            level=level,
            reason=reason,
            action=action
        )
