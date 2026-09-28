from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.schemas import SessionState

class ConcernManager:
    """Classifies, validates, and manages active elder safety concerns."""

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

        # Backend rule verification: if fall is reported in elder status or reason
        elder_status = session.elderStatus
        if elder_status.get("fallReported") is True:
            level = "high"
            if not reason:
                reason = "Fall reported by elder."
            if not action:
                action = "Check on elder immediately and verify emergency assistance."

        if level == "none":
            return None

        concern_record = {
            "level": level,
            "reason": reason or "Elder noted concern",
            "action": action or "Monitor elder status closely",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add to active concerns if not a verbatim duplicate
        for c in session.activeConcerns:
            if c.get("reason") == concern_record["reason"] and c.get("level") == concern_record["level"]:
                return None

        session.activeConcerns.append(concern_record)
        return concern_record
