from typing import Dict, Any, List
from app.schemas import SessionState

class StatusManager:
    """Manages merging and preserving structured elder status."""

    ALLOWED_FIELDS = {
        "mood", "sleep", "breakfast", "lunch", "dinner",
        "medicationTaken", "painPresent", "painLocation",
        "painSeverity", "dizziness", "fallReported"
    }

    @staticmethod
    def update_status(session: SessionState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge only supplied non-null/specified fields into session.elderStatus.
        Never overwrite unrelated fields with null.
        """
        if not updates or not isinstance(updates, dict):
            return session.elderStatus

        for key, value in updates.items():
            if key in StatusManager.ALLOWED_FIELDS and value is not None:
                session.elderStatus[key] = value

        return session.elderStatus
