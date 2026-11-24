import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.schemas import SessionState
from app.state.events import SessionEvent

class StateReducer:
    """
    Deterministic State Reducer:
    Reconciles event streams into canonical elder status, resolved memories, and active concerns.
    Replaces uncontrolled array appending with temporal and semantic resolution.
    """

    ALLOWED_STATUS_FIELDS = {
        "mood", "sleep", "breakfast", "lunch", "dinner",
        "medicationTaken", "painPresent", "painLocation",
        "painSeverity", "dizziness", "fallReported", "hydration"
    }

    @staticmethod
    def apply_event(session: SessionState, event: SessionEvent) -> None:
        """Applies an individual event to mutate the session state deterministically."""
        # Record event in event store if tracked
        if not hasattr(session, "events") or session.events is None:
            session.events = []
        session.events.append(event.model_dump())

        etype = event.event_type
        payload = event.payload

        if etype == "SLOT_UPDATED":
            for field, val in payload.items():
                if field in StateReducer.ALLOWED_STATUS_FIELDS and val is not None:
                    session.elderStatus[field] = val

        elif etype == "SYMPTOM_REPORTED":
            if "painPresent" in payload:
                session.elderStatus["painPresent"] = payload["painPresent"]
            if "painLocation" in payload and payload["painLocation"]:
                session.elderStatus["painLocation"] = payload["painLocation"]
            if "painSeverity" in payload and payload["painSeverity"]:
                session.elderStatus["painSeverity"] = payload["painSeverity"]
            if "dizziness" in payload:
                session.elderStatus["dizziness"] = payload["dizziness"]

        elif etype == "EMERGENCY_TRIGGERED":
            session.elderStatus["fallReported"] = True
            # Escalate an emergency concern
            StateReducer._upsert_concern(
                session,
                level="high",
                reason=payload.get("reason", "Immediate emergency or fall reported"),
                action=payload.get("action", "Contact emergency services or caregiver immediately")
            )

        elif etype == "MEMORY_ACQUIRED":
            StateReducer._merge_memory(
                session,
                category=payload.get("category", "general"),
                value=payload.get("value", ""),
                importance=payload.get("importance", "medium")
            )

        elif etype == "CONCERN_RECORDED":
            StateReducer._upsert_concern(
                session,
                level=payload.get("level", "none"),
                reason=payload.get("reason"),
                action=payload.get("action")
            )

        elif etype == "CONCERN_RESOLVED":
            target_reason = payload.get("reason", "").lower().strip()
            if not target_reason:
                return
            session.activeConcerns = [
                c for c in session.activeConcerns 
                if target_reason not in str(c.get("reason", "")).lower()
            ]

    @staticmethod
    def _merge_memory(session: SessionState, category: str, value: str, importance: str) -> None:
        val_clean = str(value).strip()
        if not val_clean:
            return
        cat_clean = str(category).strip().lower()
        imp_clean = str(importance).strip().lower()

        # Check for matching existing memory
        for existing in session.importantMemory:
            existing_val = str(existing.get("value", "")).strip().lower()
            existing_cat = str(existing.get("category", "")).strip().lower()

            # Exact match or semantic overlap
            if existing_cat == cat_clean and (existing_val == val_clean.lower() or val_clean.lower() in existing_val):
                if imp_clean == "high":
                    existing["importance"] = "high"
                return

        session.importantMemory.append({
            "category": cat_clean,
            "value": val_clean,
            "importance": imp_clean
        })

    @staticmethod
    def _upsert_concern(session: SessionState, level: str, reason: Optional[str], action: Optional[str]) -> Optional[Dict[str, Any]]:
        lvl = str(level).lower().strip()
        if lvl not in {"low", "medium", "high"}:
            return None

        # Fall verification rule: fall is always high
        if session.elderStatus.get("fallReported") is True:
            lvl = "high"

        reason_text = reason or "Health or safety concern noted"
        action_text = action or "Monitor elder condition"

        # Update existing matching concern if present
        for c in session.activeConcerns:
            if c.get("reason", "").lower() == reason_text.lower():
                c["level"] = lvl
                c["action"] = action_text
                return c

        new_concern = {
            "level": lvl,
            "reason": reason_text,
            "action": action_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        session.activeConcerns.append(new_concern)
        return new_concern

    @staticmethod
    def batch_reduce(
        session: SessionState,
        status_updates: Optional[Dict[str, Any]] = None,
        memory_updates: Optional[List[Dict[str, Any]]] = None,
        concern_update: Optional[Dict[str, Any]] = None,
        turn_index: int = 0
    ) -> None:
        """Helper to apply a batch of updates as discrete, traceable events."""
        call_id = session.callId

        if status_updates:
            event = SessionEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                call_id=call_id,
                turn_index=turn_index,
                event_type="SLOT_UPDATED",
                payload=status_updates
            )
            StateReducer.apply_event(session, event)

        if memory_updates:
            for mem in memory_updates:
                if isinstance(mem, dict):
                    event = SessionEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:8]}",
                        call_id=call_id,
                        turn_index=turn_index,
                        event_type="MEMORY_ACQUIRED",
                        payload=mem
                    )
                    StateReducer.apply_event(session, event)

        if concern_update and isinstance(concern_update, dict):
            lvl = concern_update.get("level", "none")
            if lvl in {"low", "medium", "high"}:
                event = SessionEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    call_id=call_id,
                    turn_index=turn_index,
                    event_type="CONCERN_RECORDED",
                    payload=concern_update
                )
                StateReducer.apply_event(session, event)
