from datetime import datetime, timezone
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

EventType = Literal[
    "SLOT_UPDATED",
    "SYMPTOM_REPORTED",
    "EMERGENCY_TRIGGERED",
    "MEMORY_ACQUIRED",
    "CONCERN_RECORDED",
    "CONCERN_RESOLVED",
    "TURN_RECORDED"
]

class SessionEvent(BaseModel):
    event_id: str
    call_id: str
    turn_index: int
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload: Dict[str, Any] = Field(default_factory=dict)
