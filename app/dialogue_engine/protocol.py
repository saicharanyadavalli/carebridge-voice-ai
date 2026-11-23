from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ProtocolState(str, Enum):
    GREETING = "GREETING"
    MEDICATION_CHECK = "MEDICATION_CHECK"
    MEAL_CHECK = "MEAL_CHECK"
    SLEEP_CHECK = "SLEEP_CHECK"
    SYMPTOM_SCREENING = "SYMPTOM_SCREENING"
    SYMPTOM_INVESTIGATION = "SYMPTOM_INVESTIGATION"
    HYDRATION_CHECK = "HYDRATION_CHECK"
    WELLBEING_WRAPUP = "WELLBEING_WRAPUP"
    EMERGENCY_TRIAGE = "EMERGENCY_TRIAGE"

class DialogueSlot(BaseModel):
    name: str
    asked: bool = False
    resolved: bool = False
    value: Any = None
    asked_at_turn: Optional[int] = None
    resolved_at_turn: Optional[int] = None

class ProtocolPlan(BaseModel):
    current_state: ProtocolState = ProtocolState.MEDICATION_CHECK
    active_slot: Optional[str] = "medicationTaken"
    slots: Dict[str, DialogueSlot] = Field(default_factory=lambda: {
        "medicationTaken": DialogueSlot(name="medicationTaken"),
        "breakfast": DialogueSlot(name="breakfast"),
        "lunch": DialogueSlot(name="lunch"),
        "sleep": DialogueSlot(name="sleep"),
        "painPresent": DialogueSlot(name="painPresent"),
        "painLocation": DialogueSlot(name="painLocation"),
        "painSeverity": DialogueSlot(name="painSeverity"),
        "dizziness": DialogueSlot(name="dizziness"),
        "hydration": DialogueSlot(name="hydration"),
        "mood": DialogueSlot(name="mood")
    })
    history: List[str] = Field(default_factory=list)
