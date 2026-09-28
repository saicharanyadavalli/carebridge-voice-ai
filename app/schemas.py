from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ElderStatus(BaseModel):
    mood: Optional[str] = None
    sleep: Optional[str] = None
    breakfast: Optional[Any] = None  # bool or string describing it
    lunch: Optional[Any] = None
    dinner: Optional[Any] = None
    medicationTaken: Optional[bool] = None
    painPresent: Optional[bool] = None
    painLocation: Optional[str] = None
    painSeverity: Optional[str] = None
    dizziness: Optional[bool] = None
    fallReported: Optional[bool] = None

class MemoryItem(BaseModel):
    category: str = "general"
    value: str
    importance: str = "medium"

class ConcernInfo(BaseModel):
    level: str = "none"  # "none" | "low" | "medium" | "high"
    reason: Optional[str] = None
    action: Optional[str] = None
    timestamp: Optional[str] = None

class TranscriptItem(BaseModel):
    role: str  # "user" | "assistant"
    text: str
    timestamp: str

class SessionState(BaseModel):
    callId: str
    startedAt: str
    endedAt: Optional[str] = None
    language: str = "en"
    recentTurns: List[Dict[str, str]] = Field(default_factory=list)  # [{"role": "user"|"assistant", "text": "..."}]
    rollingSummary: str = ""
    importantMemory: List[Dict[str, Any]] = Field(default_factory=list)
    elderStatus: Dict[str, Any] = Field(default_factory=lambda: {
        "mood": None,
        "sleep": None,
        "breakfast": None,
        "lunch": None,
        "dinner": None,
        "medicationTaken": None,
        "painPresent": None,
        "painLocation": None,
        "painSeverity": None,
        "dizziness": None,
        "fallReported": None
    })
    activeConcerns: List[Dict[str, Any]] = Field(default_factory=list)
    transcript: List[Dict[str, str]] = Field(default_factory=list)

class LLMTurnResponse(BaseModel):
    reply: str
    memory_updates: List[Dict[str, Any]] = Field(default_factory=list)
    status_updates: Dict[str, Any] = Field(default_factory=dict)
    concern: Dict[str, Any] = Field(default_factory=lambda: {"level": "none", "reason": None, "action": None})

class FinalSummaryResponse(BaseModel):
    summary: str
    overall_status: str  # "stable" | "attention_needed" | "urgent"
    key_points: List[str] = Field(default_factory=list)
    elder_status: Dict[str, Any] = Field(default_factory=dict)
    concerns: List[Dict[str, Any]] = Field(default_factory=list)
    caregiver_actions: List[str] = Field(default_factory=list)
