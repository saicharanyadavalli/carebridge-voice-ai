import pytest
from app.session_store import SessionStore
from app.status_manager import StatusManager
from app.memory_manager import MemoryManager
from app.concern_manager import ConcernManager
from app.context_manager import ContextManager

def test_session_creation_and_retrieval():
    store = SessionStore()
    session = store.create_session()
    assert session.callId.startswith("call_")
    assert session.startedAt is not None
    assert session.endedAt is None
    
    fetched = store.get_session(session.callId)
    assert fetched is not None
    assert fetched.callId == session.callId

def test_elder_status_updates():
    store = SessionStore()
    session = store.create_session()
    
    # Update sleep and medication
    StatusManager.update_status(session, {"sleep": "poor", "medicationTaken": False})
    assert session.elderStatus["sleep"] == "poor"
    assert session.elderStatus["medicationTaken"] is False
    assert session.elderStatus["mood"] is None  # untouched
    
    # Later: medication taken becomes True without losing sleep
    StatusManager.update_status(session, {"medicationTaken": True, "painPresent": True, "painLocation": "knee"})
    assert session.elderStatus["medicationTaken"] is True
    assert session.elderStatus["sleep"] == "poor"
    assert session.elderStatus["painLocation"] == "knee"

def test_memory_manager_deduplication():
    store = SessionStore()
    session = store.create_session()
    
    memories = [
        {"category": "symptom", "value": "mild knee pain", "importance": "medium"},
        {"category": "symptom", "value": "mild knee pain", "importance": "high"}, # duplicate
        {"category": "preference", "value": "likes tea in morning", "importance": "low"}
    ]
    MemoryManager.add_memories(session, memories)
    assert len(session.importantMemory) == 2
    # Verify importance upgraded
    knee_mem = next(m for m in session.importantMemory if m["value"] == "mild knee pain")
    assert knee_mem["importance"] == "high"

def test_concern_detection_and_emergency_rules():
    store = SessionStore()
    session = store.create_session()
    
    # Fall reported triggers high level
    session.elderStatus["fallReported"] = True
    concern = ConcernManager.process_concern(session, {"level": "low", "reason": "elder tripped"})
    assert concern["level"] == "high"
    assert len(session.activeConcerns) == 1

def test_compact_context_builder():
    store = SessionStore()
    session = store.create_session()
    session.elderStatus["mood"] = "cheerful"
    session.importantMemory.append({"category": "event", "value": "grandchild visiting tomorrow", "importance": "high"})
    ContextManager.append_turn(session, "assistant", "Hello! How are you?", "2026-09-27T12:00:00Z")
    
    context = ContextManager.build_compact_context(session, "I'm doing well, thank you.")
    assert len(context) >= 3  # system instruction + assistant turn + user message
    assert "CURRENT ELDER STATUS" in context[0]["content"]
    assert "grandchild visiting tomorrow" in context[0]["content"]
