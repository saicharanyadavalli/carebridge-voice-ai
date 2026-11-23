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

def test_hindi_context_directive():
    store = SessionStore()
    session = store.create_session()
    session.language = "hi"
    context = ContextManager.build_compact_context(session, "नमस्ते, मैं ठीक हूँ।")
    assert "LANGUAGE DIRECTIVE — HIGH QUALITY HINDI" in context[0]["content"]
    assert "Devanagari" in context[0]["content"]

@pytest.mark.asyncio
async def test_gemini_key_rotation(monkeypatch):
    from app.llm_service import LLMService
    from app.config import config
    import httpx

    from app.config import Config

    # Mock GEMINI_API_KEYS property to return 3 test keys
    monkeypatch.setattr(Config, "GEMINI_API_KEYS", property(lambda self: ["fake_key_1", "fake_key_2", "fake_key_3"]))

    attempts = []

    async def mock_post(self, url, *args, **kwargs):
        attempts.append(str(url))
        if "fake_key_1" in str(url):
            return httpx.Response(429, text="Rate limit exceeded", request=httpx.Request("POST", str(url)))
        elif "fake_key_2" in str(url):
            return httpx.Response(500, text="Internal server error", request=httpx.Request("POST", str(url)))
        elif "fake_key_3" in str(url):
            json_body = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"reply": "नमस्ते जी, आप कैसे हैं?", "memory_updates": [], "status_updates": {}, "concern": {"level": "none", "reason": null, "action": null}}'
                        }]
                    }
                }]
            }
            return httpx.Response(200, json=json_body, request=httpx.Request("POST", str(url)))
        return httpx.Response(404, request=httpx.Request("POST", str(url)))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    messages = [{"role": "system", "content": "You are CareBridge"}, {"role": "user", "content": "Hello"}]
    res = await LLMService._call_gemini(messages)
    assert res is not None
    assert "नमस्ते जी" in res["reply"]
    # Check that all 3 keys were attempted in order
    assert len(attempts) == 3
    assert "fake_key_1" in attempts[0]
    assert "fake_key_2" in attempts[1]
    assert "fake_key_3" in attempts[2]

@pytest.mark.asyncio
async def test_full_fallback_hierarchy(monkeypatch):
    from app.llm_service import LLMService
    from app.session_store import SessionStore

    # Mock all remote calls to fail (simulating rate-limits / expired keys / network downtime)
    async def mock_fail(*args, **kwargs):
        return None

    monkeypatch.setattr(LLMService, "_call_gemini", mock_fail)
    monkeypatch.setattr(LLMService, "_call_openrouter", mock_fail)
    monkeypatch.setattr(LLMService, "_call_nvidia", mock_fail)

    store = SessionStore()
    session = store.create_session()

    # User reports pain
    messages = [
        {"role": "assistant", "content": "Hello! How are you feeling today?"},
        {"role": "user", "content": "My knee is hurting badly today."}
    ]

    result = await LLMService.chat_turn(messages, session=session)
    assert result is not None
    assert "knee" in result["reply"].lower()
    assert result["status_updates"].get("painPresent") is True
    assert result["status_updates"].get("painLocation") == "knee"
    assert result["concern"].get("level") == "low"

@pytest.mark.asyncio
async def test_wrapup_and_graceful_termination(monkeypatch):
    from app.llm_service import LLMService
    from app.session_store import SessionStore
    from app.dialogue_engine.fsm import DialogueFSM

    async def mock_fail(*args, **kwargs):
        return None

    monkeypatch.setattr(LLMService, "_call_gemini", mock_fail)
    monkeypatch.setattr(LLMService, "_call_openrouter", mock_fail)
    monkeypatch.setattr(LLMService, "_call_nvidia", mock_fail)

    store = SessionStore()
    session = store.create_session()

    # Pre-fill session with all routine checks already completed
    session.elderStatus = {
        "medicationTaken": True,
        "breakfast": True,
        "lunch": True,
        "sleep": "good",
        "painPresent": False,
        "painLocation": None,
        "painSeverity": None,
        "dizziness": False,
        "fallReported": False,
        "hydration": True,
        "mood": "calm"
    }

    # Simulate assistant just asked wrap up inquiry
    session.recentTurns = [
        {"role": "assistant", "text": "Is there anything else you would like to discuss or need help with today?"}
    ]

    # Elder says they have nothing else
    messages = [
        {"role": "assistant", "content": "Is there anything else you would like to discuss or need help with today?"},
        {"role": "user", "content": "No, that's all. Thank you."}
    ]

    res = await LLMService.chat_turn(messages, session=session)
    assert res is not None
    assert res.get("call_ended") is True
    assert "thank you" in res["reply"].lower() or "goodbye" in res["reply"].lower()



