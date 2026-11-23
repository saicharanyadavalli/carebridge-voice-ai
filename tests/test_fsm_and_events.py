import pytest
import os
import uuid
from app.session_store import SessionStore
from app.schemas import SessionState
from app.dialogue_engine.fsm import DialogueFSM
from app.state.events import SessionEvent
from app.state.reducer import StateReducer
from app.guardrails import SafetyGuardrail

def test_fsm_zero_repetition_progression():
    """Verify that DialogueFSM deterministically advances and never repeats asked topics."""
    session = SessionState(callId="test_fsm_1", startedAt="2026-11-22T00:00:00Z", clientHour=9)
    
    # 1. In the morning, medication is the first target
    topic1, q1_en, _ = DialogueFSM.determine_next_objective(session)
    assert topic1 == "medicationTaken"
    assert "medication" in q1_en.lower()
    
    # Mark medication as asked by adding assistant turn
    session.recentTurns.append({"role": "assistant", "text": q1_en})
    assert "medicationTaken" in DialogueFSM.get_asked_topics(session)
    
    # 2. Next objective must be breakfast
    topic2, q2_en, _ = DialogueFSM.determine_next_objective(session)
    assert topic2 == "breakfast"
    assert "breakfast" in q2_en.lower()
    
    # Mark breakfast as answered
    session.elderStatus["breakfast"] = True
    session.recentTurns.append({"role": "assistant", "text": q2_en})
    
    # 3. Next objective must be sleep
    topic3, q3_en, _ = DialogueFSM.determine_next_objective(session)
    assert topic3 == "sleep"

def test_fsm_time_of_day_meals_and_greetings():
    """Verify time-of-day contextual meals (breakfast, lunch, dinner) and greetings in EN & HI."""
    # Morning (9 AM)
    session_morn = SessionState(callId="morn", startedAt="2026-11-22T09:00:00Z", clientHour=9)
    assert DialogueFSM.get_time_period(9) == "morning"
    g_en = DialogueFSM.get_initial_greeting("en", 9)
    g_hi = DialogueFSM.get_initial_greeting("hi", 9)
    assert "morning" in g_en.lower()
    assert "शुभ प्रभात" in g_hi
    seq_morn = DialogueFSM.get_protocol_sequence(session_morn)
    assert "breakfast" in seq_morn

    # Afternoon (1 PM)
    session_aft = SessionState(callId="aft", startedAt="2026-11-22T13:00:00Z", clientHour=13)
    assert DialogueFSM.get_time_period(13) == "afternoon"
    g_en = DialogueFSM.get_initial_greeting("en", 13)
    g_hi = DialogueFSM.get_initial_greeting("hi", 13)
    assert "afternoon" in g_en.lower() and "lunch" in g_en.lower()
    assert "दोपहर का खाना" in g_hi
    seq_aft = DialogueFSM.get_protocol_sequence(session_aft)
    assert seq_aft[0] == "lunch"

    # Evening (7 PM)
    session_eve = SessionState(callId="eve", startedAt="2026-11-22T19:00:00Z", clientHour=19)
    assert DialogueFSM.get_time_period(19) == "evening"
    g_en = DialogueFSM.get_initial_greeting("en", 19)
    g_hi = DialogueFSM.get_initial_greeting("hi", 19)
    assert "evening" in g_en.lower() and "dinner" in g_en.lower()
    assert "शुभ संध्या" in g_hi and "रात का खाना" in g_hi
    seq_eve = DialogueFSM.get_protocol_sequence(session_eve)
    assert seq_eve[0] == "dinner"

    # Night (11 PM)
    assert DialogueFSM.get_time_period(23) == "night"
    g_en = DialogueFSM.get_initial_greeting("en", 23)
    g_hi = DialogueFSM.get_initial_greeting("hi", 23)
    assert "tonight" in g_en.lower()
    assert "आज रात" in g_hi

def test_fsm_prioritizes_symptom_investigation():
    """If pain is reported but location is unknown, FSM must prioritize investigating it."""
    session = SessionState(callId="test_fsm_pain", startedAt="2026-11-22T00:00:00Z")
    session.elderStatus["painPresent"] = True
    session.elderStatus["painLocation"] = None
    
    topic, q_en, _ = DialogueFSM.determine_next_objective(session)
    assert topic == "pain_investigation"
    assert "where does the pain bother you" in q_en.lower()

def test_safety_guardrails_sub_millisecond_interception():
    """Safety guardrail must immediately intercept falls and acute chest pain without LLM."""
    session = SessionState(callId="test_safety", startedAt="2026-11-22T00:00:00Z")
    
    # Test Fall
    fall_res = SafetyGuardrail.check_emergency("I tripped over the rug and fell down, I can't get up", session)
    assert fall_res is not None
    assert fall_res["triggered"] is True
    assert fall_res["status_updates"]["fallReported"] is True
    assert fall_res["concern"]["level"] == "high"
    assert "remain completely still" in fall_res["reply"]
    
    # Test Acute Chest Pain
    chest_res = SafetyGuardrail.check_emergency("I'm feeling severe chest pain and pressure", session)
    assert chest_res is not None
    assert chest_res["triggered"] is True
    assert chest_res["status_updates"]["painLocation"] == "chest"
    assert chest_res["concern"]["level"] == "high"

def test_state_reducer_event_sourcing_and_conflict_resolution():
    """State reducer must record immutable events and resolve state accurately."""
    session = SessionState(callId="test_reducer", startedAt="2026-11-22T00:00:00Z")
    
    # Apply slot event: medication taken
    evt1 = SessionEvent(
        event_id="evt_1",
        call_id=session.callId,
        turn_index=1,
        event_type="SLOT_UPDATED",
        payload={"medicationTaken": True, "sleep": "good"}
    )
    StateReducer.apply_event(session, evt1)
    assert session.elderStatus["medicationTaken"] is True
    assert session.elderStatus["sleep"] == "good"
    assert len(session.events) == 1
    
    # Correction: elder later says they actually forgot
    evt2 = SessionEvent(
        event_id="evt_2",
        call_id=session.callId,
        turn_index=3,
        event_type="SLOT_UPDATED",
        payload={"medicationTaken": False}
    )
    StateReducer.apply_event(session, evt2)
    assert session.elderStatus["medicationTaken"] is False  # updated cleanly
    assert session.elderStatus["sleep"] == "good"            # preserved
    assert len(session.events) == 2

def test_session_store_sqlite_persistence(tmp_path):
    """SessionStore must persist sessions in SQLite and restore them upon reload."""
    db_file = os.path.join(tmp_path, "test_store.db")
    store1 = SessionStore(db_path=db_file)
    
    session1 = store1.create_session()
    call_id = session1.callId
    session1.elderStatus["mood"] = "cheerful"
    session1.elderStatus["breakfast"] = True
    session1.transcript.append({"role": "user", "text": "I had oatmeal", "timestamp": "2026-11-22T00:00:00Z"})
    store1.save_session(session1)
    
    # Simulate a separate process or server restart with a fresh store instance
    store2 = SessionStore(db_path=db_file)
    reloaded = store2.get_session(call_id)
    
    assert reloaded is not None
    assert reloaded.callId == call_id
    assert reloaded.elderStatus["mood"] == "cheerful"
    assert reloaded.elderStatus["breakfast"] is True
    assert len(reloaded.transcript) == 1
    assert reloaded.transcript[0]["text"] == "I had oatmeal"
