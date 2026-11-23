import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.session_store import session_store

client = TestClient(app)

def test_simulate_complete_english_conversation_flow():
    """Simulates a full end-to-end English conversation from start to wrap-up to summary."""
    # 1. Start Session in English
    start_resp = client.post("/api/session/start", json={"language": "en"})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    call_id = start_data["callId"]
    assert "CareBridge" in start_data["greeting"]

    # 2. Turn 1: Medication inquiry answered
    r1 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "Yes, I took my blood pressure medicine with water."
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["elderStatus"]["medicationTaken"] is True
    assert d1["call_ended"] is False

    # 3. Turn 2: Breakfast inquiry answered
    r2 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "Yes, I had oatmeal and a banana for breakfast."
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["elderStatus"]["breakfast"] is True
    assert d2["call_ended"] is False

    # 4. Turn 3: User mentions pain in knee
    r3 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "My left knee has been hurting a bit when I walk."
    })
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["elderStatus"]["painPresent"] is True
    assert "knee" in str(d3["elderStatus"]["painLocation"]).lower()

    # 5. Turn 4: User wraps up after routine topics
    # Simulate assistant asked: "Is there anything else you would like to discuss or need help with today?"
    session = session_store.get_session(call_id)
    session.recentTurns.append({
        "role": "assistant",
        "text": "Is there anything else you would like to discuss or need help with today?"
    })

    r4 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "No, that's everything for today. Thank you so much!"
    })
    assert r4.status_code == 200
    d4 = r4.json()
    assert d4["call_ended"] is True
    assert "thank you" in d4["reply"].lower() or "goodbye" in d4["reply"].lower()

    # 6. End Session and generate summary
    end_resp = client.post("/api/session/end", json={"callId": call_id})
    assert end_resp.status_code == 200
    end_data = end_resp.json()
    report = end_data.get("report", end_data)
    assert "summary" in report
    assert "overall_status" in report
    assert "key_points" in report
    assert "caregiver_actions" in report


def test_simulate_complete_hindi_conversation_flow():
    """Simulates a full end-to-end Hindi conversation from start to wrap-up to summary."""
    # 1. Start Session in Hindi
    start_resp = client.post("/api/session/start", json={"language": "hi"})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    call_id = start_data["callId"]
    assert "नमस्ते" in start_data["greeting"]

    # 2. Turn 1: Elder confirms medication in Hindi
    r1 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "हाँ, मैंने अपनी सुबह की दवाई ले ली है।"
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["elderStatus"]["medicationTaken"] is True

    # 3. Turn 2: Elder mentions knee pain in Hindi
    r2 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "आज मेरे घुटने में थोड़ा दर्द हो रहा है।"
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["elderStatus"]["painPresent"] is True
    assert "knee" in str(d2["elderStatus"]["painLocation"]).lower()

    # 4. Turn 3: Simulate Wrap-up inquiry
    session = session_store.get_session(call_id)
    session.recentTurns.append({
        "role": "assistant",
        "text": "क्या आप किसी और चीज़ के बारे में बात करना चाहते हैं, या आपको किसी और सहायता की आवश्यकता है?"
    })

    r3 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "नहीं, कुछ नहीं। धन्यवाद।"
    })
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["call_ended"] is True
    assert "धन्यवाद" in d3["reply"] or "नमस्ते" in d3["reply"]

    # 5. End Session & generate summary
    end_resp = client.post("/api/session/end", json={"callId": call_id})
    assert end_resp.status_code == 200
    end_data = end_resp.json()
    report = end_data.get("report", end_data)
    assert "summary" in report
    assert report["overall_status"] in ["stable", "attention_needed", "urgent"]
