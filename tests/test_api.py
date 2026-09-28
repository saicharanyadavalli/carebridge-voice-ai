import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "provider" in data
    assert "has_openrouter_key" in data

def test_session_lifecycle_and_chat():
    # 1. Start session
    start_resp = client.post("/api/session/start", json={"language": "en"})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert "callId" in start_data
    assert "greeting" in start_data
    call_id = start_data["callId"]

    # 2. Chat turn - Medication check
    chat_resp1 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "Yes, I took my blood pressure medicine this morning."
    })
    assert chat_resp1.status_code == 200
    chat_data1 = chat_resp1.json()
    assert "reply" in chat_data1
    assert chat_data1["elderStatus"]["medicationTaken"] is True

    # 3. Chat turn - Pain / Concern detection
    chat_resp2 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "My left knee has been hurting a lot today."
    })
    assert chat_resp2.status_code == 200
    chat_data2 = chat_resp2.json()
    assert chat_data2["elderStatus"]["painPresent"] is True
    assert "knee" in str(chat_data2["elderStatus"]["painLocation"]).lower()

    # 4. State correction check
    chat_resp3 = client.post("/api/chat", json={
        "callId": call_id,
        "message": "Actually, I forgot my afternoon pills, I haven't taken them."
    })
    assert chat_resp3.status_code == 200
    chat_data3 = chat_resp3.json()
    assert chat_data3["elderStatus"]["medicationTaken"] is False

    # 5. Fetch session details
    get_resp = client.get(f"/api/session/{call_id}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["transcript"]) >= 4

    # 6. End session
    end_resp = client.post("/api/session/end", json={"callId": call_id})
    assert end_resp.status_code == 200
    end_data = end_resp.json()
    assert "report" in end_data
    assert "summary" in end_data["report"]
    assert "overall_status" in end_data["report"]

def test_api_root():
    # Verify backend API root info loads at /
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "CareBridge" in root_resp.text
    data = root_resp.json()
    assert data["status"] == "online"
    assert "endpoints" in data
