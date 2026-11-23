import os
import re
import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from app.main import app

REQUIRED_DOM_IDS = [
    "btnLang",
    "btnTheme",
    "btnMute",
    "bedsideClock",
    "errorBanner",
    "voiceOrb",
    "stateBadge",
    "stateText",
    "activeSpeechText",
    "btnStart",
    "btnEnd",
    "textChatForm",
    "manualInput",
    "btnSendText",
    "transcriptList",
    "alertsContainer",
    "statusIndicators",
    "summaryModal",
    "sumOverallStatus",
    "sumOverview",
    "sumKeyPoints",
    "sumConcerns",
    "sumCaregiverActions",
]

def test_html_contains_all_required_dom_ids():
    html_path = os.path.join("static", "index.html")
    assert os.path.exists(html_path), "static/index.html must exist"
    
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    soup = BeautifulSoup(content, "html.parser")
    missing_ids = []
    for elem_id in REQUIRED_DOM_IDS:
        if not soup.find(id=elem_id):
            missing_ids.append(elem_id)
            
    assert not missing_ids, f"static/index.html is missing required DOM IDs: {missing_ids}"

def test_css_contains_direction_3_design_tokens():
    css_path = os.path.join("static", "style.css")
    assert os.path.exists(css_path), "static/style.css must exist"
    
    with open(css_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    required_tokens = [
        "--primary-sage",
        "--accent-terracotta",
        "--canvas-bg",
        "--text-ink",
        "voice-orb",
        "state-LISTENING",
        "state-SPEAKING",
        "state-THINKING"
    ]
    missing_tokens = [tok for tok in required_tokens if tok not in content]
    assert not missing_tokens, f"static/style.css is missing Direction 3 design tokens: {missing_tokens}"

def test_backend_api_endpoints_exist():
    client = TestClient(app)
    
    # 1. Health endpoint
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json().get("status") == "ok"
    
    # 2. Session Start endpoint
    start_resp = client.post("/api/session/start", json={"language": "en"})
    assert start_resp.status_code == 200
    data = start_resp.json()
    assert "callId" in data
    assert "greeting" in data
    call_id = data["callId"]
    
    # 3. Chat endpoint
    chat_resp = client.post("/api/chat", json={"callId": call_id, "message": "Good morning"})
    assert chat_resp.status_code == 200
    assert "reply" in chat_resp.json()
    
    # 4. Session End endpoint
    end_resp = client.post("/api/session/end", json={"callId": call_id})
    assert end_resp.status_code == 200
    assert "report" in end_resp.json()
