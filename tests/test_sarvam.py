import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.sarvam_service import SarvamService

client = TestClient(app)

def test_health_check_reports_sarvam():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "has_sarvam_key" in data
    assert data["has_sarvam_key"] is True
    assert "sarvam_tts_model" in data
    assert "sarvam_stt_model" in data

@pytest.mark.asyncio
async def test_sarvam_tts_live():
    res = await SarvamService.text_to_speech("Good morning, have a wonderful day.", language="en")
    assert res is not None
    assert res["status"] == "ok"
    assert "audio_base64" in res
    assert len(res["audio_base64"]) > 1000

@pytest.mark.asyncio
async def test_sarvam_tts_and_stt_roundtrip():
    # 1. Generate speech with Sarvam TTS
    tts_res = await SarvamService.text_to_speech("Hello, I am feeling very well today.", language="en")
    assert tts_res is not None
    assert "audio_base64" in tts_res
    
    # 2. Transcribe back using Sarvam STT
    import base64
    audio_bytes = base64.b64decode(tts_res["audio_base64"])
    stt_res = await SarvamService.speech_to_text(audio_bytes, language="en")
    assert stt_res is not None
    assert stt_res["status"] == "ok"
    assert "transcript" in stt_res
    assert len(stt_res["transcript"]) > 0

def test_voice_tts_endpoint():
    resp = client.post("/api/voice/tts", json={
        "text": "How are you feeling today?",
        "language": "en"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "audio_base64" in data
    assert len(data["audio_base64"]) > 500

def test_session_start_and_chat_native_speech_integration():
    # Verify /api/session/start returns clean text greeting ready for native browser speech synthesis
    start_resp = client.post("/api/session/start", json={"language": "en"})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert "greeting" in start_data
    assert len(start_data["greeting"]) > 5
    assert start_data.get("audio_base64") is None  # Offloaded to zero-latency browser speech synthesis
    
    call_id = start_data["callId"]
    
    # Verify /api/chat returns conversational reply without blocking Sarvam TTS roundtrip
    chat_resp = client.post("/api/chat", json={
        "callId": call_id,
        "message": "Yes, I took my morning medicine."
    })
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "reply" in chat_data
    assert len(chat_data["reply"]) > 5
    assert chat_data.get("audio_base64") is None

def test_session_start_and_chat_hindi_sarvam_audio():
    # Verify Hindi session generates high-fidelity Sarvam Bulbul TTS audio for native Hindi speech
    start_resp = client.post("/api/session/start", json={"language": "hi"})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert "greeting" in start_data
    assert "audio_base64" in start_data
    assert start_data["audio_base64"] is not None
    assert len(start_data["audio_base64"]) > 1000

    call_id = start_data["callId"]

    # Verify /api/chat delivers Sarvam audio in Hindi
    chat_resp = client.post("/api/chat", json={
        "callId": call_id,
        "message": "हाँ मैंने अपनी दवाई ले ली है।"
    })
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "reply" in chat_data
    assert "audio_base64" in chat_data
    assert chat_data["audio_base64"] is not None
    assert len(chat_data["audio_base64"]) > 1000
