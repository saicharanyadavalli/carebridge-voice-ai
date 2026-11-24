import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import config
from app.schemas import SessionState, LLMTurnResponse, FinalSummaryResponse
from app.session_store import session_store
from app.context_manager import ContextManager
from app.status_manager import StatusManager
from app.memory_manager import MemoryManager
from app.concern_manager import ConcernManager
from app.state.reducer import StateReducer
from app.guardrails import SafetyGuardrail
from app.llm_service import LLMService
from app.summary_service import SummaryService
from app.sarvam_service import SarvamService
from app.dialogue_engine.fsm import DialogueFSM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("carebridge.main")

app = FastAPI(title="CareBridge Voice AI", version="2.0.0")

# Configure CORS for local development and production deployments
raw_origins = getattr(config, "ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
if not origins or "*" in origins:
    cors_origins = ["*"]
    allow_credentials = False
else:
    cors_origins = origins
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class StartSessionRequest(BaseModel):
    language: Optional[str] = "en"
    clientHour: Optional[int] = None

class ChatRequest(BaseModel):
    callId: str
    message: str

class EndSessionRequest(BaseModel):
    callId: str

class VoiceTTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    speaker: Optional[str] = None

@app.get("/api/health")
async def health_check():
    gemini_keys = config.GEMINI_API_KEYS
    return {
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "gemini_model": config.GEMINI_MODEL,
        "gemini_keys_count": len(gemini_keys),
        "has_gemini_key": len(gemini_keys) > 0,
        "openrouter_model": config.OPENROUTER_MODEL,
        "nvidia_model": config.NVIDIA_MODEL,
        "has_openrouter_key": bool(config.OPENROUTER_API_KEY and not config.OPENROUTER_API_KEY.startswith("your_")),
        "has_nvidia_key": bool(config.NVIDIA_API_KEY and not config.NVIDIA_API_KEY.startswith("your_")),
        "has_sarvam_key": bool(config.SARVAM_API_KEY and not config.SARVAM_API_KEY.startswith("your_")),
        "sarvam_tts_model": config.SARVAM_TTS_MODEL,
        "sarvam_tts_speaker": config.SARVAM_TTS_SPEAKER,
        "sarvam_stt_model": config.SARVAM_STT_MODEL,
        "language": config.APP_LANGUAGE
    }

@app.post("/api/voice/tts")
async def voice_tts(req: VoiceTTSRequest):
    """Generates natural voice speech using Sarvam AI Bulbul models with graceful fallback."""
    res = await SarvamService.text_to_speech(req.text, language=req.language, speaker=req.speaker)
    if not res:
        return {"status": "fallback", "audio_base64": None, "message": "Sarvam TTS unavailable, using browser speech synthesis"}
    return res

MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

@app.post("/api/voice/stt")
async def voice_stt(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """Transcribes audio using Sarvam AI Saaras speech recognition."""
    if file.size and file.size > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 10MB limit")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided")
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 10MB limit")

    res = await SarvamService.speech_to_text(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
        content_type=file.content_type or "audio/wav",
        language=language
    )
    if not res:
        raise HTTPException(status_code=500, detail="Sarvam STT transcription failed or key unconfigured")
    return res

@app.post("/api/session/start")
async def start_session(req: Optional[StartSessionRequest] = None):
    """Starts a new conversation session and provides a time-aware greeting in English or Hindi with Sarvam TTS."""
    session = session_store.create_session()
    lang = (req.language if req and req.language else config.APP_LANGUAGE).lower()
    session.language = lang
    client_hour = req.clientHour if (req and req.clientHour is not None) else datetime.now().hour
    session.clientHour = client_hour

    initial_greeting = DialogueFSM.get_initial_greeting(lang, client_hour)
    
    # Record initial assistant greeting in transcript
    now_iso = datetime.now(timezone.utc).isoformat()
    ContextManager.append_turn(session, "assistant", initial_greeting, now_iso)
    session_store.save_session(session)
    
    # Generate Sarvam TTS for Hindi; English uses zero-latency browser speech synthesis
    audio_base64 = None
    if lang.startswith("hi"):
        tts_res = await SarvamService.text_to_speech(initial_greeting, language=lang)
        audio_base64 = tts_res["audio_base64"] if tts_res else None
    
    return {
        "callId": session.callId,
        "greeting": initial_greeting,
        "language": lang,
        "audio_base64": audio_base64,
        "session": session.model_dump()
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handles one conversation turn with low-latency execution & deterministic safety:
    1. Reconciles session state from SQLite/cache
    2. Sub-millisecond Safety Guardrail Check
    3. Deterministic FSM context generation
    4. Resilient LLM turn resolution
    5. Event-sourced State Reducer updates
    6. Persists canonical state & returns voice payload
    """
    session = session_store.get_session(request.callId)
    if not session:
        logger.warning(f"Session {request.callId} not found (server was likely restarted). Automatically re-creating session.")
        session = session_store.create_session(call_id=request.callId)

    user_text = request.message.strip()
    if not user_text:
        return {
            "reply": "I'm listening whenever you're ready.",
            "session": session.model_dump()
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    turn_index = len(session.transcript) if session.transcript else 0

    # 1. Deterministic Sub-millisecond Safety Guardrail Check
    emergency = SafetyGuardrail.check_emergency(user_text, session=session)
    if emergency:
        logger.warning(f"Safety guardrail triggered for call {session.callId}: {emergency.get('concern')}")
        ContextManager.append_turn(session, "user", user_text, now_iso)
        
        StateReducer.batch_reduce(
            session=session,
            status_updates=emergency.get("status_updates"),
            memory_updates=emergency.get("memory_updates"),
            concern_update=emergency.get("concern"),
            turn_index=turn_index
        )

        reply_text = emergency["reply"]
        ContextManager.append_turn(session, "assistant", reply_text, datetime.now(timezone.utc).isoformat())
        session_store.save_session(session)

        audio_b64 = None
        if session.language.startswith("hi"):
            tts_res = await SarvamService.text_to_speech(reply_text, language=session.language)
            audio_b64 = tts_res["audio_base64"] if tts_res else None

        return {
            "reply": reply_text,
            "call_ended": False,
            "audio_base64": audio_b64,
            "memory_updates": emergency.get("memory_updates", []),
            "status_updates": emergency.get("status_updates", {}),
            "concern": emergency.get("concern") or {"level": "high"},
            "elderStatus": session.elderStatus,
            "activeConcerns": session.activeConcerns,
            "session": session.model_dump()
        }

    # 2. Record user turn
    ContextManager.append_turn(session, "user", user_text, now_iso)

    # 3. Build compact, FSM-guided context
    context_messages = ContextManager.build_compact_context(session, user_text)

    # 4. Call LLM Service (Gemini with multi-key rotation -> OpenRouter -> NVIDIA -> Clinical Engine)
    llm_result = await LLMService.chat_turn(context_messages, session=session)

    reply_text = llm_result.get("reply", "I am here with you. How can I help?")

    # 5. Process state updates through deterministic event reducer
    status_updates = llm_result.get("status_updates", {})
    memory_updates = llm_result.get("memory_updates", [])
    concern_update = llm_result.get("concern")

    StateReducer.batch_reduce(
        session=session,
        status_updates=status_updates,
        memory_updates=memory_updates,
        concern_update=concern_update,
        turn_index=turn_index
    )

    # 6. Record assistant turn
    reply_timestamp = datetime.now(timezone.utc).isoformat()
    ContextManager.append_turn(session, "assistant", reply_text, reply_timestamp)

    # 7. Persist updated canonical session
    session_store.save_session(session)

    active_concern_payload = concern_update if (concern_update and concern_update.get("level") != "none") else {"level": "none"}

    # Generate Sarvam TTS for Hindi; English uses zero-latency browser speech synthesis
    audio_b64 = None
    if session.language.startswith("hi"):
        tts_res = await SarvamService.text_to_speech(reply_text, language=session.language)
        audio_b64 = tts_res["audio_base64"] if tts_res else None

    return {
        "reply": reply_text,
        "call_ended": bool(llm_result.get("call_ended", False)),
        "audio_base64": audio_b64,
        "memory_updates": memory_updates,
        "status_updates": status_updates,
        "concern": active_concern_payload,
        "elderStatus": session.elderStatus,
        "activeConcerns": session.activeConcerns,
        "session": session.model_dump()
    }

@app.post("/api/session/end")
async def end_session(request: EndSessionRequest):
    """Ends the session, persists final state, and returns structured caregiver summary."""
    session = session_store.get_session(request.callId)
    if not session:
        logger.warning(f"Session {request.callId} not found on end_session. Creating session to generate summary.")
        session = session_store.create_session(call_id=request.callId)

    session_store.end_session(request.callId)
    summary_report = await SummaryService.generate_summary(session)

    return {
        "callId": session.callId,
        "report": summary_report,
        "session": session.model_dump()
    }

@app.get("/api/session/{call_id}")
async def get_session(call_id: str):
    """Retrieves session details and full transcript from store."""
    session = session_store.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()

# Static directory configuration
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if not os.path.exists(static_dir):
    static_dir = "static"

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """Root endpoint: serves frontend UI for browsers, or API info for API clients."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file, media_type="text/html")
    return {
        "service": "CareBridge Voice AI — Backend API",
        "status": "online",
        "message": "CareBridge backend server is awake and healthy.",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "start_session": "/api/session/start",
            "chat": "/api/chat",
            "end_session": "/api/session/end",
            "frontend": "/static/index.html"
        },
        "frontend_app": "/static/index.html"
    }

@app.api_route("/app", methods=["GET", "HEAD"])
async def app_view():
    """Dedicated endpoint to view the frontend directly."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend file not found")

@app.api_route("/static", methods=["GET", "HEAD"])
async def static_redirect():
    """Redirect /static to /static/ to avoid 404."""
    return RedirectResponse(url="/static/", status_code=307)

@app.api_route("/style.css", methods=["GET", "HEAD"])
async def get_root_style():
    style_file = os.path.join(static_dir, "style.css")
    if os.path.exists(style_file):
        return FileResponse(style_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")

@app.api_route("/app.js", methods=["GET", "HEAD"])
async def get_root_app_js():
    js_file = os.path.join(static_dir, "app.js")
    if os.path.exists(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")

@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
async def favicon():
    return Response(status_code=204)

# Mount frontend static files under /static
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

@app.api_route("/{filename:path}", methods=["GET", "HEAD"])
async def serve_static_root(filename: str):
    """Serves root-level static assets (icons, images, audio) requested by the frontend."""
    file_path = os.path.join(static_dir, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
