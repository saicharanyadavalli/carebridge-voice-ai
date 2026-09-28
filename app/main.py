import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
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
from app.llm_service import LLMService
from app.summary_service import SummaryService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("carebridge.main")

app = FastAPI(title="CareBridge Voice AI", version="1.0.0")

# Configure CORS for local development and production deployments
raw_origins = getattr(config, "ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
if not origins or "*" in origins:
    cors_origins = ["*"]
else:
    cors_origins = origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class StartSessionRequest(BaseModel):
    language: Optional[str] = "en"

class ChatRequest(BaseModel):
    callId: str
    message: str

class EndSessionRequest(BaseModel):
    callId: str

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "openrouter_model": config.OPENROUTER_MODEL,
        "nvidia_model": config.NVIDIA_MODEL,
        "has_openrouter_key": bool(config.OPENROUTER_API_KEY and not config.OPENROUTER_API_KEY.startswith("your_")),
        "has_nvidia_key": bool(config.NVIDIA_API_KEY and not config.NVIDIA_API_KEY.startswith("your_")),
        "language": config.APP_LANGUAGE
    }

@app.post("/api/session/start")
async def start_session(req: Optional[StartSessionRequest] = None):
    """Starts a new conversation session and provides an initial greeting in English or Hindi."""
    session = session_store.create_session()
    lang = (req.language if req and req.language else config.APP_LANGUAGE).lower()
    session.language = lang
    
    if lang.startswith("hi"):
        initial_greeting = "नमस्ते! मैं CareBridge हूँ, आपकी आवाज़ साथी। आज आप कैसा महसूस कर रहे हैं, और क्या आपने अपनी सुबह की दवाई ले ली है?"
    else:
        initial_greeting = "Hello! I am CareBridge, your voice companion. It's so nice to speak with you today. How are you feeling, and did you remember to take your morning medication yet?"
    
    # Record initial assistant greeting in transcript
    now_iso = datetime.now(timezone.utc).isoformat()
    ContextManager.append_turn(session, "assistant", initial_greeting, now_iso)
    
    return {
        "callId": session.callId,
        "greeting": initial_greeting,
        "language": lang,
        "session": session.model_dump()
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handles one conversation turn:
    1. Records user message
    2. Builds compact context
    3. Calls OpenRouter LLM for structured JSON
    4. Updates memory, status, concerns
    5. Records assistant reply
    6. Returns speech reply + updated session state
    """
    session = session_store.get_session(request.callId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_text = request.message.strip()
    if not user_text:
        return {
            "reply": "I'm listening whenever you're ready.",
            "session": session.model_dump()
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    
    # 1. Record user turn
    ContextManager.append_turn(session, "user", user_text, now_iso)

    # 2. Build compact context
    context_messages = ContextManager.build_compact_context(session, user_text)

    # 3. Call LLM for single turn structured response
    llm_result = await LLMService.chat_turn(context_messages, session=session)

    reply_text = llm_result.get("reply", "I am here with you. How can I help?")

    # 4. Merge updates
    status_updates = llm_result.get("status_updates", {})
    if status_updates:
        StatusManager.update_status(session, status_updates)

    memory_updates = llm_result.get("memory_updates", [])
    if memory_updates:
        MemoryManager.add_memories(session, memory_updates)

    concern_update = llm_result.get("concern")
    new_concern = ConcernManager.process_concern(session, concern_update)

    # 5. Record assistant turn
    reply_timestamp = datetime.now(timezone.utc).isoformat()
    ContextManager.append_turn(session, "assistant", reply_text, reply_timestamp)

    return {
        "reply": reply_text,
        "memory_updates": memory_updates,
        "status_updates": status_updates,
        "concern": new_concern or {"level": "none"},
        "elderStatus": session.elderStatus,
        "activeConcerns": session.activeConcerns,
        "session": session.model_dump()
    }

@app.post("/api/session/end")
async def end_session(request: EndSessionRequest):
    """Ends the session and returns structured caregiver summary."""
    session = session_store.get_session(request.callId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session_store.end_session(request.callId)
    summary_report = await SummaryService.generate_summary(session)

    return {
        "callId": session.callId,
        "report": summary_report,
        "session": session.model_dump()
    }

@app.get("/api/session/{call_id}")
async def get_session(call_id: str):
    """Retrieves session details and full transcript."""
    session = session_store.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Mount frontend static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if not os.path.exists(static_dir):
    static_dir = "static"

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_root")
