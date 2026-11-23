# CareBridge — English Voice Elderly-Care Companion

A complete production-ready web application providing a continuous, natural voice conversation between an elderly user and an empathetic AI companion.

## Features

- **Continuous Voice Interaction Loop**: State machine managing `IDLE → LISTENING → THINKING → SPEAKING → (loop) → ENDING → ENDED`.
- **Speech Recognition (STT)**: Native browser `SpeechRecognition` / `webkitSpeechRecognition` (en-US & hi-IN) with auto-restart, silence timeout handling, and gentle check-in prompts.
- **Text-to-Speech (TTS)**: Native browser `SpeechSynthesis` with isolated abstraction (`tts.speak()`, `tts.stop()`), mute/unmute control, and auto-restart listening after speech ends.
- **Compact Context Architecture**: Constructs a compact payload with system instructions, rolling summary, important memories, structured elder status, and recent turns without resending the entire transcript.
- **Multi-Tier LLM Architecture with Key Rotation**:
  - **Tier 1 (Primary)**: Google Gemini (`gemini-3.5-flash-lite`) with automatic failover rotation across 1–3 Google API keys if rate limits (429) or error codes occur.
  - **Tier 2 (Secondary Fallback)**: OpenRouter with active resilient models.
  - **Tier 3 (Tertiary Fallback)**: NVIDIA NIM API.
  - **Tier 4 (Guaranteed Fail-Safe)**: Built-in state-aware clinical healthcare conversation engine.
- **Bilingual Voice Support (English & Hindi)**:
  - English (`en-US`) and Hindi (`hi-IN` / हिन्दी) toggled directly from the UI.
  - High-quality, elder-respectful Hindi conversational prompting (using 'आप' and respectful particles like 'जी') in native Devanagari script.
  - Neural Hindi voice preference (Google, Microsoft Swara/Madhur Natural) with a calibrated calm speaking rate (0.90) for elderly comprehension.
- **Safety Concern Detection**: Classifies concerns (`none`, `low`, `medium`, `high`) and displays real-time alerts on the caregiver panel.
- **Elderly-Friendly UI**: High-contrast, large readable typography, accessible buttons, live transcript, and dynamic status indicators.
- **End-of-Call Caregiver Report**: Generates clinical call summary, overall status (`stable`, `attention_needed`, `urgent`), key observations, and caregiver action items.
- **Dual Voice & Text Fallback**: Dual-input bar allows testing and interacting via typing or voice seamlessly.

---

## Architecture & Project Structure

```
carebridge-voice-ai/
├── app/
│   ├── config.py            # Environment & model configuration
│   ├── schemas.py           # Pydantic data models
│   ├── session_store.py     # Thread-safe in-memory session manager
│   ├── status_manager.py    # Structured elder status updates
│   ├── memory_manager.py    # Memory deduplication and importance
│   ├── concern_manager.py   # Safety concern validation & alert engine
│   ├── context_manager.py   # Compact context builder & turn tracking
│   ├── llm_service.py       # Multi-provider LLM client with clinical fallback
│   ├── summary_service.py   # Caregiver end-of-call report generator
│   └── main.py              # FastAPI server & REST endpoints
├── static/
│   ├── index.html           # Elderly-friendly web UI
│   ├── style.css            # Clean, high-legibility styling
│   └── app.js               # State machine, STT, and TTS audio loop
├── tests/
│   ├── test_carebridge.py   # Backend unit tests
│   └── test_api.py          # API lifecycle & integration tests
├── .env.example             # Template for API keys
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment process declaration
├── render.yaml              # Render blueprint specification
├── pytest.ini               # Pytest configuration
├── run.py                   # Production server entrypoint
└── README.md
```

---

## Local Development & Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your `OPENROUTER_API_KEY` or `NVIDIA_API_KEY` if desired. The application will automatically engage the clinical conversation engine if keys are omitted.

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

### 4. Run Demonstration Flow
```bash
python demo_flow.py
```

### 5. Start the Server
```bash
python run.py
```
Open **http://127.0.0.1:5500** in **Google Chrome** or **Microsoft Edge**.

---

## Production Deployment (Render)

### Backend (Web Service)
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/api/health`
- **Environment Variables**:
  - `LLM_PROVIDER`: `auto` (or `openrouter` / `nvidia`)
  - `OPENROUTER_API_KEY`: your secret API key
  - `NVIDIA_API_KEY`: your secret API key
  - `ALLOWED_ORIGINS`: `*` (or your frontend domain)

### Frontend (Static Site or Monolithic)
- Render Static Site:
  - **Build Command**: `echo Frontend ready`
  - **Publish Directory**: `static`
  - Set `CAREBRIDGE_API_URL` query param or config to point to the deployed backend.
- Or served directly by the FastAPI backend web service at root `/`.
