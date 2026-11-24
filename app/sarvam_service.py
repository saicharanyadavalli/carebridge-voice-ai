import base64
import logging
import re
from typing import Optional, Dict, Any
import httpx
from app.config import config

logger = logging.getLogger("carebridge.sarvam")

class SarvamService:
    """
    High-fidelity Indian Voice AI integration via Sarvam AI:
    - Text-to-Speech (TTS) via Bulbul models (e.g. bulbul:v4-flash)
    - Speech-to-Text (STT) via Saaras models (e.g. saaras:v4)
    Supports English (en-IN) and natural conversational Hindi (hi-IN).
    """

    TTS_URL = "https://api.sarvam.ai/text-to-speech"
    STT_URL = "https://api.sarvam.ai/speech-to-text"
    _client: Optional[httpx.AsyncClient] = None
    _client_loop: Any = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        """Returns or creates a shared persistent httpx client with connection pooling bound to the active event loop."""
        import asyncio
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            cls._client is None
            or cls._client.is_closed
            or getattr(cls, "_client_loop", None) != current_loop
        ):
            cls._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
            cls._client_loop = current_loop
        return cls._client

    @classmethod
    async def close_client(cls):
        """Closes persistent httpx client on shutdown."""
        if cls._client and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None

    @classmethod
    def _resolve_language_code(cls, lang: Optional[str], text: Optional[str] = None) -> str:
        """Determines BCP-47 language code: 'en-IN' or 'hi-IN'."""
        if lang:
            l = lang.lower().strip()
            if l.startswith("hi"):
                return "hi-IN"
            if l.startswith("en"):
                return "en-IN"

        # Check for Devanagari script characters in text
        if text and any('\u0900' <= ch <= '\u097f' for ch in text):
            return "hi-IN"

        return "en-IN"

    @classmethod
    async def text_to_speech(
        cls,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Converts text to natural human speech audio (base64 WAV) using Sarvam AI Bulbul.
        """
        api_key = config.SARVAM_API_KEY
        if not api_key or api_key.startswith("your_"):
            logger.warning("Sarvam API key not configured. Skipping Sarvam TTS.")
            return None

        # Clean text of markdown artifacts
        clean_text = re.sub(r'[*_#`~]', '', text).strip()
        if not clean_text:
            return None

        lang_code = cls._resolve_language_code(language, clean_text)
        tts_model = model or config.SARVAM_TTS_MODEL
        tts_speaker = speaker or config.SARVAM_TTS_SPEAKER

        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": clean_text,
            "language_code": lang_code,
            "model": tts_model,
            "speaker": tts_speaker,
            "speech_sample_rate": 22050
        }

        try:
            client = cls.get_client()
            response = await client.post(cls.TTS_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                audios = data.get("audios", [])
                if audios:
                    logger.info(f"Successfully generated Sarvam TTS ({lang_code}, {tts_model})")
                    return {
                        "status": "ok",
                        "audio_base64": audios[0],
                        "format": "audio/wav",
                        "language_code": lang_code
                    }
                logger.warning("Sarvam TTS returned HTTP 200 but audios array was empty.")
            else:
                logger.warning(f"Sarvam TTS failed ({response.status_code}): {response.text[:200]}")
        except Exception as e:
            logger.warning(f"Exception during Sarvam TTS request: {e}")

        return None

    @classmethod
    async def speech_to_text(
        cls,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribes voice audio using Sarvam AI Saaras speech recognition.
        """
        api_key = config.SARVAM_API_KEY
        if not api_key or api_key.startswith("your_"):
            logger.warning("Sarvam API key not configured. Skipping Sarvam STT.")
            return None

        if not audio_bytes:
            return None

        lang_code = cls._resolve_language_code(language)
        stt_model = model or config.SARVAM_STT_MODEL

        headers = {
            "api-subscription-key": api_key
        }
        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data = {
            "model": stt_model,
            "language_code": lang_code
        }

        try:
            client = cls.get_client()
            response = await client.post(cls.STT_URL, headers=headers, files=files, data=data)
            if response.status_code == 200:
                res_json = response.json()
                transcript = res_json.get("transcript", "").strip()
                logger.info(f"Successfully transcribed audio via Sarvam STT ({lang_code}): '{transcript[:60]}'")
                return {
                    "status": "ok",
                    "transcript": transcript,
                    "language_code": res_json.get("language_code", lang_code),
                    "request_id": res_json.get("request_id")
                }
            else:
                logger.warning(f"Sarvam STT failed ({response.status_code}): {response.text[:200]}")
        except Exception as e:
            logger.warning(f"Exception during Sarvam STT request: {e}")

        return None
