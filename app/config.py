import os
from typing import Optional, List
from dotenv import load_dotenv

DOTENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

class Config:
    @property
    def GEMINI_API_KEYS(self) -> List[str]:
        """Returns non-empty, non-placeholder Google API keys for rotation."""
        keys: List[str] = []
        # Check standard GEMINI_API_KEY (or comma-separated)
        primary = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if primary:
            for k in primary.split(","):
                k_clean = k.strip()
                if k_clean and not k_clean.startswith("your_") and k_clean not in keys:
                    keys.append(k_clean)

        # Check secondary key
        key2 = os.getenv("GEMINI_API_KEY_2", "").strip()
        if key2 and not key2.startswith("your_") and key2 not in keys:
            keys.append(key2)

        # Check tertiary key
        key3 = os.getenv("GEMINI_API_KEY_3", "").strip()
        if key3 and not key3.startswith("your_") and key3 not in keys:
            keys.append(key3)

        return keys

    @property
    def GEMINI_API_KEY(self) -> str:
        keys = self.GEMINI_API_KEYS
        return keys[0] if keys else ""

    @property
    def GEMINI_MODEL(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def OPENROUTER_API_KEY(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def OPENROUTER_MODEL(self) -> str:
        return os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")

    @property
    def NVIDIA_API_KEY(self) -> str:
        return os.getenv("NVIDIA_API_KEY", "")

    @property
    def NVIDIA_MODEL(self) -> str:
        return os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    @property
    def NVIDIA_BASE_URL(self) -> str:
        return os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    @property
    def LLM_PROVIDER(self) -> str:
        return os.getenv("LLM_PROVIDER", "gemini").lower()

    @property
    def APP_LANGUAGE(self) -> str:
        return os.getenv("APP_LANGUAGE", "en").lower()

    @property
    def PORT(self) -> int:
        return int(os.getenv("PORT", "5500"))

    @property
    def HOST(self) -> str:
        return os.getenv("HOST", "0.0.0.0")

    @property
    def ALLOWED_ORIGINS(self) -> str:
        return os.getenv("ALLOWED_ORIGINS", "*")

    @property
    def SARVAM_API_KEY(self) -> str:
        return os.getenv("SARVAM_API_KEY", "").strip()

    @property
    def SARVAM_TTS_MODEL(self) -> str:
        return os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash").strip()

    @property
    def SARVAM_TTS_SPEAKER(self) -> str:
        return os.getenv("SARVAM_TTS_SPEAKER", "simran_en_customer").strip()

    @property
    def SARVAM_STT_MODEL(self) -> str:
        return os.getenv("SARVAM_STT_MODEL", "saaras:v4").strip()

    REQUEST_TIMEOUT: float = 25.0
    MAX_RECENT_TURNS: int = 6

config = Config()
