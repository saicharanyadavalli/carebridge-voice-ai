import os
from typing import Optional
from dotenv import load_dotenv

DOTENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

class Config:
    @property
    def OPENROUTER_API_KEY(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def OPENROUTER_MODEL(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        return os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")

    @property
    def NVIDIA_API_KEY(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        return os.getenv("NVIDIA_API_KEY", "")

    @property
    def NVIDIA_MODEL(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        return os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    @property
    def NVIDIA_BASE_URL(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        return os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    @property
    def LLM_PROVIDER(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        return os.getenv("LLM_PROVIDER", "auto").lower()

    @property
    def APP_LANGUAGE(self) -> str:
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
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

    REQUEST_TIMEOUT: float = 25.0
    MAX_RECENT_TURNS: int = 6

config = Config()
