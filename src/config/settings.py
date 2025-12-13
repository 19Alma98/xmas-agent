import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings:
    """Application settings."""

    LLM_PROVIDER = LLMProvider(os.getenv("LLM_PROVIDER", "openai"))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-40-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))

    RECIPE_CATEGORIES = ["appetizer", "main_dish", "second_plate", "dessert"]
    DIETARY_OPTIONS = ["vegan", "vegetarian", "gluten_free", "diary_free", "nut_free"]

    @classmethod
    def is_ollama(cls) -> bool:
        "Check if provider is Ollama"
        return cls.LLM_PROVIDER.value.lower() == "ollama"

    @classmethod
    def is_openai(cls) -> bool:
        "Check if provider is OpenAI"
        return cls.LLM_PROVIDER.value.lower() == "openai"

    @classmethod
    def get_model(cls) -> str:
        "Get configured model based on the provider"
        if cls.is_ollama():
            return cls.OLLAMA_MODEL
        return cls.DEFAULT_MODEL

    @classmethod
    def get_provider_info(cls) -> str:
        "Get provider general info"
        if cls.is_ollama():
            return f"Ollama ({cls.OLLAMA_MODEL}) at {cls.OLLAMA_BASE_URL}"
        return f"OpenAI ({cls.DEFAULT_MODEL})"


settings = Settings()
