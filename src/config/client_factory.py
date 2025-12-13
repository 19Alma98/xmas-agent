from datapizza.clients.openai import OpenAIClient  # type: ignore

from loguru import logger
from .settings import settings


def create_client(
    api_key: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> OpenAIClient:
    use_ollama = (provider or settings.LLM_PROVIDER.value).lower() == "ollama"

    if use_ollama:
        # TODO: create Ollama client class
        raise ValueError("Still not implemented")

    return _create_openai_client(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
    )


def _create_openai_client(
    api_key: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> OpenAIClient:
    """Create OpenAI client"""
    return OpenAIClient(
        api_key=api_key or settings.OPENAI_API_KEY,
        model=model or settings.DEFAULT_MODEL,
        system_prompt=system_prompt,
        temperature=temperature or settings.TEMPERATURE,
    )


def get_provider_name() -> str:
    """Get current provider name"""
    return "Ollama" if settings.is_ollama() else "OpenAI"


def get_model_name() -> str:
    """Get current model name"""
    return settings.get_model()


def test_connection() -> bool:
    try:
        client = create_client()
        response = client.invoke("Say 'Christmas', just one word.")
    except Exception as e:
        logger.exception(f"Connection test failed: {e}")
    return response is not None
