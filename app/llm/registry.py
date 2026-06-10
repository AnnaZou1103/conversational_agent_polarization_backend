from app.config import settings
from app.llm.anthropic import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.openai_provider import AzureOpenAIProvider, OpenAIProvider

_provider_instance: LLMProvider | None = None
_fast_provider_instance: LLMProvider | None = None


def _build_provider(model: str) -> LLMProvider:
    """Construct a provider for the configured backend at the given model."""
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.llm_api_key,
            model=model,
        )
    elif settings.llm_provider == "openai":
        return OpenAIProvider(
            api_key=settings.llm_api_key,
            model=model,
            base_url=settings.llm_base_url,
        )
    elif settings.llm_provider == "azure":
        if not settings.llm_base_url:
            raise ValueError(
                "Azure provider requires LLM_BASE_URL (your Azure endpoint)"
            )
        return AzureOpenAIProvider(
            api_key=settings.llm_api_key,
            model=model,
            base_url=settings.llm_base_url,
            api_version=settings.llm_api_version,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_provider() -> LLMProvider:
    """Get or create the configured LLM provider singleton."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = _build_provider(settings.llm_model)
    return _provider_instance


def get_fast_provider() -> LLMProvider:
    """Provider for internal pipeline steps (OBSERVE / THINK / stage eval).

    Uses settings.fast_llm_model when set; otherwise returns the main provider
    so the behavior is identical to before. Same provider backend and API key.
    """
    global _fast_provider_instance
    if not settings.fast_llm_model or settings.fast_llm_model == settings.llm_model:
        return get_provider()
    if _fast_provider_instance is None:
        _fast_provider_instance = _build_provider(settings.fast_llm_model)
    return _fast_provider_instance
