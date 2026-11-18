"""Build the LLM client that the settings ask for."""

from groundwork.config import Settings
from groundwork.llm.anthropic_client import AnthropicClient
from groundwork.llm.base import LLMClient
from groundwork.llm.openai_compatible import OpenAICompatibleClient


def build_llm(settings: Settings) -> LLMClient:
    match settings.llm_provider:
        case "openai_compatible":
            if not settings.llm_model:
                raise ValueError("LLM_PROVIDER=openai_compatible requires LLM_MODEL to be set")
            return OpenAICompatibleClient(
                settings.llm_base_url,
                settings.llm_api_key.get_secret_value(),
                settings.llm_model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
            )
        case "anthropic":
            if not settings.anthropic_model:
                raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_MODEL to be set")
            return AnthropicClient(
                settings.anthropic_api_key.get_secret_value(),
                settings.anthropic_model,
                timeout=settings.llm_timeout_seconds,
            )
