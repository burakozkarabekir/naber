"""LLM provider package: interface + provider selection."""

from __future__ import annotations

import logging

from app.config import LLMProviderName, Settings
from app.llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger("hermes.llm")


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the LLM provider selected by configuration.

    Defaults to the local provider so email content stays on-device. The cloud
    provider is opt-in and triggers a loud warning (emitted by the caller in
    ``app.main`` at startup).
    """
    if settings.llm_provider == LLMProviderName.anthropic:
        # Imported lazily so the anthropic SDK isn't required for local-only use.
        from app.llm.anthropic_provider import AnthropicProvider

        logger.info("Using cloud LLM provider (anthropic).")
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )

    from app.llm.local_provider import LocalLLMProvider

    logger.info("Using local LLM provider.")
    return LocalLLMProvider(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
    )


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "build_provider",
]
