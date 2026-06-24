"""Tests for configuration and LLM provider selection."""

from __future__ import annotations

from unittest.mock import patch

from app.config import LLMProviderName, Settings
from app.llm import build_provider
from app.llm.local_provider import LocalLLMProvider


def _settings(**overrides) -> Settings:
    base = dict(
        llm_provider=LLMProviderName.local,
        local_llm_base_url="http://localhost:1234/v1",
        local_llm_model="qwen2.5-7b-instruct",
        anthropic_api_key="",
        anthropic_model="claude-sonnet-4-6",
        app_host="127.0.0.1",
        app_port=8000,
        allow_send=False,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_defaults_are_safe():
    s = _settings()
    assert s.llm_provider == LLMProviderName.local  # private by default
    assert s.app_host == "127.0.0.1"  # localhost only
    assert s.allow_send is False  # never send
    assert s.is_cloud_llm is False


def test_build_provider_local_is_default():
    provider = build_provider(_settings())
    assert isinstance(provider, LocalLLMProvider)
    assert provider.uses_native_tools is False


def test_build_provider_anthropic_when_selected():
    s = _settings(llm_provider=LLMProviderName.anthropic, anthropic_api_key="sk-test")
    with patch("app.llm.anthropic_provider.anthropic.Anthropic"):
        provider = build_provider(s)
    assert provider.name == "anthropic"
    assert provider.uses_native_tools is True
    assert s.is_cloud_llm is True


def test_anthropic_provider_missing_key_raises():
    s = _settings(llm_provider=LLMProviderName.anthropic, anthropic_api_key="")
    try:
        build_provider(s)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when ANTHROPIC_API_KEY is empty")
