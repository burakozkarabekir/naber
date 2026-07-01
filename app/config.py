"""Typed application configuration loaded from environment / .env.

Security note: this module reads configuration only. Secrets that must never
touch disk in plaintext (the OAuth token) are handled by the Keychain layer in
``app.security.auth_google`` — NOT here. The only secret read here is
``ANTHROPIC_API_KEY`` (used solely for the opt-in cloud provider), and it is
never logged.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(str, Enum):
    """Selectable LLM backends. ``local`` keeps email content on-device."""

    local = "local"
    anthropic = "anthropic"


class Settings(BaseSettings):
    """All runtime configuration for Hermes.

    Values come from environment variables (and a local ``.env`` during
    development). Field names map to upper-case env vars case-insensitively.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM selection ---
    # Default to the local provider so email content never leaves the machine.
    llm_provider: LLMProviderName = LLMProviderName.local

    # Local LLM (LM Studio, OpenAI-compatible endpoint).
    local_llm_base_url: str = "http://localhost:1234/v1"
    local_llm_model: str = "qwen2.5-7b-instruct"

    # Cloud LLM (only used when llm_provider == anthropic).
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # --- Google OAuth client ---
    google_credentials_path: str = "./credentials.json"

    # --- Server (localhost only; enforced at bind time in main.py) ---
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # --- Safety guardrail ---
    # MUST remain False for the MVP. Hermes never sends email; it only drafts.
    allow_send: bool = Field(default=False)

    # --- Demo mode ---
    # When true, Hermes runs entirely on in-memory sample data with a scripted
    # LLM — no Google OAuth, no LM Studio, no API key. For UI/UX testing only.
    # Never enable against a real mailbox; it ignores real Gmail entirely.
    demo_mode: bool = Field(default=False)

    log_level: str = "INFO"

    @property
    def is_cloud_llm(self) -> bool:
        return self.llm_provider == LLMProviderName.anthropic


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
