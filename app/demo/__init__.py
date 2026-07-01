"""Demo mode: in-memory sample mailbox + scripted LLM (no external services)."""

from __future__ import annotations

from app.demo.fake_gmail import FakeGmailClient
from app.demo.provider import DemoLLMProvider

__all__ = ["FakeGmailClient", "DemoLLMProvider"]
