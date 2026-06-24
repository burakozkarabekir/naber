"""Gmail integration package."""

from __future__ import annotations

from app.gmail.client import (
    DRAFTS_URL,
    DraftResult,
    EmailSummary,
    GmailClient,
    ThreadMessage,
    ThreadView,
)

__all__ = [
    "DRAFTS_URL",
    "DraftResult",
    "EmailSummary",
    "GmailClient",
    "ThreadMessage",
    "ThreadView",
]
