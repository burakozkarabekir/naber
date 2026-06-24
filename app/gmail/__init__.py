"""Gmail integration package."""

from __future__ import annotations

from app.gmail.client import (
    EmailSummary,
    GmailClient,
    ThreadMessage,
    ThreadView,
)

__all__ = [
    "EmailSummary",
    "GmailClient",
    "ThreadMessage",
    "ThreadView",
]
