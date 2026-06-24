"""Gmail client — read, search, thread retrieval, and draft creation.

SECURITY / PRIVACY notes (audit here):
  * DRAFTS ONLY — NEVER SENDS. This module has no send code path. ``create_draft``
    calls ``users().drafts().create`` exclusively; there is no call to
    ``messages().send`` or ``drafts().send`` anywhere, and none may be added in
    this MVP. The ``ALLOW_SEND`` flag exists only to make this guarantee
    auditable — see ``app.main`` for the startup guard.
  * Email content (bodies, subjects, addresses) returned/accepted by these
    methods is passed to the LLM and the UI, but is NEVER written to logs.
    Logging uses operational metadata only (IDs, counts, error types) via
    ``app.security.redact.safe_meta``.
  * Bodies are not persisted to disk; everything is processed in memory.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import asdict, dataclass, field
from email.message import EmailMessage
from typing import Any

from googleapiclient.discovery import build

from app.config import Settings
from app.security.auth_google import get_credentials
from app.security.redact import safe_meta

logger = logging.getLogger("hermes.gmail")

# Where the user finds created drafts in the Gmail web UI.
DRAFTS_URL = "https://mail.google.com/mail/u/0/#drafts"

# Headers we read for message summaries. Kept minimal.
_SUMMARY_HEADERS = ["From", "Subject", "Date"]
# Headers needed to thread a reply correctly.
_REPLY_HEADERS = ["Message-ID", "References", "Subject", "From"]

# Crude HTML-to-text fallback used only when a message has no text/plain part.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


@dataclass
class EmailSummary:
    """Lightweight inbox row — no full body, just a snippet."""

    id: str
    thread_id: str
    sender: str
    subject: str
    snippet: str
    date: str
    unread: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThreadMessage:
    """A single message within a thread, including its plain-text body."""

    id: str
    sender: str
    to: str
    date: str
    subject: str
    body: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThreadView:
    thread_id: str
    messages: list[ThreadMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "messages": [m.to_dict() for m in self.messages],
        }


@dataclass
class DraftResult:
    """Result of creating a Gmail draft. Carries a clear, relayable confirmation.

    The ``reminder`` is intentionally part of the structured result so the agent
    always surfaces 'review & send manually' to the user — Hermes never sends.
    """

    draft_id: str
    thread_id: str
    is_reply: bool
    location: str = DRAFTS_URL
    reminder: str = (
        "Draft created. It was NOT sent. Open Gmail → Drafts to review and send "
        "it yourself."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GmailClient:
    """Thin wrapper over the Gmail API resource.

    Construct with an explicit service (tests inject a mock) or via
    :meth:`from_settings`, which obtains Keychain-stored credentials.
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    @classmethod
    def from_settings(cls, settings: Settings) -> "GmailClient":
        # allow_interactive=False: the server never blocks a request on a browser
        # consent flow. Authorization is performed explicitly via /auth/login.
        creds = get_credentials(settings, allow_interactive=False)
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return cls(service)

    # --- Read operations -------------------------------------------------

    def list_recent_emails(
        self, max_results: int = 10, query: str | None = None
    ) -> list[EmailSummary]:
        """List recent messages (optionally filtered by a Gmail query)."""
        max_results = max(1, min(int(max_results or 10), 50))
        resp = (
            self._service.users()
            .messages()
            .list(userId="me", maxResults=max_results, q=query)
            .execute()
        )
        ids = [m["id"] for m in resp.get("messages", [])]
        summaries = [self._get_summary(mid) for mid in ids]
        logger.info("list_recent_emails %s", safe_meta(count=len(summaries), q_present=bool(query)))
        return summaries

    def search_emails(self, query: str) -> list[EmailSummary]:
        """Search using Gmail query syntax (e.g. ``from:x is:unread newer_than:7d``)."""
        resp = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=50)
            .execute()
        )
        ids = [m["id"] for m in resp.get("messages", [])]
        summaries = [self._get_summary(mid) for mid in ids]
        logger.info("search_emails %s", safe_meta(count=len(summaries)))
        return summaries

    def get_thread(self, thread_id: str) -> ThreadView:
        """Retrieve a full thread (all messages, plain-text bodies)."""
        thread = (
            self._service.users()
            .threads()
            .get(userId="me", id=thread_id, format="full")
            .execute()
        )
        messages = [self._parse_full(m) for m in thread.get("messages", [])]
        logger.info(
            "get_thread %s",
            safe_meta(thread_id=thread_id, message_count=len(messages)),
        )
        return ThreadView(thread_id=thread_id, messages=messages)

    # --- Draft creation (DRAFTS ONLY — never sends) ----------------------

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        in_reply_to_message_id: str | None = None,
    ) -> DraftResult:
        """Create a Gmail **draft**. Never sends — there is no send path.

        For a reply (``in_reply_to_message_id`` provided), the original message
        is read to obtain its RFC822 ``Message-ID`` and ``threadId`` so the draft
        threads correctly (In-Reply-To / References headers + threadId).

        Args:
            to: recipient address(es).
            subject: subject line (e.g. "Re: ...").
            body: the message body (plain text), already composed by the agent.
            in_reply_to_message_id: optional Gmail message id to reply to.

        Returns:
            A :class:`DraftResult` with the draft id, thread id, and a reminder
            that the user must review and send it manually.
        """
        mime = EmailMessage()
        mime["To"] = to
        mime["Subject"] = subject
        # set_content handles encoding; body is plain text only (no HTML send path).
        mime.set_content(body or "")

        thread_id: str | None = None
        if in_reply_to_message_id:
            thread_id = self._apply_reply_headers(mime, in_reply_to_message_id)

        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        message_resource: dict[str, Any] = {"raw": raw}
        if thread_id:
            message_resource["threadId"] = thread_id

        # SECURITY: drafts().create ONLY. No .send anywhere.
        draft = (
            self._service.users()
            .drafts()
            .create(userId="me", body={"message": message_resource})
            .execute()
        )

        draft_id = draft.get("id", "")
        result_thread = (
            thread_id or draft.get("message", {}).get("threadId", "") or ""
        )
        logger.info(
            "create_draft %s",
            safe_meta(
                draft_id=draft_id,
                is_reply=bool(in_reply_to_message_id),
                outcome="draft_created",
            ),
        )
        return DraftResult(
            draft_id=draft_id,
            thread_id=result_thread,
            is_reply=bool(in_reply_to_message_id),
        )

    def _apply_reply_headers(self, mime: EmailMessage, message_id: str) -> str | None:
        """Read the original message and set reply threading headers.

        Returns the original's threadId (so the draft attaches to the thread).
        """
        original = (
            self._service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=_REPLY_HEADERS,
            )
            .execute()
        )
        headers = _headers_map(original.get("payload", {}))
        orig_msgid = headers.get("message-id")
        if orig_msgid:
            mime["In-Reply-To"] = orig_msgid
            refs = headers.get("references", "").strip()
            # Append the original Message-ID to References, but avoid duplicating
            # it if it's already the last reference (common when replying to the
            # most recent message in a thread).
            if refs.split()[-1:] == [orig_msgid]:
                mime["References"] = refs
            else:
                mime["References"] = f"{refs} {orig_msgid}".strip()
        return original.get("threadId")

    # --- Internal helpers ------------------------------------------------

    def _get_summary(self, message_id: str) -> EmailSummary:
        msg = (
            self._service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=_SUMMARY_HEADERS,
            )
            .execute()
        )
        headers = _headers_map(msg.get("payload", {}))
        label_ids = msg.get("labelIds", []) or []
        return EmailSummary(
            id=msg.get("id", message_id),
            thread_id=msg.get("threadId", ""),
            sender=headers.get("from", ""),
            subject=headers.get("subject", ""),
            snippet=msg.get("snippet", ""),
            date=headers.get("date", ""),
            unread="UNREAD" in label_ids,
        )

    def _parse_full(self, msg: dict[str, Any]) -> ThreadMessage:
        payload = msg.get("payload", {})
        headers = _headers_map(payload)
        return ThreadMessage(
            id=msg.get("id", ""),
            sender=headers.get("from", ""),
            to=headers.get("to", ""),
            date=headers.get("date", ""),
            subject=headers.get("subject", ""),
            body=_extract_body(payload),
        )


def _headers_map(payload: dict[str, Any]) -> dict[str, str]:
    return {
        h.get("name", "").lower(): h.get("value", "")
        for h in payload.get("headers", [])
    }


def _decode_b64(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode(
            "utf-8", errors="replace"
        )
    except Exception:  # noqa: BLE001 - never raise on a single bad part
        return ""


def _html_to_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _extract_body(payload: dict[str, Any]) -> str:
    """Extract a plain-text body from a message payload.

    Prefers ``text/plain``; falls back to stripped ``text/html`` only if no
    plain-text part exists. Walks nested multipart structures.
    """
    mime = payload.get("mimeType", "")
    data = payload.get("body", {}).get("data")

    if mime == "text/plain" and data:
        return _decode_b64(data).strip()

    parts = payload.get("parts", []) or []
    # First pass: prefer any text/plain anywhere in the tree.
    for part in parts:
        text = _extract_body(part)
        if text:
            return text

    # Fallback: this node is HTML with no plain-text sibling.
    if mime == "text/html" and data:
        return _html_to_text(_decode_b64(data))

    return ""
