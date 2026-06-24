"""Tests for the Gmail client (Gmail API fully mocked — no network)."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

from app.gmail.client import GmailClient, _extract_body


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def _fake_service_for_list(message_ids, message_get_results):
    """Build a mock Gmail service whose list/get chain returns canned data."""
    service = MagicMock()
    users = service.users.return_value
    messages = users.messages.return_value

    # users().messages().list(...).execute()
    messages.list.return_value.execute.return_value = {
        "messages": [{"id": mid} for mid in message_ids]
    }

    # users().messages().get(...).execute() — keyed by message id
    def _get(userId, id, **kwargs):  # noqa: N803 - match API kwarg names
        call = MagicMock()
        call.execute.return_value = message_get_results[id]
        return call

    messages.get.side_effect = _get
    return service


def test_list_recent_emails_maps_headers_and_unread():
    results = {
        "m1": {
            "id": "m1",
            "threadId": "t1",
            "snippet": "Quick question about the report",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "Subject", "value": "Report"},
                    {"name": "Date", "value": "Mon, 1 Jan 2026 10:00:00 +0000"},
                ]
            },
        }
    }
    client = GmailClient(_fake_service_for_list(["m1"], results))
    emails = client.list_recent_emails(max_results=5)

    assert len(emails) == 1
    e = emails[0]
    assert e.id == "m1"
    assert e.thread_id == "t1"
    assert e.sender == "Alice <alice@example.com>"
    assert e.subject == "Report"
    assert e.snippet.startswith("Quick question")
    assert e.unread is True


def test_list_recent_emails_clamps_max_results():
    client = GmailClient(_fake_service_for_list([], {}))
    client.list_recent_emails(max_results=999)
    _, kwargs = client._service.users().messages().list.call_args
    assert kwargs["maxResults"] == 50  # clamped to the upper bound


def test_search_emails_passes_query():
    client = GmailClient(_fake_service_for_list([], {}))
    client.search_emails("from:bob@x.com is:unread")
    _, kwargs = client._service.users().messages().list.call_args
    assert kwargs["q"] == "from:bob@x.com is:unread"


def test_get_thread_parses_messages_and_bodies():
    service = MagicMock()
    threads = service.users.return_value.threads.return_value
    threads.get.return_value.execute.return_value = {
        "messages": [
            {
                "id": "m1",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "alice@example.com"},
                        {"name": "To", "value": "me@example.com"},
                        {"name": "Subject", "value": "Budget"},
                        {"name": "Date", "value": "Mon, 1 Jan 2026"},
                    ],
                    "body": {"data": _b64("Here is the Q3 budget summary.")},
                },
            }
        ]
    }
    client = GmailClient(service)
    thread = client.get_thread("t1")

    assert thread.thread_id == "t1"
    assert len(thread.messages) == 1
    m = thread.messages[0]
    assert m.sender == "alice@example.com"
    assert m.subject == "Budget"
    assert "Q3 budget" in m.body


def test_extract_body_prefers_plain_over_html():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64("plain version")}},
            {"mimeType": "text/html", "body": {"data": _b64("<p>html version</p>")}},
        ],
    }
    assert _extract_body(payload) == "plain version"


def test_extract_body_falls_back_to_html_stripped():
    payload = {
        "mimeType": "text/html",
        "body": {"data": _b64("<div>Hello <b>world</b></div>")},
    }
    out = _extract_body(payload)
    assert "Hello" in out and "world" in out
    assert "<" not in out  # tags stripped


# --- create_draft (DRAFTS ONLY) ------------------------------------------


def _decode_raw(raw_b64: str) -> str:
    return base64.urlsafe_b64decode(raw_b64.encode("ascii")).decode("utf-8")


def test_create_draft_new_email_uses_drafts_create_only():
    service = MagicMock()
    drafts = service.users.return_value.drafts.return_value
    drafts.create.return_value.execute.return_value = {
        "id": "draft123",
        "message": {"id": "msg123", "threadId": "thr123"},
    }
    client = GmailClient(service)

    result = client.create_draft(
        to="bob@example.com", subject="Hello", body="Hi Bob, how are you?"
    )

    assert result.draft_id == "draft123"
    assert result.is_reply is False
    assert "drafts" in result.location.lower()
    assert "review" in result.reminder.lower()

    # The raw MIME contains our fields.
    _, kwargs = drafts.create.call_args
    raw = kwargs["body"]["message"]["raw"]
    mime = _decode_raw(raw)
    assert "To: bob@example.com" in mime
    assert "Subject: Hello" in mime
    assert "Hi Bob" in mime
    # A brand-new email has no threadId.
    assert "threadId" not in kwargs["body"]["message"]

    # SECURITY: the send endpoints were never touched.
    service.users().drafts().send.assert_not_called()
    service.users().messages().send.assert_not_called()


def test_create_draft_reply_sets_threading_headers():
    service = MagicMock()
    users = service.users.return_value
    # Original message lookup (for Message-ID + threadId).
    users.messages.return_value.get.return_value.execute.return_value = {
        "id": "m1",
        "threadId": "thread-A",
        "payload": {
            "headers": [
                {"name": "Message-ID", "value": "<orig@mail.example.com>"},
                {"name": "References", "value": "<older@mail.example.com>"},
                {"name": "Subject", "value": "Budget"},
            ]
        },
    }
    drafts = users.drafts.return_value
    drafts.create.return_value.execute.return_value = {"id": "draftR", "message": {}}

    client = GmailClient(service)
    result = client.create_draft(
        to="alice@example.com",
        subject="Re: Budget",
        body="Looks good.",
        in_reply_to_message_id="m1",
    )

    assert result.is_reply is True
    assert result.thread_id == "thread-A"

    _, kwargs = drafts.create.call_args
    # Draft is attached to the original thread.
    assert kwargs["body"]["message"]["threadId"] == "thread-A"
    mime = _decode_raw(kwargs["body"]["message"]["raw"])
    assert "In-Reply-To: <orig@mail.example.com>" in mime
    assert "<orig@mail.example.com>" in mime  # appears in References too
    assert "<older@mail.example.com>" in mime  # prior References preserved

    service.users().drafts().send.assert_not_called()


def test_create_draft_reply_does_not_duplicate_references():
    # If the original References already ends with the original Message-ID,
    # we must not append it twice.
    service = MagicMock()
    users = service.users.return_value
    users.messages.return_value.get.return_value.execute.return_value = {
        "id": "m1",
        "threadId": "thread-A",
        "payload": {
            "headers": [
                {"name": "Message-ID", "value": "<orig@mail.example.com>"},
                {"name": "References", "value": "<a@x> <orig@mail.example.com>"},
            ]
        },
    }
    drafts = users.drafts.return_value
    drafts.create.return_value.execute.return_value = {"id": "d", "message": {}}
    client = GmailClient(service)
    client.create_draft(
        to="a@x.com", subject="Re", body="b", in_reply_to_message_id="m1"
    )
    _, kwargs = drafts.create.call_args
    mime = _decode_raw(kwargs["body"]["message"]["raw"])
    # Exactly one occurrence of the original id in the References header value.
    refs_line = next(
        line for line in mime.splitlines() if line.startswith("References:")
    )
    assert refs_line.count("<orig@mail.example.com>") == 1


def test_gmail_client_has_no_send_method():
    # Hard guarantee at the client level: no send-capable public method exists.
    client = GmailClient(MagicMock())
    public = [a for a in dir(client) if not a.startswith("_")]
    assert "create_draft" in public
    assert not any("send" in name.lower() for name in public)


def test_create_draft_produces_valid_parseable_mime():
    import email as email_pkg
    from email.policy import default as default_policy

    service = MagicMock()
    drafts = service.users.return_value.drafts.return_value
    drafts.create.return_value.execute.return_value = {"id": "d", "message": {}}
    client = GmailClient(service)
    client.create_draft(to="x@y.com", subject="Subject Line", body="Body content")

    _, kwargs = drafts.create.call_args
    raw = base64.urlsafe_b64decode(kwargs["body"]["message"]["raw"])
    parsed = email_pkg.message_from_bytes(raw, policy=default_policy)
    assert parsed["To"] == "x@y.com"
    assert parsed["Subject"] == "Subject Line"
    assert "Body content" in parsed.get_content()


def test_create_draft_handles_unicode_body():
    # Turkish characters must not break MIME encoding.
    service = MagicMock()
    drafts = service.users.return_value.drafts.return_value
    drafts.create.return_value.execute.return_value = {"id": "d", "message": {}}
    client = GmailClient(service)

    result = client.create_draft(
        to="ali@example.com",
        subject="Toplantı",
        body="Merhaba, yarın görüşmek üzere. Şükranlar, İ.",
    )
    assert result.draft_id == "d"

    import email as email_pkg
    from email.policy import default as default_policy

    _, kwargs = drafts.create.call_args
    raw = base64.urlsafe_b64decode(kwargs["body"]["message"]["raw"])
    parsed = email_pkg.message_from_bytes(raw, policy=default_policy)
    # The body decodes back to the original Turkish text.
    assert "görüşmek" in parsed.get_content()
