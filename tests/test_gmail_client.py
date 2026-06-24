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
