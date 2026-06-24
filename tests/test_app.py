"""Integration tests for the FastAPI app (endpoints, wiring, draft flow).

The LLM provider and Gmail client are injected via ``app.state`` so no network
or real credentials are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.gmail.client import DraftResult, EmailSummary
from app.llm.base import LLMResponse, ToolCall
from app.main import create_app


class _FakeProvider:
    def __init__(self, responses, uses_native_tools=True):
        self._responses = list(responses)
        self.uses_native_tools = uses_native_tools
        self._ping = True

    def chat(self, messages, tools=None):
        return self._responses.pop(0)

    def ping(self):
        return self._ping


def _client(provider=None, gmail=None):
    app = create_app()
    if provider is not None:
        app.state.llm = provider
    if gmail is not None:
        app.state.gmail = gmail
    return TestClient(app)


def test_health_reports_status():
    provider = _FakeProvider([])
    c = _client(provider=provider)
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_reachable"] is True
    assert body["allow_send"] is False
    assert "gmail_authorized" in body


def test_auth_status_endpoint():
    c = _client(provider=_FakeProvider([]))
    assert "authorized" in c.get("/auth/status").json()


def test_chat_rejects_non_user_last_message():
    c = _client(provider=_FakeProvider([]), gmail=MagicMock())
    r = c.post("/api/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert r.status_code == 400


def test_chat_direct_answer():
    provider = _FakeProvider([LLMResponse(text="Merhaba!")])
    c = _client(provider=provider, gmail=MagicMock())
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "selam"}]})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] == "Merhaba!"
    assert data["tools_used"] == []


def test_chat_triage_uses_tool():
    gmail = MagicMock()
    gmail.list_recent_emails.return_value = [
        EmailSummary("m1", "t1", "a@x.com", "Hi", "snip", "date", True)
    ]
    provider = _FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("list_recent_emails", {"max_results": 5}, "tu1")]),
            LLMResponse(text="1 e-posta yanıt bekliyor."),
        ]
    )
    c = _client(provider=provider, gmail=gmail)
    r = c.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "bugün neler önemli?"}]},
    )
    data = r.json()
    assert data["tools_used"] == ["list_recent_emails"]
    assert "yanıt" in data["reply"]


def test_chat_create_draft_flow():
    gmail = MagicMock()
    gmail.create_draft.return_value = DraftResult(
        draft_id="d1", thread_id="t1", is_reply=False
    )
    provider = _FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "create_draft",
                        {"to": "a@x.com", "subject": "Hi", "body": "Hello"},
                        "tu1",
                    )
                ]
            ),
            LLMResponse(text="Taslak oluşturuldu, lütfen Gmail'den inceleyip gönderin."),
        ]
    )
    c = _client(provider=provider, gmail=gmail)
    r = c.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "a@x.com'a kısa bir e-posta yaz"}]},
    )
    data = r.json()
    assert data["tools_used"] == ["create_draft"]
    assert "Taslak" in data["reply"]
    gmail.create_draft.assert_called_once()


def test_chat_handles_agent_failure_gracefully():
    provider = _FakeProvider([])  # empty -> chat() will pop from empty list -> error

    gmail = MagicMock()
    c = _client(provider=provider, gmail=gmail)
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "error" in r.json()


def test_index_served():
    c = _client(provider=_FakeProvider([]))
    r = c.get("/")
    assert r.status_code == 200
    assert "Hermes" in r.text
