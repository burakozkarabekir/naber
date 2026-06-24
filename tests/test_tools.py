"""Tests for agent tools (dispatch, validation) and the orchestrator loop."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agent import orchestrator, tools
from app.agent.tools import ToolError, execute_tool
from app.gmail.client import DraftResult, EmailSummary, ThreadMessage, ThreadView
from app.llm.base import LLMResponse, ToolCall


# --- Tool dispatch --------------------------------------------------------


def _gmail_stub():
    g = MagicMock()
    g.list_recent_emails.return_value = [
        EmailSummary("m1", "t1", "a@x.com", "Hi", "snippet", "date", True)
    ]
    g.search_emails.return_value = [
        EmailSummary("m2", "t2", "b@x.com", "Re: Hi", "snip2", "date2", False)
    ]
    g.get_thread.return_value = ThreadView(
        "t1", [ThreadMessage("m1", "a@x.com", "me", "date", "Hi", "body text")]
    )
    g.create_draft.return_value = DraftResult(
        draft_id="d1", thread_id="t1", is_reply=False
    )
    return g


def test_execute_list_recent_emails():
    g = _gmail_stub()
    out = execute_tool("list_recent_emails", {"max_results": 5}, g)
    assert isinstance(out, list)
    assert out[0]["id"] == "m1"
    g.list_recent_emails.assert_called_once_with(max_results=5, query=None)


def test_execute_search_requires_query():
    g = _gmail_stub()
    with pytest.raises(ToolError):
        execute_tool("search_emails", {}, g)


def test_execute_get_thread_requires_thread_id():
    g = _gmail_stub()
    with pytest.raises(ToolError):
        execute_tool("get_thread", {}, g)


def test_execute_unknown_tool_raises():
    with pytest.raises(ToolError):
        execute_tool("send_email", {}, _gmail_stub())  # there is no send tool


def test_no_send_tool_exists():
    # Hard guarantee: no tool can send mail in this MVP. create_draft is the
    # only write tool and it only ever creates a draft.
    assert "send_email" not in tools.ALLOWED_TOOLS
    assert "send" not in tools.ALLOWED_TOOLS
    assert tools.ALLOWED_TOOLS == {
        "list_recent_emails",
        "search_emails",
        "get_thread",
        "create_draft",
    }
    # No tool name hints at sending.
    assert not any("send" in name for name in tools.ALLOWED_TOOLS)


def test_execute_create_draft_new_email():
    g = _gmail_stub()
    out = execute_tool(
        "create_draft",
        {"to": "a@x.com", "subject": "Hello", "body": "Hi there"},
        g,
    )
    assert out["draft_id"] == "d1"
    assert out["is_reply"] is False
    assert "review" in out["reminder"].lower()
    g.create_draft.assert_called_once_with(
        to="a@x.com", subject="Hello", body="Hi there", in_reply_to_message_id=None
    )


def test_execute_create_draft_reply_passes_message_id():
    g = _gmail_stub()
    execute_tool(
        "create_draft",
        {
            "to": "a@x.com",
            "subject": "Re: Hi",
            "body": "Thanks!",
            "in_reply_to_message_id": "m1",
        },
        g,
    )
    g.create_draft.assert_called_once_with(
        to="a@x.com", subject="Re: Hi", body="Thanks!", in_reply_to_message_id="m1"
    )


def test_execute_create_draft_validates_required_fields():
    g = _gmail_stub()
    with pytest.raises(ToolError):
        execute_tool("create_draft", {"subject": "s", "body": "b"}, g)  # no 'to'
    with pytest.raises(ToolError):
        execute_tool("create_draft", {"to": "a@x.com", "body": "b"}, g)  # no subject
    with pytest.raises(ToolError):
        execute_tool("create_draft", {"to": "a@x.com", "subject": "s"}, g)  # no body
    with pytest.raises(ToolError):
        execute_tool(
            "create_draft",
            {"to": "a@x.com", "subject": " ", "body": "b"},  # blank subject
            g,
        )


# --- Orchestrator loop (native tool path) ---------------------------------


class _FakeProvider:
    """Scripted provider returning a queued sequence of LLMResponses."""

    def __init__(self, responses, uses_native_tools=True):
        self._responses = list(responses)
        self.uses_native_tools = uses_native_tools
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return self._responses.pop(0)

    def ping(self):
        return True


def test_agent_native_tool_then_answer():
    provider = _FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("list_recent_emails", {"max_results": 3}, "tu1")]),
            LLMResponse(text="Bugün 1 önemli e-posta var."),
        ],
        uses_native_tools=True,
    )
    result = orchestrator.run_agent(provider, _gmail_stub(), [{"role": "user", "content": "neler önemli?"}])

    assert "önemli" in result.reply
    assert result.tools_used == ["list_recent_emails"]
    assert result.iterations == 2


def test_agent_local_json_tool_then_answer():
    provider = _FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("search_emails", {"query": "is:unread"})]),
            LLMResponse(text="No unread from that sender."),
        ],
        uses_native_tools=False,
    )
    result = orchestrator.run_agent(provider, _gmail_stub(), [{"role": "user", "content": "did bob reply?"}])

    assert result.reply == "No unread from that sender."
    assert result.tools_used == ["search_emails"]
    # Local path injects the JSON tool catalog into the system prompt.
    system_msg = provider.calls[0][0]
    assert system_msg["role"] == "system"
    assert "list_recent_emails" in system_msg["content"]


def test_agent_direct_answer_no_tools():
    provider = _FakeProvider([LLMResponse(text="Merhaba!")])
    result = orchestrator.run_agent(provider, _gmail_stub(), [{"role": "user", "content": "selam"}])
    assert result.reply == "Merhaba!"
    assert result.tools_used == []
    assert result.iterations == 1


def test_agent_tool_error_is_handled_gracefully():
    # First the model asks for a tool with bad args (ToolError), then answers.
    provider = _FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("get_thread", {}, "tu1")]),  # missing thread_id
            LLMResponse(text="I couldn't open that thread."),
        ]
    )
    result = orchestrator.run_agent(provider, _gmail_stub(), [{"role": "user", "content": "summarize"}])
    assert result.reply == "I couldn't open that thread."
    # The failed tool is not recorded as used.
    assert result.tools_used == []


def test_agent_create_draft_flow_native():
    # User asks for a reply; model reads thread then drafts.
    provider = _FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("get_thread", {"thread_id": "t1"}, "tu1")]),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "create_draft",
                        {
                            "to": "a@x.com",
                            "subject": "Re: Hi",
                            "body": "Kısa ve resmi yanıt.",
                            "in_reply_to_message_id": "m1",
                        },
                        "tu2",
                    )
                ]
            ),
            LLMResponse(
                text="Taslak oluşturuldu. Gmail → Taslaklar'dan inceleyip gönderin."
            ),
        ]
    )
    g = _gmail_stub()
    g.create_draft.return_value = DraftResult(
        draft_id="d9", thread_id="t1", is_reply=True
    )
    result = orchestrator.run_agent(
        provider, g, [{"role": "user", "content": "buna kısa resmi bir yanıt yaz"}]
    )
    assert result.tools_used == ["get_thread", "create_draft"]
    assert "Taslak" in result.reply
    g.create_draft.assert_called_once()


def test_agent_bounded_by_max_iterations():
    # Model keeps requesting tools forever; loop must terminate and ask for a
    # final answer.
    loop_responses = [
        LLMResponse(tool_calls=[ToolCall("list_recent_emails", {}, f"tu{i}")])
        for i in range(orchestrator.MAX_TOOL_ITERATIONS)
    ]
    final = LLMResponse(text="Final answer.")
    provider = _FakeProvider(loop_responses + [final])
    result = orchestrator.run_agent(provider, _gmail_stub(), [{"role": "user", "content": "go"}])
    assert result.reply == "Final answer."
    assert result.iterations == orchestrator.MAX_TOOL_ITERATIONS
