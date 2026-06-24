"""Tests for the LLM providers (mocked — no real network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.llm.base import LLMResponse, ToolCall
from app.llm.local_provider import LocalLLMProvider


def _chat_completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_local_provider_returns_text():
    provider = LocalLLMProvider(base_url="http://localhost:1234/v1", model="m")

    fake_resp = MagicMock()
    fake_resp.json.return_value = _chat_completion("Merhaba, nasıl yardımcı olabilirim?")
    fake_resp.raise_for_status.return_value = None

    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp

    with patch("app.llm.local_provider.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = fake_client
        result = provider.chat([{"role": "user", "content": "selam"}])

    assert isinstance(result, LLMResponse)
    assert "Merhaba" in result.text
    assert not result.wants_tool


def test_local_provider_extracts_tool_call_from_json():
    provider = LocalLLMProvider(base_url="http://localhost:1234/v1", model="m")

    content = (
        'Sure, I will look. {"tool": "list_recent_emails", '
        '"arguments": {"max_results": 5}}'
    )
    fake_resp = MagicMock()
    fake_resp.json.return_value = _chat_completion(content)
    fake_resp.raise_for_status.return_value = None

    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp

    tools = [{"name": "list_recent_emails"}]
    with patch("app.llm.local_provider.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = fake_client
        result = provider.chat([{"role": "user", "content": "show emails"}], tools=tools)

    assert result.wants_tool
    call = result.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "list_recent_emails"
    assert call.arguments == {"max_results": 5}


def test_local_provider_malformed_tool_json_is_treated_as_text():
    provider = LocalLLMProvider(base_url="http://localhost:1234/v1", model="m")

    fake_resp = MagicMock()
    fake_resp.json.return_value = _chat_completion("no json here at all")
    fake_resp.raise_for_status.return_value = None

    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp

    with patch("app.llm.local_provider.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = fake_client
        result = provider.chat([{"role": "user", "content": "x"}], tools=[{"name": "t"}])

    assert not result.wants_tool
    assert result.text == "no json here at all"


def test_local_provider_ping_ok():
    provider = LocalLLMProvider(base_url="http://localhost:1234/v1", model="m")
    fake_resp = MagicMock(status_code=200)
    fake_client = MagicMock()
    fake_client.get.return_value = fake_resp

    with patch("app.llm.local_provider.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = fake_client
        assert provider.ping() is True


def test_anthropic_provider_parses_text_and_tool_calls():
    from app.llm.anthropic_provider import AnthropicProvider

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Taslak oluşturdum."

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "create_draft"
    tool_block.input = {"to": "x@y.com", "subject": "s", "body": "b"}
    tool_block.id = "tu_1"

    fake_msg = MagicMock()
    fake_msg.content = [text_block, tool_block]

    with patch("app.llm.anthropic_provider.anthropic.Anthropic") as anthropic_cls:
        client = anthropic_cls.return_value
        client.messages.create.return_value = fake_msg
        provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-6")
        result = provider.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "reply please"},
            ],
            tools=[{"name": "create_draft"}],
        )

    assert "Taslak" in result.text
    assert result.wants_tool
    assert result.tool_calls[0].name == "create_draft"
    # The system message must be lifted out of the turn list.
    _, kwargs = client.messages.create.call_args
    assert kwargs["system"] == "sys"
    assert all(m["role"] != "system" for m in kwargs["messages"])


def test_anthropic_provider_requires_api_key():
    from app.llm.anthropic_provider import AnthropicProvider

    try:
        AnthropicProvider(api_key="", model="claude-sonnet-4-6")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing API key")
