"""Cloud LLM provider — Anthropic Messages API.

PRIVACY WARNING: when this provider is active, email content is sent to
Anthropic's API. It must only be used with organizational approval and/or
non-sensitive test data. ``app.main`` prints a loud startup banner when this
provider is selected; do not remove that warning.

Uses native tool calling (the Messages API supports it well), unlike the local
provider which drives tools via a JSON instruction pattern.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from app.llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger("hermes.llm.anthropic")

# Keep responses bounded but roomy enough for summaries / drafts.
_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    uses_native_tools = True

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        # Extract any leading system message into the top-level `system` field,
        # which the Messages API expects separately from the turn list.
        system_text, turns = _split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "messages": turns,
        }
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(name=block.name, arguments=dict(block.input), id=block.id)
                )

        return LLMResponse(text="".join(text_parts), tool_calls=tool_calls)

    def ping(self) -> bool:
        """Minimal reachability check. Sends a trivial, content-free prompt."""
        try:
            self._client.messages.create(
                model=self._model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except anthropic.APIError as exc:
            logger.warning("Anthropic ping failed: %s", type(exc).__name__)
            return False


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull system messages out of the turn list into a single system string."""
    system_chunks: list[str] = []
    turns: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_chunks.append(content)
        else:
            turns.append(msg)
    return "\n\n".join(system_chunks), turns
