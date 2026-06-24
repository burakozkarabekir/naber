"""LLMProvider interface shared by the local and cloud backends.

The agent orchestrator talks only to this interface, so swapping providers (or
adding new ones later) never touches agent logic. The key privacy property is
encoded by *which* implementation is selected: the ``local`` provider keeps
email content on the machine; the ``anthropic`` provider sends it to the cloud.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool/function invocation requested by the model."""

    name: str
    arguments: dict[str, Any]
    # Provider-specific id used to correlate a tool result back to this call
    # (Anthropic tool_use id). May be empty for the local JSON-pattern path.
    id: str = ""


@dataclass
class LLMResponse:
    """Normalized response from any provider."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Minimal interface every LLM backend implements."""

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run one chat completion.

        Args:
            messages: chat messages in the common ``{"role", "content"}`` shape.
            tools: optional list of tool schemas (JSON-Schema style) the model
                may call. ``None`` means plain text generation.

        Returns:
            An :class:`LLMResponse` with text and/or requested tool calls.
        """
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> bool:
        """Lightweight reachability check. Must not send email content.

        Returns True if the backend appears reachable/usable, else False.
        """
        raise NotImplementedError
