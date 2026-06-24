"""Local LLM provider — LM Studio's OpenAI-compatible endpoint.

Privacy: this is the DEFAULT provider. All requests go to a server on the local
machine (``http://localhost:1234/v1`` by default), so email content never
leaves the device.

Tool calling: small local models (7-8B) have weak native tool calling, so we
do NOT rely on the OpenAI ``tools`` parameter. Instead the orchestrator injects
a strict-JSON instruction and we parse the model's reply defensively here,
extracting a ``{"tool": ..., "arguments": {...}}`` object if present.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger("hermes.llm.local")


class LocalLLMProvider(LLMProvider):
    name = "local"
    uses_native_tools = False

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
        }
        # We intentionally do not pass `tools` to the local endpoint; tool
        # selection is driven by the JSON-instruction pattern in the prompt.
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Local LLM request failed: %s", type(exc).__name__)
            raise

        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )

        if tools:
            tool_call = self._extract_tool_call(content)
            if tool_call is not None:
                return LLMResponse(text="", tool_calls=[tool_call])

        return LLMResponse(text=content)

    @staticmethod
    def _extract_tool_call(content: str) -> ToolCall | None:
        """Best-effort parse of a tool request from free-form model output.

        Expected (instructed) shape:
            {"tool": "list_recent_emails", "arguments": {"max_results": 10}}

        Local models often wrap that JSON in prose or markdown fences. We:
          1. try parsing the whole (stripped) content as JSON;
          2. otherwise scan for *balanced* ``{...}`` objects and try each one,
             returning the first that looks like a tool call.

        A naive greedy ``{.*}`` regex would span from the first ``{`` to the
        last ``}`` and fail to parse whenever the model emits any other braces —
        silently dropping a valid tool call. The balanced scan avoids that.

        Returns None if no well-formed tool object is found (the orchestrator
        then treats the output as a final text answer).
        """
        stripped = content.strip()
        # Strip a leading/trailing markdown code fence if present.
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            # Drop an optional language tag on the first line (e.g. ```json).
            stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped

        candidates: list[str] = []
        whole = stripped.strip()
        if whole.startswith("{"):
            candidates.append(whole)
        candidates.extend(_balanced_json_objects(content))

        for candidate in candidates:
            tool_call = _as_tool_call(candidate)
            if tool_call is not None:
                return tool_call
        return None

    def ping(self) -> bool:
        """Check LM Studio reachability via the models endpoint (no content)."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self._base_url}/models")
                return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("Local LLM ping failed: %s", type(exc).__name__)
            return False


def _balanced_json_objects(text: str) -> list[str]:
    """Yield top-level balanced ``{...}`` substrings, ignoring braces in strings.

    Tracks string state and escapes so braces inside JSON string values don't
    throw off the depth count. Returns candidates in order of appearance.
    """
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start : i + 1])
                    start = -1
    return objects


def _as_tool_call(candidate: str) -> ToolCall | None:
    """Parse a JSON string into a ToolCall if it names a tool, else None."""
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    tool_name = obj.get("tool") or obj.get("name")
    if not tool_name or not isinstance(tool_name, str):
        return None
    args = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=tool_name, arguments=args)
