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
import re
from typing import Any

import httpx

from app.llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger("hermes.llm.local")

# Matches the first balanced-looking JSON object in a string. Local models often
# wrap JSON in prose or markdown fences, so we search rather than json.loads().
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class LocalLLMProvider(LLMProvider):
    name = "local"

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

        Returns None if no well-formed tool object is found (the orchestrator
        then treats the output as a final text answer).
        """
        match = _JSON_OBJ_RE.search(content)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
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

    def ping(self) -> bool:
        """Check LM Studio reachability via the models endpoint (no content)."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self._base_url}/models")
                return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("Local LLM ping failed: %s", type(exc).__name__)
            return False
