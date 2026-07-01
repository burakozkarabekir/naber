"""Agent orchestrator — a small, bounded tool-calling loop.

Works with both providers behind the same loop:
  * Native tool calling (Anthropic): tool schemas are passed to the provider and
    tool calls/results are threaded back through the message list.
  * JSON-pattern tools (local models): the tool catalog is injected into the
    system prompt and the provider parses a ``{"tool", "arguments"}`` object.

The loop is bounded by ``MAX_TOOL_ITERATIONS`` and degrades gracefully on
malformed tool output. Logging is metadata-only — no email content, no model
text, no prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent import tools as toolkit
from app.gmail.client import GmailClient
from app.llm.base import LLMProvider
from app.security.redact import safe_meta

logger = logging.getLogger("hermes.agent")

MAX_TOOL_ITERATIONS = 5

# Hermes persona. Read-only triage/summarize/Q&A in Phase 1; drafting in Phase 2.
SYSTEM_PROMPT = """You are Hermes, a concise, professional email assistant for a single user.

You help the user manage their Gmail inbox: you triage, summarize, answer
questions about their email, and DRAFT replies and new emails. You do NOT send
email — you never can, and you must never claim to have sent anything. Every
email you compose is created as a Gmail DRAFT for the user to review and send
themselves.

Guidelines:
- Be concise and professional. Lead with the answer.
- Answer in the user's language. The user writes mostly in Turkish — match their
  language (reply in Turkish when they write in Turkish).
- Use the tools to read real inbox data before answering questions about email.
  Never invent emails, senders, or contents.
- For triage requests ("what needs my attention", "neler önemli"), fetch recent
  emails and group them into these categories:
    1. Needs reply (yanıt bekliyor)
    2. Awaiting others (başkalarından yanıt bekleniyor)
    3. FYI (bilgi amaçlı)
    4. Newsletters & promotions (bülten ve tanıtım)
  Keep each item to a one-line summary (sender + subject + why).
- For "summarize the thread about X": find the thread, read it, then summarize
  the key points, decisions, and any open action items.

Drafting (drafts only — NEVER send):
- Only create a draft when the user asks you to write/draft a reply or a new
  email. Never draft unprompted.
- Compose the body yourself in the requested tone and length:
    * length: short | medium (default) — keep it tight unless asked otherwise;
    * tone: neutral (default) | formal | friendly.
  If the user does not specify, use a concise, neutral, professional tone.
- To reply: first read the relevant thread (get_thread), then call create_draft
  with the original message id as in_reply_to_message_id and subject "Re: ...".
  Derive the recipient from the message you are replying to.
- For a brand-new email: call create_draft with to, subject, and body.
- After create_draft succeeds, tell the user: (a) the draft was created (not
  sent), (b) where to find it (Gmail → Drafts), and (c) that they must review
  and send it manually. Use the reminder/location from the tool result.

- When you have enough information to answer, answer — do not call more tools."""


@dataclass
class AgentResult:
    reply: str
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0


class AgentError(Exception):
    """Raised when the agent loop cannot proceed (e.g. Gmail unavailable)."""


def run_agent(
    provider: LLMProvider,
    gmail: GmailClient,
    user_messages: list[dict[str, Any]],
    memory_text: str | None = None,
) -> AgentResult:
    """Run the bounded tool-calling loop and return the assistant's reply.

    Args:
        provider: the configured LLM provider.
        gmail: a ready Gmail client (already authorized).
        user_messages: prior conversation as ``[{role, content}, ...]`` (the
            latest user turn last). Roles are ``user`` / ``assistant``.
        memory_text: user-authored memory notes (rendered as a bullet list) to
            honor in every reply — signature, tone preferences, standing
            instructions. Injected into the system prompt.
    """
    system_prompt = SYSTEM_PROMPT
    if memory_text:
        system_prompt += (
            "\n\nUser memory — durable preferences and standing instructions the "
            "user has saved. Honor these in every reply and every draft (e.g. "
            "signature, tone, language):\n" + memory_text
        )
    if not provider.uses_native_tools:
        # Local models: drive tools via the JSON instruction in the prompt.
        system_prompt = f"{system_prompt}\n\n{toolkit.render_tools_for_prompt()}"

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(user_messages)

    tools_arg = toolkit.TOOL_SCHEMAS if provider.uses_native_tools else None
    tools_used: list[str] = []

    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        response = provider.chat(messages, tools=tools_arg)

        if not response.wants_tool:
            logger.info(
                "agent_done %s",
                safe_meta(iterations=iteration, tools_used=tools_used),
            )
            return AgentResult(
                reply=response.text.strip(),
                tools_used=tools_used,
                iterations=iteration,
            )

        # Execute the requested tool call(s) and append results.
        if provider.uses_native_tools:
            _apply_native_tool_calls(messages, response, gmail, tools_used)
        else:
            _apply_json_tool_call(messages, response, gmail, tools_used)

    # Loop exhausted: ask once more for a final answer without tools.
    logger.info("agent_max_iterations %s", safe_meta(tools_used=tools_used))
    final = provider.chat(
        messages
        + [
            {
                "role": "user",
                "content": "Please give your best final answer now using what you have.",
            }
        ],
        tools=None,
    )
    return AgentResult(
        reply=final.text.strip(),
        tools_used=tools_used,
        iterations=MAX_TOOL_ITERATIONS,
    )


def _apply_native_tool_calls(
    messages: list[dict[str, Any]],
    response: Any,
    gmail: GmailClient,
    tools_used: list[str],
) -> None:
    """Thread Anthropic-style tool_use / tool_result blocks back into messages."""
    # Reconstruct the assistant turn with its tool_use blocks.
    assistant_content: list[dict[str, Any]] = []
    if response.text:
        assistant_content.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        assistant_content.append(
            {
                "type": "tool_use",
                "id": call.id or call.name,
                "name": call.name,
                "input": call.arguments,
            }
        )
    messages.append({"role": "assistant", "content": assistant_content})

    tool_results: list[dict[str, Any]] = []
    for call in response.tool_calls:
        result_block = _run_one_tool(call.name, call.arguments, gmail, tools_used)
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": call.id or call.name,
                "content": toolkit.tool_result_to_text(result_block["result"]),
                "is_error": result_block["is_error"],
            }
        )
    messages.append({"role": "user", "content": tool_results})


def _apply_json_tool_call(
    messages: list[dict[str, Any]],
    response: Any,
    gmail: GmailClient,
    tools_used: list[str],
) -> None:
    """Feed a local-model JSON tool result back as a plain message pair."""
    call = response.tool_calls[0]
    # Echo the model's tool choice as the assistant turn (text form).
    messages.append(
        {
            "role": "assistant",
            "content": toolkit.tool_result_to_text(
                {"tool": call.name, "arguments": call.arguments}
            ),
        }
    )
    result_block = _run_one_tool(call.name, call.arguments, gmail, tools_used)
    payload = toolkit.tool_result_to_text(result_block["result"])
    prefix = "TOOL ERROR" if result_block["is_error"] else "TOOL RESULT"
    messages.append(
        {
            "role": "user",
            "content": (
                f"{prefix} for {call.name}: {payload}\n"
                "Use this to answer the user, or call another tool if needed."
            ),
        }
    )


def _run_one_tool(
    name: str,
    arguments: dict[str, Any],
    gmail: GmailClient,
    tools_used: list[str],
) -> dict[str, Any]:
    """Execute a single tool, capturing errors as data (never raising)."""
    try:
        result = toolkit.execute_tool(name, arguments, gmail)
        tools_used.append(name)
        return {"result": result, "is_error": False}
    except toolkit.ToolError as exc:
        logger.warning("tool_error %s", safe_meta(tool=name, error_type=type(exc).__name__))
        return {"result": f"Error: {exc}", "is_error": True}
    except Exception as exc:  # noqa: BLE001 - surface type only, keep loop alive
        logger.warning("tool_exception %s", safe_meta(tool=name, error_type=type(exc).__name__))
        return {"result": f"Error executing {name}: {type(exc).__name__}", "is_error": True}
