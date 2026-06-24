"""Agent tools — schemas and dispatch to the Gmail client.

Each tool is defined once (name, description, JSON-Schema inputs) and used by
both providers:
  * the Anthropic provider passes ``TOOL_SCHEMAS`` to native tool calling;
  * the local provider receives the same schemas rendered into a strict-JSON
    instruction (see :func:`render_tools_for_prompt`).

Phase 1 exposes read-only tools. ``create_draft`` (Phase 2) will be appended
here as a draft-only tool — there is intentionally no send tool, ever.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.gmail.client import GmailClient
from app.security.redact import safe_meta

logger = logging.getLogger("hermes.agent.tools")

# Anthropic-style tool schemas (also the single source of truth for the local
# JSON-instruction pattern). Keep descriptions prescriptive about *when* to use
# each tool — small models benefit from explicit trigger conditions.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_recent_emails",
        "description": (
            "List the most recent emails in the inbox. Use this for triage "
            "requests like 'what needs my attention' or 'what's new'. Optionally "
            "pass a Gmail query to narrow the list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "How many emails to fetch (1-50). Default 10.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional Gmail query, e.g. 'is:unread newer_than:3d'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_emails",
        "description": (
            "Search emails using Gmail query syntax (e.g. "
            "'from:alice@x.com is:unread newer_than:7d', 'subject:budget'). Use "
            "this to answer questions like 'did X reply yet?' or to find a thread."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A Gmail search query.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_thread",
        "description": (
            "Fetch the full content of an email thread by its thread_id, for "
            "summarization or detailed questions. Get the thread_id first from "
            "list_recent_emails or search_emails."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The Gmail thread_id to retrieve.",
                },
            },
            "required": ["thread_id"],
        },
    },
    {
        "name": "create_draft",
        "description": (
            "Create a Gmail DRAFT — this NEVER sends. Use only when the user asks "
            "you to draft/write a reply or a new email. Compose the 'body' "
            "yourself in the tone and length the user requested (short / neutral "
            "/ formal; default neutral and concise). For a reply, first read the "
            "thread, then pass the original message id as in_reply_to_message_id "
            "and set subject to 'Re: ...'. After it succeeds, tell the user the "
            "draft was created, where to find it, and that they must review and "
            "send it manually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address(es), comma-separated.",
                },
                "subject": {"type": "string", "description": "Subject line."},
                "body": {
                    "type": "string",
                    "description": "The composed message body (plain text).",
                },
                "in_reply_to_message_id": {
                    "type": "string",
                    "description": (
                        "Optional. The Gmail message id this is a reply to, so the "
                        "draft threads correctly."
                    ),
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
]

# Tool names the agent is allowed to call in this phase.
ALLOWED_TOOLS = {schema["name"] for schema in TOOL_SCHEMAS}


class ToolError(Exception):
    """Raised when a tool cannot be executed (bad args, unknown tool)."""


def _tool_list_recent_emails(client: GmailClient, args: dict[str, Any]) -> Any:
    return [
        e.to_dict()
        for e in client.list_recent_emails(
            max_results=args.get("max_results", 10),
            query=args.get("query"),
        )
    ]


def _tool_search_emails(client: GmailClient, args: dict[str, Any]) -> Any:
    query = args.get("query")
    if not query or not isinstance(query, str):
        raise ToolError("search_emails requires a non-empty 'query' string.")
    return [e.to_dict() for e in client.search_emails(query)]


def _tool_get_thread(client: GmailClient, args: dict[str, Any]) -> Any:
    thread_id = args.get("thread_id")
    if not thread_id or not isinstance(thread_id, str):
        raise ToolError("get_thread requires a 'thread_id' string.")
    return client.get_thread(thread_id).to_dict()


def _tool_create_draft(client: GmailClient, args: dict[str, Any]) -> Any:
    """Create a Gmail draft (never sends). Validates required fields."""
    to = args.get("to")
    subject = args.get("subject")
    body = args.get("body")
    if not to or not isinstance(to, str):
        raise ToolError("create_draft requires a 'to' address.")
    if not isinstance(subject, str) or not subject.strip():
        raise ToolError("create_draft requires a non-empty 'subject'.")
    if not isinstance(body, str) or not body.strip():
        raise ToolError("create_draft requires a non-empty 'body'.")
    in_reply_to = args.get("in_reply_to_message_id")
    if in_reply_to is not None and not isinstance(in_reply_to, str):
        raise ToolError("'in_reply_to_message_id' must be a string if provided.")

    result = client.create_draft(
        to=to,
        subject=subject,
        body=body,
        in_reply_to_message_id=in_reply_to or None,
    )
    # The result carries the draft location and the review-&-send reminder so the
    # agent always relays them to the user.
    return result.to_dict()


# Dispatch table mapping tool name -> handler(client, args) -> JSON-able result.
#
# Phase 3 (TODO — do NOT build yet; clean extension points only):
#   * "suggest_labels"   -> Gmail label suggestions (read labels + classify).
#   * "daily_digest"     -> compose a digest (reuses list/search/get_thread).
#   * calendar tools     -> a new connector (see app/connectors note below).
# Adding a tool = (1) append a schema to TOOL_SCHEMAS, (2) add a _tool_* handler,
# (3) register it here. No orchestrator changes needed.
_DISPATCH: dict[str, Callable[[GmailClient, dict[str, Any]], Any]] = {
    "list_recent_emails": _tool_list_recent_emails,
    "search_emails": _tool_search_emails,
    "get_thread": _tool_get_thread,
    "create_draft": _tool_create_draft,
}


def execute_tool(name: str, arguments: dict[str, Any], client: GmailClient) -> Any:
    """Execute a tool by name and return a JSON-serializable result.

    Raises:
        ToolError: for unknown tools or invalid arguments.
    """
    if name not in _DISPATCH:
        raise ToolError(f"Unknown tool: {name}")
    handler = _DISPATCH[name]
    # Log the call as metadata only — never the arguments (may contain queries
    # with addresses) or the result (email content).
    logger.info("tool_call %s", safe_meta(tool=name, arg_keys=sorted(arguments.keys())))
    return handler(client, arguments or {})


def render_tools_for_prompt() -> str:
    """Render the tool catalog as a strict-JSON instruction for local models.

    Used only by the local provider path. Anthropic uses native tool calling.
    """
    lines = [
        "You can call ONE tool by replying with ONLY a JSON object on its own, "
        'in the form: {"tool": "<name>", "arguments": { ... }}.',
        "Do not wrap it in prose or markdown. If you do NOT need a tool, just "
        "answer the user normally in plain text.",
        "",
        "Available tools:",
    ]
    for schema in TOOL_SCHEMAS:
        props = schema["input_schema"].get("properties", {})
        arg_desc = ", ".join(
            f"{k} ({v.get('type', 'any')})" for k, v in props.items()
        )
        lines.append(f"- {schema['name']}: {schema['description']}")
        lines.append(f"    arguments: {arg_desc or 'none'}")
    return "\n".join(lines)


def tool_result_to_text(result: Any) -> str:
    """Serialize a tool result for feeding back into the conversation."""
    return json.dumps(result, ensure_ascii=False, default=str)
