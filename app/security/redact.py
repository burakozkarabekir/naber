"""PII / email-content redaction helpers for logging.

SECURITY-CRITICAL. Hermes must never write email bodies, subjects-with-content,
email addresses, or model prompts/outputs containing email text to logs. Logs
may contain only operational metadata: timestamps, message IDs, tool names,
action outcomes, error *types*.

Use :func:`safe_meta` to construct anything that gets logged. When in doubt,
log a count or an ID, not the content.
"""

from __future__ import annotations

import re
from typing import Any

# Matches the local-part and domain of most email addresses. Used to scrub
# addresses that may accidentally appear in free-text passed to the logger.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Long digit runs (phone numbers, IDs that look like phone numbers, etc.).
_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")

_REDACTED = "[REDACTED]"


def redact_email(address: str | None) -> str:
    """Reduce an email address to a non-identifying placeholder.

    We keep only the first character of the local part and the TLD so logs can
    distinguish entries without revealing the address, e.g.
    ``alice@example.com`` -> ``a***@***.com``.
    """
    if not address:
        return ""
    match = _EMAIL_RE.search(address)
    if not match:
        return _REDACTED
    full = match.group(0)
    local, _, domain = full.partition("@")
    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    first = local[0] if local else "*"
    return f"{first}***@***.{tld}" if tld else f"{first}***@***"


def redact_text(text: str | None) -> str:
    """Scrub a free-text string of emails and long digit runs.

    This is a defensive net for strings that *might* contain PII. It does NOT
    make arbitrary email bodies safe to log — bodies should simply never be
    passed to the logger. Prefer logging lengths/counts instead.
    """
    if not text:
        return ""
    scrubbed = _EMAIL_RE.sub(_REDACTED, text)
    scrubbed = _LONG_DIGITS_RE.sub(_REDACTED, scrubbed)
    return scrubbed


def safe_meta(**fields: Any) -> dict[str, Any]:
    """Build a dict of operational metadata that is safe to log.

    Allowed values pass through unchanged (IDs, counts, tool names, outcomes,
    error types). Any key whose name suggests content is replaced with a length
    marker rather than risk leaking the value.

    Two match modes:
      * substring match for content words (``body``, ``subject``, ...) that are
        long enough to be unambiguous;
      * exact match for short address-ish field names (``to``, ``cc``, ...)
        which would over-match if treated as substrings (``to`` ⊂ ``tool``).
    """
    blocked_substrings = (
        "body",
        "subject",
        "snippet",
        "text",
        "address",
        "email",
        "prompt",
        "content",
        "sender",
        "recipient",
    )
    blocked_exact = {"to", "cc", "bcc", "from"}
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if lowered in blocked_exact or any(b in lowered for b in blocked_substrings):
            # Replace the value with a length marker, never the content itself.
            safe[key] = f"<{_length_marker(value)}>"
            continue
        safe[key] = value
    return safe


def _length_marker(value: Any) -> str:
    try:
        return f"len={len(value)}"  # type: ignore[arg-type]
    except TypeError:
        return "present"
