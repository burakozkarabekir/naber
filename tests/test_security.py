"""Tests for security-critical helpers: redaction, OAuth scopes, no-send guard."""

from __future__ import annotations

from app.config import Settings
from app.security import auth_google
from app.security.redact import redact_email, redact_text, safe_meta


def test_redact_email_hides_local_and_domain():
    assert redact_email("alice@example.com") == "a***@***.com"
    assert redact_email("") == ""
    assert redact_email("not-an-email") == "[REDACTED]"


def test_redact_text_scrubs_emails_and_long_digits():
    out = redact_text("contact bob@corp.io or call 5551234567")
    assert "bob@corp.io" not in out
    assert "5551234567" not in out
    assert "[REDACTED]" in out


def test_safe_meta_drops_content_keys():
    meta = safe_meta(
        message_id="abc123",
        count=3,
        tool="list_recent_emails",
        subject="Confidential Q3 numbers",
        body="secret body text",
        to="ceo@corp.io",
    )
    # Operational metadata passes through.
    assert meta["message_id"] == "abc123"
    assert meta["count"] == 3
    assert meta["tool"] == "list_recent_emails"
    # Content-bearing keys are replaced with length markers, never the value.
    assert "Confidential" not in str(meta["subject"])
    assert "secret body" not in str(meta["body"])
    assert "ceo@corp.io" not in str(meta["to"])
    assert meta["subject"].startswith("<")


def test_oauth_scopes_are_minimal_and_locked():
    # Hard guarantee: only readonly + compose, nothing that can send/modify.
    assert auth_google.SCOPES == [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ]
    joined = " ".join(auth_google.SCOPES)
    assert "gmail.send" not in joined
    assert "gmail.modify" not in joined
    assert "https://mail.google.com/" not in joined  # full-access scope


def test_oauth_redirect_bound_to_loopback():
    assert auth_google._OAUTH_LOOPBACK_HOST == "127.0.0.1"


def test_allow_send_defaults_false():
    # The MVP must never enable sending.
    settings = Settings(_env_file=None)
    assert settings.allow_send is False
