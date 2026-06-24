"""Google OAuth (installed-app flow) with macOS Keychain token storage.

SECURITY-CRITICAL — review carefully:

* Scopes are LOCKED to the two minimal scopes required by the MVP:
  ``gmail.readonly`` (read) and ``gmail.compose`` (create drafts). No
  ``gmail.modify``, no ``gmail.send``, no full-account scopes. Requesting
  ``gmail.compose`` lets Hermes create drafts but NOT send them.
* The OAuth *token* (access + refresh) is stored in the OS keyring
  (macOS Keychain), never in a plaintext file on disk.
* The OAuth *client* secret (``credentials.json``) stays on disk but is
  gitignored; it is the app identity, not the user's mailbox token.
* Nothing in this module logs token material, email addresses, or message
  content.
"""

from __future__ import annotations

import json
import logging
import os

import keyring
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.config import Settings

logger = logging.getLogger("hermes.auth")

# LOCKED minimal scopes. Do not add to this list. gmail.compose creates drafts
# only — it does not grant send. gmail.readonly is read-only.
SCOPES: list[str] = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

# Keyring identifiers. The token JSON is stored as a single secret entry.
_KEYRING_SERVICE = "hermes-gmail"
_KEYRING_USERNAME = "oauth-token"

# Loopback bind for the OAuth consent redirect. 127.0.0.1 only.
_OAUTH_LOOPBACK_HOST = "127.0.0.1"


class GoogleAuthError(RuntimeError):
    """Raised when authorization cannot be completed."""


def _load_token_from_keychain() -> Credentials | None:
    """Read stored credentials from the OS keyring, if present."""
    raw = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Stored token is unparseable; ignoring and re-authorizing.")
        return None
    return Credentials.from_authorized_user_info(data, SCOPES)


def _save_token_to_keychain(creds: Credentials) -> None:
    """Persist credentials to the OS keyring (macOS Keychain).

    Stored as the standard authorized-user JSON. We never write this to a file.
    """
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, creds.to_json())
    logger.info("OAuth token saved to OS keyring.")


def clear_token() -> None:
    """Remove the stored token from the keyring (used on logout / revoke)."""
    try:
        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        logger.info("OAuth token cleared from OS keyring.")
    except keyring.errors.PasswordDeleteError:
        logger.info("No OAuth token present to clear.")


def has_token() -> bool:
    """Return True if a token exists in the keyring (no validity guarantee)."""
    return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME) is not None


def get_credentials(settings: Settings, *, allow_interactive: bool = True) -> Credentials:
    """Return valid Google credentials, refreshing or authorizing as needed.

    Flow:
      1. Load token from the keyring.
      2. If valid, return it.
      3. If expired but refreshable, refresh silently and re-save.
      4. Otherwise, if ``allow_interactive``, run the installed-app consent
         flow (opens a browser, binds the redirect listener to 127.0.0.1 only)
         and save the resulting token.

    Raises:
        GoogleAuthError: if no valid token can be obtained.
    """
    creds = _load_token_from_keychain()

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token_to_keychain(creds)
            logger.info("OAuth token refreshed.")
            return creds
        except Exception as exc:  # noqa: BLE001 - log type only, never token data
            logger.warning("Token refresh failed (%s); re-authorizing.", type(exc).__name__)

    if not allow_interactive:
        raise GoogleAuthError("No valid token and interactive auth is disabled.")

    creds = _run_installed_app_flow(settings)
    _save_token_to_keychain(creds)
    return creds


def _run_installed_app_flow(settings: Settings) -> Credentials:
    """Run the OAuth installed-app (desktop) consent flow on the loopback."""
    creds_path = os.path.abspath(settings.google_credentials_path)
    if not os.path.exists(creds_path):
        raise GoogleAuthError(
            "credentials.json not found. Download the OAuth *Desktop app* client "
            "from Google Cloud and place it at the path set in "
            "GOOGLE_CREDENTIALS_PATH. See README for setup."
        )

    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    # Bind the temporary consent-redirect server to loopback only. Port 0 lets
    # the OS choose a free ephemeral port.
    creds = flow.run_local_server(host=_OAUTH_LOOPBACK_HOST, port=0, open_browser=True)
    logger.info("Completed interactive OAuth consent.")
    return creds
