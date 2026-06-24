"""Hermes FastAPI application.

Security-critical points (audit here):
  * The server is bound to 127.0.0.1 only (see ``run()`` and the README run
    command). It must never bind 0.0.0.0 / be exposed to the network.
  * When the cloud LLM provider is selected, a loud startup banner warns that
    email content will leave the machine.
  * Logging is configured to emit operational metadata only. Do not add email
    content to logs anywhere in the app.
  * OAuth uses the locked minimal scopes (gmail.readonly + gmail.compose);
    the token lives in the macOS Keychain. The server never sends email.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.config import LLMProviderName, Settings, get_settings
from app.gmail import GmailClient
from app.llm import LLMProvider, build_provider
from app.security import auth_google
from app.security.redact import safe_meta

logger = logging.getLogger("hermes")

# Loopback only. Centralized so there is a single, auditable definition.
LOCALHOST = "127.0.0.1"

_STATIC_DIR = Path(__file__).parent / "web" / "static"


# --- Request/response models ---------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    # Full conversation history; the latest user turn is last. The server is
    # stateless — the client keeps the history.
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str] = []


# --- Setup helpers --------------------------------------------------------


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _emit_cloud_warning(settings: Settings) -> None:
    """Print a prominent warning when the cloud provider is active."""
    if settings.llm_provider == LLMProviderName.anthropic:
        banner = (
            "\n" + "!" * 72 + "\n"
            "  HERMES: CLOUD LLM PROVIDER ACTIVE (anthropic)\n"
            "  Email content WILL be sent to Anthropic's API.\n"
            "  Use only with organizational approval / non-sensitive test data.\n"
            "  For private use, set LLM_PROVIDER=local (the default).\n"
            + "!" * 72 + "\n"
        )
        print(banner)
        logger.warning("Cloud LLM provider active: email content leaves the machine.")


def _emit_allow_send_notice(settings: Settings) -> None:
    """Make the no-send guarantee explicit even if ALLOW_SEND is flipped on.

    Hermes has NO send code path. ``ALLOW_SEND`` exists only to make this
    auditable: it does not enable sending because there is nothing to enable.
    If it is set true, we say so loudly so an operator is never misled into
    thinking the app can send mail.
    """
    if settings.allow_send:
        banner = (
            "\n" + "*" * 72 + "\n"
            "  HERMES: ALLOW_SEND=true has NO EFFECT.\n"
            "  This MVP has no send capability — only Gmail DRAFTS are created.\n"
            "  Set ALLOW_SEND=false to reflect reality.\n"
            + "*" * 72 + "\n"
        )
        print(banner)
        logger.warning("ALLOW_SEND=true ignored: Hermes only creates drafts, never sends.")


def _get_gmail(app: FastAPI) -> GmailClient:
    """Return a cached Gmail client, building it on first use.

    Raises ``auth_google.GoogleAuthError`` if there is no stored token (the
    caller turns this into a friendly 'please authorize' response). We never
    trigger an interactive browser flow from inside a request.
    """
    if app.state.gmail is None:
        app.state.gmail = GmailClient.from_settings(app.state.settings)
    return app.state.gmail


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)
    _emit_cloud_warning(settings)
    _emit_allow_send_notice(settings)

    app = FastAPI(title="Hermes", version="0.2.0-phase2")

    provider: LLMProvider = build_provider(settings)
    app.state.settings = settings
    app.state.llm = provider
    app.state.gmail = None  # built lazily after authorization

    # --- Health -----------------------------------------------------------

    @app.get("/health")
    def health() -> JSONResponse:
        llm_ok = False
        try:
            llm_ok = app.state.llm.ping()
        except Exception as exc:  # noqa: BLE001 - report type only
            logger.warning("LLM ping raised: %s", type(exc).__name__)

        return JSONResponse(
            {
                "status": "ok",
                "llm_provider": settings.llm_provider.value,
                "llm_reachable": llm_ok,
                "gmail_authorized": auth_google.has_token(),
                "allow_send": settings.allow_send,  # must be false for the MVP
            }
        )

    # --- Auth -------------------------------------------------------------

    @app.get("/auth/status")
    def auth_status() -> JSONResponse:
        return JSONResponse({"authorized": auth_google.has_token()})

    @app.post("/auth/login")
    def auth_login() -> JSONResponse:
        """Run the interactive OAuth consent flow (opens a browser locally).

        This is the only place an interactive flow runs — never from /api/chat.
        The redirect listener binds to 127.0.0.1 only (see auth_google).
        """
        try:
            auth_google.get_credentials(settings, allow_interactive=True)
        except auth_google.GoogleAuthError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        # Reset the cached client so it rebuilds with the fresh token.
        app.state.gmail = None
        logger.info("auth_login %s", safe_meta(outcome="authorized"))
        return JSONResponse({"authorized": True})

    # --- Chat -------------------------------------------------------------

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> Any:
        if not req.messages or req.messages[-1].role != "user":
            return JSONResponse(
                status_code=400,
                content={"error": "Last message must be from the user."},
            )

        try:
            gmail = _get_gmail(app)
        except auth_google.GoogleAuthError:
            return ChatResponse(
                reply=(
                    "Gmail is not authorized yet. Please authorize access first "
                    "(POST /auth/login), then try again."
                ),
                tools_used=[],
            )

        history = [{"role": m.role, "content": m.content} for m in req.messages]
        logger.info("chat %s", safe_meta(turns=len(history)))

        try:
            result = run_agent(app.state.llm, gmail, history)
        except Exception as exc:  # noqa: BLE001 - surface type only
            logger.error("chat_error %s", safe_meta(error_type=type(exc).__name__))
            return JSONResponse(
                status_code=502,
                content={"error": f"Assistant failed: {type(exc).__name__}"},
            )

        return ChatResponse(reply=result.reply, tools_used=result.tools_used)

    # --- Static chat UI ---------------------------------------------------

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))

    logger.info(
        "Hermes started (provider=%s, allow_send=%s).",
        settings.llm_provider.value,
        settings.allow_send,
    )
    return app


app = create_app()


def run() -> None:
    """Run the dev server bound to localhost only."""
    import uvicorn

    settings = get_settings()
    # SECURITY: host is forced to 127.0.0.1; settings.app_host defaults to it
    # and we never pass 0.0.0.0.
    host = settings.app_host if settings.app_host == LOCALHOST else LOCALHOST
    uvicorn.run("app.main:app", host=host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    run()
