"""Hermes FastAPI application — Phase 0 foundation.

Security-critical points (audit here):
  * The server is bound to 127.0.0.1 only (see ``run()`` and the README run
    command). It must never bind 0.0.0.0 / be exposed to the network.
  * When the cloud LLM provider is selected, a loud startup banner warns that
    email content will leave the machine.
  * Logging is configured to emit operational metadata only. Do not add email
    content to logs anywhere in the app.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import LLMProviderName, Settings, get_settings
from app.llm import LLMProvider, build_provider

logger = logging.getLogger("hermes")

# Loopback only. Centralized so there is a single, auditable definition.
LOCALHOST = "127.0.0.1"


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
        # Printed (not just logged) so it is impossible to miss at startup.
        print(banner)
        logger.warning("Cloud LLM provider active: email content leaves the machine.")


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)
    _emit_cloud_warning(settings)

    app = FastAPI(title="Hermes", version="0.0.1-phase0")

    # Build the LLM provider once and stash it on app state.
    provider: LLMProvider = build_provider(settings)
    app.state.settings = settings
    app.state.llm = provider

    @app.get("/health")
    def health() -> JSONResponse:
        """Liveness + dependency check.

        Returns operational status only — never email content. Includes a
        no-op "ping" of the configured LLM so the operator can confirm the
        backend is reachable.
        """
        llm_ok = False
        try:
            llm_ok = provider.ping()
        except Exception as exc:  # noqa: BLE001 - report type only
            logger.warning("LLM ping raised: %s", type(exc).__name__)

        return JSONResponse(
            {
                "status": "ok",
                "llm_provider": settings.llm_provider.value,
                "llm_reachable": llm_ok,
                "allow_send": settings.allow_send,  # must be false for the MVP
            }
        )

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
