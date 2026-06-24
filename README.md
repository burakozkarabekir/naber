# Hermes — Secure Internal AI Email Assistant (MVP)

Hermes is an **internal-use**, **single-user** AI assistant that helps an employee
manage their email. It reads and triages the inbox, summarizes threads, answers
questions about emails, and **drafts** replies. **Gmail is the only integration**
in this MVP; the code is structured so additional connectors (Calendar, etc.) can
be added later behind a clean interface.

> **Security and data privacy are the #1 priority** — above features and polish.
> See [Security & Data Handling](#security--data-handling).

This repository is being built **phase by phase**. The current state is:

| Phase | Scope | Status |
|------|-------|--------|
| **0** | Foundation & security: scaffold, config, OAuth + Keychain, LLM abstraction, `/health` | ✅ done |
| **1** | Read & understand: Gmail read/search/thread, chat UI, triage, summarization, Q&A | ✅ done |
| 2 | Assist: draft replies & new emails (Gmail **drafts only**) | ⏳ not started |
| 3 | Stretch: digest, label suggestions, Calendar connector | ⏳ extension points only |

---

## The non-negotiable guarantees

1. **Never sends email.** Hermes can create Gmail **drafts** only. There is no send
   code path. `ALLOW_SEND` defaults to `false` and must stay `false` for the MVP.
2. **Minimal OAuth scopes:** only `gmail.readonly` + `gmail.compose`. Never
   `gmail.send`, `gmail.modify`, or full-account access. The scope list is locked
   in `app/security/auth_google.py` and asserted by a test.
3. **Secrets never touch git.** `credentials.json`, the OAuth token, and `.env` are
   all gitignored.
4. **Tokens live in the macOS Keychain** (via `keyring`), not in plaintext on disk.
5. **No logging of email content or PII** — only operational metadata.
6. **Localhost only.** The server binds `127.0.0.1`, never `0.0.0.0`.
7. **No third-party telemetry.** Only Gmail + the chosen LLM are contacted.
8. **Local LLM by default**, so email content never leaves the machine. The cloud
   provider is opt-in and clearly flagged.
9. **Minimize data at rest.** Email is processed in memory; bodies are not persisted.

---

## Tech stack

Python 3.11+ · FastAPI + uvicorn · google-api-python-client / google-auth /
google-auth-oauthlib · keyring (Keychain) · pydantic + pydantic-settings ·
httpx (local LLM) · anthropic SDK (cloud, opt-in) · pytest. Frontend is a single
static `index.html` (vanilla JS + Tailwind CDN, no build step) — added in Phase 1.

---

## Project layout

```
app/
  main.py                 # FastAPI app, /health, localhost bind, cloud warning
  config.py               # pydantic-settings config from .env
  security/
    auth_google.py        # OAuth flow + Keychain token storage (locked scopes)
    redact.py             # PII/body redaction helpers for logs
  gmail/
    client.py             # read, search, get_thread (create_draft in Phase 2)
  llm/
    base.py               # LLMProvider interface (+ uses_native_tools flag)
    local_provider.py     # LM Studio (httpx, OpenAI-compatible)
    anthropic_provider.py # Anthropic SDK (opt-in cloud)
    __init__.py           # build_provider() selects provider from config
  agent/
    tools.py              # read-only tool schemas + dispatch to Gmail client
    orchestrator.py       # bounded tool-calling loop + Hermes system prompt
  web/static/index.html   # minimal chat UI (vanilla JS + Tailwind CDN)
tests/                    # mocked provider + security tests
```

---

## Setup

### 1. Python environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in values
```

### 2. Google Cloud setup

1. Create a Google Cloud project.
2. Enable the **Gmail API**.
3. **OAuth consent screen** → User type: **Internal** (only your Workspace org's
   users can authorize; no Google verification needed and data stays in the org).
   Add the two scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.compose`
4. **Credentials** → Create OAuth client ID → Application type: **Desktop app** →
   download as `credentials.json` into the project root (it is gitignored).
5. The first app run that touches Gmail opens a browser for consent; the resulting
   token is stored in the **macOS Keychain** (service `hermes-gmail`). The redirect
   listener binds to `127.0.0.1` only.
6. *(Later org-wide rollout — documented, NOT implemented now)* a service account
   with domain-wide delegation could replace the per-user installed-app flow.

### 3. (Optional) Local LLM — recommended for privacy

Open **LM Studio**, load a model (e.g. `qwen2.5-7b-instruct` or
`llama-3.1-8b-instruct`), and start its server on port `1234`. Hermes talks to its
OpenAI-compatible endpoint at `http://localhost:1234/v1`. Email content stays on
the machine.

### 4. (Optional) Cloud LLM — opt-in, higher quality

Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` in `.env`. **When this is
active, email content is sent to Anthropic's API.** Hermes prints a loud startup
warning. Only use with organizational approval / non-sensitive test data.

---

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
# or: python -m app.main
```

Then check health:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","llm_provider":"local","llm_reachable":true,"allow_send":false}
```

`llm_reachable` is `true` only when the configured LLM backend (LM Studio or
Anthropic) responds to a no-op ping.

Then open the chat UI at **http://127.0.0.1:8000**.

### Using Hermes (Phase 1)

1. **Authorize Gmail** — click *Authorize Gmail* in the UI (or `POST /auth/login`).
   This opens a browser for Google consent once; the token is saved to the
   Keychain. The server never blocks a chat request on a browser flow — chat
   replies with a friendly "please authorize" message until a token exists.
2. **Chat.** Ask things like:
   - *"Bugün neler önemli?"* / *"What needs my attention today?"* → triage into
     **Needs reply / Awaiting others / FYI / Newsletters & promotions**.
   - *"Summarize the thread about the Q3 budget."* → finds and summarizes the thread.
   - *"Did Alice reply to my proposal yet?"* → searches the inbox and answers.

Hermes replies in the user's language (Turkish when you write Turkish).

### HTTP API

| Endpoint | Method | Purpose |
|---------|--------|---------|
| `/health` | GET | status + `llm_reachable` + `gmail_authorized` + `allow_send` |
| `/auth/status` | GET | whether a Gmail token is stored |
| `/auth/login` | POST | run the interactive OAuth consent flow (opens a browser) |
| `/api/chat` | POST | `{"messages":[{"role":"user","content":"..."}]}` → `{"reply","tools_used"}` |
| `/` , `/static/*` | GET | chat UI |

The agent runs a **bounded** tool-calling loop (max 5 tool iterations). It exposes
three **read-only** tools — `list_recent_emails`, `search_emails`, `get_thread` —
and tolerates malformed tool output. There is no send tool.

---

## Configuration (`.env`)

See `.env.example` for the full list. Key entries:

| Variable | Default | Notes |
|---------|---------|-------|
| `LLM_PROVIDER` | `local` | `local` (private) or `anthropic` (opt-in, sends data) |
| `LOCAL_LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio endpoint |
| `LOCAL_LLM_MODEL` | `qwen2.5-7b-instruct` | local model name |
| `ANTHROPIC_API_KEY` | *(empty)* | required only for the cloud provider |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | cloud model |
| `GOOGLE_CREDENTIALS_PATH` | `./credentials.json` | OAuth client (gitignored) |
| `APP_HOST` | `127.0.0.1` | localhost only — enforced in code |
| `APP_PORT` | `8000` | |
| `ALLOW_SEND` | `false` | **must stay false for the MVP** |
| `LOG_LEVEL` | `INFO` | |

---

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Phase 0 covers: both LLM providers (mocked, no network), the redaction helpers,
the locked OAuth scopes, the loopback redirect, and the `ALLOW_SEND=false` default.

---

## Security & Data Handling

**LLM provider abstraction is the key privacy decision.** With the default `local`
provider, email content is sent only to a server on your own machine. The
`anthropic` provider is opt-in, sends content to Anthropic's API, and triggers a
prominent startup warning.

**Data at rest.** Hermes processes email in memory and does **not** persist email
bodies. There is no persistent cache in the MVP. If caching is added later for
performance, it will store only message IDs + subjects + short snippets in a local
SQLite file with a short TTL, and it will be documented here.

**Logging.** Logs contain only operational metadata: timestamps, message IDs, tool
names, action outcomes, error *types*. Email bodies, subjects-with-content,
addresses, and model I/O containing email text are never logged. Use
`app.security.redact.safe_meta(...)` to build anything that gets logged.

**Network.** The server binds `127.0.0.1` only. Outbound calls are limited to (a)
the Gmail API and (b) the configured LLM provider. No analytics or telemetry.

**Auditing the security-critical code.** These files are commented for review:
- `app/security/auth_google.py` — locked minimal scopes, Keychain token storage,
  loopback OAuth redirect.
- `app/security/redact.py` — PII/body redaction for logs.
- `app/main.py` — localhost bind, cloud-provider warning, metadata-only logging.
- The no-send guarantee — there is simply no send code path; `create_draft`
  (Phase 2) creates a Gmail draft only.
