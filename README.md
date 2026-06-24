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
| **2** | Assist: draft replies & new emails with tone/length controls (Gmail **drafts only**) | ✅ done |
| 3 | Stretch: digest, label suggestions, Calendar connector | ⏳ extension points + TODOs only (by design) |

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
    tools.py              # tool schemas + dispatch (read tools + create_draft)
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
   - *"Reply to Alice — short and formal — that I'll send the numbers Friday."*
     → reads the thread, composes the reply, and **creates a Gmail draft**.
   - *"Draft a new email to bob@x.com asking to reschedule to Tuesday."*
     → **creates a new draft**.

Hermes replies in the user's language (Turkish when you write Turkish).

### Drafting (Phase 2) — drafts only, never sends

- Hermes only drafts when you ask it to. It composes the body in the requested
  **tone** (neutral / formal / friendly) and **length** (short / medium),
  defaulting to concise and neutral.
- Replies are threaded correctly (In-Reply-To / References headers + threadId).
- Every draft is created via the Gmail **drafts** API — there is no send path
  anywhere. After creating a draft, Hermes tells you it was **not** sent, points
  you to **Gmail → Drafts**, and reminds you to review and send it manually.

### HTTP API

| Endpoint | Method | Purpose |
|---------|--------|---------|
| `/health` | GET | status + `llm_reachable` + `gmail_authorized` + `allow_send` |
| `/auth/status` | GET | whether a Gmail token is stored |
| `/auth/login` | POST | run the interactive OAuth consent flow (opens a browser) |
| `/api/chat` | POST | `{"messages":[{"role":"user","content":"..."}]}` → `{"reply","tools_used"}` |
| `/` , `/static/*` | GET | chat UI |

The agent runs a **bounded** tool-calling loop (max 5 tool iterations) and
tolerates malformed tool output. It exposes four tools:
`list_recent_emails`, `search_emails`, `get_thread` (read-only), and
`create_draft` (**draft-only** — it calls the Gmail drafts API and never sends).
There is no send tool, and the Gmail client has no send method.

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

The suite (no network, all mocked) covers:
- **LLM providers** — local JSON-tool parsing, Anthropic block parsing, ping.
- **Gmail client** — list/search/thread parsing, body extraction, and
  `create_draft` (new email + reply threading); asserts the client has no send
  method and that the send endpoints are never called.
- **Tools** — dispatch, argument validation, and the no-send-tool guarantee.
- **Orchestrator** — direct answers, native + JSON tool paths, the draft flow,
  graceful tool-error handling, and the iteration bound.
- **App** — `/health`, `/auth/status`, chat (direct / triage / draft), bad
  requests, agent-failure handling, and the served UI.
- **Security** — redaction, locked OAuth scopes, loopback redirect,
  `ALLOW_SEND=false` default.

```
pytest        # 43 passing
```

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

**The no-send guarantee (drafts only).** There is no send code path anywhere:
- `app/gmail/client.py` `create_draft` calls `users().drafts().create` only —
  never `messages().send` / `drafts().send`. A test asserts those endpoints are
  never called and that the client exposes no `*send*` method.
- The agent has no send tool (`tests/test_tools.py::test_no_send_tool_exists`).
- `ALLOW_SEND` defaults to `false`. It exists only to make the guarantee
  auditable — it enables nothing, because there is nothing to enable. If set
  `true`, `app/main.py` prints a loud "has no effect" notice at startup.

**Auditing the security-critical code.** These files are commented for review:
- `app/security/auth_google.py` — locked minimal scopes, Keychain token storage,
  loopback OAuth redirect.
- `app/security/redact.py` — PII/body redaction for logs.
- `app/gmail/client.py` — drafts-only, metadata-only logging.
- `app/main.py` — localhost bind, cloud-provider warning, ALLOW_SEND notice,
  metadata-only logging.

---

## Roadmap & extending Hermes (Phase 3 — not built yet, by design)

Phase 3 items (daily digest, label suggestions, a Calendar connector) are
intentionally **not** implemented. The architecture leaves clean extension
points so they slot in without refactors:

- **New email tool** (e.g. `suggest_labels`, `daily_digest`): (1) append a schema
  to `TOOL_SCHEMAS`, (2) add a `_tool_*` handler, (3) register it in `_DISPATCH`
  in `app/agent/tools.py`. The orchestrator and providers need no changes — both
  the native and JSON tool paths pick it up automatically. TODO markers are in
  `app/agent/tools.py`.
- **New connector** (e.g. Calendar): add a client alongside `app/gmail/client.py`
  (same shape — a thin wrapper over a Google API resource, built from Keychain
  credentials) and expose its operations as tools. Adding a Calendar scope would
  be the one security-relevant change and must be done deliberately (it would
  widen the locked scope list in `app/security/auth_google.py`).
- **Org-wide rollout**: documented in [Google Cloud setup](#2-google-cloud-setup)
  — a service account with domain-wide delegation — intentionally not implemented.

Any Phase 3 work must preserve every guarantee above: drafts-only, minimal
scopes, no content logging, localhost-only, local-LLM-by-default.
