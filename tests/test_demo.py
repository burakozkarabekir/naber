"""Tests for DEMO_MODE: fake mailbox, scripted provider, and app wiring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.demo.fake_gmail import FakeGmailClient
from app.demo.provider import DemoLLMProvider
from app.gmail.client import DraftResult, EmailSummary, ThreadView
from app.main import create_app


# --- FakeGmailClient ------------------------------------------------------


def test_fake_gmail_lists_and_searches():
    g = FakeGmailClient()
    emails = g.list_recent_emails()
    assert len(emails) == 8
    assert all(isinstance(e, EmailSummary) for e in emails)

    hits = g.search_emails("bütçe")
    assert any("Ayşe" in e.sender for e in hits)

    unread = g.search_emails("is:unread")
    assert unread and all(e.unread for e in unread)


def test_fake_gmail_get_thread():
    g = FakeGmailClient()
    thread = g.get_thread("t-budget")
    assert isinstance(thread, ThreadView)
    assert len(thread.messages) == 3


def test_fake_gmail_create_draft_never_sends():
    g = FakeGmailClient()
    res = g.create_draft(to="x@y.com", subject="Hi", body="Body")
    assert isinstance(res, DraftResult)
    assert res.draft_id.startswith("demo-draft-")
    assert res.is_reply is False
    assert g.created_drafts[0]["to"] == "x@y.com"
    # No send-capable method exists.
    assert not any("send" in a.lower() for a in dir(g) if not a.startswith("_"))


# --- DemoLLMProvider through the real orchestrator ------------------------


def _run(query: str):
    from app.agent import run_agent

    provider = DemoLLMProvider()
    gmail = FakeGmailClient()
    return run_agent(provider, gmail, [{"role": "user", "content": query}])


def test_demo_triage_groups_all_categories():
    r = _run("Bugün neler önemli?")
    assert r.tools_used == ["list_recent_emails"]
    for label in ("Yanıt bekliyor", "Başkalarından", "Bilgi amaçlı", "Bülten"):
        assert label in r.reply


def test_demo_summary_has_action_item():
    r = _run("Q3 bütçe yazışmasını özetle")
    assert r.tools_used == ["get_thread"]
    assert "Açık aksiyon" in r.reply
    assert "onay" in r.reply.lower()


def test_demo_search_reply_check():
    r = _run("David teklife döndü mü?")
    assert r.tools_used == ["search_emails"]
    assert "David" in r.reply


def test_demo_draft_reply_formal_short():
    r = _run("Ayşe'ye kısa ve resmi bir yanıt taslağı yaz: rakamları Cuma göndereceğim.")
    assert r.tools_used == ["create_draft"]
    assert "Taslak oluşturuldu" in r.reply
    assert "gönderilmedi" in r.reply.lower()
    assert "ayse.yilmaz@acmeholding.com" in r.reply
    assert "Sayın Ayşe" in r.reply  # formal tone
    assert "Cuma" in r.reply  # gist carried through


def test_demo_draft_new_email_to_address():
    r = _run("bob@northwind.io'ya toplantıyı Salı'ya erteleme taslağı yaz")
    assert r.tools_used == ["create_draft"]
    assert "bob@northwind.io" in r.reply
    assert "ertele" in r.reply.lower()


def test_demo_greeting_lists_capabilities():
    r = _run("selam")
    assert r.tools_used == []
    assert "Hermes" in r.reply


def test_demo_draft_without_recipient_asks():
    # "yaz" with no address and no known name -> ask, don't crash.
    r = _run("bir taslak yaz")
    assert r.tools_used == []
    assert "kime" in r.reply.lower()


# --- App wired in demo mode (no network, no auth) -------------------------


def _demo_client() -> TestClient:
    # create_app() reads the module-level get_settings(); point it at a
    # demo-configured Settings for the duration of the build.
    from app import main as main_mod

    real = main_mod.get_settings
    main_mod.get_settings = lambda: Settings(_env_file=None, demo_mode=True)
    try:
        return TestClient(create_app())
    finally:
        main_mod.get_settings = real


def test_app_demo_health_and_chat():
    c = _demo_client()
    h = c.get("/health").json()
    assert h["demo"] is True
    assert h["gmail_authorized"] is True
    assert h["allow_send"] is False

    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "Bugün neler önemli?"}]})
    data = r.json()
    assert data["tools_used"] == ["list_recent_emails"]
    assert "Yanıt bekliyor" in data["reply"]
