"""Tests for the user memory feature: store, prompt injection, API, demo use."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent import run_agent, orchestrator
from app.config import Settings
from app.demo.fake_gmail import FakeGmailClient
from app.demo.provider import DemoLLMProvider
from app.llm.base import LLMResponse
from app.main import create_app
from app.memory import MAX_NOTES, MemoryStore


# --- MemoryStore ----------------------------------------------------------


def test_store_roundtrip_persists_across_instances(tmp_path):
    path = tmp_path / "mem.json"
    store = MemoryStore(path)
    assert store.get_notes() == []

    store.set_notes(["İmza: Burak", "Yanıtlar kısa olsun"])
    # A fresh instance reads the same file.
    again = MemoryStore(path)
    assert again.get_notes() == ["İmza: Burak", "Yanıtlar kısa olsun"]


def test_store_defaults_until_first_save(tmp_path):
    path = tmp_path / "mem.json"
    store = MemoryStore(path, defaults=["örnek not"])
    # Defaults are visible but nothing is written to disk yet.
    assert store.get_notes() == ["örnek not"]
    assert not path.exists()
    # Saving replaces the defaults and creates the file.
    store.set_notes(["gerçek not"])
    assert path.exists()
    assert store.get_notes() == ["gerçek not"]


def test_store_cleans_and_bounds_notes(tmp_path):
    store = MemoryStore(tmp_path / "mem.json")
    saved = store.set_notes(["  a  ", "", "a", "b" * 1000] + [f"n{i}" for i in range(50)])
    assert saved[0] == "a"          # trimmed, deduplicated
    assert len(saved[1]) == 300     # clamped to MAX_NOTE_LEN
    assert len(saved) <= MAX_NOTES  # bounded


def test_store_add_note_dedupes(tmp_path):
    store = MemoryStore(tmp_path / "mem.json")
    store.add_note("İmza: Burak")
    store.add_note("İmza: Burak")
    assert store.get_notes() == ["İmza: Burak"]


def test_as_prompt_renders_bullets(tmp_path):
    store = MemoryStore(tmp_path / "mem.json")
    assert store.as_prompt() == ""
    store.set_notes(["bir", "iki"])
    assert store.as_prompt() == "- bir\n- iki"


def test_store_survives_corrupt_file(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = MemoryStore(path, defaults=["yedek"])
    assert store.get_notes() == ["yedek"]


# --- Orchestrator injection -------------------------------------------------


class _CaptureProvider:
    uses_native_tools = True

    def __init__(self):
        self.seen_system = ""

    def chat(self, messages, tools=None):
        self.seen_system = messages[0]["content"]
        return LLMResponse(text="ok")

    def ping(self):
        return True


def test_run_agent_injects_memory_into_system_prompt():
    provider = _CaptureProvider()
    run_agent(provider, FakeGmailClient(), [{"role": "user", "content": "selam"}],
              memory_text="- İmza: Burak\n- Ton: resmi")
    assert "User memory" in provider.seen_system
    assert "İmza: Burak" in provider.seen_system
    # Base prompt is still intact.
    assert provider.seen_system.startswith(orchestrator.SYSTEM_PROMPT.split("\n")[0])


def test_run_agent_without_memory_unchanged():
    provider = _CaptureProvider()
    run_agent(provider, FakeGmailClient(), [{"role": "user", "content": "selam"}])
    assert "User memory" not in provider.seen_system


# --- Demo provider honors memory ---------------------------------------------


def _demo(memory: MemoryStore):
    provider = DemoLLMProvider(memory=memory)
    return lambda q: run_agent(provider, FakeGmailClient(),
                               [{"role": "user", "content": q}],
                               memory_text=memory.as_prompt())


def test_demo_draft_uses_memory_signature_and_tone(tmp_path):
    mem = MemoryStore(tmp_path / "mem.json")
    mem.set_notes(["İmza: Burak Özkarabekir", "Varsayılan ton: resmi"])
    ask = _demo(mem)
    r = ask("Ayşe'ye yanıt taslağı yaz: rakamları Cuma göndereceğim")
    assert "Burak Özkarabekir" in r.reply   # signature from memory
    assert "Sayın Ayşe" in r.reply          # formal default from memory


def test_demo_explicit_tone_overrides_memory(tmp_path):
    mem = MemoryStore(tmp_path / "mem.json")
    mem.set_notes(["Varsayılan ton: resmi"])
    ask = _demo(mem)
    r = ask("Ayşe'ye samimi bir yanıt taslağı yaz: teşekkürler")
    assert "Selam Ayşe" in r.reply


def test_demo_memory_show_lists_notes(tmp_path):
    mem = MemoryStore(tmp_path / "mem.json")
    mem.set_notes(["İmza: Burak"])
    ask = _demo(mem)
    r = ask("hafızanda ne var?")
    assert "İmza: Burak" in r.reply
    assert r.tools_used == []


def test_demo_memory_add_via_chat(tmp_path):
    mem = MemoryStore(tmp_path / "mem.json")
    ask = _demo(mem)
    r = ask("hafızana ekle: yanıtlar her zaman İngilizce olsun")
    assert "ekledim" in r.reply.lower()
    assert "yanıtlar her zaman İngilizce olsun" in mem.get_notes()


# --- API ---------------------------------------------------------------------


def _client(tmp_path) -> TestClient:
    from app import main as main_mod

    real = main_mod.get_settings
    main_mod.get_settings = lambda: Settings(
        _env_file=None, demo_mode=True, memory_path=str(tmp_path / "mem.json")
    )
    try:
        return TestClient(create_app())
    finally:
        main_mod.get_settings = real


def test_memory_api_roundtrip(tmp_path):
    c = _client(tmp_path)
    # Demo defaults are visible initially.
    initial = c.get("/api/memory").json()["notes"]
    assert any("İmza" in n for n in initial)

    r = c.put("/api/memory", json={"notes": ["İmza: Test Kullanıcı", "Kısa yaz"]})
    assert r.status_code == 200
    assert r.json()["notes"] == ["İmza: Test Kullanıcı", "Kısa yaz"]
    assert c.get("/api/memory").json()["notes"] == ["İmza: Test Kullanıcı", "Kısa yaz"]


def test_memory_api_rejects_oversized_note(tmp_path):
    c = _client(tmp_path)
    r = c.put("/api/memory", json={"notes": ["x" * 500]})
    assert r.status_code == 400


def test_memory_flows_into_chat(tmp_path):
    c = _client(tmp_path)
    c.put("/api/memory", json={"notes": ["İmza: Genel Müdür", "Varsayılan ton: resmi"]})
    r = c.post("/api/chat", json={"messages": [
        {"role": "user", "content": "Ayşe'ye yanıt taslağı yaz: onaylıyorum"}
    ]})
    data = r.json()
    assert data["tools_used"] == ["create_draft"]
    assert "Genel Müdür" in data["reply"]     # signature honored end-to-end
    assert "Sayın Ayşe" in data["reply"]      # tone honored end-to-end
