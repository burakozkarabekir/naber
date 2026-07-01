"""Scripted LLM provider for DEMO_MODE.

Implements the ``LLMProvider`` interface with simple, deterministic intent
detection so the full agent pipeline (orchestrator → tools → FakeGmailClient)
runs end-to-end with NO external LLM. It recognizes the core demo intents
(triage, summarize, search/"did X reply", draft) in Turkish and English, drives
the real tools, and composes data-driven answers.

This is for UI/UX testing only — it is not a general model.
"""

from __future__ import annotations

import re
from typing import Any

from app.demo.fake_gmail import ME, SEED
from app.llm.base import LLMProvider, LLMResponse, ToolCall
from app.memory import MemoryStore

CATEGORY_LABELS = {
    "needs_reply": "🔴 Yanıt bekliyor",
    "awaiting_others": "🟡 Başkalarından bekleniyor",
    "fyi": "🔵 Bilgi amaçlı",
    "newsletter": "⚪ Bülten & tanıtım",
}
_CATEGORY_ORDER = ["needs_reply", "awaiting_others", "fyi", "newsletter"]
_ID_CATEGORY = {d["id"]: d["category"] for d in SEED}
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_TR_MAP = str.maketrans("şıİöüçğ", "siioucg")


def _fold(s: str) -> str:
    return s.translate(_TR_MAP).lower()


def _short_sender(sender: str) -> str:
    return sender.split("<")[0].strip() or sender


def _first_name(sender: str) -> str:
    return _short_sender(sender).split()[0]


def _email_of(sender: str) -> str:
    m = _EMAIL_RE.search(sender)
    return m.group(0) if m else ""


# name key -> (display, email, thread_id, subject, message_id)
def _build_people() -> dict[str, tuple[str, str, str, str, str]]:
    people: dict[str, tuple[str, str, str, str, str]] = {}
    for d in SEED:
        first = _first_name(d["sender"])
        info = (
            _short_sender(d["sender"]),
            _email_of(d["sender"]),
            d["thread_id"],
            d["subject"],
            d["id"],
        )
        people.setdefault(first.lower(), info)
        people.setdefault(_fold(first), info)
    return people


_PEOPLE = _build_people()


class DemoLLMProvider(LLMProvider):
    name = "demo"
    uses_native_tools = True  # tool results come back as structured blocks

    def __init__(self, memory: MemoryStore | None = None) -> None:
        # Optional user-memory store: signature, default tone, standing notes.
        self._memory = memory

    # --- memory helpers ----------------------------------------------------

    def _notes(self) -> list[str]:
        return self._memory.get_notes() if self._memory else []

    def _signature(self) -> str:
        for n in self._notes():
            if _fold(n).startswith("imza"):
                _, _, value = n.partition(":")
                if value.strip():
                    return value.strip()
        return ME

    def _default_tone(self) -> str:
        for n in self._notes():
            f = _fold(n)
            if "ton" in f:
                if "resmi" in f or "formal" in f:
                    return "formal"
                if "samimi" in f or "friendly" in f or "sicak" in f:
                    return "friendly"
        return "neutral"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        pending = self._pending_tool_result(messages)
        user_text = self._last_user_text(messages)

        if pending is not None:
            name, result = pending
            return LLMResponse(text=self._compose(name, result, user_text))

        intent = self._intent(user_text)
        if intent == "memory_add":
            return LLMResponse(text=self._memory_add(user_text))
        if intent == "memory_show":
            return LLMResponse(text=self._memory_show())
        if intent == "triage":
            return LLMResponse(
                tool_calls=[ToolCall("list_recent_emails", {"max_results": 10}, "d1")]
            )
        if intent == "summarize":
            return LLMResponse(
                tool_calls=[
                    ToolCall("get_thread", {"thread_id": self._thread_for(user_text)}, "d1")
                ]
            )
        if intent == "search":
            return LLMResponse(
                tool_calls=[
                    ToolCall("search_emails", {"query": self._search_query(user_text)}, "d1")
                ]
            )
        if intent == "draft":
            args = self._draft_args(user_text)
            if not args.get("to"):
                return LLMResponse(
                    text=(
                        "Kime yazmamı istersin? Bir e-posta adresi ver (ör. "
                        "`bob@northwind.io`) ya da bir kişi belirt (ör. *Ayşe'ye yanıt yaz*)."
                    )
                )
            return LLMResponse(tool_calls=[ToolCall("create_draft", args, "d1")])
        return LLMResponse(text=self._help())

    def ping(self) -> bool:
        return True

    # --- message plumbing ------------------------------------------------

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return ""

    @staticmethod
    def _pending_tool_result(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, Any] | None:
        if not messages:
            return None
        last = messages[-1]
        if last.get("role") != "user" or not isinstance(last.get("content"), list):
            return None
        tr = next(
            (b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"),
            None,
        )
        if tr is None:
            return None
        name = None
        for msg in reversed(messages[:-1]):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                tu = next(
                    (b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "tool_use"),
                    None,
                )
                if tu:
                    name = tu.get("name")
                    break
        import json

        try:
            result = json.loads(tr.get("content") or "null")
        except Exception:  # noqa: BLE001
            result = None
        return (name, result)

    # --- intent ----------------------------------------------------------

    def _intent(self, text: str) -> str:
        t = _fold(text)
        if any(k in t for k in ["hafizana ekle", "hafizaya ekle", "aklinda tut",
                                "unutma:", "not al:", "remember:"]):
            return "memory_add"
        if any(k in t for k in ["hafizanda ne", "hafizani goster", "neleri hatirliyorsun",
                                "notlarin neler", "hafizan", "memory"]):
            return "memory_show"
        draft_kw = ["taslak", "draft", "reply", "yanitla", "cevap yaz", "yanit yaz",
                    "e-posta yaz", "eposta yaz", "mail yaz", "email yaz"]
        if any(k in t for k in draft_kw):
            return "draft"
        if "yaz" in t and ("@" in text or "mail" in t or "posta" in t
                           or any(k in t for k in _PEOPLE)):
            return "draft"
        if any(k in t for k in ["ozet", "summar", "konusul", "yazismasi", "yazisma"]):
            return "summarize"
        if any(k in t for k in ["dondu mu", "yanit verdi mi", "cevap verdi mi",
                                "reply yet", "replied", "geldi mi", "buldun mu",
                                "yaniti geldi"]):
            return "search"
        if any(k in t for k in ["onemli", "neler var", "gelen kutu", "triage",
                                "attention", "bugun ne", "ne var", "durum", "inbox",
                                "onceliklendir"]):
            return "triage"
        return "help"

    def _thread_for(self, text: str) -> str:
        t = _fold(text)
        if any(k in t for k in ["butce", "budget", "q3", "ayse"]):
            return "t-budget"
        if any(k in t for k in ["teklif", "proposal", "david", "northwind"]):
            return "t-proposal"
        return "t-budget"

    def _search_query(self, text: str) -> str:
        t = _fold(text)
        if "david" in t or "proposal" in t or "teklif" in t:
            return "proposal"
        if "ayse" in t or "butce" in t or "budget" in t:
            return "bütçe"
        m = _EMAIL_RE.search(text)
        if m:
            return m.group(0)
        words = [w for w in re.split(r"\W+", text) if len(w) > 3]
        return " ".join(words[:3])

    def _draft_args(self, text: str) -> dict[str, Any]:
        t = _fold(text)
        # Memory sets the default tone; an explicit request overrides it.
        tone = self._default_tone()
        if any(k in t for k in ["resmi", "formal"]):
            tone = "formal"
        elif any(k in t for k in ["samimi", "friendly", "sicak"]):
            tone = "friendly"
        short = any(k in t for k in ["kisa", "short"])
        gist = text.split(":", 1)[1].strip().strip('"“”') if ":" in text else ""
        if not gist:
            gist = self._default_gist(t)

        email = _EMAIL_RE.search(text)
        if email:
            to = email.group(0)
            greet = to.split("@")[0].split(".")[0].capitalize()
            subject = self._subject_for(t, gist)
            return {
                "to": to,
                "subject": subject,
                "body": self._body(greet, gist, tone, short),
            }

        # Reply to a known person mentioned by name.
        for key, (display, addr, _tid, subject, msg_id) in _PEOPLE.items():
            if key in t:
                greet = _first_name(display)
                return {
                    "to": addr,
                    "subject": f"Re: {subject}",
                    "body": self._body(greet, gist, tone, short),
                    "in_reply_to_message_id": msg_id,
                }
        return {"to": "", "subject": "", "body": ""}

    @staticmethod
    def _default_gist(t: str) -> str:
        if "ertele" in t or "reschedul" in t or "toplant" in t:
            return ("Planladığımız toplantıyı ertelememiz gerekiyor. Sizin için uygun "
                    "olabilecek yeni bir zaman önerebilir misiniz?")
        if "tesekkur" in t or "thank" in t:
            return "Desteğiniz için içtenlikle teşekkür ederim."
        if "teklif" in t or "proposal" in t:
            return "Teklifle ilgili detayları en kısa sürede sizinle paylaşacağım."
        return ""

    @staticmethod
    def _subject_for(t: str, gist: str) -> str:
        if "ertele" in t or "reschedul" in t or "toplant" in t:
            return "Toplantı ertelemesi"
        if "tesekkur" in t or "thank" in t:
            return "Teşekkürler"
        if "teklif" in t or "proposal" in t:
            return "Teklif hakkında"
        return "Bilgilendirme"

    def _body(self, greet: str, gist: str, tone: str, short: bool) -> str:
        sig = self._signature()
        line = gist if gist else "Konuyla ilgili en kısa sürede size dönüş yapacağım."
        if not line.endswith((".", "!", "?")):
            line += "."
        line = line[0].upper() + line[1:] if line else line
        if tone == "formal":
            opener, closer = f"Sayın {greet},", f"Saygılarımla,\n{sig}"
        elif tone == "friendly":
            opener, closer = f"Selam {greet},", f"Sevgiler,\n{sig}"
        else:
            opener, closer = f"Merhaba {greet},", f"İyi çalışmalar,\n{sig}"
        if short:
            return f"{opener}\n\n{line}\n\n{closer}"
        return f"{opener}\n\n{line}\n\nHerhangi bir sorunuz olursa memnuniyetle yardımcı olurum.\n\n{closer}"

    # --- memory intents ----------------------------------------------------

    def _memory_add(self, text: str) -> str:
        if self._memory is None:
            return "Hafıza bu modda kullanılamıyor."
        note = text.split(":", 1)[1].strip() if ":" in text else ""
        if not note:
            # Strip the command phrase, keep the rest as the note.
            t = text
            for kw in ["hafızana ekle", "hafızaya ekle", "aklında tut", "unutma"]:
                idx = _fold(t).find(_fold(kw))
                if idx != -1:
                    t = (t[:idx] + t[idx + len(kw):]).strip(" ,.:;")
                    break
            note = t.strip()
        if not note:
            return "Neyi hatırlamamı istersin? Örn: *hafızana ekle: imza: Burak Özkarabekir*"
        self._memory.add_note(note)
        return (
            f"🧠 Hafızama ekledim: “{note}”\n\n"
            "Bundan sonraki yanıt ve taslaklarımda bunu dikkate alacağım. "
            "Notları soldaki **Hafıza** sekmesinden görüntüleyip düzenleyebilirsin."
        )

    def _memory_show(self) -> str:
        notes = self._notes()
        if not notes:
            return (
                "Hafızam şu an boş. Soldaki **Hafıza** sekmesinden not ekleyebilir "
                "ya da bana *hafızana ekle: …* diyebilirsin."
            )
        lines = ["🧠 Hafızamda şunlar var:", ""]
        lines += [f"  • {n}" for n in notes]
        lines += ["", "Bunları soldaki **Hafıza** sekmesinden düzenleyebilirsin."]
        return "\n".join(lines)

    # --- compose answers -------------------------------------------------

    def _compose(self, name: str | None, result: Any, user_text: str) -> str:
        if name == "list_recent_emails":
            return self._compose_triage(result)
        if name == "search_emails":
            return self._compose_search(result, user_text)
        if name == "get_thread":
            return self._compose_summary(result)
        if name == "create_draft":
            return self._compose_draft(result, user_text)
        return self._help()

    def _compose_triage(self, emails: list[dict[str, Any]]) -> str:
        buckets: dict[str, list[dict[str, Any]]] = {c: [] for c in _CATEGORY_ORDER}
        for e in emails or []:
            cat = _ID_CATEGORY.get(e.get("id"), "fyi")
            buckets.setdefault(cat, []).append(e)
        lines = [f"Gelen kutunda **{len(emails or [])} e-posta** var. Önceliğe göre:"]
        for cat in _CATEGORY_ORDER:
            items = buckets.get(cat) or []
            if not items:
                continue
            lines.append("")
            lines.append(f"{CATEGORY_LABELS[cat]}")
            for e in items:
                lines.append(f"  • {_short_sender(e['sender'])} — {e['subject']}")
        lines.append("")
        lines.append("İstersen bunlardan birine yanıt taslağı hazırlayayım. ✍️")
        return "\n".join(lines)

    def _compose_search(self, emails: list[dict[str, Any]], user_text: str) -> str:
        if not emails:
            return "Bu kritere uyan bir e-posta bulamadım."
        t = _fold(user_text)
        asked_reply = any(k in t for k in ["dondu mu", "verdi mi", "reply", "geldi mi"])
        first = emails[0]
        if asked_reply:
            return (
                f"Evet ✅ **{_short_sender(first['sender'])}** {first['date']} tarihinde döndü:\n\n"
                f"“{first['snippet']}”\n\nİstersen bir yanıt taslağı hazırlayayım."
            )
        lines = [f"**{len(emails)} e-posta** buldum:"]
        for e in emails:
            flag = "🔵 okunmadı" if e.get("unread") else ""
            lines.append(f"  • {_short_sender(e['sender'])} — {e['subject']} {flag}".rstrip())
        return "\n".join(lines)

    def _compose_summary(self, thread: dict[str, Any]) -> str:
        msgs = (thread or {}).get("messages", [])
        if not msgs:
            return "Bu konuda özetlenecek bir yazışma bulamadım."
        subject = msgs[0].get("subject", "Yazışma")
        lines = [f"**{subject}** — özet ({len(msgs)} mesaj):", ""]
        for m in msgs:
            lines.append(f"  • {_short_sender(m['sender'])} ({m['date']}): {_trim(m['body'])}")
        tid = thread.get("thread_id")
        if tid == "t-budget":
            lines += [
                "",
                "📌 **Açık aksiyon:** Ayşe, pazarlamadaki %15 artış için **Cuma'ya kadar senin onayını** bekliyor (gerekçe: Eylül ürün lansmanı).",
            ]
        elif tid == "t-proposal":
            lines += [
                "",
                "📌 **Açık aksiyon:** David iki soru sordu (12 ay sabit fiyat? Eylül başı teslim?) — yanıt bekliyor.",
            ]
        return "\n".join(lines)

    def _compose_draft(self, result: dict[str, Any], user_text: str) -> str:
        args = self._draft_args(user_text)
        return (
            "✅ **Taslak oluşturuldu — gönderilmedi.**\n\n"
            f"**Kime:** {args.get('to', '')}\n"
            f"**Konu:** {args.get('subject', '')}\n\n"
            "— — —\n"
            f"{args.get('body', '')}\n"
            "— — —\n\n"
            "📌 Gmail → **Taslaklar**'dan inceleyip kendin gönderebilirsin. "
            "Hermes hiçbir zaman otomatik göndermez."
        )

    @staticmethod
    def _help() -> str:
        return (
            "Merhaba! Ben **Hermes**, e-posta asistanınım. Şunları deneyebilirsin:\n\n"
            "  • *Bugün neler önemli?* — gelen kutusu triyajı\n"
            "  • *Q3 bütçe yazışmasını özetle* — konu özeti\n"
            "  • *David teklife döndü mü?* — yanıt kontrolü\n"
            "  • *Ayşe'ye kısa, resmi bir yanıt taslağı yaz* — taslak (yalnızca taslak, asla göndermem)\n"
            "  • *hafızana ekle: imza: Burak Özkarabekir* — kalıcı tercih kaydet\n\n"
            "_Demo modu: örnek verilerle çalışıyorum; gerçek Gmail'e bağlı değilim._"
        )


_GREETING_RE = re.compile(r"^(merhaba|selam|sayın|hi|hello|dear)\b", re.IGNORECASE)


def _trim(text: str, limit: int = 130) -> str:
    """First substantive line of a message body (skips greeting lines)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    line = next((ln for ln in lines if not _GREETING_RE.match(ln)), lines[0] if lines else "")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
