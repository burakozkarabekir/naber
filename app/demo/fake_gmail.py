"""In-memory sample mailbox for DEMO_MODE.

FakeGmailClient mirrors the real ``GmailClient`` interface (same method
signatures, same return dataclasses) so the agent, tools, and API run unchanged
against sample data — no Google account, no OAuth, no network.

SECURITY: this is for UI/UX testing only. It never touches a real mailbox and,
like the real client, only ever *creates* drafts — it cannot send.
"""

from __future__ import annotations

import re
from typing import Any

from app.gmail.client import DraftResult, EmailSummary, ThreadMessage, ThreadView

ME = "Burak"

# Each sample email carries a triage ``category`` used by the demo LLM to group
# the inbox. Categories: needs_reply | awaiting_others | fyi | newsletter.
SEED: list[dict[str, Any]] = [
    {
        "id": "e1",
        "thread_id": "t-budget",
        "sender": "Ayşe Yılmaz <ayse.yilmaz@acmeholding.com>",
        "subject": "Q3 bütçe revizyonu — onayınız?",
        "snippet": "Revize bütçeyi ekte gönderdim. Pazarlama +%15 arttı; Cuma'ya kadar onayınıza ihtiyacım var.",
        "date": "Bugün 09:14",
        "unread": True,
        "category": "needs_reply",
    },
    {
        "id": "e2",
        "thread_id": "t-proposal",
        "sender": "David Chen <david@northwind.io>",
        "subject": "Re: Proposal — a couple of questions",
        "snippet": "Thanks for the proposal. Two quick questions on pricing and timeline before we sign.",
        "date": "Bugün 08:02",
        "unread": True,
        "category": "needs_reply",
    },
    {
        "id": "e3",
        "thread_id": "t-contract",
        "sender": "Mehmet Demir <mehmet.demir@acmeholding.com>",
        "subject": "Sözleşme taslağı — hukuktan dönüş bekleniyor",
        "snippet": "Taslağı hukuka ilettim. Dönüş gelir gelmez sizinle paylaşacağım.",
        "date": "Dün 17:40",
        "unread": False,
        "category": "awaiting_others",
    },
    {
        "id": "e4",
        "thread_id": "t-po",
        "sender": "Satınalma <satinalma@tedarikci.com>",
        "subject": "PO #4471 onay sürecinde",
        "snippet": "Talebiniz onay akışında. Bir güncelleme olduğunda bilgilendireceğiz.",
        "date": "Dün 15:22",
        "unread": False,
        "category": "awaiting_others",
    },
    {
        "id": "e5",
        "thread_id": "t-it",
        "sender": "BT Bildirim <noreply-it@acmeholding.com>",
        "subject": "Planlı bakım: Cumartesi 02:00–04:00",
        "snippet": "Bu süre boyunca bazı sistemler kısa süreli erişilemez olabilir.",
        "date": "Dün 11:00",
        "unread": False,
        "category": "fyi",
    },
    {
        "id": "e6",
        "thread_id": "t-cal",
        "sender": "Google Takvim <calendar-notification@google.com>",
        "subject": "Yarın: Ürün sync 15:00",
        "snippet": "Etkinlik hatırlatması: Ürün sync toplantısı.",
        "date": "Dün 09:30",
        "unread": False,
        "category": "fyi",
    },
    {
        "id": "e7",
        "thread_id": "t-news",
        "sender": "TechWeekly <newsletter@techweekly.com>",
        "subject": "Bu hafta: 10 yapay zeka haberi",
        "snippet": "Haftanın öne çıkan gelişmeleri ve okumalar.",
        "date": "Dün 07:15",
        "unread": False,
        "category": "newsletter",
    },
    {
        "id": "e8",
        "thread_id": "t-promo",
        "sender": "ShopMax Kampanya <promo@shopmax.com>",
        "subject": "🎉 Sadece bugün: %40 indirim",
        "snippet": "Seçili ürünlerde bugüne özel fırsat. Kaçırmayın!",
        "date": "Dün 06:00",
        "unread": False,
        "category": "newsletter",
    },
]

# Full threads for summarization / Q&A.
THREADS: dict[str, list[dict[str, str]]] = {
    "t-budget": [
        {
            "id": "e1",
            "sender": "Ayşe Yılmaz <ayse.yilmaz@acmeholding.com>",
            "to": "burak@acmeholding.com",
            "date": "Bugün 09:14",
            "subject": "Q3 bütçe revizyonu — onayınız?",
            "body": (
                "Merhaba Burak,\n\nQ3 için revize bütçeyi hazırladım. En önemli değişiklik "
                "pazarlama kaleminde: yeni kampanya lansmanı nedeniyle %15 artış var. "
                "Toplam bütçe yine de onaylı üst sınırın altında.\n\nCuma gününe kadar "
                "onayını rica ederim; sonrasında finansa iletmem gerekiyor.\n\nTeşekkürler,\nAyşe"
            ),
        },
        {
            "id": "e1b",
            "sender": "Burak <burak@acmeholding.com>",
            "to": "ayse.yilmaz@acmeholding.com",
            "date": "Bugün 09:40",
            "subject": "Re: Q3 bütçe revizyonu — onayınız?",
            "body": "Ayşe, pazarlamadaki %15 artışın gerekçesini biraz açar mısın?",
        },
        {
            "id": "e1c",
            "sender": "Ayşe Yılmaz <ayse.yilmaz@acmeholding.com>",
            "to": "burak@acmeholding.com",
            "date": "Bugün 10:05",
            "subject": "Re: Q3 bütçe revizyonu — onayınız?",
            "body": (
                "Artışın tamamı Eylül'deki ürün lansmanının dijital reklam ve etkinlik "
                "giderlerinden kaynaklanıyor. Detaylı kırılımı ekteki tabloda bulabilirsin. "
                "Onayınla birlikte hemen ilerleyebilirim."
            ),
        },
    ],
    "t-proposal": [
        {
            "id": "e2",
            "sender": "David Chen <david@northwind.io>",
            "to": "burak@acmeholding.com",
            "date": "Bugün 08:02",
            "subject": "Re: Proposal — a couple of questions",
            "body": (
                "Hi Burak,\n\nThanks for the proposal — looks strong. Two questions before we sign:\n"
                "1) Is the pricing fixed for 12 months?\n2) Can delivery start in early September?\n\n"
                "Best,\nDavid"
            ),
        }
    ],
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _summary(d: dict[str, Any]) -> EmailSummary:
    return EmailSummary(
        id=d["id"],
        thread_id=d["thread_id"],
        sender=d["sender"],
        subject=d["subject"],
        snippet=d["snippet"],
        date=d["date"],
        unread=d["unread"],
    )


class FakeGmailClient:
    """Duck-typed stand-in for GmailClient backed by SEED / THREADS."""

    def __init__(self) -> None:
        self._draft_seq = 0
        # Exposed so a UI could show created drafts; not used by the agent.
        self.created_drafts: list[dict[str, Any]] = []

    def list_recent_emails(
        self, max_results: int = 10, query: str | None = None
    ) -> list[EmailSummary]:
        rows = SEED
        if query:
            rows = self._filter(query)
        return [_summary(d) for d in rows[: max(1, min(int(max_results or 10), 50))]]

    def search_emails(self, query: str) -> list[EmailSummary]:
        return [_summary(d) for d in self._filter(query)]

    def get_thread(self, thread_id: str) -> ThreadView:
        msgs = THREADS.get(thread_id, [])
        return ThreadView(
            thread_id=thread_id,
            messages=[
                ThreadMessage(
                    id=m["id"],
                    sender=m["sender"],
                    to=m["to"],
                    date=m["date"],
                    subject=m["subject"],
                    body=m["body"],
                )
                for m in msgs
            ],
        )

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        in_reply_to_message_id: str | None = None,
    ) -> DraftResult:
        # DRAFT ONLY — nothing is ever sent, exactly like the real client.
        self._draft_seq += 1
        draft_id = f"demo-draft-{self._draft_seq}"
        thread_id = ""
        for d in SEED:
            if d["id"] == in_reply_to_message_id:
                thread_id = d["thread_id"]
                break
        self.created_drafts.append(
            {"draft_id": draft_id, "to": to, "subject": subject, "body": body}
        )
        return DraftResult(
            draft_id=draft_id,
            thread_id=thread_id,
            is_reply=bool(in_reply_to_message_id),
        )

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _filter(query: str) -> list[dict[str, Any]]:
        q = query.lower()
        unread_only = "is:unread" in q
        # Strip Gmail operators, keep plain words.
        words = [
            w
            for w in re.split(r"\s+", re.sub(r"\b\w+:\S+", " ", q)).__iter__()
            if w and len(w) > 2
        ]
        out = []
        for d in SEED:
            if unread_only and not d["unread"]:
                continue
            hay = f"{d['sender']} {d['subject']} {d['snippet']}".lower()
            if not words or any(w in hay for w in words):
                out.append(d)
        return out
