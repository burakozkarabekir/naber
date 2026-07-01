"""User memory store — durable preferences the assistant honors in every reply.

PRIVACY notes (audit here):
  * Memory holds ONLY user-authored notes (signature, tone preferences, standing
    instructions). It must never be used to persist email bodies or any mailbox
    content — that stays in memory-only processing per the data-at-rest rule.
  * Stored as a small local JSON file (gitignored). Not a secret store: never
    put credentials here (tokens live in the Keychain).
  * Note *content* is never logged; logging uses counts only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger("hermes.memory")

MAX_NOTES = 30
MAX_NOTE_LEN = 300


class MemoryStore:
    """File-backed list of short user notes.

    ``defaults`` are returned while no file exists (used by demo mode to show
    example notes without writing to disk); the file is created only when the
    user actually saves.
    """

    def __init__(self, path: str | Path, defaults: list[str] | None = None) -> None:
        self._path = Path(path)
        self._defaults = [n.strip() for n in (defaults or []) if n and n.strip()]
        self._lock = Lock()

    def get_notes(self) -> list[str]:
        with self._lock:
            return self._read()

    def set_notes(self, notes: list[str]) -> list[str]:
        clean = _clean(notes)
        with self._lock:
            self._write(clean)
        logger.info("memory_saved count=%d", len(clean))
        return clean

    def add_note(self, note: str) -> list[str]:
        with self._lock:
            notes = self._read()
            cleaned = note.strip()[:MAX_NOTE_LEN]
            if cleaned and cleaned not in notes:
                notes.append(cleaned)
            clean = _clean(notes)
            self._write(clean)
        logger.info("memory_saved count=%d", len(clean))
        return clean

    def as_prompt(self) -> str:
        """Render notes as a bullet list for system-prompt injection ('' if none)."""
        notes = self.get_notes()
        return "\n".join(f"- {n}" for n in notes)

    # --- file I/O (callers hold the lock) ---------------------------------

    def _read(self) -> list[str]:
        if not self._path.exists():
            return list(self._defaults)
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("memory_read_failed %s", type(exc).__name__)
            return list(self._defaults)
        notes = data.get("notes", []) if isinstance(data, dict) else []
        return [str(n) for n in notes if str(n).strip()]

    def _write(self, notes: list[str]) -> None:
        self._path.write_text(
            json.dumps({"notes": notes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _clean(notes: list[str]) -> list[str]:
    out: list[str] = []
    for n in notes:
        n = (n or "").strip()[:MAX_NOTE_LEN]
        if n and n not in out:
            out.append(n)
    return out[:MAX_NOTES]
