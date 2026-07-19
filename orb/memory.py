"""
Orb persistent memory — SQLite-backed episodic and semantic memory.

Three stores:
  episodes — past Q&A interactions with quality scores
  lessons  — generalised lessons extracted from experience
  facts    — discrete factual claims worth remembering

Retrieval uses keyword matching (BM25-like term frequency) — genuinely
functional without requiring a vector database or external embedding model.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from math import log

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "memory.db",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT    NOT NULL,
    response TEXT    NOT NULL,
    critique TEXT    DEFAULT '',
    lesson   TEXT    DEFAULT '',
    score    REAL    DEFAULT 0.0,
    ts       REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson  TEXT NOT NULL,
    domain  TEXT DEFAULT '',
    ts      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fact    TEXT NOT NULL,
    source  TEXT DEFAULT '',
    ts      REAL NOT NULL
);
"""

# Common stopwords excluded from keyword retrieval
_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","i","you","he","she","it",
    "we","they","what","which","who","whom","this","that","these",
    "those","and","but","or","nor","for","yet","so","in","on","at",
    "to","of","by","with","from","about","as","into","through","during",
    "before","after","above","below","up","down","out","off","over","under",
}


@dataclass
class Episode:
    id: int
    question: str
    response: str
    critique: str
    lesson: str
    score: float
    ts: float

    def summary(self) -> str:
        q = self.question[:70].rstrip()
        r = self.response[:70].rstrip()
        return f"Q: {q!r} → {r!r}"


def _keywords(text: str) -> list[str]:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _score_match(query_words: list[str], document: str) -> float:
    """Simple TF-based relevance score — exact word overlap."""
    doc_words = _keywords(document)
    if not doc_words:
        return 0.0
    doc_set = set(doc_words)
    hits = sum(1 for w in query_words if w in doc_set)
    # Normalise by query length to avoid penalising short queries
    return hits / max(len(set(query_words)), 1)


class Memory:
    """
    Persistent memory for Orb.

    All methods open their own connection so the object is safe to use
    across Gradio's threaded request model.
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # ── Episodes ──────────────────────────────────────────────────────────────

    def add_episode(
        self,
        question: str,
        response: str,
        critique: str = "",
        lesson: str = "",
        score: float = 0.0,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO episodes (question,response,critique,lesson,score,ts)"
                " VALUES (?,?,?,?,?,?)",
                (question, response, critique, lesson, score, time.time()),
            )
            return cur.lastrowid

    def retrieve_similar(self, query: str, k: int = 3) -> list[Episode]:
        """
        Return up to k episodes most relevant to `query` using keyword overlap.
        Scans the 300 most-recent episodes for efficiency.
        """
        q_words = _keywords(query)
        if not q_words:
            return self.get_recent(k)

        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM episodes ORDER BY ts DESC LIMIT 300"
            ).fetchall()

        scored: list[tuple[float, Episode]] = []
        for row in rows:
            doc = row["question"] + " " + row["response"]
            s = _score_match(q_words, doc)
            if s > 0:
                scored.append((s, Episode(**dict(row))))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:k]]

    def get_recent(self, k: int = 3) -> list[Episode]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (k,)
            ).fetchall()
        return [Episode(**dict(row)) for row in rows]

    # ── Lessons ───────────────────────────────────────────────────────────────

    def add_lesson(self, lesson: str, domain: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO lessons (lesson,domain,ts) VALUES (?,?,?)",
                (lesson, domain, time.time()),
            )

    def get_lessons(self, k: int = 5) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT lesson FROM lessons ORDER BY ts DESC LIMIT ?", (k,)
            ).fetchall()
        return [r["lesson"] for r in rows]

    # ── Facts ─────────────────────────────────────────────────────────────────

    def add_fact(self, fact: str, source: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO facts (fact,source,ts) VALUES (?,?,?)",
                (fact, source, time.time()),
            )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> dict[str, int]:
        with self._conn() as c:
            return {
                "episodes": c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "lessons":  c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
                "facts":    c.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
            }
