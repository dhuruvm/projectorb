"""
Obscuro persistent memory — SQLite-backed episodic memory with TF-IDF retrieval.

Three stores:
  episodes — past Q&A interactions (with quality scores)
  lessons  — generalised lessons extracted from experience
  facts    — discrete factual claims

Retrieval uses TF-IDF cosine similarity — genuine semantic relevance without
requiring an embedding model or vector database. Significantly better than
the previous keyword-overlap approach: "fast" and "quick" now score similarly.
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

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

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","i","you","he","she","it",
    "we","they","what","which","who","whom","this","that","these",
    "those","and","but","or","nor","for","yet","so","in","on","at",
    "to","of","by","with","from","about","as","into","through",
    "before","after","above","below","up","down","out","off","over",
    "under","if","then","else","not","no","yes","very","just","also",
    "get","got","make","made","use","used","its","our","my","your",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, length ≥ 2, stopwords removed."""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute a TF-IDF vector for a token list given a pre-computed IDF table."""
    if not tokens:
        return {}
    tf    = Counter(tokens)
    total = len(tokens)
    return {
        w: (count / total) * idf.get(w, 1.0)
        for w, count in tf.items()
    }


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot    = sum(a[w] * b[w] for w in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Episode:
    id:       int
    question: str
    response: str
    critique: str
    lesson:   str
    score:    float
    ts:       float

    def summary(self) -> str:
        q = self.question[:70].rstrip()
        r = self.response[:70].rstrip()
        return f"Q: {q!r} → {r!r}"


class Memory:
    """
    Persistent memory for Obscuro with TF-IDF cosine-similarity retrieval.

    Thread-safe: each method opens its own SQLite connection so it is safe
    to use under Gradio's threaded request model.
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

    # ── IDF ──────────────────────────────────────────────────────────────────

    def _compute_idf(self, corpus: list[str]) -> dict[str, float]:
        """Compute smoothed IDF over a corpus of document strings."""
        N = len(corpus)
        if N == 0:
            return {}
        df: dict[str, int] = {}
        for doc in corpus:
            for w in set(_tokenize(doc)):
                df[w] = df.get(w, 0) + 1
        return {
            w: math.log((N + 1) / (count + 1)) + 1.0
            for w, count in df.items()
        }

    # ── Episodes ─────────────────────────────────────────────────────────────

    def add_episode(
        self,
        question: str,
        response: str,
        critique: str = "",
        lesson:   str = "",
        score:    float = 0.0,
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
        Return up to k episodes most relevant to `query` using TF-IDF cosine
        similarity over the 500 most-recent episodes.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM episodes ORDER BY ts DESC LIMIT 500"
            ).fetchall()

        if not rows:
            return []

        # Build corpus for IDF
        corpus = [r["question"] + " " + r["response"] for r in rows]
        idf    = self._compute_idf(corpus)

        # Query vector
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self.get_recent(k)
        q_vec = _tfidf_vector(q_tokens, idf)

        # Score each episode
        scored: list[tuple[float, Episode]] = []
        for row, doc in zip(rows, corpus):
            d_tokens = _tokenize(doc)
            d_vec    = _tfidf_vector(d_tokens, idf)
            s        = _cosine(q_vec, d_vec)
            if s > 0.0:
                scored.append((s, Episode(**dict(row))))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:k]]

    def get_recent(self, k: int = 3) -> list[Episode]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (k,)
            ).fetchall()
        return [Episode(**dict(row)) for row in rows]

    # ── Lessons ──────────────────────────────────────────────────────────────

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

    def get_facts(self, k: int = 10) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT fact FROM facts ORDER BY ts DESC LIMIT ?", (k,)
            ).fetchall()
        return [r["fact"] for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> dict[str, int]:
        with self._conn() as c:
            return {
                "episodes": c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "lessons":  c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
                "facts":    c.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
            }
