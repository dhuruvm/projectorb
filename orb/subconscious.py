"""
Subconscious Processor — background cognitive processes beneath explicit reasoning.

What this models
────────────────
  Priming            before the ReAct loop starts, "warm-up" the context with
                     relevant memories and successful past tool sequences.
                     Like hippocampal broadcasting in biological cognition.

  Pattern registry   remember which tool sequences worked for which query types,
                     so the agent does NOT have to rediscover them each time.

  Episodic           merge very similar short-term memories into richer,
  consolidation      longer-term ones to prevent the memory store from growing
                     unboundedly with near-duplicate entries.

  Activation         rank memories by recency × relevance (not just relevance),
  broadcasting       mimicking how the mammalian brain gives recency a boost.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import Memory


# ── DB path ───────────────────────────────────────────────────────────────────

_SC_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "subconscious.db",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_patterns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_type TEXT    NOT NULL,
    tool_seq   TEXT    NOT NULL,
    success    INTEGER DEFAULT 1,
    score      REAL    DEFAULT 1.0,
    used_at    REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS consolidated (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT    NOT NULL,
    summary    TEXT    NOT NULL,
    source_ids TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qt ON tool_patterns(query_type);
"""


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PrimingContext:
    """
    The subconscious "warm start" injected before explicit reasoning.
    Contains relevant memories + the best-known tool sequence.
    """
    relevant_memories:       list[str]  = field(default_factory=list)
    suggested_tool_sequence: list[str]  = field(default_factory=list)
    activation_score:        float      = 0.0   # 0–1, how primed this query is
    register_hint:           str        = "analytical"

    def to_prompt_fragment(self) -> str:
        """Render as a short system-level hint to prepend to the task."""
        parts: list[str] = []
        if self.relevant_memories:
            parts.append(
                "Background context from prior experience:\n"
                + "\n".join(f"  • {m}" for m in self.relevant_memories[:3])
            )
        if self.suggested_tool_sequence:
            parts.append(
                "Suggested approach (from past successes): "
                + " → ".join(self.suggested_tool_sequence)
            )
        return "\n\n".join(parts)

    def is_active(self) -> bool:
        return self.activation_score > 0.15 or bool(self.relevant_memories)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _recency_weight(ts: float, now: float, half_life_hours: float = 48.0) -> float:
    """Exponential decay: 1.0 at time-of-creation, halves every half_life_hours."""
    age_h = (now - ts) / 3600.0
    return math.exp(-math.log(2) * age_h / half_life_hours)


# ── Processor ────────────────────────────────────────────────────────────────

class SubconsciousProcessor:
    """
    Background cognitive substrate.
    All operations are synchronous and cheap (~1–5 ms).
    """

    def __init__(self, db_path: str = _SC_DB) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ── Priming ───────────────────────────────────────────────────────────────

    def prime(
        self,
        query: str,
        memory: "Memory",
        register: str = "analytical",
        k: int = 3,
    ) -> PrimingContext:
        """
        Retrieve a priming context before explicit reasoning begins.
        Blends TF-IDF similarity with recency decay so recent episodes
        get a mild bonus over older equally-relevant ones.
        """
        # Retrieve candidate episodes
        episodes = memory.retrieve_similar(query, k=k * 2)

        # Re-rank with recency bonus
        now = time.time()
        scored = []
        for ep in episodes:
            sim     = _jaccard(query, ep.question)   # fast proxy; TF-IDF was already used to filter
            recency = _recency_weight(ep.ts, now)
            combined = 0.75 * sim + 0.25 * recency
            scored.append((combined, ep))
        scored.sort(key=lambda x: -x[0])
        top = [ep for _, ep in scored[:k]]

        summaries: list[str] = []
        for ep in top:
            q_preview = ep.question[:75].strip()
            a_preview = (ep.response[:75] if ep.response else "").strip()
            summaries.append(f"Q: {q_preview!r}  →  {a_preview!r}" if a_preview
                             else f"Prior: {q_preview!r}")

        tool_seq   = self.suggest_pattern(register)
        activation = min(1.0, len(top) / max(k, 1)) if top else 0.0
        if scored:
            activation = max(activation, round(scored[0][0], 3))

        return PrimingContext(
            relevant_memories=summaries,
            suggested_tool_sequence=tool_seq,
            activation_score=round(activation, 3),
            register_hint=register,
        )

    # ── Tool pattern learning ─────────────────────────────────────────────────

    def record_pattern(
        self,
        query_type: str,
        tool_sequence: list[str],
        score: float = 1.0,
        success: bool = True,
    ) -> None:
        """Record a tool sequence that worked (or failed) for a query type."""
        if not tool_sequence:
            return
        self._db.execute(
            "INSERT INTO tool_patterns (query_type, tool_seq, success, score, used_at)"
            " VALUES (?,?,?,?,?)",
            (query_type, json.dumps(tool_sequence), int(success), score, time.time()),
        )
        self._db.commit()

    def suggest_pattern(self, query_type: str) -> list[str]:
        """Return the highest-scoring tool sequence for this query type."""
        row = self._db.execute(
            """SELECT tool_seq FROM tool_patterns
               WHERE query_type=? AND success=1
               ORDER BY score DESC, used_at DESC LIMIT 1""",
            (query_type,),
        ).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    # ── Episodic consolidation ────────────────────────────────────────────────

    def consolidate(
        self,
        memory: "Memory",
        similarity_threshold: float = 0.60,
        min_episodes: int = 8,
    ) -> int:
        """
        Greedy consolidation: find pairs of very similar recent episodes and
        record a merged summary entry in the subconscious consolidated table.
        Does NOT delete originals — just indexes them.

        Returns number of clusters created.
        """
        stats = memory.count()
        if stats.get("episodes", 0) < min_episodes:
            return 0

        recent = memory.get_recent(k=30)
        if len(recent) < 4:
            return 0

        seen:    set[int]  = set()
        merged:  int       = 0

        for i, a in enumerate(recent):
            if i in seen:
                continue
            group     = [a]
            group_ids = [a.id]

            for j, b in enumerate(recent[i + 1:], start=i + 1):
                if j in seen:
                    continue
                if _jaccard(a.question, b.question) >= similarity_threshold:
                    group.append(b)
                    group_ids.append(b.id)
                    seen.add(j)

            if len(group) >= 2:
                topic   = group[0].question[:60].strip()
                summary = " | ".join(
                    g.response[:80].strip() for g in group if g.response
                )[:450]
                if summary:
                    self._db.execute(
                        "INSERT INTO consolidated (topic, summary, source_ids, created_at)"
                        " VALUES (?,?,?,?)",
                        (topic, summary, json.dumps(group_ids), time.time()),
                    )
                    merged += 1
            seen.add(i)

        if merged:
            self._db.commit()
        return merged

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        patterns     = self._db.execute("SELECT COUNT(*) FROM tool_patterns").fetchone()[0]
        consolidated = self._db.execute("SELECT COUNT(*) FROM consolidated").fetchone()[0]
        return {"patterns_recorded": patterns, "consolidated_clusters": consolidated}
