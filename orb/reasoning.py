"""
Orb multi-path reasoning engine.

Generates N candidate responses at distinct temperatures, scores each with
a combination of model perplexity and fast heuristics, then returns the
best candidate along with the full ranked list for inspection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import OrbModel


# ── Heuristic scoring ─────────────────────────────────────────────────────────

def _heuristic_score(question: str, response: str) -> float:
    """
    Fast quality estimate without a model call.

    Three components (weighted):
      length_score — reward responses in the 80–400 char sweet spot
      uniqueness   — penalise 4-gram repetition
      relevance    — fraction of question keywords present in response
    """
    if not response:
        return 0.0

    n = len(response)
    if n < 20:
        length_score = 0.1
    elif n < 80:
        length_score = 0.5
    elif n <= 400:
        length_score = 1.0
    elif n <= 800:
        length_score = 0.75
    else:
        length_score = 0.5

    words = response.lower().split()
    if len(words) >= 4:
        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        uniqueness = len(set(ngrams)) / len(ngrams)
    else:
        uniqueness = 1.0

    q_words = set(re.findall(r'\w+', question.lower()))
    r_words = set(re.findall(r'\w+', response.lower()))
    relevance = len(q_words & r_words) / max(len(q_words), 1)

    return 0.4 * length_score + 0.35 * uniqueness + 0.25 * relevance


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RankedResponse:
    text: str
    heuristic_score: float
    model_score: float
    temperature: float

    @property
    def combined_score(self) -> float:
        return 0.6 * self.heuristic_score + 0.4 * self.model_score

    def __repr__(self) -> str:
        return (
            f"RankedResponse(T={self.temperature}, "
            f"h={self.heuristic_score:.3f}, m={self.model_score:.3f}, "
            f"combined={self.combined_score:.3f})"
        )


# ── Reasoner ─────────────────────────────────────────────────────────────────

class MultiPathReasoner:
    """
    Explores 3 candidate responses at different temperatures and returns the
    highest-scoring one.

    Temperature spread:
      0.60 — conservative, low variance, safer phrasing
      0.85 — balanced
      1.10 — creative, higher variance
    """

    TEMPERATURES = [0.60, 0.85, 1.10]

    def __init__(self, model: "OrbModel") -> None:
        self._model = model

    def run(
        self,
        message: str,
        history: list[dict],
        *,
        max_new_tokens: int = 200,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        four_stream: bool = False,
        seed: int = 42,
    ) -> tuple[str, list[RankedResponse]]:
        """
        Returns (best_response_text, all_ranked_candidates).
        Candidates are sorted best-first by combined_score.
        """
        prompt = self._model.build_prompt(message, history, four_stream=four_stream)
        candidates: list[RankedResponse] = []

        for i, temp in enumerate(self.TEMPERATURES):
            text = self._model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed + i,
            )
            if not text:
                continue
            h = _heuristic_score(message, text)
            m = self._model.score(prompt, text)
            candidates.append(RankedResponse(text, h, m, temp))

        if not candidates:
            return "", []

        candidates.sort(key=lambda c: -c.combined_score)
        return candidates[0].text, candidates
