"""
Orb curiosity engine — knowledge gap detection and clarification.

Two capabilities:
  1. needs_clarification(question) — heuristic check for ambiguous questions
  2. generate_clarifying_question(question) — ask the model for a targeted
     clarification when the question lacks enough context for a good answer
  3. identify_gaps(question, response) — ask the model what the response missed

All three are genuinely functional. Output quality is bounded by GPT-2's
117M parameter capacity; the mechanism is real even when the output is rough.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import OrbModel


# Questions matching these patterns often lack enough context to answer well
_AMBIGUITY_PATTERNS = [
    r'^\s*\w[\w\s]{0,20}\?\s*$',                       # very short (< ~4 words)
    r'\b(it|this|that|they|them|he|she)\b.{0,40}\?',   # vague pronoun reference
    r'\b(best|better|worse|good|bad)\b(?!.*\bfor\b)',   # comparative without "for X"
    r'\b(should i|how do i|what do i|can i)\b',         # requires personal context
]

_CLARIFY_TEMPLATE = """\
The user asked: {question}

This question needs more context to answer well. Write ONE short clarifying
question (under 12 words) that would help give a better answer.

Clarifying question:"""

_GAP_TEMPLATE = """\
Question: {question}
Response given: {response}

What important aspects of this question were NOT addressed in the response?
List 1-2 specific gaps, or write "none" if the response is complete:"""


class CuriosityEngine:
    """Detects knowledge gaps and generates targeted clarifying questions."""

    def __init__(self, model: "OrbModel") -> None:
        self._model = model

    def needs_clarification(self, question: str) -> bool:
        """
        Fast heuristic — no model call.
        Returns True when the question is likely too vague to answer well.
        """
        q = question.strip()
        word_count = len(q.split())
        if word_count < 4:
            return True
        for pattern in _AMBIGUITY_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return True
        return False

    def generate_clarifying_question(
        self,
        question: str,
        *,
        seed: int = 99,
    ) -> str:
        """Ask the model to produce a single targeted clarifying question."""
        return self._model.generate(
            _CLARIFY_TEMPLATE.format(question=question),
            max_new_tokens=40,
            temperature=0.70,
            top_p=0.90,
            top_k=40,
            repetition_penalty=1.1,
            seed=seed,
        )

    def identify_gaps(
        self,
        question: str,
        response: str,
        *,
        seed: int = 77,
    ) -> list[str]:
        """
        Ask the model what gaps exist in the given response.
        Returns a list of gap descriptions (may be empty).
        """
        raw = self._model.generate(
            _GAP_TEMPLATE.format(question=question, response=response),
            max_new_tokens=80,
            temperature=0.60,
            top_p=0.90,
            top_k=40,
            repetition_penalty=1.1,
            seed=seed,
        )
        if not raw or re.search(r'\bnone\b', raw[:40], re.IGNORECASE):
            return []
        # Split numbered / bulleted items
        items = re.split(r'[\n]+|(?:\d+[\.\)])', raw)
        return [item.strip(" -•*") for item in items if len(item.strip()) > 15][:2]
