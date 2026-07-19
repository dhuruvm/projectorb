"""
Orb constitutional self-critique and revision (OASIS Phase 2).

Two-phase process:
  1. Critique  — identify specific weaknesses against the Orb constitution
  2. Revision  — produce an improved response addressing the critique

The critic also exposes a lightweight improved() check so the agent
can detect when the revision is substantively different from the original
(and skip it if it isn't, rather than silently regressing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import OrbModel


_CRITIQUE_TEMPLATE = """\
Review this response and identify specific weaknesses:

Question: {question}
Response: {response}

Apply these checks — be critical and specific:
1. Accuracy — any claims that might be wrong or unverified?
2. Completeness — important aspects missing?
3. Clarity — would a non-expert understand it?
4. Honesty — is uncertainty appropriately expressed?
5. Helpfulness — does it actually address the user's real need?

Brief critique (2-4 sentences, be specific):"""

_REVISION_TEMPLATE = """\
Improve this response based on the critique.

Original question: {question}
Original response: {response}
Critique: {critique}

Improved response that addresses the identified issues:"""


@dataclass
class CritiqueResult:
    original: str
    critique: str
    revised: str

    def improved(self) -> bool:
        """
        True if the revision is meaningfully different from the original.
        Measured by word-level overlap — identical or near-identical revisions
        are considered failures and the original is kept instead.
        """
        if not self.revised:
            return False
        orig = set(self.original.lower().split())
        rev  = set(self.revised.lower().split())
        if not orig:
            return bool(rev)
        overlap = len(orig & rev) / len(orig)
        return overlap < 0.88  # more than 12% new words → substantive revision

    @property
    def best(self) -> str:
        """Return the revision if it improved on the original, else the original."""
        return self.revised if self.improved() else self.original


class Critic:
    """Constitutional self-critique for Orb responses."""

    def __init__(self, model: "OrbModel") -> None:
        self._model = model

    def run(
        self,
        question: str,
        response: str,
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.75,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int = 43,
    ) -> CritiqueResult:
        # Phase 1 — critique
        critique = self._model.generate(
            _CRITIQUE_TEMPLATE.format(question=question, response=response),
            max_new_tokens=min(max_new_tokens, 140),
            temperature=max(temperature - 0.1, 0.5),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        # Phase 2 — revision
        revised = self._model.generate(
            _REVISION_TEMPLATE.format(
                question=question,
                response=response,
                critique=critique or "No specific issues found.",
            ),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed + 1,
        )

        return CritiqueResult(original=response, critique=critique, revised=revised)
