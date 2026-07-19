"""
Orb constitutional self-critique — multi-dimensional scoring + revision.

v2: five-axis dimensional scoring.
Each axis is scored 0–10 by fast heuristics (no model call for scoring).
A model-based critique + revision runs only when at least one axis < 6.

Axes
────
  accuracy          factual precision; uncertainty acknowledged
  completeness      all aspects of the question addressed
  safety            no harmful, misleading, or dangerous content
  clarity           accessible to a non-expert; well-structured
  epistemic_honesty appropriate hedging on uncertain claims

Reference: Anthropic Constitutional AI (Bai et al. 2022)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import OrbModel


# ── Templates ─────────────────────────────────────────────────────────────────

_CRITIQUE_TMPL = """\
Review this response against the Orb constitution. Be critical and specific.

Question : {question}
Response : {response}

Evaluate against these principles:
1. ACCURACY — are all claims correct and well-supported?
2. COMPLETENESS — are important aspects missing?
3. CLARITY — would a non-expert understand it?
4. HONESTY — is uncertainty appropriately expressed?
5. HELPFULNESS — does it directly address the user's real need?

Concise critique (2–4 sentences, name specific issues):"""

_REVISION_TMPL = """\
Improve this response based on the critique below.

Original question : {question}
Original response : {response}
Critique          : {critique}

Improved response that fully addresses the identified issues:"""


# ── Dimensional scoring ───────────────────────────────────────────────────────

@dataclass
class DimensionalScore:
    accuracy:         float = 8.0
    completeness:     float = 8.0
    safety:           float = 10.0
    clarity:          float = 8.0
    epistemic_honesty: float = 8.0

    @property
    def overall(self) -> float:
        return (
            self.accuracy * 0.30
            + self.completeness * 0.20
            + self.safety * 0.25
            + self.clarity * 0.15
            + self.epistemic_honesty * 0.10
        )

    @property
    def weakest_axis(self) -> tuple[str, float]:
        axes = {
            "accuracy":         self.accuracy,
            "completeness":     self.completeness,
            "safety":           self.safety,
            "clarity":          self.clarity,
            "epistemic_honesty": self.epistemic_honesty,
        }
        k = min(axes, key=lambda x: axes[x])
        return k, axes[k]

    def passes(self, threshold: float = 6.0) -> bool:
        """All axes ≥ threshold — no revision needed."""
        return all(
            v >= threshold
            for v in (
                self.accuracy, self.completeness,
                self.safety, self.clarity, self.epistemic_honesty,
            )
        )

    def report(self) -> str:
        ax, val = self.weakest_axis
        return (
            f"acc={self.accuracy:.0f} comp={self.completeness:.0f} "
            f"safe={self.safety:.0f} clar={self.clarity:.0f} "
            f"epist={self.epistemic_honesty:.0f} "
            f"→ overall={self.overall:.1f} "
            f"[weakest: {ax}={val:.0f}]"
        )


def _score_heuristic(question: str, response: str) -> DimensionalScore:
    """
    Fast heuristic dimensional scoring — no model call.
    Produces approximate scores; the model critique handles nuance.
    """
    r  = response.strip()
    rl = r.lower()
    q  = question.strip().lower()
    n  = len(r.split())

    # ── Accuracy ──────────────────────────────────────────────────────────────
    # Penalise contradictions, "I don't know" with no follow-up, very short answers
    accuracy = 8.0
    if n < 8:
        accuracy -= 2.0
    if re.search(r"\bi don'?t know\b", rl) and n < 20:
        accuracy -= 1.5
    if re.search(r"\b(always|never|everyone|no one)\b", rl):
        accuracy -= 0.5   # overly absolute claims

    # ── Completeness ──────────────────────────────────────────────────────────
    completeness = 8.0
    q_words  = set(re.findall(r'\w{4,}', q))
    r_words  = set(re.findall(r'\w{4,}', rl))
    coverage = len(q_words & r_words) / max(len(q_words), 1)
    completeness += (coverage - 0.3) * 4.0   # ±coverage bonus/penalty
    if n < 15:
        completeness -= 2.0
    completeness = max(1.0, min(10.0, completeness))

    # ── Safety ────────────────────────────────────────────────────────────────
    safety = 10.0
    danger_patterns = [
        r'\b(harm|kill|attack|exploit|malware|virus|illegal|weapon)\b',
        r'\b(credit card|password|ssn|social security)\b',
    ]
    for pat in danger_patterns:
        if re.search(pat, rl):
            safety -= 3.0
    safety = max(0.0, safety)

    # ── Clarity ───────────────────────────────────────────────────────────────
    clarity = 8.0
    # Very long sentences are hard to follow
    sentences = re.split(r'[.!?]+', r)
    avg_sent = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if avg_sent > 35:
        clarity -= 1.5
    # Code blocks and lists improve clarity
    if re.search(r'```|^\d+[\.\)]|\* ', r, re.MULTILINE):
        clarity += 0.5
    clarity = max(1.0, min(10.0, clarity))

    # ── Epistemic honesty ─────────────────────────────────────────────────────
    epistemic = 8.0
    hedges = ["might", "may", "could", "possibly", "likely", "uncertain",
              "not sure", "i believe", "as far as", "to my knowledge"]
    if n > 40 and not any(h in rl for h in hedges):
        epistemic -= 1.5   # long answer with zero hedging — suspicious
    if re.search(r"\b(definitely|certainly|absolutely|guaranteed|100%)\b", rl):
        epistemic -= 1.0
    epistemic = max(1.0, min(10.0, epistemic))

    return DimensionalScore(
        accuracy=max(1.0, min(10.0, accuracy)),
        completeness=completeness,
        safety=safety,
        clarity=clarity,
        epistemic_honesty=epistemic,
    )


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class CritiqueResult:
    original:         str
    critique:         str
    revised:          str
    dimensional_scores: DimensionalScore = field(default_factory=DimensionalScore)

    def improved(self) -> bool:
        """True if the revision is meaningfully different from the original."""
        if not self.revised:
            return False
        orig = set(self.original.lower().split())
        rev  = set(self.revised.lower().split())
        if not orig:
            return bool(rev)
        overlap = len(orig & rev) / len(orig)
        return overlap < 0.88

    @property
    def best(self) -> str:
        return self.revised if self.improved() else self.original

    @property
    def passed(self) -> bool:
        return self.dimensional_scores.passes()


# ── Critic ────────────────────────────────────────────────────────────────────

class Critic:
    """Constitutional self-critique for Orb responses — now with dimensional scoring."""

    def __init__(self, model: "OrbModel") -> None:
        self._model = model

    def score(self, question: str, response: str) -> DimensionalScore:
        """
        Fast heuristic scoring — no model call, ~0 ms.
        Returns a DimensionalScore.
        """
        return _score_heuristic(question, response)

    def run(
        self,
        question: str,
        response: str,
        *,
        max_new_tokens:     int   = 200,
        temperature:        float = 0.75,
        top_p:              float = 0.95,
        top_k:              int   = 50,
        repetition_penalty: float = 1.1,
        seed:               int   = 43,
        skip_threshold:     float = 7.0,   # skip model critique if all axes ≥ this
    ) -> CritiqueResult:
        """
        Two-phase constitutional critique.

        Phase 0 — heuristic scoring (always runs, ~0 ms)
        Phase 1 — model critique  (skipped if heuristic passes all axes ≥ skip_threshold)
        Phase 2 — model revision  (skipped if critique found nothing substantial)
        """
        dim = _score_heuristic(question, response)

        # Fast path: heuristics say it's fine — skip expensive model calls
        if dim.passes(skip_threshold):
            return CritiqueResult(
                original=response,
                critique="[Passed heuristic screening — no issues found]",
                revised=response,
                dimensional_scores=dim,
            )

        # Phase 1 — model critique
        critique = self._model.generate(
            _CRITIQUE_TMPL.format(question=question, response=response),
            max_new_tokens=min(max_new_tokens, 150),
            temperature=max(temperature - 0.1, 0.45),
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ) or "No specific issues identified."

        # Phase 2 — revision (only if critique is substantive)
        substantive = (
            len(critique.split()) > 10
            and "no specific issues" not in critique.lower()
            and "no issues" not in critique.lower()
        )

        if substantive:
            revised = self._model.generate(
                _REVISION_TMPL.format(
                    question=question, response=response, critique=critique,
                ),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed + 1,
            ) or ""
        else:
            revised = ""

        return CritiqueResult(
            original=response,
            critique=critique,
            revised=revised,
            dimensional_scores=dim,
        )
