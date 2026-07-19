"""
Orb multi-path reasoning engine — with self-consistency voting.

Generates N candidate responses at distinct temperatures, scores each with
a combination of model perplexity and fast heuristics, then:
  1. Clusters candidates by answer similarity (Jaccard on lowercased words)
  2. Picks the majority cluster's best representative  ← self-consistency
  3. Returns a consistency_score: fraction of candidates agreeing with winner

Self-consistency reference: Wang et al. 2022 (Self-Consistency Improves
Chain-of-Thought Reasoning in Language Models).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import OrbModel


# ── Heuristic scoring ─────────────────────────────────────────────────────────

def _heuristic_score(question: str, response: str) -> float:
    """
    Fast quality estimate without a model call.

    Components (weighted):
      length_score — reward responses in the 80–500 char sweet spot
      uniqueness   — penalise 4-gram repetition
      relevance    — fraction of question keywords present in response
      structure    — bonus for numbered lists, code blocks, clear answers
    """
    if not response:
        return 0.0

    n = len(response)
    if n < 20:
        length_score = 0.1
    elif n < 80:
        length_score = 0.5
    elif n <= 500:
        length_score = 1.0
    elif n <= 900:
        length_score = 0.80
    else:
        length_score = 0.55

    words = response.lower().split()
    if len(words) >= 4:
        ngrams    = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        uniqueness = len(set(ngrams)) / len(ngrams)
    else:
        uniqueness = 1.0

    q_words   = set(re.findall(r'\w+', question.lower()))
    r_words   = set(re.findall(r'\w+', response.lower()))
    relevance = len(q_words & r_words) / max(len(q_words), 1)

    # Structure bonus: numbered lists, code blocks, explicit answers
    structure = 0.0
    if re.search(r'^\d+[\.\)]', response, re.MULTILINE):
        structure += 0.08
    if "```" in response or "    " in response:
        structure += 0.06
    if re.search(r'(answer|result|therefore|conclusion)[:\s]', response.lower()):
        structure += 0.05

    raw = 0.37 * length_score + 0.30 * uniqueness + 0.22 * relevance + 0.11 * structure
    return min(1.0, raw)


# ── Similarity / clustering ────────────────────────────────────────────────────

def _extract_answer_core(text: str) -> str:
    """
    Strip preamble to get to the actual answer for clustering.
    Uses the last 40% of the response as a proxy for the conclusion.
    """
    stripped = text.strip().lower()
    # Remove common preambles
    for prefix in ("certainly", "sure", "of course", "here is", "here's", "let me"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].lstrip(" ,!:")
    return stripped[-max(60, len(stripped) // 2):]   # tail for comparison


def _jaccard_sim(a: str, b: str) -> float:
    ta = set(re.findall(r'\w+', a))
    tb = set(re.findall(r'\w+', b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cluster_responses(
    candidates: "list[RankedResponse]",
    threshold:  float = 0.28,
) -> "list[list[RankedResponse]]":
    """
    Greedy single-linkage clustering by answer-core Jaccard similarity.
    threshold: responses with similarity ≥ threshold are merged into one cluster.
    """
    clusters: list[list[RankedResponse]] = []
    for cand in candidates:
        core  = _extract_answer_core(cand.text)
        placed = False
        for cluster in clusters:
            rep = _extract_answer_core(cluster[0].text)
            if _jaccard_sim(core, rep) >= threshold:
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])
    return clusters


def self_consistency_vote(
    candidates: "list[RankedResponse]",
    threshold:  float = 0.28,
) -> "tuple[RankedResponse, float]":
    """
    Cluster candidates by answer similarity and return the best member of the
    largest cluster, along with a consistency_score (0–1).

    consistency_score = size_of_winning_cluster / total_candidates
    """
    if not candidates:
        raise ValueError("No candidates to vote on")
    if len(candidates) == 1:
        return candidates[0], 1.0

    clusters = _cluster_responses(candidates, threshold)
    # Largest cluster wins; ties broken by highest combined score
    winning  = max(clusters, key=lambda c: (len(c), max(x.combined_score for x in c)))
    best     = max(winning, key=lambda x: x.combined_score)
    score    = len(winning) / len(candidates)
    return best, round(score, 3)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RankedResponse:
    text:            str
    heuristic_score: float
    model_score:     float
    temperature:     float

    @property
    def combined_score(self) -> float:
        return 0.6 * self.heuristic_score + 0.4 * self.model_score

    def __repr__(self) -> str:
        return (
            f"RankedResponse(T={self.temperature}, "
            f"h={self.heuristic_score:.3f}, m={self.model_score:.3f}, "
            f"combined={self.combined_score:.3f})"
        )


@dataclass
class ReasoningResult:
    """Full output from MultiPathReasoner.run()"""
    best_text:         str
    candidates:        list[RankedResponse] = field(default_factory=list)
    consistency_score: float                = 1.0   # fraction of candidates agreeing
    winner_cluster:    int                  = 1     # size of winning cluster
    total_candidates:  int                  = 0

    @property
    def is_consistent(self) -> bool:
        return self.consistency_score >= 0.67   # 2 of 3 agree


# ── Reasoner ─────────────────────────────────────────────────────────────────

class MultiPathReasoner:
    """
    Generates 3 (or 4) candidate responses, ranks by combined score,
    then applies self-consistency voting to select the final answer.

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
        max_new_tokens:     int   = 200,
        top_p:              float = 0.95,
        top_k:              int   = 50,
        repetition_penalty: float = 1.1,
        four_stream:        bool  = False,
        seed:               int   = 42,
    ) -> tuple[str, list[RankedResponse]]:
        """
        Legacy interface — returns (best_text, candidates).
        See run_full() for the structured ReasoningResult.
        """
        res = self.run_full(
            message, history,
            max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
            four_stream=four_stream, seed=seed,
        )
        return res.best_text, res.candidates

    def run_full(
        self,
        message: str,
        history: list[dict],
        *,
        max_new_tokens:     int   = 200,
        top_p:              float = 0.95,
        top_k:              int   = 50,
        repetition_penalty: float = 1.1,
        four_stream:        bool  = False,
        seed:               int   = 42,
    ) -> ReasoningResult:
        """
        Full structured result including self-consistency score.
        """
        prompt     = self._model.build_prompt(message, history, four_stream=four_stream)
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
            return ReasoningResult(best_text="", candidates=[], total_candidates=0)

        # Sort by combined score (fallback if consistency vote is inconclusive)
        candidates.sort(key=lambda c: -c.combined_score)

        # Self-consistency voting
        best, consistency = self_consistency_vote(candidates)

        return ReasoningResult(
            best_text=best.text,
            candidates=candidates,
            consistency_score=consistency,
            winner_cluster=round(consistency * len(candidates)),
            total_candidates=len(candidates),
        )
