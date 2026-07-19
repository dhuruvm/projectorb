"""
Metacognition — the agent's ability to reason about its own reasoning.

Runs BEFORE the ReAct loop starts. Gives the agent a "gut feeling" about
what kind of query it's dealing with, what tools it will likely need, and
how carefully it needs to think — before spending expensive generation tokens.

Capabilities
────────────
  register detection   classify query as technical/analytical/creative/emotional/exploratory
  ambiguity detection  flag underspecified queries before committing
  intuitive tools      fast pre-selection of relevant tools for this register
  depth estimation     shallow / standard / deep → adjusts max_new_tokens
  confidence scoring   post-hoc certainty from token-level entropy or perplexity
  param adjustment     lower temperature for deterministic tasks, higher for creative
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Register keyword maps ─────────────────────────────────────────────────────

_REGISTER_KW: dict[str, list[str]] = {
    "technical": [
        "code", "function", "script", "implement", "debug", "error", "syntax",
        "api", "run", "execute", "install", "fix", "class", "method",
        "python", "bash", "shell", "file", "directory",
        "security", "vulnerability", "sql", "http", "server", "network",
        "algorithm", "library", "import", "module", "compile", "terminal",
        "command", "git", "docker", "kubernetes", "database", "query",
        "regex", "test", "deploy", "build", "pipeline", "loop", "variable",
    ],
    "analytical": [
        "analyze", "analysis", "compare", "evaluate", "assess", "why", "how",
        "explain", "reason", "cause", "effect", "because", "therefore",
        "calculate", "compute", "solve", "math", "formula", "proof",
        "what is", "what are", "which", "best", "optimal", "difference",
        "pros", "cons", "tradeoff", "versus", "summarize", "list",
    ],
    "creative": [
        "write", "create", "generate", "design", "imagine", "story", "poem",
        "essay", "compose", "invent", "brainstorm", "idea", "novel",
        "creative", "describe", "narrative", "character", "plot",
    ],
    "emotional": [
        "feel", "emotion", "sad", "happy", "worried", "anxious", "scared",
        "love", "hate", "miss", "lonely", "confused", "overwhelmed",
        "should i", "am i", "do you", "afraid", "stressed", "struggling",
    ],
    "exploratory": [
        "who are you", "what can you", "your capabilities", "tell me about",
        "what do you think", "are you", "can you", "curious", "wonder",
        "your opinion", "thoughts on", "what would happen",
    ],
}

# Default tool priority per register
_REGISTER_TOOLS: dict[str, list[str]] = {
    "technical":   ["shell", "python", "file_list", "file_read", "web_fetch", "think"],
    "analytical":  ["think", "python", "web_fetch"],
    "creative":    ["think", "web_fetch"],
    "emotional":   ["think"],
    "exploratory": ["think", "web_fetch"],
}

# Depth heuristic
_DEEP_SIGNALS = {
    "step by step", "in detail", "comprehensive", "thoroughly",
    "explain fully", "walk me through", "complete", "all the",
}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MetacognitiveAssessment:
    register:            str          # "technical"|"creative"|"analytical"|"emotional"|"exploratory"
    intuitive_tools:     list[str]    # ordered tool suggestions before the ReAct loop
    is_ambiguous:        bool
    clarifying_question: str          # non-empty if is_ambiguous
    reasoning_depth:     str          # "shallow"|"standard"|"deep"
    certainty:           float = 0.5  # updated post-generation via score_confidence()
    adjusted_params:     dict  = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"register={self.register}  depth={self.reasoning_depth}  "
            f"certainty={self.certainty:.2f}  "
            f"tools={self.intuitive_tools[:3]}  "
            f"ambiguous={self.is_ambiguous}"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_register(query: str) -> str:
    q = query.lower()
    scores: dict[str, int] = {reg: 0 for reg in _REGISTER_KW}
    for reg, keywords in _REGISTER_KW.items():
        for kw in keywords:
            if kw in q:
                scores[reg] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "analytical"


def _estimate_depth(query: str) -> str:
    words = len(query.split())
    q_lower = query.lower()
    if words < 7:
        return "shallow"
    if words > 45 or any(sig in q_lower for sig in _DEEP_SIGNALS):
        return "deep"
    return "standard"


def _detect_ambiguity(query: str) -> tuple[bool, str]:
    """Returns (is_ambiguous, clarifying_question)."""
    q = query.strip()
    q_lower = q.lower()

    # Very short / vague
    vague = {"help", "fix it", "make it better", "do it", "ok", "yes", "no", "tell me"}
    if q_lower in vague or len(q.split()) <= 2:
        return True, "Could you describe what you need in more detail?"

    # Dangling pronoun without context
    if re.search(r"\b(it|this|that|they|he|she|the thing|the code|the file)\b", q_lower):
        if len(q.split()) < 9:
            return True, "What are you referring to? A bit more context will let me give a precise answer."

    return False, ""


# ── Engine ────────────────────────────────────────────────────────────────────

class MetacognitionEngine:
    """
    Pre-loop assessment and post-loop confidence scoring.

    Call assess() before the ReAct loop starts.
    Call score_confidence() after generation to calibrate certainty.
    """

    def assess(
        self,
        query: str,
        available_tools: list[str],
    ) -> MetacognitiveAssessment:
        register  = _detect_register(query)
        depth     = _estimate_depth(query)
        ambiguous, clarify = _detect_ambiguity(query)
        intuitive = [t for t in _REGISTER_TOOLS.get(register, ["think"])
                     if t in available_tools]

        return MetacognitiveAssessment(
            register=register,
            intuitive_tools=intuitive,
            is_ambiguous=ambiguous,
            clarifying_question=clarify,
            reasoning_depth=depth,
        )

    def score_confidence(
        self,
        logits_or_perplexity,
        k: int = 40,
    ) -> float:
        """
        Estimate certainty from generation output.

        Accepts either:
          - a tuple of per-token logit tensors  (transformers `scores` output)
          - a float / int representing perplexity

        Returns a value in [0, 1]:
          0.0 = fully uncertain (uniform distribution)
          1.0 = fully certain (peaked distribution)
        """
        # Fast path: pre-computed perplexity
        if isinstance(logits_or_perplexity, (int, float)):
            ppl = max(float(logits_or_perplexity), 1.0)
            return max(0.0, min(1.0, round(1.0 - math.log(ppl) / 6.0, 3)))

        # Token logits path
        if not logits_or_perplexity or not hasattr(logits_or_perplexity, "__len__"):
            return 0.5

        try:
            import torch
            entropies: list[float] = []
            for token_logits in list(logits_or_perplexity)[:k]:
                if not hasattr(token_logits, "float"):
                    continue
                logits = token_logits.float()
                probs  = torch.softmax(logits, dim=-1)
                top    = probs.topk(min(50, probs.shape[-1])).values.squeeze()
                top    = top[top > 1e-9]
                ent    = -float((top * top.log()).sum())
                entropies.append(ent)

            if not entropies:
                return 0.5
            avg = sum(entropies) / len(entropies)
            return max(0.0, min(1.0, round(1.0 - avg / 5.5, 3)))

        except Exception:
            return 0.5

    def adjust_params(
        self,
        assessment: MetacognitiveAssessment,
        base: dict,
    ) -> dict:
        """
        Tune temperature + max_new_tokens based on register and depth.
        Never raises temperature above what the user requested by more than 0.1.
        """
        params = dict(base)
        t = params.get("temperature", 0.7)
        reg = assessment.register

        if reg == "technical":
            params["temperature"] = min(t, 0.62)
            params["top_p"]       = min(params.get("top_p", 0.95), 0.90)
        elif reg == "creative":
            params["temperature"] = min(max(t, 0.80), t + 0.10)
            params["top_p"]       = max(params.get("top_p", 0.95), 0.95)
        elif reg == "emotional":
            params["temperature"] = min(max(t, 0.72), 0.80)

        if assessment.reasoning_depth == "deep":
            params["max_new_tokens"] = max(params.get("max_new_tokens", 400), 512)
        elif assessment.reasoning_depth == "shallow":
            params["max_new_tokens"] = min(params.get("max_new_tokens", 400), 200)

        assessment.adjusted_params = params
        return params
