"""
Orb Executive Controller — the cognitive loop.

Each call to run() executes the full loop:
  Observe → Remember → Reason → Critique → Learn → Respond
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .model import OrbModel
from .memory import Memory
from .reasoning import MultiPathReasoner, RankedResponse
from .critic import Critic, CritiqueResult
from .curiosity import CuriosityEngine


@dataclass
class AgentOptions:
    max_new_tokens:     int   = 200
    temperature:        float = 0.85
    top_p:              float = 0.95
    top_k:              int   = 50
    repetition_penalty: float = 1.1
    seed:               int   = 42
    four_stream:        bool  = False
    use_critique:       bool  = False
    use_memory:         bool  = True
    multi_path:         bool  = True


@dataclass
class AgentResponse:
    response:        str
    critique:        str                  = ""
    reasoning_paths: list[RankedResponse] = field(default_factory=list)
    memory_hits:     list[str]            = field(default_factory=list)
    elapsed_ms:      int                  = 0


class OrbAgent:
    """
    Executive controller for the Orb cognitive loop.

    Usage:
        agent = OrbAgent()                       # load once at startup
        result = agent.run(message, history)     # call per user turn
    """

    def __init__(self) -> None:
        self.model     = OrbModel()
        self.memory    = Memory()
        self.reasoner  = MultiPathReasoner(self.model)
        self.critic    = Critic(self.model)
        self.curiosity = CuriosityEngine(self.model)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        message: str,
        history: list[dict],
        options: AgentOptions | None = None,
    ) -> AgentResponse:
        if options is None:
            options = AgentOptions()

        t0 = time.monotonic()
        result = AgentResponse(response="")

        # 1. Observe
        message = message.strip()
        if not message:
            result.response = "_(empty message)_"
            return result

        # 2. Remember
        if options.use_memory:
            hits = self.memory.retrieve_similar(message, k=3)
            result.memory_hits = [ep.summary() for ep in hits]

        # 3. Reason
        if options.multi_path:
            response, paths = self.reasoner.run(
                message, history,
                max_new_tokens=options.max_new_tokens,
                top_p=options.top_p,
                top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                four_stream=options.four_stream,
                seed=options.seed,
            )
            result.reasoning_paths = paths
        else:
            prompt = self.model.build_prompt(
                message, history, four_stream=options.four_stream
            )
            response = self.model.generate(
                prompt,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                seed=options.seed,
            )
            paths = []

        if not response:
            response = (
                "_(Obscuro returned an empty response — "
                "try rephrasing or lowering temperature.)_"
            )

        # 4. Critique and revise (optional — 3× slower)
        if options.use_critique:
            cr: CritiqueResult = self.critic.run(
                message, response,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                seed=options.seed,
            )
            response = cr.best
            result.critique = cr.critique

        result.response = response

        # 5. Learn
        if options.use_memory:
            score = paths[0].combined_score if paths else 0.0
            self.memory.add_episode(
                question=message,
                response=response,
                critique=result.critique,
                score=score,
            )

        result.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return result

    # ── Introspection ─────────────────────────────────────────────────────────

    def memory_stats(self) -> dict[str, int]:
        return self.memory.count()

    @property
    def model_label(self) -> str:
        return self.model.label
