"""
Obscuro Unified Agent — single entry point for the full cognitive system.

The model and agent are not separate things. This module wires them together:
  - OrbModel     → the language core
  - CognitiveLoop → ReAct execution (Thought → Action → Observation → Answer)
  - Memory        → TF-IDF persistent episodic store
  - ToolRegistry  → full action space (shell, python, filesystem, web)
  - MultiPathReasoner, Critic, CuriosityEngine → retained for non-tool mode

Usage:
    agent = OrbAgent()
    result = agent.run(message, history)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .model     import OrbModel
from .memory    import Memory
from .loop      import CognitiveLoop, LoopResult, LoopStep
from .tools     import ToolRegistry
from .reasoning import MultiPathReasoner, RankedResponse
from .critic    import Critic, CritiqueResult
from .curiosity import CuriosityEngine


@dataclass
class AgentOptions:
    max_new_tokens:     int   = 400
    temperature:        float = 0.7
    top_p:              float = 0.95
    top_k:              int   = 50
    repetition_penalty: float = 1.1
    seed:               int   = 42
    four_stream:        bool  = False
    use_critique:       bool  = False
    use_memory:         bool  = True
    multi_path:         bool  = True
    use_tools:          bool  = True   # ReAct loop with full tool access


@dataclass
class AgentResponse:
    response:        str
    critique:        str                  = ""
    reasoning_paths: list[RankedResponse] = field(default_factory=list)
    memory_hits:     list[str]            = field(default_factory=list)
    elapsed_ms:      int                  = 0
    loop_steps:      list[LoopStep]       = field(default_factory=list)
    used_tools:      list[str]            = field(default_factory=list)
    iterations:      int                  = 0


class OrbAgent:
    """
    Obscuro Unified Agent.

    Two execution modes:
      1. ReAct / Agentic (use_tools=True, default)
         The model generates Thought+Action sequences; the loop executes real
         tools (shell, Python, filesystem, web) and injects observations back.
         This is the primary mode for autonomous, multi-step, and expert tasks.

      2. Multi-path reasoning (use_tools=False)
         Three candidate responses generated at different temperatures, ranked
         by a combined heuristic+model score. Optional constitutional critique.
         Retained for fast conversational responses.
    """

    def __init__(self) -> None:
        self.model     = OrbModel()
        self.memory    = Memory()
        self.tools     = ToolRegistry()
        self.loop      = CognitiveLoop(self.model, self.memory, self.tools)
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

        message = message.strip()
        if not message:
            return AgentResponse(response="_(empty message)_")

        # ── Mode 1: ReAct loop (agentic, tools enabled) ───────────────────────
        if options.use_tools:
            loop_result: LoopResult = self.loop.run(
                message, history,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                use_tools=True,
                use_memory=options.use_memory,
                seed=options.seed,
            )
            return AgentResponse(
                response=loop_result.answer,
                memory_hits=loop_result.memory_hits,
                elapsed_ms=loop_result.elapsed_ms,
                loop_steps=loop_result.steps,
                used_tools=loop_result.used_tools,
                iterations=loop_result.iterations,
            )

        # ── Mode 2: Multi-path reasoning (no tools) ───────────────────────────
        t0     = time.monotonic()
        result = AgentResponse(response="")

        if options.use_memory:
            hits = self.memory.retrieve_similar(message, k=3)
            result.memory_hits = [ep.summary() for ep in hits]

        if options.multi_path:
            response, paths = self.reasoner.run(
                message, history,
                max_new_tokens=options.max_new_tokens,
                top_p=options.top_p, top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                four_stream=options.four_stream,
                seed=options.seed,
            )
            result.reasoning_paths = paths
        else:
            prompt = self.model.build_prompt(
                message, history,
                use_tools=False,
                four_stream=options.four_stream,
            )
            response = self.model.generate(
                prompt,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_p=options.top_p, top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                seed=options.seed,
            )

        if not response:
            response = "_(empty response — try rephrasing or lowering temperature)_"

        if options.use_critique:
            cr: CritiqueResult = self.critic.run(
                message, response,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_p=options.top_p, top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                seed=options.seed,
            )
            response       = cr.best
            result.critique = cr.critique

        result.response = response

        if options.use_memory:
            score = result.reasoning_paths[0].combined_score if result.reasoning_paths else 0.0
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
