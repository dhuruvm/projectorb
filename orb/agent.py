"""
Obscuro Unified Agent — single entry point for the full cognitive system.

v2: metacognition + subconscious wired into the core.

Architecture
────────────
  OrbModel              the language core (Llama-3.2-1B + LoRA adapters)
  CognitiveLoop         ReAct execution — model output IS action selection
  Memory                TF-IDF persistent episodic store
  ToolRegistry          full action space (shell, python, filesystem, web)
  MetacognitionEngine   pre-loop register/depth/intuition assessment
  SubconsciousProcessor priming, pattern learning, episodic consolidation
  MultiPathReasoner     multi-temp candidate ranking + self-consistency vote
  Critic                constitutional critique with 5-axis dimensional scoring
  CuriosityEngine       clarifying question generation

Usage:
    agent = OrbAgent()
    result = agent.run(message, history)          # default: ReAct + tools
    result = agent.run(message, history, options) # custom options
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .model          import OrbModel
from .memory         import Memory
from .loop           import CognitiveLoop, LoopResult, LoopStep
from .tools          import ToolRegistry
from .reasoning      import MultiPathReasoner, RankedResponse, ReasoningResult
from .critic         import Critic, CritiqueResult, DimensionalScore
from .curiosity      import CuriosityEngine
from .metacognition  import MetacognitionEngine
from .subconscious   import SubconsciousProcessor
from .types          import AgentOptions, AgentResponse


class OrbAgent:
    """
    Obscuro Unified Agent.

    Two execution modes:
      1. ReAct / Agentic (use_tools=True, default)
         Metacognition → Subconscious priming → Planning → ReAct loop
         → Reflection → Confidence scoring → Pattern recording

      2. Multi-path reasoning (use_tools=False)
         Three candidate responses ranked by combined heuristic+model score
         + self-consistency vote → optional 5-axis constitutional critique.
    """

    def __init__(self) -> None:
        self.model          = OrbModel()
        self.memory         = Memory()
        self.tools          = ToolRegistry()
        self.metacognition  = MetacognitionEngine()
        self.subconscious   = SubconsciousProcessor()
        self.loop           = CognitiveLoop(
            model         = self.model,
            memory        = self.memory,
            tools         = self.tools,
            metacognition = self.metacognition,
            subconscious  = self.subconscious,
        )
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
                max_new_tokens = options.max_new_tokens,
                temperature    = options.temperature,
                use_tools      = True,
                use_memory     = options.use_memory,
                seed           = options.seed,
            )
            return AgentResponse(
                response     = loop_result.answer,
                memory_hits  = loop_result.memory_hits,
                elapsed_ms   = loop_result.elapsed_ms,
                loop_steps   = loop_result.steps,
                used_tools   = loop_result.used_tools,
                iterations   = loop_result.iterations,
                confidence   = loop_result.confidence,
                register     = loop_result.register,
                plan         = loop_result.plan,
                primed       = loop_result.primed,
            )

        # ── Mode 2: Multi-path reasoning (no tools) ───────────────────────────
        t0     = time.monotonic()
        result = AgentResponse(response="")

        # Metacognition assessment (for register + param tuning)
        assessment = self.metacognition.assess(message, [])
        result.register = assessment.register
        # Tune params
        adjusted       = self.metacognition.adjust_params(
            assessment,
            {
                "max_new_tokens": options.max_new_tokens,
                "temperature":    options.temperature,
                "top_p":          options.top_p,
            },
        )
        max_tokens = adjusted.get("max_new_tokens", options.max_new_tokens)
        temperature = adjusted.get("temperature",    options.temperature)
        top_p       = adjusted.get("top_p",          options.top_p)

        # Memory
        if options.use_memory:
            hits            = self.memory.retrieve_similar(message, k=3)
            result.memory_hits = [ep.summary() for ep in hits]

        if options.multi_path:
            reasoning: ReasoningResult = self.reasoner.run_full(
                message, history,
                max_new_tokens     = max_tokens,
                top_p              = top_p,
                top_k              = options.top_k,
                repetition_penalty = options.repetition_penalty,
                four_stream        = options.four_stream,
                seed               = options.seed,
            )
            response               = reasoning.best_text
            result.reasoning_paths = reasoning.candidates
            result.consistency_score = reasoning.consistency_score
        else:
            prompt = self.model.build_prompt(
                message, history,
                use_tools   = False,
                four_stream = options.four_stream,
            )
            response = self.model.generate(
                prompt,
                max_new_tokens     = max_tokens,
                temperature        = temperature,
                top_p              = top_p,
                top_k              = options.top_k,
                repetition_penalty = options.repetition_penalty,
                seed               = options.seed,
            )

        if not response:
            response = "_(empty response — try rephrasing or lowering temperature)_"

        # Constitutional critique with dimensional scoring
        if options.use_critique:
            cr: CritiqueResult = self.critic.run(
                message, response,
                max_new_tokens     = max_tokens,
                temperature        = temperature,
                top_p              = top_p,
                top_k              = options.top_k,
                repetition_penalty = options.repetition_penalty,
                seed               = options.seed,
            )
            response                = cr.best
            result.critique         = cr.critique
            result.dimensional_scores = cr.dimensional_scores
        else:
            # Fast heuristic scoring always runs (free)
            result.dimensional_scores = self.critic.score(message, response)

        # Confidence from perplexity
        ppl              = self.model.score("", response)
        result.confidence = self.metacognition.score_confidence(ppl)

        result.response = response

        if options.use_memory:
            qual = result.reasoning_paths[0].combined_score if result.reasoning_paths else 0.0
            qual = min(qual, 0.5 + 0.5 * result.confidence)
            self.memory.add_episode(
                question=message,
                response=response,
                critique=result.critique,
                score=qual,
            )

        result.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return result

    # ── Maintenance ───────────────────────────────────────────────────────────

    def consolidate(self) -> int:
        """Run subconscious episodic consolidation. Returns clusters created."""
        return self.subconscious.consolidate(self.memory)

    # ── Introspection ──────────────────────────────────────────────────────────

    def memory_stats(self) -> dict:
        stats = self.memory.count()
        stats.update(self.subconscious.stats())
        return stats

    @property
    def model_label(self) -> str:
        return self.model.label
