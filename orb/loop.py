"""
Obscuro Unified Cognitive Loop — ReAct: Thought → Action → Observation → Answer.

This IS the agent. The model and agent are one system.
The model generates structured thought+action sequences; the loop executes
them and feeds observations back until a Final Answer emerges.

Design principle: generation IS action selection. The model's output drives
real execution. There is no separate planner or policy network.

v2 additions
────────────
  Planning phase    before the loop, the model drafts a short numbered plan
                    ("I will: 1. list files  2. read X  3. …")
  Reflection        after each tool observation, a fast heuristic checks
                    if we're on track — consecutive failures trigger recovery
  Metacognition     pre-loop assessment of register/depth/intuitive tools
  Subconscious      priming context injected as background knowledge
  Confidence        post-generation certainty score on the final answer
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .tools import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from .model      import OrbModel
    from .memory     import Memory
    from .metacognition import MetacognitionEngine, MetacognitiveAssessment
    from .subconscious  import SubconsciousProcessor


_MAX_ITERATIONS = 12
_MAX_SCRATCHPAD = 8000

_ACTION_RE = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL,
)
_JSON_BLOB_RE = re.compile(r'\{[^{}]*"tool"[^{}]*\}', re.DOTALL)
_FINAL_RE     = re.compile(
    r'(?:Final Answer|FINAL ANSWER|Final:|Answer:)\s*(.+)',
    re.DOTALL | re.IGNORECASE,
)

# Error signals used by heuristic reflection
_FAILURE_SIGNALS = (
    "error:", "traceback", "exception:", "not found", "permission denied",
    "command not found", "no such file", "syntax error", "nameerror",
    "typeerror", "valueerror", "oserror", "filenotfounderror",
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_action(text: str) -> tuple[str, dict] | None:
    """
    Extract a tool call from model output.
    Tries canonical format first, then a lenient JSON scan.
    Returns (tool_name, args_dict) or None.
    """
    m = _ACTION_RE.search(text)
    if m:
        try:
            args = json.loads(m.group(2))
            if isinstance(args, dict):
                return m.group(1), args
        except json.JSONDecodeError:
            pass

    for blob in _JSON_BLOB_RE.finditer(text):
        try:
            obj = json.loads(blob.group(0))
            if isinstance(obj, dict) and "tool" in obj:
                tool = obj["tool"]
                args = obj.get("args") if isinstance(obj.get("args"), dict) else {
                    k: v for k, v in obj.items() if k != "tool"
                }
                return str(tool), args
        except json.JSONDecodeError:
            continue

    return None


def _reflect(observation: str) -> str:
    """
    Heuristic reflection: assess whether the observation signals success.
    Returns 'on_track' | 'failed' | 'empty'.
    """
    if not observation or len(observation.strip()) < 4:
        return "empty"
    obs_lower = observation.lower()
    if any(sig in obs_lower for sig in _FAILURE_SIGNALS):
        return "failed"
    return "on_track"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class LoopStep:
    iteration:    int
    thought:      str
    action:       str | None        = None
    observation:  str | None        = None
    tool_result:  ToolResult | None = None
    reflection:   str               = "on_track"   # "on_track" | "failed" | "empty"

    def summary(self) -> str:
        if self.action:
            status = "✓" if (self.tool_result and self.tool_result.success) else "✗"
            tool   = self.tool_result.tool if self.tool_result else "?"
            return f"[{self.iteration}] Action: {tool} {status}  reflect={self.reflection}"
        return f"[{self.iteration}] Thought"


@dataclass
class LoopResult:
    answer:       str
    steps:        list[LoopStep] = field(default_factory=list)
    iterations:   int            = 0
    elapsed_ms:   int            = 0
    memory_hits:  list[str]      = field(default_factory=list)
    used_tools:   list[str]      = field(default_factory=list)
    plan:         str            = ""      # the pre-loop execution plan
    confidence:   float          = 0.5    # post-hoc certainty score
    register:     str            = "analytical"
    primed:       bool           = False  # was subconscious priming active?


# ── Cognitive Loop ────────────────────────────────────────────────────────────

class CognitiveLoop:
    """
    The Unified Cognitive Loop for Obscuro.

    v2: planning → priming → ReAct → reflection → confidence scoring
    """

    def __init__(
        self,
        model:          "OrbModel",
        memory:         "Memory",
        tools:          ToolRegistry | None           = None,
        metacognition:  "MetacognitionEngine | None"  = None,
        subconscious:   "SubconsciousProcessor | None" = None,
    ) -> None:
        self.model         = model
        self.memory        = memory
        self.tools         = tools or ToolRegistry()
        self.metacognition = metacognition
        self.subconscious  = subconscious

    # ── Planning ──────────────────────────────────────────────────────────────

    def _plan(self, task: str, max_tokens: int = 90) -> str:
        """Generate a short numbered execution plan before the ReAct loop."""
        planning_prompt = (
            f"Task: {task[:280]}\n\n"
            "Write a concise numbered plan (3–5 steps) to solve this task. "
            "Each step should name a specific action or tool.\nPlan:"
        )
        plan = self.model.generate(
            planning_prompt,
            max_new_tokens=max_tokens,
            temperature=0.45,
            seed=77,
        )
        return plan.strip() if plan else ""

    # ── Synchronous run ───────────────────────────────────────────────────────

    def run(
        self,
        message:    str,
        history:    list[dict],
        *,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        use_tools:      bool  = True,
        use_memory:     bool  = True,
        seed:           int   = 42,
    ) -> LoopResult:
        t0     = time.monotonic()
        result = LoopResult(answer="")

        # ── Metacognition pre-assessment ──────────────────────────────────────
        assessment: "MetacognitiveAssessment | None" = None
        if self.metacognition:
            assessment      = self.metacognition.assess(message, self.tools.names)
            result.register = assessment.register
            # Adjust generation params based on register + depth
            adjusted = self.metacognition.adjust_params(
                assessment,
                {"max_new_tokens": max_new_tokens, "temperature": temperature},
            )
            max_new_tokens = adjusted.get("max_new_tokens", max_new_tokens)
            temperature    = adjusted.get("temperature", temperature)

        # ── Memory retrieval ──────────────────────────────────────────────────
        if use_memory:
            hits             = self.memory.retrieve_similar(message, k=3)
            result.memory_hits = [ep.summary() for ep in hits]

        # ── Subconscious priming ──────────────────────────────────────────────
        priming_fragment = ""
        if self.subconscious and use_memory:
            reg     = assessment.register if assessment else "analytical"
            priming = self.subconscious.prime(message, self.memory, register=reg)
            if priming.is_active():
                priming_fragment = priming.to_prompt_fragment()
                result.primed    = True

        # ── Pre-loop execution plan ───────────────────────────────────────────
        plan = ""
        if use_tools:
            plan        = self._plan(message)
            result.plan = plan

        # ── Build initial context ─────────────────────────────────────────────
        context_parts: list[str] = []
        if priming_fragment:
            context_parts.append(priming_fragment)
        if result.memory_hits:
            context_parts.append(
                "Relevant past experience:\n"
                + "\n".join(f"  • {h}" for h in result.memory_hits)
            )
        if plan:
            context_parts.append(f"Execution plan:\n{plan}")

        task_input = ("\n\n".join(context_parts) + "\n\n" if context_parts else "") + message
        scratchpad = ""

        # Consecutive failure counter — triggers recovery strategy
        consec_failures   = 0
        _MAX_CONSEC_FAIL  = 2

        # ── ReAct loop ────────────────────────────────────────────────────────
        for iteration in range(_MAX_ITERATIONS):
            result.iterations = iteration + 1

            prompt_msg = task_input
            if scratchpad:
                prompt_msg = task_input + "\n\n" + scratchpad.rstrip()

            # Recovery hint after consecutive failures
            if consec_failures >= _MAX_CONSEC_FAIL:
                prompt_msg += (
                    "\n\nNote: Previous attempts failed. "
                    "Try a completely different approach or tool."
                )
                consec_failures = 0

            prompt = self.model.build_prompt(
                prompt_msg, history, use_tools=use_tools,
            )

            raw = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed + iteration,
            )

            if not raw:
                break

            # Check for Final Answer
            final_m = _FINAL_RE.search(raw)
            if final_m:
                result.answer = final_m.group(1).strip()
                result.steps.append(LoopStep(iteration, raw))
                break

            # Execute tool action
            if use_tools:
                action = _extract_action(raw)
                if action:
                    tool_name, args = action
                    tool_result     = self.tools.call(tool_name, args)
                    result.used_tools.append(tool_name)

                    observation = tool_result.format()
                    reflection  = _reflect(observation)

                    if reflection == "failed":
                        consec_failures += 1
                    else:
                        consec_failures = 0

                    step = LoopStep(
                        iteration=iteration,
                        thought=raw,
                        action=json.dumps({"tool": tool_name, "args": args}),
                        observation=observation,
                        tool_result=tool_result,
                        reflection=reflection,
                    )
                    result.steps.append(step)

                    entry = f"\nThought: {raw[:900]}\nObservation: {observation}\n"
                    scratchpad += entry
                    if len(scratchpad) > _MAX_SCRATCHPAD:
                        scratchpad = "...[earlier steps trimmed]...\n" + scratchpad[-_MAX_SCRATCHPAD:]
                    continue

            # No tool action, no Final Answer — treat as answer
            result.steps.append(LoopStep(iteration, raw))
            result.answer = raw
            break

        # Fallback
        if not result.answer:
            result.answer = result.steps[-1].thought if result.steps else "(no response — try rephrasing)"

        # ── Post-generation confidence scoring ────────────────────────────────
        if self.metacognition and result.answer:
            ppl = self.model.score("", result.answer)
            result.confidence = self.metacognition.score_confidence(ppl)
            if assessment:
                assessment.certainty = result.confidence

        # ── Memory + subconscious pattern recording ───────────────────────────
        if use_memory:
            quality = 0.5 + min(0.4, len(result.used_tools) * 0.08)
            quality = min(quality, 0.5 + 0.5 * result.confidence)
            self.memory.add_episode(
                question=message,
                response=result.answer,
                score=quality,
            )

        if self.subconscious and result.used_tools:
            reg = assessment.register if assessment else "analytical"
            self.subconscious.record_pattern(
                query_type=reg,
                tool_sequence=result.used_tools,
                score=quality if use_memory else 0.5,
                success=result.confidence > 0.3,
            )

        result.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return result

    # ── Streaming variant ─────────────────────────────────────────────────────

    def stream(
        self,
        message:        str,
        history:        list[dict],
        *,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        use_tools:      bool  = True,
        use_memory:     bool  = True,
        seed:           int   = 42,
    ) -> Iterator[dict]:
        """Generator variant — yields step dicts as they happen (SSE / sandbox)."""
        t0 = time.monotonic()

        # Metacognition
        assessment = None
        register   = "analytical"
        if self.metacognition:
            assessment = self.metacognition.assess(message, self.tools.names)
            register   = assessment.register
            adjusted   = self.metacognition.adjust_params(
                assessment,
                {"max_new_tokens": max_new_tokens, "temperature": temperature},
            )
            max_new_tokens = adjusted.get("max_new_tokens", max_new_tokens)
            temperature    = adjusted.get("temperature", temperature)
            yield {
                "phase": "metacognition", "status": "complete",
                "register": register,
                "depth": assessment.reasoning_depth,
                "intuitive_tools": assessment.intuitive_tools,
            }

        # Memory
        hits: list = []
        if use_memory:
            hits = self.memory.retrieve_similar(message, k=3)
        yield {
            "phase": "memory", "status": "complete",
            "hits": [ep.summary() for ep in hits],
            "count": len(hits),
        }

        # Subconscious priming
        priming_fragment = ""
        if self.subconscious and use_memory:
            priming = self.subconscious.prime(message, self.memory, register=register)
            if priming.is_active():
                priming_fragment = priming.to_prompt_fragment()
                yield {
                    "phase": "priming", "status": "complete",
                    "activation": priming.activation_score,
                    "memories": len(priming.relevant_memories),
                    "suggested_tools": priming.suggested_tool_sequence,
                }

        # Planning
        plan = ""
        if use_tools:
            plan = self._plan(message)
            if plan:
                yield {"phase": "plan", "status": "complete", "content": plan}

        # Build context
        ctx_parts: list[str] = []
        if priming_fragment:
            ctx_parts.append(priming_fragment)
        if hits:
            ctx_parts.append(
                "Relevant past experience:\n"
                + "\n".join(f"  • {ep.summary()}" for ep in hits)
            )
        if plan:
            ctx_parts.append(f"Execution plan:\n{plan}")

        task_input   = ("\n\n".join(ctx_parts) + "\n\n" if ctx_parts else "") + message
        scratchpad   = ""
        final_answer = ""
        used_tools:  list[str] = []
        consec_fail  = 0

        for iteration in range(_MAX_ITERATIONS):
            prompt_msg = task_input + ("\n\n" + scratchpad.rstrip() if scratchpad else "")
            if consec_fail >= 2:
                prompt_msg += "\n\nNote: Previous attempts failed. Try a different approach."
                consec_fail = 0

            prompt = self.model.build_prompt(prompt_msg, history, use_tools=use_tools)
            yield {"phase": "thinking", "status": "running", "iteration": iteration + 1}

            raw = self.model.generate(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed + iteration,
            )

            if not raw:
                break

            final_m = _FINAL_RE.search(raw)
            if final_m:
                final_answer = final_m.group(1).strip()
                yield {"phase": "answer", "status": "complete",
                       "content": final_answer, "thought": raw}
                break

            if use_tools:
                action = _extract_action(raw)
                if action:
                    tool_name, args = action
                    yield {"phase": "action", "status": "running",
                           "tool": tool_name, "args": args, "thought": raw[:600]}

                    tool_result = self.tools.call(tool_name, args)
                    used_tools.append(tool_name)
                    observation = tool_result.format()
                    reflection  = _reflect(observation)

                    consec_fail = (consec_fail + 1) if reflection == "failed" else 0

                    yield {"phase": "observation", "status": "complete",
                           "tool": tool_name, "output": observation,
                           "success": tool_result.success,
                           "elapsed_ms": tool_result.elapsed_ms,
                           "reflection": reflection}

                    scratchpad += f"\nThought: {raw[:900]}\nObservation: {observation}\n"
                    if len(scratchpad) > _MAX_SCRATCHPAD:
                        scratchpad = "...[trimmed]...\n" + scratchpad[-_MAX_SCRATCHPAD:]
                    continue

            final_answer = raw
            yield {"phase": "answer", "status": "complete", "content": raw}
            break

        if not final_answer:
            final_answer = "(no response generated)"

        # Confidence
        confidence = 0.5
        if self.metacognition:
            ppl        = self.model.score("", final_answer)
            confidence = self.metacognition.score_confidence(ppl)

        # Learn
        if use_memory:
            quality = min(0.9, 0.5 + min(0.4, len(used_tools) * 0.08) + 0.2 * confidence)
            self.memory.add_episode(question=message, response=final_answer, score=quality)

        if self.subconscious and used_tools:
            self.subconscious.record_pattern(
                query_type=register,
                tool_sequence=used_tools,
                score=confidence,
                success=confidence > 0.3,
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        yield {
            "phase": "done", "status": "complete",
            "elapsed_ms": elapsed_ms, "tools_used": used_tools,
            "iterations": iteration + 1, "confidence": confidence,
            "register": register,
        }
