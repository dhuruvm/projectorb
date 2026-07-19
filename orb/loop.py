"""
Obscuro Unified Cognitive Loop — ReAct: Thought → Action → Observation → Answer.

This IS the agent. The model and agent are one system.
The model generates structured thought+action sequences; the loop executes
them and feeds observations back until a Final Answer emerges.

Design principle: generation IS action selection. The model's output drives
real execution. There is no separate planner or policy network.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .tools import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from .model import OrbModel
    from .memory import Memory


_MAX_ITERATIONS  = 12     # max tool calls per turn (prevents infinite loops)
_MAX_SCRATCHPAD  = 8000   # max chars of accumulated context kept between iterations

# Match: {"tool":"name","args":{...}}
_ACTION_RE = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL,
)
# Fallback: any JSON object containing "tool" key
_JSON_BLOB_RE = re.compile(r'\{[^{}]*"tool"[^{}]*\}', re.DOTALL)

_FINAL_RE = re.compile(
    r'(?:Final Answer|FINAL ANSWER|Final:|Answer:)\s*(.+)',
    re.DOTALL | re.IGNORECASE,
)


def _extract_action(text: str) -> tuple[str, dict] | None:
    """
    Extract a tool call from model output.
    Tries canonical format first, then a lenient JSON scan.
    Returns (tool_name, args_dict) or None.
    """
    # Primary: {"tool":"name","args":{...}}
    m = _ACTION_RE.search(text)
    if m:
        try:
            args = json.loads(m.group(2))
            if isinstance(args, dict):
                return m.group(1), args
        except json.JSONDecodeError:
            pass

    # Lenient: scan for any JSON blob with a "tool" key
    for blob in _JSON_BLOB_RE.finditer(text):
        try:
            obj = json.loads(blob.group(0))
            if isinstance(obj, dict) and "tool" in obj:
                tool = obj["tool"]
                # Support both {"tool":"x","args":{}} and flat {"tool":"x","cmd":"ls"}
                args = obj.get("args") if isinstance(obj.get("args"), dict) else {
                    k: v for k, v in obj.items() if k != "tool"
                }
                return str(tool), args
        except json.JSONDecodeError:
            continue

    return None


@dataclass
class LoopStep:
    iteration:   int
    thought:     str
    action:      str | None       = None
    observation: str | None       = None
    tool_result: ToolResult | None = None

    def summary(self) -> str:
        if self.action:
            status = "✓" if (self.tool_result and self.tool_result.success) else "✗"
            tool = self.tool_result.tool if self.tool_result else "?"
            return f"[{self.iteration}] Action: {tool} {status}"
        return f"[{self.iteration}] Thought"


@dataclass
class LoopResult:
    answer:      str
    steps:       list[LoopStep] = field(default_factory=list)
    iterations:  int            = 0
    elapsed_ms:  int            = 0
    memory_hits: list[str]      = field(default_factory=list)
    used_tools:  list[str]      = field(default_factory=list)


class CognitiveLoop:
    """
    The Unified Cognitive Loop for Obscuro.

    Implements ReAct (Reason + Act):
      Thought:     model reasons about what to do next
      Action:      model emits a tool call as JSON
      Observation: system executes tool, injects result into context
      ... repeat until Final Answer ...

    The model IS the agent. There is no separation.
    """

    def __init__(
        self,
        model:  "OrbModel",
        memory: "Memory",
        tools:  ToolRegistry | None = None,
    ) -> None:
        self.model  = model
        self.memory = memory
        self.tools  = tools or ToolRegistry()

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

        # 1 — Memory retrieval
        if use_memory:
            hits = self.memory.retrieve_similar(message, k=3)
            result.memory_hits = [ep.summary() for ep in hits]

        # 2 — Build initial context
        memory_ctx = ""
        if result.memory_hits:
            memory_ctx = (
                "Relevant past experience:\n"
                + "\n".join(f"  • {h}" for h in result.memory_hits)
                + "\n\n"
            )

        task_input  = memory_ctx + message
        scratchpad  = ""   # accumulates thought+observation across iterations

        # 3 — ReAct loop
        for iteration in range(_MAX_ITERATIONS):
            result.iterations = iteration + 1

            prompt_msg = task_input
            if scratchpad:
                prompt_msg = task_input + "\n\n" + scratchpad.rstrip()

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

            # Try to extract and execute a tool action
            if use_tools:
                action = _extract_action(raw)
                if action:
                    tool_name, args = action
                    tool_result = self.tools.call(tool_name, args)
                    result.used_tools.append(tool_name)

                    observation = tool_result.format()
                    step = LoopStep(
                        iteration=iteration,
                        thought=raw,
                        action=json.dumps({"tool": tool_name, "args": args}),
                        observation=observation,
                        tool_result=tool_result,
                    )
                    result.steps.append(step)

                    # Grow scratchpad, trim if too long
                    entry = f"\nThought: {raw[:900]}\nObservation: {observation}\n"
                    scratchpad += entry
                    if len(scratchpad) > _MAX_SCRATCHPAD:
                        scratchpad = "...[earlier steps trimmed]...\n" + scratchpad[-_MAX_SCRATCHPAD:]
                    continue

            # No tool action, no Final Answer — treat raw output as the answer
            result.steps.append(LoopStep(iteration, raw))
            result.answer = raw
            break

        # Fallback — should rarely trigger
        if not result.answer:
            if result.steps:
                result.answer = result.steps[-1].thought
            else:
                result.answer = "(no response — try rephrasing)"

        # 4 — Learn from this interaction
        if use_memory:
            quality = 0.5 + min(0.4, len(result.used_tools) * 0.08)
            self.memory.add_episode(
                question=message,
                response=result.answer,
                score=quality,
            )

        result.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return result

    # ── Streaming variant (for SSE / sandbox) ─────────────────────────────────

    def stream(
        self,
        message: str,
        history: list[dict],
        *,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        use_tools:      bool  = True,
        use_memory:     bool  = True,
        seed:           int   = 42,
    ) -> Iterator[dict]:
        """
        Generator variant — yields step dicts as they happen.
        Useful for SSE streaming in the evaluation sandbox.
        """
        t0 = time.monotonic()

        # Memory
        hits: list = []
        if use_memory:
            hits = self.memory.retrieve_similar(message, k=3)
        yield {
            "phase": "memory", "status": "complete",
            "hits": [ep.summary() for ep in hits],
            "count": len(hits),
        }

        memory_ctx = ""
        if hits:
            memory_ctx = "Relevant past experience:\n" + "\n".join(
                f"  • {ep.summary()}" for ep in hits
            ) + "\n\n"

        task_input = memory_ctx + message
        scratchpad = ""
        final_answer = ""
        used_tools: list[str] = []

        for iteration in range(_MAX_ITERATIONS):
            prompt_msg = task_input + ("\n\n" + scratchpad.rstrip() if scratchpad else "")
            prompt = self.model.build_prompt(prompt_msg, history, use_tools=use_tools)

            yield {"phase": "thinking", "status": "running", "iteration": iteration + 1}

            raw = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed + iteration,
            )

            if not raw:
                break

            # Final Answer?
            final_m = _FINAL_RE.search(raw)
            if final_m:
                final_answer = final_m.group(1).strip()
                yield {
                    "phase": "answer", "status": "complete",
                    "content": final_answer,
                    "thought": raw,
                }
                break

            # Tool action?
            if use_tools:
                action = _extract_action(raw)
                if action:
                    tool_name, args = action
                    yield {
                        "phase": "action", "status": "running",
                        "tool": tool_name, "args": args,
                        "thought": raw[:600],
                    }

                    tool_result = self.tools.call(tool_name, args)
                    used_tools.append(tool_name)
                    observation = tool_result.format()

                    yield {
                        "phase": "observation", "status": "complete",
                        "tool": tool_name,
                        "output": observation,
                        "success": tool_result.success,
                        "elapsed_ms": tool_result.elapsed_ms,
                    }

                    scratchpad += f"\nThought: {raw[:900]}\nObservation: {observation}\n"
                    if len(scratchpad) > _MAX_SCRATCHPAD:
                        scratchpad = "...[trimmed]...\n" + scratchpad[-_MAX_SCRATCHPAD:]
                    continue

            # No action, no Final Answer
            final_answer = raw
            yield {"phase": "answer", "status": "complete", "content": raw}
            break

        if not final_answer:
            final_answer = "(no response generated)"

        # Learn
        if use_memory:
            quality = 0.5 + min(0.4, len(used_tools) * 0.08)
            self.memory.add_episode(
                question=message,
                response=final_answer,
                score=quality,
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        yield {
            "phase": "done", "status": "complete",
            "elapsed_ms": elapsed_ms,
            "tools_used": used_tools,
            "iterations": iteration + 1,
        }
