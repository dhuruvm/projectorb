"""
Obscuro — Unified Autonomous Intelligence.
Created by The Director.

Public API:
  OrbAgent      — unified cognitive system (model + loop + tools + memory)
  AgentOptions  — per-call configuration
  AgentResponse — structured result with loop steps, tool trace, memory hits
"""
from .agent   import OrbAgent, AgentOptions, AgentResponse
from .model   import OrbModel
from .memory  import Memory, Episode
from .tools   import ToolRegistry, ToolResult
from .loop    import CognitiveLoop, LoopResult, LoopStep

__all__ = [
    "OrbAgent", "AgentOptions", "AgentResponse",
    "OrbModel",
    "Memory", "Episode",
    "ToolRegistry", "ToolResult",
    "CognitiveLoop", "LoopResult", "LoopStep",
]
