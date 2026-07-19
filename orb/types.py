"""
orb/types.py — pure-Python data types shared across the cognitive system.

Deliberately has ZERO heavy dependencies (no torch, no transformers, no orb.model).
This lets AgentOptions and AgentResponse be imported in any environment,
including test environments without ML libraries installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loop   import LoopStep
    from .reasoning import RankedResponse
    from .critic import DimensionalScore


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
    response:           str
    critique:           str                        = ""
    reasoning_paths:    list                       = field(default_factory=list)   # list[RankedResponse]
    memory_hits:        list[str]                  = field(default_factory=list)
    elapsed_ms:         int                        = 0
    loop_steps:         list                       = field(default_factory=list)   # list[LoopStep]
    used_tools:         list[str]                  = field(default_factory=list)
    iterations:         int                        = 0
    # v2 additions
    confidence:         float                      = 0.5
    register:           str                        = "analytical"
    plan:               str                        = ""
    consistency_score:  float                      = 1.0
    dimensional_scores: "DimensionalScore | None"  = None
    primed:             bool                       = False
