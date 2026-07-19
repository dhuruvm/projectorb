"""
Obscuro — Unified Autonomous Intelligence.
Created by The Director.

Heavy ML dependencies (torch / transformers) are imported lazily inside
OrbModel.__init__ so this package is importable in environments that only
have the pure-Python modules (e.g. during unit testing).
"""

# ── Always-available (pure Python, no torch) ─────────────────────────────────
from .memory        import Memory, Episode
from .tools         import ToolRegistry, ToolResult
from .reasoning     import MultiPathReasoner, RankedResponse, ReasoningResult, self_consistency_vote
from .critic        import Critic, CritiqueResult, DimensionalScore
from .metacognition import MetacognitionEngine, MetacognitiveAssessment
from .subconscious  import SubconsciousProcessor, PrimingContext
from .loop          import CognitiveLoop, LoopResult, LoopStep
from .types         import AgentOptions, AgentResponse   # no torch required

# ── ML-dependent (torch / transformers) — guarded import ─────────────────────
try:
    from .model     import OrbModel
    from .curiosity import CuriosityEngine
    from .agent     import OrbAgent
    _ML_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _ML_AVAILABLE = False

__all__ = [
    # Always
    "Memory", "Episode",
    "ToolRegistry", "ToolResult",
    "MultiPathReasoner", "RankedResponse", "ReasoningResult", "self_consistency_vote",
    "Critic", "CritiqueResult", "DimensionalScore",
    "MetacognitionEngine", "MetacognitiveAssessment",
    "SubconsciousProcessor", "PrimingContext",
    "CognitiveLoop", "LoopResult", "LoopStep",
    # ML-dependent (only when torch is available)
    "OrbModel",
    "CuriosityEngine",
    "OrbAgent", "AgentOptions", "AgentResponse",
]
