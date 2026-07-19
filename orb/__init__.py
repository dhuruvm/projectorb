"""
Orb — OASIS Cognitive Agent
Orthogonal Autonomous Self-Improving System

Package layout:
  model     — GPT-2 wrapper: loading, generation, scoring
  memory    — SQLite-backed episodic + semantic memory
  reasoning — Multi-path generation with candidate ranking
  critic    — Constitutional self-critique and revision
  curiosity — Knowledge gap detection and clarification
  agent     — Executive controller: cognitive loop
"""
from .agent import OrbAgent, AgentOptions, AgentResponse

__all__ = ["OrbAgent", "AgentOptions", "AgentResponse"]
