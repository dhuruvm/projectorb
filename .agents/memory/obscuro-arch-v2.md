---
name: Obscuro Architecture v2
description: The unified autonomous agent architecture — ReAct loop, tool registry, TF-IDF memory, unified model+agent design
---

## Core principle
Model and agent are NOT separate. The model's generated output IS action selection. No separate planner or policy network.

## Execution modes
1. **ReAct/Agentic** (`use_tools=True`, default) — `orb/loop.py` CognitiveLoop: Thought→Action→Observation→Answer loop with real tool execution, up to 12 iterations
2. **Multi-path** (`use_tools=False`) — 3 temperature candidates (0.6/0.85/1.1) ranked by heuristic+perplexity score; optional constitutional critique

## New files added
- `orb/tools.py` — ToolRegistry with 8 built-in tools: shell, python, file_read, file_write, file_delete, file_list, web_fetch, think
- `orb/loop.py` — CognitiveLoop (ReAct), also has a `stream()` generator for SSE

## Key design decisions
- `_extract_action()` in loop.py tries canonical `{"tool":"name","args":{}}` format first, then lenient JSON scan — handles imperfect model JSON output
- Max scratchpad is 8000 chars (trimmed with `...[earlier steps trimmed]...` prefix) to stay within 3072 token context window
- Memory upgraded from BM25 keyword overlap to TF-IDF cosine similarity in memory.py — "fast" and "quick" now score similarly
- `build_prompt()` in model.py takes `use_tools: bool` — agentic mode injects full tool schema + ReAct instructions into system prompt; conversational mode uses lighter prompt
- Scoring: perplexity still used for multi-path ranking; tool-use tasks get quality = 0.5 + 0.08×(tool_count) as memory score
- Context window extended from 2048 to 3072 tokens for ReAct mode

**Why:** A 1B parameter model becomes "superhuman" not by having more weights but by having real tool access, structured reasoning, and persistent memory. The architecture is the intelligence amplifier.

**How to apply:** When extending with new tools, register via `ToolRegistry.register(name, fn)` in agent __init__. For new domains, extend `_SYSTEM_AGENTIC` in model.py.
