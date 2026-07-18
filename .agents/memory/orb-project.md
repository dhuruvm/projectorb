---
name: Orb Model Project
description: Key facts about the Orb project — naming, architecture, file layout, and what's been built
---

## What it is
GPT-2 117M renamed to "Orb" with a full training pipeline and research blueprint.
All UI references must say "Orb" — never "GPT-2".

## Model auto-detection in app.py
The Gradio app loads the best available checkpoint automatically:
- `models/orb-star/` (best — post-STaR)
- `models/orb-sft-lora/` (good — post-SFT)
- `models/gpt2/` (base — no fine-tuning)

**Why:** Users should always see the most capable version automatically.

## Special tokens added
`<|system|>`, `<|user|>`, `<|orb|>`, `<|end|>` — these are added to the tokenizer during SFT. Any adapter trained with these tokens requires the tokenizer from the adapter directory, NOT the base models/gpt2/ tokenizer.

## Dependency install quirk
The `pyproject.toml` has `requires-python = ">=3.13,<3.14"` (had to pin to avoid uv resolution failure with transformers on Linux for Python 3.14+). Core deps installed via `uv pip install` not `uv sync`.

## Research docs location
`Docs/orb-research/` — 7 documents covering gap analysis, all training methods, OASIS design, roadmap, implementation, and GGUF compilation. The OASIS doc (04) is the core novel contribution.
