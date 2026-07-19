# Obscuro — OASIS Cognitive Agent
*Created by The Director*

An autonomous AI agent built on Llama-3.2-1B (1.24B params) using the **OASIS** framework. The architecture implements a real cognitive loop — multi-path reasoning, persistent memory, constitutional self-critique, and a curiosity engine — as a clean Python package.

## How to run

The app starts automatically via the **GPT-2 Gradio App** workflow:

```
python3 app.py
```

Serves on `PORT` (default 8000) at `0.0.0.0`.

## Architecture

```
orb/
  model.py      — GPT-2 wrapper: loading, generation, perplexity scoring
  memory.py     — SQLite-backed episodic + semantic memory with keyword retrieval
  reasoning.py  — Multi-path generation (3 temperatures) with candidate ranking
  critic.py     — Constitutional self-critique and revision (OASIS Phase 2)
  curiosity.py  — Ambiguity detection and knowledge gap identification
  agent.py      — Executive controller: cognitive loop (Observe→Remember→Reason→Critique→Learn)
```

**Cognitive loop per turn:**
```
Observe → Remember → Reason → Critique → Learn → Respond
```

## Key files

| File/Dir | Purpose |
|----------|---------|
| `app.py` | Gradio UI — thin wrapper around OrbAgent |
| `orb/` | Cognitive agent package (see above) |
| `data/memory.db` | Persistent episodic memory (auto-created) |
| `data/*.jsonl` | Training datasets (GSM8K, synthetic) |
| `models/gpt2/` | Base GPT-2 weights |
| `models/orb-sft-lora/` | SFT LoRA adapter (created by training) |
| `models/orb-star/` | STaR self-improved adapter (created by training) |
| `scripts/` | Training pipeline scripts |
| `Docs/orb-research/` | Full OASIS research blueprint |

## Training pipeline (optional — runs overnight on CPU)

```bash
# 1. Download/prepare training data
python scripts/download_dataset.py
python scripts/generate_synthetic_data.py

# 2. SFT fine-tuning with LoRA (~8-14 hours on CPU)
nohup python scripts/train_sft_lora.py > logs/sft_training.log 2>&1 &

# 3. STaR self-improvement loop (~3-6 hours on CPU)
nohup python scripts/star_loop.py > logs/star_loop.log 2>&1 &

# 4. Convert to GGUF for llama.cpp (see Docs/orb-research/07-gguf-compilation.md)
python scripts/convert_to_gguf.py
```

The app auto-detects and loads the best available checkpoint at startup:
`orb-star` → `orb-sft-lora` → `gpt2` (base)

## OASIS training phases

| Phase | Method | Script |
|-------|--------|--------|
| 0 | SFT with LoRA | `scripts/train_sft_lora.py` |
| 1 | Four-stream reasoning (prompt-level) | Built into `orb/reasoning.py` |
| 2 | Constitutional self-critique | Built into `orb/critic.py` |
| 3 | DPO (planned) | — |
| 4 | STaR self-improvement loop | `scripts/star_loop.py` |

## User preferences

- Keep Python project structure; no JS/Node tooling.
- Do not migrate to pnpm workspace unless explicitly asked.
