# Orb — Advanced Language Model

A GPT-2-based language model being developed into a self-improving, constitutionally-aligned AI using the OASIS training framework.

## Run & Operate

- `python app.py` — run the Gradio chat interface (auto-detects best available model)
- `python scripts/download_dataset.py` — download SFT + math training data
- `python scripts/train_sft_lora.py` — OASIS Phase 0: LoRA SFT fine-tuning (~8-14h CPU)
- `python scripts/star_loop.py` — OASIS Phase 4: STaR self-improvement loop
- `python scripts/merge_lora.py` — merge LoRA adapter into base weights for export
- `python scripts/convert_to_gguf.py` — compile Orb to GGUF format for llama.cpp

## Stack

- Python 3.13, PyTorch, HuggingFace Transformers, PEFT (LoRA)
- Gradio 6.x (chat UI)
- llama.cpp (GGUF inference, C++)
- Base model: GPT-2 117M (HuggingFace, loaded from `models/gpt2/`)

## Where Things Live

```
app.py                        — Gradio chat app (auto-loads best model)
models/gpt2/                  — Base GPT-2 weights + tokenizer
models/orb-sft-lora/          — LoRA adapter after Phase 0 SFT
models/orb-star/              — LoRA adapter after Phase 4 STaR self-improvement
models/orb-merged/            — Merged full weights (pre-GGUF)
models/orb-f16.gguf           — GGUF F16 (full quality)
models/orb-Q4_K_M.gguf       — GGUF Q4_K_M (recommended deployment)
scripts/                      — All training scripts
data/                         — Downloaded training datasets
Docs/orb-research/            — Full research documentation (7 documents)
Modelfile                     — Ollama model definition
```

## Architecture Decisions

- **LoRA over full fine-tuning**: Only 295K of 117M parameters are trainable — makes training feasible on CPU and produces a tiny 1.5MB adapter that can be stacked on any GPT-2 checkpoint.
- **OASIS training pipeline**: Custom framework combining SFT → Orthogonal Reasoning → Constitutional DPO → SPIN → STaR. Designed specifically for small models without access to human labels or GPU compute.
- **STaR for self-improvement**: Model generates its own reasoning training data by attempting problems and keeping correct chains. No human labels needed.
- **GGUF/llama.cpp target**: Produces a portable, efficient binary that runs at 20-40 tok/s on CPU vs 8-15 tok/s with PyTorch.
- **Model auto-detection in app.py**: Loads best available checkpoint automatically (STaR > SFT-LoRA > base GPT-2).

## Product

Orb is a self-hosted, open-weight language model that:
- Follows instructions (after SFT fine-tuning)
- Reasons step-by-step using four orthogonal perspectives
- Self-corrects via constitutional critique
- Improves its own math reasoning through STaR loops
- Exports to portable GGUF format for llama.cpp / Ollama

## Research

See `Docs/orb-research/` for the full technical blueprint:
- `01-gap-analysis.md` — Honest analysis of GPT-2 vs frontier model gap
- `02-training-methods.md` — Deep analysis of RLHF, DPO, CAI, STaR, SPIN, GRPO, etc.
- `03-self-awareness-autonomy.md` — What self-awareness & autonomy actually mean in AI
- `04-novel-method-OASIS.md` — The OASIS training method design
- `05-roadmap.md` — Phase-by-phase scaling roadmap (117M → 1.3B → 7B → 70B → Frontier)
- `06-replit-implementation.md` — Full implementation guide for Replit CPU
- `07-gguf-compilation.md` — GGUF conversion and llama.cpp deployment

## User Preferences

- Model must always be named "Orb" — never "GPT-2" in any UI or output
- Research-first approach: understand before building
- Goal: reach frontier quality through OASIS training, then scale with compute

## Gotchas

- Training on CPU is slow (8-14h for SFT). Run overnight with `nohup`.
- LoRA adapters require the same tokenizer vocab size as the model they were trained on. Always save tokenizer alongside adapter.
- GGUF conversion requires llama.cpp to be built from source — the build step takes 2-5 minutes.
- GPT-2 has a 1024 token context limit — keep prompts + responses under this.
