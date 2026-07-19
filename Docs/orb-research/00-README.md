# Orb Research: Full Technical Blueprint
> *From GPT-2 to a Self-Improving, Autonomous Language Model*

## Index

| # | Document | What It Covers |
|---|----------|----------------|
| 01 | [Gap Analysis](./01-gap-analysis.md) | Honest breakdown of GPT-2 vs frontier models — what the gap actually is and what it means |
| 02 | [Training Methods](./02-training-methods.md) | Every major training method: RLHF, Constitutional AI, DPO, GRPO, STaR, SPIN, ReST, PRM, MoE, LoRA — deep analysis of each |
| 03 | [Self-Awareness & Autonomy](./03-self-awareness-autonomy.md) | What "self-awareness", "self-improvement", and "autonomous" actually mean in AI — what's real, what's theoretical, what's buildable |
| 04 | [OASIS — New Training Method](./04-novel-method-OASIS.md) | The new training method designed for Orb: **Orthogonal Autonomous Self-Improving System** — how it works, why it's novel, how it compares |
| 05 | [Roadmap](./05-roadmap.md) | Phase-by-phase roadmap from Orb (current) → Orb-7B → Orb-70B → Orb-Frontier |
| 06 | [Replit Implementation](./06-replit-implementation.md) | What we can build RIGHT NOW on Replit CPU — self-critique, STaR loop, LoRA fine-tuning plan |
| 07 | [GGUF / llama.cpp Compilation](./07-gguf-compilation.md) | Step-by-step: convert Orb to GGUF, quantize, run with llama.cpp |

---

## Executive Summary

**The honest situation:**
- Orb is currently GPT-2 (117M parameters, 2019 architecture)
- Claude Opus 3 is estimated at ~2 trillion parameters, trained on ~10 trillion tokens, using thousands of H100 GPUs for months
- You cannot close that gap on Replit. No training method eliminates a 17,000× parameter gap.

**What IS achievable:**
- A well-crafted fine-tuned Orb that punches significantly above its weight class
- Real self-critique and self-correction behaviors (built into training, not faked)
- A novel training method (OASIS) that makes small models more capable per parameter than any existing approach
- A complete roadmap to scale Orb to frontier level when compute is available
- Compilation to llama.cpp GGUF format for efficient deployment

**The goal of this research:**
Design the *best possible* small model, with the *right architecture of intelligence*, and a clear path to scale it. This is how every major AI lab started.

---

## Key Terms

| Term | Simple Meaning |
|------|----------------|
| Parameters | The "brain cells" of an AI model — more = more capacity |
| Fine-tuning | Training an existing model on new data to change behavior |
| LoRA | Efficient fine-tuning that only updates a small subset of parameters |
| RLHF | Training using human feedback on which answers are better |
| Constitutional AI | Training using AI feedback guided by written principles |
| DPO | A simpler alternative to RLHF — no reward model needed |
| STaR | Self-improvement: model generates reasoning, learns from what worked |
| GGUF | File format used by llama.cpp for fast CPU inference |
| Quantization | Compressing model weights to use less memory (e.g., from 32-bit to 4-bit) |
