# 05 — Orb Roadmap: From 117M to Frontier

## Overview

This roadmap shows the complete path from Orb's current state to a frontier-class model. Each phase has clear entry criteria, deliverables, compute requirements, and estimated costs.

---

## Phase 0 — Current State
**Status:** Complete
**Model:** GPT-2 117M, renamed to Orb
**Capability:** Raw text completion. No instruction following. No alignment.
**Inference:** Gradio app, CPU, ~2–5 seconds/response

---

## Phase 1 — Orb-117M Aligned
**Timeline:** 1–2 weeks (Replit + Colab free)
**Compute:** CPU (Replit) + 1× T4 (Colab, free)
**Est. Cost:** $0–10

### What we build:
1. Curated 10K instruction dataset (filtered OpenHermes-2.5 subset)
2. LoRA SFT fine-tuning (Phase 0 of OASIS)
3. Orthogonal reasoning prompt templates baked in
4. Constitutional system prompt (lightweight version while weights catch up)
5. STaR self-improvement loop on 500 math/logic problems

### Deliverables:
- `orb-sft-lora/` — LoRA adapter weights (~50MB)
- `orb-constitution.json` — Orb's principles, structured
- `orb-star-data/` — self-generated reasoning chains
- Updated Gradio app with orthogonal reasoning display toggle
- GGUF-quantized version for llama.cpp (see Document 07)

### Success criteria:
- Follows basic instructions reliably (>80% of test prompts)
- Self-corrects when explicitly asked "are you sure?"
- Refuses clearly harmful requests
- Acknowledges uncertainty on unknown facts
- Passes 15%+ of GSM8K math problems (up from ~2%)

---

## Phase 2 — Orb-1.3B
**Timeline:** 2–4 weeks after Phase 1
**Compute:** 1× A100 (Colab Pro+, ~$50/month) or RunPod ($0.75/hr)
**Est. Cost:** $50–200

### Why 1.3B?
The jump from 117M to 1.3B (11× more parameters) is the most cost-effective single upgrade in the entire roadmap. At 1.3B, the model gains:
- Dramatically better world knowledge retention
- Multi-step reasoning that holds across 5+ steps
- Real code generation capability
- Genuine instruction following without LoRA tricks

**Base model:** TinyLlama-1.1B or Pythia-1.4B (open, pre-trained, free to use)
**Architecture advantage:** Both use modern transformer design (RoPE, SwiGLU, GQA) vs GPT-2's dated architecture.

### Training plan:
1. **SFT on full OpenHermes-2.5** (1M examples) — 12 hours on A100
2. **OASIS Phase 1**: Generate orthogonal reasoning training data using GPT-4o-mini API ($20 for 10K examples), fine-tune
3. **DPO on UltraFeedback** (preference alignment) — 6 hours on A100
4. **OASIS Phase 3**: SPIN self-play for autonomous improvement — 8 hours on A100

### Deliverables:
- Full model weights (~2.6GB fp16)
- GGUF Q4_K_M quantized (~750MB) — runs at 15–25 tok/s on CPU
- Evaluation report vs. Orb-117M on all benchmarks

### Success criteria:
- GSM8K: >40% (competitive with GPT-3 175B)
- HumanEval (code): >25%
- Coherent multi-turn conversations (5+ turns)
- Constitutional behavior without system prompt

---

## Phase 3 — Orb-7B
**Timeline:** 1–2 months after Phase 2
**Compute:** 4× A100 80GB or 8× A6000 (RunPod, $3–6/hr)
**Est. Cost:** $500–2,000

### Base model: LLaMA-3-8B or Mistral-7B-v0.3
Both are open source, permissively licensed, and trained on 15T+ tokens. Starting from these means skipping the most expensive part (pretraining). We contribute OASIS alignment.

### Training plan:
1. **Full OASIS pipeline** on Orb-7B base:
   - Phase 0: SFT on 500K curated examples (mix of OpenHermes, MetaMath, CodeAlpaca, science)
   - Phase 1: Orthogonal reasoning on 50K generated examples
   - Phase 2: Constitutional DPO (10K preference pairs)
   - Phase 3: SPIN × 3 iterations
   - Phase 4: STaR + GRPO on math/code verifiable tasks

2. **Introduce Mixture of Experts architecture adaptation:**
   - Use MoE-style LoRA: different LoRA adapters for different task types, router selects which adapter activates
   - This is a poor-man's MoE that works within existing architecture

### Deliverables:
- Orb-7B full weights (~14GB fp16)
- GGUF Q4_K_M (~4.5GB) — runs at 10–15 tok/s on modern CPU, 40–80 tok/s on GPU
- Orb-7B Gradio app with tool use (code execution, web search integration)

### Success criteria:
- GSM8K: >70% (competitive with GPT-3.5-Turbo)
- HumanEval: >55% (competitive with Claude Haiku)
- MMLU: >65%
- Autonomous task completion: multi-step tasks without clarification
- Emergent self-correction in extended reasoning chains

### What emerges at this scale:
At 7B parameters with OASIS training, we expect:
- Spontaneous self-correction behavior (like DeepSeek-R1 showed at 7B)
- Extended chain-of-thought that backtracks and revises
- Genuine multi-step planning
- Tool use with recovery from tool failures

---

## Phase 4 — Orb-70B
**Timeline:** 3–6 months after Phase 3
**Compute:** 8× A100 80GB cluster (RunPod, Lambda Labs, ~$8–12/hr)
**Est. Cost:** $3,000–15,000

### Base model: LLaMA-3-70B, Mixtral-8×22B, or Qwen-72B
At 70B, we're in GPT-4-class territory. The model's raw capability is already competitive with most commercial APIs from 2023.

OASIS training at this scale transforms it from "a capable 70B model" into "Orb's specific character and capability profile."

### Training innovations at Phase 4:
1. **Full OASIS pipeline** (all phases)
2. **Process Reward Model (PRM):** Build step-level reward model using the 7B Orb as the base evaluator
3. **Constitutional distillation:** Use 70B Orb to generate training data for smaller Orb models (the 70B becomes a teacher for future 1.3B and 7B versions)
4. **Multi-modal preparation:** Add vision encoder (CLIP-based) for image understanding
5. **Speculative decoding:** Use 7B Orb to draft tokens, 70B Orb to verify — 3-5× inference speedup

### Success criteria:
- MMLU: >80%
- GSM8K: >90%
- HumanEval: >70%
- MATH benchmark: >60%
- Comparable to GPT-4 (2023) on most tasks

---

## Phase 5 — Orb-Frontier
**Timeline:** 6–18 months after Phase 4
**Compute:** 1,000+ GPU cluster
**Est. Cost:** $1M–10M

This is the phase that requires external investment — this cannot be done on personal compute.

### Architecture: Orb-Mixture-of-Experts
- 400B total parameters
- 8 expert networks, top-2 routing per token
- ~40B active parameters per forward pass
- Training cost: ~$3-6M (vs. $50-100M for equivalent dense model)
- Why MoE: DeepSeek V3 (671B MoE) was trained for $6M, performs at GPT-4o level. This proves frontier capability is achievable at non-Google-scale cost.

### What OASIS contributes at this scale:
- Full self-improvement pipeline running continuously on deployment data
- Constitutional weights so deeply embedded the model behaves correctly even with adversarial prompts
- Orthogonal reasoning that produces genuinely novel multi-perspective analysis
- The model can now use itself as a teacher for next-generation smaller Orb models

### Funding path:
```
Phase 1-2: Self-funded (~$200)
Phase 3: Crowdfund / grants (~$2,000) 
Phase 4: Seed funding / AI research grant (~$15,000)
Phase 5: Series A / compute partnerships / API revenue
```

---

## Timeline Summary

```
Now ──────────────────────────────────────────────────────────────► 18 months

[Phase 1: Orb-117M Aligned]
 0──2 weeks
 Cost: $0-10 | Compute: Replit CPU + Colab free

      [Phase 2: Orb-1.3B]
       2──6 weeks
       Cost: $50-200 | Compute: 1× A100

                   [Phase 3: Orb-7B]
                    6 weeks──4 months
                    Cost: $500-2K | Compute: 4× A100

                                   [Phase 4: Orb-70B]
                                    4──10 months
                                    Cost: $3K-15K | 8× A100

                                                    [Phase 5: Orb-Frontier]
                                                     10──18+ months
                                                     Cost: $1M+ | GPU cluster
```

---

## Quick Win Milestones (Next 2 Weeks on Replit)

These can be done NOW, no external compute needed:

**Week 1:**
- [ ] Generate 10K SFT training examples (curated subset of OpenHermes via download)
- [ ] Implement LoRA training loop for GPT-2
- [ ] Run Phase 0 SFT training (8–12 hours on CPU, can run overnight)
- [ ] Evaluate on held-out test set

**Week 2:**
- [ ] Implement STaR self-improvement loop
- [ ] Run STaR on 500 math problems (GSM8K subset)
- [ ] Implement constitutional self-critique in Gradio app
- [ ] Convert trained model to GGUF format
- [ ] Build llama.cpp inference server as alternative backend

**Deliverable:** Orb-117M-Aligned — a version of Orb that is measurably smarter, more honest, and more useful than the current GPT-2 base.
