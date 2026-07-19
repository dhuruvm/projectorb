---
name: OASIS Training Pipeline
description: The novel training method designed for Orb — phases, key innovations, and implementation notes
---

## OASIS = Orthogonal Autonomous Self-Improving System

Designed for small models (<1B params) on CPU-only compute. No human labels needed after initial SFT dataset.

## Phases (in order)
1. **Phase 0 — SFT**: LoRA fine-tuning on 10K curated OpenHermes-2.5 examples. Script: `scripts/train_sft_lora.py`
2. **Phase 1 — Orthogonal Seeding**: Train on 4-stream reasoning format (ANALYTICAL/SKEPTICAL/CONCRETE/SYNTHESIS). Requires ~5K generated examples from a teacher model ($5-50 via API).
3. **Phase 2 — Constitutional Compression**: DPO on critique-revision pairs generated using Orb's constitution. Embeds principles in weights.
4. **Phase 3 — SPIN**: Self-play where current model beats previous checkpoint. No external labels.
5. **Phase 4 — STaR Bootstrap**: Math/code self-improvement. Script: `scripts/star_loop.py`. Uses GSM8K.
6. **Phase 5 — Continuous Loop**: Deployment self-improvement using implicit feedback signals.

## Key Innovation
The orthogonal 4-stream reasoning format forces the model to approach problems from multiple independent angles before synthesizing. This is the core novel contribution of OASIS — no existing method does this.

## LoRA config for GPT-2
```
r=16, lora_alpha=32, dropout=0.05
target_modules=["c_attn", "c_proj"]
~295K trainable params (0.24% of total)
```

**Why:** c_attn and c_proj are the query/key/value and output projections in GPT-2's attention — the most impactful layers to adapt.

## Scripts
- `scripts/download_dataset.py` — OpenHermes-2.5 + GSM8K
- `scripts/train_sft_lora.py` — Phase 0
- `scripts/star_loop.py` — Phase 4
- `scripts/merge_lora.py` — Pre-GGUF merge
- `scripts/convert_to_gguf.py` — GGUF + Q4_K_M quantization
