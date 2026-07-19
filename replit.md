# Orb — OASIS Model

A GPT-2-based AI chat assistant built on the **OASIS** (Orthogonal Autonomous Self-Improving System) framework. Features constitutional self-critique, four-stream reasoning, and a Gradio web UI.

## How to run

The app starts automatically via the **GPT-2 Gradio App** workflow, which runs:

```
python3 app.py
```

It serves on `PORT` (default 8000) at `0.0.0.0`.

## Stack

- **Model**: GPT-2 (117M params) via HuggingFace Transformers; optional LoRA adapter support via PEFT
- **UI**: Gradio 6
- **Training data**: `data/` — GSM8K samples + synthetic Orb data
- **Model weights**: `models/gpt2/` (downloaded at setup; not committed to git)
- **Fine-tuned adapters**: `models/orb-sft-lora/` and `models/orb-star/` (created by training scripts; auto-detected at startup)

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio app — model loading, chat logic, UI |
| `main.py` | Stub entry point (unused) |
| `data/` | Training datasets (GSM8K, synthetic) |
| `models/gpt2/` | Base GPT-2 weights |
| `Docs/orb-research/` | Full research blueprint for OASIS training |
| `Docs/glasswing/` | Project announcement and overview docs |
| `Modelfile` | Ollama Modelfile for future deployment |

## OASIS training phases

1. **SFT** — supervised fine-tuning on curated data  
2. **Orthogonal Reasoning** — four-stream analytical prompting  
3. **Constitutional DPO** — preference optimization guided by Orb constitution  
4. **SPIN** — self-play improvement  
5. **STaR** — self-taught reasoning loop

Training scripts are described in `Docs/orb-research/06-replit-implementation.md`.

## User preferences

- Keep project structure as-is; do not migrate to pnpm workspace unless asked.
