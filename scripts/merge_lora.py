"""
Merge LoRA adapter weights permanently into the base GPT-2 model.
Required before GGUF conversion — GGUF needs a single set of weights.

Usage:
    python scripts/merge_lora.py

Output: models/orb-merged/  (full merged model, ~500MB)
"""

import os
import sys

import torch

try:
    from peft import PeftModel
except ImportError:
    os.system(f"{sys.executable} -m pip install peft -q")
    from peft import PeftModel

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def merge_lora(
    base_model_path: str = "models/gpt2",
    output_path: str = "models/orb-merged",
):
    # Choose best available adapter (STaR > SFT > none)
    if os.path.exists("models/orb-star/adapter_config.json"):
        adapter_path = "models/orb-star"
        print("Using Orb-STaR adapter (self-improved)")
    elif os.path.exists("models/orb-sft-lora/adapter_config.json"):
        adapter_path = "models/orb-sft-lora"
        print("Using Orb-SFT adapter (instruction-tuned)")
    else:
        print("No LoRA adapter found. Train first:")
        print("  python scripts/train_sft_lora.py")
        print("  python scripts/star_loop.py")
        sys.exit(1)

    print(f"\nLoading base model from {base_model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    base = GPT2LMHeadModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    )
    base.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging LoRA weights into base model (irreversible)...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Report size
    total_bytes = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
    )
    print(f"\n✓ Merge complete!")
    print(f"  Output: {output_path}")
    print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")
    print("  Next: python scripts/convert_to_gguf.py")


if __name__ == "__main__":
    merge_lora()
