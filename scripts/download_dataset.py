"""
Download and curate training data for Orb Phase 0 (SFT) and Phase 4 (STaR).
Target: 10K high-quality instruction pairs + 500 math problems.

Usage:
    python scripts/download_dataset.py
"""

import json
import os
import sys


def download_sft_data(output_path="data/sft_10k.jsonl", n_samples=10000):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset

    print(f"Downloading OpenHermes-2.5 subset ({n_samples} examples)...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    examples = []
    skipped = 0
    for item in ds:
        if len(examples) >= n_samples:
            break

        convo = item.get("conversations", [])
        if len(convo) < 2:
            skipped += 1
            continue

        user_msg = next((c["value"] for c in convo if c["from"] == "human"), "").strip()
        asst_msg = next((c["value"] for c in convo if c["from"] == "gpt"), "").strip()

        # Quality filters
        if not user_msg or not asst_msg:
            skipped += 1
            continue
        if len(asst_msg) < 80 or len(asst_msg) > 2000:
            skipped += 1
            continue
        # Skip heavy code examples — GPT-2 tokenizer handles code poorly
        if asst_msg.count("```") > 3:
            skipped += 1
            continue

        examples.append({
            "instruction": user_msg,
            "response": asst_msg,
            "source": "openhermes",
        })

        if len(examples) % 1000 == 0:
            print(f"  Collected {len(examples)}/{n_samples} (skipped {skipped})")

    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"✓ Saved {len(examples)} SFT examples → {output_path}")
    return examples


def download_math_data(output_path="data/gsm8k_500.jsonl", n_samples=500):
    try:
        from datasets import load_dataset
    except ImportError:
        from datasets import load_dataset

    print(f"Downloading GSM8K math problems ({n_samples} examples)...")
    ds = load_dataset("gsm8k", "main", split="train")

    examples = []
    for item in list(ds)[:n_samples]:
        answer_text = item["answer"]
        # GSM8K answers end with "#### <number>"
        numeric = answer_text.split("####")[-1].strip().replace(",", "")
        examples.append({
            "question": item["question"],
            "answer": answer_text,
            "numeric_answer": numeric,
        })

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"✓ Saved {len(examples)} math problems → {output_path}")
    return examples


if __name__ == "__main__":
    download_sft_data()
    download_math_data()
    print("\nAll data downloaded. Run scripts/train_sft_lora.py next.")
