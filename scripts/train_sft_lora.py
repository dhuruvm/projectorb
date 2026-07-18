"""
OASIS Phase 0 — LoRA SFT fine-tuning for Orb.

Trains a LoRA adapter on Orb's curated synthetic dataset + any additional data.
Only ~295K parameters are updated (0.24% of total) — feasible on CPU.

Usage:
    python scripts/train_sft_lora.py

Estimated time:
    CPU (Replit): 2–4 hours for 2 epochs on ~94 synthetic examples
    Colab T4:     5–10 minutes

Output: models/orb-sft-lora/
"""

import json, os, sys
import torch
from torch.utils.data import Dataset

def ensure_deps():
    for pkg in ["peft", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}…")
            os.system(f"{sys.executable} -m pip install {pkg} -q")

ensure_deps()

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL   = "models/gpt2"
DATA_PATHS   = ["data/orb_synthetic.jsonl"]   # will add more if available
OUTPUT_DIR   = "models/orb-sft-lora"
MAX_LENGTH   = 512
SPECIAL_TOKS = ["<|system|>", "<|user|>", "<|orb|>", "<|end|>"]

ORB_SYSTEM = (
    "You are Orb, an advanced language model built on the OASIS framework. "
    "You reason carefully, acknowledge uncertainty honestly, "
    "and always aim for accurate, complete, and helpful responses."
)

def fmt_prompt(instruction: str, response: str = "") -> str:
    p = f"<|system|>\n{ORB_SYSTEM}\n<|user|>\n{instruction}\n<|orb|>\n"
    if response:
        p += response + "<|end|>"
    return p

# ── Dataset ───────────────────────────────────────────────────────────────────

class OrbDataset(Dataset):
    def __init__(self, paths, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = []
        for path in paths:
            if not os.path.isfile(path):
                print(f"  [skip] {path} not found")
                continue
            with open(path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item.get("instruction") and item.get("response"):
                        self.examples.append(item)
        print(f"  Dataset: {len(self.examples)} examples from {[p for p in paths if os.path.isfile(p)]}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        full   = fmt_prompt(item["instruction"], item["response"])
        prompt = fmt_prompt(item["instruction"])

        enc = self.tokenizer(
            full, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        prompt_len = min(prompt_len, self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100   # mask prompt tokens from loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ── Model ─────────────────────────────────────────────────────────────────────

def setup():
    print("Loading tokenizer…")
    tok = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKS})

    print("Loading base weights…")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tok))

    # Check if a prior adapter exists — continue training from it
    if os.path.isfile(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        print(f"  Continuing from existing adapter: {OUTPUT_DIR}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    return model, tok

# ── Training ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model, tok = setup()

    # Load all available data
    all_data_paths = DATA_PATHS[:]
    if os.path.isfile("data/sft_10k.jsonl"):          # if full download succeeded
        all_data_paths.append("data/sft_10k.jsonl")

    dataset = OrbDataset(all_data_paths, tok)
    if len(dataset) == 0:
        print("ERROR: No training data found. Run: python scripts/generate_synthetic_data.py")
        sys.exit(1)

    n_train = max(int(0.9 * len(dataset)), len(dataset) - 5)
    n_eval  = len(dataset) - n_train
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [n_train, n_eval])

    is_gpu = torch.cuda.is_available()
    print(f"\nTraining config:")
    print(f"  Train: {n_train}  Eval: {n_eval}")
    print(f"  Device: {'GPU' if is_gpu else 'CPU'}")
    print(f"  Output: {OUTPUT_DIR}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 8
        per_device_eval_batch_size=2,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=30,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=is_gpu,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("\nStarting OASIS Phase 0 — SFT LoRA training…")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\n✓ OASIS Phase 0 complete → {OUTPUT_DIR}")
    print("  Next: python scripts/star_loop.py")

if __name__ == "__main__":
    train()
