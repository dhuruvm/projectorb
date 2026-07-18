"""
OASIS Phase 4 — STaR Self-Improvement Loop for Orb.

Runs multiple rounds of self-improvement on math reasoning (GSM8K).
Each round: attempt → check → train on correct chains → repeat.
Model improves its own reasoning with no human labels.

Usage:
    python scripts/star_loop.py

Estimated time per round: 2–4 hours on CPU for 500 problems.

Output: models/orb-star/  (improved LoRA adapter)
"""

import json
import os
import re
import sys

import torch

# ── Ensure deps ───────────────────────────────────────────────────────────────
try:
    from peft import PeftModel
except ImportError:
    os.system(f"{sys.executable} -m pip install peft -q")
    from peft import PeftModel

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ── Prompt ────────────────────────────────────────────────────────────────────

FOUR_STREAM_MATH_PROMPT = """\
<|system|>
You are Orb, an advanced reasoning model. Solve problems using four perspectives.
<|user|>
Solve step by step using four analytical streams:

[ANALYTICAL] Break the problem into logical steps.
[SKEPTICAL]  Check for hidden assumptions or common errors.
[CONCRETE]   Work through the actual numbers carefully.
[SYNTHESIS]  Combine insights and state the final numeric answer.

Problem: {question}
<|orb|>
[ANALYTICAL]"""

RATIONALIZATION_PROMPT = """\
<|system|>
You are Orb. Show clear mathematical reasoning.
<|user|>
Problem: {question}
The correct answer is: {answer}

Using the four-stream method [ANALYTICAL, SKEPTICAL, CONCRETE, SYNTHESIS],
show the step-by-step reasoning that leads to the answer {answer}:
<|orb|>
[ANALYTICAL]"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from a chain-of-thought response."""
    text = text.lower().replace(",", "")
    patterns = [
        r"####\s*([\-]?\d+(?:\.\d+)?)",
        r"\[synthesis\].*?(?:answer|result|total)\s+(?:is|=|:)?\s*([\-]?\d+(?:\.\d+)?)",
        r"(?:final\s+answer|answer)\s+(?:is|=|:)\s*([\-]?\d+(?:\.\d+)?)",
        r"=\s*([\-]?\d+(?:\.\d+)?)\s*(?:\.|$|\n)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return None


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 350) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=650
    )
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.75,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


# ── One STaR round ────────────────────────────────────────────────────────────

def run_star_round(model, tokenizer, problems: list, round_num: int) -> tuple[list, float]:
    training_examples = []
    n_correct = 0
    n_rationalized = 0

    for i, prob in enumerate(problems):
        question = prob["question"]
        correct = prob["numeric_answer"].strip().replace(",", "")

        prompt = FOUR_STREAM_MATH_PROMPT.format(question=question)
        response = generate(model, tokenizer, prompt)
        predicted = extract_numeric_answer(response)

        if predicted and predicted == correct:
            training_examples.append({
                "input": prompt,
                "output": response,
                "weight": 1.0,
                "type": "direct",
            })
            n_correct += 1
        else:
            rat_prompt = RATIONALIZATION_PROMPT.format(
                question=question, answer=correct
            )
            rationalization = generate(model, tokenizer, rat_prompt)
            training_examples.append({
                "input": prompt,
                "output": rationalization,
                "weight": 0.5,
                "type": "rationalized",
            })
            n_rationalized += 1

        if (i + 1) % 50 == 0:
            acc = n_correct / (i + 1)
            print(
                f"  Round {round_num} | {i+1}/{len(problems)} problems | "
                f"Accuracy: {acc:.1%} | Rationalized: {n_rationalized}"
            )

    accuracy = n_correct / len(problems)
    print(f"\n  Round {round_num} summary:")
    print(f"    Direct correct:  {n_correct}/{len(problems)} = {accuracy:.1%}")
    print(f"    Rationalized:    {n_rationalized}/{len(problems)}")

    return training_examples, accuracy


# ── LoRA micro-update ─────────────────────────────────────────────────────────

def micro_update(model, tokenizer, training_examples: list, round_dir: str):
    """Fine-tune the model on this round's training examples."""
    from torch.utils.data import Dataset as TorchDataset
    from transformers import TrainingArguments, Trainer

    class RoundDataset(TorchDataset):
        def __init__(self, examples, tokenizer, max_length=512):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            full_text = ex["input"] + ex["output"] + "<|end|>"
            prompt_text = ex["input"]

            enc = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            attention_mask = enc["attention_mask"].squeeze()

            prompt_len = len(
                self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
            )
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    dataset = RoundDataset(training_examples, tokenizer)

    args = TrainingArguments(
        output_dir=round_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=50,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    return model


# ── Main loop ─────────────────────────────────────────────────────────────────

def star_loop(
    base_model_path: str = "models/gpt2",
    adapter_path: str = "models/orb-sft-lora",
    math_data_path: str = "data/gsm8k_500.jsonl",
    n_rounds: int = 3,
    output_dir: str = "models/orb-star",
):
    if not os.path.exists(math_data_path):
        print(f"Math data not found at {math_data_path}")
        print("Run: python scripts/download_dataset.py")
        sys.exit(1)

    # Determine which adapter to load
    load_path = adapter_path if os.path.exists(adapter_path) else None

    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        load_path or base_model_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    base = GPT2LMHeadModel.from_pretrained(base_model_path)
    base.resize_token_embeddings(len(tokenizer))

    if load_path:
        model = PeftModel.from_pretrained(base, load_path)
        print(f"  Loaded adapter from {load_path}")
    else:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"], bias="none",
        )
        model = get_peft_model(base, lora_cfg)
        print("  No adapter found — starting from base GPT-2")

    model.train()

    # Load math problems
    problems = []
    with open(math_data_path) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"  Loaded {len(problems)} math problems")

    # Track accuracy progress
    accuracies = []

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'═'*60}")
        print(f"  STaR Round {round_num}/{n_rounds}")
        print(f"{'═'*60}")

        model.eval()
        training_examples, accuracy = run_star_round(model, tokenizer, problems, round_num)
        accuracies.append(accuracy)

        # Save round data
        round_dir = f"{output_dir}/round_{round_num}"
        os.makedirs(round_dir, exist_ok=True)
        with open(f"{round_dir}/training_data.jsonl", "w") as f:
            for ex in training_examples:
                f.write(json.dumps(ex) + "\n")

        # Fine-tune on this round's data
        print(f"\n  Fine-tuning on {len(training_examples)} examples...")
        model.train()
        model = micro_update(model, tokenizer, training_examples, round_dir)

    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'═'*60}")
    print("  STaR Loop Complete!")
    print(f"{'═'*60}")
    for i, acc in enumerate(accuracies, 1):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  Round {i}: [{bar}] {acc:.1%}")
    if len(accuracies) > 1:
        delta = accuracies[-1] - accuracies[0]
        print(f"\n  Improvement: +{delta:.1%} over {n_rounds} rounds")
    print(f"\n  Model saved → {output_dir}")
    print("  Next: python scripts/merge_lora.py")


if __name__ == "__main__":
    star_loop()
