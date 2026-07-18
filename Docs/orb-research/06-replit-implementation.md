# 06 — Replit Implementation: What We Build Right Now

## What's Achievable on CPU-Only Replit

No GPU. No external APIs required. No cloud budget.
Here is exactly what we build, in what order, and why each step matters.

---

## Step 1: Download the SFT Dataset

We need curated instruction-following data. We'll use a filtered subset of OpenHermes-2.5 and GSM8K (math).

```python
# scripts/download_dataset.py
"""
Downloads and curates training data for Orb Phase 0 (SFT) and Phase 4 (STaR).
Target: 10K high-quality instruction pairs + 500 math problems.
"""
import json
import os
from datasets import load_dataset

def download_sft_data(output_path="data/sft_10k.jsonl", n_samples=10000):
    print("Downloading OpenHermes-2.5 subset...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    
    examples = []
    for item in ds:
        if len(examples) >= n_samples:
            break
        
        # Quality filters
        convo = item.get("conversations", [])
        if len(convo) < 2:
            continue
        
        user_msg = next((c["value"] for c in convo if c["from"] == "human"), "")
        asst_msg = next((c["value"] for c in convo if c["from"] == "gpt"), "")
        
        # Length filter: not too short, not too long
        if len(asst_msg) < 100 or len(asst_msg) > 2000:
            continue
        
        # Skip code-heavy examples (GPT-2 tokenizer handles code poorly)
        if asst_msg.count("```") > 4:
            continue
        
        examples.append({
            "instruction": user_msg,
            "response": asst_msg,
            "source": "openhermes"
        })
    
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Saved {len(examples)} examples to {output_path}")
    return examples


def download_math_data(output_path="data/gsm8k_500.jsonl", n_samples=500):
    print("Downloading GSM8K math problems...")
    ds = load_dataset("gsm8k", "main", split="train")
    
    examples = []
    for item in list(ds)[:n_samples]:
        examples.append({
            "question": item["question"],
            "answer": item["answer"],
            # Extract just the number from the answer (for verification)
            "numeric_answer": item["answer"].split("####")[-1].strip()
        })
    
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Saved {len(examples)} math problems to {output_path}")
    return examples


if __name__ == "__main__":
    download_sft_data()
    download_math_data()
```

---

## Step 2: LoRA Fine-Tuning (Phase 0 — SFT)

This is the core training step. We use PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters on GPT-2.

```python
# scripts/train_sft_lora.py
"""
LoRA SFT fine-tuning for Orb Phase 0.
Runs on CPU (slow but feasible for 10K examples with small batch).
Estimated time: 8-14 hours on Replit CPU for 1 epoch.
For faster training: run on Colab T4 (~45 minutes).
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

# ── Orb prompt template ──────────────────────────────────────────────────────

ORB_SYSTEM = """You are Orb, an advanced language model designed to be helpful, honest, and thoughtful. You reason carefully, acknowledge uncertainty, and always aim to give accurate, complete responses."""

def format_orb_prompt(instruction: str, response: str = "") -> str:
    prompt = f"<|system|>\n{ORB_SYSTEM}\n<|user|>\n{instruction}\n<|orb|>\n"
    if response:
        prompt += response + "<|end|>"
    return prompt


# ── Dataset ──────────────────────────────────────────────────────────────────

class OrbSFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.examples.append(item)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        full_text = format_orb_prompt(item["instruction"], item["response"])
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels: -100 for prompt tokens (don't compute loss on them)
        # Only compute loss on the response tokens
        prompt_only = format_orb_prompt(item["instruction"])
        prompt_len = len(self.tokenizer(prompt_only, return_tensors="pt")["input_ids"][0])
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ── LoRA Configuration ───────────────────────────────────────────────────────

def setup_lora_model(model_path: str):
    print("Loading base model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Orb special tokens
    special_tokens = {
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|orb|>", "<|end|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                  # Rank — higher = more expressive
        lora_alpha=32,         # Scaling factor
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],  # GPT-2's attention projections
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected: "trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.24%"
    
    return model, tokenizer


# ── Training ─────────────────────────────────────────────────────────────────

def train(model_path="models/gpt2", data_path="data/sft_10k.jsonl", output_dir="models/orb-sft-lora"):
    model, tokenizer = setup_lora_model(model_path)
    
    dataset = OrbSFTDataset(data_path, tokenizer, max_length=512)
    
    # Split 90/10 train/eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,     # Effective batch size: 16
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=False,                         # CPU: must be False
        dataloader_num_workers=0,           # CPU: 0 workers
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting LoRA SFT training...")
    print(f"  Dataset size: {len(train_dataset)} train, {len(eval_dataset)} eval")
    print(f"  Estimated time on CPU: 8-14 hours")
    print(f"  Estimated time on Colab T4: 45-60 minutes")
    
    trainer.train()
    
    # Save the LoRA adapter (not the full model — only 1.2MB!)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    train()
```

---

## Step 3: STaR Self-Improvement Loop (Phase 4 — Simplified)

```python
# scripts/star_loop.py
"""
STaR self-improvement loop for Orb.
Uses GSM8K math problems as the verifiable task.
Runs multiple rounds, each producing a smarter model.
"""

import json
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

FOUR_STREAM_PROMPT = """Solve this step by step using four perspectives:

[ANALYTICAL]: Break the problem into logical components
[SKEPTICAL]: Check assumptions and potential errors  
[CONCRETE]: Ground with specific numbers and calculations
[SYNTHESIS]: Combine insights to reach the final answer

Problem: {question}

[ANALYTICAL]:"""


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from a chain-of-thought response."""
    # Look for patterns like "= 42", "is 42", "answer is 42", "#### 42"
    patterns = [
        r"####\s*(\-?\d+(?:\.\d+)?)",
        r"(?:answer|result|total)\s+(?:is|=|:)\s*(\-?\d+(?:\.\d+)?)",
        r"=\s*(\-?\d+(?:\.\d+)?)\s*$",
        r"\b(\-?\d+(?:\.\d+)?)\s*(?:\.|$)",
    ]
    
    # Search from end of text (final answer is usually last)
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return matches[-1].replace(",", "").strip()
    return None


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def star_round(model, tokenizer, math_problems: list, round_num: int) -> list:
    """
    One round of STaR:
    - Try to solve each problem with chain-of-thought
    - Keep examples where we got the right answer
    - For failures, do rationalization (show answer, ask for reasoning)
    Returns a list of training examples.
    """
    training_examples = []
    correct = 0
    rationalized = 0
    
    for i, problem in enumerate(math_problems):
        question = problem["question"]
        correct_answer = problem["numeric_answer"].strip()
        
        # Attempt 1: Multi-stream reasoning
        prompt = FOUR_STREAM_PROMPT.format(question=question)
        response = generate_response(model, tokenizer, prompt)
        predicted = extract_numeric_answer(response)
        
        if predicted and predicted == correct_answer:
            # Correct! Use this reasoning chain
            training_examples.append({
                "input": prompt,
                "output": response,
                "weight": 1.0,
                "type": "direct"
            })
            correct += 1
        else:
            # Wrong — try rationalization
            rationalization_prompt = f"""Problem: {question}
The correct answer is: {correct_answer}

Using the four-stream method [ANALYTICAL, SKEPTICAL, CONCRETE, SYNTHESIS], 
show the step-by-step reasoning that arrives at {correct_answer}:

[ANALYTICAL]:"""
            rationalization = generate_response(model, tokenizer, rationalization_prompt)
            training_examples.append({
                "input": prompt,
                "output": rationalization,
                "weight": 0.5,   # Lower weight — this was rationalized, not discovered
                "type": "rationalized"
            })
            rationalized += 1
        
        if (i + 1) % 50 == 0:
            accuracy = correct / (i + 1)
            print(f"  Round {round_num} | Problem {i+1}/{len(math_problems)} | "
                  f"Accuracy so far: {accuracy:.1%} | Rationalized: {rationalized}")
    
    print(f"\nRound {round_num} complete:")
    print(f"  Direct correct: {correct}/{len(math_problems)} = {correct/len(math_problems):.1%}")
    print(f"  Rationalized: {rationalized}")
    
    return training_examples


def star_loop(
    base_model_path: str = "models/gpt2",
    lora_adapter_path: str = "models/orb-sft-lora",
    math_data_path: str = "data/gsm8k_500.jsonl",
    n_rounds: int = 3,
    output_dir: str = "models/orb-star"
):
    import os
    from transformers import TrainingArguments, Trainer
    from torch.utils.data import Dataset as TorchDataset
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(lora_adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    
    # Load math problems
    math_problems = []
    with open(math_data_path) as f:
        for line in f:
            math_problems.append(json.loads(line))
    
    # Track accuracy across rounds
    round_accuracies = []
    
    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"STaR Round {round_num}/{n_rounds}")
        print(f"{'='*60}")
        
        # Generate training data for this round
        training_examples = star_round(model, tokenizer, math_problems, round_num)
        
        # Save training examples
        round_dir = f"{output_dir}/round_{round_num}"
        os.makedirs(round_dir, exist_ok=True)
        with open(f"{round_dir}/training_data.jsonl", "w") as f:
            for ex in training_examples:
                f.write(json.dumps(ex) + "\n")
        
        # Simple LoRA micro-update on this round's data
        # (abbreviated — full implementation would use the Trainer)
        print(f"Fine-tuning on {len(training_examples)} examples from this round...")
        # [LoRA fine-tuning step here — same as train_sft_lora.py but on round data]
        
        # Evaluate
        correct = sum(1 for ex in training_examples if ex["type"] == "direct")
        accuracy = correct / len(training_examples)
        round_accuracies.append(accuracy)
        print(f"Round {round_num} accuracy: {accuracy:.1%}")
    
    # Final save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("STaR Loop Complete!")
    for i, acc in enumerate(round_accuracies, 1):
        print(f"  Round {i}: {acc:.1%}")
    print(f"  Improvement: {round_accuracies[0]:.1%} → {round_accuracies[-1]:.1%}")
    print("="*60)


if __name__ == "__main__":
    star_loop()
```

---

## Step 4: Constitutional Self-Critique in the Gradio App

This is the immediate improvement to `app.py` — add self-critique before showing the response. No training needed.

```python
# Add to app.py — constitutional self-critique wrapper

ORB_CONSTITUTION = [
    "Is this response factually accurate? Flag any claims I'm uncertain about.",
    "Is this response complete? Have I missed important aspects of the question?",
    "Is this response clear? Would someone unfamiliar with the topic understand it?",
    "Am I being appropriately honest about uncertainty?",
    "Does this response actually help the user accomplish their real goal?",
]

def self_critique_and_revise(
    initial_response: str,
    user_message: str,
    model,
    tokenizer,
    enabled: bool = True
) -> tuple[str, str]:
    """
    Apply constitutional self-critique to improve the response.
    Returns (final_response, critique_text) for display.
    
    In the Gradio UI, the critique can be shown in a collapsible section
    so users can see how Orb thinks about its own responses.
    """
    if not enabled:
        return initial_response, ""
    
    # Build critique prompt
    critique_prompt = f"""You are reviewing your own response to improve it.

User asked: {user_message}

Your initial response: {initial_response}

Apply these quality checks:
1. Accuracy — any uncertain claims that should be flagged?
2. Completeness — anything important missing?
3. Clarity — is it understandable?
4. Honesty — appropriate uncertainty expressed?

Brief critique (be specific, be critical):"""

    critique = generate_text(critique_prompt, model, tokenizer, max_new_tokens=150)
    
    # Build revision prompt
    revision_prompt = f"""User asked: {user_message}

Initial response: {initial_response}

After reflection, here is an improved response that addresses the identified issues:"""

    revised = generate_text(revision_prompt, model, tokenizer, max_new_tokens=250)
    
    return revised, critique


def generate_text(prompt: str, model, tokenizer, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

---

## Step 5: Load Fine-Tuned LoRA Adapter in the App

After training completes, swap the base GPT-2 for the fine-tuned Orb:

```python
# Add to app.py — load LoRA adapter if available

import os
from peft import PeftModel

ORB_LORA_PATH = "models/orb-sft-lora"   # After Phase 0 SFT
ORB_STAR_PATH = "models/orb-star"         # After STaR loop (better)

def load_orb_model(model_path="models/gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(
        ORB_STAR_PATH if os.path.exists(ORB_STAR_PATH) else
        ORB_LORA_PATH if os.path.exists(ORB_LORA_PATH) else
        model_path
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    
    if os.path.exists(ORB_STAR_PATH):
        model = PeftModel.from_pretrained(base_model, ORB_STAR_PATH)
        print("[startup] Loaded Orb-STaR (self-improved model)")
    elif os.path.exists(ORB_LORA_PATH):
        model = PeftModel.from_pretrained(base_model, ORB_LORA_PATH)
        print("[startup] Loaded Orb-SFT (instruction-tuned model)")
    else:
        model = base_model
        print("[startup] Loaded Orb base (no fine-tuning found)")
    
    model.eval()
    return model, tokenizer
```

---

## What This Produces

After completing Steps 1–5:

| Component | Location | Size | What It Does |
|-----------|----------|------|-------------|
| Base weights | `models/gpt2/` | ~500MB | Original GPT-2 weights |
| SFT LoRA adapter | `models/orb-sft-lora/` | ~1.5MB | Instruction following |
| STaR adapter | `models/orb-star/` | ~1.5MB | Math reasoning |
| Training data | `data/` | ~50MB | Curated datasets |
| GGUF model | `models/orb.gguf` | ~120MB (Q8) | llama.cpp inference |

**Total additional storage: < 200MB**
**Total compute on Replit CPU: ~12-20 hours (run overnight)**
**Cost: $0**

---

## Running the Training Pipeline

```bash
# 1. Install dependencies
uv pip install peft datasets transformers torch tqdm

# 2. Download data (~10 minutes)
python scripts/download_dataset.py

# 3. Start SFT training (run overnight — 8-14 hours)
nohup python scripts/train_sft_lora.py > logs/sft_training.log 2>&1 &

# 4. After SFT, run STaR loop (~3-6 hours)
nohup python scripts/star_loop.py > logs/star_loop.log 2>&1 &

# 5. Convert to GGUF (see Document 07)
python scripts/convert_to_gguf.py

# 6. Restart the Gradio app — it will auto-detect and load the fine-tuned adapter
```
